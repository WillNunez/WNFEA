"""
Block Compressed Sparse Row (BSR 6x6) Matrix Storage & Operators for WNFEA.

Represents structural finite element stiffness matrices as collections of dense 6x6 nodal
blocks. In 3D continuum and frame mechanics, each node naturally possesses 6 degrees of
freedom (3 translations, 3 rotations). Storing stiffness matrices in BSR 6x6 format:
1. Eliminates 97.2% of column index memory traffic (1 column index per 36 values).
2. Enables 128-bit vectorized GPU memory transactions (float4 / double2).
3. Replaces irregular indirect pointer chasing with dense in-register block operations.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp


@dataclass
class BSR6x6Matrix:
    """
    Block Compressed Sparse Row matrix with fixed 6x6 block size.

    Attributes:
        block_row_ptr: [n_block_rows + 1] int32 array indexing block rows.
        block_col_idx: [nnz_blocks] int32 array of block column indices.
        values: [nnz_blocks, 36] contiguous float32/float64 array of block entries (row-major 6x6).
        n_nodes: Number of block rows and block columns (n_dofs = n_nodes * 6).
        nnz_blocks: Total number of non-zero 6x6 blocks (nnz_scalars = nnz_blocks * 36).
        dtype: np.float32 or np.float64.
    """
    block_row_ptr: np.ndarray
    block_col_idx: np.ndarray
    values: np.ndarray
    n_nodes: int
    nnz_blocks: int
    dtype: np.dtype

    @property
    def shape(self) -> tuple[int, int]:
        n_dofs = self.n_nodes * 6
        return (n_dofs, n_dofs)

    @property
    def nnz(self) -> int:
        return self.nnz_blocks * 36


def csr_to_bsr6x6(K_csr: sp.csr_matrix, dtype=np.float32) -> BSR6x6Matrix:
    """
    Convert a scalar CSR stiffness matrix to Block CSR 6x6 format.

    Args:
        K_csr: [6*N, 6*N] symmetric CSR matrix.
        dtype: Target floating-point precision (np.float32 or np.float64).

    Returns:
        BSR6x6Matrix instance.
    """
    if not isinstance(K_csr, sp.csr_matrix):
        K_csr = K_csr.tocsr()

    n_dofs = K_csr.shape[0]
    if n_dofs % 6 != 0:
        raise ValueError(f"Matrix dimension ({n_dofs}) must be a multiple of 6 for BSR 6x6.")

    n_nodes = n_dofs // 6
    K_csr.sort_indices()

    row_ptr = K_csr.indptr
    col_idx = K_csr.indices
    csr_data = K_csr.data

    # 1. Determine block structure by union of block column indices across the 6 DOFs of each node
    b_row_ptr = np.zeros(n_nodes + 1, dtype=np.int32)
    blocks_per_row = []

    for br in range(n_nodes):
        r_start = br * 6
        r_end = r_start + 6
        # Gather all column indices for the 6 rows of node br
        cols_in_node = col_idx[row_ptr[r_start] : row_ptr[r_end]]
        if len(cols_in_node) > 0:
            block_cols = np.unique(cols_in_node // 6)
        else:
            block_cols = np.array([br], dtype=np.int32)
        blocks_per_row.append(block_cols.astype(np.int32))
        b_row_ptr[br + 1] = b_row_ptr[br] + len(block_cols)

    nnz_blocks = int(b_row_ptr[-1])
    b_col_idx = np.concatenate(blocks_per_row).astype(np.int32)
    b_values = np.zeros((nnz_blocks, 36), dtype=dtype)

    # 2. Populate 6x6 block values
    # For fast lookup, map (br, bc) to block index
    for br in range(n_nodes):
        b_start = b_row_ptr[br]
        b_end = b_row_ptr[br + 1]
        row_b_cols = b_col_idx[b_start:b_end]
        col_to_blk = {bc: b_start + i for i, bc in enumerate(row_b_cols)}

        for local_r in range(6):
            r = br * 6 + local_r
            k_start = row_ptr[r]
            k_end = row_ptr[r + 1]
            cols = col_idx[k_start:k_end]
            vals = csr_data[k_start:k_end]

            for c, v in zip(cols, vals):
                bc = c // 6
                local_c = c % 6
                blk_idx = col_to_blk[bc]
                b_values[blk_idx, local_r * 6 + local_c] = v

    return BSR6x6Matrix(
        block_row_ptr=b_row_ptr,
        block_col_idx=b_col_idx,
        values=b_values,
        n_nodes=n_nodes,
        nnz_blocks=nnz_blocks,
        dtype=np.dtype(dtype),
    )


def bsr6x6_to_csr(K_bsr: BSR6x6Matrix, dtype=None) -> sp.csr_matrix:
    """
    Convert a BSR 6x6 matrix back to standard scalar scipy.sparse.csr_matrix.
    """
    if dtype is None:
        dtype = K_bsr.dtype

    n_nodes = K_bsr.n_nodes
    n_dofs = n_nodes * 6
    nnz_blocks = K_bsr.nnz_blocks

    # Vectorized unpacking of 6x6 blocks to COO triplets
    # Template for 6x6 block indices
    r_grid, c_grid = np.meshgrid(np.arange(6), np.arange(6), indexing='ij')
    r_flat = r_grid.ravel()
    c_flat = c_grid.ravel()

    # Repeat per block
    row_starts = np.repeat(np.arange(n_nodes), np.diff(K_bsr.block_row_ptr)) * 6
    col_starts = K_bsr.block_col_idx * 6

    coo_rows = (row_starts[:, None] + r_flat).ravel()
    coo_cols = (col_starts[:, None] + c_flat).ravel()
    coo_data = K_bsr.values.ravel().astype(dtype)

    # Build CSR matrix (sums duplicates if any)
    K_csr = sp.csr_matrix((coo_data, (coo_rows, coo_cols)), shape=(n_dofs, n_dofs))
    return K_csr


def bsr6x6_spmv_cpu(
    K_bsr: BSR6x6Matrix,
    x: np.ndarray,
    y: np.ndarray | None = None,
) -> np.ndarray:
    """
    Reference CPU SpMV implementation for BSR 6x6 format: y = K * x.

    Args:
        K_bsr: BSR6x6Matrix instance.
        x: [6*N] input vector.
        y: [6*N] output vector (optional pre-allocated buffer).

    Returns:
        [6*N] product vector y.
    """
    n_nodes = K_bsr.n_nodes
    n_dofs = n_nodes * 6
    x_vec = np.asarray(x, dtype=K_bsr.dtype).reshape(n_nodes, 6)

    if y is None:
        y_vec = np.zeros((n_nodes, 6), dtype=K_bsr.dtype)
    else:
        y_vec = y.reshape(n_nodes, 6)
        y_vec.fill(0.0)

    b_row_ptr = K_bsr.block_row_ptr
    b_col_idx = K_bsr.block_col_idx
    b_values = K_bsr.values.reshape(-1, 6, 6)

    for br in range(n_nodes):
        start = b_row_ptr[br]
        end = b_row_ptr[br + 1]
        for b in range(start, end):
            bc = b_col_idx[b]
            # y[br] += Ke[b] @ x[bc]
            y_vec[br] += b_values[b] @ x_vec[bc]

    return y_vec.ravel()
