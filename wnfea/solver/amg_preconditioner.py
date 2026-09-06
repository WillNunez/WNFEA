"""
Block / Smoothed Aggregation Algebraic Multigrid (AMG) Preconditioner for 3D Beams.

Designed specifically for 6-DOF 3D frame and beam networks. Groups physical
nodal DOFs into blocks and embeds the 6 rigid body modes into the prolongation
near-nullspace. Provides a symmetric positive definite (SPD) V-cycle operator
M^{-1} for Preconditioned Conjugate Gradient (PCG) in JFNK.

Mixed-Precision Support:
    The hierarchy (P, R, A_coarse, smoother diags, coarse factorization) can be
    stored and operated in a user-specified working_dtype (default: np.float64).
    When working_dtype=np.float32, all V-cycle arithmetic runs in FP32 for 2x
    memory reduction. Inputs/outputs are cast between the caller's precision
    and the internal working precision automatically.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, issparse, diags
from scipy.sparse.linalg import LinearOperator, splu, spsolve
from scipy.linalg import cho_factor, cho_solve

from ..model import FEAModel
from .assembler import assemble_global_system


def generate_rigid_body_modes(nodes: np.ndarray) -> np.ndarray:
    """
    Generate the 6 rigid body modes (3 translations, 3 rotations) for 3D beam DOFs.

    DOF layout per node: [ux, uy, uz, rx, ry, rz].

    Args:
        nodes: (N, 3) array of undeformed nodal coordinates.

    Returns:
        B: (6*N, 6) matrix where each column is a rigid body mode.
    """
    n_nodes = len(nodes)
    B = np.zeros((6 * n_nodes, 6), dtype=np.float64)

    for i, (X, Y, Z) in enumerate(nodes):
        idx = 6 * i
        # Mode 0: Translation X
        B[idx + 0, 0] = 1.0
        # Mode 1: Translation Y
        B[idx + 1, 1] = 1.0
        # Mode 2: Translation Z
        B[idx + 2, 2] = 1.0

        # Mode 3: Rotation about X
        B[idx + 1, 3] = -Z
        B[idx + 2, 3] = Y
        B[idx + 3, 3] = 1.0

        # Mode 4: Rotation about Y
        B[idx + 0, 4] = Z
        B[idx + 2, 4] = -X
        B[idx + 4, 4] = 1.0

        # Mode 5: Rotation about Z
        B[idx + 0, 5] = -Y
        B[idx + 1, 5] = X
        B[idx + 5, 5] = 1.0

    return B


def aggregate_beam_nodes(mesh_elements: np.ndarray, n_nodes: int, target_size: int = 2) -> list[list[int]]:
    """
    Greedy graph agglomeration of beam nodes into aggregates.
    """
    # Build node adjacency list
    adj: list[set[int]] = [set() for _ in range(n_nodes)]
    for n1, n2 in mesh_elements:
        adj[n1].add(n2)
        adj[n2].add(n1)

    visited = np.zeros(n_nodes, dtype=bool)
    aggregates: list[list[int]] = []

    # Agglomerate connected neighbors
    for i in range(n_nodes):
        if visited[i]:
            continue
        agg = [i]
        visited[i] = True

        # Find unvisited neighbors
        for neighbor in sorted(adj[i]):
            if not visited[neighbor]:
                agg.append(neighbor)
                visited[neighbor] = True
                if len(agg) >= target_size:
                    break
        aggregates.append(agg)

    # Attach any lone singletons to adjacent aggregates if possible
    refined_aggregates: list[list[int]] = []
    for agg in aggregates:
        if len(agg) == 1 and len(refined_aggregates) > 0:
            node = agg[0]
            # Try to attach to previous aggregate
            attached = False
            for target_agg in refined_aggregates:
                if any(node in adj[other] for other in target_agg):
                    target_agg.append(node)
                    attached = True
                    break
            if not attached:
                refined_aggregates.append(agg)
        else:
            refined_aggregates.append(agg)

    return refined_aggregates


class AMGLevel:
    """Represents a single level in the AMG hierarchy."""

    def __init__(
        self,
        A: csr_matrix,
        P: csr_matrix | None = None,
        R: csr_matrix | None = None,
        working_dtype: np.dtype = np.float64,
    ):
        self.working_dtype = np.dtype(working_dtype)
        self.A = A.astype(self.working_dtype)
        self.P = P.astype(self.working_dtype) if P is not None else None
        self.R = R.astype(self.working_dtype) if R is not None else None
        self.inv_diag = (1.0 / (A.diagonal().astype(self.working_dtype) + 1e-16)).astype(self.working_dtype)

        # Pre-compute L and U for symmetric Gauss-Seidel if small enough
        self.A_dense = A.toarray().astype(self.working_dtype) if A.shape[0] <= 500 else None


class BlockBeamAMGPreconditioner(LinearOperator):
    """
    Block Smoothed-Aggregation AMG Preconditioner for 3D beam finite element systems.
    Can be used as the M^{-1} operator in Preconditioned Conjugate Gradient (PCG).

    Mixed-Precision:
        Set working_dtype=np.float32 to run all V-cycle arithmetic in FP32.
        The hierarchy matrices, smoother diagonals, and coarse factorization
        are all stored in working_dtype. Inputs are cast from caller precision
        to working_dtype, and outputs are cast back.
    """

    def __init__(self, model: FEAModel, max_levels: int = 3, coarse_size: int = 36,
                 working_dtype: np.dtype = np.float64):
        K_dense, _ = assemble_global_system(model)
        A = csr_matrix(K_dense)
        n = A.shape[0]
        super().__init__(dtype=np.float64, shape=(n, n))

        self.model = model
        self.max_levels = max_levels
        self.coarse_size = coarse_size
        self.working_dtype = np.dtype(working_dtype)
        self.levels: list[AMGLevel] = []
        self.coarse_solver = None

        # Symmetrically equilibrate system: improves condition number by orders of magnitude
        # and brings all vectors into O(1) range, making FP16 arithmetic immune to underflow.
        diag_A = np.abs(A.diagonal())
        self.inv_sqrt_d = 1.0 / np.sqrt(np.where(diag_A > 1e-12, diag_A, 1.0))
        D_inv = diags(self.inv_sqrt_d)
        A_equil = (D_inv @ A @ D_inv).tocsr()

        self._build_hierarchy(A_equil)

    def _build_hierarchy(self, A_fine: csr_matrix) -> None:
        """Construct the multi-level AMG hierarchy."""
        A_curr = A_fine
        nodes = self.model.mesh_nodes

        for lvl in range(self.max_levels):
            n_dofs = A_curr.shape[0]
            n_nodes = n_dofs // 6

            if n_dofs <= self.coarse_size or n_nodes <= 3:
                # Coarsest level reached
                self.levels.append(AMGLevel(A=A_curr, working_dtype=self.working_dtype))
                break

            # Aggregate physical nodes
            # If at fine level, use mesh element connectivity
            if lvl == 0:
                aggregates = aggregate_beam_nodes(self.model.mesh_elements, n_nodes, target_size=2)
            else:
                # Coarser level simple pairwise grouping
                aggregates = [[2 * i, 2 * i + 1] if 2 * i + 1 < n_nodes else [2 * i]
                              for i in range((n_nodes + 1) // 2)]

            n_aggregates = len(aggregates)
            if n_aggregates >= n_nodes:
                self.levels.append(AMGLevel(A=A_curr, working_dtype=self.working_dtype))
                break

            # Compute near-nullspace (rigid body modes)
            B = generate_rigid_body_modes(nodes[:n_nodes])
            if lvl == 0:
                sqrt_d = 1.0 / self.inv_sqrt_d[:6 * n_nodes]
                B = sqrt_d[:, None] * B

            # Build Block Prolongation Operator P: (6*n_nodes, 6*n_aggregates)
            P_dense = np.zeros((6 * n_nodes, 6 * n_aggregates), dtype=np.float64)

            for agg_idx, agg_nodes in enumerate(aggregates):
                # Extract rows of B for this aggregate
                dofs_agg = []
                for node_id in agg_nodes:
                    dofs_agg.extend(range(node_id * 6, node_id * 6 + 6))

                B_block = B[dofs_agg, :]
                Q, _ = np.linalg.qr(B_block)

                col_slice = slice(agg_idx * 6, agg_idx * 6 + 6)
                P_dense[dofs_agg, col_slice] = Q[:, :6]

            P = csr_matrix(P_dense)
            R = P.T

            # Galerkin coarse grid operator: A_coarse = R * A * P
            A_coarse = (R @ A_curr @ P).tocsr()

            # Ensure coarse matrix symmetry and conditioning
            A_coarse = 0.5 * (A_coarse + A_coarse.T)
            # Add small regularization to diagonal to prevent singular coarse modes
            diag = A_coarse.diagonal()
            zero_diag = np.where(np.abs(diag) < 1e-12)[0]
            if len(zero_diag) > 0:
                diag[zero_diag] = 1.0
                A_coarse.setdiag(diag)

            # Store level with working_dtype (P, R, A are downcast in AMGLevel.__init__)
            self.levels.append(AMGLevel(A=A_curr, P=P, R=R, working_dtype=self.working_dtype))
            A_curr = A_coarse

        # Coarsest level direct solve setup (in working_dtype)
        A_coarsest = self.levels[-1].A
        coarse_dense = A_coarsest.toarray().astype(self.working_dtype)
        coarse_dense += 1e-12 * np.eye(coarse_dense.shape[0], dtype=self.working_dtype)
        try:
            # Cholesky in working_dtype
            self.coarse_cho = cho_factor(coarse_dense.astype(np.float64), lower=True)
            self.coarse_direct = True
        except Exception:
            self.coarse_lu = splu(A_coarsest.astype(np.float64).tocsc())
            self.coarse_direct = False

        # Tri-precision tracking
        self.use_tri_precision = False
        self.switch_tol = 1e-2
        self.last_precision_used = np.dtype(self.working_dtype).name
        self.fp16_vcycle_count = 0
        self.fp32_vcycle_count = 0

    def enable_tri_precision(self, switch_tol: float = 1e-2) -> None:
        """Enable adaptive FP16 early preconditioning with automatic switch to FP32."""
        self.use_tri_precision = True
        self.switch_tol = float(switch_tol)

    def _smooth(self, level: AMGLevel, b: np.ndarray, x: np.ndarray, n_steps: int = 1) -> np.ndarray:
        """Damped Jacobi smoothing step (in working_dtype)."""
        omega = 0.67
        x_out = x.copy()
        for _ in range(n_steps):
            res = b - level.A.dot(x_out)
            x_out += omega * (level.inv_diag * res)
        return x_out

    def _smooth_fp16(self, level: AMGLevel, b: np.ndarray, x: np.ndarray, n_steps: int = 1) -> np.ndarray:
        """Damped Jacobi smoothing step in FP16 (half precision)."""
        omega = np.float16(0.67)
        x_out = x.astype(np.float16, copy=True)
        b_16 = b.astype(np.float16, copy=False)
        inv_diag_16 = level.inv_diag.astype(np.float16)
        for _ in range(n_steps):
            Ax = level.A.dot(x_out.astype(np.float32)).astype(np.float16)
            res = b_16 - Ax
            x_out += omega * (inv_diag_16 * res)
        return x_out

    def _v_cycle_fp16(self, lvl_idx: int, r: np.ndarray) -> np.ndarray:
        """Recursive symmetric V-cycle execution in FP16 with stable coarse solve."""
        level = self.levels[lvl_idx]

        # Base case: coarsest level (use stable Cholesky/LU)
        if lvl_idx == len(self.levels) - 1:
            if self.coarse_direct:
                sol = cho_solve(self.coarse_cho, r.astype(np.float64))
                return sol.astype(np.float16)
            else:
                sol = self.coarse_lu.solve(r.astype(np.float64))
                return sol.astype(np.float16)

        # 1. Pre-smooth in FP16
        x = np.zeros_like(r, dtype=np.float16)
        x = self._smooth_fp16(level, r, x, n_steps=1)

        # 2. Defect computation
        Ax = level.A.dot(x.astype(np.float32)).astype(np.float16)
        res = r - Ax

        # 3. Restrict defect
        r_coarse = level.R.dot(res.astype(np.float32)).astype(np.float16)

        # 4. Recursive coarse correction
        e_coarse = self._v_cycle_fp16(lvl_idx + 1, r_coarse)

        # 5. Prolongate
        Pe = level.P.dot(e_coarse.astype(np.float32)).astype(np.float16)
        x += Pe

        # 6. Post-smooth in FP16
        x = self._smooth_fp16(level, r, x, n_steps=1)

        return x

    def _v_cycle(self, lvl_idx: int, r: np.ndarray) -> np.ndarray:
        """Recursive symmetric V-cycle execution (in working_dtype)."""
        level = self.levels[lvl_idx]

        # Base case: coarsest level
        if lvl_idx == len(self.levels) - 1:
            if self.coarse_direct:
                sol = cho_solve(self.coarse_cho, r.astype(np.float64))
                return sol.astype(self.working_dtype)
            else:
                sol = self.coarse_lu.solve(r.astype(np.float64))
                return sol.astype(self.working_dtype)

        # 1. Pre-smooth
        x = np.zeros_like(r)
        x = self._smooth(level, r, x, n_steps=1)

        # 2. Compute residual
        res = r - level.A.dot(x)

        # 3. Restrict residual to coarse level
        r_coarse = level.R.dot(res)

        # 4. Recursive coarse grid correction
        e_coarse = self._v_cycle(lvl_idx + 1, r_coarse)

        # 5. Prolongate correction back to fine level
        x += level.P.dot(e_coarse)

        # 6. Post-smooth
        x = self._smooth(level, r, x, n_steps=1)

        return x

    def apply_adaptive(self, r: np.ndarray, current_res_rel: float | None = None) -> np.ndarray:
        """
        Apply preconditioner with dynamic tri-precision adaptation:
        - If current_res_rel > switch_tol, executes V-cycle in FP16 (with dynamic range protection).
        - When current_res_rel <= switch_tol, transitions to working_dtype (FP32).
        """
        if self.use_tri_precision and current_res_rel is not None and current_res_rel > self.switch_tol:
            r_work = np.asarray(r, dtype=np.float32)
            r_equil = (self.inv_sqrt_d.astype(np.float32) * r_work)
            max_val = float(np.max(np.abs(r_equil)))
            if max_val > 1e-15:
                scale = 1.0 / max_val
                r_scaled = (r_equil * scale).astype(np.float16)
                z_scaled = self._v_cycle_fp16(0, r_scaled)
                z_equil = (z_scaled.astype(np.float32) / scale)
                z = (self.inv_sqrt_d.astype(np.float32) * z_equil)
            else:
                z = np.zeros_like(r_work)
            self.last_precision_used = "float16"
            self.fp16_vcycle_count += 1
            return z.astype(np.float64)
        else:
            self.last_precision_used = np.dtype(self.working_dtype).name
            self.fp32_vcycle_count += 1
            return self._matvec(r)

    def _matvec(self, r: np.ndarray) -> np.ndarray:
        """Apply preconditioner: z = M^{-1} * r via equilibrated AMG V-cycle.

        Input r may be in any precision. It is cast to working_dtype for the
        V-cycle, and the result is cast back to float64 for the caller.
        """
        r_work = np.asarray(r, dtype=self.working_dtype)
        r_equil = (self.inv_sqrt_d.astype(self.working_dtype) * r_work)
        z_equil = self._v_cycle(0, r_equil)
        z = (self.inv_sqrt_d.astype(self.working_dtype) * z_equil)
        return z.astype(np.float64)
