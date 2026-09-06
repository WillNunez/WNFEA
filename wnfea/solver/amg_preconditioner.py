"""
Block / Smoothed Aggregation Algebraic Multigrid (AMG) Preconditioner for 3D Beams.

Designed specifically for 6-DOF 3D frame and beam networks. Groups physical
nodal DOFs into blocks and embeds the 6 rigid body modes into the prolongation
near-nullspace. Provides a symmetric positive definite (SPD) V-cycle operator
M^{-1} for Preconditioned Conjugate Gradient (PCG) in JFNK.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, issparse
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
    ):
        self.A = A
        self.P = P  # Prolongation: coarse -> fine
        self.R = R  # Restriction: fine -> coarse (typically P.T)
        self.inv_diag = 1.0 / (A.diagonal() + 1e-16)

        # Pre-compute L and U for symmetric Gauss-Seidel if small enough
        self.A_dense = A.toarray() if A.shape[0] <= 500 else None


class BlockBeamAMGPreconditioner(LinearOperator):
    """
    Block Smoothed-Aggregation AMG Preconditioner for 3D beam finite element systems.
    Can be used as the M^{-1} operator in Preconditioned Conjugate Gradient (PCG).
    """

    def __init__(self, model: FEAModel, max_levels: int = 3, coarse_size: int = 36):
        K_dense, _ = assemble_global_system(model)
        A = csr_matrix(K_dense)
        n = A.shape[0]
        super().__init__(dtype=np.float64, shape=(n, n))

        self.model = model
        self.max_levels = max_levels
        self.coarse_size = coarse_size
        self.levels: list[AMGLevel] = []
        self.coarse_solver = None

        self._build_hierarchy(A)

    def _build_hierarchy(self, A_fine: csr_matrix) -> None:
        """Construct the multi-level AMG hierarchy."""
        A_curr = A_fine
        nodes = self.model.mesh_nodes

        for lvl in range(self.max_levels):
            n_dofs = A_curr.shape[0]
            n_nodes = n_dofs // 6

            if n_dofs <= self.coarse_size or n_nodes <= 3:
                # Coarsest level reached
                self.levels.append(AMGLevel(A=A_curr))
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
                self.levels.append(AMGLevel(A=A_curr))
                break

            # Compute near-nullspace (rigid body modes)
            B = generate_rigid_body_modes(nodes[:n_nodes])

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

            self.levels.append(AMGLevel(A=A_curr, P=P, R=R))
            A_curr = A_coarse

        # Coarsest level direct solve setup
        A_coarsest = self.levels[-1].A
        coarse_dense = A_coarsest.toarray()
        coarse_dense += 1e-12 * np.eye(coarse_dense.shape[0])
        try:
            self.coarse_cho = cho_factor(coarse_dense, lower=True)
            self.coarse_direct = True
        except Exception:
            self.coarse_lu = splu(A_coarsest.tocsc())
            self.coarse_direct = False

    def _smooth(self, level: AMGLevel, b: np.ndarray, x: np.ndarray, n_steps: int = 1) -> np.ndarray:
        """Damped Jacobi smoothing step."""
        omega = 0.67
        x_out = x.copy()
        for _ in range(n_steps):
            res = b - level.A.dot(x_out)
            x_out += omega * (level.inv_diag * res)
        return x_out

    def _v_cycle(self, lvl_idx: int, r: np.ndarray) -> np.ndarray:
        """Recursive symmetric V-cycle execution."""
        level = self.levels[lvl_idx]

        # Base case: coarsest level
        if lvl_idx == len(self.levels) - 1:
            if self.coarse_direct:
                return cho_solve(self.coarse_cho, r)
            else:
                return self.coarse_lu.solve(r)

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

    def _matvec(self, r: np.ndarray) -> np.ndarray:
        """Apply preconditioner: z = M^{-1} * r via AMG V-cycle."""
        r_arr = np.asarray(r, dtype=np.float64)
        return self._v_cycle(0, r_arr)
