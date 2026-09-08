"""
Two-Level Geometric p-Multigrid Preconditioner for Matrix-Free C3D10 Quadratic Solids.

Leverages the hierarchical geometric structure of quadratic 10-node tetrahedra:
- Coarse space V_H: 4 corner vertex nodes (linear tetrahedral continuum space).
- Fine space V_h: 4 corner vertex + 6 mid-edge nodes (quadratic continuum space).
- Canonical Prolongation P: Exact quadratic interpolation (identity at vertices, 0.5/0.5 on edges).
- Restriction R = P^T.
- Coarse operator K_H = P^T K_h P: Assembled elementally via Pe^T Ke Pe in O(N) without ever
  forming the global fine stiffness matrix.
- Smoothing: Damped Jacobi fine-scale smoother damping high-frequency edge oscillations.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, splu
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .matrix_free_c3d10 import MatrixFreeC3D10Operator


class PMultigridC3D10Preconditioner(LinearOperator):
    """
    Two-Level Geometric p-Multigrid Preconditioner M^{-1}.
    
    Acts as a symmetric positive-definite (SPD) preconditioner for Conjugate Gradient solvers
    on C3D10 solid finite element models.
    """

    def __init__(
        self,
        mf_op: MatrixFreeC3D10Operator,
        omega: float = 0.67,
        pre_sweeps: int = 1,
        post_sweeps: int = 1,
    ):
        self.mf_op = mf_op
        self.model = mf_op.model
        self.omega = omega
        self.pre_sweeps = pre_sweeps
        self.post_sweeps = post_sweeps
        self.n_dofs = mf_op.n_dofs
        self.shape = (self.n_dofs, self.n_dofs)
        self.dtype = mf_op.dtype

        self.inv_diag_K = mf_op.inv_diag_K
        self.fixed_dofs = mf_op.fixed_dofs
        self.apply_bcs = mf_op.apply_bcs

        self._setup_geometric_hierarchy()
        super().__init__(dtype=self.dtype, shape=self.shape)

    def _setup_geometric_hierarchy(self) -> None:
        elems = self.model.solid_elements
        n_fine_nodes = len(self.model.mesh_nodes)

        # 1. Identify corner vertex nodes
        self.vertex_nodes = np.unique(elems[:, 0:4])
        self.n_coarse_nodes = len(self.vertex_nodes)
        self.n_coarse_dofs = self.n_coarse_nodes * 3

        # Mapping: global_node_id -> coarse_node_id (-1 if mid-edge)
        self.v_map = np.full(n_fine_nodes, -1, dtype=np.int32)
        self.v_map[self.vertex_nodes] = np.arange(self.n_coarse_nodes, dtype=np.int32)

        # 2. Extract edge pairs for all mid-edge nodes
        edge_defs = [
            (4, 0, 1),
            (5, 1, 2),
            (6, 2, 0),
            (7, 0, 3),
            (8, 1, 3),
            (9, 2, 3),
        ]
        mid_edge_endpoints: dict[int, tuple[int, int]] = {}
        for e in elems:
            for m_idx, v_a_idx, v_b_idx in edge_defs:
                m = int(e[m_idx])
                if m not in mid_edge_endpoints:
                    mid_edge_endpoints[m] = (int(e[v_a_idx]), int(e[v_b_idx]))

        # 3. Construct sparse node-wise prolongation matrix P_node of shape (n_fine_nodes, n_coarse_nodes)
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        for k in range(n_fine_nodes):
            c_idx = self.v_map[k]
            if c_idx >= 0:
                rows.append(k)
                cols.append(c_idx)
                vals.append(1.0)
            else:
                v_a, v_b = mid_edge_endpoints[k]
                rows.extend([k, k])
                cols.extend([self.v_map[v_a], self.v_map[v_b]])
                vals.extend([0.5, 0.5])

        self.P_node = csr_matrix(
            (vals, (rows, cols)), shape=(n_fine_nodes, self.n_coarse_nodes), dtype=np.float64
        )
        self.R_node = self.P_node.T.tocsr()

        # 4. Construct coarse elemental stiffness matrix Ke_coarse = Pe.T @ Ke @ Pe
        P_elem_node = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5, 0.0],
            [0.5, 0.0, 0.0, 0.5],
            [0.0, 0.5, 0.0, 0.5],
            [0.0, 0.0, 0.5, 0.5],
        ], dtype=np.float64)
        Pe = np.kron(P_elem_node, np.eye(3))  # (30, 12)

        if self.mf_op.Ke_batch is not None:
            Ke_batch = self.mf_op.Ke_batch
        else:
            from ..elements.c3d10 import element_stiffness_c3d10
            n_elements = len(elems)
            Ke_batch = np.empty((n_elements, 30, 30), dtype=np.float64)
            for e_idx, n_idx in enumerate(elems):
                c = self.model.mesh_nodes[n_idx]
                E = self.mf_op.props_c[e_idx, 0]
                nu = self.mf_op.props_c[e_idx, 1]
                Ke_batch[e_idx] = element_stiffness_c3d10(c, E, nu)

        Ke_coarse = Pe.T @ Ke_batch @ Pe  # (E, 12, 12)

        # 5. Assemble coarse stiffness matrix K_H in CSR format
        elem_coarse_nodes = self.v_map[elems[:, 0:4]]
        elem_coarse_dofs = np.empty((len(elems), 12), dtype=np.int32)
        for i in range(4):
            for d in range(3):
                elem_coarse_dofs[:, i * 3 + d] = elem_coarse_nodes[:, i] * 3 + d

        r_idx = np.repeat(elem_coarse_dofs, 12, axis=1).ravel()
        c_idx = np.tile(elem_coarse_dofs, (1, 12)).ravel()
        K_H = csr_matrix((Ke_coarse.ravel(), (r_idx, c_idx)), shape=(self.n_coarse_dofs, self.n_coarse_dofs))

        # 6. Apply coarse Dirichlet boundary conditions
        coarse_fixed_list: list[int] = []
        if self.apply_bcs and len(self.fixed_dofs) > 0:
            for fd in self.fixed_dofs:
                nid = fd // 3
                d = fd % 3
                if nid < n_fine_nodes and self.v_map[nid] >= 0:
                    coarse_fixed_list.append(self.v_map[nid] * 3 + d)

        self.coarse_fixed = np.array(sorted(set(coarse_fixed_list)), dtype=np.int32)

        if len(self.coarse_fixed) > 0:
            K_H = K_H.tolil()
            for cfd in self.coarse_fixed:
                K_H[cfd, :] = 0.0
                K_H[:, cfd] = 0.0
                K_H[cfd, cfd] = 1.0
            K_H = K_H.tocsr()

        # 7. Pre-factorize coarse problem via SuperLU sparse factorization
        self.coarse_lu = splu(K_H.tocsc())

    def prolongate(self, u_H: np.ndarray) -> np.ndarray:
        """Interpolate coarse vertex field u_H to fine quadratic field u_h."""
        u_H_3d = u_H.reshape(self.n_coarse_nodes, 3)
        u_h_3d = self.P_node @ u_H_3d
        return u_h_3d.ravel().astype(self.dtype)

    def restrict(self, r_h: np.ndarray) -> np.ndarray:
        """Restrict fine residual field r_h to coarse vertex field r_H = P^T r_h."""
        n_fine_nodes = len(self.model.mesh_nodes)
        r_h_3d = r_h.reshape(n_fine_nodes, 3)
        r_H_3d = self.R_node @ r_h_3d
        r_H = r_H_3d.ravel()
        if len(self.coarse_fixed) > 0:
            r_H[self.coarse_fixed] = 0.0
        return r_H.astype(self.dtype)

    def _matvec(self, r: np.ndarray) -> np.ndarray:
        """
        Apply two-level symmetric V-cycle:
        1. Pre-smooth with damped Jacobi.
        2. Compute fine-scale defect d = r - K z.
        3. Restrict defect to coarse grid r_H = R d.
        4. Direct solve on coarse grid e_H = K_H^{-1} r_H.
        5. Prolongate coarse correction z += P e_H.
        6. Post-smooth with damped Jacobi.
        """
        r_arr = np.asarray(r, dtype=self.dtype)
        z = np.zeros_like(r_arr)

        # Pre-smoothing sweeps
        for _ in range(self.pre_sweeps):
            res = r_arr - self.mf_op @ z
            if self.apply_bcs and len(self.fixed_dofs) > 0:
                res[self.fixed_dofs] = 0.0
            z += self.omega * (res * self.inv_diag_K)
            if self.apply_bcs and len(self.fixed_dofs) > 0:
                z[self.fixed_dofs] = 0.0

        # Fine defect & restriction
        d = r_arr - self.mf_op @ z
        if self.apply_bcs and len(self.fixed_dofs) > 0:
            d[self.fixed_dofs] = 0.0
        r_H = self.restrict(d)

        # Coarse solve
        e_H = self.coarse_lu.solve(r_H)
        if len(self.coarse_fixed) > 0:
            e_H[self.coarse_fixed] = 0.0

        # Prolongation
        z += self.prolongate(e_H)
        if self.apply_bcs and len(self.fixed_dofs) > 0:
            z[self.fixed_dofs] = 0.0

        # Post-smoothing sweeps
        for _ in range(self.post_sweeps):
            res = r_arr - self.mf_op @ z
            if self.apply_bcs and len(self.fixed_dofs) > 0:
                res[self.fixed_dofs] = 0.0
            z += self.omega * (res * self.inv_diag_K)
            if self.apply_bcs and len(self.fixed_dofs) > 0:
                z[self.fixed_dofs] = 0.0

        return z
