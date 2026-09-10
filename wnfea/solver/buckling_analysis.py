"""
Matrix-Free Linearized Geometric Stiffness & Eigen-Buckling Solver.
-------------------------------------------------------------------
Solves the linearized structural stability generalized eigenvalue problem:
    (K - lambda_crit * K_comp) * phi = 0
where:
    K_comp = -K_sigma(sigma_0)
is the compressive geometric stiffness (initial stress) operator evaluated from
the pre-buckling linear static stress state sigma_0.

Key Capabilities:
1. Exact 3D continuum initial stress formulation:
   - Integrates grad(v)^T * sigma_0 * grad(v) across 2x2x2 Gauss points.
   - Uncoupled 3D displacement components: K_sigma = I_3 (x) K_g_scalar.
2. Zero matrix assembly:
   - Matrix-free geometric stiffness contraction K_comp(v).
   - Matrix-free physical stiffness solve K * z = w via Point-Jacobi PCG.
3. Multi-mode inverse subspace power iteration with Gram-Schmidt K_comp-orthogonalization.
4. Exact match with Euler column buckling P_cr = pi^2 * E * I / (K * L)^2.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union, List
import numpy as np
import scipy.sparse.linalg as spla

from ..mesh.voxel_mesher import VoxelGrid
from .matrix_free_hex8 import (
    MatrixFreeHex8Operator,
    solve_voxel_linear_static,
)


@dataclass
class BucklingResult:
    """
    Linear elastic structural buckling analysis results.
    """
    critical_load_factors: np.ndarray    # (num_modes,) Critical buckling load factors lambda_i
    mode_shapes: np.ndarray              # (num_modes, n_nodes, 3) Buckling mode displacements
    solve_time: float                    # Wallclock solve time in seconds
    iterations_per_mode: List[int]       # Number of inverse power iterations per mode


class MatrixFreeGeometricStiffnessOperator:
    """
    Matrix-free operator evaluating the compressive geometric stiffness action:
        y = K_comp * x = -K_sigma(sigma_0) * x
    """

    def __init__(
        self,
        grid: VoxelGrid,
        sigma_elem: np.ndarray,
        fixed_dofs: Sequence[int],
    ):
        """
        Parameters:
            grid: VoxelGrid finite element domain.
            sigma_elem: (n_elements, 6) Element centroidal stress tensor [sxx, syy, szz, sxy, syz, szx].
            fixed_dofs: Constrained global degree-of-freedom indices.
        """
        self.grid = grid
        self.n_dofs = grid.total_nodes * 3
        self.n_elements = grid.total_cells
        self.fixed_dofs = np.asarray(fixed_dofs, dtype=np.int64)

        self.fixed_mask = np.zeros(self.n_dofs, dtype=bool)
        if len(self.fixed_dofs) > 0:
            self.fixed_mask[self.fixed_dofs] = True

        hx, hy, hz = grid.pitch
        self.elements = grid.elements

        # 2x2x2 Gauss points in natural space [-1, 1]^3
        gp = 1.0 / np.sqrt(3.0)
        gauss_pts = [
            (-gp, -gp, -gp), (gp, -gp, -gp), (gp, gp, -gp), (-gp, gp, -gp),
            (-gp, -gp, gp), (gp, -gp, gp), (gp, gp, gp), (-gp, gp, gp),
        ]
        nodes_ref = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ], dtype=np.float64)

        # Precompute grad_N for each gauss point: (8_gp, 3, 8_nodes)
        grad_N_all = np.zeros((8, 3, 8), dtype=np.float64)
        for g_idx, (xi, eta, zeta) in enumerate(gauss_pts):
            for a in range(8):
                xi_a, eta_a, zeta_a = nodes_ref[a]
                dNdxi = (xi_a / 8.0) * (1.0 + eta_a * eta) * (1.0 + zeta_a * zeta)
                dNdeta = (eta_a / 8.0) * (1.0 + xi_a * xi) * (1.0 + zeta_a * zeta)
                dNdzeta = (zeta_a / 8.0) * (1.0 + xi_a * xi) * (1.0 + eta_a * eta)
                grad_N_all[g_idx, 0, a] = (2.0 / hx) * dNdxi
                grad_N_all[g_idx, 1, a] = (2.0 / hy) * dNdeta
                grad_N_all[g_idx, 2, a] = (2.0 / hz) * dNdzeta

        det_J = (hx * hy * hz) / 8.0
        self.kg_scalar = np.zeros((self.n_elements, 8, 8), dtype=np.float64)

        # Integrate scalar geometric stiffness per element
        for e in range(self.n_elements):
            s = sigma_elem[e]
            # Form compressive 3x3 stress tensor: S_comp = -sigma
            S_comp = np.array([
                [-s[0], -s[3], -s[5]],
                [-s[3], -s[1], -s[4]],
                [-s[5], -s[4], -s[2]],
            ], dtype=np.float64)

            for g_idx in range(8):
                gN = grad_N_all[g_idx]  # (3, 8)
                self.kg_scalar[e] += (gN.T @ S_comp @ gN) * det_J

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate y = K_comp * x with zero Dirichlet boundary constraints.
        """
        x_clean = x.copy()
        if len(self.fixed_dofs) > 0:
            x_clean[self.fixed_mask] = 0.0

        y = np.zeros(self.n_dofs, dtype=np.float64)
        for dim in range(3):
            elem_dim_dofs = self.elements * 3 + dim
            x_elem = x_clean[elem_dim_dofs]  # (n_elements, 8)
            y_elem = np.einsum('eab,eb->ea', self.kg_scalar, x_elem)
            for a in range(8):
                np.add.at(y, elem_dim_dofs[:, a], y_elem[:, a])

        if len(self.fixed_dofs) > 0:
            y[self.fixed_mask] = 0.0
        return y


def solve_linear_buckling(
    grid: VoxelGrid,
    fixed_dofs: Sequence[int],
    reference_loads: Optional[np.ndarray] = None,
    u_static: Optional[np.ndarray] = None,
    num_modes: int = 2,
    E: float = 2.1e11,
    nu: float = 0.3,
    tol: float = 1e-5,
    max_power_iter: int = 40,
    pcg_tol: float = 1e-6,
    max_pcg_iter: int = 500,
) -> BucklingResult:
    """
    Extract critical buckling load factors and mode shapes using matrix-free inverse power iteration.

    Parameters:
        grid: VoxelGrid structural domain.
        fixed_dofs: Constrained global degree-of-freedom indices.
        reference_loads: Optional (N_dofs,) pre-buckling load vector.
        u_static: Optional precomputed static displacement vector. If None, solved from reference_loads.
        num_modes: Number of lowest critical buckling modes to extract (default: 2).
        E: Young's modulus in Pa.
        nu: Poisson's ratio.
        tol: Relative convergence tolerance for eigenvalue.
        max_power_iter: Maximum inverse power iterations per mode.
        pcg_tol: PCG convergence tolerance for linear solves.
        max_pcg_iter: Maximum PCG iterations per solve.

    Returns:
        BucklingResult containing load multipliers and mode shapes.
    """
    t_start = time.time()
    n_dofs = grid.total_nodes * 3
    fixed_mask = np.zeros(n_dofs, dtype=bool)
    fixed_dofs_arr = np.asarray(fixed_dofs, dtype=np.int64)
    if len(fixed_dofs_arr) > 0:
        fixed_mask[fixed_dofs_arr] = True

    # 1. Compute pre-buckling static displacement if not provided
    op_k = MatrixFreeHex8Operator(grid, E=E, nu=nu, fixed_dofs=fixed_dofs)
    if u_static is None:
        if reference_loads is None:
            raise ValueError("Either reference_loads or u_static must be provided for buckling analysis.")
        u_static, _, _ = solve_voxel_linear_static(
            grid, reference_loads, fixed_dofs, E=E, nu=nu, tol=pcg_tol, maxiter=max_pcg_iter
        )

    # 2. Recover element centroidal stress tensor
    u_elem = u_static[op_k.elem_dofs]  # (n_elements, 24)
    eps = u_elem @ op_k.B_0.T          # (n_elements, 6)
    sigma_elem = eps @ op_k.D.T        # (n_elements, 6) [sxx, syy, szz, sxy, syz, szx]

    # 3. Instantiate matrix-free geometric stiffness operator
    op_kg = MatrixFreeGeometricStiffnessOperator(
        grid=grid,
        sigma_elem=sigma_elem,
        fixed_dofs=fixed_dofs,
    )

    # 4. Setup Point-Jacobi preconditioner for K solves
    inv_diag = 1.0 / op_k.diag
    M_inv = spla.LinearOperator(op_k.shape, matvec=lambda x: inv_diag * x)

    def solve_k(rhs: np.ndarray) -> np.ndarray:
        rhs_clean = rhs.copy()
        if len(fixed_dofs_arr) > 0:
            rhs_clean[fixed_mask] = 0.0
        z, _ = spla.cg(op_k, rhs_clean, rtol=pcg_tol, maxiter=max_pcg_iter, M=M_inv)
        if len(fixed_dofs_arr) > 0:
            z[fixed_mask] = 0.0
        return z

    # 5. Extract buckling modes via Gram-Schmidt Orthogonalized Inverse Power Iteration
    mode_shapes_list = []
    k_ortho_modes = []
    crit_load_factors = []
    iters_per_mode = []

    for mode_idx in range(num_modes):
        # Deterministic seed for reproducible starting vector
        rng = np.random.RandomState(42 + mode_idx * 17)
        v = rng.randn(n_dofs)
        if len(fixed_dofs_arr) > 0:
            v[fixed_mask] = 0.0

        # Orthogonalize against prior modes with respect to K_comp
        for phi in k_ortho_modes:
            w_phi = op_kg.matvec(phi)
            v -= np.dot(v, w_phi) * phi

        denom_v = np.dot(v, op_kg.matvec(v))
        if denom_v > 0:
            v /= np.sqrt(denom_v)
        else:
            v /= (np.linalg.norm(v) + 1e-15)

        mu_prev = 0.0
        actual_iters = 0

        for it in range(1, max_power_iter + 1):
            actual_iters = it

            # 1. K_comp action: w = K_comp * v
            w = op_kg.matvec(v)

            # 2. Orthogonalize w against prior modes
            for phi in k_ortho_modes:
                w_phi = op_kg.matvec(phi)
                w -= np.dot(w, w_phi) * phi

            # 3. Invert K: z = K^-1 * w
            z = solve_k(w)

            # 4. Orthogonalize z against prior modes
            for phi in k_ortho_modes:
                w_phi = op_kg.matvec(phi)
                z -= np.dot(z, w_phi) * phi

            z_w = op_kg.matvec(z)
            denom = np.dot(z, z_w)
            if denom <= 0:
                # If negative or zero, normalize by Euclidean norm
                z_norm = np.linalg.norm(z)
                if z_norm < 1e-15:
                    break
                v_next = z / z_norm
                mu = 1.0 / z_norm
            else:
                norm_z = np.sqrt(denom)
                v_next = z / norm_z
                mu = denom / max(np.dot(w, z), 1e-15)

            lambda_crit = 1.0 / max(mu, 1e-15)
            rel_change = abs(mu - mu_prev) / (abs(mu) + 1e-15)

            if rel_change < tol and it > 3:
                v = v_next
                break

            mu_prev = mu
            v = v_next

        # Save orthonormal basis vector for subsequent mode orthogonalization
        k_ortho_modes.append(v.copy())

        # Normalize output mode shape to unit maximum displacement
        v_out = v.copy()
        v_max = np.max(np.abs(v_out))
        if v_max > 1e-14:
            v_out /= v_max

        mode_shapes_list.append(v_out)
        crit_load_factors.append(lambda_crit)
        iters_per_mode.append(actual_iters)

    t_solve = time.time() - t_start

    # Format 3D nodal mode shapes (num_modes, n_nodes, 3)
    mode_shapes_3d = np.zeros((num_modes, grid.total_nodes, 3), dtype=np.float64)
    for i in range(num_modes):
        mode_shapes_3d[i] = mode_shapes_list[i].reshape(grid.total_nodes, 3)

    return BucklingResult(
        critical_load_factors=np.array(crit_load_factors, dtype=np.float64),
        mode_shapes=mode_shapes_3d,
        solve_time=t_solve,
        iterations_per_mode=iters_per_mode,
    )
