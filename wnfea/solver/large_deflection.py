"""
Total Lagrangian Large-Deflection & Geometric Non-Linearity Solver for Metals.
-----------------------------------------------------------------------------
Implements:
1. Exact Total Lagrangian kinematic formulation:
      - Deformation gradient F = I + grad_0(u).
      - Green-Lagrange strain tensor E = 1/2 (F^T F - I) = eps_lin + eps_nl.
      - Second Piola-Kirchhoff (PK2) stress tensor S.
2. Full Tangent Stiffness Decomposition:
      - K_t(u) = K_L(u) + K_sigma(S), where K_sigma is the initial stress (geometric)
        stiffness matrix capturing large rotation, stress stiffening, and snap-through.
3. Elasto-Plastic Material Coupling:
      - Interfaces directly with MetalPlasticMaterial and radial_return_mapping (Phase 22).
      - Seamless support for Aluminum 6061-T6, Al 7075-T6, Ti-6Al-4V, and S355 steel.
4. Incremental Newton-Raphson Solver:
      - Quadratic equilibrium convergence with line search / displacement increment control.
      - Cauchy true stress recovery: sigma = (1 / det(F)) * F * S * F^T.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence, List, Tuple
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from ..mesh.voxel_mesher import VoxelGrid
from ..materials.plasticity import (
    MetalPlasticMaterial,
    PlasticHistoryState,
    radial_return_mapping,
    _P_DEV,
    _M_OUTER,
)
from .plasticity_solver import _precompute_hex8_2x2x2_B


@dataclass
class LargeDeflectionStepResult:
    """Results at a converged load increment under large deflections."""
    step_index: int
    load_factor: float
    displacements: np.ndarray             # (N_dof,) displacement vector u
    pk2_stresses: np.ndarray              # (N_elem, 6) 2nd Piola-Kirchhoff stress in Voigt
    cauchy_stresses: np.ndarray           # (N_elem, 6) Cauchy true stress in Voigt
    green_lagrange_strains: np.ndarray    # (N_elem, 6) Green-Lagrange strain in Voigt
    plastic_strains: np.ndarray           # (N_elem, 6) plastic strain in Voigt
    accumulated_plastic_strain: np.ndarray # (N_elem,) equivalent plastic strain alpha_p
    yielded_mask: np.ndarray              # (N_elem,) bool mask of yielded elements
    iterations: int                       # Newton-Raphson iterations
    converged: bool                       # Convergence flag
    final_residual: float                 # Final relative residual


@dataclass
class LargeDeflectionSolveResult:
    """Full trajectory result from multi-step non-linear large deflection solve."""
    steps: List[LargeDeflectionStepResult]
    load_factors: np.ndarray
    monitor_displacements: np.ndarray
    total_solve_time: float

    @property
    def final_step(self) -> LargeDeflectionStepResult:
        return self.steps[-1]


def solve_large_deflection_increments(
    grid: VoxelGrid,
    material: MetalPlasticMaterial,
    load_vector: np.ndarray,
    fixed_dofs: Sequence[int],
    load_factors: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
    max_iter_per_step: int = 30,
    tolerance: float = 1e-6,
    monitor_dof: Optional[int] = None,
) -> LargeDeflectionSolveResult:
    """
    Solve geometrically non-linear (large deflection) structural equilibrium with material plasticity.

    Parameters:
        grid: Cartesian VoxelGrid.
        material: MetalPlasticMaterial (e.g. Al 6061-T6).
        load_vector: Reference external load vector F_ref (N_dof,).
        fixed_dofs: Boundary condition DOF indices constrained to zero.
        load_factors: Load schedule multipliers lambda_k.
        max_iter_per_step: Maximum Newton-Raphson iterations per step.
        tolerance: Relative convergence threshold ||R|| / max(||F_ref||, 1.0).
        monitor_dof: Specific global DOF to track in load-displacement curve.

    Returns:
        LargeDeflectionSolveResult containing step history and Cauchy/PK2 stresses.
    """
    start_time = time.time()
    n_nodes = grid.total_nodes
    n_dof = n_nodes * 3
    n_elem = grid.total_cells
    active_elems = grid.active_element_indices
    n_active = len(active_elems)
    hx, hy, hz = grid.pitch

    # Precompute 2x2x2 Gauss quadrature data
    # B0_gps: 8 linear B matrices (6, 24); dN_dx_gps: 8 shape function gradient matrices (3, 8)
    B0_gps, dN_dx_gps, detJ = _precompute_hex8_grad_N(hx, hy, hz)

    # Elastic constitutive tensor C_el
    k_bulk = material.E / (3.0 * (1.0 - 2.0 * material.nu))
    two_g = material.E / (1.0 + material.nu)
    C_el = k_bulk * _M_OUTER + two_g * _P_DEV

    fixed_set = set(fixed_dofs)
    free_dofs = np.array([d for d in range(n_dof) if d not in fixed_set], dtype=np.int64)
    f_ref_base = max(float(np.linalg.norm(load_vector)), 1.0)

    if monitor_dof is None:
        monitor_dof = int(np.argmax(np.abs(load_vector)))

    # Element DOFs mapping
    elem_dofs = np.zeros((n_elem, 24), dtype=np.int64)
    for e in active_elems:
        nodes = grid.elements[e]
        dofs = np.empty(24, dtype=np.int64)
        for i, node in enumerate(nodes):
            dofs[3 * i : 3 * i + 3] = [3 * node, 3 * node + 1, 3 * node + 2]
        elem_dofs[e] = dofs
    active_elem_dofs = elem_dofs[active_elems]  # (n_active, 24)

    # Triplet index patterns for sparse matrix assembly
    r_idx = np.empty((n_active, 24, 24), dtype=np.int64)
    c_idx = np.empty((n_active, 24, 24), dtype=np.int64)
    for l_idx in range(n_active):
        d_e = active_elem_dofs[l_idx]
        rg, cg = np.meshgrid(d_e, d_e, indexing="ij")
        r_idx[l_idx] = rg
        c_idx[l_idx] = cg
    rows_flat = r_idx.ravel()
    cols_flat = c_idx.ravel()

    # Pre-build G_matrices for geometric stiffness: G_k has shape (3, 24) for k in {0, 1, 2}
    # G_k[i, 3*a + k] = dN_dx[i, a]
    G_gps: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for p in range(8):
        dNx = dN_dx_gps[p]  # (3, 8)
        G0 = np.zeros((3, 24), dtype=np.float64)
        G1 = np.zeros((3, 24), dtype=np.float64)
        G2 = np.zeros((3, 24), dtype=np.float64)
        for a in range(8):
            G0[:, 3 * a + 0] = dNx[:, a]
            G1[:, 3 * a + 1] = dNx[:, a]
            G2[:, 3 * a + 2] = dNx[:, a]
        G_gps.append((G0, G1, G2))

    # Initialize state variables
    n_gp_total = n_active * 8
    state_gp = PlasticHistoryState.initialize(n_gp_total)
    state_committed_gp = PlasticHistoryState.initialize(n_gp_total)
    u_current = np.zeros(n_dof, dtype=np.float64)

    steps_history: List[LargeDeflectionStepResult] = []
    monitor_history: List[float] = []

    for step_idx, lam in enumerate(load_factors):
        F_ext = lam * load_vector
        is_unloading = (step_idx > 0 and lam < load_factors[step_idx - 1])

        converged = False
        final_res_norm = 1.0

        for it in range(max_iter_per_step):
            u_elem = u_current[active_elem_dofs]  # (n_active, 24)

            # 1. Compute displacement gradient H = grad_0(u), deformation gradient F = I + H,
            #    and Green-Lagrange strain E at all 8 Gauss points
            E_gp = np.zeros((n_active, 8, 6), dtype=np.float64)
            BL_gp = np.zeros((n_active, 8, 6, 24), dtype=np.float64)
            F_gp = np.zeros((n_active, 8, 3, 3), dtype=np.float64)

            for p in range(8):
                dNx = dN_dx_gps[p]  # (3, 8)
                B0 = B0_gps[p]      # (6, 24)

                # Displacements at 8 element nodes: shape (n_active, 8, 3)
                u_nodes = u_elem.reshape(n_active, 8, 3)
                # Displacement gradient H_ij = d(u_i) / d(X_j) = sum_a u_i,a * dN_a/dX_j: (n_active, 3, 3)
                # i = displacement component (0:u, 1:v, 2:w), j = coord (0:x, 1:y, 2:z)
                H = np.einsum("eai,ja->eij", u_nodes, dNx)  # (n_active, 3, 3)
                
                # Deformation gradient F = I + H
                I3 = np.eye(3, dtype=np.float64)[None, :, :]
                F_mat = I3 + H  # (n_active, 3, 3)
                F_gp[:, p, :, :] = F_mat

                # Green-Lagrange strain: E = 1/2 * (H + H^T + H^T * H)
                # E_xx, E_yy, E_zz
                E_xx = H[:, 0, 0] + 0.5 * np.sum(H[:, :, 0] ** 2, axis=1)
                E_yy = H[:, 1, 1] + 0.5 * np.sum(H[:, :, 1] ** 2, axis=1)
                E_zz = H[:, 2, 2] + 0.5 * np.sum(H[:, :, 2] ** 2, axis=1)

                # Engineering shear: gamma_yz = 2 * E_yz, gamma_zx = 2 * E_zx, gamma_xy = 2 * E_xy
                # 2 * E_yz = (H_12 + H_21) + (H_01*H_02 + H_11*H_12 + H_21*H_22)
                gamma_yz = (H[:, 1, 2] + H[:, 2, 1]) + np.sum(H[:, :, 1] * H[:, :, 2], axis=1)
                gamma_zx = (H[:, 0, 2] + H[:, 2, 0]) + np.sum(H[:, :, 0] * H[:, :, 2], axis=1)
                gamma_xy = (H[:, 0, 1] + H[:, 1, 0]) + np.sum(H[:, :, 0] * H[:, :, 1], axis=1)

                E_gp[:, p, 0] = E_xx
                E_gp[:, p, 1] = E_yy
                E_gp[:, p, 2] = E_zz
                E_gp[:, p, 3] = gamma_yz
                E_gp[:, p, 4] = gamma_zx
                E_gp[:, p, 5] = gamma_xy

                # Non-linear B-matrix: B_L(u) = B_0 + B_NL(u)
                # For each element e, B_NL is derived from variations delta(E)
                # delta(E_xx) = du_x/dX * d(delta u)/dX + dv_x/dX * d(delta v)/dX + ...
                # We vectorize B_L across all active elements:
                # B0: (6, 24). Let's construct B_L: (n_active, 6, 24)
                B_L = np.tile(B0, (n_active, 1, 1))  # (n_active, 6, 24)
                for a in range(8):
                    dNa_dX = dNx[0, a]
                    dNa_dY = dNx[1, a]
                    dNa_dZ = dNx[2, a]

                    # delta(E_xx) terms: sum_k H_k0 * dNa_dX * delta(u_ka)
                    B_L[:, 0, 3 * a + 0] += H[:, 0, 0] * dNa_dX
                    B_L[:, 0, 3 * a + 1] += H[:, 1, 0] * dNa_dX
                    B_L[:, 0, 3 * a + 2] += H[:, 2, 0] * dNa_dX

                    # delta(E_yy) terms: sum_k H_k1 * dNa_dY * delta(u_ka)
                    B_L[:, 1, 3 * a + 0] += H[:, 0, 1] * dNa_dY
                    B_L[:, 1, 3 * a + 1] += H[:, 1, 1] * dNa_dY
                    B_L[:, 1, 3 * a + 2] += H[:, 2, 1] * dNa_dY

                    # delta(E_zz) terms: sum_k H_k2 * dNa_dZ * delta(u_ka)
                    B_L[:, 2, 3 * a + 0] += H[:, 0, 2] * dNa_dZ
                    B_L[:, 2, 3 * a + 1] += H[:, 1, 2] * dNa_dZ
                    B_L[:, 2, 3 * a + 2] += H[:, 2, 2] * dNa_dZ

                    # delta(gamma_yz): sum_k [ H_k1 * dNa_dZ + H_k2 * dNa_dY ]
                    B_L[:, 3, 3 * a + 0] += H[:, 0, 1] * dNa_dZ + H[:, 0, 2] * dNa_dY
                    B_L[:, 3, 3 * a + 1] += H[:, 1, 1] * dNa_dZ + H[:, 1, 2] * dNa_dY
                    B_L[:, 3, 3 * a + 2] += H[:, 2, 1] * dNa_dZ + H[:, 2, 2] * dNa_dY

                    # delta(gamma_zx): sum_k [ H_k0 * dNa_dZ + H_k2 * dNa_dX ]
                    B_L[:, 4, 3 * a + 0] += H[:, 0, 0] * dNa_dZ + H[:, 0, 2] * dNa_dX
                    B_L[:, 4, 3 * a + 1] += H[:, 1, 0] * dNa_dZ + H[:, 1, 2] * dNa_dX
                    B_L[:, 4, 3 * a + 2] += H[:, 2, 0] * dNa_dZ + H[:, 2, 2] * dNa_dX

                    # delta(gamma_xy): sum_k [ H_k0 * dNa_dY + H_k1 * dNa_dX ]
                    B_L[:, 5, 3 * a + 0] += H[:, 0, 0] * dNa_dY + H[:, 0, 1] * dNa_dX
                    B_L[:, 5, 3 * a + 1] += H[:, 1, 0] * dNa_dY + H[:, 1, 1] * dNa_dX
                    B_L[:, 5, 3 * a + 2] += H[:, 2, 0] * dNa_dY + H[:, 2, 1] * dNa_dX

                BL_gp[:, p, :, :] = B_L

            # 2. Constitutive return mapping in Green-Lagrange space -> PK2 stress S
            E_gp_flat = E_gp.reshape(-1, 6)
            S_flat, state_active, C_ep_flat = radial_return_mapping(
                material=material,
                eps_total=E_gp_flat,
                state_old=state_committed_gp,
                compute_tangent=True,
            )
            S_gp = S_flat.reshape(n_active, 8, 6)
            C_ep_gp = C_ep_flat.reshape(n_active, 8, 6, 6)

            if is_unloading and it == 0:
                C_ep_gp[:] = C_el

            # 3. Assemble internal force vector F_int = sum_gp detJ * B_L^T * S
            F_int = np.zeros(n_dof, dtype=np.float64)
            f_int_elem = np.zeros((n_active, 24), dtype=np.float64)
            for p in range(8):
                # BL_gp[:, p]: (n_active, 6, 24), S_gp[:, p]: (n_active, 6)
                f_int_elem += detJ * np.einsum("eja,ej->ea", BL_gp[:, p, :, :], S_gp[:, p, :])
            for l_idx, e in enumerate(active_elems):
                np.add.at(F_int, elem_dofs[e], f_int_elem[l_idx])

            # 4. Out-of-balance residual R = F_int - F_ext
            R_free = (F_int - F_ext)[free_dofs]
            res_norm = float(np.linalg.norm(R_free)) / f_ref_base
            final_res_norm = res_norm

            if res_norm < tolerance or (step_idx == 0 and res_norm < 1e-10):
                converged = True
                state_committed_gp.eps_p[:] = state_active.eps_p
                state_committed_gp.alpha_p[:] = state_active.alpha_p
                state_committed_gp.yielded[:] = state_active.yielded
                state_gp = state_active
                break

            # 5. Assemble Tangent Stiffness: K_t = K_L(material) + K_sigma(geometric initial stress)
            Ke_all = np.zeros((n_active, 24, 24), dtype=np.float64)
            for p in range(8):
                # Material tangent stiffness: B_L^T * C^ep * B_L
                BL_p = BL_gp[:, p, :, :]  # (n_active, 6, 24)
                C_p = C_ep_gp[:, p, :, :] # (n_active, 6, 6)
                BC = np.einsum("eja,ejk->eak", BL_p, C_p)  # (n_active, 24, 6)
                Ke_all += detJ * np.einsum("eak,ekb->eab", BC, BL_p)

                # Geometric initial stress stiffness: sum_k G_k^T * S_mat * G_k
                S_p = S_gp[:, p, :]  # (n_active, 6): [xx, yy, zz, yz, zx, xy]
                # Construct 3x3 symmetric PK2 tensor for each element: (n_active, 3, 3)
                S_tensor = np.zeros((n_active, 3, 3), dtype=np.float64)
                S_tensor[:, 0, 0] = S_p[:, 0]  # S_xx
                S_tensor[:, 1, 1] = S_p[:, 1]  # S_yy
                S_tensor[:, 2, 2] = S_p[:, 2]  # S_zz
                S_tensor[:, 1, 2] = S_tensor[:, 2, 1] = S_p[:, 3]  # S_yz
                S_tensor[:, 0, 2] = S_tensor[:, 2, 0] = S_p[:, 4]  # S_zx
                S_tensor[:, 0, 1] = S_tensor[:, 1, 0] = S_p[:, 5]  # S_xy

                G0, G1, G2 = G_gps[p]  # Each (3, 24)
                # G_k^T * S_mat * G_k:
                for Gk in (G0, G1, G2):
                    SG = np.einsum("eij,jb->eib", S_tensor, Gk)  # (n_active, 3, 24)
                    Ke_all += detJ * np.einsum("ia,eib->eab", Gk, SG)

            K_global = csr_matrix((Ke_all.ravel(), (rows_flat, cols_flat)), shape=(n_dof, n_dof))
            K_free = K_global[free_dofs, :][:, free_dofs]

            # 6. Solve for displacement update: K_t * delta_u = - R
            delta_u_free = spsolve(K_free, -R_free)
            u_current[free_dofs] += delta_u_free

        # 7. Convert PK2 stress S to True Cauchy stress: sigma = (1 / det(F)) * F * S * F^T
        cauchy_gp = np.zeros((n_active, 8, 6), dtype=np.float64)
        for p in range(8):
            F_p = F_gp[:, p, :, :]  # (n_active, 3, 3)
            detF = np.linalg.det(F_p) # (n_active,)
            detF_inv = 1.0 / np.maximum(detF, 1e-6)

            S_p = S_gp[:, p, :]
            S_tensor = np.zeros((n_active, 3, 3), dtype=np.float64)
            S_tensor[:, 0, 0] = S_p[:, 0]
            S_tensor[:, 1, 1] = S_p[:, 1]
            S_tensor[:, 2, 2] = S_p[:, 2]
            S_tensor[:, 1, 2] = S_tensor[:, 2, 1] = S_p[:, 3]
            S_tensor[:, 0, 2] = S_tensor[:, 2, 0] = S_p[:, 4]
            S_tensor[:, 0, 1] = S_tensor[:, 1, 0] = S_p[:, 5]

            # sigma = (1 / J) * F * S * F^T
            FS = np.einsum("eij,ejk->eik", F_p, S_tensor)
            sigma_tensor = detF_inv[:, None, None] * np.einsum("eik,ejk->eij", FS, F_p)

            cauchy_gp[:, p, 0] = sigma_tensor[:, 0, 0]
            cauchy_gp[:, p, 1] = sigma_tensor[:, 1, 1]
            cauchy_gp[:, p, 2] = sigma_tensor[:, 2, 2]
            cauchy_gp[:, p, 3] = sigma_tensor[:, 1, 2]
            cauchy_gp[:, p, 4] = sigma_tensor[:, 0, 2]
            cauchy_gp[:, p, 5] = sigma_tensor[:, 0, 1]

        # Element-averaged output quantities
        elem_pk2 = np.zeros((n_elem, 6), dtype=np.float64)
        elem_cauchy = np.zeros((n_elem, 6), dtype=np.float64)
        elem_E = np.zeros((n_elem, 6), dtype=np.float64)
        elem_eps_p = np.zeros((n_elem, 6), dtype=np.float64)
        elem_alpha_p = np.zeros(n_elem, dtype=np.float64)
        elem_yielded = np.zeros(n_elem, dtype=bool)

        elem_pk2[active_elems] = np.mean(S_gp, axis=1)
        elem_cauchy[active_elems] = np.mean(cauchy_gp, axis=1)
        elem_E[active_elems] = np.mean(E_gp, axis=1)
        elem_eps_p[active_elems] = np.mean(state_gp.eps_p.reshape(n_active, 8, 6), axis=1)
        elem_alpha_p[active_elems] = np.mean(state_gp.alpha_p.reshape(n_active, 8), axis=1)
        elem_yielded[active_elems] = np.any(state_gp.yielded.reshape(n_active, 8), axis=1)

        step_res = LargeDeflectionStepResult(
            step_index=step_idx,
            load_factor=float(lam),
            displacements=u_current.copy(),
            pk2_stresses=elem_pk2,
            cauchy_stresses=elem_cauchy,
            green_lagrange_strains=elem_E,
            plastic_strains=elem_eps_p,
            accumulated_plastic_strain=elem_alpha_p,
            yielded_mask=elem_yielded,
            iterations=it + 1,
            converged=converged,
            final_residual=final_res_norm,
        )
        steps_history.append(step_res)
        monitor_history.append(float(u_current[monitor_dof]))

    return LargeDeflectionSolveResult(
        steps=steps_history,
        load_factors=np.array(load_factors, dtype=np.float64),
        monitor_displacements=np.array(monitor_history, dtype=np.float64),
        total_solve_time=time.time() - start_time,
    )


def _precompute_hex8_grad_N(hx: float, hy: float, hz: float) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
    """
    Precompute the linear B matrices, physical shape function gradients dN/dX,
    and Jacobian determinant for standard 2x2x2 Gauss quadrature on uniform Cartesian Hex8 cells.
    """
    gp = 1.0 / np.sqrt(3.0)
    gauss_pts = [
        (-gp, -gp, -gp),
        ( gp, -gp, -gp),
        ( gp,  gp, -gp),
        (-gp,  gp, -gp),
        (-gp, -gp,  gp),
        ( gp, -gp,  gp),
        ( gp,  gp,  gp),
        (-gp,  gp,  gp),
    ]

    xi_nodes = np.array([
        [-1.0, -1.0, -1.0],
        [ 1.0, -1.0, -1.0],
        [ 1.0,  1.0, -1.0],
        [-1.0,  1.0, -1.0],
        [-1.0, -1.0,  1.0],
        [ 1.0, -1.0,  1.0],
        [ 1.0,  1.0,  1.0],
        [-1.0,  1.0,  1.0],
    ], dtype=np.float64)

    detJ = (hx / 2.0) * (hy / 2.0) * (hz / 2.0)
    invJ = np.diag([2.0 / hx, 2.0 / hy, 2.0 / hz])

    b_list: List[np.ndarray] = []
    dNx_list: List[np.ndarray] = []

    for xi, eta, zeta in gauss_pts:
        dN_nat = np.zeros((3, 8), dtype=np.float64)
        for a in range(8):
            xa, ya, za = xi_nodes[a]
            dN_nat[0, a] = 0.125 * xa * (1.0 + ya * eta) * (1.0 + za * zeta)
            dN_nat[1, a] = 0.125 * ya * (1.0 + xa * xi) * (1.0 + za * zeta)
            dN_nat[2, a] = 0.125 * za * (1.0 + xa * xi) * (1.0 + ya * eta)

        dN_dx = invJ @ dN_nat  # (3, 8): row 0 = dN/dx, row 1 = dN/dy, row 2 = dN/dz
        dNx_list.append(dN_dx)

        B = np.zeros((6, 24), dtype=np.float64)
        for a in range(8):
            dNx, dNy, dNz = dN_dx[0, a], dN_dx[1, a], dN_dx[2, a]
            col = a * 3
            B[0, col + 0] = dNx
            B[1, col + 1] = dNy
            B[2, col + 2] = dNz
            B[3, col + 1] = dNz
            B[3, col + 2] = dNy
            B[4, col + 0] = dNz
            B[4, col + 2] = dNx
            B[5, col + 0] = dNy
            B[5, col + 1] = dNx
        b_list.append(B)

    return b_list, dNx_list, detJ
