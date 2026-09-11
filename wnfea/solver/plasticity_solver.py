"""
Non-Linear Incremental Elasto-Plastic Solver for WNFEA (Phase 22).
------------------------------------------------------------------
Implements:
1. Incremental Load Stepping: lambda_k * F_ext with full Newton-Raphson iterations.
2. Radial Return Mapping integration of constitutive J2 plasticity at each element/quadrature point.
3. Consistent Tangent Stiffness Assembly K_t(u) = sum_e V_e * B_e^T * C^ep * B_e.
4. Internal Force Evaluation F_int(u) = sum_e V_e * B_e^T * sigma_e.
5. Residual equilibrium tracking: ||R(u)|| / ||F_ext|| < tol with quadratic convergence.
6. Cyclic loading, elastic unloading, springback, and residual stress analysis.
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


@dataclass
class ElastoPlasticStepResult:
    """Results from a single converged load increment."""
    step_index: int
    load_factor: float
    displacements: np.ndarray             # (N_dof,) displacement vector
    stresses: np.ndarray                  # (N_elem, 6) element-averaged Cauchy stress tensor
    plastic_strains: np.ndarray           # (N_elem, 6) element-averaged plastic strain tensor
    accumulated_plastic_strain: np.ndarray # (N_elem,) element-averaged accumulated plastic strain
    yielded_mask: np.ndarray              # (N_elem,) bool mask of yielded elements
    iterations: int                       # Newton iterations taken
    converged: bool                       # Convergence flag
    final_residual: float                 # Final relative residual norm


@dataclass
class ElastoPlasticSolveResult:
    """Comprehensive results across all load increments."""
    steps: List[ElastoPlasticStepResult]
    load_factors: np.ndarray              # (N_steps,) load schedule
    monitor_displacements: np.ndarray     # (N_steps,) displacement at monitored DOF
    plastic_dissipation: np.ndarray       # (N_steps,) cumulative plastic work
    permanent_deflection: float           # Residual displacement remaining at zero load
    total_solve_time: float               # Solution time in seconds

    @property
    def final_step(self) -> ElastoPlasticStepResult:
        return self.steps[-1]


def solve_elastoplastic_increments(
    grid: VoxelGrid,
    material: MetalPlasticMaterial,
    load_vector: np.ndarray,
    fixed_dofs: Sequence[int],
    load_factors: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
    max_iter_per_step: int = 25,
    tolerance: float = 1e-6,
    monitor_dof: Optional[int] = None,
) -> ElastoPlasticSolveResult:
    """
    Solve non-linear elasto-plastic structural equilibrium under incremental loading.

    Parameters:
        grid: VoxelGrid representing the 3D structural domain.
        material: MetalPlasticMaterial instance.
        load_vector: (N_nodes * 3,) reference nodal load vector F_ref.
        fixed_dofs: Boundary condition DOF indices constrained to zero.
        load_factors: Sequence of scaling multipliers lambda_k for incremental loading.
        max_iter_per_step: Maximum Newton-Raphson iterations per step.
        tolerance: Relative convergence tolerance ||R|| / max(||F_ref||, 1.0).
        monitor_dof: Specific global DOF index to track load-displacement curve.

    Returns:
        ElastoPlasticSolveResult containing full history across all load steps.
    """
    start_time = time.time()
    n_nodes = grid.total_nodes
    n_dof = n_nodes * 3
    n_elem = grid.total_cells
    active_elems = grid.active_element_indices
    n_active = len(active_elems)
    hx, hy, hz = grid.pitch

    # Precompute 2x2x2 Gauss points and B-matrices
    B_gps, detJ = _precompute_hex8_2x2x2_B(hx, hy, hz)  # 8 matrices of shape (6, 24)

    # Elastic constitutive tensor C_el
    k_bulk = material.E / (3.0 * (1.0 - 2.0 * material.nu))
    two_g = material.E / (1.0 + material.nu)
    C_el = k_bulk * _M_OUTER + two_g * _P_DEV

    # Free DOFs mask
    fixed_set = set(fixed_dofs)
    free_dofs = np.array([d for d in range(n_dof) if d not in fixed_set], dtype=np.int64)

    # Base reference load norm
    f_ref_base = max(float(np.linalg.norm(load_vector)), 1.0)

    # Initialize state variables at all Gauss points (8 GP per active element)
    n_gp_total = n_active * 8
    state_gp = PlasticHistoryState.initialize(n_gp_total)
    state_committed_gp = PlasticHistoryState.initialize(n_gp_total)
    u_current = np.zeros(n_dof, dtype=np.float64)

    if monitor_dof is None:
        monitor_dof = int(np.argmax(np.abs(load_vector)))

    steps_history: List[ElastoPlasticStepResult] = []
    monitor_history: List[float] = []
    dissipation_history: List[float] = []
    cumulative_dissipation = 0.0

    # Pre-allocate element DOF connectivity map (n_elem, 24)
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

    # Loop over incremental load steps
    for step_idx, lam in enumerate(load_factors):
        F_ext = lam * load_vector
        is_unloading = (step_idx > 0 and lam < load_factors[step_idx - 1])

        converged = False
        final_res_norm = 1.0

        for it in range(max_iter_per_step):
            # 1. Compute element strains at all 8 Gauss points
            u_elem = u_current[active_elem_dofs]  # (n_active, 24)
            eps_gp = np.zeros((n_active, 8, 6), dtype=np.float64)
            for p in range(8):
                eps_gp[:, p, :] = np.einsum("ij,ej->ei", B_gps[p], u_elem)
            eps_gp_flat = eps_gp.reshape(-1, 6)

            # 2. Constitutive return mapping at all Gauss points
            stress_flat, state_active, C_ep_flat = radial_return_mapping(
                material=material,
                eps_total=eps_gp_flat,
                state_old=state_committed_gp,
                compute_tangent=True,
            )
            stress_gp = stress_flat.reshape(n_active, 8, 6)
            C_ep_gp = C_ep_flat.reshape(n_active, 8, 6, 6)

            # In elastic unloading (iteration 0 of unloading step), use elastic tangent predictor
            if is_unloading and it == 0:
                C_ep_gp[:] = C_el

            # 3. Assemble internal force vector F_int
            F_int = np.zeros(n_dof, dtype=np.float64)
            f_int_elem = np.zeros((n_active, 24), dtype=np.float64)
            for p in range(8):
                f_int_elem += detJ * np.einsum("ij,ej->ei", B_gps[p].T, stress_gp[:, p, :])
            for l_idx, e in enumerate(active_elems):
                np.add.at(F_int, elem_dofs[e], f_int_elem[l_idx])

            # 4. Out-of-balance residual vector R = F_int - F_ext
            R_free = (F_int - F_ext)[free_dofs]
            res_norm = float(np.linalg.norm(R_free)) / f_ref_base
            final_res_norm = res_norm

            if res_norm < tolerance or (step_idx == 0 and res_norm < 1e-10):
                converged = True
                # Accept and commit converged state
                state_committed_gp.eps_p[:] = state_active.eps_p
                state_committed_gp.alpha_p[:] = state_active.alpha_p
                state_committed_gp.yielded[:] = state_active.yielded
                state_gp = state_active
                break

            # 5. Assemble global tangent stiffness matrix K_t
            Ke_all = np.zeros((n_active, 24, 24), dtype=np.float64)
            for p in range(8):
                BC = np.einsum("ij,ejk->eik", B_gps[p].T, C_ep_gp[:, p, :, :])
                Ke_all += detJ * np.einsum("eij,jk->eik", BC, B_gps[p])

            K_global = csr_matrix((Ke_all.ravel(), (rows_flat, cols_flat)), shape=(n_dof, n_dof))
            K_free = K_global[free_dofs, :][:, free_dofs]

            # 6. Solve for displacement increment: K_t * delta_u = - R
            delta_u_free = spsolve(K_free, -R_free)

            # 7. Update displacement: u = u + delta_u
            u_current[free_dofs] += delta_u_free

        # Element-averaged quantities for output
        elem_stresses = np.zeros((n_elem, 6), dtype=np.float64)
        elem_plastic_strains = np.zeros((n_elem, 6), dtype=np.float64)
        elem_alpha_p = np.zeros(n_elem, dtype=np.float64)
        elem_yielded = np.zeros(n_elem, dtype=bool)

        stress_gp_reshaped = stress_gp.reshape(n_active, 8, 6)
        eps_p_reshaped = state_gp.eps_p.reshape(n_active, 8, 6)
        alpha_p_reshaped = state_gp.alpha_p.reshape(n_active, 8)
        yielded_reshaped = state_gp.yielded.reshape(n_active, 8)

        elem_stresses[active_elems] = np.mean(stress_gp_reshaped, axis=1)
        elem_plastic_strains[active_elems] = np.mean(eps_p_reshaped, axis=1)
        elem_alpha_p[active_elems] = np.mean(alpha_p_reshaped, axis=1)
        elem_yielded[active_elems] = np.any(yielded_reshaped, axis=1)

        # Evaluate plastic work dissipation increment: dW_p = sum_gp detJ * sigma_gp : d_eps_p
        d_eps_p_gp = state_gp.eps_p - state_committed_gp.eps_p
        step_dissipation = float(detJ * np.sum(stress_gp.reshape(-1, 6) * d_eps_p_gp))
        cumulative_dissipation += max(step_dissipation, 0.0)

        step_res = ElastoPlasticStepResult(
            step_index=step_idx,
            load_factor=float(lam),
            displacements=u_current.copy(),
            stresses=elem_stresses,
            plastic_strains=elem_plastic_strains,
            accumulated_plastic_strain=elem_alpha_p,
            yielded_mask=elem_yielded,
            iterations=it + 1,
            converged=converged,
            final_residual=final_res_norm,
        )
        steps_history.append(step_res)
        monitor_history.append(float(u_current[monitor_dof]))
        dissipation_history.append(cumulative_dissipation)

    perm_deflection = float(monitor_history[-1]) if load_factors[-1] == 0.0 else 0.0

    return ElastoPlasticSolveResult(
        steps=steps_history,
        load_factors=np.array(load_factors, dtype=np.float64),
        monitor_displacements=np.array(monitor_history, dtype=np.float64),
        plastic_dissipation=np.array(dissipation_history, dtype=np.float64),
        permanent_deflection=perm_deflection,
        total_solve_time=time.time() - start_time,
    )


def _precompute_hex8_2x2x2_B(hx: float, hy: float, hz: float) -> Tuple[List[np.ndarray], float]:
    """
    Precompute the 8 strain-displacement matrices B_p (6, 24) and Jacobian determinant
    for standard 2x2x2 Gauss quadrature on uniform Cartesian Hex8 cells.
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
    for xi, eta, zeta in gauss_pts:
        dN_nat = np.zeros((3, 8), dtype=np.float64)
        for a in range(8):
            xa, ya, za = xi_nodes[a]
            dN_nat[0, a] = 0.125 * xa * (1.0 + ya * eta) * (1.0 + za * zeta)
            dN_nat[1, a] = 0.125 * ya * (1.0 + xa * xi) * (1.0 + za * zeta)
            dN_nat[2, a] = 0.125 * za * (1.0 + xa * xi) * (1.0 + ya * eta)

        dN_dx = invJ @ dN_nat  # (3, 8)
        B = np.zeros((6, 24), dtype=np.float64)
        for a in range(8):
            dNx, dNy, dNz = dN_dx[0, a], dN_dx[1, a], dN_dx[2, a]
            col = a * 3
            # Voigt order: [xx, yy, zz, yz, zx, xy]
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

    return b_list, detJ
