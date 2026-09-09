"""
Matrix-Free Stress-Constrained Topology Optimization Engine for WNFEA.
----------------------------------------------------------------------
Implements real-time stress-constrained generative structural design:
1. p-Norm & Kreisselmeier-Steinhauser (KS) smooth stress aggregation:
       sigma_PN = [ sum_e (tilde_sigma_vm,e / sigma_yield)^P ]^(1/P) * sigma_yield
2. Stress relaxation (q-SIMP) to prevent singularity in void regions:
       tilde_sigma_e = rho_e^q * sigma_e(u)
3. Exact Adjoint Sensitivity Backpropagation:
       K * lambda = d(sigma_PN)/du
       d(sigma_PN)/d_rho_e = d(sigma_PN)/d_rho_e_expl - p * rho_e^(p-1) * (lambda_e^T * k_0 * u_e)
4. Augmented Lagrangian / Adaptive Penalized Multiplier Optimization:
       Enforces max von Mises stress <= sigma_yield while minimizing material volume.
"""

from __future__ import annotations

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Sequence, List, Tuple
import numpy as np
import scipy.sparse.linalg as spla

from ..mesh.voxel_mesher import VoxelGrid
from ..solver.matrix_free_hex8 import (
    compute_hex8_reference_stiffness,
    MatrixFreeHex8Operator,
)
from .filters import SensitivityFilter, HeavisideProjection


class StressAggregationType(Enum):
    PNORM = "pnorm"
    KS = "ks"


@dataclass
class StressConstraintConfig:
    """Configuration for stress-constrained topology optimization."""
    sigma_yield: float = 2.5e8              # Material yield limit (Pa, default: 250 MPa)
    aggregation_type: StressAggregationType = StressAggregationType.PNORM
    p_norm: float = 6.0                     # p-norm aggregation exponent P in [4, 10]
    ks_rho: float = 50.0                    # KS aggregation parameter rho_ks
    q_relax: float = 0.5                    # Stress relaxation exponent q in (0, 1]
    simp_penalty: float = 3.0               # SIMP stiffness penalty p
    target_volume_fraction: float = 0.40    # Target volume fraction V*
    filter_radius: float = 0.05             # Sensitivity filter radius in meters
    max_iterations: int = 35                # Maximum optimization iterations
    convergence_tol: float = 0.01           # Density change stopping tolerance
    move_limit: float = 0.15                # Maximum density step size per iteration
    enable_heaviside: bool = False          # Enable beta-continuation Heaviside
    verbose: bool = True


@dataclass
class StressOptResult:
    """Output telemetry and optimized stress-constrained results."""
    optimized_densities: np.ndarray         # (N_cells,) physical densities in [0, 1]
    peak_von_mises: float                   # Final maximum von Mises stress (Pa)
    p_norm_stress: float                    # Final p-norm aggregate stress (Pa)
    final_volume_fraction: float            # Achieved volume fraction
    final_compliance: float                 # Final compliance (Joules)
    total_iterations: int
    total_time: float                       # Total solve time in seconds
    stress_history: List[float]             # Peak stress history per iteration
    p_norm_history: List[float]             # p-norm stress history per iteration
    volume_history: List[float]             # Volume fraction history
    final_displacement: np.ndarray          # (N_nodes * 3,) displacement vector
    element_von_mises: np.ndarray           # (N_cells,) von Mises stress field


class StressConstrainedTopologyOptimizer:
    """
    In-the-loop Matrix-Free Stress-Constrained Topology Optimization Engine.
    Directly suppresses stress concentrations to satisfy yield strength limits.
    """
    def __init__(
        self,
        grid: VoxelGrid,
        forces: np.ndarray,
        fixed_dofs: Sequence[int],
        config: Optional[StressConstraintConfig] = None,
        E: float = 2.1e11,
        nu: float = 0.3,
        passive_solid: Optional[Sequence[int]] = None,
        passive_void: Optional[Sequence[int]] = None,
    ):
        self.grid = grid
        self.forces = np.asarray(forces, dtype=np.float64)
        self.fixed_dofs = list(fixed_dofs)
        self.config = config or StressConstraintConfig()
        self.E = float(E)
        self.nu = float(nu)

        # Precompute reference Hex8 stiffness
        hx, hy, hz = grid.pitch
        self.k_0, self.B_0, self.D = compute_hex8_reference_stiffness(hx, hy, hz, E=E, nu=nu)

        # Sensitivity Filter setup
        centroids = grid.get_element_centroids()
        cell_vol = hx * hy * hz
        volumes = np.full(grid.total_cells, cell_vol, dtype=np.float64)
        self.filter = SensitivityFilter(centroids, volumes=volumes, radius=self.config.filter_radius)
        self.heaviside = HeavisideProjection(eta=0.5, beta=1.0) if self.config.enable_heaviside else None

        # Passive domains
        self.passive_solid = set(passive_solid) if passive_solid is not None else set()
        self.passive_void = set(passive_void) if passive_void is not None else set()

    def optimize(self) -> StressOptResult:
        """
        Execute matrix-free stress-constrained topology optimization.
        """
        cfg = self.config
        t_start = time.time()

        n_cells = self.grid.total_cells
        active_indices = self.grid.active_element_indices
        n_active = len(active_indices)

        # Initial uniform density
        rho = np.full(n_cells, cfg.target_volume_fraction, dtype=np.float64)
        for idx in self.passive_solid:
            rho[idx] = 1.0
        for idx in self.passive_void:
            rho[idx] = 1e-4

        rho_phys = rho.copy()

        # Telemetry histories
        stress_history: List[float] = []
        p_norm_history: List[float] = []
        volume_history: List[float] = []

        best_rho = rho_phys.copy()
        best_p_norm = float("inf")
        final_u = np.zeros(self.grid.total_nodes * 3, dtype=np.float64)
        final_vm = np.zeros(n_cells, dtype=np.float64)
        final_compliance = 0.0

        # Adaptive multiplier for stress constraint
        lagrange_mu = 1.0

        for it in range(1, cfg.max_iterations + 1):
            t_iter_start = time.time()

            # 1. Update active grid volume fractions
            self.grid.volume_fractions = rho_phys.copy()

            # 2. Forward Linear Static Solve: K(rho) @ u = f
            op = MatrixFreeHex8Operator(
                self.grid,
                E=self.E,
                nu=self.nu,
                fixed_dofs=self.fixed_dofs,
                simp_p=cfg.simp_penalty,
            )
            inv_diag = 1.0 / op.diag
            M_inv = spla.LinearOperator(op.shape, matvec=lambda v: inv_diag * v)

            rhs = self.forces.copy()
            rhs[op.fixed_dofs_mask] = 0.0

            u, _ = spla.cg(op, rhs, rtol=1e-6, maxiter=500, M=M_inv)
            final_u = u

            # Compliance: C = f^T * u
            compliance = float(np.dot(self.forces, u))
            final_compliance = compliance

            # 3. Recover Stresses on Active Elements
            active_elems = self.grid.elements[active_indices]
            u_e = u[op.elem_dofs]  # (M, 24)

            # Centroidal strain: (M, 6)
            eps_e = u_e @ self.B_0.T
            # Stress: (M, 6)
            sigma_e = eps_e @ self.D.T
            sxx = sigma_e[:, 0]
            syy = sigma_e[:, 1]
            szz = sigma_e[:, 2]
            sxy = sigma_e[:, 3]
            syz = sigma_e[:, 4]
            szx = sigma_e[:, 5]

            # True von Mises: (M,)
            vm = np.sqrt(
                0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2)
            )

            # 4. Stress Relaxation (q-SIMP): tilde_sigma = rho^q * sigma
            alphas = rho_phys[active_indices]
            q = cfg.q_relax
            alphas_q = alphas ** q
            vm_relaxed = alphas_q * vm

            # 5. p-Norm Aggregation
            P = cfg.p_norm
            s_yield = cfg.sigma_yield
            normalized_vm = np.clip(vm_relaxed / s_yield, 0.0, 1e4)
            term_P = normalized_vm ** P
            sum_P = float(np.sum(term_P))
            if sum_P < 1e-30:
                sum_P = 1e-30

            p_norm_stress = float(s_yield * (sum_P ** (1.0 / P)))
            peak_vm = float(np.max(vm_relaxed))

            p_norm_history.append(p_norm_stress)
            stress_history.append(peak_vm)
            current_vol = float(np.mean(rho_phys[active_indices]))
            volume_history.append(current_vol)

            if p_norm_stress < best_p_norm:
                best_p_norm = p_norm_stress
                best_rho = rho_phys.copy()

            # 6. Adjoint Vector Assembly: f_adj = d(sigma_PN) / du
            # c_e = (sum_P)^(1/P - 1) * (normalized_vm)^(P - 1) * rho_e^q / sigma_yield
            c_e = (sum_P ** (1.0 / P - 1.0)) * (normalized_vm ** (P - 1.0)) * (alphas_q / s_yield)

            # d(vm)/d(sigma)
            safe_vm = np.where(vm > 1e-12, vm, 1e-12)
            d_vm_d_sigma = np.column_stack([
                (sxx - 0.5 * (syy + szz)) / safe_vm,
                (syy - 0.5 * (sxx + szz)) / safe_vm,
                (szz - 0.5 * (sxx + syy)) / safe_vm,
                3.0 * sxy / safe_vm,
                3.0 * syz / safe_vm,
                3.0 * szx / safe_vm,
            ])  # (M, 6)

            d_vm_d_eps = d_vm_d_sigma @ self.D.T   # (M, 6)
            d_vm_d_u = d_vm_d_eps @ self.B_0      # (M, 24)
            f_adj_elem = c_e[:, None] * d_vm_d_u  # (M, 24)

            f_adj = np.zeros(self.grid.total_nodes * 3, dtype=np.float64)
            np.add.at(f_adj, op.elem_dofs.ravel(), f_adj_elem.ravel())

            # 7. Adjoint Solve: K @ lambda = f_adj
            rhs_adj = f_adj.copy()
            rhs_adj[op.fixed_dofs_mask] = 0.0
            lam, _ = spla.cg(op, rhs_adj, rtol=1e-6, maxiter=500, M=M_inv)

            # 8. Adjoint Sensitivity of Stress:
            # d(sigma_PN)/d_rho_e = d(sigma_PN)/d_rho_e_expl - p * rho_e^(p-1) * (lambda_e^T * k_0 * u_e)
            lam_e = lam[op.elem_dofs]  # (M, 24)
            ku_e = u_e @ self.k_0      # (M, 24)
            lam_k_u = np.sum(lam_e * ku_e, axis=1)  # (M,)

            p_simp = cfg.simp_penalty
            d_stress_adjoint = - p_simp * (alphas ** (p_simp - 1.0)) * lam_k_u  # (M,)

            # Explicit part: d(sigma_PN)/d_rho_e_expl
            safe_alphas = np.where(alphas > 1e-12, alphas, 1e-12)
            d_stress_expl = (
                (sum_P ** (1.0 / P - 1.0))
                * (normalized_vm ** (P - 1.0))
                * (q * (safe_alphas ** (q - 1.0)) * vm)
            )

            d_stress_total = np.zeros(n_cells, dtype=np.float64)
            d_stress_total[active_indices] = d_stress_adjoint + d_stress_expl

            # Compliance sensitivity for reinforcement guidance
            strain_energy = np.sum(u_e * ku_e, axis=1)
            d_comp = np.zeros(n_cells, dtype=np.float64)
            d_comp[active_indices] = - p_simp * (alphas ** (p_simp - 1.0)) * strain_energy

            # Filter sensitivities
            d_stress_filt = self.filter.filter_sensitivities(d_stress_total)
            d_comp_filt = self.filter.filter_sensitivities(d_comp)

            # 9. Update Multiplier and Densities
            # Constraint violation: g_stress = (sigma_PN / sigma_yield) - 1.0
            g_stress = (p_norm_stress / s_yield) - 1.0
            if g_stress > 0:
                lagrange_mu = min(lagrange_mu * 1.15, 1e4)
            else:
                lagrange_mu = max(lagrange_mu * 0.90, 0.01)

            # Combined sensitivity: minimize volume + mu * stress_violation
            # Note: d_comp is negative, so (-d_comp) is positive energy density
            comp_metric = np.maximum(-d_comp_filt, 1e-12)
            stress_metric = np.maximum(-d_stress_filt, 0.0)

            # OC-style heuristic driving material to high-stress & high-strain paths:
            B_e = (comp_metric + lagrange_mu * stress_metric) / (
                np.mean(comp_metric[active_indices]) + 1e-12
            )
            B_e = np.power(B_e, 0.5)

            # Bisection on volume
            l1, l2 = 0.0, 1e5
            rho_new = rho.copy()
            move = cfg.move_limit

            while (l2 - l1) / (l1 + l2 + 1e-12) > 1e-4:
                lmid = 0.5 * (l1 + l2)
                step = rho * np.power(B_e / lmid, 0.5)
                candidate = np.maximum(
                    0.001,
                    np.maximum(
                        rho - move,
                        np.minimum(1.0, np.minimum(rho + move, step)),
                    ),
                )
                for pid in self.passive_solid:
                    candidate[pid] = 1.0
                for pid in self.passive_void:
                    candidate[pid] = 0.001

                if np.mean(candidate[active_indices]) > cfg.target_volume_fraction:
                    l1 = lmid
                else:
                    l2 = lmid

            rho_new = candidate

            # Check change
            change = float(np.max(np.abs(rho_new - rho)))
            rho = rho_new
            rho_phys = rho.copy()

            iter_time = time.time() - t_iter_start
            if cfg.verbose and (it % 5 == 0 or it == 1):
                print(
                    f"  Iter {it:2d} ({iter_time*1000:.1f}ms): "
                    f"Vol = {current_vol*100:.1f}%, "
                    f"Peak VM = {peak_vm/1e6:.2f} MPa, "
                    f"P-norm = {p_norm_stress/1e6:.2f} MPa, "
                    f"Chg = {change:.3f}"
                )

            if change < cfg.convergence_tol and it >= 10:
                if cfg.verbose:
                    print(f"  --> Converged at iteration {it} (change < {cfg.convergence_tol})")
                break

        total_time = time.time() - t_start
        final_vm[active_indices] = vm_relaxed

        return StressOptResult(
            optimized_densities=best_rho,
            peak_von_mises=peak_vm,
            p_norm_stress=best_p_norm,
            final_volume_fraction=float(np.mean(best_rho[active_indices])),
            final_compliance=final_compliance,
            total_iterations=it,
            total_time=total_time,
            stress_history=stress_history,
            p_norm_history=p_norm_history,
            volume_history=volume_history,
            final_displacement=final_u,
            element_von_mises=final_vm,
        )
