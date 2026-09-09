"""
High-Performance Matrix-Free SIMP Topology Optimization Engine for WNFEA.
--------------------------------------------------------------------------
Implements the real-time in-the-loop generative design pipeline:
1. Matrix-Free Solid Isotropic Material with Penalization (SIMP):
       E_e(rho) = E_min + rho_e^p * (E_0 - E_min)
2. Direct elemental strain energy evaluation:
       dC/d_rho_e = - p * rho_e^(p-1) * (u_e^T * k_0 * u_e)
   computed on-the-fly without global stiffness assembly.
3. Optimality Criteria (OC) bisection update scheme with move limits.
4. Integrated spatial filtering, Heaviside projection, and 3-axis CNC machinability constraints.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence, List, Union
import numpy as np

from ..mesh.voxel_mesher import VoxelGrid
from ..solver.matrix_free_hex8 import (
    compute_hex8_reference_stiffness,
    solve_voxel_linear_static,
)
from .filters import SensitivityFilter, HeavisideProjection
from .machinability import CNCMillingConstraint


@dataclass
class TopologyConfig:
    """
    Configuration parameters for topology optimization run.
    """
    target_volume_fraction: float = 0.40   # Desired mass fraction V* in (0, 1)
    simp_penalty: float = 3.0              # SIMP penalization power p >= 3
    filter_radius: float = 0.05            # Spatial sensitivity filter radius (m)
    max_iterations: int = 40               # Maximum optimization iterations
    convergence_tol: float = 0.01          # Stop when max(|rho_new - rho|) < tol
    move_limit: float = 0.20               # Maximum step size per iteration
    damping_factor: float = 0.50           # OC damping coefficient eta
    enable_heaviside: bool = True          # Enable beta-continuation Heaviside projection
    heaviside_start_iter: int = 10         # Iteration to activate Heaviside projection
    cnc_milling_axis: Optional[Union[str, Sequence[str]]] = None # e.g. "+z", "bi-z", or [\"+z\", \"-z\", \"+x\"]
    cnc_penalty_weight: float = 0.0        # Weight for undercut penalty
    verbose: bool = True


@dataclass
class OptimizationResult:
    """
    Output telemetry and optimized density field.
    """
    optimized_densities: np.ndarray        # (N,) physical densities in [0, 1]
    final_compliance: float                # Final compliance in Joules
    final_volume_fraction: float           # Achieved volume fraction
    total_iterations: int
    total_time: float                      # Total wallclock time (s)
    average_iter_time: float               # Average time per iteration (s)
    compliance_history: List[float]
    volume_history: List[float]
    change_history: List[float]
    discreteness_index: float              # Fraction of elements within 5% of 0 or 1
    load_case_compliances: Optional[List[float]] = None
    final_displacement: Optional[np.ndarray] = None


class TopologyOptimizer:
    """
    In-the-loop Matrix-Free Multi-Load Case Topology Optimization Engine.
    """
    def __init__(
        self,
        grid: VoxelGrid,
        forces: Union[np.ndarray, Sequence[np.ndarray]],
        fixed_dofs: Sequence[int],
        config: Optional[TopologyConfig] = None,
        load_weights: Optional[Sequence[float]] = None,
        E: float = 2.1e11,
        nu: float = 0.3,
        passive_solid: Optional[Sequence[int]] = None,
        passive_void: Optional[Sequence[int]] = None,
    ):
        self.grid = grid
        self.fixed_dofs = list(fixed_dofs)
        self.config = config or TopologyConfig()
        self.E = float(E)
        self.nu = float(nu)

        # Multi-load case processing
        if isinstance(forces, np.ndarray) and forces.ndim == 1:
            self.load_cases = [forces.astype(np.float64)]
        elif isinstance(forces, np.ndarray) and forces.ndim == 2:
            self.load_cases = [forces[i].astype(np.float64) for i in range(len(forces))]
        else:
            self.load_cases = [np.asarray(f, dtype=np.float64) for f in forces]

        self.forces = self.load_cases[0]  # Backward compatibility property

        n_lc = len(self.load_cases)
        if load_weights is not None:
            w = np.asarray(load_weights, dtype=np.float64)
            if len(w) != n_lc:
                raise ValueError(f"Length of load_weights ({len(w)}) must match number of load cases ({n_lc})")
            if np.sum(w) <= 0:
                raise ValueError("Sum of load_weights must be positive")
            self.load_weights = w / np.sum(w)
        else:
            self.load_weights = np.full(n_lc, 1.0 / n_lc, dtype=np.float64)

        self.n_elements = grid.total_cells
        hx, hy, hz = grid.pitch
        self.k_0, self.B_0, self.D = compute_hex8_reference_stiffness(hx, hy, hz, self.E, self.nu)

        # Precompute elemental connectivity DOFs
        elem_nodes = grid.elements  # (N, 8)
        node_dofs = elem_nodes[:, :, None] * 3 + np.array([0, 1, 2])[None, None, :]
        self.elem_dofs = node_dofs.reshape(self.n_elements, 24)

        # Spatial Filter setup
        centroids = grid.get_element_centroids()
        cell_vol = hx * hy * hz
        volumes = np.full(self.n_elements, cell_vol, dtype=np.float64)
        self.filter = SensitivityFilter(centroids, volumes, radius=self.config.filter_radius)

        # Heaviside Projection
        self.heaviside = HeavisideProjection()

        # Non-design domains
        self.passive_solid_mask = np.zeros(self.n_elements, dtype=bool)
        if passive_solid is not None and len(passive_solid) > 0:
            self.passive_solid_mask[passive_solid] = True

        self.passive_void_mask = np.zeros(self.n_elements, dtype=bool)
        if passive_void is not None and len(passive_void) > 0:
            self.passive_void_mask[passive_void] = True

        # CNC milling constraint
        self.cnc_constraint = None
        if self.config.cnc_milling_axis is not None:
            self.cnc_constraint = CNCMillingConstraint(grid, milling_axis=self.config.cnc_milling_axis)

    def optimize(self) -> OptimizationResult:
        """Execute in-the-loop topology optimization loop."""
        cfg = self.config
        t_start = time.perf_counter()

        # 1. Initialize design densities to target volume fraction
        rho = np.full(self.n_elements, cfg.target_volume_fraction, dtype=np.float64)
        rho[self.passive_solid_mask] = 1.0
        rho[self.passive_void_mask] = 0.0

        compliance_hist = []
        volume_hist = []
        change_hist = []

        total_vol = float(self.n_elements)
        target_vol = cfg.target_volume_fraction * total_vol

        for it in range(1, cfg.max_iterations + 1):
            t_it_start = time.perf_counter()

            # A. Density filtering & Projection
            rho_filtered = self.filter.filter_densities(rho)

            if cfg.enable_heaviside and it >= cfg.heaviside_start_iter:
                rho_phys = self.heaviside.project(rho_filtered)
                d_proj = self.heaviside.derivative(rho_filtered)
                if it % 5 == 0:
                    self.heaviside.step_continuation()
            else:
                rho_phys = rho_filtered
                d_proj = np.ones_like(rho)

            # Strict preservation of non-design domains
            rho_phys[self.passive_solid_mask] = 1.0
            rho_phys[self.passive_void_mask] = 0.0

            # Clamp physical densities to prevent singular matrices
            self.grid.volume_fractions = np.clip(rho_phys, 1e-4, 1.0)

            # B. State Solve via Matrix-Free PCG across all load cases
            p = cfg.simp_penalty
            weighted_strain_energy = np.zeros(self.n_elements, dtype=np.float64)
            compliance = 0.0
            lc_compliances = []
            max_pcg_iters = 0
            last_u = None

            for f_case, w_case in zip(self.load_cases, self.load_weights):
                u, elem_vm, pcg_iters = solve_voxel_linear_static(
                    self.grid,
                    f_case,
                    self.fixed_dofs,
                    E=self.E,
                    nu=self.nu,
                    tol=1e-5,
                    verbose=False,
                )
                last_u = u
                max_pcg_iters = max(max_pcg_iters, pcg_iters)

                u_e = u[self.elem_dofs]
                k0_ue = u_e @ self.k_0
                se_case = 0.5 * np.sum(u_e * k0_ue, axis=1)  # (N,)
                comp_case = float(np.sum((rho_phys ** p) * (2.0 * se_case)))
                lc_compliances.append(comp_case)

                weighted_strain_energy += w_case * se_case
                compliance += w_case * comp_case

            curr_vol = float(np.mean(rho_phys))
            compliance_hist.append(compliance)
            volume_hist.append(curr_vol)

            # D. Adjoint Sensitivities (weighted sum across load cases)
            # dC/d(rho_phys) = - p * rho_phys^(p-1) * (2 * weighted_strain_energy)
            dC_drho_phys = - p * (rho_phys ** (p - 1.0)) * (2.0 * weighted_strain_energy)

            # Chain rule through Heaviside and Spatial Filter
            # dC/d(rho_filtered) = dC/d(rho_phys) * d_proj
            dC_drho_filt = dC_drho_phys * d_proj
            # dC/d(rho) = H^T @ dC_drho_filt
            dC_drho = self.filter.filter_sensitivities(dC_drho_filt)

            # E. Optional CNC Machinability Penalty
            if self.cnc_constraint is not None and cfg.cnc_penalty_weight > 0:
                p_val, p_grad = self.cnc_constraint.evaluate_undercut_penalty(rho_phys)
                dC_drho += cfg.cnc_penalty_weight * p_grad

            # Zero sensitivities on non-design regions
            dC_drho[self.passive_solid_mask] = 0.0
            dC_drho[self.passive_void_mask] = 0.0

            # F. Optimality Criteria (OC) Bisection Update
            pos_sens = np.maximum(-dC_drho, 1e-14)
            design_mask = (~self.passive_solid_mask) & (~self.passive_void_mask)
            if np.any(design_mask):
                l1 = float(np.min(pos_sens[design_mask])) * 1e-4
                l2 = float(np.max(pos_sens[design_mask])) * 1e4
            else:
                l1, l2 = 1e-6, 1e2
            move = cfg.move_limit
            eta = cfg.damping_factor

            # Vectorized bisection for Lagrange multiplier lambda
            rho_new = rho.copy()
            for _ in range(60):
                l_mid = 0.5 * (l1 + l2)
                # B_e = - dC/drho / (lambda * V_e)
                # Elements with large -dC/drho want to increase density
                be = np.maximum(-dC_drho / max(l_mid, 1e-14), 1e-12)
                step = rho * (be ** eta)

                # Move limits: max(0, rho - move) <= rho_new <= min(1, rho + move)
                candidate = np.clip(step, rho - move, rho + move)
                candidate = np.clip(candidate, 0.0, 1.0)
                candidate[self.passive_solid_mask] = 1.0
                candidate[self.passive_void_mask] = 0.0

                if np.sum(candidate) > target_vol:
                    l1 = l_mid  # Lambda too small -> volume too large
                else:
                    l2 = l_mid  # Lambda too large -> volume too small

                if (l2 - l1) / (l1 + l2) < 1e-4:
                    rho_new = candidate
                    break

            # G. Check Convergence
            change = float(np.max(np.abs(rho_new - rho)))
            change_hist.append(change)
            rho = rho_new

            t_it = time.perf_counter() - t_it_start
            if cfg.verbose and (it == 1 or it % 5 == 0 or change < cfg.convergence_tol):
                print(
                    f"  Iter {it:2d}: Comp = {compliance:10.2f} J | "
                    f"Vol = {curr_vol*100:4.1f}% | "
                    f"Chg = {change:.4f} | "
                    f"PCG = {max_pcg_iters:2d} | "
                    f"Time = {t_it*1e3:5.1f} ms"
                )

            if change < cfg.convergence_tol and it >= 10:
                if cfg.verbose:
                    print(f"  [CONVERGED] Topology optimization converged at iter {it} (chg < {cfg.convergence_tol})")
                break

        t_total = time.perf_counter() - t_start
        final_phys = self.filter.filter_densities(rho)
        if cfg.enable_heaviside:
            final_phys = self.heaviside.project(final_phys)
        final_phys[self.passive_solid_mask] = 1.0
        final_phys[self.passive_void_mask] = 0.0

        # Discreteness index: proportion of elements where rho < 0.05 or rho > 0.95
        discrete_mask = (final_phys < 0.05) | (final_phys > 0.95)
        discreteness = float(np.mean(discrete_mask))

        return OptimizationResult(
            optimized_densities=final_phys,
            final_compliance=compliance_hist[-1],
            final_volume_fraction=volume_hist[-1],
            total_iterations=len(compliance_hist),
            total_time=t_total,
            average_iter_time=t_total / max(1, len(compliance_hist)),
            compliance_history=compliance_hist,
            volume_history=volume_hist,
            change_history=change_hist,
            discreteness_index=discreteness,
            load_case_compliances=lc_compliances,
            final_displacement=last_u,
        )
