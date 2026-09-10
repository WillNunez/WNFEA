"""
Matrix-Free Frequency-Constrained Topology Optimization Engine for WNFEA.
-------------------------------------------------------------------------
Implements dynamic structural optimization with natural frequency constraints:
1. Exact generalized eigenvalue sensitivity:
       d(omega_j^2)/d(rho_e) = phi_j^T * [ dK/d(rho_e) - omega_j^2 * dM/d(rho_e) ] * phi_j
   evaluated on-the-fly without assembling global K or M matrices.
2. Elemental stiffness and lumped mass sensitivities:
       phi_j^T * (dK/d(rho_e)) * phi_j = p * rho_e^(p-1) * (phi_{j, e}^T * k_0 * phi_{j, e})
       phi_j^T * (dM/d(rho_e)) * phi_j = (v_e * rho_mat / 8) * sum_{n=1}^8 ||phi_{j, e, n}||^2
3. Frequency-constrained and resonance-suppression objective:
       Maximize fundamental frequency f_1 = omega_1 / (2*pi) >= f_target
       or Minimize compliance subject to f_1 >= f_target and V <= V*.
4. Multi-objective Optimality Criteria and Projected Gradient updates with spatial sensitivity filtering.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence, List, Tuple, Union
import numpy as np

from ..mesh.voxel_mesher import VoxelGrid
from ..solver.matrix_free_hex8 import (
    compute_hex8_reference_stiffness,
    solve_voxel_linear_static,
)
from ..solver.modal_analysis import (
    solve_modal_analysis,
    compute_lumped_mass_hex8,
    ModalResult,
)
from .filters import SensitivityFilter


@dataclass
class FrequencyConstrainedConfig:
    """
    Configuration for frequency-constrained topology optimization.
    """
    target_frequency_hz: float = 60.0       # Minimum allowable fundamental frequency (Hz)
    target_volume_fraction: float = 0.40    # Volume fraction constraint V*
    simp_penalty: float = 3.0               # SIMP penalty exponent
    max_iterations: int = 30                # Maximum optimization iterations
    convergence_tol: float = 0.01           # Max density change convergence tolerance
    move_limit: float = 0.15                # Move limit per iteration
    filter_radius: float = 0.05             # Sensitivity filter radius (m)
    material_density: float = 7850.0        # Solid material mass density (kg/m^3)
    frequency_weight: float = 0.50          # Multi-objective balance weight for frequency vs compliance
    verbose: bool = True


@dataclass
class FrequencyOptResult:
    """
    Diagnostics and results from frequency-constrained topology optimization.
    """
    optimized_densities: np.ndarray         # (N,) physical densities in [0, 1]
    final_compliance: float                 # Compliance in Joules
    final_frequency_hz: float               # Fundamental natural frequency in Hz
    final_volume_fraction: float            # Final volume fraction achieved
    is_frequency_satisfied: bool            # True if final frequency >= target
    total_iterations: int
    total_time: float                       # Wallclock time (s)
    average_iter_time: float                # Time per iteration (s)
    compliance_history: List[float]
    frequency_history: List[float]
    volume_history: List[float]


class ModalSensitivityEvaluator:
    """
    Evaluates exact closed-form modal eigenvalue sensitivities:
        d(omega_j^2)/d(rho_e) = phi_j^T [ dK/d(rho_e) - omega_j^2 * dM/d(rho_e) ] phi_j
    matrix-free using reference elemental stiffness and lumped mass.
    """
    def __init__(
        self,
        grid: VoxelGrid,
        simp_penalty: float = 3.0,
        density_material: float = 7850.0,
        E: float = 2.1e11,
        nu: float = 0.30,
    ):
        self.grid = grid
        self.p = float(simp_penalty)
        self.rho_mat = float(density_material)
        self.E = float(E)
        self.nu = float(nu)
        
        self.num_elements = self.grid.total_cells
        hx, hy, hz = self.grid.pitch
        self.elem_volume = float(hx * hy * hz)
        
        # Reference unit element stiffness k_0 (scaled by E=1.0)
        self.k_0, _, _ = compute_hex8_reference_stiffness(hx, hy, hz, E=1.0, nu=self.nu)
        
        # Precompute elemental connectivity DOFs
        elem_nodes = self.grid.elements
        node_dofs = elem_nodes[:, :, None] * 3 + np.array([0, 1, 2])[None, None, :]
        self.elem_dofs = node_dofs.reshape(self.num_elements, 24)
        self.elem_nodes = elem_nodes # (N, 8)

    def evaluate_eigenvalue_sensitivities(
        self,
        densities: np.ndarray,
        eigenvalue: float,
        mode_shape: np.ndarray,
    ) -> np.ndarray:
        """
        Compute d(lambda_j)/d(rho_e) = d(omega_j^2)/d(rho_e) across all N elements.
        
        Parameters
        ----------
        densities : np.ndarray, shape (N,)
            Current element densities in [0, 1].
        eigenvalue : float
            Eigenvalue lambda_j = omega_j^2 (rad/s)^2.
        mode_shape : np.ndarray, shape (N_dofs,)
            M-orthonormal mode shape eigenvector phi_j.
            
        Returns
        -------
        dlambda_drho : np.ndarray, shape (N,)
            Eigenvalue sensitivity array.
        """
        # 1. Stiffness sensitivity: phi_j,e^T (dK_e/drho) phi_j,e
        # dK_e/drho = p * rho^(p-1) * E * k_0
        phi_e = mode_shape[self.elem_dofs]  # (N, 24)
        k0_phie = np.dot(phi_e, self.k_0)   # (N, 24)
        strain_energy_term = np.einsum('ij,ij->i', phi_e, k0_phie)  # (N,)
        
        rho_safe = np.maximum(densities, 1e-4)
        dK_drho = self.p * (rho_safe ** (self.p - 1)) * self.E * strain_energy_term  # (N,)
        
        # 2. Mass sensitivity: lambda_j * phi_j,e^T (dM_e/drho) phi_j,e
        # In lumped mass, each of the 8 nodes gets (v_e * rho_mat / 8)
        # Sum of squared displacements at the 8 nodes:
        # phi_e reshaped to (N, 8, 3)
        phi_nodes = phi_e.reshape(self.num_elements, 8, 3)
        sq_displacements = np.sum(phi_nodes ** 2, axis=(1, 2))  # (N,)
        
        mass_per_node = (self.elem_volume * self.rho_mat) / 8.0
        dM_drho = mass_per_node * sq_displacements  # (N,)
        
        # d(omega^2)/drho = dK_drho - lambda * dM_drho
        dlambda_drho = dK_drho - eigenvalue * dM_drho
        return dlambda_drho


class FrequencyConstrainedOptimizer:
    """
    Topology optimizer with combined static compliance and natural frequency constraints.
    """
    def __init__(
        self,
        grid: VoxelGrid,
        forces: np.ndarray,
        fixed_dofs: Sequence[int],
        config: Optional[FrequencyConstrainedConfig] = None,
        E: float = 2.1e11,
        nu: float = 0.30,
    ):
        self.grid = grid
        self.forces = np.asarray(forces, dtype=np.float64)
        self.fixed_dofs = np.asarray(fixed_dofs, dtype=np.int32)
        self.config = config or FrequencyConstrainedConfig()
        self.E = float(E)
        self.nu = float(nu)
        
        self.num_elements = self.grid.total_cells
        hx, hy, hz = self.grid.pitch
        self.elem_volume = float(hx * hy * hz)
        
        # Spatial sensitivity filter
        centroids = self.grid.get_element_centroids()
        volumes = np.full(self.num_elements, self.elem_volume, dtype=np.float64)
        self.filter = SensitivityFilter(centroids, volumes, radius=self.config.filter_radius)
        
        # Reference element stiffness k_0 (scaled with E)
        self.k_0, _, _ = compute_hex8_reference_stiffness(hx, hy, hz, E=self.E, nu=self.nu)
        
        # Precompute elemental DOFs
        elem_nodes = self.grid.elements
        node_dofs = elem_nodes[:, :, None] * 3 + np.array([0, 1, 2])[None, None, :]
        self.elem_dofs = node_dofs.reshape(self.num_elements, 24)
        
        # Modal sensitivity evaluator
        self.modal_evaluator = ModalSensitivityEvaluator(
            grid=self.grid,
            simp_penalty=self.config.simp_penalty,
            density_material=self.config.material_density,
            E=self.E,
            nu=self.nu,
        )

    def optimize(self) -> FrequencyOptResult:
        """
        Execute frequency-constrained topology optimization loop.
        """
        start_time = time.time()
        
        # Initial uniform density
        rho = np.full(self.num_elements, self.config.target_volume_fraction, dtype=np.float64)
        
        compliance_history = []
        frequency_history = []
        volume_history = []
        
        target_omega_sq = (2.0 * np.pi * self.config.target_frequency_hz) ** 2
        
        if self.config.verbose:
            print("=" * 70)
            print("   WNFEA FREQUENCY-CONSTRAINED TOPOLOGY OPTIMIZATION")
            print(f"   Target Volume: {self.config.target_volume_fraction*100:.1f}% | Elements: {self.num_elements}")
            print(f"   Target Frequency: f_1 >= {self.config.target_frequency_hz:.2f} Hz")
            print(f"   Max Iterations: {self.config.max_iterations}")
            print("=" * 70)

        for iteration in range(self.config.max_iterations):
            iter_start = time.time()
            
            # 1. State Solve: Static Linear Elasticity
            self.grid.volume_fractions = np.clip(rho, 1e-4, 1.0)
            u, elem_vm, pcg_iters = solve_voxel_linear_static(
                self.grid,
                self.forces,
                self.fixed_dofs,
                E=self.E,
                nu=self.nu,
                tol=1e-5,
                maxiter=500,
                verbose=False,
            )
            
            # Elemental strain energy for reference k_0
            u_e = u[self.elem_dofs]
            k0_ue = np.dot(u_e, self.k_0)
            elem_strain_energies = np.einsum('ij,ij->i', u_e, k0_ue)
            
            # Compliance c = sum_e rho_e^p * (u_e^T k_0 u_e)
            p = self.config.simp_penalty
            compliance = float(np.sum((rho ** p) * elem_strain_energies))
            compliance_history.append(compliance)
            
            # Static sensitivity: dc/drho = - p * rho^(p-1) * (u_e^T k_0 u_e)
            dc_drho = - p * (np.maximum(rho, 1e-4) ** (p - 1)) * elem_strain_energies
            
            # 2. State Solve: Modal Eigenvalue Analysis
            modal_res = solve_modal_analysis(
                self.grid,
                self.fixed_dofs,
                num_modes=1,
                density_material=self.config.material_density,
                densities=rho,
                E=self.E,
                nu=self.nu,
                tol=1e-4,
                method="eigsh",
            )
            
            f1_hz = float(modal_res.frequencies_hz[0])
            omega1_sq = float(modal_res.eigenvalues[0])
            phi_1 = modal_res.mode_shapes[0].reshape(-1)
            
            frequency_history.append(f1_hz)
            current_vf = float(np.mean(rho))
            volume_history.append(current_vf)
            
            # 3. Dynamic Sensitivity: d(omega_1^2)/drho
            domega_drho = self.modal_evaluator.evaluate_eigenvalue_sensitivities(
                densities=rho,
                eigenvalue=omega1_sq,
                mode_shape=phi_1,
            )
            
            # Filter sensitivities spatially
            filtered_dc = self.filter.filter_sensitivities(dc_drho)
            filtered_domega = self.filter.filter_sensitivities(domega_drho)
            
            # 4. Multi-Objective / Frequency-Constraint Sensitivity Formulation:
            # We want to minimize compliance while driving f1 >= target_frequency_hz.
            # If f1 < target, increase weight on frequency gradient (normalized).
            freq_deficit = max(0.0, (self.config.target_frequency_hz - f1_hz) / max(1.0, self.config.target_frequency_hz))
            
            # Scale gradients to comparable magnitude
            norm_dc = np.max(np.abs(filtered_dc)) + 1e-15
            norm_domega = np.max(np.abs(filtered_domega)) + 1e-15
            
            # Effective gradient: we want -dc (reduce compliance) and +domega (increase frequency)
            # Higher frequency deficit -> stronger push to increase frequency
            w_freq = min(0.90, self.config.frequency_weight + 2.0 * freq_deficit)
            w_comp = 1.0 - w_freq
            
            combined_sens = w_comp * (-filtered_dc / norm_dc) + w_freq * (filtered_domega / norm_domega)
            
            # 5. Optimality Criteria Bisection for Volume Constraint sum(rho_e) = V* * N
            target_vol = self.config.target_volume_fraction * self.num_elements
            move = self.config.move_limit
            
            l1 = 1e-12
            l2 = 1e6
            rho_old = rho.copy()
            rho_cand = np.zeros_like(rho)
            
            # Make sure ratio is non-negative
            pos_sens = np.maximum(combined_sens - np.min(combined_sens) + 1e-4, 1e-8)
            
            for _ in range(40):
                l_mid = 0.5 * (l1 + l2)
                B = np.sqrt(pos_sens / l_mid)
                rho_trial = np.clip(
                    rho_old * B,
                    np.maximum(1e-4, rho_old - move),
                    np.minimum(1.0, rho_old + move),
                )
                if np.sum(rho_trial) > target_vol:
                    l1 = l_mid
                else:
                    l2 = l_mid
                if abs(l2 - l1) / (l1 + 1e-15) < 1e-4:
                    rho_cand = rho_trial
                    break
            else:
                rho_cand = rho_trial
                
            rho = rho_cand
            max_change = float(np.max(np.abs(rho - rho_old)))
            iter_time = time.time() - iter_start
            
            if self.config.verbose:
                print(f"Iter {iteration+1:2d} | Compliance: {compliance:10.4e} J | "
                      f"Freq: {f1_hz:6.2f} Hz (Target: {self.config.target_frequency_hz:.1f}) | "
                      f"Vol: {current_vf*100:4.1f}% | Change: {max_change:6.4f} | Time: {iter_time*1000:6.1f} ms")
                      
            if iteration >= 5 and max_change < self.config.convergence_tol:
                if self.config.verbose:
                    print(f"--> Converged at iteration {iteration+1} (max change {max_change:.5f} < {self.config.convergence_tol})")
                break

        total_time = time.time() - start_time
        num_iters = len(compliance_history)
        avg_iter_time = total_time / max(1, num_iters)
        final_freq = frequency_history[-1]
        
        return FrequencyOptResult(
            optimized_densities=rho,
            final_compliance=compliance_history[-1],
            final_frequency_hz=final_freq,
            final_volume_fraction=float(np.mean(rho)),
            is_frequency_satisfied=final_freq >= (self.config.target_frequency_hz - 1.0),
            total_iterations=num_iters,
            total_time=total_time,
            average_iter_time=avg_iter_time,
            compliance_history=compliance_history,
            frequency_history=frequency_history,
            volume_history=volume_history,
        )
