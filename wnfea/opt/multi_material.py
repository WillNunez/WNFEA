"""
Multi-Material SIMP Topology Optimization Engine for WNFEA.
-------------------------------------------------------------
Implements multi-material structural optimization with M candidate solid materials + void:
1. Simplex partition-of-unity density interpolation:
       rho_{e, m} >= 0,  sum_{m=1}^M rho_{e, m} <= 1
2. Effective Young's modulus & mass density:
       E_e(rho_e) = E_min + sum_{m=1}^M (rho_{e, m})^p * E_m
       density_e(rho_e) = sum_{m=1}^M rho_{e, m} * density_m
3. Matrix-free elemental strain energy and exact adjoint sensitivities:
       dc / d(rho_{e, m}) = - p * (rho_{e, m})^(p-1) * E_m * (u_e^T * k_0 * u_e)
       dM / d(rho_{e, m}) = v_e * density_m
4. Vectorized Euclidean projection onto the unit simplex Delta = {x >= 0, sum x <= 1}.
5. Multi-material Optimality Criteria (MMOC) and projected gradient optimizer.
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
from .filters import SensitivityFilter


@dataclass
class MaterialProperty:
    """
    Candidate solid material definition for multi-material topology optimization.
    """
    name: str
    E: float                # Young's Modulus (Pa)
    density: float          # Mass density (kg/m^3)
    nu: float = 0.30        # Poisson's ratio
    cost_per_kg: float = 1.0 # Optional economic cost factor ($/kg)
    color: Tuple[float, float, float] = (0.5, 0.5, 0.5)


# Standard aerospace/structural material library
ALUMINUM_6061 = MaterialProperty(name="Aluminum_6061", E=69.0e9, density=2700.0, nu=0.33, cost_per_kg=5.0, color=(0.8, 0.8, 0.9))
TITANIUM_TI6AL4V = MaterialProperty(name="Titanium_Ti6Al4V", E=114.0e9, density=4430.0, nu=0.34, cost_per_kg=40.0, color=(0.6, 0.6, 0.7))
STEEL_STRUCTURAL = MaterialProperty(name="Structural_Steel", E=210.0e9, density=7850.0, nu=0.30, cost_per_kg=2.0, color=(0.3, 0.3, 0.4))
CARBON_PEEK = MaterialProperty(name="Carbon_PEEK", E=45.0e9, density=1450.0, nu=0.38, cost_per_kg=120.0, color=(0.1, 0.1, 0.1))


def project_simplex_batch(V: np.ndarray) -> np.ndarray:
    """
    Vectorized Euclidean projection of N points in R^M onto the unit simplex:
        Delta = { x in R^M : x_m >= 0, sum_{m=1}^M x_m <= 1 }
    
    Parameters
    ----------
    V : np.ndarray, shape (N, M)
        Unconstrained density candidates for N elements across M materials.
        
    Returns
    -------
    X : np.ndarray, shape (N, M)
        Admissible projected densities satisfying x_m >= 0 and sum(x) <= 1.
    """
    # 1. First clip non-negativity
    V_pos = np.maximum(V, 0.0)
    sums = np.sum(V_pos, axis=1) # (N,)
    
    # Elements already inside the simplex require no further projection
    inside_mask = sums <= 1.0
    if np.all(inside_mask):
        return V_pos

    # 2. For elements violating sum <= 1, project onto the hyperplane sum(x) = 1, x >= 0
    # Algorithm: Sort components descending: u_1 >= u_2 >= ... >= u_M
    # Find rho = max { j in [1, M] : u_j + (1 / j) * (1 - sum_{i=1}^j u_i) > 0 }
    # theta = (1 / rho) * (1 - sum_{i=1}^rho u_i)
    # x_i = max(v_i + theta, 0)
    N, M = V.shape
    X = V_pos.copy()
    
    violators = ~inside_mask
    V_viol = V[violators]
    
    # Sort descending along material axis
    U = np.sort(V_viol, axis=1)[:, ::-1]
    cssv = np.cumsum(U, axis=1)
    
    j_indices = np.arange(1, M + 1)
    cond = U + (1.0 / j_indices) * (1.0 - cssv) > 0.0
    
    # Find the largest index j where cond is True
    # If cond is all False (which should not happen for sum > 1), fallback to 0
    rho = M - 1 - np.argmax(cond[:, ::-1], axis=1) # (N_viol,)
    
    # Extract corresponding cumulative sums
    cssv_rho = cssv[np.arange(len(rho)), rho]
    theta = (1.0 - cssv_rho) / (rho + 1.0)
    
    X_viol = np.maximum(V_viol + theta[:, None], 0.0)
    X[violators] = X_viol
    return X


class MultiMaterialSIMPInterpolator:
    """
    Interpolates effective Young's modulus, mass density, and computes exact adjoint sensitivities
    for multi-material SIMP domains.
    """
    def __init__(
        self,
        materials: Sequence[MaterialProperty],
        simp_penalty: float = 3.0,
        E_min: float = 1e-6,
    ):
        self.materials = list(materials)
        self.M = len(self.materials)
        self.p = float(simp_penalty)
        self.E_min = float(E_min)
        
        # Pre-extract material vectors
        self.E_vec = np.array([mat.E for mat in self.materials], dtype=np.float64) # (M,)
        self.density_vec = np.array([mat.density for mat in self.materials], dtype=np.float64) # (M,)
        self.cost_vec = np.array([mat.cost_per_kg * mat.density for mat in self.materials], dtype=np.float64) # (M,)

    def compute_effective_modulus(self, rho: np.ndarray) -> np.ndarray:
        """
        Compute effective Young's modulus for each element:
            E_e = E_min + sum_{m=1}^M (rho_{e, m})^p * E_m
            
        Parameters
        ----------
        rho : np.ndarray, shape (N, M)
        
        Returns
        -------
        E_eff : np.ndarray, shape (N,)
        """
        penalized_rho = np.maximum(rho, 0.0) ** self.p
        return self.E_min + np.dot(penalized_rho, self.E_vec)

    def compute_effective_density(self, rho: np.ndarray) -> np.ndarray:
        """
        Compute effective physical mass density for each element:
            rho_mass, e = sum_{m=1}^M rho_{e, m} * density_m
        """
        return np.dot(np.maximum(rho, 0.0), self.density_vec)

    def compute_total_mass(self, rho: np.ndarray, elem_volume: float) -> float:
        """Compute total mass of the design in kg."""
        return float(np.sum(self.compute_effective_density(rho)) * elem_volume)

    def compute_compliance_sensitivities(
        self,
        rho: np.ndarray,
        elem_strain_energies: np.ndarray,
    ) -> np.ndarray:
        """
        Compute analytical sensitivities of compliance with respect to each material fraction:
            dc / d(rho_{e, m}) = - p * (rho_{e, m})^(p-1) * E_m * U_0_e
        where U_0_e = u_e^T k_0 u_e is the elemental strain energy for reference unit stiffness.
        
        Parameters
        ----------
        rho : np.ndarray, shape (N, M)
        elem_strain_energies : np.ndarray, shape (N,)
        
        Returns
        -------
        dc_drho : np.ndarray, shape (N, M)
        """
        rho_safe = np.maximum(rho, 1e-12)
        power_term = self.p * (rho_safe ** (self.p - 1)) # (N, M)
        # Scaled by material modulus E_m and elemental strain energy
        dc_drho = - power_term * self.E_vec[None, :] * elem_strain_energies[:, None]
        return dc_drho

    def compute_mass_sensitivities(
        self,
        num_elements: int,
        elem_volume: float,
    ) -> np.ndarray:
        """
        Compute analytical sensitivities of total mass with respect to rho_{e, m}:
            dM / d(rho_{e, m}) = elem_volume * density_m
        """
        dM_drho = np.tile(self.density_vec * elem_volume, (num_elements, 1))
        return dM_drho


@dataclass
class MultiMaterialConfig:
    """Configuration for multi-material topology optimization."""
    target_mass_fraction: float = 0.35    # Target total mass fraction relative to 100% solid of heaviest material
    simp_penalty: float = 3.0             # SIMP exponent p
    max_iterations: int = 40              # Maximum optimization iterations
    convergence_tol: float = 0.01         # Max density change convergence tolerance
    move_limit: float = 0.20              # Move limit per step
    filter_radius: float = 0.05           # Spatial sensitivity filter radius (m)
    enable_am_filter: bool = False        # Enable additive manufacturing overhang filter
    am_build_direction: str = "+z"        # Build direction: '+z', '-z', '+y', '-y', '+x', '-x'
    am_critical_angle_deg: float = 45.0   # Critical overhang angle (deg)
    am_penalty_weight: float = 0.0        # Weight for overhang penalty in sensitivities
    verbose: bool = True


@dataclass
class MultiMaterialResult:
    """Output results and diagnostics from multi-material topology optimization."""
    optimized_densities: np.ndarray       # (N, M) optimal volume fractions
    dominant_material: np.ndarray         # (N,) material index (-1 for void, 0..M-1 for solid materials)
    final_compliance: float               # Compliance in Joules
    final_mass: float                     # Mass in kg
    mass_fraction: float                  # Final mass fraction achieved
    material_volumes: List[float]         # Volume in m^3 of each material
    total_iterations: int
    total_time: float                     # Wallclock time (s)
    average_iter_time: float              # Time per iteration (s)
    compliance_history: List[float]
    mass_history: List[float]
    am_is_printable: Optional[bool] = None # True if AM overhang constraints satisfied


class MultiMaterialTopologyOptimizer:
    """
    In-the-loop Matrix-Free Multi-Material SIMP Topology Optimization Engine.
    """
    def __init__(
        self,
        grid: VoxelGrid,
        forces: np.ndarray,
        fixed_dofs: Sequence[int],
        materials: Sequence[MaterialProperty],
        config: Optional[MultiMaterialConfig] = None,
        nu_reference: float = 0.30,
    ):
        self.grid = grid
        self.forces = np.asarray(forces, dtype=np.float64)
        self.fixed_dofs = np.asarray(fixed_dofs, dtype=np.int32)
        self.materials = list(materials)
        self.M = len(self.materials)
        self.config = config or MultiMaterialConfig()
        
        self.num_elements = self.grid.total_cells
        hx, hy, hz = self.grid.pitch
        self.elem_volume = float(hx * hy * hz)
        self.total_domain_volume = self.num_elements * self.elem_volume
        
        # Interpolator
        self.interpolator = MultiMaterialSIMPInterpolator(
            materials=self.materials,
            simp_penalty=self.config.simp_penalty,
        )
        
        # Spatial sensitivity filter
        centroids = self.grid.get_element_centroids()
        volumes = np.full(self.num_elements, self.elem_volume, dtype=np.float64)
        self.filter = SensitivityFilter(centroids, volumes, radius=self.config.filter_radius)
        
        # Reference element stiffness k_0 (normalized with E=1.0)
        self.k_0, _, _ = compute_hex8_reference_stiffness(
            hx=hx,
            hy=hy,
            hz=hz,
            E=1.0,
            nu=nu_reference,
        )
        
        # Precompute elemental degrees of freedom mapping
        elem_nodes = self.grid.elements
        node_dofs = elem_nodes[:, :, None] * 3 + np.array([0, 1, 2])[None, None, :]
        self.elem_dofs = node_dofs.reshape(self.num_elements, 24)
        
        # AM Overhang filter
        if self.config.enable_am_filter:
            from .am_overhang import AMOverhangFilter
            self.am_filter = AMOverhangFilter(
                grid=self.grid,
                build_direction=self.config.am_build_direction,
                critical_angle_deg=self.config.am_critical_angle_deg,
            )
        else:
            self.am_filter = None
        
        # Baseline reference mass (if domain were 100% of the heaviest material)
        max_mat_density = max(mat.density for mat in self.materials)
        self.max_domain_mass = self.total_domain_volume * max_mat_density
        self.target_mass = self.config.target_mass_fraction * self.max_domain_mass

    def compute_element_strain_energies(self, u: np.ndarray) -> np.ndarray:
        """
        Evaluate elemental strain energies U_0_e = u_e^T k_0 u_e for reference stiffness k_0.
        Vectorized across all N elements without global matrix assembly.
        """
        # u_e has shape (N, 24)
        u_e = u[self.elem_dofs]
        # (N, 24) @ (24, 24) -> (N, 24)
        k0_ue = np.dot(u_e, self.k_0)
        # Row-wise dot product
        U_0 = np.einsum('ij,ij->i', u_e, k0_ue)
        return np.maximum(U_0, 0.0)

    def optimize(self) -> MultiMaterialResult:
        """
        Execute multi-material topology optimization loop.
        """
        start_time = time.time()
        
        # Initial uniform densities
        init_val = (self.config.target_mass_fraction / self.M) * 0.8
        rho = np.full((self.num_elements, self.M), init_val, dtype=np.float64)
        rho = project_simplex_batch(rho)
        
        compliance_history = []
        mass_history = []
        
        if self.config.verbose:
            print("=" * 70)
            print("   WNFEA MULTI-MATERIAL TOPOLOGY OPTIMIZATION")
            print(f"   Resolution: {self.grid.resolution} | Elements: {self.num_elements}")
            print(f"   Materials: {[m.name for m in self.materials]}")
            print(f"   Target Mass: {self.target_mass:.2f} kg ({self.config.target_mass_fraction*100:.1f}%)")
            print(f"   Max Iterations: {self.config.max_iterations}")
            print("=" * 70)

        for iteration in range(self.config.max_iterations):
            iter_start = time.time()
            
            # 1. Evaluate effective Young's modulus across elements
            E_eff = self.interpolator.compute_effective_modulus(rho)
            
            # 2. Solve forward matrix-free linear static equilibrium
            # Scale relative to reference E_ref so that (alpha_eff)^p * E_ref = E_eff
            E_ref = float(np.max(self.interpolator.E_vec))
            alpha_eff = np.clip((E_eff / E_ref) ** (1.0 / self.config.simp_penalty), 1e-4, 1.0)
            self.grid.volume_fractions = alpha_eff
            u, elem_vm, pcg_iters = solve_voxel_linear_static(
                self.grid,
                self.forces,
                self.fixed_dofs,
                E=E_ref,
                nu=0.30,
                tol=1e-5,
                maxiter=500,
                verbose=False,
            )
            
            # 3. Evaluate elemental reference strain energy
            U_0 = self.compute_element_strain_energies(u)
            
            # 4. Total compliance c = f^T u = sum_e E_eff_e * U_0_e
            compliance = float(np.sum(E_eff * U_0))
            current_mass = self.interpolator.compute_total_mass(rho, self.elem_volume)
            
            compliance_history.append(compliance)
            mass_history.append(current_mass)
            
            # 5. Evaluate adjoint sensitivities
            # dc/drho: (N, M)
            dc_drho = self.interpolator.compute_compliance_sensitivities(rho, U_0)
            
            # Filter compliance sensitivities spatially for each material
            filtered_dc_drho = np.zeros_like(dc_drho)
            for m in range(self.M):
                filtered_dc_drho[:, m] = self.filter.filter_sensitivities(dc_drho[:, m])
                
            # 6. Multi-Material Optimality Criteria (MMOC) bisection update
            # Sensitivity of mass: dM/drho_{e, m} = v_e * density_m
            dM_drho = self.interpolator.compute_mass_sensitivities(self.num_elements, self.elem_volume)
            
            # Move limits
            move = self.config.move_limit
            
            # Dual bisection on Lagrange multiplier lambda for the mass constraint
            l1 = 1e-12
            l2 = 1e10
            
            rho_old = rho.copy()
            rho_cand = np.zeros_like(rho)
            
            for _ in range(40):
                l_mid = 0.5 * (l1 + l2)
                
                # Compute tentative OC step for each material
                ratio = np.maximum(-filtered_dc_drho, 0.0) / (l_mid * dM_drho + 1e-15)
                B = np.sqrt(ratio)
                
                # Apply move limits
                rho_trial = np.clip(
                    rho_old * B,
                    rho_old - move,
                    rho_old + move
                )
                
                # Project onto the simplex: sum_m rho_{e, m} <= 1, rho_{e, m} >= 0
                rho_cand = project_simplex_batch(rho_trial)
                
                # Check candidate mass
                trial_mass = self.interpolator.compute_total_mass(rho_cand, self.elem_volume)
                if trial_mass > self.target_mass:
                    l1 = l_mid
                else:
                    l2 = l_mid
                    
                if abs(l2 - l1) / (l1 + 1e-15) < 1e-4:
                    break
                    
            rho = rho_cand
            
            # Apply AM overhang filter if enabled
            if self.am_filter is not None and iteration >= 2:
                tot_rho = np.sum(rho, axis=1)
                filtered_tot = self.am_filter.filter_am_densities(tot_rho)
                # Scale down any unsupported material fractions proportionally
                scale = np.where(tot_rho > 1e-6, np.minimum(1.0, filtered_tot / (tot_rho + 1e-12)), 1.0)
                rho = rho * scale[:, None]
                
            max_change = float(np.max(np.abs(rho - rho_old)))
            iter_time = time.time() - iter_start
            
            if self.config.verbose:
                print(f"Iter {iteration+1:2d} | Compliance: {compliance:10.4e} J | "
                      f"Mass: {current_mass:7.2f} kg ({current_mass/self.max_domain_mass*100:4.1f}%) | "
                      f"Change: {max_change:6.4f} | Time: {iter_time*1000:6.1f} ms")
                      
            if iteration >= 5 and max_change < self.config.convergence_tol:
                if self.config.verbose:
                    print(f"--> Converged at iteration {iteration+1} (max change {max_change:.5f} < {self.config.convergence_tol})")
                break

        total_time = time.time() - start_time
        num_iters = len(compliance_history)
        avg_iter_time = total_time / max(1, num_iters)
        
        # Dominant material determination for visualization/export:
        # Sum of fractions >= 0.20 means solid element, choose argmax; otherwise void (-1)
        tot_frac = np.sum(rho, axis=1)
        dom_mat = np.where(tot_frac >= 0.20, np.argmax(rho, axis=1), -1)
        
        mat_volumes = [float(np.sum(rho[:, m]) * self.elem_volume) for m in range(self.M)]
        final_mass = self.interpolator.compute_total_mass(rho, self.elem_volume)
        
        am_printable = None
        if self.am_filter is not None:
            diag = self.am_filter.analyze_printability(tot_frac)
            am_printable = diag.is_printable
        
        return MultiMaterialResult(
            optimized_densities=rho,
            dominant_material=dom_mat,
            final_compliance=compliance_history[-1],
            final_mass=final_mass,
            mass_fraction=final_mass / self.max_domain_mass,
            material_volumes=mat_volumes,
            total_iterations=num_iters,
            total_time=total_time,
            average_iter_time=avg_iter_time,
            compliance_history=compliance_history,
            mass_history=mass_history,
            am_is_printable=am_printable,
        )
