"""
Additive Manufacturing (AM) Overhang Angle Constraint Filter for WNFEA.
-----------------------------------------------------------------------
Enforces self-supporting printability constraints for powder bed fusion (SLM/DMLS)
and extrusion (FDM) additive manufacturing processes:
1. Critical overhang angle criterion: features with slope < alpha_crit (typically 45 deg)
   relative to the build plate require sacrificial support structures.
2. Layer-by-layer support neighborhood calculation:
       S_{i, j, k} = max_{(u, v) in N_supp(i, j)} rho_{u, v, k-1}
3. Differentiable overhang violation penalty:
       V_{overhang}(rho) = sum_{k=1}^{Nz-1} sum_{i, j} max(0, rho_{i, j, k} - S_{i, j, k})^2
   with exact reverse-mode analytical gradient backpropagation.
4. Forward printability filter: recursively enforces self-supporting geometries
   without manual support design.
5. Flexible build vector support: +z, -z, +y, -y, +x, -x.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, List, Tuple, Union, Literal
import numpy as np

from ..mesh.voxel_mesher import VoxelGrid


@dataclass
class AMFilterResult:
    """
    Diagnostics and metrics from additive manufacturing overhang analysis.
    """
    is_printable: bool                    # True if max overhang violation < tolerance
    overhang_penalty: float               # Value of V_overhang
    max_violation: float                  # Maximum local unsupported density excess
    unsupported_voxel_count: int          # Number of voxels with unsupported density > 0.1
    unsupported_volume: float             # Volume in m^3 of unsupported features
    projected_density: np.ndarray         # (N,) self-supporting printable density field


class AMOverhangFilter:
    """
    Differentiable Additive Manufacturing Overhang Angle Filter & Constraint Operator.
    """
    def __init__(
        self,
        grid: VoxelGrid,
        build_direction: Literal["+z", "-z", "+y", "-y", "+x", "-x"] = "+z",
        critical_angle_deg: float = 45.0,
        support_radius_factor: float = 1.0,
    ):
        self.grid = grid
        self.build_direction = build_direction
        self.critical_angle_rad = np.radians(float(critical_angle_deg))
        self.dx, self.dy, self.dz = grid.pitch
        self.nx, self.ny, self.nz = grid.resolution
        self.num_elements = grid.total_cells
        self.elem_volume = float(self.dx * self.dy * self.dz)
        
        # Determine coordinate mapping based on build direction
        # Re-orient so that axis 0 is the build layer (height from 0 to N_layers - 1)
        self._setup_axis_mapping()
        
        # Build 2D supporting neighborhood offsets in the layer plane
        self.supp_offsets = self._compute_neighborhood_offsets(support_radius_factor)

    def _setup_axis_mapping(self):
        """Map the 3D voxel grid axes (nz, ny, nx) to (n_layers, n_u, n_v)."""
        # Standard grid indexing is (k_z, j_y, i_x)
        if self.build_direction == "+z":
            self.layer_axis = 0
            self.flip_layer = False
            self.h_step = self.dz
            self.u_step, self.v_step = self.dy, self.dx
        elif self.build_direction == "-z":
            self.layer_axis = 0
            self.flip_layer = True
            self.h_step = self.dz
            self.u_step, self.v_step = self.dy, self.dx
        elif self.build_direction == "+y":
            self.layer_axis = 1
            self.flip_layer = False
            self.h_step = self.dy
            self.u_step, self.v_step = self.dz, self.dx
        elif self.build_direction == "-y":
            self.layer_axis = 1
            self.flip_layer = True
            self.h_step = self.dy
            self.u_step, self.v_step = self.dz, self.dx
        elif self.build_direction == "+x":
            self.layer_axis = 2
            self.flip_layer = False
            self.h_step = self.dx
            self.u_step, self.v_step = self.dz, self.dy
        elif self.build_direction == "-x":
            self.layer_axis = 2
            self.flip_layer = True
            self.h_step = self.dx
            self.u_step, self.v_step = self.dz, self.dy
        else:
            raise ValueError(f"Unsupported build direction: {self.build_direction}")

    def _to_layer_view(self, rho_flat: np.ndarray) -> np.ndarray:
        """Convert 1D (N,) density into (n_layers, n_u, n_v) 3D array."""
        rho_3d = rho_flat.reshape((self.nz, self.ny, self.nx))
        if self.layer_axis == 0:
            arr = rho_3d
        elif self.layer_axis == 1:
            arr = np.transpose(rho_3d, (1, 0, 2))
        else:
            arr = np.transpose(rho_3d, (2, 0, 1))
            
        if self.flip_layer:
            arr = arr[::-1, :, :]
        return arr

    def _from_layer_view(self, arr: np.ndarray) -> np.ndarray:
        """Convert (n_layers, n_u, n_v) array back to 1D (N,) flat array."""
        if self.flip_layer:
            arr = arr[::-1, :, :]
            
        if self.layer_axis == 0:
            rho_3d = arr
        elif self.layer_axis == 1:
            rho_3d = np.transpose(arr, (1, 0, 2))
        else:
            rho_3d = np.transpose(arr, (1, 2, 0))
            
        return rho_3d.reshape(-1)

    def _compute_neighborhood_offsets(self, radius_factor: float) -> List[Tuple[int, int]]:
        """
        Compute relative (du, dv) cell offsets in layer k-1 that can support an element in layer k.
        Support reach: R = h_step / tan(critical_angle) * radius_factor.
        """
        max_reach = (self.h_step / np.tan(self.critical_angle_rad)) * radius_factor
        
        # Max integer grid offsets
        max_du = max(1, int(np.floor(max_reach / self.u_step + 1e-5)))
        max_dv = max(1, int(np.floor(max_reach / self.v_step + 1e-5)))
        
        offsets = []
        for du in range(-max_du, max_du + 1):
            for dv in range(-max_dv, max_dv + 1):
                dist = np.sqrt((du * self.u_step) ** 2 + (dv * self.v_step) ** 2)
                if dist <= max_reach + 1e-6:
                    offsets.append((du, dv))
        return offsets

    def compute_layer_support(self, prev_layer: np.ndarray) -> np.ndarray:
        """
        Evaluate available support field in the current layer from the previous layer:
            S(u, v) = max_{(du, dv) in offsets} prev_layer(u + du, v + dv)
        using vectorized boundary padding.
        """
        nu, nv = prev_layer.shape
        S = np.zeros_like(prev_layer)
        
        # Vectorized maximum over neighborhood offsets
        for du, dv in self.supp_offsets:
            u_start_src = max(0, du)
            u_end_src = min(nu, nu + du)
            u_start_dst = max(0, -du)
            u_end_dst = min(nu, nu - du)
            
            v_start_src = max(0, dv)
            v_end_src = min(nv, nv + dv)
            v_start_dst = max(0, -dv)
            v_end_dst = min(nv, nv - dv)
            
            src_patch = prev_layer[u_start_src:u_end_src, v_start_src:v_end_src]
            S[u_start_dst:u_end_dst, v_start_dst:v_end_dst] = np.maximum(
                S[u_start_dst:u_end_dst, v_start_dst:v_end_dst],
                src_patch
            )
        return S

    def evaluate_overhang_penalty(self, rho: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute total overhang violation penalty and its analytical gradient:
            V = 0.5 * sum_{k=1}^{K-1} sum_{u, v} max(0, rho_{k, u, v} - S_{k, u, v})^2
            
        Returns
        -------
        penalty : float
            Total squared unsupported volume residual.
        grad : np.ndarray, shape (N,)
            Analytical gradient dV/d(rho).
        """
        layers = self._to_layer_view(rho)
        n_layers, nu, nv = layers.shape
        
        penalty = 0.0
        # Forward pass: compute supports and violations
        # S_all stores support for layer k from layer k-1
        S_all = np.zeros((n_layers, nu, nv), dtype=np.float64)
        # Layer 0 has full support from the build plate
        S_all[0] = 1.0
        
        violations = np.zeros((n_layers, nu, nv), dtype=np.float64)
        
        for k in range(1, n_layers):
            S_k = self.compute_layer_support(layers[k - 1])
            S_all[k] = S_k
            viol_k = np.maximum(0.0, layers[k] - S_k)
            violations[k] = viol_k
            penalty += 0.5 * np.sum(viol_k ** 2)
            
        # Backward pass: compute analytical adjoint sensitivities
        # dV/d(rho_k) = viol_k - adjoint_spread(viol_{k+1})
        grad_layers = np.zeros_like(layers)
        
        # Direct derivative from layer k
        grad_layers[1:] = violations[1:]
        
        # Reverse backpropagation of support dependency:
        # S_{k, u, v} = max_{(du, dv)} rho_{k-1, u+du, v+dv}
        # Subgradient: if rho_{k-1, u+du, v+dv} achieved the max S_{k, u, v}, it receives -violations[k]
        for k in range(n_layers - 1, 0, -1):
            viol_k = violations[k]
            prev_layer = layers[k - 1]
            S_k = S_all[k]
            
            # Find which neighbors achieved S_k
            for du, dv in self.supp_offsets:
                u_start_src = max(0, du)
                u_end_src = min(nu, nu + du)
                u_start_dst = max(0, -du)
                u_end_dst = min(nu, nu - du)
                
                v_start_src = max(0, dv)
                v_end_src = min(nv, nv + dv)
                v_start_dst = max(0, -dv)
                v_end_dst = min(nv, nv - dv)
                
                src_patch = prev_layer[u_start_src:u_end_src, v_start_src:v_end_src]
                dst_viol = viol_k[u_start_dst:u_end_dst, v_start_dst:v_end_dst]
                dst_S = S_k[u_start_dst:u_end_dst, v_start_dst:v_end_dst]
                
                # If neighbor was maximal and violation > 0, backpropagate negative gradient
                mask = (src_patch >= dst_S - 1e-10) & (dst_viol > 0.0)
                grad_layers[k - 1, u_start_src:u_end_src, v_start_src:v_end_src] -= np.where(mask, dst_viol, 0.0)
                
        grad_flat = self._from_layer_view(grad_layers)
        return float(penalty), grad_flat

    def filter_am_densities(self, rho: np.ndarray) -> np.ndarray:
        """
        Forward recursive projection: produces a 100% self-supporting printable density field
        by strictly clipping any unsupported features exceeding critical angle alpha_crit.
        
        Parameters
        ----------
        rho : np.ndarray, shape (N,)
        
        Returns
        -------
        printable_rho : np.ndarray, shape (N,)
        """
        layers = self._to_layer_view(rho).copy()
        n_layers, nu, nv = layers.shape
        
        # Layer 0 is on the build plate, completely preserved
        for k in range(1, n_layers):
            S_k = self.compute_layer_support(layers[k - 1])
            # Element cannot exceed the support available from the layer below
            layers[k] = np.minimum(layers[k], S_k)
            
        return self._from_layer_view(layers)

    def analyze_printability(self, rho: np.ndarray, tolerance: float = 0.05) -> AMFilterResult:
        """
        Evaluate full printability diagnostics and overhang statistics.
        """
        penalty, _ = self.evaluate_overhang_penalty(rho)
        printable_rho = self.filter_am_densities(rho)
        
        unsupported = np.maximum(0.0, rho - printable_rho)
        max_viol = float(np.max(unsupported))
        unsupp_mask = unsupported > tolerance
        unsupp_count = int(np.sum(unsupp_mask))
        unsupp_vol = float(unsupp_count * self.elem_volume)
        
        is_printable = max_viol <= tolerance
        
        return AMFilterResult(
            is_printable=is_printable,
            overhang_penalty=penalty,
            max_violation=max_viol,
            unsupported_voxel_count=unsupp_count,
            unsupported_volume=unsupp_vol,
            projected_density=printable_rho,
        )
