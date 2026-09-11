"""
Rate-Independent J2 von Mises Elasto-Plasticity & Isotropic Hardening Engine for WNFEA (Phase 22).
-------------------------------------------------------------------------------------------------
Formulates:
1. Classical rate-independent J2 flow theory with associative normality rule:
      f(sigma, alpha) = sigma_vm - sigma_y(alpha) <= 0
      d_eps_p = d_gamma * N = d_gamma * (3/2) * (s / sigma_vm)
2. Isotropic Hardening Laws:
      - Linear Bilinear Hardening: sigma_y(alpha) = sigma_y0 + H * alpha
      - Voce Non-Linear Saturation: sigma_y(alpha) = sigma_y0 + R_inf * (1 - exp(-b * alpha))
      - Swift Power-Law Hardening: sigma_y(alpha) = K * (eps_0 + alpha)^n
      - Perfectly Plastic (H = 0)
3. Aerospace & Structural Metal Material Presets:
      - Al 6061-T6, Al 7075-T6, Ti-6Al-4V, Structural Steel S355, 304 Stainless Steel.
4. Implicit Radial Return Mapping Algorithm:
      - Vectorized trial elastic evaluation and scalar consistency parameter d_gamma solve.
      - Exact plastic incompressibility tr(d_eps_p) = 0.
5. Consistent Algorithmic Elastoplastic Tangent Modulus C^ep = d(sigma_n+1) / d(eps_n+1):
      - Strictly symmetric positive-semidefinite tensor guaranteeing quadratic Newton-Raphson convergence.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Union
import numpy as np


class HardeningType(enum.Enum):
    """Supported isotropic hardening laws for metallic materials."""
    LINEAR = "linear"
    VOCE = "voce"
    POWER_LAW = "power_law"
    PERFECT_PLASTIC = "perfect_plastic"


@dataclass
class MetalPlasticMaterial:
    """
    Metallic elasto-plastic material with J2 von Mises yield criterion and isotropic hardening.
    """
    name: str = "Structural Steel S355"
    E: float = 210e9                   # Young's modulus (Pa)
    nu: float = 0.30                   # Poisson's ratio
    sigma_y0: float = 355e6            # Initial yield strength (Pa)
    hardening_type: HardeningType = HardeningType.LINEAR

    # Linear hardening parameters
    H: float = 2.1e9                   # Plastic modulus H = E * Et / (E - Et) (Pa)

    # Voce saturation hardening parameters: sigma_y = sigma_y0 + R_inf * (1 - exp(-b * alpha))
    R_inf: float = 0.0                 # Saturation stress increment (Pa)
    b_voce: float = 0.0                # Saturation rate parameter

    # Swift power-law parameters: sigma_y = K * (eps_0 + alpha)^n
    K_swift: float = 0.0               # Strength coefficient (Pa)
    eps_0: float = 0.0                 # Initial offset strain
    n_swift: float = 0.1               # Hardening exponent

    @property
    def G(self) -> float:
        """Shear modulus G = E / (2 * (1 + nu)) in Pa."""
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def K_bulk(self) -> float:
        """Bulk modulus K = E / (3 * (1 - 2*nu)) in Pa."""
        return self.E / (3.0 * (1.0 - 2.0 * self.nu))

    def yield_stress(self, alpha: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate flow stress sigma_y(alpha) at accumulated plastic strain alpha."""
        a = np.maximum(alpha, 0.0)
        if self.hardening_type == HardeningType.LINEAR:
            return self.sigma_y0 + self.H * a
        elif self.hardening_type == HardeningType.VOCE:
            return self.sigma_y0 + self.R_inf * (1.0 - np.exp(-self.b_voce * a))
        elif self.hardening_type == HardeningType.POWER_LAW:
            return self.K_swift * ((self.eps_0 + a) ** self.n_swift)
        elif self.hardening_type == HardeningType.PERFECT_PLASTIC:
            return np.full_like(a, self.sigma_y0) if isinstance(a, np.ndarray) else self.sigma_y0
        else:
            return self.sigma_y0 + self.H * a

    def hardening_slope(self, alpha: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate hardening slope H'(alpha) = d(sigma_y) / d(alpha) in Pa."""
        a = np.maximum(alpha, 0.0)
        if self.hardening_type == HardeningType.LINEAR:
            return np.full_like(a, self.H) if isinstance(a, np.ndarray) else self.H
        elif self.hardening_type == HardeningType.VOCE:
            return self.R_inf * self.b_voce * np.exp(-self.b_voce * a)
        elif self.hardening_type == HardeningType.POWER_LAW:
            return self.K_swift * self.n_swift * ((self.eps_0 + a) ** (self.n_swift - 1.0))
        elif self.hardening_type == HardeningType.PERFECT_PLASTIC:
            return np.zeros_like(a) if isinstance(a, np.ndarray) else 0.0
        else:
            return np.full_like(a, self.H) if isinstance(a, np.ndarray) else self.H

    def compute_elasticity_matrix(self) -> np.ndarray:
        """Compute standard 6x6 isotropic elastic stiffness matrix C_e in Voigt notation."""
        K = self.K_bulk
        G = self.G
        C_e = np.array([
            [K + 4.0 / 3.0 * G, K - 2.0 / 3.0 * G, K - 2.0 / 3.0 * G, 0.0, 0.0, 0.0],
            [K - 2.0 / 3.0 * G, K + 4.0 / 3.0 * G, K - 2.0 / 3.0 * G, 0.0, 0.0, 0.0],
            [K - 2.0 / 3.0 * G, K - 2.0 / 3.0 * G, K + 4.0 / 3.0 * G, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, G, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, G, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, G],
        ], dtype=np.float64)
        return C_e

    # Standard Aerospace & Structural Alloy Presets
    @classmethod
    def aluminum_6061_t6(cls) -> MetalPlasticMaterial:
        """Al 6061-T6 aerospace aluminum with Voce non-linear saturation hardening."""
        return cls(
            name="Aluminum 6061-T6",
            E=68.9e9,
            nu=0.33,
            sigma_y0=276e6,
            hardening_type=HardeningType.VOCE,
            R_inf=45e6,
            b_voce=55.0,
            H=800e6,
        )

    @classmethod
    def aluminum_7075_t6(cls) -> MetalPlasticMaterial:
        """Al 7075-T6 high-strength aerospace alloy with linear isotropic hardening."""
        return cls(
            name="Aluminum 7075-T6",
            E=71.7e9,
            nu=0.33,
            sigma_y0=503e6,
            hardening_type=HardeningType.LINEAR,
            H=1.85e9,
        )

    @classmethod
    def titanium_ti6al4v(cls) -> MetalPlasticMaterial:
        """Ti-6Al-4V Grade 5 titanium alloy with Voce saturation hardening."""
        return cls(
            name="Titanium Ti-6Al-4V",
            E=113.8e9,
            nu=0.342,
            sigma_y0=880e6,
            hardening_type=HardeningType.VOCE,
            R_inf=140e6,
            b_voce=40.0,
            H=2.5e9,
        )

    @classmethod
    def structural_steel_s355(cls) -> MetalPlasticMaterial:
        """Structural Steel S355 with bilinear isotropic hardening."""
        return cls(
            name="Structural Steel S355",
            E=210e9,
            nu=0.30,
            sigma_y0=355e6,
            hardening_type=HardeningType.LINEAR,
            H=2.1e9,
        )

    @classmethod
    def stainless_steel_304(cls) -> MetalPlasticMaterial:
        """304 Austenitic Stainless Steel with Swift power-law work hardening."""
        return cls(
            name="Stainless Steel 304",
            E=193e9,
            nu=0.29,
            sigma_y0=205e6,
            hardening_type=HardeningType.POWER_LAW,
            K_swift=1250e6,
            eps_0=0.0015,
            n_swift=0.45,
            H=1.5e9,
        )


@dataclass
class PlasticHistoryState:
    """
    History variables tracked per integration point or element across loading increments.
    """
    eps_p: np.ndarray        # (N, 6) or (6,) plastic strain tensor [xx, yy, zz, yz, zx, xy]
    alpha_p: np.ndarray      # (N,) or float accumulated equivalent plastic strain
    yielded: np.ndarray      # (N,) or bool active yield state flag

    @classmethod
    def initialize(cls, num_points: int = 1) -> PlasticHistoryState:
        """Create zero-strain initialized state for N evaluation points."""
        if num_points == 1:
            return cls(
                eps_p=np.zeros(6, dtype=np.float64),
                alpha_p=np.array(0.0, dtype=np.float64),
                yielded=np.array(False, dtype=bool),
            )
        else:
            return cls(
                eps_p=np.zeros((num_points, 6), dtype=np.float64),
                alpha_p=np.zeros(num_points, dtype=np.float64),
                yielded=np.zeros(num_points, dtype=bool),
            )


# Precomputed Deviatoric Projection Matrix P_dev in Voigt [xx, yy, zz, yz, zx, xy]
_P_DEV = np.array([
    [2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, 0.0, 0.0, 0.0],
    [-1.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0, 0.0, 0.0, 0.0],
    [-1.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
], dtype=np.float64)

_M_VEC = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
_M_OUTER = np.outer(_M_VEC, _M_VEC)


def radial_return_mapping(
    material: MetalPlasticMaterial,
    eps_total: np.ndarray,
    state_old: Optional[PlasticHistoryState] = None,
    compute_tangent: bool = True,
) -> Tuple[np.ndarray, PlasticHistoryState, Optional[np.ndarray]]:
    """
    Perform implicit backward-Euler radial return mapping for J2 von Mises plasticity.

    Parameters:
        material: MetalPlasticMaterial instance with elasticity and hardening parameters.
        eps_total: (6,) or (N, 6) total strain tensor at current step n+1.
        state_old: PlasticHistoryState containing eps_p and alpha_p at step n.
        compute_tangent: If True, evaluates the exact consistent algorithmic tangent C^ep.

    Returns:
        stress_new: (6,) or (N, 6) updated stress tensor in Voigt.
        state_new: Updated PlasticHistoryState instance.
        C_ep: (6, 6) or (N, 6, 6) consistent algorithmic elastoplastic tangent matrix.
    """
    eps_in = np.asarray(eps_total, dtype=np.float64)
    is_single = (eps_in.ndim == 1)

    if is_single:
        eps_arr = eps_in.reshape(1, 6)
    else:
        eps_arr = eps_in

    N_pts = eps_arr.shape[0]

    if state_old is None:
        state_old = PlasticHistoryState.initialize(N_pts if not is_single else 1)

    eps_p_old = np.atleast_2d(state_old.eps_p)
    alpha_p_old = np.atleast_1d(state_old.alpha_p)

    C_e = material.compute_elasticity_matrix()
    K_bulk = material.K_bulk
    G = material.G

    # Elastic trial strain: eps_e_tr = eps_tot - eps_p_old
    eps_e_tr = eps_arr - eps_p_old  # (N, 6)
    sig_tr = eps_e_tr @ C_e.T       # (N, 6)

    # Hydrostatic trial pressure: p_tr = (sig_xx + sig_yy + sig_zz) / 3
    p_tr = np.mean(sig_tr[:, :3], axis=1, keepdims=True)  # (N, 1)

    # Deviatoric trial stress s_tr
    s_tr = sig_tr.copy()
    s_tr[:, :3] -= p_tr  # (N, 6)

    # von Mises equivalent stress: sqrt(1.5 * (s_xx^2 + s_yy^2 + s_zz^2 + 2*(s_yz^2 + s_zx^2 + s_xy^2)))
    s_sq_sum = np.sum(s_tr[:, :3] ** 2, axis=1) + 2.0 * np.sum(s_tr[:, 3:] ** 2, axis=1)
    vm_tr = np.sqrt(1.5 * np.maximum(s_sq_sum, 1e-24))  # (N,)

    # Yield condition check: f_tr = vm_tr - sigma_y(alpha_old)
    sig_y_old = material.yield_stress(alpha_p_old)
    f_tr = vm_tr - sig_y_old  # (N,)

    # Initialize output containers
    stress_new = sig_tr.copy()
    eps_p_new = eps_p_old.copy()
    alpha_p_new = alpha_p_old.copy()
    yielded = (f_tr > 0.0)

    C_ep = np.zeros((N_pts, 6, 6), dtype=np.float64) if compute_tangent else None
    if compute_tangent:
        C_ep[:] = C_e[None, :, :]

    # Plastic step for yielding integration points
    plastic_indices = np.where(yielded)[0]

    for idx in plastic_indices:
        vm_val = vm_tr[idx]
        s_val = s_tr[idx]
        a_old = alpha_p_old[idx]
        p_val = p_tr[idx, 0]

        # Solve scalar non-linear equation for plastic multiplier d_gamma:
        # Phi(d_gamma) = vm_val - 3 * G * d_gamma - sigma_y(a_old + d_gamma) = 0
        if material.hardening_type == HardeningType.LINEAR:
            d_gamma = float(f_tr[idx] / (3.0 * G + material.H))
        elif material.hardening_type == HardeningType.PERFECT_PLASTIC:
            d_gamma = float(f_tr[idx] / (3.0 * G))
        else:
            # Local Newton-Raphson iteration for non-linear hardening (Voce / Swift)
            d_gamma = float(f_tr[idx] / (3.0 * G + material.hardening_slope(a_old)))
            for _ in range(12):
                res = vm_val - 3.0 * G * d_gamma - material.yield_stress(a_old + d_gamma)
                if abs(res) < 1e-9:
                    break
                deriv = -3.0 * G - material.hardening_slope(a_old + d_gamma)
                d_gamma -= res / deriv

        d_gamma = max(d_gamma, 0.0)
        a_new = a_old + d_gamma

        # Scaling factor beta_0 = 1 - 3 * G * d_gamma / vm_val
        beta_0 = 1.0 - 3.0 * G * d_gamma / vm_val

        # Update stress: sigma_n+1 = p_tr * m + beta_0 * s_tr
        s_new = beta_0 * s_val
        stress_new[idx, :3] = p_val + s_new[:3]
        stress_new[idx, 3:] = s_new[3:]

        # Direction of plastic flow N in Voigt notation:
        # d_eps_p_ij = d_gamma * (3 / (2 * vm_tr)) * s_ij
        # For engineering shear: gamma_ij = 2 * eps_ij => factor of 2 on shear components
        flow_n = (1.5 / vm_val) * np.array([
            s_val[0], s_val[1], s_val[2], 2.0 * s_val[3], 2.0 * s_val[4], 2.0 * s_val[5]
        ], dtype=np.float64)

        eps_p_new[idx] += d_gamma * flow_n
        alpha_p_new[idx] = a_new

        # Compute consistent algorithmic elastoplastic tangent modulus C^ep:
        # C^ep = K_bulk * (m x m) + 2 * G * beta_0 * P_dev - c_scalar * (s_tr x s_tr)
        if compute_tangent:
            H_prime = material.hardening_slope(a_new)
            c_scalar = (9.0 * (G**2) / (vm_val**2)) * (
                (1.0 / (3.0 * G + H_prime)) - (d_gamma / vm_val)
            )
            C_ep[idx] = (
                K_bulk * _M_OUTER
                + 2.0 * G * beta_0 * _P_DEV
                - c_scalar * np.outer(s_val, s_val)
            )

    state_out = PlasticHistoryState(
        eps_p=eps_p_new[0] if is_single else eps_p_new,
        alpha_p=alpha_p_new[0] if is_single else alpha_p_new,
        yielded=yielded[0] if is_single else yielded,
    )

    out_stress = stress_new[0] if is_single else stress_new
    out_tangent = C_ep[0] if (compute_tangent and is_single) else C_ep

    return out_stress, state_out, out_tangent
