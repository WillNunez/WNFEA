"""
Multiaxial Critical Plane Analysis & Dang Van Fatigue Limit Criterion for WNFEA.
---------------------------------------------------------------------------------
Evaluates:
1. Critical Plane Search:
   - Sweeps candidate material planes n(theta, phi).
   - Resolves time-dependent traction vector t(t) = sigma(t) * n.
   - Computes normal stress sigma_n(t) = n^T * sigma(t) * n and shear stress tau(t).
   - Formulates Findley, Fatemi-Socie, and Smith-Watson-Topper (SWT) parameters.
   - Identifies the critical orientation theta*, phi* maximizing fatigue damage.
2. Dang Van Mesoscopic Multiaxial Fatigue Limit Criterion:
   - Mesoscopic hydrostatic stress p_H(t) = 1/3 * tr(sigma(t)).
   - Deviatoric shakedown center s_0 and Tresca shear stress amplitude tau(t).
   - Evaluates Dang Van equivalent stress sigma_DV(t) = tau(t) + alpha_DV * p_H(t).
   - Computes multiaxial safety factor SF_DV against infinite fatigue life limit beta_DV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class CriticalPlaneResult:
    """Output summary from critical plane fatigue evaluation."""
    criterion: str
    parameter_max: float
    theta_rad: float
    phi_rad: float
    normal_vector: np.ndarray
    sigma_n_max: float
    sigma_n_mean: float
    tau_amplitude: float
    safety_factor: Optional[float] = None

    @property
    def theta_deg(self) -> float:
        return float(np.degrees(self.theta_rad))

    @property
    def phi_deg(self) -> float:
        return float(np.degrees(self.phi_rad))


def _stress_history_to_tensors(stress_history: np.ndarray) -> np.ndarray:
    """
    Convert (N_steps, 6) Voigt array [sxx, syy, szz, sxy, syz, szx]
    into (N_steps, 3, 3) symmetric stress tensors.
    """
    arr = np.asarray(stress_history, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[1:] == (3, 3):
        return arr
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError(f"Expected stress history of shape (N_steps, 6), got {arr.shape}")

    n_steps = arr.shape[0]
    tensors = np.zeros((n_steps, 3, 3), dtype=np.float64)
    tensors[:, 0, 0] = arr[:, 0]  # sxx
    tensors[:, 1, 1] = arr[:, 1]  # syy
    tensors[:, 2, 2] = arr[:, 2]  # szz

    tensors[:, 0, 1] = arr[:, 3]  # sxy
    tensors[:, 1, 0] = arr[:, 3]

    tensors[:, 1, 2] = arr[:, 4]  # syz
    tensors[:, 2, 1] = arr[:, 4]

    tensors[:, 0, 2] = arr[:, 5]  # szx
    tensors[:, 2, 0] = arr[:, 5]

    return tensors


def evaluate_critical_plane(
    stress_history: np.ndarray,
    criterion: str = "findley",
    k_param: float = 0.3,
    yield_strength: float = 355e6,
    fatigue_limit: Optional[float] = None,
    n_theta: int = 19,
    n_phi: int = 36,
) -> CriticalPlaneResult:
    """
    Identify the critical plane orientation and maximum fatigue parameter.

    Parameters:
        stress_history: Array of shape (N_steps, 6) in Voigt order or (N_steps, 3, 3).
        criterion: Critical plane model:
                   - 'findley': FP = tau_a + k * sigma_{n,max}
                   - 'fatemi_socie': FS = tau_a * (1 + k * sigma_{n,max} / sigma_y)
                   - 'swt': SWT = sigma_{n,max} * Delta_sigma_n / 2
        k_param: Material sensitivity parameter k (typically 0.2 - 0.4 for Findley).
        yield_strength: Material yield strength sigma_y (Pa) for Fatemi-Socie.
        fatigue_limit: Optional material fatigue limit (e.g. Findley f_F) to compute safety factor.
        n_theta: Number of polar angle samples in [0, pi/2].
        n_phi: Number of azimuthal angle samples in [0, pi).

    Returns:
        CriticalPlaneResult instance with critical orientation and parameters.
    """
    tensors = _stress_history_to_tensors(stress_history)
    n_steps = tensors.shape[0]

    crit_lower = criterion.lower()
    best_param = -1e30
    best_theta = 0.0
    best_phi = 0.0
    best_n = np.array([0.0, 0.0, 1.0])
    best_sn_max = 0.0
    best_sn_mean = 0.0
    best_tau_a = 0.0

    thetas = np.linspace(0.0, np.pi / 2.0, n_theta)
    phis = np.linspace(0.0, np.pi, n_phi, endpoint=False)

    for theta in thetas:
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        for phi in phis:
            n = np.array([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], dtype=np.float64)

            # Traction vectors across all time steps: t(t) = tensors(t) @ n
            # Shape: (N_steps, 3)
            tractions = np.einsum("tij,j->ti", tensors, n)

            # Normal stress: sigma_n(t) = n . t(t)
            # Shape: (N_steps,)
            sigma_n = np.einsum("ti,i->t", tractions, n)

            sn_max = float(np.max(sigma_n))
            sn_min = float(np.min(sigma_n))
            sn_mean = 0.5 * (sn_max + sn_min)
            delta_sn_half = 0.5 * (sn_max - sn_min)

            # Shear stress vector: tau_vec(t) = t(t) - sigma_n(t) * n
            tau_vec = tractions - sigma_n[:, None] * n[None, :]

            # Maximum shear amplitude across the time history
            # For non-proportional loading, tau_a = 0.5 * max_{t1, t2} ||tau(t1) - tau(t2)||
            # If N_steps is small (< 100), exact pairwise max; otherwise norm range
            if n_steps <= 128:
                # Pairwise difference
                diff = tau_vec[:, None, :] - tau_vec[None, :, :]
                dists_sq = np.sum(diff**2, axis=-1)
                tau_a = 0.5 * float(np.sqrt(np.max(dists_sq)))
            else:
                tau_mag = np.linalg.norm(tau_vec, axis=-1)
                tau_a = 0.5 * float(np.max(tau_mag) - np.min(tau_mag))

            # Compute fatigue parameter
            if crit_lower == "findley":
                param = tau_a + k_param * sn_max
            elif crit_lower == "fatemi_socie":
                param = tau_a * (1.0 + k_param * (sn_max / max(yield_strength, 1e-6)))
            elif crit_lower == "swt":
                param = max(sn_max, 0.0) * delta_sn_half
            else:
                raise ValueError(f"Unknown critical plane criterion: '{criterion}'")

            if param > best_param:
                best_param = param
                best_theta = theta
                best_phi = phi
                best_n = n.copy()
                best_sn_max = sn_max
                best_sn_mean = sn_mean
                best_tau_a = tau_a

    sf = None
    if fatigue_limit is not None and best_param > 1e-12:
        sf = float(fatigue_limit / best_param)

    return CriticalPlaneResult(
        criterion=criterion,
        parameter_max=float(best_param),
        theta_rad=float(best_theta),
        phi_rad=float(best_phi),
        normal_vector=best_n,
        sigma_n_max=float(best_sn_max),
        sigma_n_mean=float(best_sn_mean),
        tau_amplitude=float(best_tau_a),
        safety_factor=sf,
    )


def evaluate_dang_van_safety_factor(
    stress_history: np.ndarray,
    shear_fatigue_limit: float = 180e6,
    tensile_fatigue_limit: float = 300e6,
) -> Tuple[float, float, float]:
    """
    Evaluate the multiaxial Dang Van fatigue limit criterion.

    Condition for infinite fatigue life:
        tau_max(t) + alpha_DV * p_H(t) <= beta_DV

    Where:
        p_H(t) = 1/3 * tr(sigma(t)) is mesoscopic hydrostatic stress.
        tau(t) is the Tresca shear stress of the centered deviatoric stress path.
        alpha_DV and beta_DV are calibrated from pure torsion and pure bending limits:
            beta_DV = shear_fatigue_limit (tau_e)
            alpha_DV = 3 * (tau_e / sigma_e - 0.5)

    Parameters:
        stress_history: (N_steps, 6) Voigt array or (N_steps, 3, 3) tensors.
        shear_fatigue_limit: Pure shear / torsion endurance limit tau_e (Pa).
        tensile_fatigue_limit: Fully reversed tension/bending endurance limit sigma_e (Pa).

    Returns:
        safety_factor: Dang Van multiaxial safety factor SF_DV = beta_DV / max_t(sigma_DV(t)).
                       SF_DV >= 1.0 indicates safety against fatigue crack initiation.
        max_equivalent_stress: Maximum Dang Van equivalent stress max_t(tau + alpha*p_H).
        max_hydrostatic_stress: Maximum hydrostatic stress max_t(p_H(t)).
    """
    tensors = _stress_history_to_tensors(stress_history)
    n_steps = tensors.shape[0]

    # 1. Calibrate Dang Van coefficients
    tau_e = float(shear_fatigue_limit)
    sigma_e = float(tensile_fatigue_limit)
    beta_dv = tau_e
    alpha_dv = 3.0 * ((tau_e / max(sigma_e, 1e-6)) - 0.5)

    # 2. Hydrostatic stress: p_H(t) = 1/3 * (sxx + syy + szz)
    p_h = np.trace(tensors, axis1=1, axis2=2) / 3.0  # (N_steps,)

    # 3. Deviatoric stress tensor: s(t) = sigma(t) - p_H(t) * I
    dev = tensors - p_h[:, None, None] * np.eye(3)[None, :, :]

    # 4. Deviatoric shakedown center s_0 (mean or min-max midpoint)
    s_0 = 0.5 * (np.max(dev, axis=0) + np.min(dev, axis=0))
    dev_centered = dev - s_0[None, :, :]

    # 5. Mesoscopic Tresca shear stress: tau(t) = 0.5 * (s_1(t) - s_3(t))
    # Compute eigenvalues of dev_centered at each time step
    tau = np.zeros(n_steps, dtype=np.float64)
    for t in range(n_steps):
        eigvals = np.linalg.eigvalsh(dev_centered[t])
        tau[t] = 0.5 * (eigvals[2] - eigvals[0])

    # 6. Dang Van equivalent stress: sigma_DV(t) = tau(t) + alpha_DV * p_H(t)
    sigma_dv = tau + alpha_dv * p_h

    max_sigma_dv = float(np.max(sigma_dv))
    max_ph = float(np.max(p_h))

    if max_sigma_dv > 1e-12:
        sf = float(beta_dv / max_sigma_dv)
    else:
        sf = 1000.0  # Zero stress

    return sf, max_sigma_dv, max_ph
