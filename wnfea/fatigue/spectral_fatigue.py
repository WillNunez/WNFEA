"""
Frequency-Domain Spectral Fatigue Damage Models for WNFEA (Phase 19).
--------------------------------------------------------------------
Provides fast, closed-form vibration fatigue damage estimation directly from
PSD spectral moments (m_0, m_1, m_2, m_4) without time-domain realization:

1. Steinberg 3-Band Gaussian Damage Model:
      dot{D} = E[P] * [ 0.683 / N(1*sigma_RMS) + 0.271 / N(2*sigma_RMS) + 0.0433 / N(3*sigma_RMS) ]
   Executes in microseconds and standard in aerospace electronics / avionics.

2. Dirlik Four-Moment Broadband Fatigue Model (1985):
   Uses Dirlik's empirical probability density function p(S) of rainflow cycle ranges
   derived from extensive Gaussian process simulations.
   Evaluates the expected damage rate:
      dot{D} = E[P] * int_0^infty [ p(S) / N(S) ] dS
   Utilizes an EXACT closed-form analytical expression in terms of the Gamma function:
      dot{D} = (E[P] / C) * (2*sqrt(m_0))^k * [ D_1 * Q^k * Gamma(1+k) + (sqrt(2))^k * Gamma(1+k/2) * (D_2 * R^k + D_3) ]
   where k = -1/b and C = 2^(k-1) * (sigma_f')^k from Basquin's S-N relationship.

3. Narrow-Band (Bendat / Rayleigh) Model:
   Upper-bound conservative damage model for narrow-band resonant peaks (gamma -> 1).

4. Mesh-Wide Spectral Fatigue Evaluator:
   Computes damage rates and time-to-failure (seconds, hours, flight blocks) across 3D meshes.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union
import numpy as np

from .fatigue_solver import FatigueMaterial, evaluate_sn_life


def evaluate_steinberg_damage_rate(
    sigma_rms: float,
    peak_rate: float,
    material: FatigueMaterial,
) -> float:
    """
    Evaluate fatigue damage rate (damage per second) using Steinberg's 3-band Gaussian model.

    Parameters:
        sigma_rms: 1-sigma root-mean-square stress amplitude (Pa).
        peak_rate: Expected rate of peaks E[P] in Hz.
        material: FatigueMaterial definition.

    Returns:
        damage_rate: Cumulative damage rate per second (1/s).
    """
    s_rms = float(sigma_rms)
    ep = float(peak_rate)

    if s_rms <= 0.0 or ep <= 0.0:
        return 0.0

    # 1-sigma, 2-sigma, and 3-sigma stress cycle counts per second:
    # 68.3% at 1*sigma_rms, 27.1% at 2*sigma_rms, 4.33% at 3*sigma_rms
    n1 = 0.683 * ep
    n2 = 0.271 * ep
    n3 = 0.0433 * ep

    nf1 = evaluate_sn_life(1.0 * s_rms, 0.0, material, correction="none")
    nf2 = evaluate_sn_life(2.0 * s_rms, 0.0, material, correction="none")
    nf3 = evaluate_sn_life(3.0 * s_rms, 0.0, material, correction="none")

    damage_rate = 0.0
    if nf1 < material.cut_off_cycles:
        damage_rate += n1 / nf1
    if nf2 < material.cut_off_cycles:
        damage_rate += n2 / nf2
    if nf3 < material.cut_off_cycles:
        damage_rate += n3 / nf3

    return float(damage_rate)


def evaluate_narrowband_damage_rate(
    m0: float,
    m2: float,
    material: FatigueMaterial,
) -> float:
    """
    Evaluate fatigue damage rate using Bendat's narrow-band Rayleigh model:
        dot{D}_NB = (E[0] / C) * (sqrt(2 * m_0))^k * Gamma(1 + k/2)
    where k = -1/b and C = 2^(k-1) * (sigma_f')^k.
    """
    if m0 <= 0.0 or m2 <= 0.0:
        return 0.0

    e0 = math.sqrt(m2 / m0)
    sigma_rms = math.sqrt(m0)

    # Basquin constants
    k = -1.0 / material.b
    C = (2.0 ** (k - 1.0)) * (material.sigma_f_prime ** k)

    # Analytical Rayleigh damage rate
    gamma_val = math.gamma(1.0 + 0.5 * k)
    damage_rate = (e0 / C) * ((math.sqrt(2.0) * sigma_rms) ** k) * gamma_val
    return float(damage_rate)


def evaluate_dirlik_damage_rate(
    m0: float,
    m1: float,
    m2: float,
    m4: float,
    material: FatigueMaterial,
) -> float:
    """
    Evaluate fatigue damage rate (1/s) using Dirlik's four-moment spectral model.

    Parameters:
        m0: 0th spectral moment (variance of stress, Pa^2).
        m1: 1st spectral moment.
        m2: 2nd spectral moment.
        m4: 4th spectral moment.
        material: FatigueMaterial property definition.

    Returns:
        damage_rate: Expected fatigue damage per second (1/s).
    """
    if m0 <= 0.0 or m2 <= 0.0 or m4 <= 0.0 or m1 <= 0.0:
        return 0.0

    sigma_rms = math.sqrt(m0)
    ep = math.sqrt(m4 / m2)

    # Spectral parameters
    gamma = float(np.clip(m2 / math.sqrt(m0 * m4), 1e-4, 0.9999))
    xm = float((m1 / m0) * math.sqrt(m2 / m4))

    # Dirlik coefficients
    d1 = 2.0 * (xm - gamma**2) / (1.0 + gamma**2)
    # If d1 is non-positive or unphysical, fallback to narrow-band model
    if d1 <= 1e-6 or xm < gamma**2:
        return evaluate_narrowband_damage_rate(m0, m2, material)

    denom_r = max(1.0 - gamma - d1 + d1**2, 1e-8)
    R = (gamma - xm - d1**2) / denom_r
    if R <= 0.0 or R >= 1.0:
        # Fallback to narrow-band Rayleigh if R is outside physical bounds (0, 1)
        return evaluate_narrowband_damage_rate(m0, m2, material)

    d2 = denom_r / max(1.0 - R, 1e-8)
    d3 = 1.0 - d1 - d2
    Q = 1.25 * (gamma - d3 - d2 * R) / max(d1, 1e-8)
    if Q <= 0.0:
        return evaluate_narrowband_damage_rate(m0, m2, material)

    # Basquin constants: S_a = sigma_f' * (2*N_f)^b ==> N(S) = C * S^(-k)
    k = -1.0 / material.b
    C = (2.0 ** (k - 1.0)) * (material.sigma_f_prime ** k)

    # Analytical closed-form Gamma integration
    # Term 1: Exponential component
    term1 = d1 * (Q ** k) * math.gamma(1.0 + k)
    # Term 2: Two Rayleigh components
    term2 = (math.sqrt(2.0) ** k) * math.gamma(1.0 + 0.5 * k) * (d2 * (R ** k) + d3)

    damage_rate = (ep / C) * ((2.0 * sigma_rms) ** k) * (term1 + term2)

    return float(max(damage_rate, 0.0))


def evaluate_mesh_spectral_fatigue(
    spectral_moments: Dict[str, np.ndarray],
    material: FatigueMaterial,
    method: str = "dirlik",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate mesh-wide spectral fatigue damage rate and time-to-failure.

    Parameters:
        spectral_moments: Dictionary containing 'm0', 'm1', 'm2', 'm4' arrays.
        material: FatigueMaterial definition.
        method: 'dirlik', 'steinberg', or 'narrowband'.

    Returns:
        damage_rates: (N_elements,) array of damage per second (1/s).
        time_to_failure_hours: (N_elements,) array of estimated life in hours.
    """
    m0 = np.asarray(spectral_moments["m0"], dtype=np.float64)
    m1 = np.asarray(spectral_moments.get("m1", m0), dtype=np.float64)
    m2 = np.asarray(spectral_moments["m2"], dtype=np.float64)
    m4 = np.asarray(spectral_moments["m4"], dtype=np.float64)

    n_elem = len(m0)
    damage_rates = np.zeros(n_elem, dtype=np.float64)
    time_to_failure_hours = np.full(n_elem, 1e12, dtype=np.float64)

    meth = method.lower()
    for e in range(n_elem):
        if m0[e] <= 0.0 or m2[e] <= 0.0:
            continue

        if meth == "dirlik":
            d_rate = evaluate_dirlik_damage_rate(m0[e], m1[e], m2[e], m4[e], material)
        elif meth == "steinberg":
            sigma_rms = math.sqrt(m0[e])
            ep = math.sqrt(m4[e] / m2[e]) if m2[e] > 0 else 0.0
            d_rate = evaluate_steinberg_damage_rate(sigma_rms, ep, material)
        elif meth == "narrowband":
            d_rate = evaluate_narrowband_damage_rate(m0[e], m2[e], material)
        else:
            raise ValueError(f"Unknown spectral fatigue method: '{method}'")

        damage_rates[e] = d_rate
        if d_rate > 0.0:
            ttf_sec = 1.0 / d_rate
            time_to_failure_hours[e] = min(ttf_sec / 3600.0, 1e12)
        else:
            time_to_failure_hours[e] = 1e12  # Infinite life

    return damage_rates, time_to_failure_hours
