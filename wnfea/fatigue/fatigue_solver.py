"""
ASTM E1049-85 Rainflow Cycle Counting & S-N Cumulative Damage Solver for WNFEA.
--------------------------------------------------------------------------------
Provides:
1. Turning point (peaks & valleys) extraction.
2. Standard ASTM E1049-85 Section 5.4.4 Rainflow cycle counting algorithm.
3. S-N fatigue curve evaluation via Basquin formulation:
      S_a = sigma_f' * (2 * N_f)^b  ==>  N_f = 0.5 * (S_eq / sigma_f')^(1 / b)
4. Mean stress corrections:
      - Goodman:    S_eq = S_a / (1 - S_m / S_ut)
      - Gerber:     S_eq = S_a / (1 - (S_m / S_ut)^2)
      - Morrow:     S_eq = S_a / (1 - S_m / sigma_f')
      - Soderberg:  S_eq = S_a / (1 - S_m / S_y)
5. Palmgren-Miner cumulative linear damage accumulation:
      D = sum_i (n_i / N_i)
6. Element-wise fatigue damage and life evaluation across FEA stress histories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np


@dataclass(frozen=True)
class RainflowCycle:
    """Represents a rainflow-counted stress cycle or half-cycle."""
    range: float       # Peak-to-peak range Delta_sigma
    mean: float        # Mean stress sigma_m
    count: float       # 1.0 for full cycle, 0.5 for half cycle

    @property
    def amplitude(self) -> float:
        """Stress amplitude S_a = Delta_sigma / 2."""
        return 0.5 * self.range

    @property
    def stress_min(self) -> float:
        """Minimum valley stress."""
        return self.mean - 0.5 * self.range

    @property
    def stress_max(self) -> float:
        """Maximum peak stress."""
        return self.mean + 0.5 * self.range

    @property
    def r_ratio(self) -> float:
        """Stress ratio R = sigma_min / sigma_max."""
        s_max = self.stress_max
        if abs(s_max) < 1e-12:
            return -1.0
        return self.stress_min / s_max


@dataclass
class FatigueMaterial:
    """
    Fatigue property dataset for metallic alloys.
    Units in SI (Pascals).
    """
    name: str = "Structural Steel S355"
    E: float = 210e9               # Elastic modulus (Pa)
    sigma_y: float = 355e6         # Yield strength (Pa)
    sigma_ut: float = 510e6        # Ultimate tensile strength (Pa)
    sigma_f_prime: float = 950e6   # Fatigue strength coefficient (Pa)
    b: float = -0.09               # Basquin fatigue strength exponent
    endurance_limit: float = 220e6 # Fatigue endurance limit (Pa)
    cut_off_cycles: float = 1e12   # Threshold representing infinite life

    @classmethod
    def aluminum_7075_t6(cls) -> "FatigueMaterial":
        """Aerospace Grade Aluminum 7075-T6."""
        return cls(
            name="Aluminum 7075-T6",
            E=71.7e9,
            sigma_y=503e6,
            sigma_ut=572e6,
            sigma_f_prime=1313e6,
            b=-0.126,
            endurance_limit=159e6,
            cut_off_cycles=1e12,
        )

    @classmethod
    def titanium_ti6al4v(cls) -> "FatigueMaterial":
        """Aerospace Grade Titanium Ti-6Al-4V Grade 5."""
        return cls(
            name="Titanium Ti-6Al-4V",
            E=113.8e9,
            sigma_y=880e6,
            sigma_ut=950e6,
            sigma_f_prime=1600e6,
            b=-0.088,
            endurance_limit=450e6,
            cut_off_cycles=1e12,
        )

    @classmethod
    def structural_steel_s355(cls) -> "FatigueMaterial":
        """Structural Carbon Steel S355."""
        return cls(
            name="Structural Steel S355",
            E=210e9,
            sigma_y=355e6,
            sigma_ut=510e6,
            sigma_f_prime=950e6,
            b=-0.090,
            endurance_limit=220e6,
            cut_off_cycles=1e12,
        )


def extract_peaks_valleys(series: Sequence[float]) -> np.ndarray:
    """
    Filter a continuous stress time-history to strictly local reversals (peaks and valleys).
    
    Removes intermediate points along monotonic segments and removes
    consecutive identical values while preserving the endpoints.
    """
    s = np.asarray(series, dtype=np.float64)
    if len(s) < 3:
        return s.copy()

    # Step 1: Remove consecutive duplicates
    mask = np.ones(len(s), dtype=bool)
    mask[1:] = np.abs(np.diff(s)) > 1e-14
    s_dedup = s[mask]
    if len(s_dedup) < 3:
        return s_dedup

    # Step 2: Keep only points where slope changes sign
    d1 = s_dedup[1:-1] - s_dedup[:-2]
    d2 = s_dedup[2:] - s_dedup[1:-1]
    is_extrema = (d1 * d2) < 0.0

    extrema = [s_dedup[0]]
    for idx, is_ext in enumerate(is_extrema):
        if is_ext:
            extrema.append(s_dedup[idx + 1])
    extrema.append(s_dedup[-1])

    return np.array(extrema, dtype=np.float64)


def count_rainflow_cycles(
    series: Sequence[float],
    close_residuals: bool = False,
) -> List[RainflowCycle]:
    """
    Count stress cycles using the standard ASTM E1049-85 Section 5.4.4 algorithm.

    Parameters:
        series: Time history of stress (or already filtered peaks & valleys).
        close_residuals: If True, remaining unclosed half-cycles are treated as
                         closed cycles (repeating block assumption); if False,
                         they are counted as 0.5 cycles.

    Returns:
        cycles: List of RainflowCycle instances.
    """
    pv = extract_peaks_valleys(series)
    if len(pv) < 2:
        return []

    cycles: List[RainflowCycle] = []
    stack: List[float] = []

    # ASTM E1049-85 Section 5.4.4 Rainflow Counting:
    for pt in pv:
        stack.append(float(pt))
        while len(stack) >= 3:
            X = abs(stack[-1] - stack[-2])
            Y = abs(stack[-2] - stack[-3])

            if X < Y:
                break
            else:
                # X >= Y: range Y is counted
                rng = Y
                mean = 0.5 * (stack[-3] + stack[-2])
                if len(stack) == 3:
                    # Count Y as 0.5 cycle and discard the first point stack[0]
                    cycles.append(RainflowCycle(range=rng, mean=mean, count=0.5))
                    stack.pop(0)
                else:
                    # Count Y as 1.0 full cycle and delete stack[-3] and stack[-2]
                    cycles.append(RainflowCycle(range=rng, mean=mean, count=1.0))
                    del stack[-3:-1]

    # Process remaining unclosed points in stack
    mult = 1.0 if close_residuals else 0.5
    while len(stack) >= 2:
        rng = abs(stack[-1] - stack[-2])
        mean = 0.5 * (stack[-1] + stack[-2])
        if rng > 1e-12:
            cycles.append(RainflowCycle(range=rng, mean=mean, count=mult))
        stack.pop()

    return cycles


def evaluate_sn_life(
    amplitude: float,
    mean: float,
    material: FatigueMaterial,
    correction: str = "goodman",
) -> float:
    """
    Evaluate fatigue life N_f (number of cycles to failure) for a given stress
    amplitude and mean stress using S-N curves with mean stress correction.

    Parameters:
        amplitude: Alternating stress amplitude S_a (Pa).
        mean: Mean stress S_m (Pa).
        material: FatigueMaterial instance.
        correction: Mean stress correction model:
                    'goodman', 'gerber', 'morrow', 'soderberg', or 'none'.

    Returns:
        N_f: Number of cycles to failure (capped at material.cut_off_cycles).
    """
    S_a = max(float(amplitude), 0.0)
    S_m = float(mean)

    if S_a < 1e-12:
        return material.cut_off_cycles

    # 1. Mean stress correction to equivalent fully-reversed amplitude S_eq (R = -1)
    corr_lower = correction.lower()
    if corr_lower == "goodman":
        if S_m > 0.0:
            ratio = S_m / material.sigma_ut
            if ratio >= 1.0:
                return 1.0  # Static ultimate tensile failure
            S_eq = S_a / max(1.0 - ratio, 1e-6)
        else:
            # Tensile mean stress accelerates fatigue; compressive mean stress does not
            S_eq = S_a
    elif corr_lower == "gerber":
        if S_m > 0.0:
            ratio = (S_m / material.sigma_ut) ** 2
            if ratio >= 1.0:
                return 1.0
            S_eq = S_a / max(1.0 - ratio, 1e-6)
        else:
            S_eq = S_a
    elif corr_lower == "morrow":
        if S_m > 0.0:
            ratio = S_m / material.sigma_f_prime
            if ratio >= 1.0:
                return 1.0
            S_eq = S_a / max(1.0 - ratio, 1e-6)
        else:
            S_eq = S_a
    elif corr_lower == "soderberg":
        if S_m > 0.0:
            ratio = S_m / material.sigma_y
            if ratio >= 1.0:
                return 1.0
            S_eq = S_a / max(1.0 - ratio, 1e-6)
        else:
            S_eq = S_a
    elif corr_lower == "none":
        S_eq = S_a
    else:
        raise ValueError(f"Unknown mean stress correction: '{correction}'")

    # 2. Check endurance limit
    if S_eq <= material.endurance_limit:
        return material.cut_off_cycles

    # 3. Basquin power-law relationship: S_eq = sigma_f' * (2 * N_f)^b
    # ==> 2 * N_f = (S_eq / sigma_f')^(1 / b)
    # ==> N_f = 0.5 * (S_eq / sigma_f')^(1 / b)
    exponent = 1.0 / material.b
    ratio = S_eq / material.sigma_f_prime
    if ratio <= 0.0:
        return material.cut_off_cycles

    N_f = 0.5 * (ratio ** exponent)

    if N_f > material.cut_off_cycles:
        return material.cut_off_cycles
    if N_f < 1.0:
        return 1.0

    return float(N_f)


def evaluate_palmgren_miner_damage(
    cycles: Sequence[RainflowCycle],
    material: FatigueMaterial,
    correction: str = "goodman",
) -> Tuple[float, float]:
    """
    Evaluate cumulative fatigue damage D using the linear Palmgren-Miner rule:
        D = sum_i (n_i / N_i)

    Parameters:
        cycles: Sequence of RainflowCycle objects.
        material: FatigueMaterial instance.
        correction: Mean stress correction model.

    Returns:
        damage: Total cumulative damage D (D >= 1.0 indicates fatigue failure).
        cycles_to_failure: Estimated total cycles to failure under repetitions
                           of this load history.
    """
    total_damage = 0.0
    total_cycles = 0.0

    for cyc in cycles:
        total_cycles += cyc.count
        if cyc.range < 1e-12:
            continue
        n_f = evaluate_sn_life(cyc.amplitude, cyc.mean, material, correction=correction)
        if n_f < material.cut_off_cycles and n_f > 0.0:
            total_damage += cyc.count / n_f

    if total_damage <= 0.0 or total_damage < 1.0 / material.cut_off_cycles:
        cycles_to_failure = material.cut_off_cycles
    else:
        # Number of repetitions of this block until failure is 1 / D
        blocks_to_failure = 1.0 / total_damage
        cycles_to_failure = blocks_to_failure * total_cycles

    return float(total_damage), float(cycles_to_failure)


def evaluate_element_fatigue_life(
    stress_histories: np.ndarray,
    material: FatigueMaterial,
    method: str = "signed_von_mises",
    correction: str = "goodman",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate element-wise fatigue damage and fatigue life across an FEA mesh.

    Parameters:
        stress_histories: Array of shape (N_elements, N_steps) representing
                          scalar stress histories, OR shape (N_elements, N_steps, 6)
                          for Voigt stress tensors [sxx, syy, szz, sxy, syz, szx].
        material: FatigueMaterial property set.
        method: Scalar conversion method if stress_histories is 3D:
                'signed_von_mises', 'max_principal', or 'tresca'.
        correction: Mean stress correction ('goodman', 'gerber', 'morrow', 'soderberg', 'none').

    Returns:
        damage: (N_elements,) cumulative Palmgren-Miner damage array.
        fatigue_life: (N_elements,) estimated cycles to failure (capped at cut_off_cycles).
    """
    stress_arr = np.asarray(stress_histories, dtype=np.float64)

    if stress_arr.ndim == 3 and stress_arr.shape[2] == 6:
        n_elem, n_steps, _ = stress_arr.shape
        scalar_hist = np.zeros((n_elem, n_steps), dtype=np.float64)

        sxx = stress_arr[:, :, 0]
        syy = stress_arr[:, :, 1]
        szz = stress_arr[:, :, 2]
        sxy = stress_arr[:, :, 3]
        syz = stress_arr[:, :, 4]
        szx = stress_arr[:, :, 5]

        # Hydrostatic stress trace
        p_h = (sxx + syy + szz) / 3.0

        # Deviatoric stresses
        dev_xx = sxx - p_h
        dev_yy = syy - p_h
        dev_zz = szz - p_h

        # J2 invariant & von Mises
        j2 = 0.5 * (dev_xx**2 + dev_yy**2 + dev_zz**2) + (sxy**2 + syz**2 + szx**2)
        von_mises = np.sqrt(np.maximum(3.0 * j2, 0.0))

        if method.lower() == "signed_von_mises":
            # Signed von Mises: sign comes from hydrostatic stress trace
            sign = np.where(p_h >= 0.0, 1.0, -1.0)
            scalar_hist = sign * von_mises
        elif method.lower() == "tresca":
            # Approximation or direct Tresca
            scalar_hist = von_mises * (2.0 / np.sqrt(3.0))
        elif method.lower() == "max_principal":
            # Principal stress via characteristic equation roots
            for e in range(n_elem):
                for t in range(n_steps):
                    t_tensor = np.array([
                        [sxx[e, t], sxy[e, t], szx[e, t]],
                        [sxy[e, t], syy[e, t], syz[e, t]],
                        [szx[e, t], syz[e, t], szz[e, t]],
                    ])
                    eigvals = np.linalg.eigvalsh(t_tensor)
                    scalar_hist[e, t] = eigvals[2]  # Max eigenvalue
        else:
            scalar_hist = von_mises
    elif stress_arr.ndim == 2:
        scalar_hist = stress_arr
    else:
        raise ValueError(f"Unsupported stress_histories shape: {stress_arr.shape}")

    n_elements = scalar_hist.shape[0]
    damage = np.zeros(n_elements, dtype=np.float64)
    fatigue_life = np.zeros(n_elements, dtype=np.float64)

    for e in range(n_elements):
        cycles = count_rainflow_cycles(scalar_hist[e], close_residuals=False)
        d_e, n_f = evaluate_palmgren_miner_damage(cycles, material, correction=correction)
        damage[e] = d_e
        fatigue_life[e] = n_f

    return damage, fatigue_life
