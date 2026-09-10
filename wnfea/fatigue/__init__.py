"""
WNFEA Aero-Structural Fatigue Life & Cyclic Damage Estimation Engine
--------------------------------------------------------------------
Provides ASTM E1049-85 Rainflow cycle counting, Basquin & Wöhler S-N fatigue curves,
mean stress corrections (Goodman, Gerber, Morrow, Soderberg), Palmgren-Miner
cumulative damage summation, multiaxial critical plane analysis, and Dang Van
fatigue limit criteria.
"""

from .fatigue_solver import (
    RainflowCycle,
    FatigueMaterial,
    extract_peaks_valleys,
    count_rainflow_cycles,
    evaluate_sn_life,
    evaluate_palmgren_miner_damage,
    evaluate_element_fatigue_life,
)
from .critical_plane import (
    CriticalPlaneResult,
    evaluate_critical_plane,
    evaluate_dang_van_safety_factor,
)

__all__ = [
    "RainflowCycle",
    "FatigueMaterial",
    "extract_peaks_valleys",
    "count_rainflow_cycles",
    "evaluate_sn_life",
    "evaluate_palmgren_miner_damage",
    "evaluate_element_fatigue_life",
    "CriticalPlaneResult",
    "evaluate_critical_plane",
    "evaluate_dang_van_safety_factor",
]
