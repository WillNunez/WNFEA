"""
WNFEA Topology Optimization & Generative Design Package.
--------------------------------------------------------
Modules:
- topology: Core matrix-free SIMP topology optimization engine with Optimality Criteria (OC).
- filters: Spatial sensitivity convolution filters and smoothed Heaviside projection.
- machinability: 3-axis CNC milling accessibility and undercut constraint penalties.
"""

from .topology import (
    TopologyOptimizer,
    TopologyConfig,
    OptimizationResult,
)
from .filters import (
    SensitivityFilter,
    HeavisideProjection,
)
from .machinability import (
    CNCMillingConstraint,
    apply_3axis_milling_filter,
)

__all__ = [
    "TopologyOptimizer",
    "TopologyConfig",
    "OptimizationResult",
    "SensitivityFilter",
    "HeavisideProjection",
    "CNCMillingConstraint",
    "apply_3axis_milling_filter",
]
