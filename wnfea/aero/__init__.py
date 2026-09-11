"""
WNFEA Aeroelasticity, Vortex Lattice Aerodynamics & Flutter Package.
---------------------------------------------------------------------
Modules:
- vortex_lattice: Quasi-steady 3D Vortex Lattice Method (VLM), Prandtl-Glauert compressibility,
  and conservative structural surface spline interpolation.
- flutter_solver: British PK-method aeroelastic flutter analysis, Theodorsen unsteady aerodynamics,
  damping-frequency root tracking, and static aeroelastic divergence.
"""

from .vortex_lattice import (
    AeroPanel,
    VortexLatticeMesh,
    SurfaceSplineCoupler,
)

from .flutter_solver import (
    AeroelasticState,
    FlutterResult,
    TheodorsenGAF,
    solve_pk_flutter,
    solve_static_divergence,
)

__all__ = [
    "AeroPanel",
    "VortexLatticeMesh",
    "SurfaceSplineCoupler",
    "AeroelasticState",
    "FlutterResult",
    "TheodorsenGAF",
    "solve_pk_flutter",
    "solve_static_divergence",
]
