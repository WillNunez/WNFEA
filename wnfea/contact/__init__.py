"""
WNFEA Contact Mechanics & Multi-Body Non-Penetration Subpackage.
---------------------------------------------------------------
Provides spatial hash surface contact pair detection, signed gap functions,
and matrix-free Augmented Lagrangian solvers for structural assemblies.
"""

from .contact_detector import (
    SurfaceFacet,
    ContactPair,
    compute_mean_value_coordinates_quad,
    extract_hex8_surface_facets,
    SpatialHashContactDetector,
)

from .contact_solver import (
    ContactSolveResult,
    MatrixFreeContactTangentOperator,
    solve_contact_assembly,
)

__all__ = [
    "SurfaceFacet",
    "ContactPair",
    "compute_mean_value_coordinates_quad",
    "extract_hex8_surface_facets",
    "SpatialHashContactDetector",
    "ContactSolveResult",
    "MatrixFreeContactTangentOperator",
    "solve_contact_assembly",
]
