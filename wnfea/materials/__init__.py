"""
WNFEA Materials and Microstructure Homogenization Package.
----------------------------------------------------------
Modules:
- homogenization: Numerical asymptotic homogenization of 3D periodic cellular lattices
  (TPMS Gyroid, Schwarz P, Diamond, etc.) under periodic boundary conditions.
"""

from .homogenization import (
    HomogenizationResult,
    compute_isotropic_elasticity_matrix,
    solve_periodic_homogenization,
    homogenize_tpms_unit_cell,
)

__all__ = [
    "HomogenizationResult",
    "compute_isotropic_elasticity_matrix",
    "solve_periodic_homogenization",
    "homogenize_tpms_unit_cell",
]
