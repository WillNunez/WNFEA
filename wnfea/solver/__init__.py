# Solver subpackage
from .beam_solver import BeamSolver, SolverError
from .linear_static import solve_linear_static
from .assembler import solve_linear_system
from .nonlinear_solver import solve_nonlinear_jfnk, NonLinearConvergenceError
from .jfnk_operator import MatrixFreeJFNKOperator
from .amg_preconditioner import BlockBeamAMGPreconditioner
from .mixed_precision import PrecisionConfig, MIXED_FP32, TRI_PRECISION, FULL_FP64

from .thermal_solver import (
    compute_hex8_thermal_reference,
    MatrixFreeHex8ThermalOperator,
    ThermalConvectionBC,
    ThermoMechanicalResult,
    solve_steady_state_thermal,
    solve_transient_thermal,
    compute_thermal_load_vector,
    compute_thermo_mechanical_stresses,
    solve_thermo_mechanical,
)

__all__ = [
    "BeamSolver",
    "SolverError",
    "solve_linear_static",
    "solve_linear_system",
    "solve_nonlinear_jfnk",
    "NonLinearConvergenceError",
    "MatrixFreeJFNKOperator",
    "BlockBeamAMGPreconditioner",
    "PrecisionConfig",
    "MIXED_FP32",
    "TRI_PRECISION",
    "FULL_FP64",
    "compute_hex8_thermal_reference",
    "MatrixFreeHex8ThermalOperator",
    "ThermalConvectionBC",
    "ThermoMechanicalResult",
    "solve_steady_state_thermal",
    "solve_transient_thermal",
    "compute_thermal_load_vector",
    "compute_thermo_mechanical_stresses",
    "solve_thermo_mechanical",
]

