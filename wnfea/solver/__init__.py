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

from .subdomain_solver import (
    IsolatedSubdomain,
    SubdomainSolveResult,
    extract_isolated_subdomain,
    solve_isolated_subdomain,
)

from .adaptive_solve_loop import (
    AdaptiveSubdomainResult,
    run_adaptive_voxel_amr_pipeline,
)

from .outofcore_streaming import (
    ChunkedStreamingConfig,
    ChunkedStreamingTelemetry,
    ChunkedStreamingMatrixFreeOperator,
)

from .hierarchical_warmstart import (
    HierarchicalWarmStartResult,
    HierarchicalCoarseMeshWarmStart,
    project_voxel_to_fine_mesh,
    project_amr_to_fine_mesh,
    compute_hierarchical_warmstart,
)

from .transient_implicit import (
    TransientHistory,
    MatrixFreeEffectiveDynamicOperator,
    solve_pcg_transient,
    solve_transient_implicit,
)

from .buckling_analysis import (
    BucklingResult,
    MatrixFreeGeometricStiffnessOperator,
    solve_linear_buckling,
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
    "IsolatedSubdomain",
    "SubdomainSolveResult",
    "extract_isolated_subdomain",
    "solve_isolated_subdomain",
    "AdaptiveSubdomainResult",
    "run_adaptive_voxel_amr_pipeline",
    "ChunkedStreamingConfig",
    "ChunkedStreamingTelemetry",
    "ChunkedStreamingMatrixFreeOperator",
    "HierarchicalWarmStartResult",
    "HierarchicalCoarseMeshWarmStart",
    "project_voxel_to_fine_mesh",
    "project_amr_to_fine_mesh",
    "compute_hierarchical_warmstart",
    "TransientHistory",
    "MatrixFreeEffectiveDynamicOperator",
    "solve_pcg_transient",
    "solve_transient_implicit",
    "BucklingResult",
    "MatrixFreeGeometricStiffnessOperator",
    "solve_linear_buckling",
]

