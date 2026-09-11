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
    FiveAxisMachinabilityOptimizer,
    FiveAxisSetupResult,
    apply_3axis_milling_filter,
)

from .stress_opt import (
    StressConstrainedTopologyOptimizer,
    StressConstraintConfig,
    StressOptResult,
    StressAggregationType,
)

from .multi_material import (
    MaterialProperty,
    MultiMaterialConfig,
    MultiMaterialResult,
    MultiMaterialSIMPInterpolator,
    MultiMaterialTopologyOptimizer,
    project_simplex_batch,
    ALUMINUM_6061,
    TITANIUM_TI6AL4V,
    STEEL_STRUCTURAL,
    CARBON_PEEK,
)

from .am_overhang import (
    AMOverhangFilter,
    AMFilterResult,
)

from .modal_opt import (
    FrequencyConstrainedConfig,
    FrequencyOptResult,
    ModalSensitivityEvaluator,
    FrequencyConstrainedOptimizer,
)

from .tpms_lattice import (
    TPMSType,
    TPMSConfig,
    evaluate_tpms_levelset,
    density_to_levelset_threshold,
    evaluate_graded_tpms_field,
    generate_tpms_voxel_infill,
    extract_tpms_isosurface,
)

__all__ = [
    "TopologyOptimizer",
    "TopologyConfig",
    "OptimizationResult",
    "SensitivityFilter",
    "HeavisideProjection",
    "CNCMillingConstraint",
    "FiveAxisMachinabilityOptimizer",
    "FiveAxisSetupResult",
    "apply_3axis_milling_filter",
    "StressConstrainedTopologyOptimizer",
    "StressConstraintConfig",
    "StressOptResult",
    "StressAggregationType",
    "MaterialProperty",
    "MultiMaterialConfig",
    "MultiMaterialResult",
    "MultiMaterialSIMPInterpolator",
    "MultiMaterialTopologyOptimizer",
    "project_simplex_batch",
    "ALUMINUM_6061",
    "TITANIUM_TI6AL4V",
    "STEEL_STRUCTURAL",
    "CARBON_PEEK",
    "AMOverhangFilter",
    "AMFilterResult",
    "FrequencyConstrainedConfig",
    "FrequencyOptResult",
    "ModalSensitivityEvaluator",
    "FrequencyConstrainedOptimizer",
    "TPMSType",
    "TPMSConfig",
    "evaluate_tpms_levelset",
    "density_to_levelset_threshold",
    "evaluate_graded_tpms_field",
    "generate_tpms_voxel_infill",
    "extract_tpms_isosurface",
]

