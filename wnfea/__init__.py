# WNFEA — 3D Beam Finite Element Analysis Package
#
# Top-level convenience imports so that users can write:
#
#   from wnfea import FEAModel, STEPParser, BeamMesher, BeamSolver, PostProcessor

from .model import FEAModel, PipelineStage, PropertyAssignment
from .geometry.step_parser import STEPParser, STEPParseError
from .geometry.primitives import Point3D, GeometryNode, GeometryEdge, GeometryFace
from .properties.materials import MaterialDef, get_preset_material, create_custom_material, MATERIAL_PRESETS
from .properties.sections import (
    SectionDef, create_hollow_tube, create_solid_circle,
    create_square_tube, create_rectangular_tube, create_i_beam,
    create_rectangular_bar, create_general_section,
)
from .boundary.conditions import (
    SupportDef, LoadDef, DOFConstraint, DOFType,
    create_fixed_support,
)
from .boundary.coupling import RigidCoupling
from .mesh.beam_mesher import BeamMesher, MeshError
from .solver.beam_solver import BeamSolver, SolverError
from .solver.linear_static import solve_linear_static
from .solver.nonlinear_solver import solve_nonlinear_jfnk, NonLinearConvergenceError
from .solver.jfnk_operator import MatrixFreeJFNKOperator
from .solver.amg_preconditioner import BlockBeamAMGPreconditioner
from .solver.dof_manager import DOFManager
from .solver.mixed_precision import PrecisionConfig, MIXED_FP32, TRI_PRECISION, FULL_FP64
from .results.post_processor import PostProcessor, PostProcessError
from .results.result_data import StressPoint, ElementResult, ModelResults
from .elements import (
    element_stiffness_c3d10,
    element_internal_forces_c3d10,
    element_stresses_c3d10,
    b_matrix_c3d10,
    shape_functions_c3d10,
    jacobian_c3d10,
)

__all__ = [
    # Model
    "FEAModel", "PipelineStage", "PropertyAssignment",
    # Geometry
    "STEPParser", "STEPParseError",
    "Point3D", "GeometryNode", "GeometryEdge", "GeometryFace",
    # Properties
    "MaterialDef", "get_preset_material", "create_custom_material", "MATERIAL_PRESETS",
    "SectionDef", "create_hollow_tube", "create_solid_circle",
    "create_square_tube", "create_rectangular_tube", "create_i_beam",
    "create_rectangular_bar", "create_general_section",
    # Boundary & Couplings
    "SupportDef", "LoadDef", "DOFConstraint", "DOFType",
    "create_fixed_support", "RigidCoupling",
    # Mesh
    "BeamMesher", "MeshError",
    # Elements
    "element_stiffness_c3d10", "element_internal_forces_c3d10", "element_stresses_c3d10",
    "b_matrix_c3d10", "shape_functions_c3d10", "jacobian_c3d10",
    # Solver
    "BeamSolver", "SolverError", "solve_linear_static",
    "solve_nonlinear_jfnk", "NonLinearConvergenceError",
    "MatrixFreeJFNKOperator", "BlockBeamAMGPreconditioner",
    "DOFManager",
    "PrecisionConfig", "MIXED_FP32", "TRI_PRECISION", "FULL_FP64",
    # Results
    "PostProcessor", "PostProcessError",
    "StressPoint", "ElementResult", "ModelResults",
]


