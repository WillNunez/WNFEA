"""
Central FEA data model for WNFEA.

The FEAModel class is the single source of truth shared across all pipeline
stages. Every module (parser, mesher, solver, GUI panel) reads from and writes
to this model. Stage transitions validate that prerequisite data is present.

Pipeline stages:
    1. Geometry Import  → populates geometry_nodes, geometry_edges, geometry_faces
    2. Property Assign  → populates materials, sections, edge_assignments
    3. Meshing          → populates mesh_nodes, mesh_elements, element_properties
    4. Boundary Conds   → populates supports, loads
    5. Solve            → populates displacements, element_results
"""

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from .geometry.primitives import GeometryNode, GeometryEdge, GeometryFace
from .properties.materials import MaterialDef
from .properties.sections import SectionDef
from .boundary.conditions import SupportDef, LoadDef


class PipelineStage(Enum):
    """Tracks how far the model has progressed through the analysis pipeline."""
    EMPTY = auto()
    GEOMETRY_LOADED = auto()
    PROPERTIES_ASSIGNED = auto()
    MESHED = auto()
    BCS_DEFINED = auto()
    SOLVED = auto()


@dataclass
class PropertyAssignment:
    """Associates a material and cross-section with a geometry edge."""
    material_name: str
    section_name: str


@dataclass
class FEAModel:
    """
    Central data model for the finite element analysis pipeline.

    All pipeline stages read from and write to this single object.
    """

    # --- Stage 1: Geometry (populated by import) ---
    geometry_nodes: dict[int, GeometryNode] = field(default_factory=dict)
    geometry_edges: dict[int, GeometryEdge] = field(default_factory=dict)
    geometry_faces: dict[int, GeometryFace] = field(default_factory=dict)  # visualization only
    source_file: str = ""

    # --- Stage 2: Properties (populated by property assignment) ---
    materials: dict[str, MaterialDef] = field(default_factory=dict)
    sections: dict[str, SectionDef] = field(default_factory=dict)
    edge_assignments: dict[int, PropertyAssignment] = field(default_factory=dict)

    # --- Stage 3: Mesh (populated by mesher) ---
    mesh_nodes: np.ndarray | None = None       # shape (N, 3)
    mesh_elements: np.ndarray | None = None    # shape (E, 2)
    element_properties: dict[int, PropertyAssignment] = field(default_factory=dict)
    # Solid continuum elements (e.g. C3D10)
    solid_elements: np.ndarray | None = None   # shape (M, 10)
    solid_materials: dict[int, str] = field(default_factory=dict)  # elem_id -> material_name
    # Multi-point kinematic couplings (e.g. beam to solid)
    couplings: list = field(default_factory=list)
    # Maps geometry node IDs → mesh node IDs for BC propagation
    geometry_to_mesh_node_map: dict[int, int] = field(default_factory=dict)

    # --- Stage 4: Boundary Conditions ---
    supports: list[SupportDef] = field(default_factory=list)
    loads: list[LoadDef] = field(default_factory=list)

    # --- Stage 5: Results (populated by solver) ---
    displacements: np.ndarray | None = None    # shape (N*6,)
    element_results: dict | None = None
    reaction_forces: np.ndarray | None = None  # shape (N*6,)

    # --- Pipeline tracking ---
    @property
    def stage(self) -> PipelineStage:
        """Determine the current pipeline stage based on populated data."""
        if self.displacements is not None:
            return PipelineStage.SOLVED
        if self.supports or self.loads:
            return PipelineStage.BCS_DEFINED
        if self.mesh_nodes is not None and (self.mesh_elements is not None or self.solid_elements is not None):
            return PipelineStage.MESHED
        if self.edge_assignments:
            return PipelineStage.PROPERTIES_ASSIGNED
        if self.geometry_nodes:
            return PipelineStage.GEOMETRY_LOADED
        return PipelineStage.EMPTY

    # --- Stage validators ---
    def validate_for_properties(self) -> list[str]:
        """Check that geometry is loaded before property assignment."""
        errors = []
        if not self.geometry_nodes:
            errors.append("No geometry nodes loaded. Import a STEP file first.")
        if not self.geometry_edges:
            errors.append("No geometry edges loaded. Import a STEP file with edges.")
        return errors

    def validate_for_meshing(self) -> list[str]:
        """Check that properties are assigned to all edges before meshing."""
        errors = self.validate_for_properties()
        unassigned = [
            eid for eid in self.geometry_edges
            if eid not in self.edge_assignments
        ]
        if unassigned:
            errors.append(
                f"Edges without property assignments: {unassigned}. "
                "Assign material and section to all edges."
            )
        # Validate that referenced materials/sections exist
        for eid, assignment in self.edge_assignments.items():
            if assignment.material_name not in self.materials:
                errors.append(f"Edge {eid}: material '{assignment.material_name}' not defined.")
            if assignment.section_name not in self.sections:
                errors.append(f"Edge {eid}: section '{assignment.section_name}' not defined.")
        return errors

    def validate_for_solving(self) -> list[str]:
        """Check that mesh and BCs are defined before solving."""
        errors = []
        if self.mesh_nodes is None or (self.mesh_elements is None and self.solid_elements is None):
            errors.append("No mesh generated. Run the mesher first.")
        if not self.supports:
            errors.append("No supports defined. Add at least one boundary condition.")
        if not self.loads:
            errors.append("No loads defined. Add at least one load.")
        # Check that at least one support has a constraint
        if self.supports and not any(s.has_any_constraint for s in self.supports):
            errors.append("No support has any constrained DOFs.")
        # Check that at least one load is nonzero
        if self.loads and not any(l.has_any_load for l in self.loads):
            errors.append("All loads are zero. Add at least one nonzero load.")
        return errors

    # --- Mutation helpers ---
    def load_geometry(self, parse_result: dict):
        """
        Populate geometry from a STEP parser result dict.
        Clears all downstream data (properties, mesh, BCs, results).
        """
        self.geometry_nodes = parse_result.get('nodes', {})
        self.geometry_edges = parse_result.get('edges', {})
        self.geometry_faces = parse_result.get('faces', {})
        self.source_file = parse_result.get('source_file', '')

        # Clear downstream
        self.edge_assignments.clear()
        self.mesh_nodes = None
        self.mesh_elements = None
        self.element_properties.clear()
        self.geometry_to_mesh_node_map.clear()
        self.supports.clear()
        self.loads.clear()
        self.displacements = None
        self.element_results = None
        self.reaction_forces = None

    def clear_results(self):
        """Clear solver results (e.g., when mesh or BCs change)."""
        self.displacements = None
        self.element_results = None
        self.reaction_forces = None

    def clear_mesh(self):
        """Clear mesh and all downstream data."""
        self.mesh_nodes = None
        self.mesh_elements = None
        self.solid_elements = None
        self.solid_materials.clear()
        self.couplings.clear()
        self.element_properties.clear()
        self.geometry_to_mesh_node_map.clear()
        # Keep geometry-node BCs but clear mesh-node BCs
        self.supports = [s for s in self.supports if s.is_geometry_node]
        self.loads = [l for l in self.loads if l.is_geometry_node]
        self.clear_results()

    # --- Summary ---
    def summary(self) -> str:
        """Return a human-readable summary of the model state."""
        lines = [f"FEAModel -- Stage: {self.stage.name}"]
        if self.source_file:
            lines.append(f"  Source: {self.source_file}")
        lines.append(f"  Geometry: {len(self.geometry_nodes)} nodes, "
                      f"{len(self.geometry_edges)} edges, "
                      f"{len(self.geometry_faces)} faces (vis-only)")
        lines.append(f"  Materials: {len(self.materials)}, Sections: {len(self.sections)}")
        lines.append(f"  Assignments: {len(self.edge_assignments)} / {len(self.geometry_edges)} edges")
        if self.mesh_nodes is not None:
            n_beams = len(self.mesh_elements) if self.mesh_elements is not None else 0
            n_solids = len(self.solid_elements) if self.solid_elements is not None else 0
            lines.append(f"  Mesh: {len(self.mesh_nodes)} nodes, {n_beams} beam elements, {n_solids} solid elements")
        else:
            lines.append(f"  Mesh: not generated")
        lines.append(f"  BCs: {len(self.supports)} supports, {len(self.loads)} loads")
        if self.displacements is not None:
            lines.append(f"  Results: solved ({len(self.displacements)} DOFs)")
        else:
            lines.append(f"  Results: not solved")
        return "\n".join(lines)
