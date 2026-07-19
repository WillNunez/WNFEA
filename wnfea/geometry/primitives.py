"""
Geometry primitive types for the WNFEA model.

These are the internal representation of geometry imported from STEP files.
Edges are used for beam element generation. Faces are stored for visualization only.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Point3D:
    """A point in 3D space."""
    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    def distance_to(self, other: 'Point3D') -> float:
        return float(np.linalg.norm(self.to_array() - other.to_array()))

    def __repr__(self):
        return f"Point3D({self.x:.6f}, {self.y:.6f}, {self.z:.6f})"


@dataclass
class GeometryNode:
    """A named node in the geometry model, corresponding to a STEP VERTEX_POINT."""
    id: int
    label: str
    point: Point3D

    def __repr__(self):
        return f"GeometryNode(id={self.id}, label='{self.label}', {self.point})"


@dataclass
class GeometryEdge:
    """
    A straight edge connecting two geometry nodes.
    Used for beam element generation during meshing.
    """
    id: int
    label: str
    start_node_id: int
    end_node_id: int

    def __repr__(self):
        return f"GeometryEdge(id={self.id}, label='{self.label}', {self.start_node_id} -> {self.end_node_id})"


@dataclass
class GeometryFace:
    """
    A surface/shell face defined by a list of bounding node IDs.
    Stored for VISUALIZATION ONLY — no FEA functionality is tied to faces.
    """
    id: int
    label: str
    bounding_node_ids: list[int] = field(default_factory=list)

    def __repr__(self):
        return f"GeometryFace(id={self.id}, label='{self.label}', nodes={self.bounding_node_ids})"
