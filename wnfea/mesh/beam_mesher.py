"""
Beam mesher for WNFEA.

Subdivides geometry edges into 1-D beam finite elements. Each geometry edge
is divided into ``n_divisions`` equal-length segments. Nodes at geometry
vertices are shared automatically so that the connectivity is correct.

The mesher populates:
    - model.mesh_nodes        (N×3 float array)
    - model.mesh_elements     (E×2 int array, indices into mesh_nodes)
    - model.element_properties (element_id → PropertyAssignment)
    - model.geometry_to_mesh_node_map (geometry_node_id → mesh_node_id)
"""

from __future__ import annotations

import numpy as np

from ..model import FEAModel


class MeshError(Exception):
    """Raised when meshing fails."""
    pass


class BeamMesher:
    """
    Generates a finite element mesh from geometry edges.

    Parameters
    ----------
    n_divisions : int
        Number of beam elements per geometry edge (default 1).
    """

    def __init__(self, n_divisions: int = 1):
        if n_divisions < 1:
            raise ValueError(f"n_divisions must be >= 1, got {n_divisions}")
        self.n_divisions = n_divisions

    def mesh(self, model: FEAModel) -> None:
        """
        Generate the mesh and store it on *model*.

        Raises
        ------
        MeshError
            If the model fails validation for meshing.
        """
        errors = model.validate_for_meshing()
        if errors:
            raise MeshError(
                "Cannot mesh – validation errors:\n  " + "\n  ".join(errors)
            )

        # Clear any prior mesh data
        model.clear_mesh()

        # -----------------------------------------------------------
        # Phase 1 – collect unique mesh node coordinates.
        # Geometry vertices become the first mesh nodes. Interior
        # nodes (from subdivision) are appended afterwards.
        # -----------------------------------------------------------

        # Tolerance for merging coincident nodes (metres)
        merge_tol = 1e-9

        # Map: geometry_node_id → mesh_node_id
        geo_to_mesh: dict[int, int] = {}
        # List built incrementally; index == mesh_node_id
        node_coords: list[np.ndarray] = []

        # Pre-register all geometry nodes so they get low indices
        for gid in sorted(model.geometry_nodes.keys()):
            gnode = model.geometry_nodes[gid]
            coord = gnode.point.to_array()
            mesh_id = len(node_coords)
            node_coords.append(coord)
            geo_to_mesh[gid] = mesh_id

        # -----------------------------------------------------------
        # Phase 2 – subdivide each edge and create elements.
        # -----------------------------------------------------------
        elements: list[tuple[int, int]] = []
        elem_props: dict[int, object] = {}

        for edge_id in sorted(model.geometry_edges.keys()):
            edge = model.geometry_edges[edge_id]
            assignment = model.edge_assignments[edge_id]

            start_gid = edge.start_node_id
            end_gid = edge.end_node_id

            start_coord = model.geometry_nodes[start_gid].point.to_array()
            end_coord = model.geometry_nodes[end_gid].point.to_array()

            # Mesh node ids along this edge (start, interior*, end)
            edge_mesh_nodes: list[int] = [geo_to_mesh[start_gid]]

            if self.n_divisions > 1:
                for k in range(1, self.n_divisions):
                    t = k / self.n_divisions
                    interior = start_coord + t * (end_coord - start_coord)

                    # Check for coincident node (unlikely but safe)
                    found = False
                    for existing_id, existing_coord in enumerate(node_coords):
                        if np.linalg.norm(interior - existing_coord) < merge_tol:
                            edge_mesh_nodes.append(existing_id)
                            found = True
                            break
                    if not found:
                        new_id = len(node_coords)
                        node_coords.append(interior)
                        edge_mesh_nodes.append(new_id)

            edge_mesh_nodes.append(geo_to_mesh[end_gid])

            # Create beam elements from consecutive node pairs
            for seg in range(self.n_divisions):
                elem_id = len(elements)
                elements.append((edge_mesh_nodes[seg], edge_mesh_nodes[seg + 1]))
                elem_props[elem_id] = assignment

        # -----------------------------------------------------------
        # Phase 3 – store everything on the model.
        # -----------------------------------------------------------
        model.mesh_nodes = np.array(node_coords, dtype=float)
        model.mesh_elements = np.array(elements, dtype=int)
        model.element_properties = elem_props
        model.geometry_to_mesh_node_map = dict(geo_to_mesh)

        # Clear any stale results
        model.clear_results()
