"""
Spatial Hash Contact Pair Detector & Surface Penetration Engine.
---------------------------------------------------------------
Extracts external boundary facets from 3D continuum meshes and detects candidate
slave-node to master-facet contact pairs using spatial acceleration trees.

Key Capabilities:
1. Exact boundary facet extraction for Hex8 (and tetrahedral) elements.
2. Outward-pointing surface normal vector calculation.
3. Fast spatial query via scipy.spatial.cKDTree in O(N log N) / O(N).
4. Analytical orthogonal projection of slave nodes onto master polygon facets.
5. Exact generalized barycentric (Mean Value Coordinates) master weight calculation
   ensuring C^0 displacement continuity and exact kinematic contact constraints.
6. Signed normal gap function:
   g_n = (x_s - x_m) . n_m
   g_n < 0 denotes penetration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
from scipy.spatial import cKDTree


@dataclass
class SurfaceFacet:
    """
    Planar or bilinear boundary surface facet of a 3D solid element.
    """
    facet_id: int
    node_ids: np.ndarray        # (4,) for Hex8 quad face, (3,) for Tet tri face
    centroid: np.ndarray        # (3,) spatial centroid
    normal: np.ndarray          # (3,) outward unit normal
    area: float                 # Surface area in m^2
    element_id: int             # Parent solid element index


@dataclass
class ContactPair:
    """
    Kinematic contact interaction between a slave node and a master facet.
    """
    slave_node_id: int          # Global ID of slave node
    master_facet_id: int        # Index of master facet
    master_node_ids: np.ndarray # Global IDs of master facet nodes
    master_weights: np.ndarray  # Generalized barycentric weights (sum to 1.0)
    normal: np.ndarray          # (3,) Master outward unit normal
    gap_n: float                # Signed normal distance (g_n < 0 means penetration)
    slave_coords: np.ndarray    # (3,) Coordinates of slave node
    projected_coords: np.ndarray# (3,) Coordinates of contact projection on master facet


def compute_mean_value_coordinates_quad(vertices: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Compute Mean Value Coordinates (generalized barycentric coordinates) of point p
    with respect to a 4-node convex planar or slightly non-planar quadrilateral.

    Parameters:
        vertices: (4, 3) 3D coordinates of quadrilateral corners ordered counter-clockwise.
        p: (3,) 3D coordinates of query point lying approximately in the quad plane.

    Returns:
        weights: (4,) float64 interpolation weights satisfying sum(w) = 1 and sum(w_i * v_i) = p.
    """
    n = 4
    s = [vertices[i] - p for i in range(n)]
    r = [np.linalg.norm(si) for si in s]

    # Exact vertex coincidence check
    for i in range(n):
        if r[i] < 1e-12:
            w = np.zeros(n, dtype=np.float64)
            w[i] = 1.0
            return w

    weights = np.zeros(n, dtype=np.float64)
    for i in range(n):
        im1 = (i - 1) % n
        ip1 = (i + 1) % n

        cross_m = np.linalg.norm(np.cross(s[im1], s[i]))
        dot_m = np.dot(s[im1], s[i])
        denom_m = r[im1] * r[i] + dot_m
        tan_half_m = cross_m / max(denom_m, 1e-15)

        cross_p = np.linalg.norm(np.cross(s[i], s[ip1]))
        dot_p = np.dot(s[i], s[ip1])
        denom_p = r[i] * r[ip1] + dot_p
        tan_half_p = cross_p / max(denom_p, 1e-15)

        weights[i] = (tan_half_m + tan_half_p) / r[i]

    total_w = np.sum(weights)
    if total_w > 1e-15:
        return weights / total_w
    return np.full(n, 0.25, dtype=np.float64)


def extract_hex8_surface_facets(
    nodes: np.ndarray,
    elements: np.ndarray,
) -> List[SurfaceFacet]:
    """
    Extract all external boundary quadrilateral facets from a Hex8 mesh.
    Computes outward-directed unit normals and surface areas.

    Parameters:
        nodes: (N_nodes, 3) float64 nodal coordinates.
        elements: (N_elements, 8) int64 Hex8 element connectivity.

    Returns:
        List of SurfaceFacet objects representing the external surface boundary.
    """
    # 6 local quad faces for standard Hex8:
    # 0: -Z (0, 3, 2, 1)
    # 1: +Z (4, 5, 6, 7)
    # 2: -Y (0, 1, 5, 4)
    # 3: +X (1, 2, 6, 5)
    # 4: +Y (2, 3, 7, 6)
    # 5: -X (3, 0, 4, 7)
    hex_faces_local = np.array([
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ], dtype=np.int64)

    face_dict = {}  # sorted_node_tuple -> (elem_idx, local_face_nodes)

    for e_idx, elem in enumerate(elements):
        for f_local in hex_faces_local:
            f_nodes = elem[f_local]
            key = tuple(sorted(f_nodes))
            if key in face_dict:
                # Shared internal face: remove from boundary
                del face_dict[key]
            else:
                face_dict[key] = (e_idx, f_nodes)

    facets = []
    facet_id = 0

    for key, (e_idx, f_nodes) in face_dict.items():
        v = nodes[f_nodes]  # (4, 3)
        centroid = np.mean(v, axis=0)

        # Polygon normal via Newell's method or diagonal cross product:
        # n = (v2 - v0) x (v3 - v1)
        diag1 = v[2] - v[0]
        diag2 = v[3] - v[1]
        n_raw = np.cross(diag1, diag2)
        norm_len = np.linalg.norm(n_raw)
        if norm_len < 1e-15:
            continue
        normal = n_raw / norm_len

        # Verify outward direction relative to parent element centroid
        elem_nodes = nodes[elements[e_idx]]
        elem_centroid = np.mean(elem_nodes, axis=0)
        outward_dir = centroid - elem_centroid
        if np.dot(normal, outward_dir) < 0.0:
            normal = -normal
            f_nodes = f_nodes[::-1]
            v = nodes[f_nodes]

        # Area = 0.5 * |diag1 x diag2|
        area = 0.5 * norm_len

        facets.append(
            SurfaceFacet(
                facet_id=facet_id,
                node_ids=np.asarray(f_nodes, dtype=np.int64),
                centroid=centroid,
                normal=normal,
                area=area,
                element_id=e_idx,
            )
        )
        facet_id += 1

    return facets


def compute_nodal_surface_normals(
    nodes: np.ndarray,
    facets: List[SurfaceFacet],
) -> np.ndarray:
    """
    Compute outward unit normal vectors at each surface node by area-weighted averaging
    of incident surface facets.

    Parameters:
        nodes: (N_nodes, 3) nodal coordinates.
        facets: List of SurfaceFacet boundary facets.

    Returns:
        normals: (N_nodes, 3) unit outward normal vectors.
    """
    normals = np.zeros_like(nodes)
    for f in facets:
        for nid in f.node_ids:
            normals[nid] += f.normal * f.area

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    mask = norms[:, 0] > 1e-12
    normals[mask] /= norms[mask]
    return normals


class SpatialHashContactDetector:
    """
    Accelerated spatial contact pair detector between slave body nodes and master body facets.
    """

    def __init__(
        self,
        master_nodes: np.ndarray,
        master_facets: List[SurfaceFacet],
    ):
        """
        Parameters:
            master_nodes: (N_master_nodes, 3) nodal coordinates of master body.
            master_facets: List of SurfaceFacet objects defining master boundary.
        """
        self.master_nodes = master_nodes
        self.master_facets = master_facets
        self.num_facets = len(master_facets)

        # Build facet centroids and bounding radii for KDTree
        if self.num_facets > 0:
            centroids = np.array([f.centroid for f in master_facets], dtype=np.float64)
            self.kdtree = cKDTree(centroids)

            # Precalculate maximum radius of each facet from centroid
            self.radii = np.zeros(self.num_facets, dtype=np.float64)
            for i, f in enumerate(master_facets):
                v = master_nodes[f.node_ids]
                self.radii[i] = np.max(np.linalg.norm(v - f.centroid, axis=1))
        else:
            self.kdtree = None
            self.radii = np.zeros(0, dtype=np.float64)

    def detect_contact_pairs(
        self,
        slave_nodes: np.ndarray,
        search_distance: float = 0.005,
        slave_node_indices: Optional[Sequence[int]] = None,
        slave_normals: Optional[np.ndarray] = None,
        max_penetration: float = 0.05,
    ) -> List[ContactPair]:
        """
        Detect all slave nodes in contact or proximity with master facets.

        Parameters:
            slave_nodes: (N_slave_nodes, 3) coordinates of slave body nodes.
            search_distance: Maximum positive separation distance to consider for contact.
            slave_node_indices: Optional subset of slave node indices (e.g. boundary skin nodes).
            slave_normals: Optional (N_slave_nodes, 3) surface normals for opposing-face filtering.
            max_penetration: Maximum allowed negative gap before ignoring spurious penetrations.

        Returns:
            List of ContactPair objects.
        """
        if self.kdtree is None or len(slave_nodes) == 0:
            return []

        active_slave_ids = (
            slave_node_indices
            if slave_node_indices is not None
            else range(len(slave_nodes))
        )

        contact_pairs = []
        max_facet_radius = np.max(self.radii) if len(self.radii) > 0 else 0.0
        query_radius = search_distance + max_facet_radius + max_penetration

        for s_idx in active_slave_ids:
            x_s = slave_nodes[s_idx]

            # Query nearby candidate facets
            candidate_indices = self.kdtree.query_ball_point(x_s, r=query_radius)
            if not candidate_indices:
                continue

            best_pair = None
            best_abs_gap = np.inf

            for f_idx in candidate_indices:
                facet = self.master_facets[f_idx]
                n = facet.normal

                # If slave normals provided, verify opposing surfaces: n_s . n_m < -0.2
                if slave_normals is not None:
                    n_s = slave_normals[s_idx]
                    if np.dot(n_s, n) > -0.2:
                        continue

                # Normal signed distance from facet centroid plane
                gap_raw = np.dot(x_s - facet.centroid, n)

                # Check search bounds: -max_penetration <= gap_raw <= search_distance
                if gap_raw < -max_penetration or gap_raw > search_distance:
                    continue

                # Projected point onto the facet plane
                x_proj = x_s - gap_raw * n

                # Check if x_proj lies inside the quadrilateral
                v = self.master_nodes[facet.node_ids]  # (4, 3)
                is_inside = True
                for edge_idx in range(4):
                    p0 = v[edge_idx]
                    p1 = v[(edge_idx + 1) % 4]
                    edge_vec = p1 - p0
                    to_proj = x_proj - p0
                    cross_n = np.dot(np.cross(edge_vec, to_proj), n)
                    # Slight tolerance for boundary edges
                    if cross_n < -1e-8 * facet.area:
                        is_inside = False
                        break

                if not is_inside:
                    continue

                # Calculate generalized barycentric weights
                weights = compute_mean_value_coordinates_quad(v, x_proj)

                # If multiple overlapping facets, pick the closest one
                if abs(gap_raw) < best_abs_gap:
                    best_abs_gap = abs(gap_raw)
                    best_pair = ContactPair(
                        slave_node_id=int(s_idx),
                        master_facet_id=f_idx,
                        master_node_ids=facet.node_ids.copy(),
                        master_weights=weights,
                        normal=n.copy(),
                        gap_n=float(gap_raw),
                        slave_coords=x_s.copy(),
                        projected_coords=x_proj.copy(),
                    )

            if best_pair is not None:
                contact_pairs.append(best_pair)

        return contact_pairs
