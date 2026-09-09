"""
CAD-Conforming Boundary Snapping for Hierarchical Octree AMR Meshes.
-------------------------------------------------------------------
Implements geometry-respecting adaptive mesh refinement:
1. Analytical CAD Surface Representations:
   - Cylinders (holes, bosses, filleted cylinders).
   - Spheres (domes, ball joints).
   - Planes (flat mounting faces, datum pads).
   - General Signed Distance Fields (SDF) and Level-Set isosurfaces.
2. Independent Boundary Node Snapping:
   - Boundary independent nodes are projected onto exact CAD surfaces.
3. Hanging-Node Conforming Reconstruction:
   - Constrained/hanging nodes are reconstructed through C @ x_true,
     guaranteeing 100% exact inter-element C^0 continuity and zero gaps.
4. Tangling Prevention:
   - Verifies element Jacobian determinant det(J) > 0 at all 8 Gauss points,
     applying line-search step damping if an element approaches inversion.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence, List, Tuple, Callable
from dataclasses import dataclass
import numpy as np

from .octree_amr import AMRMesh


class CADSurface(ABC):
    """Abstract base class for analytical CAD boundary surfaces."""

    @abstractmethod
    def project(self, point: np.ndarray) -> np.ndarray:
        """Project a 3D point onto the analytical surface."""
        pass

    @abstractmethod
    def distance(self, point: np.ndarray) -> float:
        """Compute shortest signed/unsigned Euclidean distance to surface."""
        pass


@dataclass
class CylinderCADSurface(CADSurface):
    """
    Analytical infinite or bounded cylinder surface.
    Defined by axis point p0, unit axis vector axis_dir, and radius R.
    """
    axis_point: np.ndarray
    axis_dir: np.ndarray
    radius: float

    def __post_init__(self):
        self.axis_point = np.asarray(self.axis_point, dtype=np.float64)
        norm = np.linalg.norm(self.axis_dir)
        if norm < 1e-14:
            raise ValueError("axis_dir must be non-zero")
        self.axis_dir = np.asarray(self.axis_dir, dtype=np.float64) / norm
        self.radius = float(self.radius)

    def project(self, point: np.ndarray) -> np.ndarray:
        p = np.asarray(point, dtype=np.float64)
        d = p - self.axis_point
        d_par = np.dot(d, self.axis_dir) * self.axis_dir
        d_perp = d - d_par
        norm_perp = np.linalg.norm(d_perp)
        if norm_perp < 1e-12:
            # Point lies directly on the axis; arbitrary radial vector
            if abs(self.axis_dir[0]) < 0.9:
                radial = np.cross(self.axis_dir, np.array([1.0, 0.0, 0.0]))
            else:
                radial = np.cross(self.axis_dir, np.array([0.0, 1.0, 0.0]))
            radial = radial / np.linalg.norm(radial)
            return self.axis_point + d_par + self.radius * radial
        return self.axis_point + d_par + (self.radius / norm_perp) * d_perp

    def distance(self, point: np.ndarray) -> float:
        p = np.asarray(point, dtype=np.float64)
        d = p - self.axis_point
        d_par = np.dot(d, self.axis_dir) * self.axis_dir
        d_perp = d - d_par
        return abs(np.linalg.norm(d_perp) - self.radius)


@dataclass
class SphereCADSurface(CADSurface):
    """Analytical sphere surface defined by center point and radius R."""
    center: np.ndarray
    radius: float

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=np.float64)
        self.radius = float(self.radius)

    def project(self, point: np.ndarray) -> np.ndarray:
        p = np.asarray(point, dtype=np.float64)
        d = p - self.center
        norm_d = np.linalg.norm(d)
        if norm_d < 1e-12:
            return self.center + np.array([self.radius, 0.0, 0.0])
        return self.center + (self.radius / norm_d) * d

    def distance(self, point: np.ndarray) -> float:
        p = np.asarray(point, dtype=np.float64)
        return abs(np.linalg.norm(p - self.center) - self.radius)


@dataclass
class PlaneCADSurface(CADSurface):
    """Analytical infinite plane defined by point p0 and unit normal."""
    point: np.ndarray
    normal: np.ndarray

    def __post_init__(self):
        self.point = np.asarray(self.point, dtype=np.float64)
        n_norm = np.linalg.norm(self.normal)
        if n_norm < 1e-14:
            raise ValueError("normal must be non-zero")
        self.normal = np.asarray(self.normal, dtype=np.float64) / n_norm

    def project(self, point: np.ndarray) -> np.ndarray:
        p = np.asarray(point, dtype=np.float64)
        dist = np.dot(p - self.point, self.normal)
        return p - dist * self.normal

    def distance(self, point: np.ndarray) -> float:
        p = np.asarray(point, dtype=np.float64)
        return abs(np.dot(p - self.point, self.normal))


@dataclass
class SDFCADSurface(CADSurface):
    """
    Arbitrary CAD shape defined by a Signed Distance Field function Phi(x)
    where Phi(x) = 0 is the physical boundary surface.
    """
    sdf_func: Callable[[np.ndarray], float]
    grad_func: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def distance(self, point: np.ndarray) -> float:
        return abs(self.sdf_func(point))

    def project(self, point: np.ndarray, max_iters: int = 10, tol: float = 1e-7) -> np.ndarray:
        x = np.asarray(point, dtype=np.float64).copy()
        eps = 1e-6
        for _ in range(max_iters):
            val = self.sdf_func(x)
            if abs(val) < tol:
                break
            if self.grad_func is not None:
                grad = self.grad_func(x)
            else:
                # Finite-difference gradient
                g0 = (self.sdf_func(x + np.array([eps, 0.0, 0.0])) - val) / eps
                g1 = (self.sdf_func(x + np.array([0.0, eps, 0.0])) - val) / eps
                g2 = (self.sdf_func(x + np.array([0.0, 0.0, eps])) - val) / eps
                grad = np.array([g0, g1, g2])

            norm_g = np.linalg.norm(grad)
            if norm_g < 1e-12:
                break
            step = (val / norm_g**2) * grad
            x -= step
        return x


def compute_hex8_min_jacobian(nodes: np.ndarray, elements: np.ndarray) -> float:
    """
    Evaluate the minimum 2x2x2 Gauss-point Jacobian determinant det(J) across
    all Hex8 elements. If min det(J) > 0, the mesh has no inverted elements.
    """
    gp = 1.0 / np.sqrt(3.0)
    gauss_pts = [-gp, gp]
    xi_nodes = np.array([
        [-1.0, -1.0, -1.0], [ 1.0, -1.0, -1.0], [ 1.0,  1.0, -1.0], [-1.0,  1.0, -1.0],
        [-1.0, -1.0,  1.0], [ 1.0, -1.0,  1.0], [ 1.0,  1.0,  1.0], [-1.0,  1.0,  1.0],
    ], dtype=np.float64)

    min_det = float("inf")
    for elem in elements:
        Xe = nodes[elem]  # (8, 3)
        for xi in gauss_pts:
            for eta in gauss_pts:
                for zeta in gauss_pts:
                    dN_nat = np.zeros((3, 8), dtype=np.float64)
                    for a in range(8):
                        xa, ya, za = xi_nodes[a]
                        dN_nat[0, a] = 0.125 * xa * (1.0 + ya * eta) * (1.0 + za * zeta)
                        dN_nat[1, a] = 0.125 * ya * (1.0 + xa * xi) * (1.0 + za * zeta)
                        dN_nat[2, a] = 0.125 * za * (1.0 + xa * xi) * (1.0 + ya * eta)

                    J = dN_nat @ Xe  # (3, 3)
                    detJ = np.linalg.det(J)
                    if detJ < min_det:
                        min_det = detJ
    return float(min_det)


class CADOctreeSnapper:
    """
    Projects octree AMR mesh nodes onto analytical CAD surfaces, eliminating
    voxel staircasing while preserving hanging-node MPC conformity.
    """

    @staticmethod
    def snap_amr_mesh(
        amr_mesh: AMRMesh,
        cad_surfaces: Sequence[CADSurface],
        max_snap_dist: Optional[float] = None,
        boundary_only: bool = True,
        damping_factor: float = 0.8,
        min_allowed_detJ: float = 1e-6,
    ) -> Tuple[AMRMesh, int, float, np.ndarray]:
        """
        Snap boundary nodes of an AMRMesh onto CAD surfaces.

        Parameters:
            amr_mesh: Input conforming AMRMesh.
            cad_surfaces: Sequence of target CADSurface geometries.
            max_snap_dist: Maximum Euclidean distance for a node to be snapped (default: 0.4 * h_min).
            boundary_only: Only snap boundary nodes (incident elements < 8) to prevent interior collapse.
            damping_factor: Step damping applied if element approaches tangling.
            min_allowed_detJ: Minimum permissible element Jacobian determinant.

        Returns:
            snapped_mesh: New conforming AMRMesh with snapped boundary nodes.
            snapped_count: Number of independent boundary nodes snapped.
            min_detJ: Minimum element Jacobian determinant after snapping.
            snapped_flags: Boolean mask of shape (N_independent_nodes,) indicating snapped nodes.
        """
        indep_indices = amr_mesh.independent_node_indices
        n_true = len(indep_indices)

        # Automatically determine snapping threshold if not provided
        if max_snap_dist is None:
            # Estimate minimum element edge length
            edge_lens = np.linalg.norm(
                amr_mesh.nodes[amr_mesh.elements[:, 1]] - amr_mesh.nodes[amr_mesh.elements[:, 0]],
                axis=1,
            )
            h_min = float(np.min(edge_lens))
            snap_dist_thresh = 0.40 * h_min
        else:
            snap_dist_thresh = float(max_snap_dist)

        # Count incident elements to identify boundary vs interior nodes
        node_elem_count = np.zeros(amr_mesh.total_nodes, dtype=int)
        np.add.at(node_elem_count, amr_mesh.elements.ravel(), 1)

        # Work on independent coordinates
        x_true_orig = amr_mesh.nodes[indep_indices].copy()
        x_true_new = x_true_orig.copy()

        snapped_flags = np.zeros(n_true, dtype=bool)

        # 1. Project independent boundary nodes near CAD surfaces
        for i in range(n_true):
            nid = indep_indices[i]
            if boundary_only and node_elem_count[nid] >= 8:
                continue

            p = x_true_orig[i]
            best_surf = None
            min_dist = float("inf")
            for surf in cad_surfaces:
                d = surf.distance(p)
                if d < min_dist:
                    min_dist = d
                    best_surf = surf

            if best_surf is not None and min_dist <= snap_dist_thresh:
                proj_p = best_surf.project(p)
                x_true_new[i] = proj_p
                snapped_flags[i] = True

        snapped_count = int(np.sum(snapped_flags))

        # 2. Reconstruct all nodes including hanging nodes: x_all = C @ x_true
        x_all_snapped = amr_mesh.constraint_matrix @ x_true_new

        # 3. Quality Guardrail: Check minimum element Jacobian
        min_det = compute_hex8_min_jacobian(x_all_snapped, amr_mesh.elements)

        # If tangling or near-singular elements are detected, damp displacements
        alpha = 1.0
        while min_det < min_allowed_detJ and alpha > 0.05:
            alpha *= damping_factor
            x_true_damped = (1.0 - alpha) * x_true_orig + alpha * x_true_new
            x_all_snapped = amr_mesh.constraint_matrix @ x_true_damped
            min_det = compute_hex8_min_jacobian(x_all_snapped, amr_mesh.elements)

        snapped_mesh = AMRMesh(
            nodes=x_all_snapped,
            elements=amr_mesh.elements,
            leaf_levels=amr_mesh.leaf_levels,
            independent_node_indices=amr_mesh.independent_node_indices,
            hanging_node_indices=amr_mesh.hanging_node_indices,
            constraint_matrix=amr_mesh.constraint_matrix,
        )

        return snapped_mesh, snapped_count, min_det, snapped_flags

    @staticmethod
    def evaluate_surface_distance_error(
        mesh: AMRMesh,
        surface: CADSurface,
        node_indices: Optional[Sequence[int]] = None,
    ) -> Tuple[float, float]:
        """
        Compute mean and max geometric distance error from selected nodes to a CAD surface.

        Returns:
            mean_dist, max_dist in meters.
        """
        if node_indices is None:
            pts = mesh.nodes
        else:
            pts = mesh.nodes[node_indices]

        dists = np.array([surface.distance(p) for p in pts], dtype=np.float64)
        return float(np.mean(dists)), float(np.max(dists))
