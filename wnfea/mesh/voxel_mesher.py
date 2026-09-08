"""
Cartesian Voxelizer & Immersed Boundary (IFEM) Generator for WNFEA.
-------------------------------------------------------------------
Implements Stage 1 of the Dual-Stage Voxel / Spherical Sub-Modeling Engine:
1. Generates structured Cartesian Hex8 voxel grids over bounding boxes or CAD shapes.
2. Evaluates active volume fractions alpha_e in [0, 1] for immersed boundaries.
3. Maps boundary supports and concentrated/distributed loads onto voxel grid nodes.
4. Supports sub-sampling for smooth volume fractions and exact Level-Set/SDF geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable, Union, Sequence
import numpy as np

from ..model import FEAModel
from ..geometry.primitives import Point3D


@dataclass
class VoxelGrid:
    """
    Structured 3D Cartesian Voxel Grid representing a structural domain.
    """
    bounds: tuple[float, float, float, float, float, float]  # (xmin, xmax, ymin, ymax, zmin, zmax)
    resolution: tuple[int, int, int]                         # (nx, ny, nz)
    pitch: tuple[float, float, float]                        # (hx, hy, hz)
    nodes: np.ndarray                                        # (N_nodes, 3) grid coordinates
    elements: np.ndarray                                     # (N_elements, 8) Hex8 connectivity
    volume_fractions: np.ndarray                             # (N_elements,) in [0, 1]
    active_element_indices: np.ndarray                       # indices where volume_fractions > 0
    active_node_indices: np.ndarray                          # unique nodes of active elements
    grid_shape: tuple[int, int, int] = field(init=False)     # (nx, ny, nz)

    def __post_init__(self):
        self.grid_shape = self.resolution

    @property
    def total_cells(self) -> int:
        return len(self.elements)

    @property
    def total_active_cells(self) -> int:
        return len(self.active_element_indices)

    @property
    def total_nodes(self) -> int:
        return len(self.nodes)

    @property
    def total_active_nodes(self) -> int:
        return len(self.active_node_indices)

    def node_id(self, i: int, j: int, k: int) -> int:
        """Convert 3D grid coordinate (i, j, k) to 1D global node ID."""
        nx, ny, _ = self.resolution
        return i + j * (nx + 1) + k * (nx + 1) * (ny + 1)

    def cell_id(self, i: int, j: int, k: int) -> int:
        """Convert 3D cell coordinate (i, j, k) to 1D element ID."""
        nx, ny, _ = self.resolution
        return i + j * nx + k * nx * ny

    def get_element_centroids(self) -> np.ndarray:
        """Return (N_elements, 3) centroids of all voxel cells."""
        return np.mean(self.nodes[self.elements], axis=1)

    def get_active_mesh(self, threshold: float = 1e-4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract compact active nodes, compact Hex8 elements, and compact volume fractions.
        Useful for standard FEA solves or VTK export.

        Returns:
            compact_nodes: (N_active_nodes, 3)
            compact_elements: (N_active_elems, 8) re-indexed into compact_nodes
            compact_alphas: (N_active_elems,)
        """
        mask = self.volume_fractions >= threshold
        active_elems = self.elements[mask]
        compact_alphas = self.volume_fractions[mask]

        unique_nodes, inverse = np.unique(active_elems, return_inverse=True)
        compact_nodes = self.nodes[unique_nodes]
        compact_elements = inverse.reshape(active_elems.shape)

        return compact_nodes, compact_elements, compact_alphas

    def locate_point(self, point: Sequence[float]) -> tuple[int, int, int]:
        """
        Locate the 3D cell index (i, j, k) containing a given spatial point.
        Clamps to grid boundary if slightly outside.
        """
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        nx, ny, nz = self.resolution
        hx, hy, hz = self.pitch

        x, y, z = point
        i = int(np.clip(np.floor((x - xmin) / hx), 0, nx - 1))
        j = int(np.clip(np.floor((y - ymin) / hy), 0, ny - 1))
        k = int(np.clip(np.floor((z - zmin) / hz), 0, nz - 1))
        return i, j, k

    def sample_trilinear_weights(self, point: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the 8 host corner node indices and trilinear interpolation shape functions
        N_J(x) for an arbitrary query point in space.

        Returns:
            corner_node_ids: (8,) int64 array of global node IDs
            weights: (8,) float64 shape function values summing to 1.0
        """
        xmin, _, ymin, _, zmin, _ = self.bounds
        hx, hy, hz = self.pitch
        x, y, z = point

        i, j, k = self.locate_point(point)
        elem_idx = self.cell_id(i, j, k)
        corner_node_ids = self.elements[elem_idx]

        # Natural coordinates xi, eta, zeta in [0, 1] within the cell
        x0 = xmin + i * hx
        y0 = ymin + j * hy
        z0 = zmin + k * hz

        xi = np.clip((x - x0) / hx, 0.0, 1.0)
        eta = np.clip((y - y0) / hy, 0.0, 1.0)
        zeta = np.clip((z - z0) / hz, 0.0, 1.0)

        # Standard trilinear shape functions
        w0 = (1.0 - xi) * (1.0 - eta) * (1.0 - zeta)
        w1 = xi * (1.0 - eta) * (1.0 - zeta)
        w2 = xi * eta * (1.0 - zeta)
        w3 = (1.0 - xi) * eta * (1.0 - zeta)
        w4 = (1.0 - xi) * (1.0 - eta) * zeta
        w5 = xi * (1.0 - eta) * zeta
        w6 = xi * eta * zeta
        w7 = (1.0 - xi) * eta * zeta

        weights = np.array([w0, w1, w2, w3, w4, w5, w6, w7], dtype=np.float64)
        return corner_node_ids, weights


class VoxelMesher:
    """
    High-Performance Cartesian Voxel Grid Generator and Immersed Boundary Classifier.
    """

    @staticmethod
    def create_box_grid(
        bounds: tuple[float, float, float, float, float, float],
        resolution: tuple[int, int, int],
        inside_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        subsampling: int = 1,
    ) -> VoxelGrid:
        """
        Generate a regular Cartesian Hex8 voxel grid over a rectangular domain [xmin, xmax] x ...

        Parameters:
            bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
            resolution: (nx, ny, nz) cell counts along each axis
            inside_fn: Optional vectorized function taking (P, 3) coords and returning
                       boolean mask or signed-distance array (< 0 is inside).
            subsampling: Sub-grid sampling points per axis for volume fraction (1 = centroid only,
                         2 = 8 points, 3 = 27 points).

        Returns:
            VoxelGrid object.
        """
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        nx, ny, nz = resolution

        hx = (xmax - xmin) / nx
        hy = (ymax - ymin) / ny
        hz = (zmax - zmin) / nz
        pitch = (hx, hy, hz)

        # 1. Generate grid nodes
        xs = np.linspace(xmin, xmax, nx + 1, dtype=np.float64)
        ys = np.linspace(ymin, ymax, ny + 1, dtype=np.float64)
        zs = np.linspace(zmin, zmax, nz + 1, dtype=np.float64)

        # Meshgrid with 'ij' indexing: shape (nx+1, ny+1, nz+1)
        grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
        # Flatten with C-order where z varies fastest, then y, then x, or i + j*(nx+1) + k*(nx+1)*(ny+1)
        # Let's match: node_id(i, j, k) = i + j*(nx+1) + k*(nx+1)*(ny+1)
        # That means transpose to (nz+1, ny+1, nx+1) and flatten:
        coords = np.stack([grid_x, grid_y, grid_z], axis=-1)  # (nx+1, ny+1, nz+1, 3)
        # Transpose to (k, j, i) order for node_id = i + (nx+1)*j + (nx+1)*(ny+1)*k
        coords_ordered = np.transpose(coords, (2, 1, 0, 3)).reshape(-1, 3)

        # 2. Build Hex8 connectivity
        # Standard Hex8 node order:
        # Bottom: (0,0,0), (1,0,0), (1,1,0), (0,1,0)
        # Top:    (0,0,1), (1,0,1), (1,1,1), (0,1,1)
        n_cells = nx * ny * nz
        elements = np.zeros((n_cells, 8), dtype=np.int64)

        # Compute element node offsets
        stride_x = 1
        stride_y = nx + 1
        stride_z = (nx + 1) * (ny + 1)

        cell_i, cell_j, cell_k = np.meshgrid(
            np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
        )
        # Transpose to (k, j, i) for cell_id = i + j*nx + k*nx*ny
        cell_i = np.transpose(cell_i, (2, 1, 0)).ravel()
        cell_j = np.transpose(cell_j, (2, 1, 0)).ravel()
        cell_k = np.transpose(cell_k, (2, 1, 0)).ravel()

        base_nodes = cell_i * stride_x + cell_j * stride_y + cell_k * stride_z
        elements[:, 0] = base_nodes
        elements[:, 1] = base_nodes + stride_x
        elements[:, 2] = base_nodes + stride_x + stride_y
        elements[:, 3] = base_nodes + stride_y
        elements[:, 4] = base_nodes + stride_z
        elements[:, 5] = base_nodes + stride_x + stride_z
        elements[:, 6] = base_nodes + stride_x + stride_y + stride_z
        elements[:, 7] = base_nodes + stride_y + stride_z

        # 3. Compute Volume Fractions
        if inside_fn is None:
            # Entire bounding box is solid
            alphas = np.ones(n_cells, dtype=np.float64)
        else:
            if subsampling <= 1:
                # Evaluate at cell centroids
                centroids = (
                    np.stack([xmin + (cell_i + 0.5) * hx,
                              ymin + (cell_j + 0.5) * hy,
                              zmin + (cell_k + 0.5) * hz], axis=-1)
                )
                mask = inside_fn(centroids)
                if mask.dtype == bool:
                    alphas = mask.astype(np.float64)
                else:
                    # Signed distance: <= 0 is solid
                    alphas = (mask <= 0.0).astype(np.float64)
            else:
                # Sub-sampled quadrature for smooth volume fractions
                sub_pts = (np.arange(subsampling) + 0.5) / subsampling
                sx, sy, sz = np.meshgrid(sub_pts, sub_pts, sub_pts, indexing="ij")
                offsets = np.stack([sx.ravel() * hx, sy.ravel() * hy, sz.ravel() * hz], axis=-1)  # (M, 3)

                n_sub = len(offsets)
                alphas = np.zeros(n_cells, dtype=np.float64)

                cell_origins = np.stack([
                    xmin + cell_i * hx,
                    ymin + cell_j * hy,
                    zmin + cell_k * hz
                ], axis=-1)  # (N_cells, 3)

                for off in offsets:
                    pts = cell_origins + off[None, :]
                    mask = inside_fn(pts)
                    if mask.dtype == bool:
                        alphas += mask.astype(np.float64)
                    else:
                        alphas += (mask <= 0.0).astype(np.float64)

                alphas /= float(n_sub)

        active_elem_idx = np.where(alphas > 1e-4)[0]
        if len(active_elem_idx) > 0:
            active_nodes = np.unique(elements[active_elem_idx].ravel())
        else:
            active_nodes = np.empty(0, dtype=np.int64)

        return VoxelGrid(
            bounds=bounds,
            resolution=resolution,
            pitch=pitch,
            nodes=coords_ordered,
            elements=elements,
            volume_fractions=alphas,
            active_element_indices=active_elem_idx,
            active_node_indices=active_nodes,
        )

    @classmethod
    def voxelize_model_bounding_box(
        cls,
        model: FEAModel,
        resolution: tuple[int, int, int] = (20, 10, 10),
        padding_ratio: float = 0.0,
    ) -> VoxelGrid:
        """
        Voxelize the bounding envelope of an existing FEAModel's solid mesh.
        """
        coords = model.mesh_nodes
        xmin, ymin, zmin = np.min(coords, axis=0)
        xmax, ymax, zmax = np.max(coords, axis=0)

        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin

        pad_x = dx * padding_ratio
        pad_y = dy * padding_ratio
        pad_z = dz * padding_ratio

        bounds = (
            xmin - pad_x, xmax + pad_x,
            ymin - pad_y, ymax + pad_y,
            zmin - pad_z, zmax + pad_z
        )

        return cls.create_box_grid(bounds, resolution)
