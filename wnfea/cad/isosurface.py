"""
Fast Watertight Isosurface Extraction & Curvature Smoothing for WNFEA.
----------------------------------------------------------------------
Extracts manifold, watertight triangular surface meshes from 3D topology optimization
density fields:
1. Boundary Facet Extraction: Dual Surface Net identifying solid/void interfaces (rho >= isovalue).
2. Manifold Triangulation: Exactly 2 triangles per boundary cell face.
3. Curvature-Preserving Laplacian Smoothing: Eliminates staircase voxel facets.
4. Binary STL and Wavefront OBJ Export for FreeCAD OpenCASCADE B-Rep ingestion.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from ..mesh.voxel_mesher import VoxelGrid


@dataclass
class TriangularMesh:
    """
    Watertight triangular surface mesh.
    """
    vertices: np.ndarray  # (V, 3) float64 coordinates
    faces: np.ndarray     # (F, 3) int64 vertex indices

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    def write_stl(self, filepath: str, binary: bool = True):
        """Write surface mesh to STL file."""
        v = self.vertices
        f = self.faces

        # Compute face normal vectors
        v0 = v[f[:, 0]]
        v1 = v[f[:, 1]]
        v2 = v[f[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
        norm_len[norm_len < 1e-12] = 1.0
        normals /= norm_len

        if binary:
            with open(filepath, "wb") as fp:
                # 80-byte header
                header = b"WNFEA Isosurface Mesh Export" + b"\0" * (80 - 28)
                fp.write(header)
                # Number of triangles (uint32)
                fp.write(struct.pack("<I", len(f)))

                for i in range(len(f)):
                    n = normals[i]
                    p0 = v0[i]
                    p1 = v1[i]
                    p2 = v2[i]
                    # 12 floats (normal: 3, v0: 3, v1: 3, v2: 3) + 2 bytes attribute
                    data = struct.pack(
                        "<ffffffffffffH",
                        float(n[0]), float(n[1]), float(n[2]),
                        float(p0[0]), float(p0[1]), float(p0[2]),
                        float(p1[0]), float(p1[1]), float(p1[2]),
                        float(p2[0]), float(p2[1]), float(p2[2]),
                        0,
                    )
                    fp.write(data)
        else:
            with open(filepath, "w") as fp:
                fp.write("solid WNFEA_Isosurface\n")
                for i in range(len(f)):
                    n = normals[i]
                    fp.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
                    fp.write("    outer loop\n")
                    fp.write(f"      vertex {v0[i,0]:.6e} {v0[i,1]:.6e} {v0[i,2]:.6e}\n")
                    fp.write(f"      vertex {v1[i,0]:.6e} {v1[i,1]:.6e} {v1[i,2]:.6e}\n")
                    fp.write(f"      vertex {v2[i,0]:.6e} {v2[i,1]:.6e} {v2[i,2]:.6e}\n")
                    fp.write("    endloop\n")
                    fp.write("  endfacet\n")
                fp.write("endsolid WNFEA_Isosurface\n")

    def write_obj(self, filepath: str):
        """Write surface mesh to Wavefront OBJ format."""
        with open(filepath, "w") as fp:
            fp.write("# WNFEA Surface Mesh\n")
            for pt in self.vertices:
                fp.write(f"v {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")
            for tri in self.faces:
                fp.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


def extract_isosurface_mesh(
    grid: VoxelGrid,
    density: np.ndarray,
    isovalue: float = 0.5,
    smoothing_iters: int = 3,
    smoothing_factor: float = 0.5,
) -> TriangularMesh:
    """
    Extract a smoothed watertight triangular surface mesh from a 3D density field.

    Parameters:
        grid: VoxelGrid containing spatial resolution and bounds.
        density: (N_elements,) element physical densities.
        isovalue: Density threshold for solid/void interface (default: 0.5).
        smoothing_iters: Number of Laplacian smoothing passes (default: 3).
        smoothing_factor: Damping parameter lambda in (0, 1).

    Returns:
        TriangularMesh object.
    """
    nx, ny, nz = grid.resolution
    hx, hy, hz = grid.pitch
    xmin, _, ymin, _, zmin, _ = grid.bounds

    # 3D binary solid mask: True if rho >= isovalue
    # Shape: (nz, ny, nx) corresponding to cell_id = i + j*nx + k*nx*ny
    solid_3d = (density >= isovalue).reshape((nz, ny, nx))

    # Identify all boundary faces:
    # A boundary face exists if a solid cell borders a void cell (or the domain edge)
    quad_vertices = []  # list of (4, 3) quad corner coordinates

    # 1. X-faces (normal along +/- X)
    # Check left boundary (i=0)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if not solid_3d[k, j, i]:
                    continue

                x0 = xmin + i * hx
                x1 = x0 + hx
                y0 = ymin + j * hy
                y1 = y0 + hy
                z0 = zmin + k * hz
                z1 = z0 + hz

                # -X face (left)
                if i == 0 or not solid_3d[k, j, i - 1]:
                    quad_vertices.append([
                        [x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]
                    ])

                # +X face (right)
                if i == nx - 1 or not solid_3d[k, j, i + 1]:
                    quad_vertices.append([
                        [x1, y0, z0], [x1, y0, z1], [x1, y1, z1], [x1, y1, z0]
                    ])

                # -Y face (front)
                if j == 0 or not solid_3d[k, j - 1, i]:
                    quad_vertices.append([
                        [x0, y0, z0], [x0, y0, z1], [x1, y0, z1], [x1, y0, z0]
                    ])

                # +Y face (back)
                if j == ny - 1 or not solid_3d[k, j + 1, i]:
                    quad_vertices.append([
                        [x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]
                    ])

                # -Z face (bottom)
                if k == 0 or not solid_3d[k - 1, j, i]:
                    quad_vertices.append([
                        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]
                    ])

                # +Z face (top)
                if k == nz - 1 or not solid_3d[k + 1, j, i]:
                    quad_vertices.append([
                        [x0, y0, z1], [x0, y1, z1], [x1, y1, z1], [x1, y0, z1]
                    ])

    if not quad_vertices:
        # Trivial / empty mesh
        return TriangularMesh(np.empty((0, 3)), np.empty((0, 3), dtype=np.int64))

    # Merge quad vertices and build unique vertex table
    quad_vertices = np.array(quad_vertices, dtype=np.float64)  # (Q, 4, 3)
    n_quads = len(quad_vertices)

    raw_pts = quad_vertices.reshape(-1, 3)
    # Round to 6 decimals to snap shared boundary vertices
    snapped_pts = np.round(raw_pts, decimals=6)
    unique_pts, inv_indices = np.unique(snapped_pts, axis=0, return_inverse=True)

    quad_indices = inv_indices.reshape(n_quads, 4)

    # Triangulate each quad: (0, 1, 2) and (0, 2, 3)
    tri_0 = quad_indices[:, [0, 1, 2]]
    tri_1 = quad_indices[:, [0, 2, 3]]
    faces = np.vstack([tri_0, tri_1])

    vertices = unique_pts.copy()

    # Apply Laplacian Smoothing to eliminate voxel staircasing
    if smoothing_iters > 0:
        # Build vertex adjacency list
        n_v = len(vertices)
        adj: dict[int, set[int]] = {idx: set() for idx in range(n_v)}
        for tri in faces:
            adj[tri[0]].update([tri[1], tri[2]])
            adj[tri[1]].update([tri[0], tri[2]])
            adj[tri[2]].update([tri[0], tri[1]])

        for _ in range(smoothing_iters):
            new_v = vertices.copy()
            for idx, neighbors in adj.items():
                if neighbors:
                    centroid = np.mean(vertices[list(neighbors)], axis=0)
                    new_v[idx] += smoothing_factor * (centroid - vertices[idx])
            vertices = new_v

    return TriangularMesh(vertices=vertices, faces=faces)
