"""
Scalable 3D Continuum Solid Mesh and Model Generator for C3D10 Elements.

Provides high-throughput generation of 10-node quadratic tetrahedral (C3D10)
FEAModels with physical boundary conditions for large-scale GPU benchmarking:
1. Structured tetrahedral block subdivision (instantly scales to 1M+ DOFs).
2. Programmatic OpenCASCADE CAD meshing via Gmsh.
"""

from __future__ import annotations

import time
import numpy as np
from pathlib import Path

from ..model import FEAModel
from ..properties.materials import MaterialDef
from ..boundary.conditions import create_fixed_support, LoadDef


def generate_c3d10_structured_block(
    length: float = 10.0,
    width: float = 1.0,
    height: float = 1.0,
    nx: int = 40,
    ny: int = 4,
    nz: int = 4,
    E: float = 2.1e11,
    nu: float = 0.3,
    tip_load_total: float = 10000.0,
) -> tuple[FEAModel, dict]:
    """
    Generate a conforming 10-node quadratic tetrahedral (C3D10) solid mesh
    of a 3D rectangular cantilever beam by subdividing a structured hex grid.

    Each hexahedral cell [i, j, k] is subdivided into 5 tetrahedra.
    Corner nodes are augmented with unique mid-edge nodes to form C3D10 elements.

    Parameters
    ----------
    length, width, height : Dimensions of the cantilever beam.
    nx, ny, nz : Number of subdivisions along X, Y, Z.
    E, nu : Young's modulus and Poisson's ratio.
    tip_load_total : Total transverse shear load applied at x = length.

    Returns
    -------
    model : FEAModel populated with nodes, solid_elements, supports, and loads.
    meta : Metadata dictionary with node count, element count, DOFs, and timing.
    """
    t0 = time.perf_counter()

    # 1. Generate corner grid vertices: (nx+1, ny+1, nz+1)
    x_grid = np.linspace(0.0, length, nx + 1)
    y_grid = np.linspace(0.0, width, ny + 1)
    z_grid = np.linspace(0.0, height, nz + 1)

    # 3D grid of corner node coordinates
    X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")
    corner_coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    n_corner_nodes = len(corner_coords)

    def corner_idx(i: int, j: int, k: int) -> int:
        return i * ((ny + 1) * (nz + 1)) + j * (nz + 1) + k

    # 2. Subdivide each hexahedron into 5 tetrahedra
    # 5 tets decomposition pattern:
    hex_tets_even = [
        (0, 1, 3, 4),
        (1, 2, 3, 6),
        (1, 4, 5, 6),
        (3, 4, 6, 7),
        (1, 3, 4, 6),
    ]
    hex_tets_odd = [
        (0, 1, 2, 5),
        (0, 2, 3, 7),
        (0, 4, 5, 7),
        (2, 5, 6, 7),
        (0, 2, 5, 7),
    ]

    linear_tets: list[tuple[int, int, int, int]] = []

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                parity = (i + j + k) % 2
                tet_patterns = hex_tets_odd if parity else hex_tets_even

                # 8 corners of current hex
                c = [
                    corner_idx(i,     j,     k),
                    corner_idx(i + 1, j,     k),
                    corner_idx(i + 1, j + 1, k),
                    corner_idx(i,     j + 1, k),
                    corner_idx(i,     j,     k + 1),
                    corner_idx(i + 1, j,     k + 1),
                    corner_idx(i + 1, j + 1, k + 1),
                    corner_idx(i,     j + 1, k + 1),
                ]

                for p in tet_patterns:
                    v0, v1, v2, v3 = c[p[0]], c[p[1]], c[p[2]], c[p[3]]
                    p0 = corner_coords[v0]
                    p1 = corner_coords[v1]
                    p2 = corner_coords[v2]
                    p3 = corner_coords[v3]
                    # Determinant of edge vectors: [p1-p0, p2-p0, p3-p0]
                    # Triple product (p1 - p0) . ((p2 - p0) x (p3 - p0))
                    d1 = p1 - p0
                    d2 = p2 - p0
                    d3 = p3 - p0
                    det_vol = d1[0] * (d2[1] * d3[2] - d2[2] * d3[1]) - \
                              d1[1] * (d2[0] * d3[2] - d2[2] * d3[0]) + \
                              d1[2] * (d2[0] * d3[1] - d2[1] * d3[0])
                    if det_vol < 0.0:
                        # Swap v1 and v2 to maintain right-hand rule and positive Jacobian
                        v1, v2 = v2, v1
                    linear_tets.append((v0, v1, v2, v3))

    n_elements = len(linear_tets)

    # 3. Upgrade 4-node tets to 10-node quadratic tets (C3D10)
    edge_pairs = [
        (0, 1),  # Edge 0-1
        (1, 2),  # Edge 1-2
        (2, 0),  # Edge 2-0
        (0, 3),  # Edge 0-3
        (1, 3),  # Edge 1-3
        (2, 3),  # Edge 2-3
    ]

    edge_to_midedge: dict[tuple[int, int], int] = {}
    midedge_coords_list: list[np.ndarray] = []
    c3d10_elements = np.empty((n_elements, 10), dtype=np.int32)

    current_node_id = n_corner_nodes

    for elem_idx, (v0, v1, v2, v3) in enumerate(linear_tets):
        v = (v0, v1, v2, v3)
        c3d10_elements[elem_idx, 0:4] = v

        for edge_idx, (e_a, e_b) in enumerate(edge_pairs):
            n_a = v[e_a]
            n_b = v[e_b]
            key = (min(n_a, n_b), max(n_a, n_b))

            mid_id = edge_to_midedge.get(key)
            if mid_id is None:
                mid_id = current_node_id
                edge_to_midedge[key] = mid_id
                current_node_id += 1
                mid_pos = 0.5 * (corner_coords[n_a] + corner_coords[n_b])
                midedge_coords_list.append(mid_pos)

            c3d10_elements[elem_idx, 4 + edge_idx] = mid_id

    # Combine corner coordinates and mid-edge coordinates
    if midedge_coords_list:
        all_coords = np.vstack([corner_coords, np.array(midedge_coords_list, dtype=np.float64)])
    else:
        all_coords = corner_coords

    n_total_nodes = len(all_coords)
    n_solid_dofs = n_total_nodes * 3

    # 4. Construct FEAModel
    model = FEAModel()
    model.mesh_nodes = all_coords
    model.solid_elements = c3d10_elements

    mat = MaterialDef(name="Steel", youngs_modulus=E, poissons_ratio=nu, yield_strength=2.5e8)
    model.materials["Steel"] = mat
    model.solid_materials = {i: "Steel" for i in range(n_elements)}

    # 5. Apply Boundary Conditions
    # Fixed clamp on face x = 0
    tol = 1e-6 * length
    root_nodes = np.where(all_coords[:, 0] <= tol)[0]
    for nid in root_nodes:
        model.supports.append(create_fixed_support(int(nid), is_geometry_node=False))

    # Tip shear load Fy on face x = length
    tip_nodes = np.where(all_coords[:, 0] >= length - tol)[0]
    load_per_node = tip_load_total / max(len(tip_nodes), 1)
    for nid in tip_nodes:
        model.loads.append(
            LoadDef(
                node_id=int(nid),
                fx=0.0,
                fy=load_per_node,
                fz=0.0,
                is_geometry_node=False,
            )
        )

    build_time = time.perf_counter() - t0

    meta = {
        "length": length,
        "width": width,
        "height": height,
        "grid": (nx, ny, nz),
        "n_nodes": n_total_nodes,
        "n_elements": n_elements,
        "n_dofs": n_solid_dofs,
        "root_nodes_fixed": len(root_nodes),
        "tip_nodes_loaded": len(tip_nodes),
        "build_time_sec": build_time,
    }

    return model, meta


def generate_c3d10_cad_cantilever(
    length: float = 10.0,
    width: float = 1.0,
    height: float = 1.0,
    mesh_size_max: float = 0.4,
    mesh_size_min: float = 0.2,
    E: float = 2.1e11,
    nu: float = 0.3,
    tip_load_total: float = 10000.0,
) -> tuple[FEAModel, dict]:
    """
    Generate an unstructured C3D10 solid mesh of a cantilever beam using Gmsh OpenCASCADE.
    """
    t0 = time.perf_counter()
    from .gmsh_mesher import GmshMesher

    mesher = GmshMesher(
        mesh_size_max=mesh_size_max,
        mesh_size_min=mesh_size_min,
        element_order=2,
        optimize=True,
        verbose=False,
    )

    def build_cad_beam(occ):
        occ.addBox(0.0, 0.0, 0.0, length, width, height)

    nodes, elements, model = mesher.mesh_cad(build_cad_beam, material_name="Steel")

    mat = MaterialDef(name="Steel", youngs_modulus=E, poissons_ratio=nu, yield_strength=2.5e8)
    model.materials["Steel"] = mat
    model.solid_materials = {i: "Steel" for i in range(len(elements))}

    # Boundary conditions
    fixed_nodes = mesher.get_nodes_on_plane(nodes, axis="x", coord=0.0, tol=1e-4)
    for nid in fixed_nodes:
        model.supports.append(create_fixed_support(int(nid), is_geometry_node=False))

    tip_nodes = mesher.get_nodes_on_plane(nodes, axis="x", coord=length, tol=1e-4)
    load_per_node = tip_load_total / max(len(tip_nodes), 1)
    for nid in tip_nodes:
        model.loads.append(
            LoadDef(
                node_id=int(nid),
                fx=0.0,
                fy=load_per_node,
                fz=0.0,
                is_geometry_node=False,
            )
        )

    build_time = time.perf_counter() - t0

    meta = {
        "length": length,
        "width": width,
        "height": height,
        "n_nodes": len(nodes),
        "n_elements": len(elements),
        "n_dofs": len(nodes) * 3,
        "root_nodes_fixed": len(fixed_nodes),
        "tip_nodes_loaded": len(tip_nodes),
        "build_time_sec": build_time,
    }

    return model, meta


def generate_c3d10_ibeam(
    length: float = 4.0,
    height: float = 0.3,
    flange_width: float = 0.2,
    flange_thick: float = 0.02,
    web_thick: float = 0.012,
    mesh_size_max: float = 0.05,
    mesh_size_min: float = 0.02,
    E: float = 2.1e11,
    nu: float = 0.3,
    tip_load_total: float = -10000.0,
    load_direction: str = "z",
) -> tuple[FEAModel, dict]:
    """
    Generate an industry-standard structural I-beam (W-section / IPE profile)
    meshed with 10-node quadratic tetrahedral solid elements (C3D10).

    The beam is aligned along the X-axis:
      - X ∈ [0, length]
      - Y ∈ [-flange_width/2, flange_width/2] (horizontal flange axis)
      - Z ∈ [-height/2, height/2]             (vertical web axis)

    Boundary Conditions:
      - Root face (x = 0): Fully clamped cantilever support (ux = uy = uz = 0).
      - Tip face (x = length): Transverse shear load distributed across tip nodes.

    Parameters
    ----------
    length : Span of the cantilever beam along X.
    height : Total section height along Z (flange outer to flange outer).
    flange_width : Width of top and bottom flanges along Y.
    flange_thick : Thickness of top and bottom flanges.
    web_thick : Thickness of the central web.
    mesh_size_max, mesh_size_min : Mesh size limits for Gmsh Delaunay / HXT.
    E, nu : Elastic modulus and Poisson's ratio.
    tip_load_total : Total load in Newtons applied at tip face.
    load_direction : "z" for vertical/major bending, "y" for lateral/minor bending.

    Returns
    -------
    model : Populated FEAModel with C3D10 solid elements, supports, and loads.
    meta : Metadata dictionary with geometry, mesh stats, and theoretical analytical metrics.
    """
    t0 = time.perf_counter()
    from .gmsh_mesher import GmshMesher

    mesher = GmshMesher(
        mesh_size_max=mesh_size_max,
        mesh_size_min=mesh_size_min,
        element_order=2,
        optimize=True,
        verbose=False,
    )

    def build_ibeam_cad(occ):
        # Bottom flange: Z from -height/2 to -height/2 + flange_thick
        b1 = occ.addBox(0.0, -flange_width / 2.0, -height / 2.0, length, flange_width, flange_thick)
        # Web: Z from -height/2 + flange_thick to height/2 - flange_thick
        b2 = occ.addBox(0.0, -web_thick / 2.0, -height / 2.0 + flange_thick, length, web_thick, height - 2.0 * flange_thick)
        # Top flange: Z from height/2 - flange_thick to height/2
        b3 = occ.addBox(0.0, -flange_width / 2.0, height / 2.0 - flange_thick, length, flange_width, flange_thick)
        # Fuse into a single seamless 3D solid continuum
        occ.fuse([(3, b1), (3, b3)], [(3, b2)])

    nodes, elements, model = mesher.mesh_cad(build_ibeam_cad, material_name="Steel")

    mat = MaterialDef(name="Steel", youngs_modulus=E, poissons_ratio=nu, yield_strength=2.5e8)
    model.materials["Steel"] = mat
    model.solid_materials = {i: "Steel" for i in range(len(elements))}

    # Boundary conditions:
    # 1. Clamp root at x = 0
    fixed_nodes = mesher.get_nodes_on_plane(nodes, axis="x", coord=0.0, tol=1e-4)
    for nid in fixed_nodes:
        model.supports.append(create_fixed_support(int(nid), is_geometry_node=False))

    # 2. Tip shear load at x = length
    tip_nodes = mesher.get_nodes_on_plane(nodes, axis="x", coord=length, tol=1e-4)
    load_per_node = tip_load_total / max(len(tip_nodes), 1)

    fx_val = 0.0
    fy_val = load_per_node if load_direction.lower() == "y" else 0.0
    fz_val = load_per_node if load_direction.lower() == "z" else 0.0

    for nid in tip_nodes:
        model.loads.append(
            LoadDef(
                node_id=int(nid),
                fx=fx_val,
                fy=fy_val,
                fz=fz_val,
                is_geometry_node=False,
            )
        )

    build_time = time.perf_counter() - t0

    # Theoretical Analytical Beam Mechanics
    # Second moment of area about horizontal Y-axis (major bending in Z)
    # I_y = (1/12) * [bf * h^3 - (bf - tw) * (h - 2*tf)^3]
    h_inner = height - 2.0 * flange_thick
    I_major = (1.0 / 12.0) * (flange_width * (height ** 3) - (flange_width - web_thick) * (h_inner ** 3))
    # Second moment of area about vertical Z-axis (minor bending in Y)
    # I_z = 2 * [(1/12) * tf * bf^3] + (1/12) * (h - 2*tf) * tw^3
    I_minor = (2.0 / 12.0) * flange_thick * (flange_width ** 3) + (1.0 / 12.0) * h_inner * (web_thick ** 3)

    I_active = I_major if load_direction.lower() == "z" else I_minor
    # Theoretical Euler-Bernoulli tip deflection: delta = P * L^3 / (3 * E * I)
    theoretical_tip_disp = (tip_load_total * (length ** 3)) / (3.0 * E * I_active)

    meta = {
        "profile": "Structural I-Beam",
        "length": length,
        "height": height,
        "flange_width": flange_width,
        "flange_thick": flange_thick,
        "web_thick": web_thick,
        "I_major": I_major,
        "I_minor": I_minor,
        "theoretical_tip_deflection_m": theoretical_tip_disp,
        "n_nodes": len(nodes),
        "n_elements": len(elements),
        "n_dofs": len(nodes) * 3,
        "root_nodes_fixed": len(fixed_nodes),
        "tip_nodes_loaded": len(tip_nodes),
        "build_time_sec": build_time,
    }

    return model, meta

