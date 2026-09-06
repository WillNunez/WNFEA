"""
End-to-End Integration Test for Gmsh STEP Meshing and ParaView VTU Export.

Tests:
1. Solid STEP CAD generation & Gmsh C3D10 meshing (10-node quadratic tets).
2. Jacobian determinant validity across all generated elements (det(J) > 0).
3. Boundary condition definition and linear static solution via direct elimination.
4. ParaView XML UnstructuredGrid (.vtu) export with Displacement & Cauchy stress fields.
5. XML document structure validation against the VTK UnstructuredGrid schema.
6. Mixed-topology model export (VTK_LINE + VTK_QUADRATIC_TETRA).
"""

from __future__ import annotations

import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
try:
    import pytest
except ImportError:
    pytest = None


from wnfea.model import FEAModel
from wnfea.mesh import GmshMesher
from wnfea.results import VTUExporter, export_vtu
from wnfea.boundary.conditions import create_fixed_support, LoadDef
from wnfea.solver import solve_linear_system
from wnfea.elements.c3d10 import jacobian_c3d10, shape_functions_c3d10, GAUSS_POINTS_C3D10


def test_gmsh_step_meshing_and_vtu_export(tmp_path):
    """Test importing a STEP solid CAD file, meshing with C3D10, solving, and exporting to VTU."""
    import gmsh

    step_file = tmp_path / "cantilever_solid.stp"
    vtu_file = tmp_path / "cantilever_solid.vtu"

    # 1. Create a STEP solid geometry via Gmsh OCC
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("CAD_Beam")
    # Box: L=4.0, W=1.0, H=1.0
    gmsh.model.occ.addBox(0.0, 0.0, 0.0, 4.0, 1.0, 1.0)
    gmsh.model.occ.synchronize()
    gmsh.write(str(step_file))
    gmsh.finalize()

    assert step_file.exists(), f"STEP file was not created: {step_file}"

    # 2. Mesh the STEP file using GmshMesher
    mesher = GmshMesher(
        mesh_size_max=1.0,
        mesh_size_min=0.5,
        element_order=2,  # Quadratic C3D10
        optimize=True,
        verbose=False,
    )

    model = FEAModel()
    nodes, elements, model = mesher.mesh_file(step_file, model=model, material_name="Steel")

    assert len(nodes) > 0, "No nodes generated"
    assert len(elements) > 0, "No elements generated"
    assert elements.shape[1] == 10, f"Expected 10-node elements, got shape {elements.shape}"

    # 3. Verify Jacobian determinants on all elements at all 4 Gauss points
    for elem_id, node_ids in enumerate(elements):
        coords = nodes[node_ids]
        for xi, eta, zeta in GAUSS_POINTS_C3D10:
            _, dN_dxi = shape_functions_c3d10(xi, eta, zeta)
            _, det_J, _ = jacobian_c3d10(coords, dN_dxi)
            assert det_J > 0.0, f"Inverted or degenerate element {elem_id} (det_J = {det_J})"

    # 4. Apply Boundary Conditions:
    # Clamp face at x = 0 (fixed ux, uy, uz)
    fixed_nodes = mesher.get_nodes_on_plane(nodes, axis="x", coord=0.0, tol=1e-4)
    assert len(fixed_nodes) > 0, "Failed to identify clamped face nodes at x=0"

    for nid in fixed_nodes:
        model.supports.append(create_fixed_support(int(nid), is_geometry_node=False))


    # Apply tip transverse load Fy = 1000 N distributed on face x = 4.0
    tip_nodes = mesher.get_nodes_on_plane(nodes, axis="x", coord=4.0, tol=1e-4)
    assert len(tip_nodes) > 0, "Failed to identify tip face nodes at x=4.0"
    load_per_node = 1000.0 / len(tip_nodes)

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

    # 5. Solve the linear system
    solve_linear_system(model)

    assert model.displacements is not None, "Model displacements not populated"
    u_vec = model.displacements.reshape(-1, 6)

    # Verify boundary displacement at root is zero
    for nid in fixed_nodes:
        assert np.allclose(u_vec[nid, :3], 0.0, atol=1e-9), f"Clamped node {nid} moved: {u_vec[nid, :3]}"

    # Verify tip deflection in y direction is strictly positive
    tip_deflections = [u_vec[nid, 1] for nid in tip_nodes]
    avg_tip_deflection = float(np.mean(tip_deflections))
    assert avg_tip_deflection > 0.0, f"Tip deflection must be positive, got {avg_tip_deflection}"

    # 6. Export to ParaView VTU format
    out_path = export_vtu(model, vtu_file, compute_stresses=True)
    assert Path(out_path).exists(), f"VTU file not created: {out_path}"
    assert Path(out_path).stat().st_size > 0, "VTU file is empty"

    # 7. Validate XML structure
    tree = ET.parse(out_path)
    root = tree.getroot()
    assert root.tag == "VTKFile", f"Root element must be VTKFile, got {root.tag}"
    assert root.attrib.get("type") == "UnstructuredGrid"

    piece = root.find(".//Piece")
    assert piece is not None, "Missing <Piece> in VTU"
    assert int(piece.attrib["NumberOfPoints"]) == len(nodes)
    assert int(piece.attrib["NumberOfCells"]) == len(elements)

    # Check Points
    points_elem = piece.find(".//Points/DataArray[@Name='Points']")
    assert points_elem is not None, "Missing Points DataArray"
    assert points_elem.attrib.get("NumberOfComponents") == "3"

    # Check Cells
    conn_elem = piece.find(".//Cells/DataArray[@Name='connectivity']")
    offsets_elem = piece.find(".//Cells/DataArray[@Name='offsets']")
    types_elem = piece.find(".//Cells/DataArray[@Name='types']")
    assert conn_elem is not None and offsets_elem is not None and types_elem is not None

    types = [int(t) for t in types_elem.text.split()]
    assert all(t == 24 for t in types), "All elements must be VTK_QUADRATIC_TETRA (type 24)"

    # Check PointData (Displacements)
    disp_elem = piece.find(".//PointData/DataArray[@Name='Displacement']")
    assert disp_elem is not None, "Missing Displacement in PointData"
    assert disp_elem.attrib.get("NumberOfComponents") == "3"

    # Check CellData (VonMises and StressTensor)
    vm_elem = piece.find(".//CellData/DataArray[@Name='VonMises']")
    stress_elem = piece.find(".//CellData/DataArray[@Name='StressTensor']")
    assert vm_elem is not None, "Missing VonMises in CellData"
    assert stress_elem is not None, "Missing StressTensor in CellData"
    assert stress_elem.attrib.get("NumberOfComponents") == "6"

    cell_vm_vals = [float(v) for v in vm_elem.text.split()]
    assert len(cell_vm_vals) == len(elements)
    assert max(cell_vm_vals) > 0.0, "Stress field should be non-zero"


def test_vtu_mixed_beam_and_solid_export(tmp_path):
    """Test exporting a hybrid model containing both 1D beams and 3D solid elements."""
    vtu_file = tmp_path / "mixed_model.vtu"

    model = FEAModel()
    # 4 corner vertices + 6 mid-edges = 10 nodes for solid tet
    # + 1 additional node for a connected beam
    nodes = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [0.0, 1.0, 0.0],  # 2
        [0.0, 0.0, 1.0],  # 3
        [0.5, 0.0, 0.0],  # 4
        [0.5, 0.5, 0.0],  # 5
        [0.0, 0.5, 0.0],  # 6
        [0.0, 0.0, 0.5],  # 7
        [0.5, 0.0, 0.5],  # 8 (1-3)
        [0.0, 0.5, 0.5],  # 9 (2-3)
        [2.0, 0.0, 0.0],  # 10 (beam tip connected to node 1)
    ], dtype=np.float64)

    # 1 Solid C3D10 element
    solid_elems = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int64)
    # 1 Beam element connecting node 1 to node 10
    beam_elems = np.array([[1, 10]], dtype=np.int64)

    model.mesh_nodes = nodes
    model.solid_elements = solid_elems
    model.mesh_elements = beam_elems
    model.solid_materials[0] = "Steel"

    # Fake displacements
    model.displacements = np.zeros(len(nodes) * 6, dtype=np.float64)
    model.displacements[10 * 6 + 1] = 0.05  # Beam tip uy = 0.05

    exporter = VTUExporter()
    out_path = exporter.export(model, vtu_file, compute_stresses=False)

    tree = ET.parse(out_path)
    root = tree.getroot()
    piece = root.find(".//Piece")

    assert int(piece.attrib["NumberOfPoints"]) == 11
    assert int(piece.attrib["NumberOfCells"]) == 2  # 1 beam + 1 solid

    types_elem = piece.find(".//Cells/DataArray[@Name='types']")
    types = [int(t) for t in types_elem.text.split()]
    assert types == [3, 24], f"Expected types [3, 24] for beam and quadratic tet, got {types}"


if __name__ == "__main__":
    import tempfile
    print("Running Gmsh STEP Meshing & ParaView VTU Exporter Integration Tests...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        print("  -> Running test_gmsh_step_meshing_and_vtu_export...")
        test_gmsh_step_meshing_and_vtu_export(tmp_p)
        print("     PASSED!")

        print("  -> Running test_vtu_mixed_beam_and_solid_export...")
        test_vtu_mixed_beam_and_solid_export(tmp_p)
        print("     PASSED!")

    print("\nALL GMSH & VTU INTEGRATION TESTS PASSED 100%!")

