"""
Tests for Phase 7: FreeCAD B-Rep Reconstruction, Isosurface Extraction, and ParaView Pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import tempfile
import unittest
import numpy as np

from wnfea.mesh.voxel_mesher import VoxelMesher
from wnfea.cad.isosurface import TriangularMesh, extract_isosurface_mesh
from wnfea.cad.brep_reconstruction import BRepReconstructor
from wnfea.cad.freecad_brep import FreeCADBRepDetector, CylindricalFeature
from wnfea.results.paraview_export import export_voxel_grid_vtu, generate_paraview_macro


class TestCADParaViewPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_triangular_mesh_stl_obj_export(self):
        """Test TriangularMesh STL (binary + ascii) and OBJ serialization."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [0, 1, 4],
        ], dtype=np.int64)

        mesh = TriangularMesh(vertices=vertices, faces=faces)
        self.assertEqual(mesh.num_vertices, 5)
        self.assertEqual(mesh.num_faces, 3)

        # Test binary STL
        stl_bin_path = os.path.join(self.tmp_dir.name, "test_bin.stl")
        mesh.write_stl(stl_bin_path, binary=True)
        self.assertTrue(os.path.exists(stl_bin_path))
        # 80 bytes header + 4 bytes count + 3 * 50 bytes = 234 bytes
        self.assertEqual(os.path.getsize(stl_bin_path), 84 + 3 * 50)

        # Test ASCII STL
        stl_ascii_path = os.path.join(self.tmp_dir.name, "test_ascii.stl")
        mesh.write_stl(stl_ascii_path, binary=False)
        self.assertTrue(os.path.exists(stl_ascii_path))
        with open(stl_ascii_path, "r") as fp:
            content = fp.read()
            self.assertIn("solid WNFEA_Isosurface", content)
            self.assertIn("facet normal", content)

        # Test OBJ
        obj_path = os.path.join(self.tmp_dir.name, "test.obj")
        mesh.write_obj(obj_path)
        self.assertTrue(os.path.exists(obj_path))
        with open(obj_path, "r") as fp:
            lines = fp.readlines()
            v_lines = [l for l in lines if l.startswith("v ")]
            f_lines = [l for l in lines if l.startswith("f ")]
            self.assertEqual(len(v_lines), 5)
            self.assertEqual(len(f_lines), 3)

    def test_isosurface_extraction_and_smoothing(self):
        """Test watertight isosurface extraction from a 3D density field."""
        # Create a simple 6x6x6 voxel grid
        grid = VoxelMesher.create_box_grid(
            bounds=(0.0, 6.0, 0.0, 6.0, 0.0, 6.0),
            resolution=(6, 6, 6),
        )

        # Create a density cube in the center: elements with indices 1..4 in each direction
        density = np.zeros(grid.total_cells, dtype=np.float64)
        for k in range(1, 5):
            for j in range(1, 5):
                for i in range(1, 5):
                    cid = i + j * 6 + k * 36
                    density[cid] = 1.0

        mesh = extract_isosurface_mesh(
            grid,
            density,
            isovalue=0.5,
            smoothing_iters=2,
            smoothing_factor=0.3,
        )

        # 4x4x4 cube has 6 faces of 4x4 = 16 quads -> 96 quads = 192 triangles
        self.assertGreater(mesh.num_faces, 0)
        self.assertEqual(mesh.num_faces, 192)
        self.assertGreater(mesh.num_vertices, 0)

        # Verify all vertex coordinates are bounded within [0, 6]
        self.assertTrue(np.all(mesh.vertices >= 0.0))
        self.assertTrue(np.all(mesh.vertices <= 6.0))

    def test_paraview_vtu_and_macro_generation(self):
        """Test VTU file exporter and ParaView automated macro generator."""
        grid = VoxelMesher.create_box_grid(
            bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
            resolution=(4, 4, 4),
        )

        densities = np.full(grid.total_cells, 0.8)
        displacements = np.random.randn(grid.total_nodes * 3) * 1e-4
        von_mises = np.random.rand(grid.total_cells) * 1e6

        vtu_path = os.path.join(self.tmp_dir.name, "model.vtu")
        out_vtu = export_voxel_grid_vtu(
            grid=grid,
            filepath=vtu_path,
            displacements=displacements,
            densities=densities,
            von_mises=von_mises,
            threshold=0.1,
        )
        self.assertTrue(os.path.exists(out_vtu))

        with open(out_vtu, "r", encoding="utf-8") as fp:
            xml_text = fp.read()
            self.assertIn('<VTKFile type="UnstructuredGrid"', xml_text)
            self.assertIn('Name="Displacement"', xml_text)
            self.assertIn('Name="Density"', xml_text)
            self.assertIn('Name="VonMisesStress"', xml_text)

        # Macro generation
        py_path = os.path.join(self.tmp_dir.name, "load_paraview.py")
        out_py = generate_paraview_macro(
            vtu_filepath=out_vtu,
            output_py_path=py_path,
            warp_scale=100.0,
            color_by="VonMisesStress",
        )
        self.assertTrue(os.path.exists(out_py))
        with open(out_py, "r", encoding="utf-8") as fp:
            py_text = fp.read()
            self.assertIn("import paraview.simple", py_text)
            self.assertIn("WarpByVector", py_text)
            self.assertIn("VonMisesStress", py_text)

    def test_freecad_brep_solid_reconstruction(self):
        """Test B-Rep solid reconstruction and STEP export via FreeCAD 1.1 if installed."""
        freecad_cmd = FreeCADBRepDetector.find_freecad_cmd()
        if not freecad_cmd:
            self.skipTest("FreeCAD 1.1 executable not detected; skipping live B-Rep test.")

        # Create a small watertight box mesh
        vertices = np.array([
            [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0], [10.0, 0.0, 10.0], [10.0, 10.0, 10.0], [0.0, 10.0, 10.0],
        ], dtype=np.float64)
        # 12 triangles for 6 faces
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom (-Z)
            [4, 6, 5], [4, 7, 6],  # top (+Z)
            [0, 5, 1], [0, 4, 5],  # front (-Y)
            [3, 2, 6], [3, 6, 7],  # back (+Y)
            [0, 3, 7], [0, 7, 4],  # left (-X)
            [1, 5, 6], [1, 6, 2],  # right (+X)
        ], dtype=np.int64)
        mesh = TriangularMesh(vertices=vertices, faces=faces)

        reconstructor = BRepReconstructor(freecad_path=freecad_cmd)
        step_out = os.path.join(self.tmp_dir.name, "reconstructed_solid.step")

        # Synthetic cylindrical bolt hole
        bolt = CylindricalFeature(
            feature_id=0,
            face_index=0,
            is_internal=True,
            radius=1.5,
            diameter=3.0,
            axis=np.array([0.0, 0.0, 1.0]),
            center=np.array([5.0, 5.0, 5.0]),
            length=15.0,
            area=2.0 * np.pi * 1.5 * 15.0,
            min_proj=-2.5,
            max_proj=12.5,
        )

        res = reconstructor.reconstruct_step_solid(
            mesh=mesh,
            output_filepath=step_out,
            bolt_holes=[bolt],
            tolerance=0.05,
        )

        self.assertTrue(os.path.exists(res.step_filepath))
        self.assertGreater(os.path.getsize(res.step_filepath), 100)
        self.assertTrue(res.is_valid_solid)
        self.assertGreater(res.volume, 0.0)
        # Solid box 10x10x10 = 1000 minus cylinder pi * 1.5^2 * 10 = ~70.68 -> ~929
        self.assertAlmostEqual(res.volume, 1000.0 - np.pi * 1.5**2 * 10.0, delta=50.0)


if __name__ == "__main__":
    unittest.main()
