"""
Verification Suite: Phase 9 Matrix-Free Modal Dynamic Eigen-Solver & Octree AMR
-------------------------------------------------------------------------------
Validates:
1. Matrix-Free LOBPCG Structural Dynamics Modal Solver:
   - Natural frequencies and circular frequencies calculation.
   - Mode shape mass-orthonormality: phi_i^T * M * phi_j = delta_ij.
   - Comparison against Euler-Bernoulli analytical cantilever beam bending frequency.
   - Directional effective modal mass and mass participation ratios.
2. Hierarchical 1-Irregular Octree Adaptive Mesh Refinement (AMR):
   - 2:1 balancing constraint enforcement across adjacent refinement levels.
   - Conforming multi-point constraint (MPC) matrix generation (C).
   - Exact linear patch test interpolation on hanging nodes (C @ u_linear = u_exact).
3. ParaView Modal Animation State Generation:
   - XML UnstructuredGrid (.vtu) export with multi-frequency mode shape vectors.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tempfile
import unittest
import numpy as np

from wnfea.mesh.voxel_mesher import VoxelMesher
from wnfea.mesh.octree_amr import OctreeAMRMesher
from wnfea.solver.modal_analysis import solve_modal_analysis, compute_lumped_mass_hex8
from wnfea.results.paraview_export import export_modal_analysis_vtu, generate_modal_paraview_macro


class TestPhase9ModalAMR(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_lobpcg_modal_analysis_cantilever(self):
        """Verify natural frequencies, mode shapes, and M-orthonormality on a cantilever beam."""
        # Cantilever beam: L = 1.0 m, width = 0.05 m, height = 0.05 m
        # Grid: 20 x 2 x 2
        grid = VoxelMesher.create_box_grid(
            bounds=(0.0, 1.0, 0.0, 0.05, 0.0, 0.05),
            resolution=(20, 2, 2),
        )

        # Fix all DOFs at x = 0 (left face)
        fixed_dofs = []
        for j in range(3):
            for k in range(3):
                nid = grid.node_id(0, j, k)
                fixed_dofs.extend([nid * 3, nid * 3 + 1, nid * 3 + 2])

        E = 2.1e11       # Steel (210 GPa)
        nu = 0.3
        rho_mat = 7850.0 # Steel density (kg/m^3)
        num_modes = 4

        res = solve_modal_analysis(
            grid=grid,
            fixed_dofs=fixed_dofs,
            num_modes=num_modes,
            density_material=rho_mat,
            E=E,
            nu=nu,
            tol=1e-5,
            max_iter=50,
        )

        # 1. Monotonic frequencies > 0
        self.assertEqual(len(res.frequencies_hz), num_modes)
        for i in range(num_modes):
            self.assertGreater(res.frequencies_hz[i], 0.0)
            if i > 0:
                self.assertGreaterEqual(res.frequencies_hz[i], res.frequencies_hz[i - 1])

        # 2. Analytical Euler-Bernoulli first bending frequency check (~42 Hz)
        # 3D continuum beam with shear deformation is slightly more flexible than pure Euler-Bernoulli
        f1 = res.frequencies_hz[0]
        self.assertGreater(f1, 35.0)
        self.assertLess(f1, 55.0)

        # 3. Mass-orthonormality: phi_i^T * M * phi_j = delta_ij
        m_diag = compute_lumped_mass_hex8(grid, density_material=rho_mat)
        for i in range(num_modes):
            phi_i = res.mode_shapes[i].ravel()
            for j in range(num_modes):
                phi_j = res.mode_shapes[j].ravel()
                inner_prod = float(np.sum(phi_i * m_diag * phi_j))
                if i == j:
                    self.assertAlmostEqual(inner_prod, 1.0, delta=1e-3)
                else:
                    self.assertAlmostEqual(inner_prod, 0.0, delta=1e-2)

        # 4. Total mass and participation ratio consistency
        expected_mass = 1.0 * 0.05 * 0.05 * 7850.0  # 19.625 kg
        self.assertAlmostEqual(res.total_mass, expected_mass, delta=1e-3)
        self.assertTrue(np.all(res.mass_participation_ratio >= 0.0))
        self.assertTrue(np.all(res.mass_participation_ratio <= 1.0))

        # 5. ParaView VTU & Macro export verification
        vtu_path = os.path.join(self.tmp_dir.name, "modal_beam.vtu")
        export_modal_analysis_vtu(grid, vtu_path, res)
        self.assertTrue(os.path.exists(vtu_path))
        with open(vtu_path, "r", encoding="utf-8") as fp:
            xml_text = fp.read()
            self.assertIn("Mode_1_", xml_text)
            self.assertIn(f"Mode_1_{f1:.1f}Hz_Magnitude", xml_text)

        macro_path = os.path.join(self.tmp_dir.name, "animate_mode.py")
        generate_modal_paraview_macro(vtu_path, macro_path, f"Mode_1_{f1:.1f}Hz")
        self.assertTrue(os.path.exists(macro_path))
        with open(macro_path, "r", encoding="utf-8") as fp:
            macro_text = fp.read()
            self.assertIn("WarpByVector", macro_text)

    def test_octree_amr_2to1_balancing_and_mpc(self):
        """Verify 1-irregular octree refinement, hanging node detection, and MPC interpolation."""
        root_grid = VoxelMesher.create_box_grid(
            bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
            resolution=(2, 2, 2),  # 8 root cells
        )
        mesher = OctreeAMRMesher(root_grid)
        self.assertEqual(len(mesher.get_leaf_cells()), 8)

        # Refine cell 0 (corner cell)
        mesher.refine_cells([0], enforce_2to1_balance=True)
        # 1 parent cell subdivided into 8 children -> total leaves = 7 + 8 = 15
        leaves = mesher.get_leaf_cells()
        self.assertEqual(len(leaves), 15)

        # Build conforming AMR mesh with MPC constraints
        amr_mesh = mesher.build_conforming_mesh()
        self.assertEqual(amr_mesh.total_elements, 15)
        self.assertGreater(len(amr_mesh.hanging_node_indices), 0)
        self.assertEqual(
            amr_mesh.total_nodes,
            amr_mesh.total_true_nodes + len(amr_mesh.hanging_node_indices)
        )

        # Verify constraint matrix C (u = C @ u_true)
        C = amr_mesh.constraint_matrix
        self.assertEqual(C.shape[0], amr_mesh.total_nodes)
        self.assertEqual(C.shape[1], amr_mesh.total_true_nodes)

        # Patch test: Exact linear field u(x, y, z) = 2.0*x + 3.0*y - 1.5*z + 4.0
        # When sampled on independent nodes, C @ u_true must match analytical field on hanging nodes!
        true_nodes = amr_mesh.nodes[amr_mesh.independent_node_indices]
        u_true = 2.0 * true_nodes[:, 0] + 3.0 * true_nodes[:, 1] - 1.5 * true_nodes[:, 2] + 4.0

        u_interpolated = C.dot(u_true)
        u_exact_all = 2.0 * amr_mesh.nodes[:, 0] + 3.0 * amr_mesh.nodes[:, 1] - 1.5 * amr_mesh.nodes[:, 2] + 4.0

        max_interp_error = float(np.max(np.abs(u_interpolated - u_exact_all)))
        self.assertLess(max_interp_error, 1e-12)


if __name__ == "__main__":
    unittest.main()
