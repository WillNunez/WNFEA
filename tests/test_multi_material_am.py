"""
Unit and Integration Tests for Phase 16:
Multi-Material SIMP Topology Optimization & Additive Manufacturing Overhang Filter.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from wnfea.mesh.voxel_mesher import VoxelMesher
from wnfea.opt.multi_material import (
    MaterialProperty,
    MultiMaterialConfig,
    MultiMaterialResult,
    MultiMaterialSIMPInterpolator,
    MultiMaterialTopologyOptimizer,
    project_simplex_batch,
    ALUMINUM_6061,
    TITANIUM_TI6AL4V,
    STEEL_STRUCTURAL,
)
from wnfea.opt.am_overhang import (
    AMOverhangFilter,
    AMFilterResult,
)


class TestMultiMaterialAM(unittest.TestCase):
    """
    Verification test suite for multi-material topology optimization and AM overhang filter.
    """

    def test_01_simplex_projection(self):
        """Verify vectorized Euclidean projection onto unit simplex Delta = {x >= 0, sum x <= 1}."""
        # 1. Point strictly inside
        v_in = np.array([[0.2, 0.3, 0.4]])
        p_in = project_simplex_batch(v_in)
        np.testing.assert_allclose(p_in, v_in, atol=1e-14)
        self.assertLessEqual(np.sum(p_in), 1.0)
        self.assertTrue(np.all(p_in >= 0.0))

        # 2. Point violating sum <= 1
        v_out = np.array([[0.8, 0.8]])
        p_out = project_simplex_batch(v_out)
        self.assertAlmostEqual(float(np.sum(p_out)), 1.0, places=12)
        np.testing.assert_allclose(p_out, [[0.5, 0.5]], atol=1e-12)

        # 3. Point with negative components
        v_neg = np.array([[-0.5, 0.7, 0.6]])
        p_neg = project_simplex_batch(v_neg)
        self.assertAlmostEqual(float(np.sum(p_neg)), 1.0, places=12)
        self.assertEqual(p_neg[0, 0], 0.0)
        self.assertTrue(np.all(p_neg >= 0.0))

        # 4. Batch of random points
        np.random.seed(42)
        v_batch = np.random.uniform(-0.5, 1.5, size=(100, 4))
        p_batch = project_simplex_batch(v_batch)
        self.assertTrue(np.all(p_batch >= -1e-14))
        self.assertTrue(np.all(np.sum(p_batch, axis=1) <= 1.0 + 1e-12))

    def test_02_multi_material_sensitivities(self):
        """Verify multi-material SIMP interpolation & adjoint compliance sensitivity vs finite difference."""
        materials = [ALUMINUM_6061, TITANIUM_TI6AL4V]
        p = 3.0
        interp = MultiMaterialSIMPInterpolator(materials, simp_penalty=p)

        # Test state
        rho = np.array([[0.35, 0.45]], dtype=np.float64)
        U_0 = np.array([125.0], dtype=np.float64) # reference strain energy

        # Analytical sensitivities
        dc_drho = interp.compute_compliance_sensitivities(rho, U_0) # (1, 2)
        dM_drho = interp.compute_mass_sensitivities(1, elem_volume=0.001) # (1, 2)

        eps = 1e-6
        for m in range(2):
            # Compliance sensitivity check
            rho_plus = rho.copy()
            rho_minus = rho.copy()
            rho_plus[0, m] += eps
            rho_minus[0, m] -= eps

            E_plus = interp.compute_effective_modulus(rho_plus)
            E_minus = interp.compute_effective_modulus(rho_minus)

            # c = E_eff * U_0
            c_plus = float(E_plus[0] * U_0[0])
            c_minus = float(E_minus[0] * U_0[0])

            # In linear elasticity, dc/drho = - d(strain_energy)/drho
            # For fixed displacement (adjoint state), dc/drho = - d(E)/drho * U_0
            dE_fd = (E_plus[0] - E_minus[0]) / (2.0 * eps)
            expected_dc_drho_m = - dE_fd * U_0[0]

            rel_err_c = abs(dc_drho[0, m] - expected_dc_drho_m) / abs(expected_dc_drho_m)
            self.assertLess(rel_err_c, 1e-5, f"Compliance sensitivity mismatch for material {m}")

            # Mass sensitivity check
            M_plus = interp.compute_total_mass(rho_plus, 0.001)
            M_minus = interp.compute_total_mass(rho_minus, 0.001)
            dM_fd = (M_plus - M_minus) / (2.0 * eps)
            rel_err_M = abs(dM_drho[0, m] - dM_fd) / abs(dM_fd)
            self.assertLess(rel_err_M, 1e-9, f"Mass sensitivity mismatch for material {m}")

    def test_03_multi_material_optimization_convergence(self):
        """Verify multi-material topology optimization on a 3D cantilever beam."""
        L, W, H = 1.0, 0.2, 0.2
        grid = VoxelMesher.create_box_grid((0.0, L, 0.0, W, 0.0, H), (10, 2, 2))

        # Clamp at x=0
        fixed_dofs = []
        for node_idx, pt in enumerate(grid.nodes):
            if abs(pt[0]) < 1e-5:
                fixed_dofs.extend([node_idx * 3, node_idx * 3 + 1, node_idx * 3 + 2])

        # Downward point load at tip (x=L, z=0)
        forces = np.zeros(len(grid.nodes) * 3)
        tip_nodes = [i for i, pt in enumerate(grid.nodes) if abs(pt[0] - L) < 1e-5]
        for tn in tip_nodes:
            forces[tn * 3 + 2] = -1000.0 / len(tip_nodes)

        materials = [ALUMINUM_6061, TITANIUM_TI6AL4V]
        config = MultiMaterialConfig(
            target_mass_fraction=0.40,
            simp_penalty=3.0,
            max_iterations=12,
            filter_radius=0.15,
            move_limit=0.20,
            verbose=False,
        )

        opt = MultiMaterialTopologyOptimizer(grid, forces, fixed_dofs, materials, config)
        res = opt.optimize()

        # Check convergence & properties
        self.assertGreater(res.total_iterations, 2)
        self.assertLessEqual(res.total_iterations, 12)
        self.assertGreater(res.final_compliance, 0.0)
        self.assertLessEqual(res.mass_fraction, 0.45) # Within budget

        # Check that densities satisfy simplex bounds everywhere
        self.assertTrue(np.all(res.optimized_densities >= -1e-12))
        self.assertTrue(np.all(np.sum(res.optimized_densities, axis=1) <= 1.0 + 1e-8))

        # Material differentiation: titanium is stiffer than aluminum,
        # so titanium should be placed at the high-stress root elements
        centroids = grid.get_element_centroids()
        root_elements = np.where(centroids[:, 0] < 0.2)[0]
        ti_at_root = np.sum(res.optimized_densities[root_elements, 1])
        self.assertGreater(ti_at_root, 0.0, "Titanium should be utilized at high bending stress root")

    def test_04_am_overhang_detection_and_projection(self):
        """Verify AM overhang filter on self-supporting vs unsupported features."""
        grid = VoxelMesher.create_box_grid((0.0, 1.0, 0.0, 1.0, 0.0, 1.0), (4, 4, 4))
        am = AMOverhangFilter(grid, build_direction="+z", critical_angle_deg=45.0)

        # Case A: Solid vertical column anchored to base plate (z=0)
        rho_col = np.zeros(64)
        rho_3d = rho_col.reshape((4, 4, 4))
        rho_3d[0, :, :] = 1.0          # Base plate solid
        rho_3d[1:4, 1:3, 1:3] = 1.0    # 2x2 column rising in Z
        diag_col = am.analyze_printability(rho_col)
        self.assertTrue(diag_col.is_printable)
        self.assertEqual(diag_col.overhang_penalty, 0.0)
        self.assertEqual(diag_col.unsupported_voxel_count, 0)

        # Case B: Unsupported horizontal overhang at layer 2 with void at layer 1
        rho_overhang = np.zeros(64)
        rho_3d = rho_overhang.reshape((4, 4, 4))
        rho_3d[0, 0:2, 0:2] = 1.0      # Base support at bottom left
        rho_3d[1, 0:2, 0:2] = 1.0      # Layer 1 support at bottom left
        rho_3d[2, 2:4, 2:4] = 1.0      # Layer 2 floating overhang at top right (completely unsupported!)
        diag_overhang = am.analyze_printability(rho_overhang)
        self.assertFalse(diag_overhang.is_printable)
        self.assertGreater(diag_overhang.overhang_penalty, 0.0)
        self.assertEqual(diag_overhang.unsupported_voxel_count, 4)

        # Case C: Forward printability filter removes unsupported ceiling
        filtered_rho = am.filter_am_densities(rho_overhang)
        diag_filtered = am.analyze_printability(filtered_rho)
        self.assertTrue(diag_filtered.is_printable)
        self.assertEqual(diag_filtered.overhang_penalty, 0.0)
        self.assertEqual(diag_filtered.unsupported_voxel_count, 0)

    def test_05_am_overhang_gradient_parity(self):
        """Verify analytical adjoint gradient of AM overhang penalty matches finite difference."""
        grid = VoxelMesher.create_box_grid((0.0, 1.0, 0.0, 1.0, 0.0, 1.0), (3, 3, 3))
        am = AMOverhangFilter(grid, build_direction="+z", critical_angle_deg=45.0)

        np.random.seed(123)
        rho = np.random.uniform(0.1, 0.9, size=27)

        # Analytical penalty and gradient
        penalty_ana, grad_ana = am.evaluate_overhang_penalty(rho)

        eps = 1e-6
        # Check a few elements across layers
        for e_idx in [5, 10, 15, 20]:
            rho_p = rho.copy()
            rho_m = rho.copy()
            rho_p[e_idx] += eps
            rho_m[e_idx] -= eps

            p_p, _ = am.evaluate_overhang_penalty(rho_p)
            p_m, _ = am.evaluate_overhang_penalty(rho_m)

            grad_fd = (p_p - p_m) / (2.0 * eps)
            # Compare analytical vs numerical gradient
            diff = abs(grad_ana[e_idx] - grad_fd)
            # If gradient is non-zero, check relative error, otherwise absolute
            if abs(grad_fd) > 1e-4:
                rel_err = diff / abs(grad_fd)
                self.assertLess(rel_err, 0.05, f"Gradient mismatch at element {e_idx}")
            else:
                self.assertLess(diff, 1e-4, f"Gradient mismatch at element {e_idx}")

    def test_06_multi_material_am_coupled_optimization(self):
        """Verify coupled multi-material topology optimization with AM overhang constraint."""
        L, W, H = 0.6, 0.2, 0.2
        grid = VoxelMesher.create_box_grid((0.0, L, 0.0, W, 0.0, H), (6, 2, 2))

        # Clamp at bottom x=0
        fixed_dofs = []
        for node_idx, pt in enumerate(grid.nodes):
            if abs(pt[0]) < 1e-5:
                fixed_dofs.extend([node_idx * 3, node_idx * 3 + 1, node_idx * 3 + 2])

        forces = np.zeros(len(grid.nodes) * 3)
        tip_nodes = [i for i, pt in enumerate(grid.nodes) if abs(pt[0] - L) < 1e-5]
        for tn in tip_nodes:
            forces[tn * 3 + 2] = -500.0 / len(tip_nodes)

        materials = [ALUMINUM_6061, TITANIUM_TI6AL4V]
        config = MultiMaterialConfig(
            target_mass_fraction=0.45,
            simp_penalty=3.0,
            max_iterations=8,
            filter_radius=0.10,
            enable_am_filter=True,
            am_build_direction="+z",
            am_critical_angle_deg=45.0,
            verbose=False,
        )

        opt = MultiMaterialTopologyOptimizer(grid, forces, fixed_dofs, materials, config)
        res = opt.optimize()

        self.assertIsNotNone(res.am_is_printable)
        self.assertTrue(res.am_is_printable, "Optimized multi-material design should be printable under AM filter")
        self.assertGreater(res.total_iterations, 2)

    def test_07_am_multiple_build_directions(self):
        """Verify AM overhang filter operates seamlessly across multiple build orientations."""
        grid = VoxelMesher.create_box_grid((0.0, 1.0, 0.0, 1.0, 0.0, 1.0), (4, 4, 4))
        
        for b_dir in ["+z", "-z", "+y", "-y", "+x", "-x"]:
            am = AMOverhangFilter(grid, build_direction=b_dir, critical_angle_deg=45.0)
            rho = np.random.uniform(0.0, 1.0, size=64)
            
            # Filter densities
            p_rho = am.filter_am_densities(rho)
            diag = am.analyze_printability(p_rho)
            
            self.assertTrue(diag.is_printable, f"Printability filter failed for direction {b_dir}")
            self.assertAlmostEqual(diag.overhang_penalty, 0.0, places=10)
            self.assertEqual(diag.unsupported_voxel_count, 0)


if __name__ == "__main__":
    unittest.main()

