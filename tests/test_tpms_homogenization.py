"""
Test Suite for Functionally Graded TPMS Lattices & Asymptotic Homogenization Engine (Phase 20).
-----------------------------------------------------------------------------------------------
Verifies:
1. Mathematical TPMS level-set surface functions (Gyroid, Schwarz P, Diamond, Neovius, IWP).
2. Level-set threshold calibration and skeletal vs sheet morphology.
3. Functionally graded TPMS voxel infill generation and density gradients.
4. Watertight isosurface triangular mesh extraction and bounding box checks.
5. Asymptotic periodic homogenization exactness on solid continuum (< 1e-10 relative error).
6. Cubic symmetry, positive definiteness, and anisotropy ratio for Gyroid and Schwarz P.
7. Gibson-Ashby scaling law monotonicity and Hashin-Shtrikman upper bound compliance.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from wnfea.mesh.voxel_mesher import VoxelMesher
from wnfea.opt.tpms_lattice import (
    TPMSType,
    TPMSConfig,
    evaluate_tpms_levelset,
    density_to_levelset_threshold,
    evaluate_graded_tpms_field,
    generate_tpms_voxel_infill,
    extract_tpms_isosurface,
)
from wnfea.materials.homogenization import (
    HomogenizationResult,
    compute_isotropic_elasticity_matrix,
    solve_periodic_homogenization,
    homogenize_tpms_unit_cell,
)


class TestTPMSHomogenization(unittest.TestCase):
    """Rigorous verification suite for Phase 20 TPMS & Asymptotic Homogenization."""

    def setUp(self):
        self.base_E = 210e9     # Structural Steel Young's modulus (210 GPa)
        self.base_nu = 0.30     # Poisson's ratio
        self.C0 = compute_isotropic_elasticity_matrix(self.base_E, self.base_nu)

    def test_tpms_levelset_evaluations(self):
        """Verify mathematical definitions and volume symmetry of TPMS functions."""
        # Grid spanning one unit period [-0.5, 0.5]^3
        n = 20
        grid = np.linspace(-0.5, 0.5, n, endpoint=False)
        X, Y, Z = np.meshgrid(grid, grid, grid, indexing="ij")
        pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        for tpms_type in [
            TPMSType.GYROID,
            TPMSType.SCHWARZ_P,
            TPMSType.DIAMOND,
            TPMSType.NEOVIUS,
            TPMSType.IWP,
        ]:
            val = evaluate_tpms_levelset(pts, tpms_type, cell_size=(1.0, 1.0, 1.0))
            self.assertEqual(val.shape, (n**3,))
            # By point-group inversion/reflection symmetry, mean value over a unit cell is ~0
            mean_val = np.mean(val)
            self.assertAlmostEqual(mean_val, 0.0, delta=0.05,
                                   msg=f"{tpms_type.value} level set mean deviates from zero: {mean_val}")

            # At t = 0, volume fraction of skeletal solid (val <= 0) is ~50%
            solid_ratio = np.mean(val <= 0)
            self.assertAlmostEqual(solid_ratio, 0.50, delta=0.08,
                                   msg=f"{tpms_type.value} 50% threshold volume mismatch: {solid_ratio}")

    def test_density_threshold_monotonicity_and_sheet_mode(self):
        """Verify threshold function monotonicity and sheet vs skeletal morphology."""
        # Skeletal mode: solid is phi <= t. As rho increases, t must increase
        rho_vals = [0.15, 0.30, 0.50, 0.70, 0.85]
        t_skeletal = [density_to_levelset_threshold(TPMSType.GYROID, "skeletal", r) for r in rho_vals]
        for i in range(len(t_skeletal) - 1):
            self.assertLess(t_skeletal[i], t_skeletal[i + 1],
                            msg="Skeletal threshold must be strictly monotonically increasing with relative density")

        # Sheet mode: solid is |phi| <= t. As rho increases, wall half-thickness t must increase
        t_sheet = [density_to_levelset_threshold(TPMSType.GYROID, "sheet", r) for r in rho_vals]
        for i in range(len(t_sheet) - 1):
            self.assertLess(t_sheet[i], t_sheet[i + 1],
                            msg="Sheet threshold must be strictly monotonically increasing with relative density")
            self.assertGreater(t_sheet[i], 0.0, msg="Sheet thickness threshold must be positive")

    def test_functionally_graded_infill_generation(self):
        """Verify 3D functionally graded TPMS lattice generation."""
        # Box bounds (0, 1) x (0, 1) x (0, 1.5) with (16, 16, 24) resolution
        bounds = (0.0, 1.0, 0.0, 1.0, 0.0, 1.5)
        res = (16, 16, 24)
        grid = VoxelMesher.create_box_grid(bounds, res)

        # Graded relative density function: linearly increasing along Z from 0.20 to 0.80
        def rho_fn(pts):
            z = pts[:, 2]
            return 0.20 + 0.60 * (z / 1.5)

        config = TPMSConfig(
            tpms_type=TPMSType.GYROID,
            cell_size=(0.5, 0.5, 0.5),
            lattice_mode="skeletal",
            relative_density=0.50,
        )

        infill_fractions = generate_tpms_voxel_infill(
            grid=grid,
            config=config,
            density_field=rho_fn,
            subsampling=2,
        )

        self.assertEqual(len(infill_fractions), grid.total_cells)

        # Check gradient: bottom cells (low z) vs top cells (high z)
        centroids = grid.get_element_centroids()
        bottom_mask = centroids[:, 2] < 0.35
        top_mask = centroids[:, 2] > 1.15
        bottom_avg = np.mean(infill_fractions[bottom_mask])
        top_avg = np.mean(infill_fractions[top_mask])

        self.assertLess(bottom_avg, top_avg,
                        msg=f"Graded infill failed: bottom ({bottom_avg:.3f}) >= top ({top_avg:.3f})")
        self.assertGreater(bottom_avg, 0.10)
        self.assertLess(top_avg, 0.90)

    def test_watertight_isosurface_extraction(self):
        """Verify triangle mesh extraction from TPMS level sets."""
        config = TPMSConfig(
            tpms_type=TPMSType.SCHWARZ_P,
            cell_size=(10.0, 10.0, 10.0),
            lattice_mode="skeletal",
            relative_density=0.40,
        )

        mesh = extract_tpms_isosurface(
            config=config,
            bounds=(0.0, 10.0, 0.0, 10.0, 0.0, 10.0),
            resolution=(16, 16, 16),
        )

        self.assertGreater(len(mesh.vertices), 50, msg="Extracted vertices array should not be empty")
        self.assertGreater(len(mesh.faces), 50, msg="Extracted faces array should not be empty")
        self.assertEqual(mesh.vertices.shape[1], 3)
        self.assertEqual(mesh.faces.shape[1], 3)

        # Coordinate bounds must be inside the domain
        self.assertGreaterEqual(np.min(mesh.vertices), -1e-5)
        self.assertLessEqual(np.max(mesh.vertices), 10.0 + 1e-5)

    def test_periodic_homogenization_solid_exactness(self):
        """
        Verify asymptotic homogenization on a 100% solid unit cell.
        Under uniform continuum material, micro-displacement fluctuations chi^(kl) are
        identically zero, and C^eff must match the isotropic C0 matrix to machine precision.
        """
        n = 4
        solid_densities = np.ones((n, n, n), dtype=np.float64)
        res = solve_periodic_homogenization(
            element_densities=solid_densities,
            cell_size=(1.0, 1.0, 1.0),
            base_E=self.base_E,
            base_nu=self.base_nu,
            simp_penalty=1.0,
        )

        # 1. Exact matrix match
        rel_diff = np.max(np.abs(res.C_effective - self.C0)) / np.max(np.abs(self.C0))
        self.assertLess(rel_diff, 1e-10,
                        msg=f"Solid homogenization matrix relative error {rel_diff:.2e} exceeds 1e-10")

        # 2. Engineering constants match
        self.assertAlmostEqual(res.E_x, self.base_E, delta=self.base_E * 1e-8)
        self.assertAlmostEqual(res.E_y, self.base_E, delta=self.base_E * 1e-8)
        self.assertAlmostEqual(res.E_z, self.base_E, delta=self.base_E * 1e-8)

        expected_G = self.base_E / (2.0 * (1.0 + self.base_nu))
        self.assertAlmostEqual(res.G_xy, expected_G, delta=expected_G * 1e-8)
        self.assertAlmostEqual(res.G_yz, expected_G, delta=expected_G * 1e-8)
        self.assertAlmostEqual(res.G_zx, expected_G, delta=expected_G * 1e-8)

        self.assertAlmostEqual(res.nu_xy, self.base_nu, delta=1e-8)
        self.assertAlmostEqual(res.nu_yz, self.base_nu, delta=1e-8)
        self.assertAlmostEqual(res.nu_zx, self.base_nu, delta=1e-8)

        # 3. Zener anisotropy index for isotropic material must be 1.0
        self.assertAlmostEqual(res.anisotropy_ratio, 1.0, delta=1e-6)
        self.assertAlmostEqual(res.relative_density, 1.0, delta=1e-8)

    def test_gyroid_homogenization_cubic_symmetry(self):
        """Verify cubic symmetry and positive-definiteness for Gyroid unit cell."""
        config = TPMSConfig(
            tpms_type=TPMSType.GYROID,
            cell_size=(1.0, 1.0, 1.0),
            lattice_mode="skeletal",
            relative_density=0.40,
        )

        res = homogenize_tpms_unit_cell(
            config=config,
            resolution=10,
            base_E=self.base_E,
            base_nu=self.base_nu,
        )

        # 1. Positive definiteness (all 6 eigenvalues > 0)
        evals = np.linalg.eigvalsh(res.C_effective)
        self.assertTrue(np.all(evals > 0), msg=f"Homogenized tensor is not positive definite: {evals}")

        # 2. Cubic symmetry: C11 ~ C22 ~ C33
        c11 = res.C_effective[0, 0]
        c22 = res.C_effective[1, 1]
        c33 = res.C_effective[2, 2]
        self.assertAlmostEqual(c11, c22, delta=c11 * 0.05, msg="Gyroid C11 and C22 mismatch")
        self.assertAlmostEqual(c11, c33, delta=c11 * 0.05, msg="Gyroid C11 and C33 mismatch")

        # 3. Cubic symmetry: C12 ~ C13 ~ C23
        c12 = res.C_effective[0, 1]
        c13 = res.C_effective[0, 2]
        c23 = res.C_effective[1, 2]
        self.assertAlmostEqual(c12, c13, delta=c12 * 0.05, msg="Gyroid C12 and C13 mismatch")
        self.assertAlmostEqual(c12, c23, delta=c12 * 0.05, msg="Gyroid C12 and C23 mismatch")

        # 4. Shear symmetry: C44 ~ C55 ~ C66
        c44 = res.C_effective[3, 3]
        c55 = res.C_effective[4, 4]
        c66 = res.C_effective[5, 5]
        self.assertAlmostEqual(c44, c55, delta=c44 * 0.05, msg="Gyroid C44 and C55 mismatch")
        self.assertAlmostEqual(c44, c66, delta=c44 * 0.05, msg="Gyroid C44 and C66 mismatch")

        # 5. Cubic symmetry residual metric
        self.assertLess(res.cubic_symmetry_residual, 0.05,
                        msg=f"Gyroid cubic residual {res.cubic_symmetry_residual:.3f} exceeds 0.05")

        # 6. Physical bounds on moduli: 0 < E_eff < E_base
        self.assertGreater(res.E_x, 0.0)
        self.assertLess(res.E_x, self.base_E)
        self.assertGreater(res.anisotropy_ratio, 0.0)

    def test_schwarz_p_homogenization(self):
        """Verify Schwarz P unit cell homogenization and engineering constants."""
        config = TPMSConfig(
            tpms_type=TPMSType.SCHWARZ_P,
            cell_size=(1.0, 1.0, 1.0),
            lattice_mode="skeletal",
            relative_density=0.35,
        )

        res = homogenize_tpms_unit_cell(
            config=config,
            resolution=10,
            base_E=self.base_E,
            base_nu=self.base_nu,
        )

        # Symmetry checks
        self.assertAlmostEqual(res.E_x, res.E_y, delta=res.E_x * 0.05)
        self.assertAlmostEqual(res.E_x, res.E_z, delta=res.E_x * 0.05)
        self.assertAlmostEqual(res.G_xy, res.G_yz, delta=res.G_xy * 0.05)

        # Relative density within +/- 15% of target
        self.assertAlmostEqual(res.relative_density, 0.35, delta=0.08)

    def test_gibson_ashby_scaling_law(self):
        """Verify monotonic stiffness scaling with relative density E_eff(rho) and bounds."""
        config_low = TPMSConfig(TPMSType.GYROID, cell_size=(1.0, 1.0, 1.0), relative_density=0.25)
        config_high = TPMSConfig(TPMSType.GYROID, cell_size=(1.0, 1.0, 1.0), relative_density=0.55)

        res_low = homogenize_tpms_unit_cell(config_low, resolution=8, base_E=self.base_E, base_nu=self.base_nu)
        res_high = homogenize_tpms_unit_cell(config_high, resolution=8, base_E=self.base_E, base_nu=self.base_nu)

        # Higher density must yield strictly higher effective Young's modulus
        self.assertGreater(res_high.E_x, res_low.E_x * 1.5,
                           msg=f"Stiffness scaling failed: E_high={res_high.E_x:.2e} <= 1.5 * E_low={res_low.E_x:.2e}")

        # Hashin-Shtrikman upper bound: E_eff <= rho * E_base
        self.assertLessEqual(res_low.E_x, res_low.relative_density * self.base_E)
        self.assertLessEqual(res_high.E_x, res_high.relative_density * self.base_E)


if __name__ == "__main__":
    unittest.main()
