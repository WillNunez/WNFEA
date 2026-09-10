"""
Test Suite for Aero-Structural Fatigue Life & Cyclic Damage Estimation Engine (Phase 18).
---------------------------------------------------------------------------------------
Verifies:
1. Turning point (peaks & valleys) extraction and filtering.
2. ASTM E1049-85 Section 5.4.4 Rainflow cycle counting standard benchmark.
3. Basquin S-N curve and Goodman, Gerber, Morrow, Soderberg mean stress corrections.
4. Linear Palmgren-Miner cumulative damage accumulation.
5. Element-wise fatigue life and damage field evaluation across FEA domains.
6. Multiaxial Critical Plane analysis (Findley, Fatemi-Socie, SWT) under proportional
   and non-proportional loading.
7. Dang Van mesoscopic multiaxial fatigue limit criterion and hydrostatic sensitivity.
8. ParaView XML VTU export with fatigue damage and logarithmic life fields.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from wnfea.fatigue.fatigue_solver import (
    RainflowCycle,
    FatigueMaterial,
    extract_peaks_valleys,
    count_rainflow_cycles,
    evaluate_sn_life,
    evaluate_palmgren_miner_damage,
    evaluate_element_fatigue_life,
)
from wnfea.fatigue.critical_plane import (
    CriticalPlaneResult,
    evaluate_critical_plane,
    evaluate_dang_van_safety_factor,
)
from wnfea.mesh.voxel_mesher import VoxelGrid
from wnfea.results.paraview_export import export_voxel_grid_vtu


class TestFatigueLife(unittest.TestCase):
    """Rigorous verification suite for Phase 18 Fatigue Engine."""

    def setUp(self):
        self.steel = FatigueMaterial.structural_steel_s355()
        self.aluminum = FatigueMaterial.aluminum_7075_t6()
        self.titanium = FatigueMaterial.titanium_ti6al4v()

    def test_extract_peaks_valleys(self):
        """Verify reversal filter discards flat/monotonic points and keeps extrema."""
        # Continuous sinusoidal wave sampled at fine intervals
        t = np.linspace(0, 4 * np.pi, 200)
        signal = 100.0 * np.sin(t)
        pv = extract_peaks_valleys(signal)

        # In 2 full sine waves [0, 4*pi], there are 2 peaks (+100) and 2 valleys (-100)
        # Plus endpoints at t=0 and t=4*pi (signal=0). Total extrema = 6.
        self.assertLessEqual(len(pv), 7)
        self.assertGreaterEqual(len(pv), 5)
        # Peaks should be ~100 and valleys ~ -100
        self.assertAlmostEqual(np.max(pv), 100.0, delta=0.5)
        self.assertAlmostEqual(np.min(pv), -100.0, delta=0.5)

        # Duplicate points removal
        flat_signal = [0.0, 5.0, 5.0, 5.0, 10.0, 10.0, 3.0, 3.0, 8.0]
        pv_flat = extract_peaks_valleys(flat_signal)
        self.assertEqual(list(pv_flat), [0.0, 10.0, 3.0, 8.0])

    def test_astm_e1049_rainflow_benchmark(self):
        """
        Verify exact ASTM E1049-85 Section 5.4.4 Fig. 8 cycle count standard benchmark.
        Standard sequence: [-2, 3, -4, 4, -2, 2, -3, 5, -5, 3]
        """
        seq = [-2.0, 3.0, -4.0, 4.0, -2.0, 2.0, -3.0, 5.0, -5.0, 3.0]
        cycles = count_rainflow_cycles(seq, close_residuals=False)

        # Expected ASTM standard results:
        # Full cycles (count = 1.0):
        #   Range 4.0 (from -2 to 2), mean 0.0
        #   Range 7.0 (from -3 to 4), mean 0.5
        # Half cycles (count = 0.5):
        #   Range 5.0, mean 0.5
        #   Range 7.0, mean -0.5
        #   Range 9.0, mean 0.5
        #   Range 8.0, mean -1.0
        #   Range 10.0, mean 0.0
        full_cycles = [c for c in cycles if c.count == 1.0]
        self.assertEqual(len(full_cycles), 2)

        ranges_full = sorted([c.range for c in full_cycles])
        self.assertAlmostEqual(ranges_full[0], 4.0, places=5)
        self.assertAlmostEqual(ranges_full[1], 7.0, places=5)

        means_full = sorted([c.mean for c in full_cycles])
        self.assertAlmostEqual(means_full[0], 0.0, places=5)
        self.assertAlmostEqual(means_full[1], 0.5, places=5)

        total_cycles = sum(c.count for c in cycles)
        self.assertAlmostEqual(total_cycles, 4.5, places=5)

    def test_sn_curve_and_mean_stress_corrections(self):
        """Verify analytical Basquin S-N curve formula and mean stress corrections."""
        mat = self.steel  # sigma_f' = 950 MPa, b = -0.09, sigma_ut = 510 MPa, sigma_y = 355 MPa
        S_a = 400e6  # 400 MPa amplitude

        # 1. Uncorrected (fully reversed, S_m = 0):
        # Basquin: N_f = 0.5 * (S_a / sigma_f')^(1/b)
        expected_Nf = 0.5 * ((S_a / mat.sigma_f_prime) ** (1.0 / mat.b))
        actual_Nf = evaluate_sn_life(S_a, 0.0, mat, correction="none")
        self.assertAlmostEqual(actual_Nf, expected_Nf, delta=expected_Nf * 1e-4)

        # 2. Goodman correction with tensile mean stress:
        # S_m = 100 MPa ==> S_eq = 400 / (1 - 100/510) = 497.56 MPa
        S_m = 100e6
        expected_S_eq_goodman = S_a / (1.0 - S_m / mat.sigma_ut)
        expected_Nf_goodman = 0.5 * ((expected_S_eq_goodman / mat.sigma_f_prime) ** (1.0 / mat.b))
        actual_Nf_goodman = evaluate_sn_life(S_a, S_m, mat, correction="goodman")
        self.assertAlmostEqual(actual_Nf_goodman, expected_Nf_goodman, delta=expected_Nf_goodman * 1e-4)

        # 3. Hierarchy of mean stress conservatism:
        # Soderberg is most conservative (uses sigma_y), then Goodman (sigma_ut), then Gerber (quadratic).
        nf_soderberg = evaluate_sn_life(S_a, S_m, mat, correction="soderberg")
        nf_goodman = evaluate_sn_life(S_a, S_m, mat, correction="goodman")
        nf_gerber = evaluate_sn_life(S_a, S_m, mat, correction="gerber")
        nf_none = evaluate_sn_life(S_a, S_m, mat, correction="none")

        self.assertLess(nf_soderberg, nf_goodman)
        self.assertLess(nf_goodman, nf_gerber)
        self.assertLess(nf_gerber, nf_none)

        # 4. Compressive mean stress does not reduce life in Goodman model:
        nf_compressive = evaluate_sn_life(S_a, -100e6, mat, correction="goodman")
        self.assertAlmostEqual(nf_compressive, nf_none, places=1)

        # 5. Endurance limit cutoff:
        nf_below_endurance = evaluate_sn_life(150e6, 0.0, mat, correction="goodman")
        self.assertEqual(nf_below_endurance, mat.cut_off_cycles)

    def test_palmgren_miner_damage_accumulation(self):
        """Verify linear damage summation D = sum(n_i / N_i)."""
        mat = self.steel
        # Block of cycles:
        # 500 cycles at S_a = 450 MPa (N_1)
        # 2000 cycles at S_a = 350 MPa (N_2)
        sa_1, sa_2 = 450e6, 350e6
        n_1, n_2 = 500.0, 2000.0

        n_f1 = evaluate_sn_life(sa_1, 0.0, mat, correction="none")
        n_f2 = evaluate_sn_life(sa_2, 0.0, mat, correction="none")

        expected_d = (n_1 / n_f1) + (n_2 / n_f2)
        expected_life = (1.0 / expected_d) * (n_1 + n_2)

        cycles = [
            RainflowCycle(range=2.0 * sa_1, mean=0.0, count=n_1),
            RainflowCycle(range=2.0 * sa_2, mean=0.0, count=n_2),
        ]
        actual_d, actual_life = evaluate_palmgren_miner_damage(cycles, mat, correction="none")

        self.assertAlmostEqual(actual_d, expected_d, places=7)
        self.assertAlmostEqual(actual_life, expected_life, places=1)

    def test_element_fatigue_life_field(self):
        """Verify mesh-wide fatigue life calculation on multi-element stress histories."""
        n_elem = 20
        n_steps = 50
        time = np.linspace(0, 2 * np.pi, n_steps)

        # Elements with linearly increasing stress amplitude from 100 MPa to 500 MPa
        amps = np.linspace(100e6, 500e6, n_elem)
        hist = np.zeros((n_elem, n_steps), dtype=np.float64)
        for e in range(n_elem):
            hist[e] = amps[e] * np.sin(time)

        damage, life = evaluate_element_fatigue_life(hist, self.steel, correction="goodman")

        self.assertEqual(len(damage), n_elem)
        self.assertEqual(len(life), n_elem)

        # Elements with stress <= endurance limit (220 MPa) must have infinite life
        for e in range(n_elem):
            if amps[e] <= self.steel.endurance_limit:
                self.assertEqual(damage[e], 0.0)
                self.assertEqual(life[e], self.steel.cut_off_cycles)

        # Peak stress element must have highest damage and lowest life
        self.assertGreater(damage[-1], 0.0)
        self.assertLess(life[-1], life[0])
        self.assertAlmostEqual(damage[np.argmax(damage)], damage[-1])

    def test_critical_plane_multiaxial(self):
        """Verify multiaxial critical plane analysis for uniaxial, torsion, and 90-deg out-of-phase loading."""
        n_steps = 60
        t = np.linspace(0, 2 * np.pi, n_steps, endpoint=False)

        # 1. Pure uniaxial tension along X: sigma_xx(t) = sigma_0 * sin(t)
        sigma_0 = 300e6
        uniaxial_hist = np.zeros((n_steps, 6))
        uniaxial_hist[:, 0] = sigma_0 * np.sin(t)

        # In pure uniaxial tension, maximum shear stress occurs at 45 degrees to X axis
        res_uniaxial = evaluate_critical_plane(uniaxial_hist, criterion="findley", k_param=0.0)
        self.assertAlmostEqual(res_uniaxial.tau_amplitude, 0.5 * sigma_0, delta=sigma_0 * 0.05)

        # 2. Pure torsion in XY: tau_xy(t) = tau_0 * sin(t)
        tau_0 = 200e6
        torsion_hist = np.zeros((n_steps, 6))
        torsion_hist[:, 3] = tau_0 * np.sin(t)

        res_torsion = evaluate_critical_plane(torsion_hist, criterion="findley", k_param=0.0)
        self.assertAlmostEqual(res_torsion.tau_amplitude, tau_0, delta=tau_0 * 0.05)

        # 3. Non-proportional 90-degree out-of-phase loading:
        # sigma_xx(t) = sigma_0 * sin(t), tau_xy(t) = sigma_0 * cos(t)
        nonprop_hist = np.zeros((n_steps, 6))
        nonprop_hist[:, 0] = sigma_0 * np.sin(t)
        nonprop_hist[:, 3] = sigma_0 * np.cos(t)

        res_nonprop = evaluate_critical_plane(nonprop_hist, criterion="findley", k_param=0.3)
        self.assertGreater(res_nonprop.parameter_max, 0.0)
        self.assertGreater(res_nonprop.sigma_n_max, 0.0)

    def test_dang_van_multiaxial_limit(self):
        """Verify Dang Van mesoscopic fatigue limit criterion and hydrostatic sensitivity."""
        n_steps = 40
        t = np.linspace(0, 2 * np.pi, n_steps, endpoint=False)

        tau_e = 180e6  # Shear endurance limit
        sigma_e = 300e6 # Tensile endurance limit

        # 1. Pure alternating torsion right at endurance limit tau_0 = tau_e
        hist_torsion = np.zeros((n_steps, 6))
        hist_torsion[:, 3] = tau_e * np.sin(t)

        sf_torsion, max_sigma_dv, max_ph = evaluate_dang_van_safety_factor(
            hist_torsion, shear_fatigue_limit=tau_e, tensile_fatigue_limit=sigma_e
        )
        self.assertAlmostEqual(sf_torsion, 1.0, delta=0.02)
        self.assertAlmostEqual(max_ph, 0.0, places=5)

        # 2. Adding tensile mean hydrostatic stress decreases safety factor
        hist_tensile_hydro = hist_torsion.copy()
        hist_tensile_hydro[:, 0] += 100e6  # Tensile normal stress
        hist_tensile_hydro[:, 1] += 100e6
        hist_tensile_hydro[:, 2] += 100e6

        sf_hydro, _, max_ph_tensile = evaluate_dang_van_safety_factor(
            hist_tensile_hydro, shear_fatigue_limit=tau_e, tensile_fatigue_limit=sigma_e
        )
        self.assertGreater(max_ph_tensile, 0.0)
        self.assertLess(sf_hydro, sf_torsion)  # Safety factor reduced by tensile hydrostatic stress

    def test_paraview_fatigue_vtu_export(self):
        """Verify ParaView export of voxel grid with fatigue damage and logarithmic life fields."""
        from wnfea.mesh.voxel_mesher import VoxelMesher

        grid = VoxelMesher.create_box_grid(
            bounds=(0.0, 0.02, 0.0, 0.02, 0.0, 0.02),
            resolution=(2, 2, 2),
        )

        n_cells = 8
        damages = np.array([0.0, 0.01, 0.05, 0.12, 0.45, 0.88, 1.25, 2.50])
        log_lives = np.array([12.0, 8.5, 7.2, 6.1, 5.0, 4.3, 3.8, 3.2])

        with tempfile.TemporaryDirectory() as tmpdir:
            vtu_file = os.path.join(tmpdir, "test_fatigue.vtu")
            export_voxel_grid_vtu(
                grid=grid,
                filepath=vtu_file,
                fatigue_damage=damages,
                log_fatigue_life=log_lives,
            )
            self.assertTrue(os.path.exists(vtu_file))

            with open(vtu_file, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn('<DataArray type="Float64" Name="FatigueDamage"', content)
            self.assertIn('<DataArray type="Float64" Name="Log10_FatigueLife"', content)
            self.assertIn("1.250000e+00", content)


if __name__ == "__main__":
    unittest.main()
