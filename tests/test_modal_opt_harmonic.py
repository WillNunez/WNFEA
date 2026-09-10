"""
Unit and Integration Tests for Phase 17:
Dynamic Frequency-Constrained Generative Optimization & Harmonic Response Solver.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from wnfea.mesh.voxel_mesher import VoxelMesher
from wnfea.solver.modal_analysis import (
    solve_modal_analysis,
    compute_lumped_mass_hex8,
)
from wnfea.solver.harmonic_response import (
    HarmonicResponseResult,
    solve_harmonic_modal_superposition,
    solve_direct_harmonic_pcg,
)
from wnfea.opt.modal_opt import (
    FrequencyConstrainedConfig,
    FrequencyOptResult,
    ModalSensitivityEvaluator,
    FrequencyConstrainedOptimizer,
)


class TestModalOptHarmonic(unittest.TestCase):
    """
    Verification suite for modal eigenvalue sensitivities, frequency-constrained
    topology optimization, and matrix-free steady-state harmonic FRF analysis.
    """

    def setUp(self):
        # Standard cantilever beam domain: L=1.0m, W=0.2m, H=0.2m
        self.L, self.W, self.H = 1.0, 0.2, 0.2
        self.grid = VoxelMesher.create_box_grid((0.0, self.L, 0.0, self.W, 0.0, self.H), (6, 1, 1))
        
        # Clamp at x=0
        self.fixed_dofs = []
        for node_idx, pt in enumerate(self.grid.nodes):
            if abs(pt[0]) < 1e-5:
                self.fixed_dofs.extend([node_idx * 3, node_idx * 3 + 1, node_idx * 3 + 2])
        self.fixed_dofs = np.array(self.fixed_dofs, dtype=np.int32)
        
        # Downward tip load at x=L
        self.forces = np.zeros(len(self.grid.nodes) * 3)
        tip_nodes = [i for i, pt in enumerate(self.grid.nodes) if abs(pt[0] - self.L) < 1e-5]
        for tn in tip_nodes:
            self.forces[tn * 3 + 2] = -1000.0 / len(tip_nodes)

    def test_01_modal_eigenvalue_sensitivities_parity(self):
        """Verify exact closed-form eigenvalue sensitivities d(omega_1^2)/d(rho) vs numerical FD."""
        rho = np.array([0.8, 0.75, 0.7, 0.65, 0.6, 0.55], dtype=np.float64)
        
        # Base modal solve
        modal_base = solve_modal_analysis(
            self.grid,
            self.fixed_dofs,
            num_modes=1,
            densities=rho,
            tol=1e-6,
            method="eigsh",
        )
        omega1_sq = float(modal_base.eigenvalues[0])
        phi_1 = modal_base.mode_shapes[0].reshape(-1)
        
        evaluator = ModalSensitivityEvaluator(self.grid, simp_penalty=3.0)
        domega_drho_ana = evaluator.evaluate_eigenvalue_sensitivities(rho, omega1_sq, phi_1)
        
        # Numerical central finite difference check on all elements
        eps = 1e-5
        for e in range(len(rho)):
            rho_plus = rho.copy()
            rho_minus = rho.copy()
            rho_plus[e] += eps
            rho_minus[e] -= eps
            
            m_plus = solve_modal_analysis(self.grid, self.fixed_dofs, num_modes=1, densities=rho_plus, tol=1e-6)
            m_minus = solve_modal_analysis(self.grid, self.fixed_dofs, num_modes=1, densities=rho_minus, tol=1e-6)
            
            domega_fd = (m_plus.eigenvalues[0] - m_minus.eigenvalues[0]) / (2.0 * eps)
            rel_error = abs(domega_drho_ana[e] - domega_fd) / abs(domega_drho_ana[e])
            self.assertLess(rel_error, 1e-4, f"Eigenvalue sensitivity mismatch at element {e}")

    def test_02_cantilever_spatial_sensitivity_distribution(self):
        """Verify that root elements have positive frequency sensitivity and tip elements have negative."""
        rho = np.full(self.grid.total_cells, 0.7)
        modal = solve_modal_analysis(self.grid, self.fixed_dofs, num_modes=1, densities=rho, tol=1e-5)
        
        evaluator = ModalSensitivityEvaluator(self.grid, simp_penalty=3.0)
        sens = evaluator.evaluate_eigenvalue_sensitivities(rho, modal.eigenvalues[0], modal.mode_shapes[0].reshape(-1))
        
        # Element 0 is at root (x=0..dx): stiffness dominates -> d(omega^2)/drho > 0
        self.assertGreater(sens[0], 0.0, "Root element must have positive frequency sensitivity")
        # Element -1 is at tip (x=L-dx..L): inertial mass dominates -> d(omega^2)/drho < 0
        self.assertLess(sens[-1], 0.0, "Tip element must have negative frequency sensitivity")
        # Sensitivities must decrease strictly monotonically from root to tip
        for i in range(len(sens) - 1):
            self.assertGreater(sens[i], sens[i + 1], f"Sensitivity should decrease along cantilever length at {i}")

    def test_03_frequency_constrained_topology_optimization(self):
        """Verify frequency-constrained topology optimization drives f_1 >= target frequency."""
        # Initial natural frequency of uniform cantilever
        modal_init = solve_modal_analysis(self.grid, self.fixed_dofs, num_modes=1, densities=np.full(6, 0.40))
        f1_init = float(modal_init.frequencies_hz[0])
        
        # Set target frequency higher than initial
        target_f1 = f1_init * 1.15  # Request 15% increase in fundamental frequency
        
        config = FrequencyConstrainedConfig(
            target_frequency_hz=target_f1,
            target_volume_fraction=0.40,
            simp_penalty=3.0,
            max_iterations=12,
            filter_radius=0.15,
            move_limit=0.15,
            verbose=False,
        )
        
        optimizer = FrequencyConstrainedOptimizer(self.grid, self.forces, self.fixed_dofs, config)
        res = optimizer.optimize()
        
        # Check optimization results
        self.assertGreater(res.total_iterations, 2)
        self.assertGreater(res.final_frequency_hz, f1_init, "Frequency must increase over initial uniform design")
        self.assertAlmostEqual(res.final_volume_fraction, 0.40, delta=0.03, msg="Volume constraint must be respected")
        
        # Check material distribution: root must have higher density than tip
        self.assertGreater(res.optimized_densities[0], res.optimized_densities[-1],
                           "Material must concentrate at the root to maximize bending frequency")

    def test_04_harmonic_modal_superposition_sweep(self):
        """Verify steady-state harmonic response sweep and dynamic amplification factor Q."""
        modal = solve_modal_analysis(self.grid, self.fixed_dofs, num_modes=3, tol=1e-5)
        f1 = float(modal.frequencies_hz[0])
        
        zeta = 0.025  # 2.5% damping
        expected_q = 1.0 / (2.0 * zeta)  # Q = 20.0
        
        # Sweep across fundamental frequency +/- 50 Hz
        res = solve_harmonic_modal_superposition(
            self.grid,
            self.forces,
            self.fixed_dofs,
            modal,
            freq_start_hz=f1 * 0.2,
            freq_end_hz=f1 * 2.0,
            num_frequencies=150,
            damping_ratio=zeta,
        )
        
        # Check resonance peak
        peak_freq = res.peak_frequencies_hz[0]
        self.assertAlmostEqual(peak_freq, f1, delta=2.5, msg="Resonant peak must occur at natural frequency")
        
        # Dynamic amplification factor Q should be within 10% of theoretical 1 / (2 * zeta)
        self.assertAlmostEqual(res.dynamic_amplification_factor, expected_q, delta=expected_q * 0.15,
                               msg=f"Dynamic amplification Q ({res.dynamic_amplification_factor:.1f}) should match 1/(2*zeta) ({expected_q:.1f})")

    def test_05_direct_vs_modal_harmonic_response_parity(self):
        """Verify direct matrix-free Krylov harmonic solver matches modal superposition."""
        modal = solve_modal_analysis(self.grid, self.fixed_dofs, num_modes=25, tol=1e-5)
        f1 = float(modal.frequencies_hz[0])
        
        test_freq = f1 * 0.70  # Sub-resonant frequency
        zeta = 0.03
        
        # Modal superposition at test_freq
        res_modal = solve_harmonic_modal_superposition(
            self.grid,
            self.forces,
            self.fixed_dofs,
            modal,
            freq_start_hz=test_freq,
            freq_end_hz=test_freq,
            num_frequencies=1,
            damping_ratio=zeta,
        )
        mag_modal = res_modal.frf_magnitude[0]
        
        # Direct solve at test_freq
        u_re, u_im, mag_direct = solve_direct_harmonic_pcg(
            self.grid,
            self.forces,
            self.fixed_dofs,
            frequency_hz=test_freq,
            damping_ratio=zeta,
            reference_freq_hz=f1,
            tol=1e-5,
            maxiter=400,
        )
        
        # Compare tip displacement magnitude
        tip_dof = len(self.forces) - 1
        rel_diff = abs(mag_direct[tip_dof] - mag_modal[tip_dof]) / mag_direct[tip_dof]
        self.assertLess(rel_diff, 0.05, "Direct harmonic solve must match modal superposition to < 5%")

    def test_06_phase_angle_transition_across_resonance(self):
        """Verify dynamic phase shift: 0 deg below resonance -> -90 deg at resonance -> -180 deg above resonance."""
        modal = solve_modal_analysis(self.grid, self.fixed_dofs, num_modes=2, tol=1e-5)
        f1 = float(modal.frequencies_hz[0])
        
        zeta = 0.02
        tip_dof = [len(self.forces) - 1]
        
        # Low frequency: 0.1 * f1
        res_low = solve_harmonic_modal_superposition(
            self.grid, self.forces, self.fixed_dofs, modal,
            freq_start_hz=f1 * 0.1, freq_end_hz=f1 * 0.1, num_frequencies=1,
            damping_ratio=zeta, monitor_dofs=tip_dof
        )
        
        # At resonance: 1.0 * f1
        res_res = solve_harmonic_modal_superposition(
            self.grid, self.forces, self.fixed_dofs, modal,
            freq_start_hz=f1, freq_end_hz=f1, num_frequencies=1,
            damping_ratio=zeta, monitor_dofs=tip_dof
        )
        
        # High frequency: 2.5 * f1
        res_high = solve_harmonic_modal_superposition(
            self.grid, self.forces, self.fixed_dofs, modal,
            freq_start_hz=f1 * 2.5, freq_end_hz=f1 * 2.5, num_frequencies=1,
            damping_ratio=zeta, monitor_dofs=tip_dof
        )
        
        # Phase shift relative to quasi-static low-frequency response
        u_ref = res_low.frf_displacement[0, 0]
        shift_low = float(np.angle(res_low.frf_displacement[0, 0] / u_ref, deg=True))
        shift_res = float(np.angle(res_res.frf_displacement[0, 0] / u_ref, deg=True))
        shift_high = float(np.angle(res_high.frf_displacement[0, 0] / u_ref, deg=True))
        
        # Below resonance: in-phase (~0 deg)
        self.assertAlmostEqual(shift_low, 0.0, delta=2.0)
        # At resonance: phase lag of -90 deg
        self.assertAlmostEqual(shift_res, -90.0, delta=6.0, msg="Resonant phase lag should be near -90 degrees")
        # Above resonance: phase lag approaches -180 deg
        self.assertLess(shift_high, -150.0, msg="High frequency phase lag should approach -180 degrees")
        self.assertGreater(shift_low, shift_res)
        self.assertGreater(shift_res, shift_high)

    def test_07_multi_mode_resonance_sweep(self):
        """Verify wide-band multi-mode harmonic sweep detects multiple resonant peaks."""
        modal = solve_modal_analysis(self.grid, self.fixed_dofs, num_modes=4, tol=1e-5)
        f1 = float(modal.frequencies_hz[0])
        f3 = float(modal.frequencies_hz[2])  # Mode 3 is second Z-bending mode
        
        tip_dof = [len(self.forces) - 1]
        
        # Wide-band sweep from 50 Hz to 1200 Hz
        res = solve_harmonic_modal_superposition(
            self.grid,
            self.forces,
            self.fixed_dofs,
            modal,
            freq_start_hz=50.0,
            freq_end_hz=1200.0,
            num_frequencies=250,
            damping_ratio=0.02,
            monitor_dofs=tip_dof,
        )
        
        # Should identify multiple resonance peaks
        self.assertGreaterEqual(len(res.peak_frequencies_hz), 2)
        # First peak should match f1
        self.assertAlmostEqual(res.peak_frequencies_hz[0], f1, delta=10.0)
        # Second peak should match f3
        self.assertAlmostEqual(res.peak_frequencies_hz[1], f3, delta=15.0)


if __name__ == "__main__":
    unittest.main()
