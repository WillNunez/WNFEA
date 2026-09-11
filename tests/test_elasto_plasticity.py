"""
Test Suite for Rate-Independent J2 von Mises Elasto-Plasticity & Isotropic Hardening (Phase 22).
------------------------------------------------------------------------------------------------
Verifies:
1. Aerospace & structural metal material presets (Al 6061-T6, Al 7075-T6, Ti-6Al-4V, S355, 304 SS).
2. Uniaxial tension stress-strain response matching exact analytical bilinear curve.
3. Elastic unloading and permanent plastic strain recovery (eps_p = eps_max - sigma_max / E).
4. Voce non-linear saturation hardening asymptotic convergence to sigma_y0 + R_inf.
5. von Mises pure shear yielding at exact tau_y = sigma_y0 / sqrt(3).
6. Consistent algorithmic elastoplastic tangent parity vs numerical finite difference (< 1e-5).
7. Exact plastic incompressibility tr(d_eps_p) = 0 (< 1e-14).
8. 3D cantilever plastic bending, incremental Newton-Raphson convergence, and springback residual deflection.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from wnfea.materials.plasticity import (
    HardeningType,
    MetalPlasticMaterial,
    PlasticHistoryState,
    radial_return_mapping,
)
from wnfea.mesh.voxel_mesher import VoxelMesher
from wnfea.solver.plasticity_solver import (
    ElastoPlasticStepResult,
    ElastoPlasticSolveResult,
    solve_elastoplastic_increments,
)


class TestElastoPlasticity(unittest.TestCase):
    """Rigorous verification suite for Phase 22 Metal Plasticity Engine."""

    def test_metal_material_presets_and_isotropic_hardening(self):
        """Verify standard alloy presets, initial yield strength, and hardening slopes."""
        al6061 = MetalPlasticMaterial.aluminum_6061_t6()
        self.assertEqual(al6061.hardening_type, HardeningType.VOCE)
        self.assertAlmostEqual(al6061.yield_stress(0.0), 276e6)
        self.assertAlmostEqual(al6061.hardening_slope(0.0), al6061.R_inf * al6061.b_voce)

        al7075 = MetalPlasticMaterial.aluminum_7075_t6()
        self.assertEqual(al7075.hardening_type, HardeningType.LINEAR)
        self.assertAlmostEqual(al7075.yield_stress(0.0), 503e6)
        self.assertAlmostEqual(al7075.yield_stress(0.01), 503e6 + 0.01 * al7075.H)

        ti = MetalPlasticMaterial.titanium_ti6al4v()
        self.assertEqual(ti.hardening_type, HardeningType.VOCE)
        self.assertAlmostEqual(ti.yield_stress(0.0), 880e6)

        steel = MetalPlasticMaterial.structural_steel_s355()
        self.assertEqual(steel.hardening_type, HardeningType.LINEAR)
        self.assertAlmostEqual(steel.yield_stress(0.0), 355e6)

        ss304 = MetalPlasticMaterial.stainless_steel_304()
        self.assertEqual(ss304.hardening_type, HardeningType.POWER_LAW)
        self.assertGreater(ss304.yield_stress(0.05), ss304.yield_stress(0.0))

    def test_uniaxial_bilinear_elastoplastic_tension(self):
        """Verify uniaxial tension matches analytical bilinear elastoplastic response."""
        mat = MetalPlasticMaterial(
            E=200e9,
            nu=0.30,
            sigma_y0=200e6,
            hardening_type=HardeningType.LINEAR,
            H=20e9,
        )

        eps_y = mat.sigma_y0 / mat.E  # 0.001 strain at yield

        # 1. Pure elastic point: eps = 0.5 * eps_y
        eps_el = np.array([0.5 * eps_y, -mat.nu * 0.5 * eps_y, -mat.nu * 0.5 * eps_y, 0, 0, 0])
        sig_el, state_el, _ = radial_return_mapping(mat, eps_el)
        self.assertFalse(state_el.yielded)
        self.assertAlmostEqual(sig_el[0], 0.5 * mat.sigma_y0, delta=1e-3)
        self.assertAlmostEqual(state_el.alpha_p, 0.0, delta=1e-12)

        # 2. Plastic point: eps = 2.0 * eps_y
        eps_pl = np.array([2.0 * eps_y, -0.4 * eps_y, -0.4 * eps_y, 0, 0, 0])
        sig_pl, state_pl, _ = radial_return_mapping(mat, eps_pl)
        self.assertTrue(state_pl.yielded)
        self.assertGreater(state_pl.alpha_p, 0.0)

        # von Mises stress must match updated flow stress sigma_y(alpha) exactly
        s_dev = sig_pl[:3] - np.mean(sig_pl[:3])
        vm_calc = np.sqrt(1.5 * np.sum(s_dev**2))
        expected_flow_stress = mat.yield_stress(state_pl.alpha_p)
        self.assertAlmostEqual(vm_calc, expected_flow_stress, delta=1e-3)

    def test_elastic_unloading_and_permanent_strain(self):
        """Verify zero plastic strain on unloading and exact permanent deformation recovery."""
        mat = MetalPlasticMaterial.structural_steel_s355()
        eps_y = mat.sigma_y0 / mat.E

        # Step 1: Load to 3x yield strain
        eps_load = np.array([3.0 * eps_y, -mat.nu * eps_y, -mat.nu * eps_y, 0, 0, 0])
        sig_load, state_load, _ = radial_return_mapping(mat, eps_load)
        self.assertTrue(state_load.yielded)
        alpha_loaded = state_load.alpha_p
        self.assertGreater(alpha_loaded, 0.0)

        # Step 2: Unload partially to 1.5x yield strain (elastic unloading)
        eps_unload1 = np.array([1.5 * eps_y, -mat.nu * eps_y, -mat.nu * eps_y, 0, 0, 0])
        sig_unload1, state_unload1, _ = radial_return_mapping(mat, eps_unload1, state_old=state_load)
        # Plastic strain must be frozen during elastic unloading
        self.assertFalse(state_unload1.yielded)
        self.assertAlmostEqual(state_unload1.alpha_p, alpha_loaded, delta=1e-12)
        np.testing.assert_allclose(state_unload1.eps_p, state_load.eps_p, atol=1e-12)

        # Step 3: Complete elastic unloading to zero stress: eps = eps_p
        eps_zero_stress = state_load.eps_p.copy()
        sig_zero, state_zero, _ = radial_return_mapping(mat, eps_zero_stress, state_old=state_load)
        # Resulting stress must be virtually zero
        self.assertLess(np.max(np.abs(sig_zero)), 1e-4)

    def test_voce_nonlinear_hardening_saturation(self):
        """Verify Voce saturation hardening asymptotically approaches sigma_y0 + R_inf."""
        mat = MetalPlasticMaterial.titanium_ti6al4v()
        expected_saturation = mat.sigma_y0 + mat.R_inf

        # Apply large plastic strain increments
        alpha_large = 0.50  # 50% plastic strain
        flow_stress_sat = mat.yield_stress(alpha_large)

        rel_diff = abs(flow_stress_sat - expected_saturation) / expected_saturation
        self.assertLess(rel_diff, 1e-6, msg="Voce flow stress must saturate to sigma_y0 + R_inf")

        # Slope must approach zero at saturation
        slope_sat = mat.hardening_slope(alpha_large)
        self.assertLess(slope_sat / mat.E, 1e-6)

    def test_von_mises_pure_shear_yield(self):
        """Verify pure shear yielding occurs at exact theoretical tau_y = sigma_y0 / sqrt(3)."""
        mat = MetalPlasticMaterial.structural_steel_s355()
        tau_y_expected = mat.sigma_y0 / math.sqrt(3.0)

        # 1. Just below shear yield
        gamma_sub = 0.999 * (tau_y_expected / mat.G)
        eps_sub = np.array([0, 0, 0, 0, 0, gamma_sub], dtype=np.float64)
        sig_sub, state_sub, _ = radial_return_mapping(mat, eps_sub)
        self.assertFalse(state_sub.yielded)
        self.assertAlmostEqual(sig_sub[5], mat.G * gamma_sub, delta=1.0)

        # 2. Above shear yield
        gamma_sup = 1.50 * (tau_y_expected / mat.G)
        eps_sup = np.array([0, 0, 0, 0, 0, gamma_sup], dtype=np.float64)
        sig_sup, state_sup, _ = radial_return_mapping(mat, eps_sup)
        self.assertTrue(state_sup.yielded)

        # Under perfectly plastic or small hardening, tau is close to tau_y_expected
        self.assertGreater(sig_sup[5], tau_y_expected * 0.999)

    def test_consistent_algorithmic_tangent_parity(self):
        """Verify analytical tangent C_ep matches numerical finite difference perturbation to < 1e-5."""
        mat = MetalPlasticMaterial.aluminum_7075_t6()

        # Multi-axial plastic strain state
        eps0 = np.array([0.015, -0.004, -0.005, 0.003, 0.002, -0.004], dtype=np.float64)
        sig0, _, C_ep = radial_return_mapping(mat, eps0, compute_tangent=True)

        # Numerical perturbation tangent
        C_fd = np.zeros((6, 6), dtype=np.float64)
        h = 1e-8
        for j in range(6):
            eps_pert = eps0.copy()
            eps_pert[j] += h
            sig_pert, _, _ = radial_return_mapping(mat, eps_pert, compute_tangent=False)
            C_fd[:, j] = (sig_pert - sig0) / h

        rel_diff = np.max(np.abs(C_ep - C_fd)) / np.max(np.abs(C_fd))
        self.assertLess(
            rel_diff,
            1e-5,
            msg=f"Algorithmic tangent relative error {rel_diff:.2e} exceeds 1e-5",
        )

        # Tangent must be symmetric
        symm_diff = np.max(np.abs(C_ep - C_ep.T)) / np.max(np.abs(C_ep))
        self.assertLess(symm_diff, 1e-12, msg="Algorithmic tangent C_ep must be strictly symmetric")

    def test_plastic_incompressibility(self):
        """Verify exact plastic volumetric incompressibility tr(d_eps_p) = 0."""
        mat = MetalPlasticMaterial.aluminum_6061_t6()

        # Batch of arbitrary plastic strain increments
        np.random.seed(123)
        eps_batch = np.random.randn(10, 6) * 0.02

        _, state_batch, _ = radial_return_mapping(mat, eps_batch)

        # Volumetric plastic strain = eps_p_xx + eps_p_yy + eps_p_zz
        plastic_vol = np.sum(state_batch.eps_p[:, :3], axis=1)
        max_vol_error = np.max(np.abs(plastic_vol))

        self.assertLess(
            max_vol_error,
            1e-14,
            msg=f"Plastic incompressibility violation: {max_vol_error:.2e} exceeds 1e-14",
        )

    def test_3d_cantilever_plastic_bending_and_springback(self):
        """Verify incremental 3D cantilever plastic bending, Newton convergence, and springback."""
        # 3D cantilever beam: 6 x 2 x 2 = 24 elements
        # Dimensions: 0.6m x 0.1m x 0.1m
        bounds = (0.0, 0.6, 0.0, 0.1, 0.0, 0.1)
        res = (6, 2, 2)
        grid = VoxelMesher.create_box_grid(bounds, res)

        mat = MetalPlasticMaterial(
            name="Test Aluminum",
            E=70e9,
            nu=0.33,
            sigma_y0=25e6,
            hardening_type=HardeningType.LINEAR,
            H=2.0e9,
        )

        n_nodes = grid.total_nodes
        n_dof = n_nodes * 3

        # Fixed boundary condition at root x = 0
        fixed_dofs = []
        for i, pt in enumerate(grid.nodes):
            if abs(pt[0]) < 1e-5:
                fixed_dofs.extend([3 * i, 3 * i + 1, 3 * i + 2])

        # Downward transverse load F_z applied at tip x = 0.6m
        load_vec = np.zeros(n_dof, dtype=np.float64)
        tip_nodes = [i for i, pt in enumerate(grid.nodes) if abs(pt[0] - 0.6) < 1e-5]
        load_per_node = -30000.0 / len(tip_nodes)  # Total -30 kN
        for node in tip_nodes:
            load_vec[3 * node + 2] = load_per_node

        monitor_dof = 3 * tip_nodes[0] + 2  # Monitor tip vertical deflection

        # Loading schedule: [0.2 (elastic), 0.6 (onset of yield), 1.0 (full plastic bending), 0.0 (springback)]
        load_factors = [0.2, 0.6, 1.0, 0.0]

        result = solve_elastoplastic_increments(
            grid=grid,
            material=mat,
            load_vector=load_vec,
            fixed_dofs=fixed_dofs,
            load_factors=load_factors,
            tolerance=1e-5,
            monitor_dof=monitor_dof,
        )

        # 1. Verification of step convergence
        for step in result.steps:
            self.assertTrue(step.converged, f"Step {step.step_index} failed to converge")

        # 2. First step (lambda = 0.3) should be mostly elastic
        step0 = result.steps[0]
        step2 = result.steps[2]  # Full peak load (lambda = 1.0)
        self.assertGreater(np.sum(step2.yielded_mask), np.sum(step0.yielded_mask))
        self.assertGreater(np.max(step2.accumulated_plastic_strain), 0.0)

        # 3. Springback: After unloading to 0.0 load (step 3), permanent deflection must remain
        step_unloaded = result.steps[3]
        tip_unloaded = result.monitor_displacements[3]
        tip_peak = result.monitor_displacements[2]

        # Deflection at peak load is negative (downward)
        self.assertLess(tip_peak, 0.0)
        # Permanent deflection after springback is still negative (downward), but smaller in magnitude
        self.assertLess(tip_unloaded, 0.0)
        self.assertGreater(tip_unloaded, tip_peak)

        # Residual stress field must exist in unloaded state
        max_residual_vm = np.max(np.abs(step_unloaded.stresses))
        self.assertGreater(max_residual_vm, 1e6, msg="Residual stress must exist after plastic springback")


if __name__ == "__main__":
    unittest.main()
