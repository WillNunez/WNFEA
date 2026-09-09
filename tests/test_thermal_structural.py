"""
Verification Suite: Phase 10 Thermal-Structural Multi-Physics Engine
---------------------------------------------------------------------
Validates:
1. Reference thermal conductivity matrix k0_th conservation and positive semi-definiteness.
2. 1D steady-state heat conduction against exact linear temperature profile.
3. Volumetric heat generation against exact parabolic temperature profile.
4. Robin surface convection boundary condition against analytical energy balance.
5. Implicit Euler transient thermal conduction stability and monotonic convergence.
6. Exact hydrostatic thermal stress under fully constrained boundary conditions (sigma = -E*alpha*Delta_T / (1 - 2*nu)).
7. Exact unconstrained thermal expansion with zero residual stress.
8. End-to-end thermo-mechanical pipeline with ParaView VTU multi-physics export.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tempfile
import unittest
import numpy as np

from wnfea.mesh.voxel_mesher import VoxelMesher
from wnfea.solver.thermal_solver import (
    compute_hex8_thermal_reference,
    MatrixFreeHex8ThermalOperator,
    ThermalConvectionBC,
    solve_steady_state_thermal,
    solve_transient_thermal,
    compute_thermal_load_vector,
    compute_thermo_mechanical_stresses,
    solve_thermo_mechanical,
)
from wnfea.results.paraview_export import export_voxel_grid_vtu


class TestThermalStructuralPhysics(unittest.TestCase):
    """Unit and analytical parity tests for thermal-structural engine."""

    def test_hex8_thermal_reference_properties(self):
        """Verify mathematical integrity of Hex8 reference conductivity and coupling matrices."""
        hx, hy, hz = 0.5, 0.2, 0.1
        k_therm = 45.0
        k0_th, B0_th, H_th, vol = compute_hex8_thermal_reference(hx, hy, hz, k_therm=k_therm)

        # 1. Shape and symmetry
        self.assertEqual(k0_th.shape, (8, 8))
        self.assertTrue(np.allclose(k0_th, k0_th.T, atol=1e-14), "k0_th must be symmetric")

        # 2. Row sums must be identically 0 (constant temperature generates zero heat flow)
        row_sums = np.sum(k0_th, axis=1)
        self.assertTrue(np.allclose(row_sums, 0.0, atol=1e-14), "k0_th row sums must vanish")

        # 3. Eigenvalues: exactly 1 zero eigenvalue, 7 positive eigenvalues
        evals = np.linalg.eigvalsh(k0_th)
        self.assertTrue(np.isclose(evals[0], 0.0, atol=1e-12), "Rigid thermal mode must have zero eigenvalue")
        self.assertTrue(np.all(evals[1:] > 1e-12), "Deformation thermal modes must have positive eigenvalues")

        # 4. Reference thermo-mechanical coupling H_th (24, 8)
        self.assertEqual(H_th.shape, (24, 8))
        f_uniform = H_th @ np.ones(8)
        self.assertTrue(np.allclose(np.sum(f_uniform.reshape(8, 3), axis=0), 0.0, atol=1e-14))

        # 5. Cell volume
        self.assertTrue(np.isclose(vol, hx * hy * hz))

    def test_steady_state_1d_heat_conduction(self):
        """Verify 1D heat conduction along a bar matches exact analytical linear profile."""
        L = 2.0
        nx = 10
        grid = VoxelMesher.create_box_grid(bounds=(0.0, L, 0.0, 0.2, 0.0, 0.2), resolution=(nx, 1, 1))

        nodes = grid.nodes
        left_nodes = np.where(np.isclose(nodes[:, 0], 0.0))[0]
        right_nodes = np.where(np.isclose(nodes[:, 0], L))[0]

        T_left = 120.0
        T_right = 20.0
        fixed_T = {int(n): T_left for n in left_nodes}
        fixed_T.update({int(n): T_right for n in right_nodes})

        k_therm = 60.0
        T_sol, fluxes, iters = solve_steady_state_thermal(
            grid=grid,
            fixed_temperatures=fixed_T,
            k_therm=k_therm,
            tol=1e-12,
        )

        # Analytical solution: T(x) = T_left + (T_right - T_left) * (x / L)
        T_exact = T_left + (T_right - T_left) * (nodes[:, 0] / L)
        max_err = np.max(np.abs(T_sol - T_exact))
        self.assertLess(max_err, 1e-10, f"Max error {max_err} exceeds 1e-10")

        # Analytical heat flux: q_x = -k * dT/dx = -k * (T_right - T_left) / L
        q_exact_x = -k_therm * (T_right - T_left) / L
        act_fluxes = fluxes[grid.active_element_indices]
        self.assertTrue(np.allclose(act_fluxes[:, 0], q_exact_x, atol=1e-8))
        self.assertTrue(np.allclose(act_fluxes[:, 1], 0.0, atol=1e-8))
        self.assertTrue(np.allclose(act_fluxes[:, 2], 0.0, atol=1e-8))

    def test_internal_heat_generation_parabolic(self):
        """Verify volumetric heat generation matches exact 1D parabolic profile."""
        L = 1.0
        nx = 20
        grid = VoxelMesher.create_box_grid(bounds=(0.0, L, 0.0, 0.05, 0.0, 0.05), resolution=(nx, 1, 1))

        nodes = grid.nodes
        left_nodes = np.where(np.isclose(nodes[:, 0], 0.0))[0]
        right_nodes = np.where(np.isclose(nodes[:, 0], L))[0]

        fixed_T = {int(n): 0.0 for n in left_nodes}
        fixed_T.update({int(n): 0.0 for n in right_nodes})

        k_therm = 15.0
        Q = 2000.0  # W/m^3

        T_sol, _, _ = solve_steady_state_thermal(
            grid=grid,
            fixed_temperatures=fixed_T,
            internal_heat_generation=Q,
            k_therm=k_therm,
            tol=1e-12,
        )

        # Analytical profile: T(x) = (Q / (2 * k)) * x * (L - x)
        T_exact = (Q / (2.0 * k_therm)) * nodes[:, 0] * (L - nodes[:, 0])
        max_err = np.max(np.abs(T_sol - T_exact))
        rel_err = max_err / np.max(T_exact)
        self.assertLess(rel_err, 1e-10, f"Relative error {rel_err} exceeds 1e-10")

    def test_surface_convection_robin_bc(self):
        """Verify surface convection Robin BC against analytical 1D heat balance."""
        L = 0.5
        A_cross = 0.1 * 0.1
        grid = VoxelMesher.create_box_grid(bounds=(0.0, L, 0.0, 0.1, 0.0, 0.1), resolution=(10, 1, 1))

        nodes = grid.nodes
        left_nodes = np.where(np.isclose(nodes[:, 0], 0.0))[0]
        right_nodes = np.where(np.isclose(nodes[:, 0], L))[0]

        T_left = 100.0
        fixed_T = {int(n): T_left for n in left_nodes}

        h_conv = 20.0  # W/(m^2·K)
        T_inf = 20.0   # Ambient temperature
        k_therm = 10.0

        convection_bc = ThermalConvectionBC(
            face_nodes=list(right_nodes),
            h_conv=h_conv,
            T_inf=T_inf,
            area=A_cross,
        )

        T_sol, _, _ = solve_steady_state_thermal(
            grid=grid,
            fixed_temperatures=fixed_T,
            convection_bcs=[convection_bc],
            k_therm=k_therm,
            tol=1e-10,
        )

        # Analytical: k * (T_left - T_L) / L = h * (T_L - T_inf)
        # T_L = (k * T_left + h * L * T_inf) / (k + h * L)
        T_L_exact = (k_therm * T_left + h_conv * L * T_inf) / (k_therm + h_conv * L)
        T_L_computed = np.mean(T_sol[right_nodes])

        rel_err = abs(T_L_computed - T_L_exact) / T_L_exact
        self.assertLess(rel_err, 1e-4, f"Convection T_L relative error {rel_err} exceeds 1e-4")

    def test_transient_thermal_conduction(self):
        """Verify transient thermal implicit Euler stability and monotonic heating."""
        grid = VoxelMesher.create_box_grid(bounds=(0.0, 0.5, 0.0, 0.1, 0.0, 0.1), resolution=(5, 1, 1))

        left_nodes = np.where(np.isclose(grid.nodes[:, 0], 0.0))[0]
        fixed_T = {int(n): 100.0 for n in left_nodes}

        T_history, time_pts = solve_transient_thermal(
            grid=grid,
            initial_temperatures=20.0,
            time_steps=8,
            dt=2.0,
            fixed_temperatures=fixed_T,
            rho=7800.0,
            cp=500.0,
            k_therm=50.0,
            tol=1e-8,
        )

        self.assertEqual(T_history.shape, (9, grid.total_nodes))
        # Interior nodes start at initial temperature 20.0
        interior_nodes = [n for n in range(grid.total_nodes) if n not in left_nodes]
        self.assertTrue(np.allclose(T_history[0, interior_nodes], 20.0))

        mean_temps = np.mean(T_history, axis=1)
        self.assertTrue(np.all(np.diff(mean_temps) > 0.0), "Temperatures must increase monotonically during heating")
        self.assertGreater(mean_temps[-1], mean_temps[0])

    def test_fully_constrained_thermal_expansion_hydrostatic_stress(self):
        """
        Verify exact analytical parity for fully constrained thermal expansion:
        sigma_xx = sigma_yy = sigma_zz = -E * alpha * Delta_T / (1 - 2*nu).
        Von Mises stress must be identically 0 (pure hydrostatic compression).
        """
        grid = VoxelMesher.create_box_grid(bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), resolution=(3, 3, 3))

        T_ref = 20.0
        T_applied = 80.0
        dT = T_applied - T_ref

        all_nodes = list(range(grid.total_nodes))
        all_dofs = list(range(grid.total_nodes * 3))
        fixed_T = {n: T_applied for n in all_nodes}

        E = 2.0e11
        nu = 0.28
        alpha_cte = 1.1e-5

        res = solve_thermo_mechanical(
            grid=grid,
            fixed_temperatures=fixed_T,
            structural_fixed_dofs=all_dofs,
            T_ref=T_ref,
            E=E,
            nu=nu,
            alpha_cte=alpha_cte,
            tol=1e-12,
        )

        sigma_exact = -(E * alpha_cte * dT) / (1.0 - 2.0 * nu)

        for cell_id in grid.active_element_indices:
            sxx, syy, szz, sxy, syz, szx = res.stresses[cell_id]
            self.assertTrue(np.isclose(sxx, sigma_exact, rtol=1e-10))
            self.assertTrue(np.isclose(syy, sigma_exact, rtol=1e-10))
            self.assertTrue(np.isclose(szz, sigma_exact, rtol=1e-10))
            self.assertLess(abs(sxy), 1e-10)
            self.assertLess(abs(syz), 1e-10)
            self.assertLess(abs(szx), 1e-10)
            self.assertLess(abs(res.von_mises[cell_id]), 1e-10)

    def test_free_thermal_expansion_zero_stress(self):
        """
        Verify unconstrained thermal expansion produces exact thermal strain
        Delta L = alpha * Delta_T * L and zero residual stress.
        """
        Lx, Ly, Lz = 1.0, 0.5, 0.25
        grid = VoxelMesher.create_box_grid(bounds=(0.0, Lx, 0.0, Ly, 0.0, Lz), resolution=(4, 2, 1))

        T_ref = 20.0
        T_applied = 60.0
        dT = T_applied - T_ref

        all_nodes = list(range(grid.total_nodes))
        fixed_T = {n: T_applied for n in all_nodes}

        E = 2.1e11
        nu = 0.3
        alpha_cte = 1.2e-5

        n_origin = np.where(np.isclose(grid.nodes, [0.0, 0.0, 0.0]).all(axis=1))[0][0]
        n_x = np.where(np.isclose(grid.nodes, [Lx, 0.0, 0.0]).all(axis=1))[0][0]
        n_y = np.where(np.isclose(grid.nodes, [0.0, Ly, 0.0]).all(axis=1))[0][0]

        rigid_fixed_dofs = [
            n_origin * 3 + 0, n_origin * 3 + 1, n_origin * 3 + 2,
            n_x * 3 + 1, n_x * 3 + 2,
            n_y * 3 + 2,
        ]

        res = solve_thermo_mechanical(
            grid=grid,
            fixed_temperatures=fixed_T,
            structural_fixed_dofs=rigid_fixed_dofs,
            T_ref=T_ref,
            E=E,
            nu=nu,
            alpha_cte=alpha_cte,
            tol=1e-12,
            maxiter=1000,
        )

        # Residual stress should be zero (< 1.0 Pa vs 210 GPa)
        max_vm = np.max(res.von_mises[grid.active_element_indices])
        self.assertLess(max_vm, 1.0, f"Max von Mises {max_vm} Pa is not zero in free expansion")

        # Check corner displacement
        n_corner = np.where(np.isclose(grid.nodes, [Lx, Ly, Lz]).all(axis=1))[0][0]
        u_corner = res.displacements[n_corner * 3 : n_corner * 3 + 3]

        expected_ux = alpha_cte * dT * Lx
        expected_uy = alpha_cte * dT * Ly
        expected_uz = alpha_cte * dT * Lz

        self.assertTrue(np.isclose(u_corner[0], expected_ux, rtol=1e-5))
        self.assertTrue(np.isclose(u_corner[1], expected_uy, rtol=1e-5))
        self.assertTrue(np.isclose(u_corner[2], expected_uz, rtol=1e-5))

    def test_aerospace_bracket_thermal_gradient_and_paraview_export(self):
        """
        Verify coupled thermo-mechanical solve on a thermal gradient and test
        multi-physics ParaView VTU export.
        """
        grid = VoxelMesher.create_box_grid(bounds=(0.0, 0.6, 0.0, 0.2, 0.0, 0.1), resolution=(6, 2, 1))

        nodes = grid.nodes
        left_nodes = np.where(np.isclose(nodes[:, 0], 0.0))[0]
        right_nodes = np.where(np.isclose(nodes[:, 0], 0.6))[0]

        fixed_T = {int(n): 300.0 for n in left_nodes}
        fixed_T.update({int(n): 20.0 for n in right_nodes})

        clamp_dofs = []
        for n in left_nodes:
            clamp_dofs.extend([n * 3 + 0, n * 3 + 1, n * 3 + 2])

        res = solve_thermo_mechanical(
            grid=grid,
            fixed_temperatures=fixed_T,
            structural_fixed_dofs=clamp_dofs,
            T_ref=20.0,
            E=7.0e10,
            nu=0.33,
            alpha_cte=2.3e-5,
            k_therm=180.0,
            tol=1e-8,
        )

        self.assertEqual(len(res.temperatures), grid.total_nodes)
        self.assertEqual(len(res.displacements), grid.total_nodes * 3)
        self.assertGreater(np.max(res.von_mises), 1e6)

        with tempfile.TemporaryDirectory() as tmpdir:
            vtu_file = os.path.join(tmpdir, "thermal_bracket_test.vtu")
            export_voxel_grid_vtu(
                grid=grid,
                filepath=vtu_file,
                displacements=res.displacements,
                von_mises=res.von_mises,
                temperatures=res.temperatures,
            )

            self.assertTrue(os.path.exists(vtu_file))
            with open(vtu_file, "r", encoding="utf-8") as fp:
                vtu_text = fp.read()
                self.assertIn('Name="Displacement"', vtu_text)
                self.assertIn('Name="VonMisesStress"', vtu_text)
                self.assertIn('Name="Temperature"', vtu_text)


if __name__ == "__main__":
    unittest.main()
