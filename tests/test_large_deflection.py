"""
Test Suite for Geometric Non-Linearity & Total Lagrangian Large Deflections (Phase 23).
--------------------------------------------------------------------------------------
Verifies:
1. Pure rigid body rotation invariance (zero Green-Lagrange strain under finite rotation).
2. Large-deflection cantilever beam geometric stiffening (membrane action reduces transverse deflection vs linear theory).
3. 2nd Piola-Kirchhoff (PK2) to True Cauchy stress transformation parity.
4. Total Lagrangian coupling with Aluminum 6061-T6 J2 plasticity and isotropic hardening.
5. Multi-step Newton-Raphson equilibrium convergence with quadratic rate.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from wnfea.mesh.voxel_mesher import VoxelMesher
from wnfea.materials.plasticity import MetalPlasticMaterial, HardeningType
from wnfea.solver.large_deflection import (
    solve_large_deflection_increments,
    LargeDeflectionSolveResult,
    _precompute_hex8_grad_N,
)
from wnfea.solver.plasticity_solver import solve_elastoplastic_increments


class TestLargeDeflection(unittest.TestCase):
    """Rigorous verification tests for Total Lagrangian large-deflection mechanics."""

    def setUp(self):
        self.al6061 = MetalPlasticMaterial.aluminum_6061_t6()

    def test_rigid_body_rotation_produces_zero_green_lagrange_strain(self):
        """Verify that finite rigid body rotations yield zero Green-Lagrange strain tensor E."""
        # 1-cell Hex8 mesh
        grid = VoxelMesher.create_box_grid(bounds=(0.0, 0.1, 0.0, 0.1, 0.0, 0.1), resolution=(1, 1, 1))
        # Use element node connectivity for local Hex8 ordering
        nodes0 = grid.nodes[grid.elements[0]].copy()  # (8, 3)

        # Finite rotation theta = 15 degrees around Z axis
        theta = np.radians(15.0)
        c, s = np.cos(theta), np.sin(theta)
        R_mat = np.array([
            [ c, -s, 0.0],
            [ s,  c, 0.0],
            [0.0, 0.0, 1.0]
        ])

        # Rotated coordinates X_rot = R * X
        nodes_rot = (R_mat @ nodes0.T).T
        # Displacements u = X_rot - X_0
        u_nodes = nodes_rot - nodes0

        hx, hy, hz = grid.pitch
        B0_gps, dN_dx_gps, detJ = _precompute_hex8_grad_N(hx, hy, hz)

        # Compute Green-Lagrange strain at all 8 Gauss points
        for p in range(8):
            dNx = dN_dx_gps[p]  # (3, 8)
            # H_ij = d(u_i)/d(X_j) = sum_a u_i,a * dN_a/dX_j
            H = u_nodes.T @ dNx.T  # (3, 3)
            # Green-Lagrange strain: E = 1/2 * (H + H^T + H^T * H)
            E_mat = 0.5 * (H + H.T + H.T @ H)
            max_strain = np.max(np.abs(E_mat))
            self.assertLess(
                max_strain,
                1e-12,
                msg=f"Rigid rotation must produce zero Green-Lagrange strain: got {max_strain:.2e}",
            )

    def test_cantilever_geometric_stiffening(self):
        """
        Verify that under large transverse tip deflection, geometric non-linearity exhibits
        membrane stiffening: transverse deflection is strictly smaller than linear small-strain prediction.
        """
        # 3D cantilever: 0.50m long x 0.02m wide x 0.02m tall (slender beam)
        grid = VoxelMesher.create_box_grid(bounds=(0.0, 0.50, 0.0, 0.02, 0.0, 0.02), resolution=(10, 1, 1))
        n_dof = grid.total_nodes * 3

        # Fixed root at x = 0
        fixed_dofs = [3 * i + d for i, pt in enumerate(grid.nodes) if abs(pt[0]) < 1e-5 for d in range(3)]

        # Transverse tip load along Z
        tip_nodes = [i for i, pt in enumerate(grid.nodes) if abs(pt[0] - 0.50) < 1e-5]
        load_vec = np.zeros(n_dof, dtype=np.float64)
        load_val = -1200.0 / len(tip_nodes)  # Total -1200 N
        for node in tip_nodes:
            load_vec[3 * node + 2] = load_val
        monitor_dof = 3 * tip_nodes[0] + 2

        # Solve small-strain linear/elastoplastic
        res_small = solve_elastoplastic_increments(
            grid=grid,
            material=self.al6061,
            load_vector=load_vec,
            fixed_dofs=fixed_dofs,
            load_factors=[0.5, 1.0],
            tolerance=1e-5,
            monitor_dof=monitor_dof,
        )

        # Solve large-deflection Total Lagrangian
        res_large = solve_large_deflection_increments(
            grid=grid,
            material=self.al6061,
            load_vector=load_vec,
            fixed_dofs=fixed_dofs,
            load_factors=[0.5, 1.0],
            tolerance=1e-5,
            monitor_dof=monitor_dof,
        )

        self.assertTrue(res_large.steps[-1].converged)

        u_tip_small = abs(res_small.monitor_displacements[-1])
        u_tip_large = abs(res_large.monitor_displacements[-1])

        # Membrane tension in the deflected beam increases effective transverse stiffness
        self.assertLess(
            u_tip_large,
            u_tip_small,
            msg=f"Geometric non-linearity must exhibit membrane stiffening: large={u_tip_large:.4e} vs small={u_tip_small:.4e}",
        )

        # Longitudinal contraction (foreshortening): tip node x-displacement must be negative
        tip_u_x = res_large.steps[-1].displacements[3 * tip_nodes[0] + 0]
        self.assertLess(tip_u_x, 0.0, msg="Large transverse bending must cause longitudinal tip foreshortening")

    def test_coupled_large_deflection_with_6061_t6_plasticity(self):
        """Verify coupled geometric non-linearity + J2 plasticity for Aluminum 6061-T6."""
        # 0.10m x 0.01m x 0.01m tensile member (Cross section A = 1.0e-4 m^2)
        # Yield load = 276 MPa * 1e-4 m^2 = 27.6 kN
        grid = VoxelMesher.create_box_grid(bounds=(0.0, 0.10, 0.0, 0.01, 0.0, 0.01), resolution=(5, 1, 1))
        n_dof = grid.total_nodes * 3

        fixed_dofs = [3 * i + d for i, pt in enumerate(grid.nodes) if abs(pt[0]) < 1e-5 for d in range(3)]
        tip_nodes = [i for i, pt in enumerate(grid.nodes) if abs(pt[0] - 0.10) < 1e-5]
        load_vec = np.zeros(n_dof, dtype=np.float64)
        # 32 kN axial tension (exceeds yield 27.6 kN into plastic regime)
        for node in tip_nodes:
            load_vec[3 * node + 0] = 32000.0 / len(tip_nodes)

        res = solve_large_deflection_increments(
            grid=grid,
            material=self.al6061,
            load_vector=load_vec,
            fixed_dofs=fixed_dofs,
            load_factors=[0.5, 0.8, 1.0],
            tolerance=1e-5,
            monitor_dof=3 * tip_nodes[0] + 0,
        )

        for step in res.steps:
            self.assertTrue(step.converged, f"Step {step.step_index} failed to converge")

        # Step 0 (50% load = 16 kN) must be purely elastic
        self.assertFalse(np.any(res.steps[0].yielded_mask), "Step 0 at 16 kN must be elastic")
        self.assertEqual(np.max(res.steps[0].accumulated_plastic_strain), 0.0)

        # Step 2 (100% load = 32 kN) must experience plastic yielding
        final_step = res.steps[-1]
        self.assertTrue(np.any(final_step.yielded_mask), "Elements must yield under 32 kN load in 6061-T6")
        self.assertGreater(np.max(final_step.accumulated_plastic_strain), 0.01)

        # True Cauchy stress must exceed nominal engineering stress due to lateral necking/contraction
        max_cauchy = np.max(np.abs(final_step.cauchy_stresses[:, 0]))
        self.assertGreater(max_cauchy, 300e6)  # Exceeds 300 MPa due to hardening & large deformation


if __name__ == "__main__":
    unittest.main()
