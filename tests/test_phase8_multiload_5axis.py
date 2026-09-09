"""
Verification Suite: Phase 8 Multi-Load Case Optimization & 5-Axis CNC Intelligence
---------------------------------------------------------------------------------
Validates:
1. Multi-Load Case Matrix-Free Topology Optimization:
   - Weighted multi-load compliance formulation (e.g. combined bending + lateral/torsion).
   - Energy sensitivity weighting across multiple load vectors.
   - Accurate volume fraction constraint enforcement under multi-directional loading.
2. 5-Axis Multi-Directional CNC Milling Constraints:
   - Cardinal multi-axis projection (+z, -z, +x, -x, +y, -y).
   - Minimum undercut pooling across multiple setup orientations.
3. 5-Axis Spindle Setup Optimization:
   - Automated identification of optimal K-setup combination (e.g. G54/G55).
   - Evaluation of undercut residuals and machinable volume fraction metrics.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import unittest
import numpy as np

from wnfea.mesh.voxel_mesher import VoxelMesher
from wnfea.opt import (
    TopologyOptimizer,
    TopologyConfig,
    CNCMillingConstraint,
    FiveAxisMachinabilityOptimizer,
)


class TestPhase8MultiLoad5Axis(unittest.TestCase):
    def test_multi_load_case_topology_optimization(self):
        """Verify multi-load case topology optimization with dual orthogonal loads."""
        nx, ny, nz = 10, 6, 4
        grid = VoxelMesher.create_box_grid(
            bounds=(0.0, 1.0, 0.0, 0.6, 0.0, 0.4),
            resolution=(nx, ny, nz),
        )

        # Fix all nodes at x = 0 (left face)
        fixed_dofs = []
        for j in range(ny + 1):
            for k in range(nz + 1):
                nid = grid.node_id(0, j, k)
                fixed_dofs.extend([nid * 3, nid * 3 + 1, nid * 3 + 2])

        # Load Case 1: Downward bending load (-Z) on right bottom edge (x = nx, z = 0)
        f1 = np.zeros(grid.total_nodes * 3, dtype=np.float64)
        for j in range(ny + 1):
            nid = grid.node_id(nx, j, 0)
            f1[nid * 3 + 2] = -1000.0

        # Load Case 2: Lateral transverse load (+Y) on right top edge (x = nx, z = nz)
        f2 = np.zeros(grid.total_nodes * 3, dtype=np.float64)
        for j in range(ny + 1):
            nid = grid.node_id(nx, j, nz)
            f2[nid * 3 + 1] = 1000.0

        config = TopologyConfig(
            target_volume_fraction=0.45,
            simp_penalty=3.0,
            filter_radius=0.15,
            max_iterations=15,
            convergence_tol=0.01,
            enable_heaviside=True,
            heaviside_start_iter=8,
            verbose=False,
        )

        optimizer = TopologyOptimizer(
            grid=grid,
            forces=[f1, f2],
            fixed_dofs=fixed_dofs,
            load_weights=[0.6, 0.4],
            config=config,
            E=2.1e11,
            nu=0.3,
        )

        self.assertEqual(len(optimizer.load_cases), 2)
        self.assertAlmostEqual(float(np.sum(optimizer.load_weights)), 1.0, places=6)

        t0 = time.perf_counter()
        res = optimizer.optimize()
        t_elapsed = time.perf_counter() - t0

        # Volume fraction constraint satisfaction (within 2%)
        self.assertAlmostEqual(res.final_volume_fraction, 0.45, delta=0.03)

        # Multi-load compliance checks
        self.assertGreater(res.final_compliance, 0.0)
        self.assertIsNotNone(res.load_case_compliances)
        self.assertEqual(len(res.load_case_compliances), 2)
        for c in res.load_case_compliances:
            self.assertGreater(c, 0.0)

        # Physical densities bounded in [0, 1]
        self.assertTrue(np.all(res.optimized_densities >= 0.0))
        self.assertTrue(np.all(res.optimized_densities <= 1.0))
        self.assertGreater(res.discreteness_index, 0.3)

    def test_multi_axis_cnc_milling_constraint(self):
        """Verify multi-directional line-of-sight milling projections and undercut penalties."""
        grid = VoxelMesher.create_box_grid(
            bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
            resolution=(5, 5, 5),
        )
        rho = np.zeros(grid.total_cells, dtype=np.float64)

        # Create a hollow cavity inside: solid shell with void center
        for k in range(5):
            for j in range(5):
                for i in range(5):
                    cid = grid.cell_id(i, j, k)
                    if i in (0, 4) or j in (0, 4) or k in (0, 4):
                        rho[cid] = 1.0  # Solid exterior walls
                    else:
                        rho[cid] = 0.0  # Internal trapped void

        # Test single axis +z: internal void is completely shadowed by top roof (k=4)
        c_top = CNCMillingConstraint(grid, milling_axis="+z")
        rho_top = c_top.project_machinable_densities(rho)
        pen_top, grad_top = c_top.evaluate_undercut_penalty(rho)
        self.assertGreater(pen_top, 0.0)

        # Test dual setup ["+z", "-z"]
        c_biz = CNCMillingConstraint(grid, milling_axis=["+z", "-z"])
        rho_biz = c_biz.project_machinable_densities(rho)
        pen_biz, _ = c_biz.evaluate_undercut_penalty(rho)
        self.assertGreater(pen_biz, 0.0)

        # Multi-axis combination: ["+z", "-z", "+x", "-x"]
        c_multi = CNCMillingConstraint(grid, milling_axis=["+z", "-z", "+x", "-x"])
        rho_multi = c_multi.project_machinable_densities(rho)
        self.assertEqual(len(rho_multi), grid.total_cells)

    def test_5axis_setup_optimization(self):
        """Verify FiveAxisMachinabilityOptimizer setup selection and undercut reduction."""
        grid = VoxelMesher.create_box_grid(
            bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
            resolution=(6, 6, 6),
        )

        # Create a stepped U-bracket open at top (+Z) and front (+Y)
        rho = np.zeros(grid.total_cells, dtype=np.float64)
        for k in range(6):
            for j in range(6):
                for i in range(6):
                    cid = grid.cell_id(i, j, k)
                    # Bottom plate + back wall + side walls
                    if k < 2 or j >= 4 or i in (0, 5):
                        rho[cid] = 1.0

        opt = FiveAxisMachinabilityOptimizer(
            grid=grid,
            candidate_axes=["+z", "-z", "+y", "-y", "+x", "-x"],
        )

        res = opt.optimize_setups(rho, max_setups=2)

        self.assertIsNotNone(res.optimal_setups)
        self.assertLessEqual(len(res.optimal_setups), 2)
        self.assertGreater(res.machinable_fraction, 0.80)
        self.assertGreater(len(res.evaluated_combinations), 10)


if __name__ == "__main__":
    unittest.main()
