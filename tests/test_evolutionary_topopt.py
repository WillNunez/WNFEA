"""
Test Suite for Evolutionary Topology Optimization & 5 Diverse Candidates Engine.
--------------------------------------------------------------------------------
Verifies:
1. Morphological seed generation (X-truss, perimeter sills, central spine, monocoque).
2. Phenotypic structural distance metric and diversity quantification.
3. Multi-objective trade-offs across 5 distinct engineering archetypes.
4. 3-Axis CNC machinability evaluations for all candidates.
5. End-to-end execution of generate_5_diverse_solutions() and candidate export.
"""

from __future__ import annotations

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wnfea.mesh.voxel_mesher import VoxelMesher
from wnfea.gui.server import build_model_car_chassis_problem
from wnfea.opt.evolutionary import (
    EvolutionaryTopologyOptimizer,
    initialize_archetype_seeds,
    compute_phenotypic_distance,
    GenerativeGenome,
)


class TestEvolutionaryTopology(unittest.TestCase):
    """Rigorous tests for Evolutionary Quality-Diversity Topology Optimization."""

    def test_morphological_seeds_generation(self):
        """Verify distinct spatial distributions for seed archetypes."""
        grid = VoxelMesher.create_box_grid((0.0, 0.300, -0.050, 0.050, 0.0, 0.036), (20, 8, 4))
        seeds = initialize_archetype_seeds(grid, grid.total_cells)

        self.assertIn("uniform", seeds)
        self.assertIn("x_brace", seeds)
        self.assertIn("perimeter_sills", seeds)
        self.assertIn("center_spine", seeds)
        self.assertIn("monocoque", seeds)

        # Confirm that different seeds have non-zero phenotypic distance
        d_x_to_sills = compute_phenotypic_distance(seeds["x_brace"], seeds["perimeter_sills"])
        d_spine_to_sills = compute_phenotypic_distance(seeds["center_spine"], seeds["perimeter_sills"])

        self.assertGreater(d_x_to_sills, 0.10, "X-brace and perimeter sills must be structurally distinct")
        self.assertGreater(d_spine_to_sills, 0.15, "Center spine and perimeter sills must be distinct")

    def test_phenotypic_distance_properties(self):
        """Verify mathematical properties of the phenotypic distance metric."""
        a = np.random.uniform(0.1, 1.0, 100)
        # Distance to self must be 0
        self.assertAlmostEqual(compute_phenotypic_distance(a, a), 0.0, places=6)
        # Scaled identical field has distance 0
        self.assertAlmostEqual(compute_phenotypic_distance(a, 2.5 * a), 0.0, places=6)

    def test_generate_5_diverse_solutions_end_to_end(self):
        """Run rapid validation of the 5-solution evolutionary generator on 300mm chassis."""
        grid, load_cases, load_weights, fixed_dofs, p_solid, p_void = build_model_car_chassis_problem(
            lx=0.300,
            ly=0.100,
            lz=0.036,
            nx=15,
            ny=6,
            nz=3,
            vehicle_mass=2.0,
        )

        engine = EvolutionaryTopologyOptimizer(
            grid=grid,
            load_cases=load_cases,
            fixed_dofs=fixed_dofs,
            E=68.9e9,
            nu=0.33,
            material_density=2700.0,
            passive_solid=p_solid,
            passive_void=p_void,
            inner_iterations=6,  # Fast iterations for unit test
        )

        run_res = engine.generate_5_diverse_solutions()

        # 1. Verify exactly 5 candidates delivered
        self.assertEqual(len(run_res.candidates), 5)
        self.assertEqual(len(run_res.summary_table), 5)

        # 2. Verify all candidates have valid masses and volume fractions
        for c in run_res.candidates:
            self.assertGreater(c.mass_grams, 300.0)
            self.assertLess(c.mass_grams, 1500.0)
            self.assertGreater(c.machinability_score, 80.0)
            self.assertIsNotNone(c.densities)

        # 3. Verify structural diversity across candidates
        # Diversity matrix diagonal is 0, off-diagonals are positive
        for i in range(5):
            self.assertAlmostEqual(run_res.diversity_matrix[i, i], 0.0)
            # Compare with at least one other candidate
            other_dists = [run_res.diversity_matrix[i, j] for j in range(5) if i != j]
            self.assertTrue(any(d > 0.02 for d in other_dists), f"Candidate {i+1} must be distinct from others (max dist: {max(other_dists):.4f})")


if __name__ == "__main__":
    unittest.main()
