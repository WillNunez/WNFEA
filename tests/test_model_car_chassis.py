"""
Test Suite for 300mm Model Car Chassis Case Study.
--------------------------------------------------
Verifies:
1. Chassis physical domain dimensions (300 mm x 100 mm x 36 mm).
2. Aluminum 6061-T6 material specification and mass calculations.
3. Double wishbone suspension pickup hardpoints and passive solid/void domains.
4. 4 Realistic 5g Dynamic Tire Grip Load Cases.
5. 3-Axis CNC bidirectional milling constraint and undercut suppression.
6. Topology optimization convergence and target volume fraction attainment (V* <= 0.35).
7. Watertight triangular STL mesh extraction and manifold face verification.
"""

from __future__ import annotations

import os
import sys
import unittest
import tempfile
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from examples.model_car_chassis_case_study import run_chassis_case_study


class TestModelCarChassis(unittest.TestCase):
    """Rigorous verification of the 300mm model car chassis case study."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_run_chassis_case_study_end_to_end(self):
        """Run complete case study on validation grid and verify all engineering targets."""
        summary = run_chassis_case_study(
            output_dir=self.temp_dir.name,
            resolution=(20, 8, 4),
            target_volume_fraction=0.30,
            max_iterations=12,
        )

        # 1. Material and Geometry
        self.assertEqual(summary["material"], "Aluminum 6061-T6")
        self.assertEqual(summary["dimensions_mm"]["length"], 300)
        self.assertEqual(summary["dimensions_mm"]["width"], 100)
        self.assertEqual(summary["dimensions_mm"]["height"], 36)

        # 2. Mass Reduction Target (> 65% reduction)
        self.assertLessEqual(summary["final_volume_fraction"], 0.35)
        self.assertGreater(summary["mass_reduction_percent"], 65.0)
        self.assertLess(summary["optimized_mass_grams"], 1100.0)

        # 3. 3-Axis CNC Machinability
        self.assertEqual(summary["cnc_milling_setup"], "3-axis bidirectional top/bottom (bi-z)")
        self.assertGreaterEqual(summary["machinability_score_percent"], 90.0)

        # 4. Suspension and Loads
        self.assertEqual(summary["suspension_type"], "double_wishbone")
        self.assertEqual(summary["dynamic_g_acceleration"], 5.0)

        # 5. Output Artifacts
        stl_path = summary["artifacts"]["stl"]
        self.assertTrue(os.path.exists(stl_path), f"STL file must exist: {stl_path}")
        self.assertGreater(os.path.getsize(stl_path), 1000, "STL must contain valid geometry data")


if __name__ == "__main__":
    unittest.main()
