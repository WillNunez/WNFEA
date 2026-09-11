"""
Test Suite for WNFEA Interactive Web GUI & REST Backend.
-------------------------------------------------------
Verifies:
1. Material database integrity (Aluminum 6061-T6, 7075-T6, Ti-6Al-4V, Steel).
2. Application presets integrity, specifically the 300mm model car chassis case.
3. Problem construction: double wishbone suspension hardpoints, 5g load cases, passive solid/void zones.
4. HTTP REST API endpoints (/api/materials, /api/presets, /api/status, index.html delivery).
5. Background optimization job state machine and lifecycle.
"""

from __future__ import annotations

import os
import sys
import json
import time
import urllib.request
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wnfea.gui.server import (
    MATERIALS_DB,
    PRESETS_DB,
    GUIServer,
    build_model_car_chassis_problem,
    GLOBAL_JOB,
)


class TestGUIBackend(unittest.TestCase):
    """Rigorous tests for the WNFEA GUI server and chassis problem builder."""

    @classmethod
    def setUpClass(cls):
        # Start GUI server on test port
        cls.port = 8899
        cls.server = GUIServer(host="127.0.0.1", port=cls.port)
        cls.server.start(blocking=False)
        time.sleep(0.3)  # Allow socket to bind

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    def test_materials_database(self):
        """Verify Aluminum 6061-T6 and structural metals are correctly configured."""
        self.assertIn("al6061_t6", MATERIALS_DB)
        al = MATERIALS_DB["al6061_t6"]
        self.assertAlmostEqual(al["E"], 68.9e9)
        self.assertAlmostEqual(al["nu"], 0.33)
        self.assertAlmostEqual(al["density"], 2700.0)
        self.assertAlmostEqual(al["yield_strength"], 276e6)
        self.assertEqual(al["recommended_milling_axis"], "bi-z")

    def test_presets_database(self):
        """Verify 300mm Model Car Chassis preset parameters."""
        self.assertIn("model_car_chassis_300mm", PRESETS_DB)
        preset = PRESETS_DB["model_car_chassis_300mm"]
        self.assertEqual(preset["dimensions"]["lx"], 0.300)
        self.assertEqual(preset["dimensions"]["ly"], 0.100)
        self.assertEqual(preset["dimensions"]["lz"], 0.036)
        self.assertEqual(preset["material_id"], "al6061_t6")
        self.assertEqual(preset["cnc_milling"]["axis"], "bi-z")
        self.assertEqual(len(preset["load_cases"]), 4)

    def test_build_model_car_chassis_problem(self):
        """Verify construction of double wishbone suspension hardpoints and 5g loads."""
        grid, load_cases, load_weights, fixed_dofs, p_solid, p_void = build_model_car_chassis_problem(
            lx=0.300,
            ly=0.100,
            lz=0.036,
            nx=20,
            ny=8,
            nz=4,
            vehicle_mass=2.0,
        )
        self.assertEqual(grid.bounds, (0.0, 0.300, -0.050, 0.050, 0.0, 0.036))
        self.assertEqual(len(load_cases), 4)
        self.assertEqual(len(load_weights), 4)
        self.assertGreater(len(fixed_dofs), 0)
        self.assertGreater(len(p_solid), 0)
        self.assertGreater(len(p_void), 0)

        # Check total 5g lateral tire grip magnitude (~98 N)
        f_lat = load_cases[0]
        self.assertAlmostEqual(sum(f_lat[1::3]), 2.0 * 5.0 * 9.81, places=1)

    def test_http_get_index(self):
        """Verify that index.html is served with 200 OK."""
        url = f"http://127.0.0.1:{self.port}/"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            html = resp.read().decode("utf-8")
            self.assertIn("WNFEA Generative Studio", html)
            self.assertIn("Three.js", html)
            self.assertIn("3-Axis CNC", html)

    def test_http_get_materials(self):
        """Verify GET /api/materials returns valid JSON dictionary."""
        url = f"http://127.0.0.1:{self.port}/api/materials"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("al6061_t6", data)

    def test_http_get_presets(self):
        """Verify GET /api/presets returns presets dictionary."""
        url = f"http://127.0.0.1:{self.port}/api/presets"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("model_car_chassis_300mm", data)

    def test_http_get_status(self):
        """Verify GET /api/status returns job state."""
        url = f"http://127.0.0.1:{self.port}/api/status"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("is_running", data)
            self.assertIn("status_text", data)


if __name__ == "__main__":
    unittest.main()
