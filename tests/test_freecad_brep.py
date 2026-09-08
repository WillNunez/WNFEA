"""
Unit and Regression Tests for FreeCAD 1.1 B-Rep Cylindrical Face & Bolt Hole Detection.

Verifies:
1. Exact matching of observed hole diameters against ISO 273 and ASME B18.2.8 tables.
2. Full B-Rep analytical cylinder extraction on STEP CAD models via FreeCAD 1.1 kernel.
3. Accurate classification of internal holes vs external bosses/pins.
4. Automatic generation of bolt supports (fixed, pinned, sliding).
5. Cosine-weighted bearing pressure load distribution with total force equilibrium.
6. Multi-point kinematic spider coupling (RBE2) generation for pinned / bolted joints.
"""

import os
import sys
import tempfile
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from wnfea.cad import (
    CylindricalFeature,
    FreeCADBRepDetector,
    match_standard_bolt,
    generate_bolt_supports,
    generate_bearing_loads,
    generate_spider_coupling,
)
from wnfea.boundary.conditions import DOFType


def test_standard_bolt_table_matching():
    """Verify ISO 273 and ASME B18.2.8 standard bolt matching."""
    # Metric ISO 273
    std, nom, fit = match_standard_bolt(6.6)
    assert nom == "M6", f"Expected M6, got {nom}"
    assert fit == "medium", f"Expected medium, got {fit}"
    assert "ISO 273" in std

    std, nom, fit = match_standard_bolt(9.0)
    assert nom == "M8", f"Expected M8, got {nom}"
    assert fit == "medium"

    std, nom, fit = match_standard_bolt(11.0)
    assert nom == "M10", f"Expected M10, got {nom}"

    std, nom, fit = match_standard_bolt(14.5)
    assert nom == "M12", f"Expected M12, got {nom}"
    assert fit == "free"

    # Imperial ASME B18.2.8
    std, nom, fit = match_standard_bolt(6.76)
    assert nom == '1/4"', f"Expected 1/4\", got {nom}"
    assert "ASME B18.2.8" in std

    std, nom, fit = match_standard_bolt(13.49)
    assert nom == '1/2"', f"Expected 1/2\", got {nom}"

    # Non-standard bore (e.g. 52 mm bearing bore)
    std, nom, fit = match_standard_bolt(52.0)
    assert nom is None and std is None

    print("  [PASS] test_standard_bolt_table_matching")


def test_freecad_brep_step_detection():
    """Generate a STEP CAD model with FreeCAD and detect its cylindrical features."""
    detector = FreeCADBRepDetector()
    if not detector.freecad_cmd:
        print("  [SKIP] FreeCAD binary not found; skipping live CAD extraction.")
        return

    step_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "fixtures", "test_plate.step"))
    if not os.path.exists(step_file):
        print(f"  [SKIP] STEP fixture {step_file} not found.")
        return

    features = detector.detect_cylinders(step_file)
    assert len(features) >= 3, f"Expected at least 3 cylinders, found {len(features)}"

    # Separate internal holes vs external boss
    internal_holes = [f for f in features if f.is_internal]
    external_bosses = [f for f in features if not f.is_internal]

    assert len(internal_holes) == 2, f"Expected 2 internal holes, got {len(internal_holes)}"
    assert len(external_bosses) >= 1, f"Expected at least 1 external boss, got {len(external_bosses)}"

    # Verify M6 hole
    m6_hole = next((h for h in internal_holes if h.nominal_bolt_size == "M6"), None)
    assert m6_hole is not None, "M6 bolt hole was not identified!"
    assert np.isclose(m6_hole.radius, 3.3, atol=0.05)
    assert np.isclose(m6_hole.center[0], 15.0, atol=0.5)
    assert np.isclose(m6_hole.center[1], 25.0, atol=0.5)

    # Verify M8 hole
    m8_hole = next((h for h in internal_holes if h.nominal_bolt_size == "M8"), None)
    assert m8_hole is not None, "M8 bolt hole was not identified!"
    assert np.isclose(m8_hole.radius, 4.5, atol=0.05)
    assert np.isclose(m8_hole.center[0], 35.0, atol=0.5)

    # Verify boss radius = 6.0 mm
    boss_feat = external_bosses[0]
    assert np.isclose(boss_feat.radius, 6.0, atol=0.05)
    assert not boss_feat.is_internal

    print(f"  Detected {len(internal_holes)} standard bolt holes and {len(external_bosses)} bosses.")
    print("  [PASS] test_freecad_brep_step_detection")


def test_automated_boundary_conditions():
    """Verify automatic bolt support generation on cylindrical mesh nodes."""
    feat = CylindricalFeature(
        feature_id=0,
        face_index=1,
        is_internal=True,
        radius=5.0,
        diameter=10.0,
        axis=np.array([0.0, 0.0, 1.0]),
        center=np.array([10.0, 20.0, 0.0]),
        length=15.0,
        area=471.2,
        min_proj=0.0,
        max_proj=15.0,
        nominal_bolt_size="M10",
    )

    # Synthetic mesh nodes: 8 on cylinder surface, 4 outside
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    surf_nodes = []
    for th in angles:
        surf_nodes.append([10.0 + 5.0 * np.cos(th), 20.0 + 5.0 * np.sin(th), 7.5])
    outside_nodes = [
        [0.0, 0.0, 0.0],
        [10.0, 20.0, 0.0],       # On axis, r=0
        [10.0 + 10.0, 20.0, 7.5], # r=10
        [10.0 + 5.0, 20.0, 25.0], # Outside axial range
    ]
    mesh_nodes = np.array(surf_nodes + outside_nodes)

    supports = generate_bolt_supports(feat, mesh_nodes, bc_type="fixed")
    assert len(supports) == 8, f"Expected 8 surface supports, got {len(supports)}"
    assert all(s.ux.dof_type == DOFType.FIXED for s in supports)
    assert all(s.uy.dof_type == DOFType.FIXED for s in supports)
    assert all(s.uz.dof_type == DOFType.FIXED for s in supports)

    print("  [PASS] test_automated_boundary_conditions")


def test_bearing_load_distribution():
    """Verify cosine-weighted bearing pressure loads with exact force equilibrium."""
    feat = CylindricalFeature(
        feature_id=1,
        face_index=2,
        is_internal=True,
        radius=10.0,
        diameter=20.0,
        axis=np.array([0.0, 0.0, 1.0]),
        center=np.array([0.0, 0.0, 0.0]),
        length=20.0,
        area=1256.6,
        min_proj=0.0,
        max_proj=20.0,
        nominal_bolt_size="M20",
    )

    # 16 nodes around cylinder circumference
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    nodes_list = [[10.0 * np.cos(th), 10.0 * np.sin(th), 10.0] for th in angles]
    mesh_nodes = np.array(nodes_list)

    total_force = 10000.0
    f_dir = np.array([0.0, 1.0, 0.0])  # Force in +Y direction

    loads = generate_bearing_loads(feat, mesh_nodes, total_force=total_force, force_direction=f_dir)

    # Contact occurs on +Y half (|theta - pi/2| <= pi/2)
    assert len(loads) > 0, "No bearing loads generated!"
    sum_fy = sum(load.fy for load in loads)
    assert np.isclose(sum_fy, total_force, rtol=1e-10), f"Equilibrium violated: {sum_fy} vs {total_force}"

    # Verify no forces in unloaded half (Y < 0)
    for load in loads:
        assert mesh_nodes[load.node_id, 1] > -1e-6, "Unloaded side received load!"

    print(f"  Bearing load equilibrium verified: sum(Fy) = {sum_fy:.2f} N (target: {total_force} N)")
    print("  [PASS] test_bearing_load_distribution")


def test_spider_coupling_synthesis():
    """Verify RBE2 rigid spider coupling generation."""
    feat = CylindricalFeature(
        feature_id=2,
        face_index=3,
        is_internal=True,
        radius=6.0,
        diameter=12.0,
        axis=np.array([0.0, 0.0, 1.0]),
        center=np.array([0.0, 0.0, 0.0]),
        length=10.0,
        area=376.9,
        min_proj=0.0,
        max_proj=10.0,
    )

    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    mesh_nodes = np.array([[6.0 * np.cos(th), 6.0 * np.sin(th), 5.0] for th in angles])

    coupling = generate_spider_coupling(feat, mesh_nodes, master_node_id=100)
    assert coupling.master_node_id == 100
    assert len(coupling.slave_node_ids) == 6
    assert "SpiderCoupling" in coupling.label

    print("  [PASS] test_spider_coupling_synthesis")


def run_all():
    print("=" * 60)
    print("Running FreeCAD 1.1 B-Rep Feature & Bolt Detection Tests...")
    print("=" * 60)
    test_standard_bolt_table_matching()
    test_freecad_brep_step_detection()
    test_automated_boundary_conditions()
    test_bearing_load_distribution()
    test_spider_coupling_synthesis()
    print("=" * 60)
    print("ALL FREECAD B-REP TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
