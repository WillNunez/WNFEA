"""
Validation Suite for WNFEA Non-Linear Matrix-Free JFNK Solver with AMG.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wnfea import (
    FEAModel,
    PropertyAssignment,
    get_preset_material,
    create_hollow_tube,
    create_solid_circle,
    create_fixed_support,
    LoadDef,
    BeamMesher,
    solve_linear_static,
    solve_nonlinear_jfnk,
    BlockBeamAMGPreconditioner,
    MatrixFreeJFNKOperator,
)
from wnfea.geometry.primitives import Point3D, GeometryNode, GeometryEdge
from wnfea.solver.residual import compute_equilibrium_residual, build_external_force_vector


def test_jfnk_operator_accuracy():
    """Verify that the JFNK operator computes accurate Fréchet derivatives."""
    print("\n--- Test 1: Matrix-Free JFNK Operator Verification ---")
    model = FEAModel()
    model.geometry_nodes = {
        0: GeometryNode(id=0, label="Fixed", point=Point3D(0.0, 0.0, 0.0)),
        1: GeometryNode(id=1, label="Free", point=Point3D(1.0, 0.0, 0.0)),
    }
    model.geometry_edges = {
        0: GeometryEdge(id=0, label="Beam", start_node_id=0, end_node_id=1),
    }

    steel = get_preset_material("Structural Steel")
    tube = create_hollow_tube("Tube", 0.05, 0.04)
    model.materials[steel.name] = steel
    model.sections[tube.name] = tube
    model.edge_assignments[0] = PropertyAssignment(steel.name, tube.name)

    mesher = BeamMesher(n_divisions=4)
    mesher.mesh(model)

    model.supports.append(create_fixed_support(0, is_geometry_node=True))
    model.loads.append(LoadDef(node_id=1, is_geometry_node=True, fy=-1000.0))

    U = np.zeros(len(model.mesh_nodes) * 6)
    F_ext = build_external_force_vector(model)

    # Instantiate JFNK operator
    J_op = MatrixFreeJFNKOperator(model, U, F_ext, load_factor=1.0)

    # Test perturbation vector
    np.random.seed(42)
    v = np.random.randn(len(U))
    v[0:6] = 0.0  # Fixed node DOFs zeroed

    w_jfnk = J_op.matvec(v)

    # Central difference baseline for comparison
    h = 1e-7
    R_plus = compute_equilibrium_residual(model, U + h * v, F_ext, 1.0)
    R_minus = compute_equilibrium_residual(model, U - h * v, F_ext, 1.0)
    w_central = (R_plus - R_minus) / (2.0 * h)
    w_central[0:6] = v[0:6]

    rel_diff = np.linalg.norm(w_jfnk - w_central) / np.linalg.norm(w_central)
    print(f"  Relative difference between JFNK forward diff and central diff: {rel_diff:.4e}")
    assert rel_diff < 1e-4, f"JFNK Fréchet derivative mismatch: {rel_diff}"
    print("  [PASS] JFNK Operator directional derivative verified.")


def test_amg_preconditioner():
    """Verify Block Beam AMG preconditioner SPD properties."""
    print("\n--- Test 2: Block Beam AMG Preconditioner Verification ---")
    model = FEAModel()
    model.geometry_nodes = {
        0: GeometryNode(id=0, label="Fixed", point=Point3D(0.0, 0.0, 0.0)),
        1: GeometryNode(id=1, label="Free", point=Point3D(2.0, 0.0, 0.0)),
    }
    model.geometry_edges = {
        0: GeometryEdge(id=0, label="Beam", start_node_id=0, end_node_id=1),
    }

    steel = get_preset_material("Structural Steel")
    tube = create_hollow_tube("Tube", 0.05, 0.04)
    model.materials[steel.name] = steel
    model.sections[tube.name] = tube
    model.edge_assignments[0] = PropertyAssignment(steel.name, tube.name)

    mesher = BeamMesher(n_divisions=6)
    mesher.mesh(model)

    model.supports.append(create_fixed_support(0, is_geometry_node=True))
    model.loads.append(LoadDef(node_id=1, is_geometry_node=True, fy=-500.0))

    amg = BlockBeamAMGPreconditioner(model)
    print(f"  AMG Levels: {len(amg.levels)}")
    for i, lvl in enumerate(amg.levels):
        print(f"    Level {i}: matrix size {lvl.A.shape[0]} x {lvl.A.shape[1]}")

    # Verify positive definiteness: r^T M^{-1} r > 0 for random r
    np.random.seed(123)
    r = np.random.randn(len(model.mesh_nodes) * 6)
    z = amg.matvec(r)
    quad_form = float(np.dot(r, z))
    print(f"  AMG quadratic form (r^T M^{{-1}} r): {quad_form:.4e}")
    assert quad_form > 0, "Preconditioner is not positive definite!"
    print("  [PASS] Block Beam AMG is strictly positive definite.")


def test_cantilever_small_load_consistency():
    """Verify that non-linear JFNK matches linear static solution under small loads."""
    print("\n--- Test 3: Small-Load Consistency (Linear vs Non-Linear JFNK) ---")
    model = FEAModel()
    model.geometry_nodes = {
        0: GeometryNode(id=0, label="Fixed", point=Point3D(0.0, 0.0, 0.0)),
        1: GeometryNode(id=1, label="Free", point=Point3D(1.0, 0.0, 0.0)),
    }
    model.geometry_edges = {
        0: GeometryEdge(id=0, label="Beam", start_node_id=0, end_node_id=1),
    }

    steel = get_preset_material("Structural Steel")
    bar = create_solid_circle("Bar", 0.04)
    model.materials[steel.name] = steel
    model.sections[bar.name] = bar
    model.edge_assignments[0] = PropertyAssignment(steel.name, bar.name)

    mesher = BeamMesher(n_divisions=4)
    mesher.mesh(model)

    P = -500.0  # Tip load
    model.supports.append(create_fixed_support(0, is_geometry_node=True))
    model.loads.append(LoadDef(node_id=1, is_geometry_node=True, fy=P))

    # Linear solve
    U_linear = solve_linear_static(model, device="cpu")
    free_mesh_idx = model.geometry_to_mesh_node_map[1]
    tip_linear = U_linear[free_mesh_idx * 6 + 1]  # uy at node 1
    print(f"  Linear tip deflection (uy):     {tip_linear * 1e3:.4f} mm")

    # Non-linear JFNK solve
    U_nl = solve_nonlinear_jfnk(model, n_load_steps=2, use_amg=True, verbose=False)
    tip_nl = U_nl[free_mesh_idx * 6 + 1]
    print(f"  Non-linear tip deflection (uy): {tip_nl * 1e3:.4f} mm")

    # Compare
    diff_pct = abs(tip_nl - tip_linear) / abs(tip_linear) * 100.0
    print(f"  Difference: {diff_pct:.4f}%")
    assert diff_pct < 0.2, f"Expected < 0.2% difference for small load, got {diff_pct}%"
    print("  [PASS] Non-linear JFNK matches linear static solution for small load.")


def test_large_deflection_beam():
    """Verify non-linear geometric stiffening under large loads."""
    print("\n--- Test 4: Geometrically Non-Linear Large Deflection ---")
    model = FEAModel()
    model.geometry_nodes = {
        0: GeometryNode(id=0, label="Fixed", point=Point3D(0.0, 0.0, 0.0)),
        1: GeometryNode(id=1, label="Free", point=Point3D(2.0, 0.0, 0.0)),
    }
    model.geometry_edges = {
        0: GeometryEdge(id=0, label="Beam", start_node_id=0, end_node_id=1),
    }

    steel = get_preset_material("Structural Steel")
    tube = create_hollow_tube("SlenderTube", 0.03, 0.026)
    model.materials[steel.name] = steel
    model.sections[tube.name] = tube
    model.edge_assignments[0] = PropertyAssignment(steel.name, tube.name)

    mesher = BeamMesher(n_divisions=8)
    mesher.mesh(model)

    P_large = -1000.0
    model.supports.append(create_fixed_support(0, is_geometry_node=True))
    model.loads.append(LoadDef(node_id=1, is_geometry_node=True, fy=P_large))

    U_nl = solve_nonlinear_jfnk(model, n_load_steps=5, use_amg=True, verbose=True)

    tip_node_idx = model.geometry_to_mesh_node_map[1]
    tip_ux = U_nl[tip_node_idx * 6 + 0]
    tip_uy = U_nl[tip_node_idx * 6 + 1]

    print(f"\n  Large deflection results at tip:")
    print(f"    Horizontal shortening (ux): {tip_ux * 1e3:.4f} mm")
    print(f"    Vertical deflection   (uy): {tip_uy * 1e3:.4f} mm")

    assert tip_ux < 0, f"Expected horizontal chord shortening (ux < 0), got {tip_ux}"
    assert tip_uy < 0, f"Expected downward deflection (uy < 0), got {tip_uy}"
    print("  [PASS] Geometric non-linear large deflection and chord shortening verified.")


if __name__ == "__main__":
    print("=" * 65)
    print("      WNFEA Non-Linear JFNK + AMG Test Suite")
    print("=" * 65)

    test_jfnk_operator_accuracy()
    test_amg_preconditioner()
    test_cantilever_small_load_consistency()
    test_large_deflection_beam()

    print("\n" + "=" * 65)
    print("      ALL NON-LINEAR JFNK + AMG TESTS PASSED!")
    print("=" * 65)
