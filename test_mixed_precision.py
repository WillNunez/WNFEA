"""
Mixed-Precision Iterative Refinement Validation Suite for WNFEA.

Verifies that:
1. FP32 inner solve + FP64 outer residual converges to the same accuracy as pure FP64.
2. AMG hierarchy and PCG working vectors actually operate in the requested dtype.
3. Non-linear large-deflection problems converge with mixed precision.
4. PrecisionConfig utility functions work correctly.
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
    solve_nonlinear_jfnk,
    BlockBeamAMGPreconditioner,
    MatrixFreeJFNKOperator,
    PrecisionConfig,
    MIXED_FP32,
    FULL_FP64,
)
from wnfea.geometry.primitives import Point3D, GeometryNode, GeometryEdge
from wnfea.solver.residual import compute_equilibrium_residual, build_external_force_vector
from wnfea.solver.nonlinear_solver import pcg_solve
from wnfea.solver.mixed_precision import cast_to_inner, cast_to_outer


def _make_cantilever_model(length=1.0, n_divs=4, load_fy=-500.0):
    """Helper: build a standard cantilever beam model."""
    model = FEAModel()
    model.geometry_nodes = {
        0: GeometryNode(id=0, label="Fixed", point=Point3D(0.0, 0.0, 0.0)),
        1: GeometryNode(id=1, label="Free", point=Point3D(length, 0.0, 0.0)),
    }
    model.geometry_edges = {
        0: GeometryEdge(id=0, label="Beam", start_node_id=0, end_node_id=1),
    }

    steel = get_preset_material("Structural Steel")
    tube = create_hollow_tube("Tube", 0.05, 0.04)
    model.materials[steel.name] = steel
    model.sections[tube.name] = tube
    model.edge_assignments[0] = PropertyAssignment(steel.name, tube.name)

    mesher = BeamMesher(n_divisions=n_divs)
    mesher.mesh(model)

    model.supports.append(create_fixed_support(0, is_geometry_node=True))
    model.loads.append(LoadDef(node_id=1, is_geometry_node=True, fy=load_fy))

    return model


def test_precision_config():
    """Test PrecisionConfig utility class and cast functions."""
    print("\n--- Test 1: PrecisionConfig Utilities ---")

    # Default mixed precision config
    cfg = MIXED_FP32
    assert cfg.enabled is True
    assert cfg.inner_dtype == np.float32
    assert cfg.outer_dtype == np.float64
    assert cfg.working_dtype == np.dtype(np.float32)
    print(f"  MIXED_FP32: {cfg.summary()}")

    # FP32 eps_scale should be ~3.45e-4 (sqrt(~1.19e-7))
    assert 1e-4 < cfg.inner_eps_scale < 1e-3, f"Unexpected FP32 eps_scale: {cfg.inner_eps_scale}"
    print(f"  FP32 eps_scale: {cfg.inner_eps_scale:.6e}")

    # FP64 eps_scale should be ~1.49e-8 (sqrt(~2.22e-16))
    assert 1e-9 < cfg.outer_eps_scale < 1e-7, f"Unexpected FP64 eps_scale: {cfg.outer_eps_scale}"
    print(f"  FP64 eps_scale: {cfg.outer_eps_scale:.6e}")

    # Full FP64 config
    cfg64 = FULL_FP64
    assert cfg64.enabled is False
    assert cfg64.working_dtype == np.dtype(np.float64)
    print(f"  FULL_FP64: {cfg64.summary()}")

    # Cast functions
    arr64 = np.ones(5, dtype=np.float64)
    arr32 = cast_to_inner(arr64, cfg)
    assert arr32.dtype == np.float32, f"Expected float32, got {arr32.dtype}"

    arr_back = cast_to_outer(arr32, cfg)
    assert arr_back.dtype == np.float64, f"Expected float64, got {arr_back.dtype}"

    # Identity cast (no-op)
    arr64_same = cast_to_inner(arr64, cfg64)
    assert arr64_same.dtype == np.float64

    print("  [PASS] PrecisionConfig utilities verified.")


def test_amg_fp32_hierarchy():
    """Verify AMG hierarchy stores matrices in FP32 when working_dtype=float32."""
    print("\n--- Test 2: AMG FP32 Hierarchy Verification ---")

    model = _make_cantilever_model(length=2.0, n_divs=6, load_fy=-500.0)

    # Build AMG in FP32
    amg_fp32 = BlockBeamAMGPreconditioner(model, working_dtype=np.float32)

    # Verify all level matrices are in float32
    for i, lvl in enumerate(amg_fp32.levels):
        assert lvl.A.dtype == np.float32, f"Level {i} A matrix dtype is {lvl.A.dtype}, expected float32"
        assert lvl.inv_diag.dtype == np.float32, f"Level {i} inv_diag dtype is {lvl.inv_diag.dtype}"
        if lvl.P is not None:
            assert lvl.P.dtype == np.float32, f"Level {i} P dtype is {lvl.P.dtype}"
            assert lvl.R.dtype == np.float32, f"Level {i} R dtype is {lvl.R.dtype}"
        print(f"  Level {i}: A={lvl.A.shape}, dtype={lvl.A.dtype} OK")

    # Verify positive definiteness with FP32 hierarchy
    np.random.seed(42)
    r = np.random.randn(len(model.mesh_nodes) * 6)
    z = amg_fp32.matvec(r)
    assert z.dtype == np.float64, f"AMG output should be float64, got {z.dtype}"
    quad = float(np.dot(r, z))
    assert quad > 0, f"FP32 AMG not positive definite: r^T M^-1 r = {quad}"
    print(f"  FP32 AMG quadratic form: {quad:.4e} (positive definite OK)")

    # Compare with FP64 AMG
    amg_fp64 = BlockBeamAMGPreconditioner(model, working_dtype=np.float64)
    z64 = amg_fp64.matvec(r)
    rel_diff = np.linalg.norm(z - z64) / np.linalg.norm(z64)
    print(f"  FP32 vs FP64 AMG relative difference: {rel_diff:.4e}")
    assert rel_diff < 1e-3, f"FP32 and FP64 AMG differ too much: {rel_diff}"

    print("  [PASS] AMG FP32 hierarchy verified.")


def test_jfnk_operator_fp32():
    """Verify JFNK operator produces reasonable matvecs in FP32 mode."""
    print("\n--- Test 3: JFNK Operator FP32 Matvec ---")

    model = _make_cantilever_model(length=1.0, n_divs=4, load_fy=-1000.0)

    from wnfea.solver.dof_manager import DOFManager
    dof_mgr = DOFManager(model)
    n_dofs = dof_mgr.total_active_dofs

    U = np.zeros(n_dofs, dtype=np.float64)
    F_ext = build_external_force_vector(model, dof_mgr)

    # FP64 JFNK operator
    J_fp64 = MatrixFreeJFNKOperator(
        model, U, F_ext, load_factor=1.0, dof_mgr=dof_mgr,
        working_dtype=np.float64,
    )

    # FP32 JFNK operator
    J_fp32 = MatrixFreeJFNKOperator(
        model, U, F_ext, load_factor=1.0, dof_mgr=dof_mgr,
        working_dtype=np.float32,
    )

    # Verify internal state precision
    assert J_fp32.U.dtype == np.float32, f"JFNK U should be float32, got {J_fp32.U.dtype}"
    assert J_fp32.F_ext.dtype == np.float32, f"JFNK F_ext should be float32, got {J_fp32.F_ext.dtype}"
    assert J_fp32.R_current.dtype == np.float32, f"JFNK R_current should be float32"
    print(f"  JFNK FP32 eps_scale: {J_fp32.eps_scale:.6e}")
    print(f"  JFNK FP64 eps_scale: {J_fp64.eps_scale:.6e}")

    # Compare matvec outputs
    np.random.seed(42)
    v = np.random.randn(n_dofs)

    w64 = J_fp64.matvec(v)
    w32 = J_fp32.matvec(v)

    # Both outputs should be FP64 (LinearOperator contract)
    assert w64.dtype == np.float64, f"FP64 matvec output dtype: {w64.dtype}"
    assert w32.dtype == np.float64, f"FP32 matvec output dtype: {w32.dtype}"

    rel_diff = np.linalg.norm(w32 - w64) / np.linalg.norm(w64)
    print(f"  FP32 vs FP64 matvec relative difference: {rel_diff:.4e}")
    # FP32 finite difference will have larger error (~1e-3) than FP64 (~1e-8)
    assert rel_diff < 0.05, f"JFNK FP32 vs FP64 matvec too different: {rel_diff}"

    print("  [PASS] JFNK FP32 matvec verified.")


def test_pcg_fp32_solve():
    """Verify PCG solver works correctly with FP32 working vectors."""
    print("\n--- Test 4: PCG FP32 Working Vectors ---")

    model = _make_cantilever_model(length=1.0, n_divs=4, load_fy=-1000.0)

    from wnfea.solver.dof_manager import DOFManager
    dof_mgr = DOFManager(model)
    n_dofs = dof_mgr.total_active_dofs

    U = np.zeros(n_dofs, dtype=np.float64)
    F_ext = build_external_force_vector(model, dof_mgr)
    R = compute_equilibrium_residual(
        model, U, F_ext, load_factor=1.0, dof_mgr=dof_mgr
    )

    # Build operators in FP32 and FP64
    J_fp32 = MatrixFreeJFNKOperator(
        model, U, F_ext, load_factor=1.0, R_current=R, dof_mgr=dof_mgr,
        working_dtype=np.float32,
    )
    J_fp64 = MatrixFreeJFNKOperator(
        model, U, F_ext, load_factor=1.0, R_current=R, dof_mgr=dof_mgr,
        working_dtype=np.float64,
    )

    amg_fp32 = BlockBeamAMGPreconditioner(model, working_dtype=np.float32)
    amg_fp64 = BlockBeamAMGPreconditioner(model, working_dtype=np.float64)

    rhs = -R

    # FP64 PCG
    x64, its64, res64 = pcg_solve(J_fp64, rhs, M_prec=amg_fp64, tol=1e-6, max_iter=100,
                                    working_dtype=np.float64)

    # FP32 PCG
    x32, its32, res32 = pcg_solve(J_fp32, rhs, M_prec=amg_fp32, tol=1e-6, max_iter=100,
                                    working_dtype=np.float32)

    # Both outputs should be float64
    assert x64.dtype == np.float64, f"FP64 PCG output dtype: {x64.dtype}"
    assert x32.dtype == np.float64, f"FP32 PCG output dtype: {x32.dtype}"

    rel_diff = np.linalg.norm(x32 - x64) / (np.linalg.norm(x64) + 1e-20)
    print(f"  FP64 PCG: {its64} iters, residual {res64:.4e}")
    print(f"  FP32 PCG: {its32} iters, residual {res32:.4e}")
    print(f"  Solution relative difference: {rel_diff:.4e}")

    # FP32 inner solve should give a reasonable direction (not exact, but convergent)
    assert rel_diff < 0.1, f"FP32 and FP64 PCG solutions differ too much: {rel_diff}"

    print("  [PASS] PCG FP32 working vectors verified.")


def test_mixed_precision_small_load_accuracy():
    """
    Verify that mixed-precision iterative refinement produces FP64-accurate results.
    Compare mixed-precision JFNK solution to pure FP64 JFNK solution under small load.
    """
    print("\n--- Test 5: Mixed-Precision Accuracy (Small Load) ---")

    # Pure FP64 solve
    model_fp64 = _make_cantilever_model(length=1.0, n_divs=4, load_fy=-500.0)
    U_fp64 = solve_nonlinear_jfnk(
        model_fp64, n_load_steps=2, use_amg=True,
        use_mixed_precision=False, verbose=False,
    )

    # Mixed-precision solve (FP32 inner, FP64 outer)
    model_mp = _make_cantilever_model(length=1.0, n_divs=4, load_fy=-500.0)
    U_mp = solve_nonlinear_jfnk(
        model_mp, n_load_steps=2, use_amg=True,
        use_mixed_precision=True, verbose=False,
    )

    tip_fp64 = model_fp64.geometry_to_mesh_node_map[1]
    tip_uy_fp64 = U_fp64[tip_fp64 * 6 + 1]
    tip_uy_mp = U_mp[tip_fp64 * 6 + 1]

    rel_err = abs(tip_uy_mp - tip_uy_fp64) / abs(tip_uy_fp64)
    print(f"  FP64 tip deflection:  {tip_uy_fp64 * 1e3:.6f} mm")
    print(f"  Mixed tip deflection: {tip_uy_mp * 1e3:.6f} mm")
    print(f"  Relative error: {rel_err:.4e}")

    # Full displacement vector comparison
    norm_diff = np.linalg.norm(U_mp - U_fp64) / np.linalg.norm(U_fp64)
    print(f"  Full solution relative difference: {norm_diff:.4e}")

    assert rel_err < 1e-4, f"Mixed precision tip deflection too different: {rel_err}"
    assert norm_diff < 1e-3, f"Full solution too different: {norm_diff}"

    print("  [PASS] Mixed-precision matches FP64 to required accuracy.")


def test_mixed_precision_large_deflection():
    """
    Verify convergence and accuracy of mixed-precision under large-deflection loading.
    """
    print("\n--- Test 6: Mixed-Precision Large Deflection ---")

    # FP64 reference solve
    model_fp64 = _make_cantilever_model(length=2.0, n_divs=8, load_fy=-1000.0)
    model_fp64.materials[list(model_fp64.materials.keys())[0]] = get_preset_material("Structural Steel")
    tube = create_hollow_tube("SlenderTube", 0.03, 0.026)
    model_fp64.sections[tube.name] = tube
    for eid in model_fp64.edge_assignments:
        model_fp64.edge_assignments[eid] = PropertyAssignment(
            list(model_fp64.materials.keys())[0], tube.name
        )
    mesher = BeamMesher(n_divisions=8)
    mesher.mesh(model_fp64)
    model_fp64.supports.clear()
    model_fp64.loads.clear()
    model_fp64.supports.append(create_fixed_support(0, is_geometry_node=True))
    model_fp64.loads.append(LoadDef(node_id=1, is_geometry_node=True, fy=-1000.0))

    U_fp64 = solve_nonlinear_jfnk(
        model_fp64, n_load_steps=5, use_amg=True,
        use_mixed_precision=False, verbose=False,
    )

    # Mixed-precision solve (rebuild model to avoid state carryover)
    model_mp = _make_cantilever_model(length=2.0, n_divs=8, load_fy=-1000.0)
    model_mp.materials[list(model_mp.materials.keys())[0]] = get_preset_material("Structural Steel")
    model_mp.sections[tube.name] = tube
    for eid in model_mp.edge_assignments:
        model_mp.edge_assignments[eid] = PropertyAssignment(
            list(model_mp.materials.keys())[0], tube.name
        )
    mesher.mesh(model_mp)
    model_mp.supports.clear()
    model_mp.loads.clear()
    model_mp.supports.append(create_fixed_support(0, is_geometry_node=True))
    model_mp.loads.append(LoadDef(node_id=1, is_geometry_node=True, fy=-1000.0))

    U_mp = solve_nonlinear_jfnk(
        model_mp, n_load_steps=5, use_amg=True,
        use_mixed_precision=True, verbose=True,
    )

    tip_idx = model_fp64.geometry_to_mesh_node_map[1]
    tip_ux_fp64 = U_fp64[tip_idx * 6 + 0]
    tip_uy_fp64 = U_fp64[tip_idx * 6 + 1]
    tip_ux_mp = U_mp[tip_idx * 6 + 0]
    tip_uy_mp = U_mp[tip_idx * 6 + 1]

    print(f"\n  FP64 tip:  ux={tip_ux_fp64*1e3:.4f} mm, uy={tip_uy_fp64*1e3:.4f} mm")
    print(f"  Mixed tip: ux={tip_ux_mp*1e3:.4f} mm, uy={tip_uy_mp*1e3:.4f} mm")

    norm_diff = np.linalg.norm(U_mp - U_fp64) / np.linalg.norm(U_fp64)
    print(f"  Full solution relative difference: {norm_diff:.4e}")

    # For large deflection, mixed precision may have slightly larger error
    # due to FP32 inner solve accuracy, but should still be < 1%
    assert norm_diff < 0.01, f"Mixed precision large deflection too different: {norm_diff}"
    assert tip_ux_mp < 0, f"Expected horizontal shortening, got ux={tip_ux_mp}"
    assert tip_uy_mp < 0, f"Expected downward deflection, got uy={tip_uy_mp}"

    print("  [PASS] Mixed-precision large deflection converged and matched FP64 reference.")


def test_tri_precision_adaptive_preconditioner():
    """
    Verify tri-precision operation:
    1. AMG preconditioner executes in FP16 for early high-residual iterations,
       then automatically switches to FP32 as residual drops below switch_tol.
    2. Large-deflection solve converges with tri-precision and matches FP64 reference.
    """
    print("\n--- Test 7: Tri-Precision Adaptive Preconditioner (FP16 -> FP32 -> FP64) ---")

    model = _make_cantilever_model(length=2.0, n_divs=6, load_fy=-500.0)

    # 1. Test direct AMG adaptive precision switching
    amg = BlockBeamAMGPreconditioner(model, working_dtype=np.float32)
    amg.enable_tri_precision(switch_tol=1e-2)

    np.random.seed(42)
    r = np.random.randn(len(model.mesh_nodes) * 6)

    # Coarse residual (rel_res = 0.5 > 1e-2) -> should execute in FP16
    z_fp16 = amg.apply_adaptive(r, current_res_rel=0.5)
    print(f"  Early AMG call (rel_res=0.5): used {amg.last_precision_used}")
    assert amg.last_precision_used == "float16", f"Expected float16, got {amg.last_precision_used}"
    assert z_fp16.dtype == np.float64

    # Refined residual (rel_res = 1e-4 <= 1e-2) -> should execute in FP32
    z_fp32 = amg.apply_adaptive(r, current_res_rel=1e-4)
    print(f"  Late AMG call (rel_res=1e-4):  used {amg.last_precision_used}")
    assert amg.last_precision_used == "float32", f"Expected float32, got {amg.last_precision_used}"
    assert z_fp32.dtype == np.float64

    # Positive definiteness check in FP16 mode
    quad_fp16 = float(np.dot(r, z_fp16))
    print(f"  FP16 AMG quadratic form: {quad_fp16:.4e}")
    assert quad_fp16 > 0, "FP16 AMG preconditioner is not positive definite!"

    # 2. End-to-end tri-precision nonlinear solve
    model_tri = _make_cantilever_model(length=1.0, n_divs=4, load_fy=-500.0)
    U_tri = solve_nonlinear_jfnk(
        model_tri, n_load_steps=2, use_amg=True,
        use_tri_precision=True, switch_tol=1e-2, verbose=True,
    )

    # Reference FP64 solve
    model_ref = _make_cantilever_model(length=1.0, n_divs=4, load_fy=-500.0)
    U_ref = solve_nonlinear_jfnk(
        model_ref, n_load_steps=2, use_amg=True,
        use_mixed_precision=False, verbose=False,
    )

    tip_idx = model_tri.geometry_to_mesh_node_map[1]
    tip_uy_tri = U_tri[tip_idx * 6 + 1]
    tip_uy_ref = U_ref[tip_idx * 6 + 1]

    rel_err = abs(tip_uy_tri - tip_uy_ref) / abs(tip_uy_ref)
    print(f"\n  Tri-precision tip deflection: {tip_uy_tri * 1e3:.6f} mm")
    print(f"  FP64 reference tip:          {tip_uy_ref * 1e3:.6f} mm")
    print(f"  Relative error: {rel_err:.4e}")

    assert rel_err < 1e-4, f"Tri-precision solution differs too much from FP64: {rel_err}"
    print("  [PASS] Tri-precision adaptive preconditioning (FP16 -> FP32 -> FP64) verified.")


if __name__ == "__main__":
    print("=" * 65)
    print("      WNFEA Mixed-Precision & Tri-Precision Test Suite")
    print("=" * 65)

    test_precision_config()
    test_amg_fp32_hierarchy()
    test_jfnk_operator_fp32()
    test_pcg_fp32_solve()
    test_mixed_precision_small_load_accuracy()
    test_mixed_precision_large_deflection()
    test_tri_precision_adaptive_preconditioner()

    print("\n" + "=" * 65)
    print("      ALL MIXED & TRI-PRECISION TESTS PASSED!")
    print("=" * 65)
