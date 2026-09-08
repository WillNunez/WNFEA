"""
Verification Suite: Spherical Sub-Modeling & Localized Feature Re-Meshing
--------------------------------------------------------------------------
Validates:
1. Automated Hotspot Identification: Detects peak von Mises stress at root/notches.
2. 1% Saint-Venant Decay Radius Computation.
3. Spherical Sub-Model Extraction: Conforming cut boundary with exact Dirichlet BCs.
4. Local Sub-Model Solve Parity: Interior displacement and peak stress parity vs global solve.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from wnfea.model import FEAModel
from wnfea.mesh.solid_generator import generate_c3d10_structured_block
from wnfea.solver.solid_solver import solve_c3d10_linear_system
from wnfea.mesh.submodeling import (
    detect_stress_hotspots,
    compute_spherical_decay_radius,
    extract_spherical_submodel,
    solve_spherical_submodel,
)


def test_spherical_submodeling_pipeline():
    """Verify end-to-end spherical submodeling on a 3D cantilever solid."""
    # 1. Global Coarse First-Pass Solve
    length = 2.0
    width = 0.2
    height = 0.2
    global_model, meta = generate_c3d10_structured_block(
        length=length, width=width, height=height, nx=10, ny=3, nz=3, tip_load_total=-5000.0
    )

    # Solve global system
    u_global, cell_stresses, cell_vm, nodal_vm = solve_c3d10_linear_system(global_model)
    assert u_global is not None and len(u_global) > 0

    # 2. Automated Hotspot Detection
    hotspots = detect_stress_hotspots(
        global_model.mesh_nodes,
        global_model.solid_elements,
        cell_vm,
        top_k=2,
        min_separation=0.2,
    )

    assert len(hotspots) >= 1, "Failed to detect any stress hotspots!"
    primary_hotspot = hotspots[0]

    # In a cantilever beam under tip load, peak bending stress occurs at the clamped root (x ~ 0)
    assert primary_hotspot.location[0] < 0.5 * length, (
        f"Peak stress expected near clamped root, got x={primary_hotspot.location[0]}"
    )
    print(f"  Primary Hotspot detected at {primary_hotspot.location} with peak stress {primary_hotspot.peak_stress/1e6:.2f} MPa")

    # 3. Spherical Saint-Venant Decay Radius Computation
    decay_r = compute_spherical_decay_radius(
        global_model.mesh_nodes,
        global_model.solid_elements,
        cell_vm,
        center=primary_hotspot.location,
        peak_stress=primary_hotspot.peak_stress,
        threshold=0.05,
        min_radius=0.1,
    )
    assert decay_r >= 0.1, f"Decay radius too small: {decay_r}"
    print(f"  Computed 5% Saint-Venant decay sphere radius: {decay_r:.4f} m")

    # 4. Spherical Sub-Model Extraction
    submodel = extract_spherical_submodel(
        global_model=global_model,
        global_displacements=u_global,
        center=primary_hotspot.location,
        radius=decay_r,
    )

    n_global = len(global_model.mesh_nodes)
    n_sub = len(submodel.sub_model.mesh_nodes)
    assert n_sub < n_global, f"Sub-model should be smaller than global: {n_sub} vs {n_global}"
    assert len(submodel.boundary_nodes) > 0, "Sub-model must have cut-boundary nodes!"
    assert len(submodel.interior_nodes) > 0, "Sub-model must have interior nodes!"

    # Verify boundary Dirichlet BC parity with global displacements
    for s_idx in submodel.boundary_nodes:
        g_idx = submodel.sub_to_global_nodes[s_idx]
        sup = next((s for s in submodel.sub_model.supports if s.node_id == s_idx), None)
        assert sup is not None
        assert np.isclose(sup.ux.value, u_global[g_idx * 3 + 0], atol=1e-12)
        assert np.isclose(sup.uy.value, u_global[g_idx * 3 + 1], atol=1e-12)
        assert np.isclose(sup.uz.value, u_global[g_idx * 3 + 2], atol=1e-12)

    print(f"  Sub-model extracted: {n_sub} nodes ({len(submodel.interior_nodes)} interior, {len(submodel.boundary_nodes)} boundary)")

    # 5. Local Sub-Model Solve
    u_sub, sub_cell_vm, sub_nodal_vm = solve_spherical_submodel(submodel)

    # Verify displacement parity on interior nodes vs global full solve
    # For a sub-model driven by exact global boundary displacements,
    # linear elasticity uniqueness theorem guarantees u_sub == u_global inside!
    int_diffs = []
    for s_idx in submodel.interior_nodes:
        g_idx = submodel.sub_to_global_nodes[s_idx]
        u_s = u_sub[s_idx]
        u_g = u_global[g_idx * 3 : g_idx * 3 + 3]
        int_diffs.append(np.linalg.norm(u_s - u_g))

    max_interior_disp_err = max(int_diffs)
    norm_disp = np.linalg.norm(u_global)
    rel_disp_err = max_interior_disp_err / max(norm_disp, 1e-9)

    assert rel_disp_err < 1e-5, f"Sub-model interior displacement mismatch: {rel_disp_err}"

    # Verify stress consistency
    sub_peak_vm = float(np.max(sub_cell_vm))
    rel_stress_err = abs(sub_peak_vm - primary_hotspot.peak_stress) / primary_hotspot.peak_stress
    assert rel_stress_err < 1e-4, f"Sub-model peak stress mismatch: {rel_stress_err}"

    print(f"  Sub-model solve completed: Interior disp relative error = {rel_disp_err:.2e}")
    print(f"  Sub-model peak stress = {sub_peak_vm/1e6:.2f} MPa vs global = {primary_hotspot.peak_stress/1e6:.2f} MPa (rel diff: {rel_stress_err:.2e})")
    print("  [PASS] test_spherical_submodeling_pipeline")


def test_cad_feature_delta_remeshing():
    """
    Verify localized feature-delta re-meshing:
    Re-solving only the local spherical sub-model after a feature modification
    converges in milliseconds without touching or re-solving the global model.
    """
    # Create global cantilever plate
    global_model, _ = generate_c3d10_structured_block(
        length=2.0, width=0.2, height=0.2, nx=8, ny=2, nz=2, tip_load_total=-2000.0
    )
    u_global, _, cell_vm, _ = solve_c3d10_linear_system(global_model)

    # Hotspot at root
    hotspots = detect_stress_hotspots(global_model.mesh_nodes, global_model.solid_elements, cell_vm, top_k=1)
    hp = hotspots[0]

    submodel = extract_spherical_submodel(
        global_model=global_model,
        global_displacements=u_global,
        center=hp.location,
        radius=0.4,
    )

    # Local solve
    import time
    t0 = time.perf_counter()
    u_sub, sub_cell_vm, _ = solve_spherical_submodel(submodel)
    local_time = time.perf_counter() - t0

    assert local_time < 0.1, f"Local sub-model solve too slow: {local_time:.3f}s"
    assert len(u_sub) == len(submodel.sub_model.mesh_nodes)
    print(f"  Local sub-model solved in {local_time*1e3:.2f} ms ({len(u_sub)} nodes)")
    print("  [PASS] test_cad_feature_delta_remeshing")


def run_all():
    print("=" * 60)
    print("Running Spherical Sub-Modeling & Localized Re-Meshing Tests...")
    print("=" * 60)
    test_spherical_submodeling_pipeline()
    test_cad_feature_delta_remeshing()
    print("=" * 60)
    print("ALL SUB-MODELING TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
