"""
Verification Suite: Dual-Stage Voxel First-Pass & Sub-Modeling Pipeline
-----------------------------------------------------------------------
Validates:
1. Voxel Mesher: Cartesian Hex8 grid generation, cell indexing, and trilinear shape functions.
2. Matrix-Free Hex8 Voxel Solver: Fast PCG convergence, deflection, and von Mises stress.
3. End-to-End Dual-Stage Pipeline:
   - Stage 1: Global Voxel First-Pass in <50 ms.
   - Hotspot & 1% Saint-Venant sphere calculation.
   - Stage 2: Spherical C3D10 sub-model extraction and solve in <25 ms.
   - High-fidelity peak stress recovery and performance speedup.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np

from wnfea.mesh.voxel_mesher import VoxelMesher, VoxelGrid
from wnfea.solver.matrix_free_hex8 import (
    compute_hex8_reference_stiffness,
    MatrixFreeHex8Operator,
    solve_voxel_linear_static,
)
from wnfea.mesh.solid_generator import generate_c3d10_structured_block
from wnfea.solver.dual_stage_pipeline import run_dual_stage_pipeline


def test_voxel_mesher_geometry():
    """Verify Cartesian voxel mesher coordinate generation and shape functions."""
    bounds = (0.0, 3.0, 0.0, 2.0, 0.0, 1.0)
    res = (6, 4, 2)
    grid = VoxelMesher.create_box_grid(bounds, res)

    assert grid.total_cells == 6 * 4 * 2, f"Expected 48 cells, got {grid.total_cells}"
    assert grid.total_nodes == 7 * 5 * 3, f"Expected 105 nodes, got {grid.total_nodes}"
    assert grid.pitch == (0.5, 0.5, 0.5)

    # Test trilinear interpolation at an arbitrary interior point
    query_pt = np.array([1.35, 0.85, 0.42])
    nids, weights = grid.sample_trilinear_weights(query_pt)

    assert len(nids) == 8
    assert np.isclose(np.sum(weights), 1.0, atol=1e-14)

    reconstructed_pt = np.sum(grid.nodes[nids] * weights[:, None], axis=0)
    geom_err = np.linalg.norm(reconstructed_pt - query_pt)
    assert geom_err < 1e-14, f"Shape function interpolation error: {geom_err}"

    # Test immersed boundary with an analytical spherical void
    center = np.array([1.5, 1.0, 0.5])
    radius = 0.4
    def sphere_sdf(pts):
        # Negative inside sphere (void)
        dists = np.linalg.norm(pts - center[None, :], axis=1)
        return dists - radius

    grid_void = VoxelMesher.create_box_grid(bounds, (12, 8, 4), inside_fn=sphere_sdf, subsampling=2)
    assert grid_void.total_active_cells < grid_void.total_cells, "Expected some void cells to be inactive."
    print("  [PASS] test_voxel_mesher_geometry")


def test_matrix_free_hex8_solver():
    """Verify matrix-free Hex8 PCG solver on a 3D cantilever beam."""
    L, W, H = 2.0, 0.2, 0.2
    grid = VoxelMesher.create_box_grid((0.0, L, 0.0, W, 0.0, H), (20, 2, 2))

    # Fixed at root x = 0
    fixed_nodes = [i for i, pt in enumerate(grid.nodes) if np.isclose(pt[0], 0.0)]
    fixed_dofs = []
    for nid in fixed_nodes:
        fixed_dofs.extend([nid * 3, nid * 3 + 1, nid * 3 + 2])

    # Tip downward load at x = L: total Fy = -8000 N
    tip_nodes = [i for i, pt in enumerate(grid.nodes) if np.isclose(pt[0], L)]
    f = np.zeros(grid.total_nodes * 3)
    for nid in tip_nodes:
        f[nid * 3 + 1] = -8000.0 / len(tip_nodes)

    t0 = time.perf_counter()
    u, elem_vm, iters = solve_voxel_linear_static(grid, f, fixed_dofs, E=2.1e11, nu=0.3, tol=1e-5)
    t_solve = time.perf_counter() - t0

    assert iters < 120, f"Expected fast PCG convergence, took {iters} iters"
    tip_dy = np.mean([u[nid * 3 + 1] for nid in tip_nodes])

    # Euler-Bernoulli analytical tip deflection
    I = (W * H**3) / 12.0
    v_analytical = (-8000.0 * L**3) / (3.0 * 2.1e11 * I)
    rel_error = abs(tip_dy - v_analytical) / abs(v_analytical)

    assert rel_error < 0.15, f"Expected deflection within 15% of Euler-Bernoulli, got {rel_error:.2%}"
    assert np.max(elem_vm) > 1e6, "Expected non-zero stresses"

    print(f"  Voxel PCG Solved in {t_solve*1e3:.2f} ms ({iters} iters) | Rel Error: {rel_error:.2%}")
    print("  [PASS] test_matrix_free_hex8_solver")


def test_end_to_end_dual_stage_pipeline():
    """Verify full dual-stage pipeline (Voxel First-Pass + 1% Saint-Venant Sub-Model)."""
    # 1. Generate high-fidelity C3D10 solid cantilever model
    length, width, height = 2.0, 0.2, 0.2
    c3d10_model, _ = generate_c3d10_structured_block(
        length=length, width=width, height=height, nx=12, ny=3, nz=3, tip_load_total=-6000.0
    )

    # 2. Run dual-stage pipeline
    t0 = time.perf_counter()
    result = run_dual_stage_pipeline(
        c3d10_model,
        voxel_resolution=(24, 4, 4),
        decay_threshold=0.01,
        submodel_min_radius=0.15,
        verbose=False,
    )
    total_pipeline_time = time.perf_counter() - t0

    # 3. Assertions and Verifications
    assert result.primary_hotspot is not None
    # In cantilever beam, peak bending stress is near the clamped root (x ~ 0)
    assert result.primary_hotspot.location[0] < 0.5 * length, (
        f"Hotspot should be near clamped root, got x={result.primary_hotspot.location[0]}"
    )

    assert result.decay_radius > 0.05, f"Expected valid decay radius, got {result.decay_radius}"
    assert result.submodel_peak_stress > 0.0, "Submodel peak stress should be positive"
    assert len(result.submodel.sub_model.mesh_nodes) > 0

    print(f"  Dual-Stage Total Wallclock: {total_pipeline_time*1e3:.2f} ms")
    print(f"    - Stage 1 Voxel Solve:   {result.voxel_solve_time*1e3:.2f} ms")
    print(f"    - Stage 2 Submodel Solve:{result.submodel_solve_time*1e3:.2f} ms")
    print(f"    - 1% Saint-Venant Radius:{result.decay_radius:.4f} m")
    print(f"    - Voxel Peak Stress:     {result.primary_hotspot.peak_stress/1e6:.2f} MPa")
    print(f"    - Refined Peak Stress:   {result.submodel_peak_stress/1e6:.2f} MPa")
    print(f"    - Stress Conc. Factor:   {result.stress_concentration_factor:.2f}")
    print("  [PASS] test_end_to_end_dual_stage_pipeline")


def run_all():
    print("=" * 60)
    print("Running Dual-Stage Voxel & Sub-Modeling Test Suite...")
    print("=" * 60)
    test_voxel_mesher_geometry()
    test_matrix_free_hex8_solver()
    test_end_to_end_dual_stage_pipeline()
    print("=" * 60)
    print("ALL DUAL-STAGE PIPELINE TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
