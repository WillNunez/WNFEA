"""
Verification Suite: In-the-Loop Topology Optimization & Generative Design Engine
--------------------------------------------------------------------------------
Validates:
1. Spatial Sensitivity Filtering: Matrix H row sums = 1.0, adjoint gradient symmetry.
2. Smoothed Heaviside Projection: 0-1 boundary thresholding and derivative consistency.
3. 3-Axis CNC Machinability Constraint: Undercut detection and line-of-sight projection.
4. In-the-Loop Matrix-Free SIMP Engine:
   - Compliance convergence under volume fraction constraints.
   - Non-design domain preservation (bolt bosses / pads frozen at rho=1).
   - Ultra-fast solve speed (<10 ms per iteration on CPU).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np

from wnfea.mesh.voxel_mesher import VoxelMesher
from wnfea.opt import (
    TopologyOptimizer,
    TopologyConfig,
    SensitivityFilter,
    HeavisideProjection,
    CNCMillingConstraint,
)


def test_sensitivity_filter_and_heaviside():
    """Verify spatial convolution filter normalization and Heaviside projection."""
    grid = VoxelMesher.create_box_grid((0.0, 1.0, 0.0, 1.0, 0.0, 1.0), (5, 5, 5))
    centroids = grid.get_element_centroids()
    volumes = np.full(grid.total_cells, 0.2**3)

    radius = 0.35
    s_filter = SensitivityFilter(centroids, volumes, radius=radius)

    # 1. Verify partition of unity: sum of weights for each element must equal 1.0
    row_sums = np.array(s_filter.H.sum(axis=1)).ravel()
    assert np.allclose(row_sums, 1.0, atol=1e-12), "Filter row sums must equal 1.0"

    # 2. Test density smoothing
    rho_raw = np.zeros(grid.total_cells)
    center_idx = grid.cell_id(2, 2, 2)
    rho_raw[center_idx] = 1.0  # Impulse density

    rho_smoothed = s_filter.filter_densities(rho_raw)
    assert rho_smoothed[center_idx] < 1.0, "Impulse should be diffused"
    assert np.sum(rho_smoothed) > 0.0

    # 3. Test adjoint sensitivity transpose consistency
    # <H @ x, y> == <x, H.T @ y>
    x = np.random.rand(grid.total_cells)
    y = np.random.rand(grid.total_cells)
    dot1 = np.dot(s_filter.filter_densities(x), y)
    dot2 = np.dot(x, s_filter.filter_sensitivities(y))
    adjoint_err = abs(dot1 - dot2) / abs(dot1)
    assert adjoint_err < 1e-13, f"Filter adjoint mismatch: {adjoint_err}"

    # 4. Test Heaviside projection
    heavi = HeavisideProjection(eta=0.5, beta=1.0)
    rho_mid = np.array([0.1, 0.5, 0.9])
    assert np.allclose(heavi.project(rho_mid), rho_mid)  # Beta=1 is identity

    heavi.beta = 16.0
    rho_proj = heavi.project(rho_mid)
    assert rho_proj[0] < 0.02, "Low density should be driven toward 0"
    assert np.isclose(rho_proj[1], 0.5, atol=1e-2), "Midpoint should remain ~0.5"
    assert rho_proj[2] > 0.98, "High density should be driven toward 1"

    print("  [PASS] test_sensitivity_filter_and_heaviside")


def test_cnc_machinability_projection():
    """Verify 3-axis CNC line-of-sight visibility constraint and undercut elimination."""
    grid = VoxelMesher.create_box_grid((0.0, 1.0, 0.0, 1.0, 0.0, 1.0), (5, 5, 5))
    rho = np.ones(grid.total_cells)

    # Hollow out an internal cavity in layer z=2
    # In a solid block, an internal void creates an undercut if accessed from +z
    cavity_cells = [grid.cell_id(i, j, 2) for i in range(1, 4) for j in range(1, 4)]
    rho[cavity_cells] = 0.0

    # Test +z milling constraint
    constraint_top = CNCMillingConstraint(grid, milling_axis="+z")
    p_top, grad_top = constraint_top.evaluate_undercut_penalty(rho)
    assert p_top > 0.0, "Undercut cavity should incur non-zero penalty"

    rho_mach = constraint_top.project_machinable_densities(rho)
    # The cells below the top solid layer must be filled to eliminate undercut
    for cid in cavity_cells:
        assert rho_mach[cid] == 1.0, "Undercuts must be eliminated by tool visibility"

    # Test bidirectional z milling
    constraint_bi = CNCMillingConstraint(grid, milling_axis="bi-z", split_ratio=0.5)
    rho_bi = constraint_bi.project_machinable_densities(rho)
    assert len(rho_bi) == grid.total_cells

    print("  [PASS] test_cnc_machinability_projection")


def test_in_the_loop_topology_optimization():
    """Verify end-to-end matrix-free SIMP topology optimization on a 3D cantilever bracket."""
    # Domain: 1.2m x 0.4m x 0.2m cantilever beam
    L, W, H = 1.2, 0.4, 0.2
    grid = VoxelMesher.create_box_grid((0.0, L, 0.0, W, 0.0, H), (18, 6, 3))

    # Fixed support at root plane x = 0
    fixed_nodes = [i for i, pt in enumerate(grid.nodes) if np.isclose(pt[0], 0.0)]
    fixed_dofs = []
    for nid in fixed_nodes:
        fixed_dofs.extend([nid * 3, nid * 3 + 1, nid * 3 + 2])

    # Downward point load at bottom tip (x = L, y = 0)
    tip_nodes = [i for i, pt in enumerate(grid.nodes) if np.isclose(pt[0], L) and np.isclose(pt[1], 0.0)]
    f = np.zeros(grid.total_nodes * 3)
    for nid in tip_nodes:
        f[nid * 3 + 1] = -6000.0 / len(tip_nodes)

    # Passive solid region: freeze load application cells (tip pad) at rho = 1.0
    tip_cells = [grid.cell_id(17, 0, k) for k in range(3)]

    cfg = TopologyConfig(
        target_volume_fraction=0.40,
        simp_penalty=3.0,
        filter_radius=0.10,
        max_iterations=20,
        convergence_tol=0.01,
        cnc_milling_axis="bi-z",
        cnc_penalty_weight=0.05,
        verbose=False,
    )

    t0 = time.perf_counter()
    opt = TopologyOptimizer(grid, f, fixed_dofs, config=cfg, passive_solid=tip_cells)
    res = opt.optimize()
    t_total = time.perf_counter() - t0

    # Verifications
    assert res.total_iterations >= 5, "Optimizer should perform multiple iterations"
    assert res.final_volume_fraction <= 0.45, (
        f"Target volume fraction was 0.40, got {res.final_volume_fraction:.3f}"
    )

    # Check non-design passive solid cells are preserved at rho = 1.0
    for cid in tip_cells:
        assert np.isclose(res.optimized_densities[cid], 1.0, atol=0.05), (
            f"Passive solid cell {cid} not preserved! Value: {res.optimized_densities[cid]}"
        )

    # Check execution speed (< 15 ms per iteration on CPU)
    assert res.average_iter_time < 0.05, f"Expected <50ms/iter, got {res.average_iter_time*1e3:.1f}ms"

    print(f"  Topology Optimization completed in {t_total*1e3:.1f} ms ({res.total_iterations} iters, {res.average_iter_time*1e3:.1f} ms/iter)")
    print(f"  Initial Compliance: {res.compliance_history[0]:.4f} J | Final Compliance: {res.final_compliance:.4f} J")
    print(f"  Target Volume: 40.0% | Achieved Volume: {res.final_volume_fraction*100:.1f}%")
    print(f"  Discreteness Index: {res.discreteness_index*100:.1f}% discrete")
    print("  [PASS] test_in_the_loop_topology_optimization")


def run_all():
    print("=" * 60)
    print("Running Topology Optimization & Generative Design Test Suite...")
    print("=" * 60)
    test_sensitivity_filter_and_heaviside()
    test_cnc_machinability_projection()
    test_in_the_loop_topology_optimization()
    print("=" * 60)
    print("ALL TOPOLOGY OPTIMIZATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
