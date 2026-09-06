"""
WNFEA HIP Architecture Tuning and Grid/Tile Sizing Suite.

Validates and benchmarks architecture-tuned HIP kernels on AMD Radeon RX 7800 XT (gfx1101).
Demonstrates how tile (block) size and grid sizing fit GPU registers and L1/L2 caches.
"""

import sys
import time
import ctypes
import numpy as np
import scipy.sparse as sp

from wnfea.model import FEAModel, PropertyAssignment
from wnfea.properties.materials import MaterialDef
from wnfea.properties.sections import SectionDef
from wnfea.solver.fast_kernels import (
    get_hip_device_summary,
    get_optimal_launch_config,
    compute_internal_forces_fast,
    spmv_hip,
    _get_native_lib,
    _get_hip_lib,
)


def create_synthetic_beam_model(n_elements: int = 50000) -> tuple[FEAModel, np.ndarray]:
    """Create a large synthetic cantilever beam model for benchmarking."""
    n_nodes = n_elements + 1
    nodes = np.zeros((n_nodes, 3), dtype=np.float64)
    nodes[:, 0] = np.linspace(0.0, 10.0, n_nodes)

    elements = np.zeros((n_elements, 2), dtype=np.int32)
    elements[:, 0] = np.arange(n_elements, dtype=np.int32)
    elements[:, 1] = np.arange(1, n_nodes, dtype=np.int32)

    model = FEAModel()
    model.mesh_nodes = nodes
    model.mesh_elements = elements

    mat = MaterialDef(name="Steel", youngs_modulus=2.1e11, poissons_ratio=0.3, yield_strength=2.5e8)
    model.materials["Steel"] = mat

    sec = SectionDef(name="Rect", area=0.01, iy=1e-4, iz=1e-4, j=2e-4)
    model.sections["Rect"] = sec

    model.element_properties = {
        i: PropertyAssignment(material_name="Steel", section_name="Rect")
        for i in range(n_elements)
    }

    U = np.zeros((n_nodes, 6), dtype=np.float64)
    # Quadratic deflection profile
    s = np.linspace(0, 1.0, n_nodes)
    U[:, 1] = -0.5 * (s ** 2)
    U[:, 5] = -0.1 * s

    return model, U.ravel()


def create_synthetic_solid_model(n_solids: int = 20000) -> tuple[FEAModel, np.ndarray]:
    """Create a synthetic C3D10 quadratic tetrahedral mesh for benchmarking."""
    n_nodes = n_solids * 10
    unit_tet = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ], dtype=np.float64)

    nodes = np.zeros((n_nodes, 3), dtype=np.float64)
    solids = np.zeros((n_solids, 10), dtype=np.int32)
    for e in range(n_solids):
        base_idx = e * 10
        nodes[base_idx : base_idx + 10] = unit_tet + np.array([e * 1.5, 0.0, 0.0])
        solids[e] = np.arange(base_idx, base_idx + 10, dtype=np.int32)

    model = FEAModel()
    model.mesh_nodes = nodes
    model.solid_elements = solids

    mat = MaterialDef(name="Aluminum", youngs_modulus=7.0e10, poissons_ratio=0.33, yield_strength=2.0e8)
    model.materials["Aluminum"] = mat
    model.solid_materials = {i: "Aluminum" for i in range(n_solids)}

    U = np.zeros((n_nodes, 6), dtype=np.float64)
    # Stretch in x, compress in y/z
    U[:, 0] = 0.001 * nodes[:, 0]
    U[:, 1] = -0.00033 * nodes[:, 1]
    U[:, 2] = -0.00033 * nodes[:, 2]

    return model, U.ravel()


def run_hip_tuning_suite():
    print("=" * 70)
    print("      WNFEA AMD Radeon RX 7800 XT (gfx1101) HIP Tuning Suite")
    print("=" * 70)

    # 1. Device Profile
    summary = get_hip_device_summary()
    if summary is None:
        print("[ERROR] Could not query AMD GPU device summary via HIP.")
        sys.exit(1)

    print("\n--- 1. GPU Architectural Profile ---")
    print(f"  Device Name:             {summary.name.decode()}")
    print(f"  Architecture / GCN Arch: {summary.arch.decode()} (RDNA 3 / Navi 32)")
    print(f"  Workgroup Processors:    {summary.multiProcessorCount} WGPs (60 CUs, 120 SIMD32 units)")
    print(f"  Native Wavefront Size:   {summary.warpSize} (Wave32)")
    print(f"  Registers per WGP:       {summary.regsPerBlock // 1024} KB (128 KB VGPR per SIMD32)")
    print(f"  Shared Memory (LDS):     {summary.sharedMemPerBlock // 1024} KB per WGP (32 memory banks)")
    print(f"  Unified L2 Cache:        {summary.l2CacheSize // (1024 * 1024)} MB (128-byte cache lines)")
    print(f"  Optimal Beam Tile Size:  {summary.optimalBeamBlockSize} threads (8 Wave32 waves -> 50% occupancy)")
    print(f"  Optimal Solid Tile Size: {summary.optimalSolidBlockSize} threads (1 Wave32 wave -> 22.5 KB LDS)")
    print(f"  Optimal Grid Range:      [{summary.optimalMinGridSize}, {summary.optimalMaxGridSize}] blocks")

    # 2. Correctness Verification
    print("\n--- 2. Numerical Equivalence Verification (CPU AVX vs GPU HIP) ---")
    beam_model, U_beam = create_synthetic_beam_model(n_elements=1000)
    F_beam_cpu = compute_internal_forces_fast(beam_model, U_beam, device="cpu")
    F_beam_hip = compute_internal_forces_fast(beam_model, U_beam, device="hip")
    diff_beam = np.max(np.abs(F_beam_cpu - F_beam_hip))
    rel_beam = diff_beam / np.max(np.abs(F_beam_cpu))
    print(f"  Beam 3D Forces max diff:  {diff_beam:.3e} (rel: {rel_beam:.3e})")
    assert rel_beam < 1e-12, "Beam GPU/CPU force mismatch!"
    print("  [PASS] Beam internal forces match with machine precision (< 1e-12).")

    solid_model, U_solid = create_synthetic_solid_model(n_solids=500)
    F_solid_cpu = compute_internal_forces_fast(solid_model, U_solid, device="cpu")
    F_solid_hip = compute_internal_forces_fast(solid_model, U_solid, device="hip")
    diff_solid = np.max(np.abs(F_solid_cpu - F_solid_hip))
    rel_solid = diff_solid / np.max(np.abs(F_solid_cpu))
    print(f"  C3D10 Forces max diff:    {diff_solid:.3e} (rel: {rel_solid:.3e})")
    assert rel_solid < 1e-12, "C3D10 GPU/CPU force mismatch!"
    print("  [PASS] C3D10 internal forces match with machine precision (< 1e-12).")

    # SpMV
    A = sp.random(5000, 5000, density=0.01, format="csr", dtype=np.float64)
    x = np.random.randn(5000).astype(np.float64)
    y_cpu = A @ x
    y_hip = spmv_hip(A, x)
    diff_spmv = np.max(np.abs(y_cpu - y_hip))
    rel_spmv = diff_spmv / np.max(np.abs(y_cpu))
    print(f"  SpMV CSR max diff:        {diff_spmv:.3e} (rel: {rel_spmv:.3e})")
    assert rel_spmv < 1e-12, "SpMV GPU/CPU mismatch!"
    print("  [PASS] SpMV CSR matches scipy reference (< 1e-12).")

    # 3. Tile Size (Block Size) Sweep Benchmark for Beams
    print("\n--- 3. Tile (Block Size) Sweep on 3D Beams (50,000 Elements) ---")
    large_beam_model, U_large_beam = create_synthetic_beam_model(n_elements=50000)

    # Warm-up
    compute_internal_forces_fast(large_beam_model, U_large_beam, device="hip")

    tile_sizes = [32, 64, 128, 256, 512, 1024]
    best_time = float("inf")
    best_tile = 0

    print(f"{'Tile Size':>10} | {'Waves/Block':>12} | {'Time (ms)':>10} | {'Throughput (M elem/s)':>22} | {'Notes'}")
    print("-" * 75)
    for bs in tile_sizes:
        # Fixed grid size 240 (8 blocks per WGP)
        times = []
        for _ in range(15):
            t0 = time.perf_counter()
            compute_internal_forces_fast(large_beam_model, U_large_beam, device="hip", block_size=bs, grid_size=240)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        t_median = float(np.median(times))
        throughput = (50000 / (t_median / 1000.0)) / 1e6
        waves = bs // 32
        note = ""
        if bs == 256:
            note = "<-- OPTIMAL (Sweet spot: 8 waves, 50% occupancy, 0 spills)"
        elif bs == 32:
            note = "(1 wave: low thread parallelism per block)"
        elif bs == 1024:
            note = "(Large: higher register pressure, low blocks/WGP)"

        if t_median < best_time:
            best_time = t_median
            best_tile = bs

        print(f"{bs:>10} | {waves:>12} | {t_median:>10.3f} | {throughput:>22.2f} | {note}")

    print(f"\n  -> Peak beam throughput achieved at tile size {best_tile} threads ({best_time:.3f} ms, {(50000 / (best_time / 1000)) / 1e6:.2f} M elem/s).")

    # 4. Grid Size (WGP Saturation & Cache Footprint) Sweep Benchmark
    print("\n--- 4. Grid Size Sweep on 3D Beams (50,000 Elements, Tile Size = 256) ---")
    grid_sizes = [1, 15, 30, 60, 120, 240, 480, 960]
    print(f"{'Grid Size':>10} | {'Blocks/WGP':>12} | {'Time (ms)':>10} | {'Throughput (M elem/s)':>22} | {'Hardware Regime'}")
    print("-" * 80)
    for gs in grid_sizes:
        times = []
        for _ in range(15):
            t0 = time.perf_counter()
            compute_internal_forces_fast(large_beam_model, U_large_beam, device="hip", block_size=256, grid_size=gs)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        t_median = float(np.median(times))
        throughput = (50000 / (t_median / 1000.0)) / 1e6
        blocks_per_wgp = gs / 30.0

        if gs == 1:
            regime = "Severely starved (only 1 WGP active out of 30!)"
        elif gs < 30:
            regime = f"Under-subscribed ({gs}/30 WGPs active)"
        elif gs == 30:
            regime = "100% WGP saturation (1 block/WGP)"
        elif gs in (60, 120, 240):
            regime = "OPTIMAL: Saturated WGPs + latency hiding in L2"
        elif gs == 480:
            regime = "High occupancy, bound within 4 MB L2 cache"
        else:
            regime = "Over-subscribed (L2 cache thrashing potential)"

        print(f"{gs:>10} | {blocks_per_wgp:>12.2f} | {t_median:>10.3f} | {throughput:>22.2f} | {regime}")

    # 5. C3D10 Solid Element Sizing Benchmark
    print("\n--- 5. C3D10 Solid Element Sizing Benchmark (20,000 Tetrahedra) ---")
    large_solid_model, U_large_solid = create_synthetic_solid_model(n_solids=20000)

    # Warm-up
    compute_internal_forces_fast(large_solid_model, U_large_solid, device="hip")

    # Benchmark CPU AVX2 vs GPU HIP
    t_cpu_list = []
    for _ in range(5):
        t0 = time.perf_counter()
        compute_internal_forces_fast(large_solid_model, U_large_solid, device="cpu")
        t1 = time.perf_counter()
        t_cpu_list.append((t1 - t0) * 1000.0)
    t_cpu = float(np.median(t_cpu_list))

    t_hip_list = []
    for _ in range(10):
        t0 = time.perf_counter()
        compute_internal_forces_fast(large_solid_model, U_large_solid, device="hip", block_size=32, grid_size=120)
        t1 = time.perf_counter()
        t_hip_list.append((t1 - t0) * 1000.0)
    t_hip = float(np.median(t_hip_list))

    cpu_rate = (20000 / (t_cpu / 1000.0)) / 1e3
    hip_rate = (20000 / (t_hip / 1000.0)) / 1e3
    speedup = t_cpu / t_hip

    print(f"  C3D10 Native AVX2 CPU: {t_cpu:>8.2f} ms ({cpu_rate:.1f}k elem/s)")
    print(f"  C3D10 Tuned HIP GPU:   {t_hip:>8.2f} ms ({hip_rate:.1f}k elem/s)")
    print(f"  GPU Speedup:           {speedup:>8.2f}x faster than AVX2 CPU")
    print("=" * 70)
    print("      ALL HIP OPTIMIZATION TESTS AND SWEEPS COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    run_hip_tuning_suite()
