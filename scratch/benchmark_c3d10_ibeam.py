"""
WNFEA Structural I-Beam C3D10 Continuum Solid Cantilever Benchmark.

Benchmarks 10-node quadratic tetrahedral (C3D10) finite element performance
on an industry-standard structural I-beam cantilevered under tip transverse load:
1. Generates 3D continuum I-beam CAD solid geometry (fused flanges and web) and
   meshes with C3D10 quadratic tetrahedral elements.
2. Evaluates GPU HIP vs CPU AVX2 internal force throughput (M elem/s, GFLOP/s, FLOP/B)
   on AMD Radeon RX 7800 XT (gfx1101 / RDNA 3).
3. Solves the 3-DOF sparse elasticity system for tip deflection and compares
   against theoretical Euler-Bernoulli / Timoshenko beam mechanics.
4. Recovers Cauchy stress tensors and Von Mises equivalent stress field.
5. Exports the solved model to ParaView VTU format.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np

from wnfea.mesh.solid_generator import generate_c3d10_ibeam
from wnfea.solver.solid_solver import assemble_c3d10_sparse_3dof, solve_c3d10_system, compute_c3d10_stress_field
from wnfea.solver.fast_kernels import compute_internal_forces_fast, get_hip_device_summary
from wnfea.results import export_vtu


def run_ibeam_benchmark():
    print("=" * 85)
    print("       WNFEA Structural I-Beam C3D10 Continuum Solid Cantilever Benchmark")
    print("                AMD Radeon RX 7800 XT (gfx1101 / RDNA 3)")
    print("=" * 85)

    summary = get_hip_device_summary()
    if summary:
        print(f"  Target Hardware:  {summary.name.decode()} ({summary.arch.decode()})")
        print(f"  Compute Units:    {summary.multiProcessorCount} WGPs (60 CUs, 120 SIMD32 units)")
        print(f"  Memory Bandwidth: 624 GB/s (GDDR6) | 2.8 TB/s (64 MB Infinity Cache)")
        print(f"  Wavefront Size:   {summary.warpSize} (Wave32)")
    print("-" * 85)

    # I-Beam Cross Section Specifications (W-section / IPE profile)
    L = 4.0          # Length: 4.0 meters
    h = 0.30         # Total Section Height: 300 mm
    bf = 0.20        # Flange Width: 200 mm
    tf = 0.020       # Flange Thickness: 20 mm
    tw = 0.012       # Web Thickness: 12 mm
    E_mod = 2.1e11   # Structural Steel: 210 GPa
    nu_ratio = 0.30  # Poisson's Ratio: 0.30
    tip_load = -25000.0  # -25,000 N downward shear at tip

    # Sweep Mesh Resolutions
    resolutions = [
        ("Coarse I-Beam", {"max": 0.080, "min": 0.035}),
        ("Medium I-Beam", {"max": 0.050, "min": 0.020}),
        ("Fine I-Beam",   {"max": 0.035, "min": 0.015}),
    ]

    benchmark_rows = []
    solved_model_for_vtu = None
    vtu_output_path = Path("results/cantilever_ibeam_c3d10_solved.vtu")
    vtu_output_path.parent.mkdir(exist_ok=True)

    for name, mcfg in resolutions:
        print(f"\n>>> [Mesh Resolution: {name}]...")
        t0 = time.perf_counter()
        model, meta = generate_c3d10_ibeam(
            length=L,
            height=h,
            flange_width=bf,
            flange_thick=tf,
            web_thick=tw,
            mesh_size_max=mcfg["max"],
            mesh_size_min=mcfg["min"],
            E=E_mod,
            nu=nu_ratio,
            tip_load_total=tip_load,
            load_direction="z",
        )
        t_mesh = time.perf_counter() - t0
        n_nodes = meta["n_nodes"]
        n_elements = meta["n_elements"]
        n_dofs = meta["n_dofs"]

        print(f"  1. Meshed I-Beam Solid:        {n_nodes:,d} nodes | {n_elements:,d} C3D10 elements | {n_dofs:,d} DOFs in {t_mesh:.2f} s")

        # Theoretical Euler-Bernoulli Analytical Solution
        I_major = meta["I_major"]
        theoretical_disp_mm = meta["theoretical_tip_deflection_m"] * 1000.0
        print(f"  2. Theoretical Section Inertia: I_major = {I_major:.4e} m^4")
        print(f"     Theoretical Cantilever Deflection: {theoretical_disp_mm:.3f} mm")

        # Evaluate GPU HIP Internal Forces
        U_test = np.zeros(n_nodes * 6, dtype=np.float64)
        U_test[2::6] = -0.001 * (model.mesh_nodes[:, 0] / L) ** 2  # Parabolic test field

        # Warm-up
        compute_internal_forces_fast(model, U_test, device="hip")

        t_gpu_runs = []
        for _ in range(15):
            t0 = time.perf_counter()
            compute_internal_forces_fast(model, U_test, device="hip")
            t_gpu_runs.append(time.perf_counter() - t0)
        t_gpu = float(np.median(t_gpu_runs))

        # Evaluate CPU AVX2
        t_cpu_runs = []
        n_cpu_reps = 3 if n_elements <= 15000 else 1
        for _ in range(n_cpu_reps):
            t0 = time.perf_counter()
            compute_internal_forces_fast(model, U_test, device="cpu")
            t_cpu_runs.append(time.perf_counter() - t0)
        t_cpu = float(np.median(t_cpu_runs))

        gpu_throughput = (n_elements / t_gpu) / 1e6
        cpu_throughput = (n_elements / t_cpu) / 1e6
        speedup = t_cpu / t_gpu

        # Arithmetic Intensity: 4 Gauss points * 540 FLOPs = 2,160 FLOPs / 720 Bytes = 3.0 FLOP/Byte
        flops_per_elem = 2160
        gflops_gpu = (n_elements / t_gpu * flops_per_elem) / 1e9
        eff_bw = (n_elements / t_gpu * 720.0) / 1e9

        print(f"  3. GPU Throughput (HIP):       {t_gpu*1000:>7.2f} ms | {gpu_throughput:>6.2f} M elem/s | {gflops_gpu:>6.1f} GFLOP/s | {eff_bw:>5.1f} GB/s")
        print(f"     CPU Throughput (AVX2):      {t_cpu*1000:>7.2f} ms | {cpu_throughput:>6.2f} M elem/s | Speedup: {speedup:>5.1f}x")

        # Assemble and Solve
        t0 = time.perf_counter()
        K_csr, F, fixed_dofs = assemble_c3d10_sparse_3dof(model)
        t_assem = time.perf_counter() - t0

        t0 = time.perf_counter()
        u_sol, solve_stats = solve_c3d10_system(K_csr, F, method="direct")
        t_solve = solve_stats["solve_time_sec"]

        # Compute tip deflection
        from wnfea.mesh.gmsh_mesher import GmshMesher
        mesher = GmshMesher()
        tip_node_indices = mesher.get_nodes_on_plane(model.mesh_nodes, axis="x", coord=L, tol=1e-4)
        tip_disps = [u_sol[int(nid) * 3 + 2] for nid in tip_node_indices]
        c3d10_disp_mm = float(np.mean(tip_disps)) * 1000.0

        # Stress recovery
        cell_sig, cell_vm, nodal_vm = compute_c3d10_stress_field(model, u_sol)
        max_vm_mpa = float(np.max(nodal_vm)) / 1e6

        rel_theory_diff = abs(c3d10_disp_mm - theoretical_disp_mm) / abs(theoretical_disp_mm) * 100.0
        print(f"  4. Static Cantilever Solution:  {c3d10_disp_mm:.3f} mm (Euler-Bernoulli: {theoretical_disp_mm:.3f} mm, Diff: {rel_theory_diff:.1f}%)")
        print(f"     Max Root Von Mises Stress:  {max_vm_mpa:.2f} MPa")
        print(f"     Solve Time ({solve_stats['method']}): {t_solve:.2f} s")

        benchmark_rows.append({
            "name": name,
            "nodes": n_nodes,
            "elements": n_elements,
            "dofs": n_dofs,
            "t_gpu_ms": t_gpu * 1000.0,
            "gpu_m_s": gpu_throughput,
            "gflops": gflops_gpu,
            "speedup": speedup,
            "c3d10_disp_mm": c3d10_disp_mm,
            "theory_disp_mm": theoretical_disp_mm,
            "max_vm_mpa": max_vm_mpa,
        })

        # Save the medium resolution model for VTU export
        if name == "Medium I-Beam" or solved_model_for_vtu is None:
            u_6dof = np.zeros(n_nodes * 6, dtype=np.float64)
            u_6dof.reshape(-1, 6)[:, 0:3] = u_sol.reshape(-1, 3)
            model.displacements = u_6dof
            solved_model_for_vtu = model

    # Export VTU
    print("\n" + "=" * 85)
    print("5. Exporting Solved C3D10 I-Beam to ParaView VTU...")
    t0 = time.perf_counter()
    export_vtu(solved_model_for_vtu, vtu_output_path, compute_stresses=True)
    vtu_size_mb = vtu_output_path.stat().st_size / (1024 * 1024)
    print(f"   Exported: {vtu_output_path.resolve()} ({vtu_size_mb:.2f} MB) in {time.perf_counter()-t0:.2f} s")

    # Summary Table
    print("\n" + "=" * 105)
    print("                          STRUCTURAL I-BEAM C3D10 BENCHMARK SUMMARY")
    print("=" * 105)
    print(f"{'Mesh Resolution':<16} | {'DOFs':>9} | {'Elements':>9} | {'GPU Time':>9} | {'Throughput':>12} | {'GFLOP/s':>8} | {'Speedup':>8} | {'Tip Disp':>9} | {'Theory':>9}")
    print("-" * 105)
    for r in benchmark_rows:
        print(f"{r['name']:<16} | {r['dofs']:>9,d} | {r['elements']:>9,d} | {r['t_gpu_ms']:>7.2f} ms | {r['gpu_m_s']:>7.2f} M/s | {r['gflops']:>8.1f} | {r['speedup']:>7.1f}x | {r['c3d10_disp_mm']:>7.2f} mm | {r['theory_disp_mm']:>7.2f} mm")
    print("=" * 105)
    print(f"[SUCCESS] ParaView file ready: {vtu_output_path.resolve()}")


if __name__ == "__main__":
    run_ibeam_benchmark()
