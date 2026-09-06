"""
Roofline Model Benchmarking & Analysis for WNFEA.

Constructs an empirical Roofline Model for the current hardware environment,
evaluating the arithmetic intensity (FLOPs/Byte) and attainable throughput (GFLOP/s)
for key finite element computational kernels:
1. Classical Assembled Sparse Matrix-Vector Multiplication (SpMV) [Memory-Bound Baseline]
2. Matrix-Free Residual Evaluation (FP64)
3. Matrix-Free JFNK Operator (FP64 vs FP32)
4. AMG Preconditioner V-Cycle (FP64 vs FP32 vs FP16)
5. Inner PCG Iteration (FP64 vs Mixed FP32 vs Tri-Precision)

Outputs:
- Terminal performance table with arithmetic intensity and roofline percentage
- Formatted ASCII Roofline visualization
- High-resolution matplotlib chart saved to 'roofline_benchmark.png'
"""

from __future__ import annotations

import time
import os
import sys
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp

from ..model import FEAModel
from ..geometry.primitives import Point3D, GeometryNode, GeometryEdge
from ..properties.materials import get_preset_material
from ..properties.sections import create_hollow_tube
from ..model import PropertyAssignment
from ..boundary.conditions import create_fixed_support, LoadDef
from ..mesh.beam_mesher import BeamMesher
from ..solver.residual import compute_equilibrium_residual, build_external_force_vector
from ..solver.jfnk_operator import MatrixFreeJFNKOperator
from ..solver.amg_preconditioner import BlockBeamAMGPreconditioner
from ..solver.nonlinear_solver import pcg_solve
from ..solver.assembler import assemble_global_system


@dataclass
class KernelBenchmarkResult:
    name: str
    precision: str
    time_ms: float
    gflops: float
    traffic_mb: float
    arithmetic_intensity: float  # FLOPs / Byte
    throughput_gflops: float     # GFLOP/s
    pct_roofline: float
    bound_type: str              # "Memory-Bound" or "Compute-Bound"


class RooflineBenchmark:
    """
    Empirical Roofline Model and kernel performance analyzer for WNFEA.
    """

    def __init__(self):
        self.peak_bandwidth_gbs = 0.0
        self.peak_gflops_fp64 = 0.0
        self.peak_gflops_fp32 = 0.0
        self.ridge_point_fp64 = 0.0
        self.ridge_point_fp32 = 0.0
        self.results: list[KernelBenchmarkResult] = []

    def calibrate_hardware(self, verbose: bool = True) -> tuple[float, float, float]:
        """
        Measure empirical memory bandwidth and peak compute on this host.
        """
        if verbose:
            print("  Measuring empirical hardware boundaries...")

        # 1. STREAM Triad benchmark (Memory Bandwidth)
        # Array size = 6,000,000 doubles = ~48 MB each (exceeds L3 cache)
        N_stream = 6_000_000
        a = np.empty(N_stream, dtype=np.float64)
        b = np.ones(N_stream, dtype=np.float64)
        c = np.ones(N_stream, dtype=np.float64)
        scalar = 3.14159265

        # Warmup
        a[:] = b + scalar * c

        n_trials = 5
        t_start = time.perf_counter()
        for _ in range(n_trials):
            a[:] = b + scalar * c
        t_stream = (time.perf_counter() - t_start) / n_trials

        # STREAM Triad transfers 3 arrays (read b, read c, write a) = 3 * N * 8 bytes
        bytes_transferred = 3 * N_stream * 8
        self.peak_bandwidth_gbs = (bytes_transferred / t_stream) / 1e9

        # 2. Peak Compute (Dense matrix multiplication)
        N_gemm = 1200
        A_64 = np.random.randn(N_gemm, N_gemm).astype(np.float64)
        B_64 = np.random.randn(N_gemm, N_gemm).astype(np.float64)

        # Warmup
        _ = A_64 @ B_64

        t0 = time.perf_counter()
        _ = A_64 @ B_64
        t_gemm_64 = time.perf_counter() - t0
        flops_gemm = 2.0 * (N_gemm ** 3)
        self.peak_gflops_fp64 = (flops_gemm / t_gemm_64) / 1e9

        A_32 = A_64.astype(np.float32)
        B_32 = B_64.astype(np.float32)
        t0 = time.perf_counter()
        _ = A_32 @ B_32
        t_gemm_32 = time.perf_counter() - t0
        self.peak_gflops_fp32 = (flops_gemm / t_gemm_32) / 1e9

        # Ridge points: I_ridge = Peak_GFLOPs / Peak_GBs
        self.ridge_point_fp64 = self.peak_gflops_fp64 / max(self.peak_bandwidth_gbs, 1e-3)
        self.ridge_point_fp32 = self.peak_gflops_fp32 / max(self.peak_bandwidth_gbs, 1e-3)

        if verbose:
            print(f"    Measured Memory Bandwidth: {self.peak_bandwidth_gbs:.2f} GB/s")
            print(f"    Peak Compute Throughput:   {self.peak_gflops_fp64:.2f} GFLOP/s (FP64), {self.peak_gflops_fp32:.2f} GFLOP/s (FP32)")
            print(f"    Machine Ridge Point:       {self.ridge_point_fp64:.2f} FLOPs/Byte (FP64), {self.ridge_point_fp32:.2f} FLOPs/Byte (FP32)")

        return self.peak_bandwidth_gbs, self.peak_gflops_fp64, self.peak_gflops_fp32

    def benchmark_model_kernels(self, n_divisions: int = 24, n_runs: int = 15) -> list[KernelBenchmarkResult]:
        """
        Benchmark all WNFEA FEA kernels on a parameterized beam network.
        """
        # Build benchmark model
        model = FEAModel()
        model.geometry_nodes = {
            0: GeometryNode(id=0, label="Fixed", point=Point3D(0.0, 0.0, 0.0)),
            1: GeometryNode(id=1, label="Free", point=Point3D(3.0, 0.0, 0.0)),
        }
        model.geometry_edges = {
            0: GeometryEdge(id=0, label="Beam", start_node_id=0, end_node_id=1),
        }
        steel = get_preset_material("Structural Steel")
        tube = create_hollow_tube("Tube", 0.05, 0.04)
        model.materials[steel.name] = steel
        model.sections[tube.name] = tube
        model.edge_assignments[0] = PropertyAssignment(steel.name, tube.name)

        mesher = BeamMesher(n_divisions=n_divisions)
        mesher.mesh(model)

        model.supports.append(create_fixed_support(0, is_geometry_node=True))
        model.loads.append(LoadDef(node_id=1, is_geometry_node=True, fy=-1000.0))

        from ..solver.dof_manager import DOFManager
        dof_mgr = DOFManager(model)
        n_dofs = dof_mgr.total_active_dofs
        n_elems = len(model.mesh_elements)

        U = np.zeros(n_dofs, dtype=np.float64)
        F_ext = build_external_force_vector(model, dof_mgr)

        # -------------------------------------------------------------
        # Kernel 1: Traditional Assembled SpMV (Baseline Memory-Bound)
        # -------------------------------------------------------------
        K_dense, _ = assemble_global_system(model)
        K_csr = sp.csr_matrix(K_dense)
        nnz = K_csr.nnz
        v = np.random.randn(K_csr.shape[1])

        # Warmup
        _ = K_csr.dot(v)
        t0 = time.perf_counter()
        for _ in range(n_runs * 10):
            _ = K_csr.dot(v)
        t_spmv = (time.perf_counter() - t0) / (n_runs * 10)

        # SpMV: 2 * nnz FLOPs, traffic: data (8B) + col indices (4B) + x (8B) + y (8B)
        flops_spmv = 2.0 * nnz
        bytes_spmv = nnz * 12.0 + n_dofs * 16.0
        self._record_result(
            "Assembled SpMV (CSR)", "FP64", t_spmv, flops_spmv, bytes_spmv
        )

        # -------------------------------------------------------------
        # Kernel 2: Matrix-Free Residual Evaluation R(U) in FP64
        # -------------------------------------------------------------
        # Warmup
        _ = compute_equilibrium_residual(model, U, F_ext, load_factor=1.0, dof_mgr=dof_mgr)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = compute_equilibrium_residual(model, U, F_ext, load_factor=1.0, dof_mgr=dof_mgr)
        t_res = (time.perf_counter() - t0) / n_runs

        # Beam co-rotational evaluation: ~850 FLOPs per element
        flops_res = n_elems * 850.0 + n_dofs * 3.0
        # Memory traffic: reading nodal coords (6*8B), U (12*8B), writing F_int (12*8B)
        bytes_res = n_elems * (30 * 8.0) + n_dofs * 16.0
        self._record_result(
            "Matrix-Free Residual R(U)", "FP64", t_res, flops_res, bytes_res
        )

        # -------------------------------------------------------------
        # Kernel 3: Matrix-Free JFNK Operator (FP64 vs FP32)
        # -------------------------------------------------------------
        J_fp64 = MatrixFreeJFNKOperator(model, U, F_ext, load_factor=1.0, dof_mgr=dof_mgr, working_dtype=np.float64)
        J_fp32 = MatrixFreeJFNKOperator(model, U, F_ext, load_factor=1.0, dof_mgr=dof_mgr, working_dtype=np.float32)

        v64 = np.random.randn(n_dofs).astype(np.float64)
        v32 = v64.astype(np.float32)

        # FP64 JFNK
        _ = J_fp64.matvec(v64)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = J_fp64.matvec(v64)
        t_jfnk_64 = (time.perf_counter() - t0) / n_runs
        flops_jfnk = 2.0 * flops_res + n_dofs * 4.0
        bytes_jfnk_64 = 2.0 * bytes_res + n_dofs * (4 * 8.0)
        self._record_result(
            "Matrix-Free JFNK Matvec", "FP64", t_jfnk_64, flops_jfnk, bytes_jfnk_64
        )

        # FP32 JFNK (halves perturbation vector traffic)
        _ = J_fp32.matvec(v32)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = J_fp32.matvec(v32)
        t_jfnk_32 = (time.perf_counter() - t0) / n_runs
        bytes_jfnk_32 = bytes_res + n_dofs * (4 * 4.0)
        self._record_result(
            "Matrix-Free JFNK Matvec", "FP32", t_jfnk_32, flops_jfnk, bytes_jfnk_32
        )

        # -------------------------------------------------------------
        # Kernel 4: AMG Preconditioner V-Cycle (FP64 vs FP32 vs FP16)
        # -------------------------------------------------------------
        amg_64 = BlockBeamAMGPreconditioner(model, working_dtype=np.float64)
        amg_32 = BlockBeamAMGPreconditioner(model, working_dtype=np.float32)
        amg_16 = BlockBeamAMGPreconditioner(model, working_dtype=np.float32)
        amg_16.enable_tri_precision(switch_tol=1e-2)

        r64 = np.random.randn(len(model.mesh_nodes) * 6)

        # Count total non-zeros in AMG hierarchy for FLOPs/bytes calculation
        total_nnz = sum(lvl.A.nnz for lvl in amg_64.levels)
        if len(amg_64.levels) > 1:
            total_nnz += sum(lvl.P.nnz + lvl.R.nnz for lvl in amg_64.levels if lvl.P is not None)
        # V-cycle: 2 smoothings (pre/post) + defect + restriction + prolongation
        flops_vcycle = total_nnz * 6.0

        # FP64 AMG
        _ = amg_64.matvec(r64)
        t0 = time.perf_counter()
        for _ in range(n_runs * 5):
            _ = amg_64.matvec(r64)
        t_amg_64 = (time.perf_counter() - t0) / (n_runs * 5)
        bytes_amg_64 = total_nnz * 12.0 + n_dofs * 48.0
        self._record_result(
            "AMG V-Cycle Preconditioner", "FP64", t_amg_64, flops_vcycle, bytes_amg_64
        )

        # FP32 AMG
        _ = amg_32.matvec(r64)
        t0 = time.perf_counter()
        for _ in range(n_runs * 5):
            _ = amg_32.matvec(r64)
        t_amg_32 = (time.perf_counter() - t0) / (n_runs * 5)
        bytes_amg_32 = total_nnz * 6.0 + n_dofs * 24.0
        self._record_result(
            "AMG V-Cycle Preconditioner", "FP32", t_amg_32, flops_vcycle, bytes_amg_32
        )

        # FP16 AMG (adaptive V-cycle)
        _ = amg_16.apply_adaptive(r64, current_res_rel=0.5)
        t0 = time.perf_counter()
        for _ in range(n_runs * 5):
            _ = amg_16.apply_adaptive(r64, current_res_rel=0.5)
        t_amg_16 = (time.perf_counter() - t0) / (n_runs * 5)
        bytes_amg_16 = total_nnz * 4.0 + n_dofs * 12.0
        self._record_result(
            "AMG V-Cycle Preconditioner", "FP16", t_amg_16, flops_vcycle, bytes_amg_16
        )

        # -------------------------------------------------------------
        # Kernel 5: PCG Inner Linear Solve (FP64 vs Mixed FP32 vs Tri)
        # -------------------------------------------------------------
        rhs = -F_ext
        # FP64 PCG
        _, its_64, _ = pcg_solve(J_fp64, rhs, M_prec=amg_64, tol=1e-4, max_iter=25, working_dtype=np.float64)
        t0 = time.perf_counter()
        for _ in range(n_runs // 2):
            _, its_64, _ = pcg_solve(J_fp64, rhs, M_prec=amg_64, tol=1e-4, max_iter=25, working_dtype=np.float64)
        t_pcg_64 = (time.perf_counter() - t0) / (n_runs // 2)
        flops_pcg_64 = its_64 * (flops_jfnk + flops_vcycle + n_dofs * 10.0)
        bytes_pcg_64 = its_64 * (bytes_jfnk_64 + bytes_amg_64 + n_dofs * 40.0)
        self._record_result(
            "PCG Linear Solve", "FP64", t_pcg_64, flops_pcg_64, bytes_pcg_64
        )

        # Mixed FP32 PCG
        _, its_32, _ = pcg_solve(J_fp32, rhs, M_prec=amg_32, tol=1e-4, max_iter=25, working_dtype=np.float32)
        t0 = time.perf_counter()
        for _ in range(n_runs // 2):
            _, its_32, _ = pcg_solve(J_fp32, rhs, M_prec=amg_32, tol=1e-4, max_iter=25, working_dtype=np.float32)
        t_pcg_32 = (time.perf_counter() - t0) / (n_runs // 2)
        flops_pcg_32 = its_32 * (flops_jfnk + flops_vcycle + n_dofs * 10.0)
        bytes_pcg_32 = its_32 * (bytes_jfnk_32 + bytes_amg_32 + n_dofs * 20.0)
        self._record_result(
            "PCG Linear Solve", "Mixed FP32", t_pcg_32, flops_pcg_32, bytes_pcg_32
        )

        # Tri-Precision PCG (FP16 AMG + FP32 JFNK)
        _, its_tri, _ = pcg_solve(J_fp32, rhs, M_prec=amg_16, tol=1e-4, max_iter=25, working_dtype=np.float32)
        t0 = time.perf_counter()
        for _ in range(n_runs // 2):
            _, its_tri, _ = pcg_solve(J_fp32, rhs, M_prec=amg_16, tol=1e-4, max_iter=25, working_dtype=np.float32)
        t_pcg_tri = (time.perf_counter() - t0) / (n_runs // 2)
        flops_pcg_tri = its_tri * (flops_jfnk + flops_vcycle + n_dofs * 10.0)
        bytes_pcg_tri = its_tri * (bytes_jfnk_32 + bytes_amg_16 + n_dofs * 16.0)
        self._record_result(
            "PCG Linear Solve", "Tri-Precision", t_pcg_tri, flops_pcg_tri, bytes_pcg_tri
        )

        return self.results

    def _record_result(self, name: str, precision: str, time_s: float, flops: float, bytes_transferred: float):
        """Record and compute roofline metrics for a kernel."""
        time_ms = time_s * 1e3
        gflops = (flops / time_s) / 1e9 if time_s > 0 else 0.0
        traffic_mb = bytes_transferred / 1e6
        ai = flops / max(bytes_transferred, 1.0)

        # Attainable roofline ceiling for this arithmetic intensity
        peak_gflops = self.peak_gflops_fp32 if "32" in precision or "16" in precision or "Tri" in precision else self.peak_gflops_fp64
        attainable_gflops = min(peak_gflops, self.peak_bandwidth_gbs * ai)
        pct_roof = (gflops / max(attainable_gflops, 1e-3)) * 100.0

        bound_type = "Memory-Bound" if ai < (peak_gflops / max(self.peak_bandwidth_gbs, 1e-3)) else "Compute-Bound"

        self.results.append(
            KernelBenchmarkResult(
                name=name,
                precision=precision,
                time_ms=time_ms,
                gflops=flops / 1e9,
                traffic_mb=traffic_mb,
                arithmetic_intensity=ai,
                throughput_gflops=gflops,
                pct_roofline=min(pct_roof, 100.0),
                bound_type=bound_type,
            )
        )

    def print_summary_table(self):
        """Print formatted benchmark results table to console."""
        print("\n" + "=" * 95)
        print("                   WNFEA SOLVER ROOFLINE BENCHMARK SUMMARY")
        print("=" * 95)
        print(f"  Hardware: Peak BW = {self.peak_bandwidth_gbs:.1f} GB/s | Peak FP64 = {self.peak_gflops_fp64:.1f} GFLOP/s | Peak FP32 = {self.peak_gflops_fp32:.1f} GFLOP/s")
        print(f"  Machine Ridge: FP64 = {self.ridge_point_fp64:.2f} FLOPs/B | FP32 = {self.ridge_point_fp32:.2f} FLOPs/B")
        print("-" * 95)
        hdr = f"{'Kernel':<28} | {'Prec':<13} | {'Time (ms)':<9} | {'Traffic(MB)':<11} | {'AI (FL/B)':<9} | {'GFLOP/s':<8} | {'% Roof':<6} | {'Bound'}"
        print(hdr)
        print("-" * 95)
        for r in self.results:
            print(
                f"{r.name:<28} | {r.precision:<13} | {r.time_ms:9.3f} | {r.traffic_mb:11.3f} | "
                f"{r.arithmetic_intensity:9.2f} | {r.throughput_gflops:8.2f} | {r.pct_roofline:5.1f}% | {r.bound_type}"
            )
        print("=" * 95)

    def print_ascii_roofline(self):
        """Print an ASCII visualization of the roofline chart."""
        print("\n  ASCII Roofline Chart (Throughput vs Operational Intensity):")
        print("  GFLOP/s")
        print("    ^")
        print(f"    |  Peak FP32: {self.peak_gflops_fp32:.1f} GFLOP/s  ------------------------+ (Compute Ceiling)")
        print(f"    |  Peak FP64: {self.peak_gflops_fp64:.1f} GFLOP/s  --------------+          |")
        print("    |                              /             |          |")
        print(f"    |  Bandwidth: {self.peak_bandwidth_gbs:.1f} GB/s         /              |          |")
        print("    |                       /                    |          |")
        print("    |   SpMV (FP64)        /   JFNK (FP32/64)    |          |")
        print("    |     [*]             /        [*]           |          |")
        print("    |                    /   AMG (FP16/32)       |          |")
        print("    |                   /        [*]             |          |")
        print("    |                  /  Residual (FP64)        |          |")
        print("    |                 /        [*]               |          |")
        print("    +----------------+---------------------------+----------+---> Arithmetic Intensity (FLOPs/Byte)")
        print("                   10^-1                        10^0       10^1\n")

    def save_plot(self, output_filepath: str = "roofline_benchmark.png"):
        """
        Generate and save a publication-quality roofline chart using matplotlib.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

        # Operational Intensity axis range (log scale)
        ai_range = np.logspace(-2, 2.5, 500)

        # Theoretical rooflines
        roof_fp64 = np.minimum(self.peak_gflops_fp64, self.peak_bandwidth_gbs * ai_range)
        roof_fp32 = np.minimum(self.peak_gflops_fp32, self.peak_bandwidth_gbs * ai_range)

        # Plot rooflines
        ax.loglog(ai_range, roof_fp32, "r--", linewidth=2.0, label=f"Peak FP32 Roofline ({self.peak_gflops_fp32:.1f} GFLOP/s)")
        ax.loglog(ai_range, roof_fp64, "b-", linewidth=2.0, label=f"Peak FP64 Roofline ({self.peak_gflops_fp64:.1f} GFLOP/s)")

        # Plot memory bandwidth ceiling
        bw_line = self.peak_bandwidth_gbs * ai_range
        ax.loglog(ai_range, bw_line, color="gray", linestyle=":", alpha=0.7,
                  label=f"Memory Bandwidth ({self.peak_bandwidth_gbs:.1f} GB/s)")

        # Ridge points
        ax.axvline(self.ridge_point_fp64, color="blue", linestyle=":", alpha=0.4)
        ax.axvline(self.ridge_point_fp32, color="red", linestyle=":", alpha=0.4)

        # Color and marker mapping for kernels
        color_map = {
            "FP64": "#1f77b4",
            "FP32": "#2ca02c",
            "FP16": "#d62728",
            "Mixed FP32": "#9467bd",
            "Tri-Precision": "#ff7f0e",
        }
        marker_map = {
            "Assembled SpMV (CSR)": "s",
            "Matrix-Free Residual R(U)": "o",
            "Matrix-Free JFNK Matvec": "^",
            "AMG V-Cycle Preconditioner": "D",
            "PCG Linear Solve": "P",
        }

        # Scatter plot benchmark points
        plotted_kernel_names = set()
        for r in self.results:
            c = color_map.get(r.precision, "#333333")
            m = marker_map.get(r.name, "o")
            lbl = r.name if r.name not in plotted_kernel_names else None
            if lbl:
                plotted_kernel_names.add(r.name)
            ax.scatter(r.arithmetic_intensity, r.throughput_gflops, color=c, marker=m, s=95, zorder=5, edgecolors="black", linewidths=1.0, label=lbl)

            # Annotate point with precision
            ax.annotate(
                f" {r.precision}",
                (r.arithmetic_intensity, r.throughput_gflops),
                fontsize=7.5,
                fontweight="bold",
                color=c,
                xytext=(3, 3),
                textcoords="offset points",
            )

        ax.set_title("WNFEA Solver Roofline Model Benchmark\nMatrix-Free JFNK & Tri-Precision (FP16/FP32/FP64) AMG", fontsize=13, fontweight="bold")
        ax.set_xlabel("Arithmetic Intensity [FLOPs / Byte]", fontsize=11, fontweight="semibold")
        ax.set_ylabel("Attainable Performance [GFLOP/s]", fontsize=11, fontweight="semibold")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.set_ylim(bottom=5e-3, top=max(self.peak_gflops_fp32 * 1.8, 100))
        ax.set_xlim(left=5e-2, right=200)

        # Legend
        ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
        plt.tight_layout()

        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        plt.savefig(output_filepath)
        plt.close()
        print(f"\n  Roofline plot saved successfully to: {output_filepath}")


def run_roofline_benchmark(output_chart_path: str = "roofline_benchmark.png") -> RooflineBenchmark:
    """Execute full roofline calibration, kernel benchmark, and chart generation."""
    print("=" * 65)
    print("        WNFEA ROOFLINE BENCHMARK & KERNEL PROFILER")
    print("=" * 65)

    bench = RooflineBenchmark()
    bench.calibrate_hardware(verbose=True)
    bench.benchmark_model_kernels(n_divisions=20, n_runs=10)
    bench.print_summary_table()
    bench.print_ascii_roofline()
    bench.save_plot(output_chart_path)

    return bench


if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "roofline_benchmark.png")
    run_roofline_benchmark(out_path)
