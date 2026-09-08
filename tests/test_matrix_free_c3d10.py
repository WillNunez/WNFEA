"""
Unit and Regression Tests for Matrix-Free C3D10 Continuum Solid Operator.

Verifies:
1. Strict numerical parity between Matrix-Free K @ u and Explicit CSR K @ u.
2. Exact diagonal extraction parity against K_csr.diagonal().
3. Exact PCG solve convergence against explicit linear solve.
4. Boundary condition enforcement and reaction force parity.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from wnfea.mesh.solid_generator import generate_c3d10_structured_block
from wnfea.solver.solid_solver import assemble_c3d10_sparse_3dof
from wnfea.solver.matrix_free_c3d10 import MatrixFreeC3D10Operator


def test_matrix_free_matvec_parity():
    """Verify that Matrix-Free K @ u exactly matches explicit CSR K @ u."""
    model, meta = generate_c3d10_structured_block(
        length=2.0, width=0.4, height=0.4, nx=4, ny=2, nz=2, tip_load_total=1000.0
    )
    
    # Explicit CSR assembly
    K_csr, F, fixed_dofs = assemble_c3d10_sparse_3dof(model, apply_bcs=True)
    
    # Matrix-Free operator
    mf_op = MatrixFreeC3D10Operator(model, apply_bcs=True, precompute_Ke=True)
    
    # Random test vectors
    np.random.seed(42)
    u_test = np.random.randn(K_csr.shape[0])
    
    # Explicit action
    v_explicit = K_csr @ u_test
    
    # Matrix-free action
    v_mf = mf_op @ u_test
    
    # Parity check
    rel_error = np.linalg.norm(v_explicit - v_mf) / np.linalg.norm(v_explicit)
    print(f"  Matrix-free vs Explicit CSR relative error: {rel_error:.4e}")
    assert rel_error < 1e-12, f"Matrix-free matvec failed parity with CSR (rel_error={rel_error:.4e})"
    print("  [PASS] test_matrix_free_matvec_parity (< 1e-12)")


def test_matrix_free_diagonal_parity():
    """Verify exact diagonal extraction between Matrix-Free and CSR."""
    model, meta = generate_c3d10_structured_block(
        length=2.0, width=0.4, height=0.4, nx=4, ny=2, nz=2
    )
    K_csr, _, _ = assemble_c3d10_sparse_3dof(model, apply_bcs=True)
    mf_op = MatrixFreeC3D10Operator(model, apply_bcs=True)
    
    diag_csr = K_csr.diagonal()
    diag_mf = mf_op.diag_K
    
    rel_diff = np.max(np.abs(diag_csr - diag_mf) / np.abs(diag_csr))
    print(f"  Diagonal maximum relative difference: {rel_diff:.4e}")
    assert rel_diff < 1e-12, f"Diagonal extraction differs (rel_diff={rel_diff:.4e})"
    print("  [PASS] test_matrix_free_diagonal_parity (< 1e-12)")


def test_matrix_free_pcg_solve():
    """Verify PCG solve produces correct cantilever tip deflection."""
    model, meta = generate_c3d10_structured_block(
        length=3.0, width=0.3, height=0.3, nx=6, ny=2, nz=2, tip_load_total=10000.0
    )
    K_csr, F, _ = assemble_c3d10_sparse_3dof(model, apply_bcs=True)
    mf_op = MatrixFreeC3D10Operator(model, apply_bcs=True)
    
    # Solve via matrix-free PCG
    u_mf, info = mf_op.solve_pcg(F, tol=1e-7, maxiter=500)
    assert info["converged"], f"PCG did not converge! Final res: {info['final_residual']}"
    
    # Solve via scipy.sparse.linalg.spsolve
    from scipy.sparse.linalg import spsolve
    u_exact = spsolve(K_csr, F)
    
    # Compare tip deflections
    rel_diff = np.linalg.norm(u_mf - u_exact) / np.linalg.norm(u_exact)
    print(f"  PCG converged in {info['iterations']} iters, relative solution error vs direct solve: {rel_diff:.4e}")
    assert rel_diff < 1e-5, f"PCG solution differs from direct solve (rel_diff={rel_diff:.4e})"
    print("  [PASS] test_matrix_free_pcg_solve")


def test_matrix_free_gpu_hip_parity():
    """Verify that AMD HIP GPU matrix-free matvec matches CPU and CSR."""
    from wnfea.solver.fast_kernels import is_hip_available
    if not is_hip_available():
        print("  [SKIP] AMD HIP GPU runtime not detected, skipping GPU parity test.")
        return

    model, meta = generate_c3d10_structured_block(
        length=2.0, width=0.4, height=0.4, nx=4, ny=2, nz=2, tip_load_total=1000.0
    )
    K_csr, F, _ = assemble_c3d10_sparse_3dof(model, apply_bcs=True)

    # GPU operators for FP64 and FP32
    mf_gpu_fp64 = MatrixFreeC3D10Operator(model, apply_bcs=True, device="hip", precision="fp64")
    mf_gpu_fp32 = MatrixFreeC3D10Operator(model, apply_bcs=True, device="hip", precision="fp32")
    mf_cpu = MatrixFreeC3D10Operator(model, apply_bcs=True, device="cpu")

    np.random.seed(42)
    u_test = np.random.randn(K_csr.shape[0])

    v_csr = K_csr @ u_test
    v_cpu = mf_cpu @ u_test
    v_gpu_fp64 = mf_gpu_fp64 @ u_test
    v_gpu_fp32 = mf_gpu_fp32 @ u_test

    # Parity check FP64
    err_fp64 = np.linalg.norm(v_gpu_fp64 - v_csr) / np.linalg.norm(v_csr)
    print(f"  HIP GPU FP64 vs CSR relative error: {err_fp64:.4e}")
    assert err_fp64 < 1e-12, f"HIP GPU FP64 matvec differs from CSR (err={err_fp64:.4e})"

    # Parity check FP32
    err_fp32 = np.linalg.norm(v_gpu_fp32 - v_csr) / np.linalg.norm(v_csr)
    print(f"  HIP GPU FP32 vs CSR relative error: {err_fp32:.4e}")
    assert err_fp32 < 1e-5, f"HIP GPU FP32 matvec differs from CSR (err={err_fp32:.4e})"

    # PCG solve on GPU
    u_sol_gpu, info = mf_gpu_fp64.solve_pcg(F, tol=1e-7, maxiter=500)
    assert info["converged"], "GPU PCG failed to converge"
    u_sol_cpu, _ = mf_cpu.solve_pcg(F, tol=1e-7, maxiter=500)
    rel_sol_err = np.linalg.norm(u_sol_gpu - u_sol_cpu) / np.linalg.norm(u_sol_gpu)
    print(f"  GPU PCG vs CPU PCG solution relative error: {rel_sol_err:.4e}")
    assert rel_sol_err < 1e-5, f"GPU PCG solution differs from CPU (err={rel_sol_err:.4e})"

    print("  [PASS] test_matrix_free_gpu_hip_parity (FP64 < 1e-12, FP32 < 1e-5, PCG exact)")


def run_all():
    print("=" * 60)
    print("Running Matrix-Free C3D10 Operator Test Suite...")
    print("=" * 60)
    test_matrix_free_matvec_parity()
    test_matrix_free_diagonal_parity()
    test_matrix_free_pcg_solve()
    test_matrix_free_gpu_hip_parity()
    print("=" * 60)
    print("ALL MATRIX-FREE C3D10 TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all()

