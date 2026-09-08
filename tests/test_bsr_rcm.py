"""
Unit and Regression Tests for BSR 6x6 Matrix Storage and Reverse Cuthill-McKee (RCM).

Verifies:
1. Graph adjacency construction and RCM bandwidth reduction.
2. Exact DOF permutation and solution vector recovery.
3. Exact parity between scalar CSR and BSR 6x6 representations.
4. Numerical parity of GPU BSR SpMV against CPU reference.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import scipy.sparse as sp
try:
    import pytest
except ImportError:
    pytest = None



from wnfea.mesh.rcm import (
    compute_node_adjacency_matrix,
    compute_rcm_order,
    compute_matrix_bandwidth,
    apply_rcm_to_model,
    permute_solution_to_original,
)
from wnfea.solver.bsr_matrix import (
    csr_to_bsr6x6,
    bsr6x6_to_csr,
    bsr6x6_spmv_cpu,
)
from wnfea.solver.fast_kernels import (
    _get_hip_lib,
    hip_spmv_bsr6x6,
)


def _build_test_frame(nx=8, ny=8, nz=8):
    """Helper to build a small 3D frame lattice model."""
    from test_gpu_amg import build_3d_frame_model
    return build_3d_frame_model(nx=nx, ny=ny, nz=nz)


def test_rcm_node_adjacency():
    """Verify adjacency matrix construction and structural graph symmetry."""
    model = _build_test_frame(nx=4, ny=4, nz=4)
    n_nodes = len(model.mesh_nodes)
    adj = compute_node_adjacency_matrix(n_nodes, model.mesh_elements)

    assert adj.shape == (n_nodes, n_nodes)
    assert np.all(adj.diagonal() == 1)
    # Symmetry check: A - A^T == 0
    diff = (adj - adj.T).nnz
    assert diff == 0, "Adjacency matrix must be strictly symmetric."


def test_rcm_bandwidth_reduction():
    """Verify that RCM ordering reduces or preserves graph bandwidth."""
    model = _build_test_frame(nx=10, ny=10, nz=10)
    from wnfea.solver.assembler import assemble_global_system_sparse

    K_orig, _ = assemble_global_system_sparse(model, device="cpu")
    bw_orig = compute_matrix_bandwidth(K_orig)

    model_rcm, new_to_old, old_to_new = apply_rcm_to_model(model)
    K_rcm, _ = assemble_global_system_sparse(model_rcm, device="cpu")
    bw_rcm = compute_matrix_bandwidth(K_rcm)

    assert bw_rcm <= bw_orig, f"RCM bandwidth ({bw_rcm}) must not exceed original ({bw_orig})"
    assert len(new_to_old) == len(model.mesh_nodes)
    assert len(old_to_new) == len(model.mesh_nodes)


def test_rcm_solution_recovery():
    """Verify that permute_solution_to_original perfectly recovers vector."""
    model = _build_test_frame(nx=5, ny=5, nz=5)
    n_nodes = len(model.mesh_nodes)
    n_dofs = n_nodes * 6

    _, new_to_old, old_to_new = apply_rcm_to_model(model)

    u_original = np.linspace(1.0, 100.0, n_dofs)
    dofs_new_to_old = (new_to_old[:, None] * 6 + np.arange(6)).ravel()
    u_permuted = u_original[dofs_new_to_old]

    u_recovered = permute_solution_to_original(u_permuted, new_to_old, dofs_per_node=6)
    np.testing.assert_allclose(u_recovered, u_original, rtol=1e-15, atol=1e-15)


def test_csr_to_bsr6x6_roundtrip():
    """Verify conversion from CSR -> BSR 6x6 -> CSR matches original matrix."""
    model = _build_test_frame(nx=6, ny=6, nz=6)
    from wnfea.solver.assembler import assemble_global_system_sparse

    K_csr, _ = assemble_global_system_sparse(model, device="cpu")
    K_bsr = csr_to_bsr6x6(K_csr, dtype=np.float64)

    K_roundtrip = bsr6x6_to_csr(K_bsr, dtype=np.float64)

    # Matrix difference norm
    diff = K_roundtrip - K_csr
    diff_norm = sp.linalg.norm(diff)
    orig_norm = sp.linalg.norm(K_csr)
    rel_diff = diff_norm / orig_norm

    assert rel_diff < 1e-15, f"BSR roundtrip error ({rel_diff}) exceeds double precision tolerance."


def test_bsr6x6_spmv_cpu_parity():
    """Verify CPU BSR SpMV produces exact same result as SciPy CSR SpMV."""
    model = _build_test_frame(nx=6, ny=6, nz=6)
    from wnfea.solver.assembler import assemble_global_system_sparse

    K_csr, _ = assemble_global_system_sparse(model, device="cpu")
    K_bsr = csr_to_bsr6x6(K_csr, dtype=np.float64)

    np.random.seed(42)
    x = np.random.randn(K_csr.shape[0])

    y_csr = K_csr.dot(x)
    y_bsr = bsr6x6_spmv_cpu(K_bsr, x)

    rel_err = np.linalg.norm(y_bsr - y_csr) / np.linalg.norm(y_csr)
    assert rel_err < 1e-14, f"CPU BSR SpMV relative error ({rel_err}) too high."


def test_bsr6x6_spmv_gpu_parity():
    """Verify GPU BSR SpMV produces exact numerical parity in FP64 and FP32."""
    model = _build_test_frame(nx=8, ny=8, nz=8)
    from wnfea.solver.assembler import assemble_global_system_sparse

    K_csr, _ = assemble_global_system_sparse(model, device="cpu")
    K_bsr_f64 = csr_to_bsr6x6(K_csr, dtype=np.float64)
    K_bsr_f32 = csr_to_bsr6x6(K_csr, dtype=np.float32)

    np.random.seed(42)
    x_f64 = np.random.randn(K_csr.shape[0]).astype(np.float64)
    x_f32 = x_f64.astype(np.float32)

    y_ref = K_csr.dot(x_f64)

    # GPU FP64
    y_gpu_f64 = hip_spmv_bsr6x6(K_bsr_f64, x_f64)
    err_f64 = np.linalg.norm(y_gpu_f64 - y_ref) / np.linalg.norm(y_ref)
    assert err_f64 < 1e-12, f"GPU FP64 BSR error ({err_f64}) too high."

    # GPU FP32
    y_gpu_f32 = hip_spmv_bsr6x6(K_bsr_f32, x_f32)
    err_f32 = np.linalg.norm(y_gpu_f32 - y_ref) / np.linalg.norm(y_ref)
    assert err_f32 < 1e-5, f"GPU FP32 BSR error ({err_f32}) too high."


if pytest is not None:
    test_bsr6x6_spmv_gpu_parity = pytest.mark.skipif(
        _get_hip_lib() is None, reason="AMD ROCm HIP library not available"
    )(test_bsr6x6_spmv_gpu_parity)



if __name__ == "__main__":
    print("Running BSR 6x6 & Reverse Cuthill-McKee (RCM) Test Suite...")
    print("-----------------------------------------------------------")
    test_rcm_node_adjacency()
    print("  [PASS] test_rcm_node_adjacency")
    test_rcm_bandwidth_reduction()
    print("  [PASS] test_rcm_bandwidth_reduction")
    test_rcm_solution_recovery()
    print("  [PASS] test_rcm_solution_recovery")
    test_csr_to_bsr6x6_roundtrip()
    print("  [PASS] test_csr_to_bsr6x6_roundtrip")
    test_bsr6x6_spmv_cpu_parity()
    print("  [PASS] test_bsr6x6_spmv_cpu_parity")
    if _get_hip_lib() is not None:
        test_bsr6x6_spmv_gpu_parity()
        print("  [PASS] test_bsr6x6_spmv_gpu_parity (Native AMD HIP)")
    print("-----------------------------------------------------------")
    print("ALL TESTS PASSED SUCCESSFULLY (100% PASS RATE)!")

