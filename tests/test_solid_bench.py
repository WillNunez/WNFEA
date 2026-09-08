import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
try:
    import pytest
except ImportError:
    pytest = None
from scipy.sparse.linalg import spsolve

from wnfea.mesh.solid_generator import generate_c3d10_structured_block
from wnfea.solver.solid_solver import assemble_c3d10_sparse_3dof, solve_c3d10_pcg, compute_c3d10_stress_field


def test_c3d10_generator_and_pcg_solver():
    """Verify that 3-DOF PCG matches direct sparse solve to high precision."""
    model, meta = generate_c3d10_structured_block(
        length=5.0,
        width=1.0,
        height=1.0,
        nx=10,
        ny=3,
        nz=3,
        tip_load_total=1000.0,
    )

    assert meta["n_nodes"] > 0
    assert meta["n_elements"] > 0
    assert meta["n_dofs"] == meta["n_nodes"] * 3

    K_csr, F, fixed_dofs = assemble_c3d10_sparse_3dof(model)
    assert K_csr.shape == (meta["n_dofs"], meta["n_dofs"])
    assert len(fixed_dofs) > 0

    # Direct solve
    u_direct = spsolve(K_csr, F)

    # PCG solve
    u_pcg, iters, rel_res, _ = solve_c3d10_pcg(K_csr, F, rtol=1e-7, preconditioner="jacobi")
    assert rel_res < 1e-7

    rel_diff = np.linalg.norm(u_direct - u_pcg) / np.linalg.norm(u_direct)
    assert rel_diff < 1e-6, f"PCG solution diverged from direct solve: {rel_diff}"

    # Stress recovery
    cell_sig, cell_vm, nodal_vm = compute_c3d10_stress_field(model, u_pcg)
    assert len(cell_sig) == meta["n_elements"]
    assert np.all(cell_vm >= 0.0)
    assert np.all(nodal_vm >= 0.0)
    assert np.max(nodal_vm) > 0.0
    print("  [PASS] C3D10 generator and 3-DOF PCG solver verified successfully!")


if __name__ == "__main__":
    test_c3d10_generator_and_pcg_solver()
