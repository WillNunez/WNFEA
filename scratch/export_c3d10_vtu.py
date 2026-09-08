"""
Export Solved Large C3D10 Cantilever Solid Model to ParaView VTU.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np

from wnfea.mesh.solid_generator import generate_c3d10_structured_block
from wnfea.solver.solid_solver import assemble_c3d10_sparse_3dof, solve_c3d10_pcg
from wnfea.results import export_vtu

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
vtu_path = results_dir / "cantilever_c3d10_large_solved.vtu"

print("=" * 70)
print(" Generating & Solving Large C3D10 Continuum Solid Cantilever...")
print("=" * 70)

# 60x8x8 -> 20,449 nodes -> 61,347 DOFs, 19,200 C3D10 quadratic tets
t0 = time.perf_counter()
model, meta = generate_c3d10_structured_block(
    length=10.0,
    width=1.0,
    height=1.0,
    nx=60,
    ny=8,
    nz=8,
    E=2.1e11,
    nu=0.3,
    tip_load_total=50000.0,
)
print(f"1. Generated C3D10 Mesh: {meta['n_nodes']:,d} nodes, {meta['n_elements']:,d} elements ({meta['n_dofs']:,d} DOFs) [{time.perf_counter()-t0:.2f}s]")

t0 = time.perf_counter()
K_csr, F, fixed_dofs = assemble_c3d10_sparse_3dof(model)
print(f"2. Assembled 3-DOF Sparse K: {K_csr.shape}, {K_csr.nnz:,d} non-zeros, {len(fixed_dofs)} fixed DOFs [{time.perf_counter()-t0:.2f}s]")

t0 = time.perf_counter()
u_3dof, iters, rel_res, t_solve = solve_c3d10_pcg(K_csr, F, rtol=1e-6, max_iter=2500, preconditioner="jacobi")
print(f"3. Solved with Jacobi PCG in {iters} iterations! Rel Residual: {rel_res:.2e} [{t_solve:.2f}s]")

# Store displacements into model (expand to 6 DOFs per node for exporter compatibility)
u_6dof = np.zeros(len(model.mesh_nodes) * 6, dtype=np.float64)
u_6dof.reshape(-1, 6)[:, 0:3] = u_3dof.reshape(-1, 3)
model.displacements = u_6dof

t0 = time.perf_counter()
print(f"4. Exporting to ParaView VTU: {vtu_path}...")
export_vtu(model, vtu_path, compute_stresses=True)
vtu_size = vtu_path.stat().st_size / 1e6
print(f"5. Export completed in {time.perf_counter()-t0:.2f}s! File size: {vtu_size:.2f} MB")
print("\n[SUCCESS] Large C3D10 model ready for visualization in ParaView!")
print(f"Filepath: {vtu_path.resolve()}")
