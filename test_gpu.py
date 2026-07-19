import sys
sys.path.insert(0, '.')
import numpy as np
from scipy.sparse import csr_matrix
from wnfea.solver.mfem_solver_wrapper import solve_with_mfem

print("Testing MFEM HIP solver backend...")

# Create a simple 1D Poisson-like stiffness matrix
n = 100
diags = [-1.0 * np.ones(n), 2.0 * np.ones(n), -1.0 * np.ones(n)]
K = np.diag(diags[1]) + np.diag(diags[0][:-1], k=-1) + np.diag(diags[2][:-1], k=1)
F = np.ones(n)

# Force boundary conditions at ends
K[0, :] = 0.0
K[0, 0] = 1.0
F[0] = 0.0

K[n-1, :] = 0.0
K[n-1, n-1] = 1.0
F[n-1] = 0.0

K_sparse = csr_matrix(K)

try:
    print("\n[1] Running on CPU...")
    U_cpu = solve_with_mfem(K_sparse, F, device="cpu")
    print("CPU solve successful.")
    print("CPU displacements (first 5):", U_cpu[:5])

    print("\n[2] Running on AMD GPU (HIP)...")
    U_gpu = solve_with_mfem(K_sparse, F, device="hip")
    print("HIP solve successful.")
    print("HIP displacements (first 5):", U_gpu[:5])

    # Check differences
    diff = np.linalg.norm(U_cpu - U_gpu)
    print(f"\nNorm of difference between CPU and HIP: {diff:.6e}")
    if diff < 1e-10:
        print("SUCCESS: GPU and CPU results match perfectly!")
    else:
        print("WARNING: GPU and CPU results differ.")

except Exception as e:
    print("Error during solver testing:")
    import traceback
    traceback.print_exc()
