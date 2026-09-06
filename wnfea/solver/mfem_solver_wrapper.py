import os
import subprocess
import tempfile
import numpy as np
from scipy.sparse import csr_matrix

def solve_with_mfem(K: np.ndarray, F: np.ndarray, device: str = "cpu") -> np.ndarray:
    """
    Solves K * U = F using the C++ MFEM solver backend.

    Args:
        K: Global stiffness matrix (dense or sparse).
        F: Load vector.
        device: MFEM device selection (e.g. "cpu").

    Returns:
        The displacement vector U.
    """
    # 1. Convert K to scipy csr_matrix if it is not already
    if not isinstance(K, csr_matrix):
        K_sparse = csr_matrix(K)
    else:
        K_sparse = K

    nrows, ncols = K_sparse.shape
    nnz = K_sparse.nnz

    # 2. Write to a temporary file
    temp_dir = tempfile.gettempdir()
    input_file = os.path.join(temp_dir, "K_F_input.bin")
    output_file = os.path.join(temp_dir, "U_output.bin")

    # Force cleanup of old output file if it exists
    if os.path.exists(output_file):
        try:
            os.remove(output_file)
        except OSError:
            pass

    with open(input_file, 'wb') as f:
        # Header
        f.write(np.int32(nrows).tobytes())
        f.write(np.int32(ncols).tobytes())
        f.write(np.int32(nnz).tobytes())
        # CSR arrays
        f.write(K_sparse.indptr.astype(np.int32).tobytes())
        f.write(K_sparse.indices.astype(np.int32).tobytes())
        f.write(K_sparse.data.astype(np.float64).tobytes())
        # Load vector F
        f.write(F.astype(np.float64).tobytes())

    # 3. Locate the C++ solver executable
    # Find relative to this python file (located in wnfea/solver/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

    # Check common build directories
    possible_paths = [
        os.path.join(workspace_root, "build_hip", "mfem_beam_solver.exe"),
        os.path.join(workspace_root, "build", "Release", "mfem_beam_solver.exe"),
        os.path.join(workspace_root, "build", "Debug", "mfem_beam_solver.exe"),
        os.path.join(workspace_root, "build", "mfem_beam_solver.exe"),
        os.path.join(workspace_root, "mfem_beam_solver.exe"),
    ]

    exe_path = None
    for path in possible_paths:
        if os.path.exists(path):
            exe_path = path
            break

    if exe_path is None:
        raise FileNotFoundError(
            f"MFEM beam solver executable not found in any of the expected locations: {possible_paths}. "
            "Please compile the C++ project first."
        )

    # 4. Invoke the executable
    env = os.environ.copy()
    rocm_bin = r"C:\Program Files\AMD\ROCm\7.1\bin"
    if os.path.exists(rocm_bin):
        env["PATH"] = rocm_bin + os.pathsep + env.get("PATH", "")

    cmd = [exe_path, "--device", device, "--input", input_file, "--output", output_file]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"C++ solver failed with exit code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    # Print solver stdout logs for transparency
    print(result.stdout)

    # 5. Read the solved U vector
    if not os.path.exists(output_file):
        raise FileNotFoundError("Solver execution finished, but output file was not created.")

    with open(output_file, 'rb') as f:
        U = np.frombuffer(f.read(), dtype=np.float64)

    # Clean up temp files
    try:
        os.remove(input_file)
        os.remove(output_file)
    except OSError:
        pass

    return U
