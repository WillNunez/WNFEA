"""
Linear static solver for WNFEA.

Solves the system K * U = F using scipy's direct solver.
"""

import numpy as np
from scipy.linalg import solve

from ..model import FEAModel
from .assembler import assemble_global_system


def solve_linear_static(model: FEAModel, device: str = "cpu") -> np.ndarray:
    """
    Solve the linear static FEA problem: K * U = F.

    Assembles the global system, solves for displacements, and stores
    the result in model.displacements.

    Args:
        model: FEAModel with mesh, properties, supports, and loads defined.
        device: MFEM device selection (e.g. "cpu" or "hip").

    Returns:
        The displacement vector U (also stored in model.displacements).

    Raises:
        ValueError: If the model is not ready for solving.
        np.linalg.LinAlgError: If the system is singular.
    """
    K, F = assemble_global_system(model)

    # Check for singularity
    diag = np.diag(K)
    zero_diag = np.where(np.abs(diag) < 1e-30)[0]
    if len(zero_diag) > 0:
        raise np.linalg.LinAlgError(
            f"Singular stiffness matrix: {len(zero_diag)} zero-diagonal entries "
            f"at DOFs {zero_diag[:10].tolist()}... "
            "Check boundary conditions (model may be under-constrained)."
        )

    from .dof_manager import DOFManager
    dof_mgr = DOFManager(model)

    from .mfem_solver_wrapper import solve_with_mfem
    U_active = solve_with_mfem(K, F, device=device)
    U_full = dof_mgr.expand_displacements(U_active)
    model.displacements = U_full

    return U_full
