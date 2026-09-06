"""
Matrix-Free Equilibrium Residual Evaluation for WNFEA.

Computes the global internal force vector F_int(u) and the equilibrium residual
R(u) = F_int(u) - lambda * F_ext without allocating or assembling any global
tangent stiffness matrix. Supports both 6-DOF beams and 3-DOF solids with
master-slave kinematic direct elimination.
"""

from __future__ import annotations

import numpy as np

from ..model import FEAModel
from ..boundary.conditions import DOFType
from .corotational_beam import compute_corotational_element_forces
from .dof_manager import DOFManager


def build_external_force_vector(model: FEAModel, dof_mgr: DOFManager | None = None) -> np.ndarray:
    """
    Build the unscaled external nodal force vector F_ext.
    If dof_mgr is provided, returns active DOFs (total_active_dofs,).
    Otherwise returns full nodal array (n_nodes * 6,).
    """
    n_nodes = len(model.mesh_nodes)
    F_full = np.zeros(n_nodes * 6, dtype=np.float64)

    for load in model.loads:
        if load.is_geometry_node:
            mesh_node_id = model.geometry_to_mesh_node_map.get(load.node_id)
            if mesh_node_id is None:
                continue
        else:
            mesh_node_id = load.node_id

        if mesh_node_id >= n_nodes:
            continue

        fv = load.force_vector
        for i in range(6):
            F_full[mesh_node_id * 6 + i] += fv[i]

    if dof_mgr is not None:
        return dof_mgr.condense_forces(F_full)

    return F_full


def get_boundary_constraints(
    model: FEAModel,
    dof_mgr: DOFManager | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract indices of constrained DOFs and their prescribed displacement values.
    Maps to active global DOFs if dof_mgr is provided.

    Returns:
        constrained_dofs: 1D array of active global DOF indices.
        prescribed_values: 1D array of prescribed values for each constrained DOF.
    """
    n_nodes = len(model.mesh_nodes)
    c_dofs: list[int] = []
    p_vals: list[float] = []

    for support in model.supports:
        if support.is_geometry_node:
            mesh_node_id = model.geometry_to_mesh_node_map.get(support.node_id)
            if mesh_node_id is None:
                continue
        else:
            mesh_node_id = support.node_id

        if mesh_node_id >= n_nodes:
            continue

        for local_dof, constraint in enumerate(support.constraints):
            if dof_mgr is not None:
                global_dof = dof_mgr.node_dof_to_global.get((mesh_node_id, local_dof))
                if global_dof is None:
                    continue
            else:
                global_dof = mesh_node_id * 6 + local_dof

            if constraint.dof_type == DOFType.FIXED:
                c_dofs.append(global_dof)
                p_vals.append(0.0)
            elif constraint.dof_type == DOFType.PRESCRIBED:
                c_dofs.append(global_dof)
                p_vals.append(float(constraint.value))

    return np.array(c_dofs, dtype=np.int64), np.array(p_vals, dtype=np.float64)


from .fast_kernels import compute_internal_forces_fast


def compute_internal_forces(
    model: FEAModel,
    U_full: np.ndarray,
    device: str = "cpu",
    block_size: int = 0,
    grid_size: int = 0,
) -> np.ndarray:
    """
    Compute the global internal nodal force vector F_int(U) of shape (N*6,).
    Evaluated with coalesced memory access using native HIP GPU kernels,
    native C++/AVX kernels, or batched SIMD arrays without assembling any global matrix.

    Args:
        model: FEAModel with mesh and properties.
        U_full: Full nodal displacement/rotation vector (N*6,).
        device: "cpu", "hip", or "gpu".
        block_size: Custom block size (0 = auto-tuned).
        grid_size: Custom grid size (0 = auto-tuned).

    Returns:
        F_int: Full global internal force vector (N*6,).
    """
    return compute_internal_forces_fast(
        model, U_full, device=device, block_size=block_size, grid_size=grid_size
    )


def compute_equilibrium_residual(
    model: FEAModel,
    U: np.ndarray,
    F_ext: np.ndarray,
    load_factor: float = 1.0,
    constrained_dofs: np.ndarray | None = None,
    prescribed_vals: np.ndarray | None = None,
    dof_mgr: DOFManager | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute the equilibrium residual vector:
        R(U) = F_int(U) - load_factor * F_ext
    With Dirichlet boundary conditions enforced at constrained DOFs.
    Applies direct elimination for slave nodes and inactive rotational DOFs.
    """
    if dof_mgr is None:
        dof_mgr = DOFManager(model)

    if constrained_dofs is None or prescribed_vals is None:
        constrained_dofs, prescribed_vals = get_boundary_constraints(model, dof_mgr)

    # Kinematic expansion: active DOFs -> full nodal DOFs
    if len(U) == dof_mgr.total_active_dofs:
        U_full = dof_mgr.expand_displacements(U)
    else:
        U_full = U

    # Element internal forces
    F_int_full = compute_internal_forces(model, U_full, device=device)

    # Kinematic condensation: full nodal forces -> active independent DOFs
    F_int_active = dof_mgr.condense_forces(F_int_full)

    # Active residual
    R = F_int_active - load_factor * F_ext

    # Enforce Dirichlet conditions in active residual
    if len(constrained_dofs) > 0:
        R[constrained_dofs] = U[constrained_dofs] - load_factor * prescribed_vals

    return R
