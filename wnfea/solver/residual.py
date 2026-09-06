"""
Matrix-Free Equilibrium Residual Evaluation for WNFEA.

Computes the global internal force vector F_int(u) and the equilibrium residual
R(u) = F_int(u) - lambda * F_ext without allocating or assembling any global
tangent stiffness matrix.
"""

from __future__ import annotations

import numpy as np

from ..model import FEAModel
from ..boundary.conditions import DOFType
from .corotational_beam import compute_corotational_element_forces


def build_external_force_vector(model: FEAModel) -> np.ndarray:
    """
    Build the unscaled external nodal force vector F_ext of shape (N*6,).
    """
    n_nodes = len(model.mesh_nodes)
    F_ext = np.zeros(n_nodes * 6, dtype=np.float64)

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
            F_ext[mesh_node_id * 6 + i] += fv[i]

    return F_ext


def get_boundary_constraints(model: FEAModel) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract indices of constrained DOFs and their prescribed displacement values.

    Returns:
        constrained_dofs: 1D array of global DOF indices.
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
            global_dof = mesh_node_id * 6 + local_dof
            if constraint.dof_type == DOFType.FIXED:
                c_dofs.append(global_dof)
                p_vals.append(0.0)
            elif constraint.dof_type == DOFType.PRESCRIBED:
                c_dofs.append(global_dof)
                p_vals.append(float(constraint.value))

    return np.array(c_dofs, dtype=np.int64), np.array(p_vals, dtype=np.float64)


def compute_internal_forces(model: FEAModel, U: np.ndarray) -> np.ndarray:
    """
    Compute the global internal nodal force vector F_int(U) of shape (N*6,).
    Evaluated element-by-element without assembling any global matrix.

    Args:
        model: FEAModel with mesh and properties.
        U: Global displacement/rotation vector (N*6,).

    Returns:
        F_int: Global internal force vector (N*6,).
    """
    n_nodes = len(model.mesh_nodes)
    F_int = np.zeros(n_nodes * 6, dtype=np.float64)
    nodes = model.mesh_nodes

    for elem_id, (n1_idx, n2_idx) in enumerate(model.mesh_elements):
        node1 = nodes[n1_idx]
        node2 = nodes[n2_idx]

        # Gather element DOFs
        dofs1 = slice(n1_idx * 6, n1_idx * 6 + 6)
        dofs2 = slice(n2_idx * 6, n2_idx * 6 + 6)
        u_e = np.concatenate([U[dofs1], U[dofs2]])

        assignment = model.element_properties[elem_id]
        mat = model.materials[assignment.material_name]
        sec = model.sections[assignment.section_name]

        f_e = compute_corotational_element_forces(
            node1, node2, u_e,
            mat.youngs_modulus, mat.shear_modulus,
            sec.area, sec.iy, sec.iz, sec.j,
        )

        # Scatter to global F_int
        F_int[dofs1] += f_e[0:6]
        F_int[dofs2] += f_e[6:12]

    return F_int


def compute_equilibrium_residual(
    model: FEAModel,
    U: np.ndarray,
    F_ext: np.ndarray,
    load_factor: float = 1.0,
    constrained_dofs: np.ndarray | None = None,
    prescribed_vals: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute the equilibrium residual vector:
        R(U) = F_int(U) - load_factor * F_ext
    With Dirichlet boundary conditions enforced at constrained DOFs:
        R[dof] = U[dof] - load_factor * prescribed_val

    Args:
        model: FEAModel.
        U: Global displacement vector (N*6,).
        F_ext: External force vector (N*6,).
        load_factor: Current scalar load factor lambda in [0, 1].
        constrained_dofs: Cached array of constrained DOF indices.
        prescribed_vals: Cached array of prescribed values.

    Returns:
        R: Equilibrium residual vector (N*6,).
    """
    if constrained_dofs is None or prescribed_vals is None:
        constrained_dofs, prescribed_vals = get_boundary_constraints(model)

    F_int = compute_internal_forces(model, U)
    R = F_int - load_factor * F_ext

    # Enforce Dirichlet conditions in residual
    if len(constrained_dofs) > 0:
        R[constrained_dofs] = U[constrained_dofs] - load_factor * prescribed_vals

    return R
