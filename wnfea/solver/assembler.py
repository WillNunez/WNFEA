"""
Global stiffness matrix assembler for WNFEA.

Assembles the global stiffness matrix K and force vector F from the mesh
elements, applying boundary conditions (fixed, prescribed) before solving.
"""

import numpy as np
from ..model import FEAModel
from ..boundary.conditions import DOFType


def build_element_stiffness_3d_beam(
    node1: np.ndarray,
    node2: np.ndarray,
    E: float,
    G: float,
    A: float,
    Iy: float,
    Iz: float,
    J: float,
) -> np.ndarray:
    """
    Compute the 12x12 local stiffness matrix for a 3D Euler-Bernoulli beam element.

    DOF ordering per node: [ux, uy, uz, rx, ry, rz]
    Element DOFs: [node1(6), node2(6)]

    Args:
        node1, node2: 3D coordinates of the two nodes.
        E: Young's modulus.
        G: Shear modulus.
        A, Iy, Iz, J: Cross-section properties.

    Returns:
        12x12 local stiffness matrix.
    """
    L = float(np.linalg.norm(node2 - node1))
    if L < 1e-12:
        return np.zeros((12, 12))

    Ke = np.zeros((12, 12))

    # Axial stiffness
    ax = E * A / L
    Ke[0, 0] = Ke[6, 6] = ax
    Ke[0, 6] = Ke[6, 0] = -ax

    # Torsional stiffness
    tor = G * J / L
    Ke[3, 3] = Ke[9, 9] = tor
    Ke[3, 9] = Ke[9, 3] = -tor

    # Bending about Z-axis (loads in Y direction)
    bz1 = 12 * E * Iz / L**3
    bz2 = 6 * E * Iz / L**2
    bz3 = 4 * E * Iz / L
    bz4 = 2 * E * Iz / L

    Ke[1, 1] = Ke[7, 7] = bz1
    Ke[1, 7] = Ke[7, 1] = -bz1
    Ke[1, 5] = Ke[5, 1] = bz2
    Ke[1, 11] = Ke[11, 1] = bz2
    Ke[5, 5] = Ke[11, 11] = bz3
    Ke[5, 11] = Ke[11, 5] = bz4
    Ke[5, 7] = Ke[7, 5] = -bz2
    Ke[7, 11] = Ke[11, 7] = -bz2

    # Bending about Y-axis (loads in Z direction)
    by1 = 12 * E * Iy / L**3
    by2 = 6 * E * Iy / L**2
    by3 = 4 * E * Iy / L
    by4 = 2 * E * Iy / L

    Ke[2, 2] = Ke[8, 8] = by1
    Ke[2, 8] = Ke[8, 2] = -by1
    Ke[2, 4] = Ke[4, 2] = -by2
    Ke[2, 10] = Ke[10, 2] = -by2
    Ke[4, 4] = Ke[10, 10] = by3
    Ke[4, 10] = Ke[10, 4] = by4
    Ke[4, 8] = Ke[8, 4] = by2
    Ke[8, 10] = Ke[10, 8] = by2

    return Ke


def build_transformation_matrix(node1: np.ndarray, node2: np.ndarray) -> np.ndarray:
    """
    Build the 12x12 coordinate transformation matrix T for a beam element.

    Transforms from local element coordinates to global coordinates.
    K_global = T^T @ K_local @ T

    Args:
        node1, node2: 3D coordinates of the element nodes.

    Returns:
        12x12 transformation matrix.
    """
    vector = node2 - node1
    L = float(np.linalg.norm(vector))
    if L < 1e-12:
        return np.eye(12)

    # Local x-axis: along the element
    local_x = vector / L

    # Determine local y and z axes
    if np.isclose(abs(vector[0]), L):
        # Element aligned with global X
        lx = np.array([1.0 if vector[0] > 0 else -1.0, 0.0, 0.0])
        ly = np.array([0.0, 1.0, 0.0])
        lz = np.array([0.0, 0.0, 1.0])
    elif np.isclose(abs(vector[1]), L):
        # Element aligned with global Y
        lx = np.array([0.0, 1.0 if vector[1] > 0 else -1.0, 0.0])
        ly = np.array([1.0, 0.0, 0.0])
        lz = np.array([0.0, 0.0, 1.0])
    elif np.isclose(abs(vector[2]), L):
        # Element aligned with global Z
        lx = np.array([0.0, 0.0, 1.0 if vector[2] > 0 else -1.0])
        ly = np.array([1.0, 0.0, 0.0])
        lz = np.array([0.0, 1.0, 0.0])
    else:
        # General orientation
        lx = local_x
        global_z = np.array([0.0, 0.0, 1.0])
        ly = np.cross(global_z, lx)
        if np.linalg.norm(ly) < 1e-9:
            ly = np.cross(np.array([0.0, 1.0, 0.0]), lx)
        ly = ly / np.linalg.norm(ly)
        lz = np.cross(lx, ly)
        lz = lz / np.linalg.norm(lz)

    # 3x3 rotation matrix
    R = np.array([lx, ly, lz])

    # 6x6 block for one node
    T6 = np.zeros((6, 6))
    T6[:3, :3] = R
    T6[3:, 3:] = R

    # 12x12 block-diagonal
    T = np.zeros((12, 12))
    T[:6, :6] = T6
    T[6:12, 6:12] = T6

    return T


def assemble_global_system(model: FEAModel) -> tuple[np.ndarray, np.ndarray]:
    """
    Assemble the global stiffness matrix K and force vector F.

    Applies boundary conditions:
      - FIXED DOFs: row/col zeroed, diagonal set to 1, F set to 0
      - PRESCRIBED DOFs: row/col zeroed, diagonal set to 1, F set to prescribed value

    Args:
        model: FEAModel with mesh, properties, supports, and loads.

    Returns:
        Tuple of (K, F) ready for solving.

    Raises:
        ValueError: If validation fails.
    """
    errors = model.validate_for_solving()
    if errors:
        raise ValueError("Cannot assemble:\n  " + "\n  ".join(errors))

    n_nodes = len(model.mesh_nodes)
    K_full = np.zeros((n_nodes * 6, n_nodes * 6), dtype=np.float64)
    F_full = np.zeros(n_nodes * 6, dtype=np.float64)

    # 1. Assemble beam elements
    if model.mesh_elements is not None:
        for elem_id, (n1_idx, n2_idx) in enumerate(model.mesh_elements):
            node1 = model.mesh_nodes[n1_idx]
            node2 = model.mesh_nodes[n2_idx]

            # Get element properties
            assignment = model.element_properties[elem_id]
            mat = model.materials[assignment.material_name]
            sec = model.sections[assignment.section_name]

            E = mat.youngs_modulus
            G = mat.shear_modulus

            Ke_local = build_element_stiffness_3d_beam(
                node1, node2, E, G, sec.area, sec.iy, sec.iz, sec.j
            )
            T = build_transformation_matrix(node1, node2)
            Ke_global = T.T @ Ke_local @ T

            dofs = np.concatenate([
                np.arange(n1_idx * 6, n1_idx * 6 + 6),
                np.arange(n2_idx * 6, n2_idx * 6 + 6),
            ])

            for i in range(12):
                for j in range(12):
                    K_full[dofs[i], dofs[j]] += Ke_global[i, j]

    # 2. Assemble C3D10 solid elements
    if getattr(model, "solid_elements", None) is not None and len(model.solid_elements) > 0:
        from ..elements.c3d10 import element_stiffness_c3d10
        for elem_id, node_indices in enumerate(model.solid_elements):
            coords = model.mesh_nodes[node_indices]
            mat_name = model.solid_materials.get(elem_id)
            if not mat_name and model.materials:
                mat_name = next(iter(model.materials))
            mat = model.materials[mat_name]

            Ke_solid = element_stiffness_c3d10(coords, mat.youngs_modulus, mat.poissons_ratio)

            # Map 30 DOFs (3 DOFs per node) to K_full (6 DOFs per node)
            solid_dofs = []
            for n_idx in node_indices:
                solid_dofs.extend([n_idx * 6 + 0, n_idx * 6 + 1, n_idx * 6 + 2])

            for i in range(30):
                for j in range(30):
                    K_full[solid_dofs[i], solid_dofs[j]] += Ke_solid[i, j]

    # 3. Apply loads to F_full
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

    # 4. Direct Elimination via DOFManager
    from .dof_manager import DOFManager
    dof_mgr = DOFManager(model)

    if dof_mgr.total_active_dofs < n_nodes * 6:
        T_proj = dof_mgr.build_projection_matrix()
        K = T_proj.T @ K_full @ T_proj
        F = T_proj.T @ F_full
    else:
        K = K_full
        F = F_full

    # 5. Apply boundary conditions in active DOF space
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
            global_dof = dof_mgr.node_dof_to_global.get((mesh_node_id, local_dof))
            if global_dof is None:
                continue

            if constraint.dof_type == DOFType.FIXED:
                K[global_dof, :] = 0.0
                K[:, global_dof] = 0.0
                K[global_dof, global_dof] = 1.0
                F[global_dof] = 0.0
            elif constraint.dof_type == DOFType.PRESCRIBED:
                K[global_dof, :] = 0.0
                K[:, global_dof] = 0.0
                K[global_dof, global_dof] = 1.0
                F[global_dof] = constraint.value

    return K, F


def assemble_global_system_sparse(model: FEAModel):
    """
    Assemble the global stiffness matrix directly in scipy.sparse.csr_matrix format.
    Suitable for large meshes (>100k DOFs) with minimal memory footprint.

    Returns:
        Tuple of (K_csr, F_full)
    """
    from scipy.sparse import coo_matrix, csr_matrix

    n_nodes = len(model.mesh_nodes)
    n_dofs = n_nodes * 6
    F_full = np.zeros(n_dofs, dtype=np.float64)

    rows = []
    cols = []
    vals = []

    if model.mesh_elements is not None and len(model.mesh_elements) > 0:
        for elem_id, (n1_idx, n2_idx) in enumerate(model.mesh_elements):
            node1 = model.mesh_nodes[n1_idx]
            node2 = model.mesh_nodes[n2_idx]

            assignment = model.element_properties[elem_id]
            mat = model.materials[assignment.material_name]
            sec = model.sections[assignment.section_name]

            Ke_local = build_element_stiffness_3d_beam(
                node1, node2, mat.youngs_modulus, mat.shear_modulus,
                sec.area, sec.iy, sec.iz, sec.j
            )
            T = build_transformation_matrix(node1, node2)
            Ke_global = T.T @ Ke_local @ T

            dofs = np.concatenate([
                np.arange(n1_idx * 6, n1_idx * 6 + 6),
                np.arange(n2_idx * 6, n2_idx * 6 + 6),
            ])

            r_grid, c_grid = np.meshgrid(dofs, dofs, indexing='ij')
            rows.append(r_grid.ravel())
            cols.append(c_grid.ravel())
            vals.append(Ke_global.ravel())

    if getattr(model, "solid_elements", None) is not None and len(model.solid_elements) > 0:
        from ..elements.c3d10 import element_stiffness_c3d10
        for elem_id, node_indices in enumerate(model.solid_elements):
            coords = model.mesh_nodes[node_indices]
            mat_name = model.solid_materials.get(elem_id) if hasattr(model, "solid_materials") else None
            if not mat_name and model.materials:
                mat_name = next(iter(model.materials))
            mat = model.materials[mat_name]

            Ke_solid = element_stiffness_c3d10(coords, mat.youngs_modulus, mat.poissons_ratio)
            solid_dofs = []
            for n_idx in node_indices:
                solid_dofs.extend([n_idx * 6 + 0, n_idx * 6 + 1, n_idx * 6 + 2])
            solid_dofs = np.array(solid_dofs, dtype=np.int32)
            r_grid, c_grid = np.meshgrid(solid_dofs, solid_dofs, indexing='ij')
            rows.append(r_grid.ravel())
            cols.append(c_grid.ravel())
            vals.append(Ke_solid.ravel())

    if rows:
        row_arr = np.concatenate(rows)
        col_arr = np.concatenate(cols)
        val_arr = np.concatenate(vals)
        K_csr = coo_matrix((val_arr, (row_arr, col_arr)), shape=(n_dofs, n_dofs)).tocsr()
    else:
        K_csr = csr_matrix((n_dofs, n_dofs), dtype=np.float64)

    # Loads
    for load in model.loads:
        mesh_node_id = model.geometry_to_mesh_node_map.get(load.node_id) if load.is_geometry_node else load.node_id
        if mesh_node_id is not None and mesh_node_id < n_nodes:
            F_full[mesh_node_id * 6 : mesh_node_id * 6 + 6] += load.force_vector

    # Supports / BCs
    from .dof_manager import DOFManager
    dof_mgr = DOFManager(model)
    fixed_dofs = []
    for support in model.supports:
        mesh_node_id = model.geometry_to_mesh_node_map.get(support.node_id) if support.is_geometry_node else support.node_id
        if mesh_node_id is not None and mesh_node_id < n_nodes:
            for local_dof, constraint in enumerate(support.constraints):
                global_dof = dof_mgr.node_dof_to_global.get((mesh_node_id, local_dof))
                if global_dof is not None and constraint.dof_type == DOFType.FIXED:
                    fixed_dofs.append(global_dof)

    if fixed_dofs:
        fixed_set = set(fixed_dofs)
        indptr = K_csr.indptr
        indices = K_csr.indices
        data = K_csr.data

        is_fixed = np.zeros(n_dofs, dtype=bool)
        is_fixed[list(fixed_set)] = True

        row_indices = np.repeat(np.arange(n_dofs), np.diff(indptr))
        mask_row = is_fixed[row_indices]
        mask_col = is_fixed[indices]

        # Zero out off-diagonal entries connected to fixed DOFs
        mask_offdiag = (mask_row | mask_col) & (row_indices != indices)
        data[mask_offdiag] = 0.0

        # Set diagonal of fixed DOFs to 1.0
        mask_diag = mask_row & (row_indices == indices)
        data[mask_diag] = 1.0

        F_full[is_fixed] = 0.0
        K_csr.eliminate_zeros()

    return K_csr, F_full


def solve_linear_system(model: FEAModel) -> FEAModel:
    """
    Assemble and solve the global linear FEA system for beam, solid, or mixed models.
    Direct elimination ensures zero singular zero-rows for pure solid nodes and
    rigid kinematic multi-point interfaces.

    Populates:
      - model.displacements: full nodal displacements array (N*6,)
    """
    from .dof_manager import DOFManager
    dof_mgr = DOFManager(model)
    K, F = assemble_global_system(model)
    u_active = np.linalg.solve(K, F)
    model.displacements = dof_mgr.expand_displacements(u_active)
    return model

