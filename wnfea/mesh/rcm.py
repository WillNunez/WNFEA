"""
Reverse Cuthill-McKee (RCM) Mesh Renumbering for WNFEA.

Optimizes node ordering in structural meshes to minimize matrix bandwidth, profile,
and wavefront width. In GPU sparse solvers, narrow matrix bandwidth confines the
active slice of the displacement/search vector x to the 4 MB L2 and 64 MB AMD Infinity
Cache, maximizing effective memory throughput toward the theoretical ceiling.
"""

from __future__ import annotations

import copy
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csg

from ..model import FEAModel


def compute_node_adjacency_matrix(
    n_nodes: int,
    beam_elements: np.ndarray | None = None,
    solid_elements: np.ndarray | None = None,
) -> sp.csr_matrix:
    """
    Construct the symmetric undirected graph adjacency matrix representing nodal connectivity.

    Args:
        n_nodes: Total number of mesh nodes.
        beam_elements: [E_beam, 2] int32 element connectivity array.
        solid_elements: [E_solid, 10] int32 element connectivity array.

    Returns:
        Symmetric boolean CSR adjacency matrix of shape (n_nodes, n_nodes).
    """
    rows = []
    cols = []

    # 1. Beam elements (2 nodes per beam)
    if beam_elements is not None and len(beam_elements) > 0:
        n1 = beam_elements[:, 0]
        n2 = beam_elements[:, 1]
        rows.append(n1)
        cols.append(n2)
        rows.append(n2)
        cols.append(n1)

    # 2. Solid elements (C3D10: 10 nodes per tet)
    if solid_elements is not None and len(solid_elements) > 0:
        for i in range(10):
            for j in range(i + 1, 10):
                ni = solid_elements[:, i]
                nj = solid_elements[:, j]
                rows.append(ni)
                cols.append(nj)
                rows.append(nj)
                cols.append(ni)

    # Self-loops on the diagonal
    diag = np.arange(n_nodes, dtype=np.int32)
    rows.append(diag)
    cols.append(diag)

    all_rows = np.concatenate(rows)
    all_cols = np.concatenate(cols)
    data = np.ones(len(all_rows), dtype=np.int8)

    adj = sp.csr_matrix((data, (all_rows, all_cols)), shape=(n_nodes, n_nodes))
    adj.data[:] = 1
    return adj


def compute_rcm_order(adj: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Reverse Cuthill-McKee permutation for the given node adjacency graph.

    Args:
        adj: (N, N) symmetric sparse CSR adjacency matrix.

    Returns:
        (new_to_old, old_to_new) permutation vectors where:
        - new_to_old[new_idx] = old_idx
        - old_to_new[old_idx] = new_idx
    """
    order = csg.reverse_cuthill_mckee(adj, symmetric_mode=True)
    new_to_old = np.asarray(order, dtype=np.int32)

    n_nodes = len(new_to_old)
    old_to_new = np.empty(n_nodes, dtype=np.int32)
    old_to_new[new_to_old] = np.arange(n_nodes, dtype=np.int32)

    return new_to_old, old_to_new


def compute_matrix_bandwidth(A: sp.spmatrix) -> int:
    """
    Compute the half-bandwidth beta = max |i - j| for all non-zeros in matrix A.
    """
    r, c = A.nonzero()
    if len(r) == 0:
        return 0
    return int(np.max(np.abs(r - c)))


def apply_rcm_to_model(
    model: FEAModel,
) -> tuple[FEAModel, np.ndarray, np.ndarray]:
    """
    Apply Reverse Cuthill-McKee reordering to an FEAModel.

    Returns a reordered model with minimal matrix bandwidth along with the permutation
    vectors required to restore solution vectors to the original node ordering.

    Args:
        model: Meshed FEAModel with boundary conditions.

    Returns:
        (permuted_model, new_to_old, old_to_new)
    """
    if model.mesh_nodes is None or len(model.mesh_nodes) == 0:
        raise ValueError("Cannot apply RCM to an unmeshed model.")

    n_nodes = len(model.mesh_nodes)
    adj = compute_node_adjacency_matrix(
        n_nodes=n_nodes,
        beam_elements=model.mesh_elements,
        solid_elements=getattr(model, "solid_elements", None),
    )

    new_to_old, old_to_new = compute_rcm_order(adj)

    # Construct shallow copy of model with permuted topology
    reordered_model = copy.copy(model)

    # 1. Permute nodal coordinates
    reordered_model.mesh_nodes = model.mesh_nodes[new_to_old].copy()

    # 2. Permute beam element connectivity
    if model.mesh_elements is not None and len(model.mesh_elements) > 0:
        reordered_model.mesh_elements = old_to_new[model.mesh_elements].copy()

    # 3. Permute solid element connectivity
    if getattr(model, "solid_elements", None) is not None and len(model.solid_elements) > 0:
        reordered_model.solid_elements = old_to_new[model.solid_elements].copy()

    # 4. Remap supports and loads to new node IDs
    new_supports = []
    for supp in model.supports:
        s_copy = copy.deepcopy(supp)
        if not s_copy.is_geometry_node:
            s_copy.node_id = int(old_to_new[s_copy.node_id])
        new_supports.append(s_copy)
    reordered_model.supports = new_supports

    new_loads = []
    for load in model.loads:
        l_copy = copy.deepcopy(load)
        if not l_copy.is_geometry_node:
            l_copy.node_id = int(old_to_new[l_copy.node_id])
        new_loads.append(l_copy)
    reordered_model.loads = new_loads

    return reordered_model, new_to_old, old_to_new


def permute_solution_to_original(
    u_permuted: np.ndarray,
    new_to_old: np.ndarray,
    dofs_per_node: int = 6,
) -> np.ndarray:
    """
    Restore a solution vector computed on the RCM-permuted model back to the original
    user node numbering.

    Args:
        u_permuted: [N * dofs_per_node] solution array in RCM ordering.
        new_to_old: [N] mapping where new_to_old[new_idx] = old_idx.
        dofs_per_node: Degrees of freedom per node (6 for beams, 3 for solids).

    Returns:
        [N * dofs_per_node] solution array in original user node ordering.
    """
    u_perm = np.asarray(u_permuted).ravel()
    n_nodes = len(new_to_old)
    n_total = n_nodes * dofs_per_node

    if len(u_perm) != n_total:
        raise ValueError(f"Length of u_permuted ({len(u_perm)}) does not match {n_nodes} nodes * {dofs_per_node} DOFs.")

    # Vectorized restoration
    dof_offsets = np.arange(dofs_per_node)
    dof_new_to_old = (new_to_old[:, None] * dofs_per_node + dof_offsets).ravel()

    u_original = np.empty_like(u_perm)
    u_original[dof_new_to_old] = u_perm
    return u_original
