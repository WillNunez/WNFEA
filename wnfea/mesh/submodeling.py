"""
Spherical Sub-Modeling and Localized Feature Re-Meshing Engine for WNFEA.
-------------------------------------------------------------------------
Implements the core tenet requested for accelerating large structural models:
1. First-pass solve (coarse/voxel/macro-mesh) produces global displacement and stress fields.
2. Automated Stress Hotspot Detection: Scans von Mises stress field to identify stress concentrations.
3. Spherical Boundary Determination: Sizes a bounding sphere around each hotspot at the
   Saint-Venant decay radius where stress/displacement changes by less than 1% per radial shell.
4. Sub-Model Domain Extraction: Isolates elements and nodes inside the sphere.
5. Prescribed Displacement Boundary Conditions: Applies interpolated global displacements
   onto cut-boundary nodes (Dirichlet interface).
6. Local Sub-Model Solve: Resolves peak stress and local gradients with refined fidelity
   in milliseconds without re-solving the global model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union, Callable
import numpy as np

from ..model import FEAModel
from ..boundary.conditions import SupportDef, DOFConstraint, DOFType, LoadDef


@dataclass
class StressHotspot:
    """
    Identified localized stress concentration point.
    """
    hotspot_id: int
    peak_stress: float          # von Mises stress in Pa
    location: np.ndarray        # (3,) coordinates in meters
    element_id: int
    decay_radius: float = 0.0   # 1% Saint-Venant boundary radius


def detect_stress_hotspots(
    nodes: np.ndarray,
    elements: np.ndarray,
    element_von_mises: np.ndarray,
    top_k: int = 3,
    min_separation: float = 0.05,
) -> list[StressHotspot]:
    """
    Identify localized stress concentration points (hotspots).

    Parameters:
        nodes: (N, 3) nodal coordinates.
        elements: (M, 10) or (M, 4) element connectivity.
        element_von_mises: (M,) scalar von Mises stress per element.
        top_k: Number of distinct hotspots to return.
        min_separation: Minimum spatial distance between distinct hotspots.

    Returns:
        hotspots: List of StressHotspot objects ordered by descending stress.
    """
    elem_stresses = np.asarray(element_von_mises, dtype=np.float64)
    if len(elem_stresses) == 0:
        return []

    # Element centroids
    elem_centroids = np.mean(nodes[elements[:, :4]], axis=1)

    # Sort elements descending by stress
    sorted_indices = np.argsort(-elem_stresses)

    hotspots: list[StressHotspot] = []
    for idx in sorted_indices:
        pt = elem_centroids[idx]
        stress_val = float(elem_stresses[idx])

        # Check separation from existing hotspots
        too_close = False
        for h in hotspots:
            if np.linalg.norm(pt - h.location) < min_separation:
                too_close = True
                break

        if not too_close:
            hotspots.append(StressHotspot(
                hotspot_id=len(hotspots),
                peak_stress=stress_val,
                location=pt,
                element_id=int(idx),
            ))

        if len(hotspots) >= top_k:
            break

    return hotspots


def compute_spherical_decay_radius(
    nodes: np.ndarray,
    elements: np.ndarray,
    element_von_mises: np.ndarray,
    center: np.ndarray,
    peak_stress: float,
    threshold: float = 0.01,
    min_radius: float = 0.01,
    max_radius: Optional[float] = None,
    num_shells: int = 20,
) -> float:
    """
    Determine the spherical sub-model radius R around a stress peak where
    stress variation between successive radial shells drops below threshold (default: 1%).
    """
    elem_centroids = np.mean(nodes[elements[:, :4]], axis=1)
    dists = np.linalg.norm(elem_centroids - center[None, :], axis=1)

    max_dist = np.max(dists) if max_radius is None else min(float(max_radius), np.max(dists))
    if max_dist <= min_radius:
        return float(min_radius)

    r_bins = np.linspace(min_radius, max_dist, num_shells + 1)
    shell_stresses = []

    for i in range(num_shells):
        r_inner = r_bins[i]
        r_outer = r_bins[i + 1]
        mask = (dists >= r_inner) & (dists < r_outer)
        if np.any(mask):
            shell_stresses.append(float(np.mean(element_von_mises[mask])))
        else:
            shell_stresses.append(shell_stresses[-1] if shell_stresses else peak_stress)

    # Find first radius where radial derivative |S_{k+1} - S_k| / peak_stress < threshold
    chosen_r = max_dist
    for i in range(1, num_shells - 1):
        diff = abs(shell_stresses[i + 1] - shell_stresses[i])
        rel_diff = diff / max(peak_stress, 1e-6)
        if rel_diff <= threshold and shell_stresses[i] < 0.5 * peak_stress:
            chosen_r = float(r_bins[i + 1])
            break

    return max(chosen_r, min_radius)


@dataclass
class SphericalSubmodel:
    """
    Represents an isolated spherical sub-model extracted from a global FEA model.
    """
    center: np.ndarray          # (3,) coordinate of sphere center
    radius: float               # Radius of sub-model boundary
    sub_model: FEAModel         # Self-contained sub-model FEAModel
    global_to_sub_nodes: dict[int, int]   # global node ID -> sub-model node ID
    sub_to_global_nodes: dict[int, int]   # sub-model node ID -> global node ID
    boundary_nodes: list[int]             # sub-model node IDs on the cut spherical boundary
    interior_nodes: list[int]             # sub-model node IDs strictly inside
    boundary_displacement_error: float = 0.0


def extract_spherical_submodel(
    global_model: FEAModel,
    global_displacements: np.ndarray,
    center: np.ndarray,
    radius: float,
    buffer_margin: float = 0.05,
) -> SphericalSubmodel:
    """
    Extract a conforming spherical sub-model domain from global FEAModel.
    
    Assigns prescribed displacement boundary conditions (from global_displacements)
    to all cut-boundary nodes lying on or near the spherical shell r = radius.
    """
    nodes = global_model.mesh_nodes
    elements = global_model.solid_elements
    if nodes is None or elements is None:
        raise ValueError("Global model must have mesh_nodes and solid_elements.")

    center = np.asarray(center, dtype=np.float64)
    elem_centroids = np.mean(nodes[elements[:, :4]], axis=1)
    elem_dists = np.linalg.norm(elem_centroids - center[None, :], axis=1)

    # Elements inside sphere of radius R (with small buffer margin)
    inside_elem_mask = elem_dists <= (radius * (1.0 + buffer_margin))
    sub_elem_indices = np.where(inside_elem_mask)[0]

    if len(sub_elem_indices) == 0:
        raise ValueError(f"No elements found within radius {radius} of {center}.")

    sub_elements_global = elements[sub_elem_indices]
    unique_global_nodes = np.unique(sub_elements_global)

    # Build node mapping
    global_to_sub = {int(gid): idx for idx, gid in enumerate(unique_global_nodes)}
    sub_to_global = {idx: int(gid) for idx, gid in enumerate(unique_global_nodes)}

    # Sub-model nodal coordinates
    sub_nodes = nodes[unique_global_nodes]
    sub_elements = np.vectorize(global_to_sub.get)(sub_elements_global)

    # Classify nodes: interior vs cut-boundary
    node_dists = np.linalg.norm(sub_nodes - center[None, :], axis=1)
    # A node is a boundary node if its distance is near or beyond radius,
    all_global_elem_nodes = set(np.unique(elements[~inside_elem_mask]))
    global_supported_nodes = {sup.node_id for sup in global_model.supports if not sup.is_geometry_node}

    boundary_nodes = []
    interior_nodes = []
    for sub_idx, gid in sub_to_global.items():
        if gid in all_global_elem_nodes or gid in global_supported_nodes or node_dists[sub_idx] >= (radius * 0.95):
            boundary_nodes.append(sub_idx)
        else:
            interior_nodes.append(sub_idx)

    # Create sub-model FEAModel
    sub_feamodel = FEAModel()
    sub_feamodel.mesh_nodes = sub_nodes
    sub_feamodel.solid_elements = sub_elements

    # Copy materials
    if hasattr(global_model, "materials"):
        sub_feamodel.materials = dict(global_model.materials)
    for new_idx, orig_elem_idx in enumerate(sub_elem_indices):
        mat_name = global_model.solid_materials.get(orig_elem_idx, "Default")
        sub_feamodel.solid_materials[new_idx] = mat_name

    # Prescribe displacements on cut boundary nodes from global solution
    # u_global shape: (N*6,) or (N*3,)
    is_6dof = (len(global_displacements) == len(nodes) * 6)
    stride = 6 if is_6dof else 3

    for sub_idx in boundary_nodes:
        gid = sub_to_global[sub_idx]
        ux = float(global_displacements[gid * stride + 0])
        uy = float(global_displacements[gid * stride + 1])
        uz = float(global_displacements[gid * stride + 2])

        sub_feamodel.supports.append(SupportDef(
            node_id=sub_idx,
            is_geometry_node=False,
            ux=DOFConstraint(DOFType.PRESCRIBED, ux),
            uy=DOFConstraint(DOFType.PRESCRIBED, uy),
            uz=DOFConstraint(DOFType.PRESCRIBED, uz),
            label=f"SubModel Spherical Cut Boundary (Global Node {gid})",
        ))

    return SphericalSubmodel(
        center=center,
        radius=radius,
        sub_model=sub_feamodel,
        global_to_sub_nodes=global_to_sub,
        sub_to_global_nodes=sub_to_global,
        boundary_nodes=boundary_nodes,
        interior_nodes=interior_nodes,
    )


def solve_spherical_submodel(
    submodel: SphericalSubmodel,
    E_default: float = 2.1e11,
    nu_default: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the spherical sub-model locally with prescribed cut-boundary displacements.

    Returns:
        u_sub: (N_sub, 3) nodal displacements.
        cell_vm: (M_sub,) element von Mises stress.
        nodal_vm: (N_sub,) smoothed nodal von Mises stress.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    from ..elements.c3d10 import element_stiffness_c3d10, element_stresses_c3d10

    model = submodel.sub_model
    nodes = model.mesh_nodes
    elements = model.solid_elements
    n_nodes = len(nodes)
    n_elems = len(elements)
    n_dofs = n_nodes * 3

    rows_list = []
    cols_list = []
    vals_list = []

    for elem_id in range(n_elems):
        conn = elements[elem_id]
        coords = nodes[conn]
        mat_name = model.solid_materials.get(elem_id)
        if hasattr(model, "materials") and mat_name in model.materials:
            m = model.materials[mat_name]
            E, nu = m.youngs_modulus, m.poissons_ratio
        else:
            E, nu = E_default, nu_default

        Ke = element_stiffness_c3d10(coords, E, nu)
        edofs = np.zeros(30, dtype=np.int64)
        for i in range(10):
            edofs[i*3 : i*3+3] = [conn[i]*3, conn[i]*3+1, conn[i]*3+2]

        r_idx = np.repeat(edofs, 30)
        c_idx = np.tile(edofs, 30)
        rows_list.append(r_idx)
        cols_list.append(c_idx)
        vals_list.append(Ke.ravel())

    K_full = coo_matrix(
        (np.concatenate(vals_list), (np.concatenate(rows_list), np.concatenate(cols_list))),
        shape=(n_dofs, n_dofs)
    ).tocsr()

    prescribed_dofs = {}
    for sup in model.supports:
        nid = sup.node_id
        for ld, constraint in enumerate([sup.ux, sup.uy, sup.uz]):
            if constraint.dof_type == DOFType.PRESCRIBED:
                prescribed_dofs[nid * 3 + ld] = constraint.value
            elif constraint.dof_type == DOFType.FIXED:
                prescribed_dofs[nid * 3 + ld] = 0.0

    p_dofs = np.array(sorted(prescribed_dofs.keys()), dtype=np.int64)
    u_p = np.array([prescribed_dofs[d] for d in p_dofs], dtype=np.float64)

    all_dofs = np.arange(n_dofs, dtype=np.int64)
    f_dofs = np.setdiff1d(all_dofs, p_dofs)

    if len(f_dofs) > 0:
        K_ff = K_full[f_dofs, :][:, f_dofs]
        K_fp = K_full[f_dofs, :][:, p_dofs]
        rhs_f = -K_fp.dot(u_p)
        u_f = spsolve(K_ff, rhs_f)
    else:
        u_f = np.zeros(0, dtype=np.float64)

    u_total = np.zeros(n_dofs, dtype=np.float64)
    u_total[p_dofs] = u_p
    if len(f_dofs) > 0:
        u_total[f_dofs] = u_f

    u_sub = u_total.reshape((n_nodes, 3))

    cell_vm = np.zeros(n_elems, dtype=np.float64)
    nodal_vm_accum = np.zeros(n_nodes, dtype=np.float64)
    nodal_vm_counts = np.zeros(n_nodes, dtype=np.int32)

    for elem_id in range(n_elems):
        conn = elements[elem_id]
        coords = nodes[conn]
        mat_name = model.solid_materials.get(elem_id)
        if hasattr(model, "materials") and mat_name in model.materials:
            m = model.materials[mat_name]
            E, nu = m.youngs_modulus, m.poissons_ratio
        else:
            E, nu = E_default, nu_default

        u_elem = u_sub[conn].ravel()
        gp_sig, gp_vm, _ = element_stresses_c3d10(coords, u_elem, E, nu)
        avg_vm = float(np.mean(gp_vm))
        cell_vm[elem_id] = avg_vm

        for nid in conn:
            nodal_vm_accum[nid] += avg_vm
            nodal_vm_counts[nid] += 1

    nodal_vm = np.zeros(n_nodes, dtype=np.float64)
    valid = nodal_vm_counts > 0
    nodal_vm[valid] = nodal_vm_accum[valid] / nodal_vm_counts[valid]

    return u_sub, cell_vm, nodal_vm
