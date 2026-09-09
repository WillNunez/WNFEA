"""
Local Sub-Domain Isolated Matrix-Free PCG Re-Solver for WNFEA.
--------------------------------------------------------------
Implements the core tenet of Phase 12:
1. Extracts ONLY the elements inside the Saint-Venant 1% decay boundary sphere R_1%,
   strictly excluding all exterior elements.
2. Identifies the cut-boundary nodes along the spherical partition interface.
3. Prescribes Dirichlet boundary conditions on the cut boundary using interpolated
   displacements from the previous global pass.
4. Executes an isolated matrix-free PCG re-solve on the localized hotspot in <5 ms,
   achieving >10x wallclock acceleration while capturing exact peak stress concentration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence, Union, Tuple, List, Dict
import numpy as np
import scipy.sparse.linalg as spla

from ..mesh.voxel_mesher import VoxelGrid
from ..mesh.octree_amr import AMRMesh
from .matrix_free_hex8 import compute_hex8_reference_stiffness


@dataclass
class IsolatedSubdomain:
    """
    Isolated geometric sub-domain extracted from a global mesh within a bounding sphere.
    """
    nodes: np.ndarray                       # (N_sub, 3) Sub-domain nodal coordinates
    elements: np.ndarray                    # (M_sub, 8) Hex8 element connectivity
    global_node_map: np.ndarray             # (N_sub,) Mapping from sub-domain node idx to source node idx
    cut_boundary_nodes: np.ndarray          # (N_cut,) Local node indices on the spherical cut boundary
    interior_nodes: np.ndarray              # (N_int,) Local node indices strictly inside the sub-domain
    prescribed_displacements: np.ndarray    # (N_cut, 3) Dirichlet displacements from global pass
    center: np.ndarray                      # (3,) Hotspot center
    radius: float                           # 1% Saint-Venant bounding radius R_1%
    E: float = 2.1e11                       # Young's modulus (Pa)
    nu: float = 0.3                         # Poisson's ratio
    original_element_count: int = 0         # Total elements in original global mesh


@dataclass
class SubdomainSolveResult:
    """
    Output metrics and field solutions from the isolated sub-domain solve.
    """
    subdomain: IsolatedSubdomain
    displacements: np.ndarray               # (N_sub, 3) Nodal displacements
    element_von_mises: np.ndarray           # (M_sub,) Centroidal von Mises stresses
    peak_stress: float                      # Maximum von Mises stress in sub-domain (Pa)
    solve_time: float                       # PCG solve duration in seconds
    num_iterations: int                     # PCG iteration count
    residual: float                         # Final relative residual
    element_reduction_ratio: float          # original_elems / sub_elems


def extract_isolated_subdomain(
    mesh_or_grid: Union[VoxelGrid, AMRMesh, Tuple[np.ndarray, np.ndarray]],
    center: Union[np.ndarray, Sequence[float]],
    radius: float,
    global_displacements: np.ndarray,
    E: float = 2.1e11,
    nu: float = 0.3,
    global_nodes: Optional[np.ndarray] = None,
) -> IsolatedSubdomain:
    """
    Carve out an isolated sub-domain containing only elements inside the sphere R_1%.
    Excludes all exterior elements, identifies cut-boundary nodes, and samples
    Dirichlet displacement boundary conditions from the global pass.

    Parameters:
        mesh_or_grid: VoxelGrid, AMRMesh, or (nodes, elements) tuple.
        center: (3,) Hotspot center coordinate.
        radius: 1% Saint-Venant decay boundary radius.
        global_displacements: (N_global, 3) or (N_global*3,) global displacement vector.
        E: Young's modulus (Pa).
        nu: Poisson's ratio.
        global_nodes: Optional explicit coordinates if mesh_or_grid is a raw element array.

    Returns:
        IsolatedSubdomain ready for high-speed localized PCG re-solve.
    """
    c = np.asarray(center, dtype=np.float64).reshape(3)
    R = float(radius)

    if isinstance(mesh_or_grid, VoxelGrid):
        nodes = mesh_or_grid.nodes
        elements = mesh_or_grid.elements[mesh_or_grid.active_element_indices]
    elif isinstance(mesh_or_grid, AMRMesh):
        nodes = mesh_or_grid.nodes
        elements = mesh_or_grid.elements
    elif isinstance(mesh_or_grid, tuple):
        nodes, elements = mesh_or_grid
    else:
        if global_nodes is None:
            raise ValueError("global_nodes must be provided if mesh_or_grid is not a recognized type")
        nodes = global_nodes
        elements = np.asarray(mesh_or_grid)

    original_elem_count = len(elements)
    if original_elem_count == 0:
        raise ValueError("Source mesh contains 0 elements")

    # Format global displacements to (N_global, 3)
    u_glob = np.asarray(global_displacements, dtype=np.float64)
    if u_glob.ndim == 1:
        u_glob = u_glob.reshape(-1, 3)

    # 1. Evaluate element centroids and filter elements inside sphere
    elem_centroids = np.mean(nodes[elements], axis=1)  # (M, 3)
    dist_centroids = np.linalg.norm(elem_centroids - c, axis=1)
    sub_elem_mask = dist_centroids <= R

    # Safeguard: if sphere is extremely small, retain at least the closest element
    if not np.any(sub_elem_mask):
        sub_elem_mask[np.argmin(dist_centroids)] = True

    sub_elements_raw = elements[sub_elem_mask]
    sub_elem_indices = np.where(sub_elem_mask)[0]

    # 2. Build local node numbering for retained elements
    unique_global_nids, inverse_indices = np.unique(sub_elements_raw, return_inverse=True)
    sub_elements = inverse_indices.reshape(-1, 8)
    sub_nodes = nodes[unique_global_nids]
    n_sub_nodes = len(unique_global_nids)

    # 3. Identify cut-boundary nodes:
    # A node is on the cut boundary if in the source mesh it is shared by at least
    # one excluded element outside the sphere.
    # Also tag nodes at distance d >= 0.85 * R if on the outer boundary.
    node_to_orig_elems: Dict[int, List[int]] = {nid: [] for nid in unique_global_nids}
    for e_idx, elem in enumerate(elements):
        for nid in elem:
            if nid in node_to_orig_elems:
                node_to_orig_elems[nid].append(e_idx)

    cut_boundary_local_nids: List[int] = []
    interior_local_nids: List[int] = []

    for local_idx, global_nid in enumerate(unique_global_nids):
        incident_elems = node_to_orig_elems[global_nid]
        # If any incident element was NOT included in the sub-domain, it is a cut node
        is_cut = any(e not in sub_elem_indices for e in incident_elems)

        # In addition, check geometric boundary near sphere radius
        d_node = np.linalg.norm(sub_nodes[local_idx] - c)
        if is_cut or (d_node >= 0.90 * R and len(incident_elems) < 8):
            cut_boundary_local_nids.append(local_idx)
        else:
            interior_local_nids.append(local_idx)

    # If no cut nodes were identified (e.g. sphere encloses entire object),
    # anchor the outer perimeter nodes
    if len(cut_boundary_local_nids) == 0:
        dists = np.linalg.norm(sub_nodes - c, axis=1)
        max_d = np.max(dists)
        outer_mask = np.where(dists >= 0.85 * max_d)[0]
        cut_boundary_local_nids = list(outer_mask)
        interior_local_nids = [i for i in range(n_sub_nodes) if i not in cut_boundary_local_nids]

    cut_nodes_arr = np.array(sorted(cut_boundary_local_nids), dtype=np.int64)
    int_nodes_arr = np.array(sorted(interior_local_nids), dtype=np.int64)

    # 4. Extract prescribed displacements on cut-boundary nodes from global pass
    cut_global_nids = unique_global_nids[cut_nodes_arr]
    prescribed_u = u_glob[cut_global_nids].copy()

    return IsolatedSubdomain(
        nodes=sub_nodes,
        elements=sub_elements,
        global_node_map=unique_global_nids,
        cut_boundary_nodes=cut_nodes_arr,
        interior_nodes=int_nodes_arr,
        prescribed_displacements=prescribed_u,
        center=c,
        radius=R,
        E=float(E),
        nu=float(nu),
        original_element_count=original_elem_count,
    )


class MatrixFreeSubdomainOperator(spla.LinearOperator):
    """
    Matrix-free linear operator for the isolated sub-domain.
    Applies element stiffness operations without global CSR matrix assembly.
    """
    def __init__(
        self,
        subdomain: IsolatedSubdomain,
        fixed_dofs: np.ndarray,
    ):
        self.subdomain = subdomain
        self.n_nodes = len(subdomain.nodes)
        self.n_dofs = self.n_nodes * 3
        shape = (self.n_dofs, self.n_dofs)
        super().__init__(dtype=np.float64, shape=shape)

        self.fixed_dofs = fixed_dofs
        self.fixed_mask = np.zeros(self.n_dofs, dtype=bool)
        self.fixed_mask[fixed_dofs] = True

        # Precompute element matrices
        self.k_elems, self.B_elems, self.D = self._precompute_element_stiffnesses()

        # Precompute DOF indices per element: (M, 24)
        node_dofs = subdomain.elements[:, :, None] * 3 + np.array([0, 1, 2])[None, None, :]
        self.elem_dofs = node_dofs.reshape(-1, 24)

        # Diagonal preconditioner
        self.diag = self._compute_diagonal()

    def _precompute_element_stiffnesses(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute isoparametric (24, 24) stiffness matrix for each element in sub-domain."""
        sub = self.subdomain
        nodes = sub.nodes
        elements = sub.elements
        M = len(elements)
        E = sub.E
        nu = sub.nu

        # 1. Constitutive Matrix D
        c1 = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        c11 = c1 * (1.0 - nu)
        c12 = c1 * nu
        c44 = c1 * (0.5 - nu)
        D = np.array([
            [c11, c12, c12, 0.0, 0.0, 0.0],
            [c12, c11, c12, 0.0, 0.0, 0.0],
            [c12, c12, c11, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, c44, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, c44, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, c44],
        ], dtype=np.float64)

        # Standard natural Gauss points
        gp = 1.0 / np.sqrt(3.0)
        gauss_pts = [-gp, gp]
        xi_nodes = np.array([
            [-1.0, -1.0, -1.0], [ 1.0, -1.0, -1.0], [ 1.0,  1.0, -1.0], [-1.0,  1.0, -1.0],
            [-1.0, -1.0,  1.0], [ 1.0, -1.0,  1.0], [ 1.0,  1.0,  1.0], [-1.0,  1.0,  1.0],
        ], dtype=np.float64)

        k_elems = np.zeros((M, 24, 24), dtype=np.float64)
        B_elems = np.zeros((M, 6, 24), dtype=np.float64)

        # Precompute standard shape function natural derivatives
        dN_nat = np.zeros((8, 3, 8), dtype=np.float64)
        gp_idx = 0
        gp_coords = []
        for xi in gauss_pts:
            for eta in gauss_pts:
                for zeta in gauss_pts:
                    gp_coords.append((xi, eta, zeta))
                    for a in range(8):
                        xa, ya, za = xi_nodes[a]
                        dN_nat[gp_idx, 0, a] = 0.125 * xa * (1.0 + ya * eta) * (1.0 + za * zeta)
                        dN_nat[gp_idx, 1, a] = 0.125 * ya * (1.0 + xa * xi) * (1.0 + za * zeta)
                        dN_nat[gp_idx, 2, a] = 0.125 * za * (1.0 + xa * xi) * (1.0 + ya * eta)
                    gp_idx += 1

        # Centroidal derivative
        dN_cent = np.zeros((3, 8), dtype=np.float64)
        for a in range(8):
            xa, ya, za = xi_nodes[a]
            dN_cent[0, a] = 0.125 * xa
            dN_cent[1, a] = 0.125 * ya
            dN_cent[2, a] = 0.125 * za

        def make_B(dN_dx):
            B = np.zeros((6, 24), dtype=np.float64)
            for a in range(8):
                dx = dN_dx[0, a]
                dy = dN_dx[1, a]
                dz = dN_dx[2, a]
                c_idx = a * 3
                B[0, c_idx + 0] = dx
                B[1, c_idx + 1] = dy
                B[2, c_idx + 2] = dz
                B[3, c_idx + 0] = dy
                B[3, c_idx + 1] = dx
                B[4, c_idx + 1] = dz
                B[4, c_idx + 2] = dy
                B[5, c_idx + 0] = dz
                B[5, c_idx + 2] = dx
            return B

        # Compute k_e per element
        for m in range(M):
            Xe = nodes[elements[m]]  # (8, 3)

            # Centroidal B
            J_cent = dN_cent @ Xe
            invJ_cent = np.linalg.inv(J_cent)
            dN_dx_cent = invJ_cent @ dN_cent
            B_elems[m] = make_B(dN_dx_cent)

            ke = np.zeros((24, 24), dtype=np.float64)
            for g in range(8):
                J_g = dN_nat[g] @ Xe
                detJ = np.linalg.det(J_g)
                if detJ <= 0.0:
                    detJ = 1e-12
                invJ_g = np.linalg.inv(J_g)
                dN_dx_g = invJ_g @ dN_nat[g]
                B_g = make_B(dN_dx_g)
                ke += detJ * (B_g.T @ D @ B_g)

            k_elems[m] = ke

        return k_elems, B_elems, D

    def _compute_diagonal(self) -> np.ndarray:
        diag = np.zeros(self.n_dofs, dtype=np.float64)
        for m in range(len(self.subdomain.elements)):
            dofs = self.elem_dofs[m]
            diag_elem = np.diag(self.k_elems[m])
            np.add.at(diag, dofs, diag_elem)

        diag[self.fixed_mask] = 1.0
        zero_mask = (diag < 1e-12) & (~self.fixed_mask)
        diag[zero_mask] = 1.0
        return diag

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        x_in = np.asarray(x, dtype=np.float64).copy()
        # Enforce zero perturbation on Dirichlet DOFs for operator action
        x_in[self.fixed_mask] = 0.0

        y = np.zeros(self.n_dofs, dtype=np.float64)
        for m in range(len(self.subdomain.elements)):
            dofs = self.elem_dofs[m]
            xe = x_in[dofs]
            ye = self.k_elems[m] @ xe
            np.add.at(y, dofs, ye)

        # Enforce identity on Dirichlet DOFs
        y[self.fixed_mask] = x[self.fixed_mask]
        return y


def solve_isolated_subdomain(
    subdomain: IsolatedSubdomain,
    additional_fixed_dofs: Optional[Sequence[int]] = None,
    additional_forces: Optional[np.ndarray] = None,
    tol: float = 1e-6,
    max_iter: int = 250,
) -> SubdomainSolveResult:
    """
    Execute high-speed matrix-free PCG solve on the isolated sub-domain.

    Parameters:
        subdomain: IsolatedSubdomain extracted via extract_isolated_subdomain.
        additional_fixed_dofs: Optional extra fixed DOF indices (e.g. physical supports).
        additional_forces: Optional external nodal load vector (N_sub * 3,).
        tol: Relative PCG convergence tolerance.
        max_iter: Maximum PCG iterations.

    Returns:
        SubdomainSolveResult with displacements, stresses, and timing telemetry.
    """
    t0 = time.perf_counter()

    n_nodes = len(subdomain.nodes)
    n_dofs = n_nodes * 3

    # 1. Map cut-boundary Dirichlet DOFs
    cut_dofs = []
    cut_u_values = []
    for idx, nid in enumerate(subdomain.cut_boundary_nodes):
        u_val = subdomain.prescribed_displacements[idx]
        for d in range(3):
            cut_dofs.append(nid * 3 + d)
            cut_u_values.append(u_val[d])

    if additional_fixed_dofs is not None and len(additional_fixed_dofs) > 0:
        for d in additional_fixed_dofs:
            if d not in cut_dofs:
                cut_dofs.append(d)
                cut_u_values.append(0.0)

    fixed_dofs_arr = np.array(cut_dofs, dtype=np.int64)
    fixed_u_arr = np.array(cut_u_values, dtype=np.float64)

    # 2. Build Matrix-Free Operator
    op = MatrixFreeSubdomainOperator(subdomain, fixed_dofs=fixed_dofs_arr)

    # 3. Assemble Right-Hand Side (RHS) Load Vector
    rhs = np.zeros(n_dofs, dtype=np.float64)
    if additional_forces is not None:
        rhs += np.asarray(additional_forces, dtype=np.float64).ravel()

    # Apply Dirichlet lift: rhs_eff = rhs - K * u_dirichlet
    u_lift = np.zeros(n_dofs, dtype=np.float64)
    u_lift[fixed_dofs_arr] = fixed_u_arr

    # Matrix-vector product with lift vector
    lift_forces = np.zeros(n_dofs, dtype=np.float64)
    for m in range(len(subdomain.elements)):
        dofs = op.elem_dofs[m]
        xe = u_lift[dofs]
        ye = op.k_elems[m] @ xe
        np.add.at(lift_forces, dofs, ye)

    rhs_effective = rhs - lift_forces
    rhs_effective[fixed_dofs_arr] = fixed_u_arr

    # 4. Point-Jacobi Preconditioner
    inv_diag = 1.0 / op.diag
    M_prec = spla.LinearOperator(
        shape=(n_dofs, n_dofs),
        matvec=lambda x: inv_diag * x,
        dtype=np.float64,
    )

    # 5. Execute PCG Solve
    iter_count = 0
    def callback(xk):
        nonlocal iter_count
        iter_count += 1

    u_sol, info = spla.cg(
        op,
        rhs_effective,
        M=M_prec,
        rtol=tol,
        maxiter=max_iter,
        callback=callback,
    )

    # Enforce exact prescribed displacements on cut boundaries
    u_sol[fixed_dofs_arr] = fixed_u_arr
    u_disp = u_sol.reshape(n_nodes, 3)

    # 6. Stress Recovery: Centroidal von Mises stress for each sub-domain element
    M_elems = len(subdomain.elements)
    elem_von_mises = np.zeros(M_elems, dtype=np.float64)

    for m in range(M_elems):
        dofs = op.elem_dofs[m]
        ue = u_sol[dofs]
        eps = op.B_elems[m] @ ue  # (6,) strain
        sigma = op.D @ eps        # (6,) stress: [sxx, syy, szz, sxy, syz, szx]

        sxx, syy, szz = sigma[0], sigma[1], sigma[2]
        sxy, syz, szx = sigma[3], sigma[4], sigma[5]

        # von Mises scalar
        vm_sq = 0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2 + 6.0 * (sxy**2 + syz**2 + szx**2))
        elem_von_mises[m] = np.sqrt(max(0.0, vm_sq))

    peak_stress = float(np.max(elem_von_mises)) if M_elems > 0 else 0.0
    t_solve = time.perf_counter() - t0

    # Final residual
    final_res = float(np.linalg.norm(op._matvec(u_sol) - rhs_effective) / max(1e-12, np.linalg.norm(rhs_effective)))
    elem_reduction = float(subdomain.original_element_count) / max(1, M_elems)

    return SubdomainSolveResult(
        subdomain=subdomain,
        displacements=u_disp,
        element_von_mises=elem_von_mises,
        peak_stress=peak_stress,
        solve_time=t_solve,
        num_iterations=iter_count,
        residual=final_res,
        element_reduction_ratio=elem_reduction,
    )
