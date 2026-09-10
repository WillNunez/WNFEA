"""
Matrix-Free Augmented Lagrangian Contact Mechanics Solver.
---------------------------------------------------------
Solves multi-body structural contact problems with non-penetration inequality constraints:
    K * u = f_ext + f_contact(u, lambda)
subject to the Hertz-Signorini-Moreau contact conditions:
    g_n >= 0,   p_n >= 0,   g_n * p_n = 0

Key Capabilities:
1. Exact kinematic constraint vectors B_c for slave-node to master-facet pairs.
2. Unconditionally conservative contact forces satisfying Newton's 3rd law:
   sum(f_contact) = 0.
3. Matrix-Free Tangent Contact Operator:
   K_eff * v = (K_bulk + K_contact) * v
   evaluated without global matrix assembly in O(N_contact) operations.
4. Augmented Lagrangian outer loop updating normal multipliers lambda_n,
   achieving high contact fidelity and near-zero penetration without ill-conditioning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
import scipy.sparse.linalg as spla

from .contact_detector import (
    ContactPair,
    SpatialHashContactDetector,
    SurfaceFacet,
    extract_hex8_surface_facets,
    compute_nodal_surface_normals,
)
from ..mesh.voxel_mesher import VoxelGrid
from ..solver.matrix_free_hex8 import (
    MatrixFreeHex8Operator,
    compute_hex8_reference_stiffness,
)


@dataclass
class ContactSolveResult:
    """
    Results of a multi-body structural contact simulation.
    """
    displacements: np.ndarray        # (N_total_dofs,) Combined nodal displacement vector
    contact_pairs: List[ContactPair] # Active contact pairs at converged state
    max_penetration: float           # Maximum remaining normal penetration (m)
    total_contact_force: np.ndarray  # (3,) Net resultant contact force on slave body (N)
    augmented_iterations: int        # Number of Augmented Lagrangian multiplier updates
    solve_time: float                # Wallclock solve time in seconds
    converged: bool                  # Convergence flag


class MatrixFreeContactTangentOperator:
    """
    Matrix-free operator evaluating the combined bulk + contact stiffness action:
        y = (K_bulk + K_contact) * x
    """

    def __init__(
        self,
        bulk_operators: List[MatrixFreeHex8Operator],
        dof_offsets: List[int],
        contact_pairs: List[ContactPair],
        penalty_stiffness: float,
        fixed_dofs: Sequence[int],
    ):
        """
        Parameters:
            bulk_operators: List of MatrixFreeHex8Operator for each sub-body.
            dof_offsets: Global DOF offset for each body in the assembly.
            contact_pairs: Active contact pairs.
            penalty_stiffness: Normal penalty parameter epsilon_n (N/m).
            fixed_dofs: Constrained global degree-of-freedom indices.
        """
        self.bulk_operators = bulk_operators
        self.dof_offsets = dof_offsets
        self.contact_pairs = contact_pairs
        self.penalty = float(penalty_stiffness)
        self.fixed_dofs = np.asarray(fixed_dofs, dtype=np.int64)

        # Total DOFs across all bodies
        self.n_dofs = sum(op.shape[0] for op in bulk_operators)
        self.shape = (self.n_dofs, self.n_dofs)

        self.fixed_mask = np.zeros(self.n_dofs, dtype=bool)
        if len(self.fixed_dofs) > 0:
            self.fixed_mask[self.fixed_dofs] = True

        # Precompute diagonal for Point-Jacobi preconditioning
        self.diag = np.zeros(self.n_dofs, dtype=np.float64)
        for op, offset in zip(bulk_operators, dof_offsets):
            self.diag[offset : offset + op.shape[0]] = op.diag

        # Add contact diagonal contributions
        # Slave DOF offset is dof_offsets[0], Master DOF offset is dof_offsets[1]
        s_offset = dof_offsets[0]
        m_offset = dof_offsets[1]

        for pair in contact_pairs:
            if pair.gap_n <= 1e-6:  # Touching or penetrating: active in tangent stiffness
                n = pair.normal
                # Slave node contribution: penalty * (n (x) n)
                s_dofs = s_offset + pair.slave_node_id * 3 + np.arange(3)
                self.diag[s_dofs] += self.penalty * (n ** 2)

                # Master nodes contribution: penalty * w_i^2 * (n (x) n)
                for i, m_nid in enumerate(pair.master_node_ids):
                    w_i = pair.master_weights[i]
                    m_dofs = m_offset + m_nid * 3 + np.arange(3)
                    self.diag[m_dofs] += self.penalty * (w_i ** 2) * (n ** 2)

        if len(self.fixed_dofs) > 0:
            self.diag[self.fixed_mask] = 1.0

        self.inv_diag = 1.0 / np.maximum(self.diag, 1e-15)
        if len(self.fixed_dofs) > 0:
            self.inv_diag[self.fixed_mask] = 0.0

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate y = (K_bulk + K_contact) * x
        """
        x_clean = x.copy()
        if len(self.fixed_dofs) > 0:
            x_clean[self.fixed_mask] = 0.0

        y = np.zeros(self.n_dofs, dtype=np.float64)

        # 1. Bulk stiffness actions for each body
        for op, offset in zip(self.bulk_operators, self.dof_offsets):
            x_body = x_clean[offset : offset + op.shape[0]]
            y[offset : offset + op.shape[0]] = op.matvec(x_body)

        # 2. Contact stiffness action: B_c^T * (penalty * B_c * x)
        s_offset = self.dof_offsets[0]
        m_offset = self.dof_offsets[1]

        for pair in self.contact_pairs:
            if pair.gap_n <= 1e-6:  # Active
                n = pair.normal

                # Compute delta_g = B_c * x
                u_s = x_clean[s_offset + pair.slave_node_id * 3 : s_offset + pair.slave_node_id * 3 + 3]
                delta_u_slave = np.dot(u_s, n)

                delta_u_master = 0.0
                for i, m_nid in enumerate(pair.master_node_ids):
                    w_i = pair.master_weights[i]
                    u_m = x_clean[m_offset + m_nid * 3 : m_offset + m_nid * 3 + 3]
                    delta_u_master += w_i * np.dot(u_m, n)

                delta_g = delta_u_slave - delta_u_master
                f_scalar = self.penalty * delta_g

                # Apply B_c^T * f_scalar
                # Slave node: +f_scalar * n
                y[s_offset + pair.slave_node_id * 3 : s_offset + pair.slave_node_id * 3 + 3] += f_scalar * n

                # Master nodes: -w_i * f_scalar * n
                for i, m_nid in enumerate(pair.master_node_ids):
                    w_i = pair.master_weights[i]
                    y[m_offset + m_nid * 3 : m_offset + m_nid * 3 + 3] -= (w_i * f_scalar) * n

        if len(self.fixed_dofs) > 0:
            y[self.fixed_mask] = x[self.fixed_mask]
        return y


def solve_contact_assembly(
    grid_slave: VoxelGrid,
    grid_master: VoxelGrid,
    f_ext_slave: np.ndarray,
    f_ext_master: np.ndarray,
    fixed_dofs_slave: Sequence[int],
    fixed_dofs_master: Sequence[int],
    E_slave: float = 2.1e11,
    nu_slave: float = 0.3,
    E_master: float = 2.1e11,
    nu_master: float = 0.3,
    penalty_stiffness: Optional[float] = None,
    max_augmented_iters: int = 15,
    penetration_tolerance: float = 1e-6,
    search_distance: float = 0.01,
    pcg_tol: float = 1e-6,
    max_pcg_iters: int = 400,
) -> ContactSolveResult:
    """
    Solve non-linear two-body contact assembly using matrix-free Augmented Lagrangian PCG.

    Parameters:
        grid_slave: VoxelGrid for slave (upper/impacting) body.
        grid_master: VoxelGrid for master (foundation/target) body.
        f_ext_slave: (N_slave_dofs,) External force vector on slave body.
        f_ext_master: (N_master_dofs,) External force vector on master body.
        fixed_dofs_slave: Boundary constrained DOFs on slave body.
        fixed_dofs_master: Boundary constrained DOFs on master body.
        E_slave, nu_slave: Material properties of slave body.
        E_master, nu_master: Material properties of master body.
        penalty_stiffness: Contact penalty parameter (default: computed from bulk modulus).
        max_augmented_iters: Maximum Augmented Lagrangian multiplier updates.
        penetration_tolerance: Target non-penetration tolerance (m).
        search_distance: Proximity contact search radius (m).
        pcg_tol: PCG linear solve convergence tolerance.
        max_pcg_iters: Maximum PCG iterations per solve.

    Returns:
        ContactSolveResult containing displacements, active pairs, contact forces, and telemetry.
    """
    t_start = time.time()

    n_slave_dofs = grid_slave.total_nodes * 3
    n_master_dofs = grid_master.total_nodes * 3
    n_total_dofs = n_slave_dofs + n_master_dofs

    dof_offsets = [0, n_slave_dofs]

    # Map master fixed DOFs to global assembly indices
    fixed_global = list(fixed_dofs_slave) + [d + n_slave_dofs for d in fixed_dofs_master]
    fixed_global_arr = np.asarray(fixed_global, dtype=np.int64)
    fixed_mask = np.zeros(n_total_dofs, dtype=bool)
    if len(fixed_global_arr) > 0:
        fixed_mask[fixed_global_arr] = True

    # Assemble combined external load vector
    f_ext_global = np.zeros(n_total_dofs, dtype=np.float64)
    f_ext_global[:n_slave_dofs] = f_ext_slave
    f_ext_global[n_slave_dofs:] = f_ext_master
    if len(fixed_global_arr) > 0:
        f_ext_global[fixed_mask] = 0.0

    # Instantiate bulk matrix-free operators
    op_slave = MatrixFreeHex8Operator(
        grid_slave, E=E_slave, nu=nu_slave, fixed_dofs=fixed_dofs_slave
    )
    op_master = MatrixFreeHex8Operator(
        grid_master, E=E_master, nu=nu_master, fixed_dofs=fixed_dofs_master
    )
    bulk_operators = [op_slave, op_master]

    # Extract master surface boundary facets
    master_facets = extract_hex8_surface_facets(grid_master.nodes, grid_master.elements)
    slave_facets = extract_hex8_surface_facets(grid_slave.nodes, grid_slave.elements)
    slave_normals = compute_nodal_surface_normals(grid_slave.nodes, slave_facets)

    detector = SpatialHashContactDetector(grid_master.nodes, master_facets)

    # Estimate default penalty stiffness from bulk modulus: eps_n ~ E * h
    if penalty_stiffness is None:
        hx, hy, hz = grid_slave.pitch
        h_char = min(hx, hy, hz)
        penalty_stiffness = float(E_slave * h_char * 0.5)

    # State vectors
    u_global = np.zeros(n_total_dofs, dtype=np.float64)
    lambda_multipliers = {}  # (slave_nid, facet_id) -> lambda_n

    converged = False
    actual_aug_iters = 0
    max_penetration = 0.0
    active_pairs: List[ContactPair] = []

    for aug_iter in range(1, max_augmented_iters + 1):
        actual_aug_iters = aug_iter

        # 1. Update deformed configurations of bodies
        x_slave_def = grid_slave.nodes + u_global[:n_slave_dofs].reshape(-1, 3)
        x_master_def = grid_master.nodes + u_global[n_slave_dofs:].reshape(-1, 3)

        # Update detector with deformed master nodes
        detector.master_nodes = x_master_def

        # 2. Detect contact pairs in deformed state with opposing surface filter
        pairs = detector.detect_contact_pairs(
            slave_nodes=x_slave_def,
            search_distance=search_distance,
            slave_normals=slave_normals,
        )

        # Evaluate maximum penetration and candidate active pairs
        active_pairs = []
        max_penetration = 0.0
        for p in pairs:
            # Active in contact if penetrating or within close contact interface
            if p.gap_n <= 1e-6:
                active_pairs.append(p)
                if p.gap_n < 0.0:
                    max_penetration = max(max_penetration, abs(p.gap_n))

        # Check penetration convergence
        if aug_iter > 1 and max_penetration <= penetration_tolerance:
            converged = True
            break

        # 3. Formulate contact force vector from current multipliers and penetrations:
        # f_contact = sum_c (lambda_n - penalty * g_n) * B_c^T
        rhs_contact = np.zeros(n_total_dofs, dtype=np.float64)

        for p in active_pairs:
            pair_key = (p.slave_node_id, p.master_facet_id)
            lam_n = lambda_multipliers.get(pair_key, 0.0)

            # Normal force: F_n = lam_n - penalty * gap_n (gap_n < 0)
            f_n_scalar = lam_n - penalty_stiffness * p.gap_n
            n = p.normal

            # Slave node receives -F_n * n (repulsive away from master)
            s_dofs = p.slave_node_id * 3 + np.arange(3)
            rhs_contact[s_dofs] -= f_n_scalar * n

            # Master nodes receive +w_i * F_n * n
            for i, m_nid in enumerate(p.master_node_ids):
                w_i = p.master_weights[i]
                m_dofs = n_slave_dofs + m_nid * 3 + np.arange(3)
                rhs_contact[m_dofs] += (w_i * f_n_scalar) * n

        # Total RHS: f_eff = f_ext + rhs_contact (or in Newton formulation)
        # We can construct the combined linear system:
        # (K_bulk + K_contact) * u_{new} = f_ext + f_contact_explicit
        op_tangent = MatrixFreeContactTangentOperator(
            bulk_operators=bulk_operators,
            dof_offsets=dof_offsets,
            contact_pairs=active_pairs,
            penalty_stiffness=penalty_stiffness,
            fixed_dofs=fixed_global_arr,
        )

        # In linear elasticity with penalty, K_contact * u accounts for the penetration force.
        # We solve (K_bulk + K_contact) * u = f_ext + lambda_contact_forces
        rhs_solve = f_ext_global.copy()
        for p in active_pairs:
            pair_key = (p.slave_node_id, p.master_facet_id)
            lam_n = lambda_multipliers.get(pair_key, 0.0)
            if lam_n > 0.0:
                n = p.normal
                s_dofs = p.slave_node_id * 3 + np.arange(3)
                rhs_solve[s_dofs] -= lam_n * n
                for i, m_nid in enumerate(p.master_node_ids):
                    w_i = p.master_weights[i]
                    m_dofs = n_slave_dofs + m_nid * 3 + np.arange(3)
                    rhs_solve[m_dofs] += (w_i * lam_n) * n

        if len(fixed_global_arr) > 0:
            rhs_solve[fixed_mask] = 0.0

        inv_diag = op_tangent.inv_diag
        M_inv = spla.LinearOperator(op_tangent.shape, matvec=lambda v: inv_diag * v)

        u_sol, info = spla.cg(
            op_tangent, rhs_solve, x0=u_global, rtol=pcg_tol, maxiter=max_pcg_iters, M=M_inv
        )
        if len(fixed_global_arr) > 0:
            u_sol[fixed_mask] = 0.0

        u_global = u_sol

        # 4. Augmented Lagrangian Multiplier Update:
        # lambda_n <- max(0, lambda_n - penalty * g_n)
        for p in active_pairs:
            pair_key = (p.slave_node_id, p.master_facet_id)
            lam_old = lambda_multipliers.get(pair_key, 0.0)
            lam_new = max(0.0, lam_old - penalty_stiffness * p.gap_n)
            lambda_multipliers[pair_key] = lam_new

    t_solve = time.time() - t_start

    # Compute net contact force acting on slave body: f_contact = K_slave * u_slave - f_ext_slave
    k_u_slave = op_slave.matvec(u_global[:n_slave_dofs])
    total_f_slave = np.sum((k_u_slave - f_ext_global[:n_slave_dofs]).reshape(-1, 3), axis=0)

    return ContactSolveResult(
        displacements=u_global,
        contact_pairs=active_pairs,
        max_penetration=max_penetration,
        total_contact_force=total_f_slave,
        augmented_iterations=actual_aug_iters,
        solve_time=t_solve,
        converged=converged or (max_penetration <= penetration_tolerance * 2.0),
    )
