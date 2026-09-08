"""
DOF Manager and Direct Elimination Engine for WNFEA.

Handles mixed systems of 6-DOF beam nodes and 3-DOF solid continuum nodes:
1. Directly eliminates inactive rotational DOFs at pure solid nodes.
2. Directly eliminates slave nodal DOFs at beam-to-solid rigid kinematic interfaces.
3. Provides bijective mapping between independent global DOFs and nodal components.
"""

from __future__ import annotations

import numpy as np

from ..model import FEAModel


class DOFManager:
    """
    Manages active degree of freedom allocation and master-slave direct elimination
    for mixed 6-DOF (beam) and 3-DOF (solid) finite element models.
    """

    def __init__(self, model: FEAModel):
        self.model = model
        self.n_nodes = len(model.mesh_nodes)

        # Classify nodes: beam nodes vs solid nodes
        self.beam_nodes: set[int] = set()
        if model.mesh_elements is not None and len(model.mesh_elements) > 0:
            for n1, n2 in model.mesh_elements:
                self.beam_nodes.add(int(n1))
                self.beam_nodes.add(int(n2))

        self.solid_nodes: set[int] = set()
        if getattr(model, "solid_elements", None) is not None and len(model.solid_elements) > 0:
            for elem in model.solid_elements:
                for n_idx in elem:
                    self.solid_nodes.add(int(n_idx))

        # Identify slave nodes from rigid couplings
        self.slave_nodes: set[int] = set()
        self.slave_to_coupling: dict[int, tuple[int, np.ndarray]] = {}  # slave_id -> (master_id, T_matrix)

        couplings = getattr(model, "couplings", [])
        for coup in couplings:
            master_coord = model.mesh_nodes[coup.master_node_id]
            slave_coords = [model.mesh_nodes[sid] for sid in coup.slave_node_ids]
            t_matrices = coup.compute_coupling_matrices(master_coord, slave_coords)

            for sid, T_i in zip(coup.slave_node_ids, t_matrices):
                self.slave_nodes.add(int(sid))
                self.slave_to_coupling[int(sid)] = (coup.master_node_id, T_i)

        # Build active DOF map
        self.node_dof_to_global: dict[tuple[int, int], int] = {}
        self.global_to_node_dof: list[tuple[int, int]] = []
        self.active_dofs_per_node: dict[int, list[int]] = {}

        global_idx = 0
        for node_id in range(self.n_nodes):
            if node_id in self.slave_nodes:
                # Slave node: all DOFs directly eliminated from independent system
                self.active_dofs_per_node[node_id] = []
                continue

            # Check whether node has beam connectivity or is purely solid
            if node_id in self.beam_nodes or (node_id not in self.solid_nodes and len(self.solid_nodes) == 0):
                # 6 active DOFs: [ux, uy, uz, rx, ry, rz]
                local_active = [0, 1, 2, 3, 4, 5]
            else:
                # Pure solid node: rotational DOFs (3, 4, 5) directly eliminated!
                local_active = [0, 1, 2]

            self.active_dofs_per_node[node_id] = local_active
            for ld in local_active:
                self.node_dof_to_global[(node_id, ld)] = global_idx
                self.global_to_node_dof.append((node_id, ld))
                global_idx += 1

        self.total_active_dofs = global_idx
        self.map_nodes = np.array([item[0] for item in self.global_to_node_dof], dtype=np.int32)
        self.map_dofs = np.array([item[1] for item in self.global_to_node_dof], dtype=np.int32)
        self.flat_active_indices = self.map_nodes * 6 + self.map_dofs

    def expand_displacements(self, u_active: np.ndarray) -> np.ndarray:
        """
        Expand an active displacement vector of shape (total_active_dofs,)
        into the full nodal displacement representation of shape (n_nodes * 6,).
        Kinematically calculates slave displacements from master beam motions.
        """
        u_full = np.zeros((self.n_nodes, 6), dtype=np.float64)

        # 1. Vectorized fill of independent active DOFs
        u_full[self.map_nodes, self.map_dofs] = u_active

        # 2. Kinematic substitution for slave nodes: u_s = T_i * [u_m; theta_m]
        couplings = getattr(self.model, "couplings", [])
        for coup in couplings:
            m_id = coup.master_node_id
            u_m_full = u_full[m_id, :]  # (6,)
            master_coord = self.model.mesh_nodes[m_id]
            slave_coords = [self.model.mesh_nodes[sid] for sid in coup.slave_node_ids]
            t_matrices = coup.compute_coupling_matrices(master_coord, slave_coords)

            for sid, T_i in zip(coup.slave_node_ids, t_matrices):
                # Translational motion
                u_full[sid, 0:3] = T_i @ u_m_full
                u_full[sid, 3:6] = 0.0

        return u_full.reshape(-1)

    def condense_forces(self, f_full: np.ndarray) -> np.ndarray:
        """
        Condense a full nodal force/moment vector of shape (n_nodes * 6,)
        into the active independent DOF space of shape (total_active_dofs,).
        Transfers slave forces and induced moments onto master nodes.
        """
        f_work = np.asarray(f_full, dtype=np.float64).reshape((self.n_nodes, 6)).copy()

        # Gather slave forces and moments onto master nodes
        couplings = getattr(self.model, "couplings", [])
        for coup in couplings:
            m_id = coup.master_node_id
            master_coord = self.model.mesh_nodes[m_id]

            for sid in coup.slave_node_ids:
                f_slave = f_work[sid, 0:3]
                slave_coord = self.model.mesh_nodes[sid]
                r = slave_coord - master_coord

                # Force equilibrium: F_m += F_s
                f_work[m_id, 0:3] += f_slave
                # Moment equilibrium: M_m += r x F_s
                f_work[m_id, 3:6] += np.cross(r, f_slave)

        # Vectorized extraction of active independent DOFs
        return f_work[self.map_nodes, self.map_dofs]

    def condense_displacements(self, u_full: np.ndarray) -> np.ndarray:
        """
        Extract active independent DOFs from a full displacement vector
        of shape (n_nodes * 6,) or (n_nodes, 6).
        """
        u_work = np.asarray(u_full, dtype=np.float64).reshape((self.n_nodes, 6))
        return u_work[self.map_nodes, self.map_dofs]

    def build_projection_matrix(self) -> np.ndarray:
        """
        Build the transformation matrix T_proj of shape (6*n_nodes, total_active_dofs)
        such that u_full = T_proj @ u_active.
        """
        T_proj = np.zeros((6 * self.n_nodes, self.total_active_dofs), dtype=np.float64)

        # 1. Independent active DOFs: identity mapping
        for g_idx, (node_id, local_dof) in enumerate(self.global_to_node_dof):
            T_proj[node_id * 6 + local_dof, g_idx] = 1.0

        # 2. Master-slave kinematic coupling: u_s = T_i @ [u_m; theta_m]
        couplings = getattr(self.model, "couplings", [])
        for coup in couplings:
            m_id = coup.master_node_id
            master_coord = self.model.mesh_nodes[m_id]
            slave_coords = [self.model.mesh_nodes[sid] for sid in coup.slave_node_ids]
            t_matrices = coup.compute_coupling_matrices(master_coord, slave_coords)

            # Master active DOF indices
            master_g_dofs = [self.node_dof_to_global.get((m_id, ld)) for ld in range(6)]

            for sid, T_i in zip(coup.slave_node_ids, t_matrices):
                for row in range(3):
                    for col in range(6):
                        m_g_idx = master_g_dofs[col]
                        if m_g_idx is not None:
                            T_proj[sid * 6 + row, m_g_idx] += T_i[row, col]

        return T_proj

