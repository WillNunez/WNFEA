"""
3D Euler–Bernoulli beam solver for WNFEA.

Assembles the global stiffness matrix, applies boundary conditions (supports),
builds the load vector, and solves the linear system  K·U = F  using a direct
sparse or dense solver.

The element formulation follows the classical 12-DOF beam element with:
    - Axial stiffness  (DOFs 0, 6)
    - Torsional stiffness (DOFs 3, 9)
    - Bending about local z (DOFs 1, 5, 7, 11)
    - Bending about local y (DOFs 2, 4, 8, 10)

Coordinate transformation from the local element frame to the global frame
uses a 12×12 block-diagonal rotation matrix built from the 3×3 direction
cosine matrix of the element axis.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve as scipy_solve

from ..model import FEAModel
from ..boundary.conditions import DOFType


class SolverError(Exception):
    """Raised when the solver encounters an unrecoverable error."""
    pass


class BeamSolver:
    """
    Solves the assembled linear system for a beam FEA model.

    Usage::

        solver = BeamSolver()
        solver.solve(model)
        # model.displacements is now populated
    """

    def solve(self, model: FEAModel, device: str = "cpu") -> None:
        """
        Assemble and solve K·U = F.

        After a successful solve, ``model.displacements`` contains the
        global displacement vector of shape ``(N*6,)`` and
        ``model.element_results`` contains per-element result dicts.

        Raises
        ------
        SolverError
            If validation fails or the system is singular.
        """
        errors = model.validate_for_solving()
        if errors:
            raise SolverError(
                "Cannot solve – validation errors:\n  " + "\n  ".join(errors)
            )

        nodes = model.mesh_nodes         # (N, 3)
        elements = model.mesh_elements   # (E, 2)
        n_nodes = len(nodes)
        n_dof = n_nodes * 6

        # ------------------------------------------------------------------
        # 1. Assemble global stiffness matrix
        # ------------------------------------------------------------------
        K = np.zeros((n_dof, n_dof))

        for elem_id in range(len(elements)):
            n1_idx, n2_idx = elements[elem_id]
            node1 = nodes[n1_idx]
            node2 = nodes[n2_idx]

            # Look up material and section for this element
            assignment = model.element_properties[elem_id]
            material = model.materials[assignment.material_name]
            section = model.sections[assignment.section_name]

            E = material.youngs_modulus
            G = material.shear_modulus
            A = section.area
            Iy = section.iy
            Iz = section.iz
            J = section.j

            Ke_local = self._local_stiffness(E, G, A, Iy, Iz, J, node1, node2)
            T = self._transformation_matrix(node1, node2)

            # Transform to global: Ke_global = T^T · Ke_local · T
            Ke_global = T.T @ Ke_local @ T

            # DOF mapping for this element
            dofs = np.concatenate([
                np.arange(n1_idx * 6, n1_idx * 6 + 6),
                np.arange(n2_idx * 6, n2_idx * 6 + 6),
            ])

            # Scatter into global K
            for i in range(12):
                for j in range(12):
                    K[dofs[i], dofs[j]] += Ke_global[i, j]

        # ------------------------------------------------------------------
        # 2. Build the global force vector
        # ------------------------------------------------------------------
        F = np.zeros(n_dof)

        for load in model.loads:
            mesh_node_id = self._resolve_node_id(load.node_id, load.is_geometry_node, model)
            if mesh_node_id is None:
                continue
            base_dof = mesh_node_id * 6
            fv = load.force_vector  # [Fx, Fy, Fz, Mx, My, Mz]
            for i in range(6):
                F[base_dof + i] += fv[i]

        # ------------------------------------------------------------------
        # 3. Apply boundary conditions (penalty-free row/col zeroing)
        # ------------------------------------------------------------------
        # Keep a copy of K for reaction force calculation later
        K_original = K.copy()
        F_original = F.copy()

        for support in model.supports:
            mesh_node_id = self._resolve_node_id(
                support.node_id, support.is_geometry_node, model
            )
            if mesh_node_id is None:
                continue

            base_dof = mesh_node_id * 6
            constraints = support.constraints  # list of 6 DOFConstraint

            for local_i, constraint in enumerate(constraints):
                global_dof = base_dof + local_i

                if constraint.dof_type == DOFType.FIXED:
                    K[global_dof, :] = 0.0
                    K[:, global_dof] = 0.0
                    K[global_dof, global_dof] = 1.0
                    F[global_dof] = 0.0

                elif constraint.dof_type == DOFType.PRESCRIBED:
                    prescribed_val = constraint.value
                    # Move known displacement contribution to RHS
                    F -= K[:, global_dof] * prescribed_val
                    K[global_dof, :] = 0.0
                    K[:, global_dof] = 0.0
                    K[global_dof, global_dof] = 1.0
                    F[global_dof] = prescribed_val

        # ------------------------------------------------------------------
        # 4. Solve  K · U = F
        # ------------------------------------------------------------------
        try:
            from .mfem_solver_wrapper import solve_with_mfem
            U = solve_with_mfem(K, F, device=device)
        except Exception as exc:
            raise SolverError(f"MFEM solver failed: {exc}") from exc

        model.displacements = U

        # ------------------------------------------------------------------
        # 5. Compute reaction forces:  R = K_original · U  − F_original
        # ------------------------------------------------------------------
        reactions = K_original @ U - F_original
        model.reaction_forces = reactions

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _local_stiffness(
        E: float, G: float, A: float,
        Iy: float, Iz: float, J: float,
        node1: np.ndarray, node2: np.ndarray,
    ) -> np.ndarray:
        """Build the 12×12 local beam stiffness matrix."""
        L = float(np.linalg.norm(node2 - node1))
        if L < 1e-12:
            return np.zeros((12, 12))

        Ke = np.zeros((12, 12))

        # Axial
        axial = E * A / L
        Ke[0, 0] = Ke[6, 6] = axial
        Ke[0, 6] = Ke[6, 0] = -axial

        # Torsion
        torsion = G * J / L
        Ke[3, 3] = Ke[9, 9] = torsion
        Ke[3, 9] = Ke[9, 3] = -torsion

        # Bending in x-z plane (about local y → uses Iy)
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

        # Bending in x-y plane (about local z → uses Iz)
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

        return Ke

    @staticmethod
    def _rotation_matrix_3x3(node1: np.ndarray, node2: np.ndarray) -> np.ndarray:
        """
        Build the 3×3 direction cosine matrix from global to local coordinates.

        The local x-axis is along the element (node1 → node2).
        The local y and z axes are constructed via a reference vector.
        """
        vec = node2 - node1
        L = float(np.linalg.norm(vec))
        if L < 1e-12:
            return np.eye(3)

        local_x = vec / L

        # Choose a reference vector that is NOT parallel to local_x
        if np.abs(np.abs(local_x[0]) - 1.0) < 0.1:
            # Element is roughly along global X → use global Z as reference
            ref = np.array([0.0, 0.0, 1.0])
        elif np.abs(np.abs(local_x[1]) - 1.0) < 0.1:
            # Element is roughly along global Y → use global X as reference
            ref = np.array([1.0, 0.0, 0.0])
        elif np.abs(np.abs(local_x[2]) - 1.0) < 0.1:
            # Element is roughly along global Z → use global X as reference
            ref = np.array([1.0, 0.0, 0.0])
        else:
            # General orientation → use global Z as reference
            ref = np.array([0.0, 0.0, 1.0])

        local_y = np.cross(ref, local_x)
        norm_y = np.linalg.norm(local_y)
        if norm_y < 1e-9:
            # Fallback if ref is parallel to local_x
            ref = np.array([0.0, 1.0, 0.0])
            local_y = np.cross(ref, local_x)
            norm_y = np.linalg.norm(local_y)
        local_y = local_y / norm_y

        local_z = np.cross(local_x, local_y)
        local_z = local_z / np.linalg.norm(local_z)

        # Rows = local axes expressed in global coordinates
        R = np.array([local_x, local_y, local_z])
        return R

    @classmethod
    def _transformation_matrix(cls, node1: np.ndarray, node2: np.ndarray) -> np.ndarray:
        """Build the 12×12 block-diagonal transformation matrix."""
        R = cls._rotation_matrix_3x3(node1, node2)
        T = np.zeros((12, 12))
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R
        return T

    @staticmethod
    def _resolve_node_id(
        node_id: int,
        is_geometry_node: bool,
        model: FEAModel,
    ) -> int | None:
        """
        Resolve a node_id to a mesh node index.

        If ``is_geometry_node`` is True, look up the mapping; otherwise
        return the id directly.  Returns None if the mapping is missing.
        """
        if is_geometry_node:
            return model.geometry_to_mesh_node_map.get(node_id)
        return node_id
