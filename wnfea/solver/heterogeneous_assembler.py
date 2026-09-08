"""
Heterogeneous CPU + GPU Subsystem Assembler & Solver for WNFEA.

Optimized for large structural models where:
- 99% of DOFs are 3D solid continuum elements (C3D10 quadratic tetrahedra).
- 1% of DOFs are beam elements (6 DOFs per node: stringers, stiffeners, bolts, pins).

Architecture:
1. Solid Subsystem (GPU): 100% matrix-free elemental evaluation in native AMD HIP Wave32
   (or vectorized AVX-FMA CPU fallback), avoiding gigabytes of global sparse matrix storage.
2. Beam Subsystem (CPU): Compact 12x12 Euler-Bernoulli elements assembled into a sparse
   CSR matrix on host CPU threads.
3. Kinematic Master-Slave Coupling: Transferred matrix-free via DOFManager projection
   without Lagrange multipliers, preserving strict symmetry and positive-definiteness.
4. Krylov PCG Pipeline: Runs seamlessly across active independent DOFs with zero CPU-GPU
   transfer thrashing.
"""

from __future__ import annotations

import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator
from typing import Optional, Callable

from ..model import FEAModel
from ..boundary.conditions import DOFType
from ..boundary.body_loads import AccelerationField, SpatialPointLoad
from .dof_manager import DOFManager
from .matrix_free_c3d10 import MatrixFreeC3D10Operator
from .assembler import build_element_stiffness_3d_beam, build_transformation_matrix


class HeterogeneousOperator(LinearOperator):
    """
    Heterogeneous LinearOperator coupling matrix-free solid GPU compute
    with sparse CPU beam assembly and kinematic direct elimination.
    """

    def __init__(
        self,
        model: FEAModel,
        apply_bcs: bool = True,
        precompute_Ke: bool = False,
        device: str = "auto",
        precision: str = "fp64",
    ):
        self.model = model
        self.apply_bcs = apply_bcs
        self.device = device
        self.precompute_Ke = precompute_Ke
        self.precision = precision.lower()
        self.dtype = np.float32 if self.precision == "fp32" else np.float64

        self.n_nodes = len(model.mesh_nodes)
        self.dof_mgr = DOFManager(model)
        self.n_dofs = self.dof_mgr.total_active_dofs
        self.shape = (self.n_dofs, self.n_dofs)

        # 1. Setup Solid Subsystem (Matrix-Free GPU/CPU)
        has_solids = getattr(model, "solid_elements", None) is not None and len(model.solid_elements) > 0
        self.has_solids = has_solids
        if has_solids:
            self.solid_op = MatrixFreeC3D10Operator(
                model,
                apply_bcs=False,  # Boundary conditions handled at active DOF level
                precompute_Ke=precompute_Ke,
                device=device,
                precision=precision,
            )
        else:
            self.solid_op = None

        # 2. Setup Beam Subsystem (Sparse CSR on CPU)
        has_beams = model.mesh_elements is not None and len(model.mesh_elements) > 0
        self.has_beams = has_beams
        if has_beams:
            self.K_beam = self._build_beam_csr()
        else:
            self.K_beam = None

        # 3. Identify Fixed Dirichlet Boundary Conditions in Active DOF space
        self.fixed_dofs: list[int] = []
        if apply_bcs and model.supports:
            for support in model.supports:
                if support.is_geometry_node:
                    mesh_node_id = model.geometry_to_mesh_node_map.get(support.node_id)
                    if mesh_node_id is None:
                        continue
                else:
                    mesh_node_id = support.node_id

                if mesh_node_id >= self.n_nodes:
                    continue

                for local_dof, constraint in enumerate(support.constraints):
                    global_dof = self.dof_mgr.node_dof_to_global.get((mesh_node_id, local_dof))
                    if global_dof is not None and constraint.dof_type == DOFType.FIXED:
                        self.fixed_dofs.append(global_dof)

        self.fixed_dofs = np.array(sorted(set(self.fixed_dofs)), dtype=np.int32)

        # 4. Extract diagonal for Point Jacobi preconditioning
        self.diag_K = self._compute_diagonal()
        self.inv_diag_K = np.where(np.abs(self.diag_K) > 1e-30, 1.0 / self.diag_K, 1.0)

        super().__init__(dtype=self.dtype, shape=self.shape)

    def _build_beam_csr(self) -> csr_matrix:
        """Assemble the 1% beam stiffness matrix into a compact CSR on CPU."""
        n_elem = len(self.model.mesh_elements)
        rows_list = []
        cols_list = []
        vals_list = []

        for e_idx, (n1_idx, n2_idx) in enumerate(self.model.mesh_elements):
            node1 = self.model.mesh_nodes[n1_idx]
            node2 = self.model.mesh_nodes[n2_idx]

            prop = self.model.element_properties.get(e_idx)
            mat_name = prop.material_name if prop else None
            sec_name = prop.section_name if prop else None

            mat = self.model.materials.get(mat_name) if mat_name else next(iter(self.model.materials.values()))
            sec = self.model.sections.get(sec_name) if sec_name else next(iter(self.model.sections.values()))

            E = mat.youngs_modulus
            G = getattr(mat, "shear_modulus", E / (2.0 * (1.0 + mat.poissons_ratio)))

            Ke_local = build_element_stiffness_3d_beam(
                node1, node2, E, G, sec.area, sec.iy, sec.iz, sec.j
            )
            T = build_transformation_matrix(node1, node2)
            Ke_global = T.T @ Ke_local @ T

            elem_dofs = np.concatenate([
                np.arange(n1_idx * 6, n1_idx * 6 + 6),
                np.arange(n2_idx * 6, n2_idx * 6 + 6),
            ])

            r_grid, c_grid = np.meshgrid(elem_dofs, elem_dofs, indexing="ij")
            rows_list.append(r_grid.ravel())
            cols_list.append(c_grid.ravel())
            vals_list.append(Ke_global.ravel())

        all_rows = np.concatenate(rows_list)
        all_cols = np.concatenate(cols_list)
        all_vals = np.concatenate(vals_list)

        n_full_dofs = self.n_nodes * 6
        return csr_matrix((all_vals, (all_rows, all_cols)), shape=(n_full_dofs, n_full_dofs), dtype=np.float64)

    def _compute_diagonal(self) -> np.ndarray:
        """Compute the exact global diagonal in active DOF space."""
        diag_full = np.zeros((self.n_nodes, 6), dtype=np.float64)

        if self.has_solids and self.solid_op is not None:
            diag_solid = self.solid_op.diag_K.reshape(self.n_nodes, 3)
            diag_full[:, 0:3] += diag_solid

        if self.has_beams and self.K_beam is not None:
            diag_beam = np.array(self.K_beam.diagonal()).reshape(self.n_nodes, 6)
            diag_full += diag_beam

        # Accumulate slave node stiffness onto master nodes for RigidCouplings
        couplings = getattr(self.model, "couplings", [])
        for coup in couplings:
            m_id = coup.master_node_id
            m_coord = self.model.mesh_nodes[m_id]
            for s_id in coup.slave_node_ids:
                s_coord = self.model.mesh_nodes[s_id]
                r = s_coord - m_coord
                rx, ry, rz = r
                d_solid = diag_full[s_id, 0:3]  # (dx, dy, dz)
                # Translations
                diag_full[m_id, 0:3] += d_solid
                # Rotations from offset moment arms
                diag_full[m_id, 3] += d_solid[1] * (rz**2) + d_solid[2] * (ry**2)
                diag_full[m_id, 4] += d_solid[0] * (rz**2) + d_solid[2] * (rx**2)
                diag_full[m_id, 5] += d_solid[0] * (ry**2) + d_solid[1] * (rx**2)

        # Map to active DOFs (fully vectorized)
        diag_active = diag_full[self.dof_mgr.map_nodes, self.dof_mgr.map_dofs]

        if self.apply_bcs and len(self.fixed_dofs) > 0:
            diag_active[self.fixed_dofs] = 1.0

        return diag_active

    def _matvec(self, u: np.ndarray) -> np.ndarray:
        """
        Evaluate global coupled action v = K @ u:
        1. Expand active independent displacements into full nodal DOFs via kinematics.
        2. Evaluate 99% solid continuum on GPU (matrix-free).
        3. Evaluate 1% beam subsystem on CPU (sparse CSR SpMV).
        4. Condense forces & moments back to active DOF space.
        5. Enforce Dirichlet boundary conditions.
        """
        u_arr = np.asarray(u, dtype=self.dtype)

        # Apply Dirichlet zeroing to input vector
        if self.apply_bcs and len(self.fixed_dofs) > 0:
            u_eval = np.copy(u_arr)
            u_eval[self.fixed_dofs] = 0.0
        else:
            u_eval = u_arr

        # 1. Expand to full 6-DOF per node representation (O(N) kinematics)
        u_full = self.dof_mgr.expand_displacements(u_eval)  # shape (n_nodes * 6,)
        v_full = np.zeros((self.n_nodes, 6), dtype=np.float64)

        # 2. Evaluate Solid Subsystem on GPU (matrix-free Wave32)
        if self.has_solids and self.solid_op is not None:
            u_solid = u_full.reshape(self.n_nodes, 6)[:, 0:3].ravel()
            v_solid = self.solid_op._matvec(u_solid)
            v_full[:, 0:3] += v_solid.reshape(self.n_nodes, 3)

        # 3. Evaluate Beam Subsystem on CPU
        if self.has_beams and self.K_beam is not None:
            v_beam = self.K_beam @ u_full
            v_full += v_beam.reshape(self.n_nodes, 6)

        # 4. Condense forces and moments across rigid master-slave couplings
        v_active = self.dof_mgr.condense_forces(v_full.ravel())

        # 5. Enforce Dirichlet identity action on fixed DOFs
        if self.apply_bcs and len(self.fixed_dofs) > 0:
            v_active[self.fixed_dofs] = u_arr[self.fixed_dofs]

        return v_active.astype(self.dtype)

    def assemble_rhs(
        self,
        acceleration: Optional[Union[AccelerationField, tuple[float, float, float]]] = None,
    ) -> np.ndarray:
        """
        Assemble the active RHS force vector F from applied nodal loads,
        acceleration fields (gravity/inertia), and spatial point loads.
        """
        F_full = np.zeros((self.n_nodes, 6), dtype=np.float64)

        # 1. Concentrated nodal loads
        for load in self.model.loads:
            if load.is_geometry_node:
                mesh_node_id = self.model.geometry_to_mesh_node_map.get(load.node_id)
                if mesh_node_id is None:
                    continue
            else:
                mesh_node_id = load.node_id

            if mesh_node_id >= self.n_nodes:
                continue

            fv = load.force_vector
            for i in range(6):
                F_full[mesh_node_id, i] += fv[i]

        # 2. Spatial point loads
        if hasattr(self.model, "spatial_point_loads") and self.model.spatial_point_loads:
            for sp_load in self.model.spatial_point_loads:
                dist_loads = sp_load.distribute_to_mesh(self.model.mesh_nodes)
                for dl in dist_loads:
                    nid = dl.node_id
                    if nid < self.n_nodes:
                        fv = dl.force_vector
                        for i in range(6):
                            F_full[nid, i] += fv[i]

        # 3. Applied acceleration field (gravity / inertia)
        target_acc: Optional[AccelerationField] = None
        if acceleration is not None:
            if isinstance(acceleration, (list, tuple, np.ndarray)):
                target_acc = AccelerationField(ax=float(acceleration[0]), ay=float(acceleration[1]), az=float(acceleration[2]))
            else:
                target_acc = acceleration
        elif hasattr(self.model, "applied_accelerations") and self.model.applied_accelerations:
            for acc_f in self.model.applied_accelerations:
                F_body, _ = acc_f.apply_to_model(self.model)
                F_full += F_body
        elif getattr(self.model, "acceleration", None) is not None:
            acc = self.model.acceleration
            target_acc = AccelerationField(ax=float(acc[0]), ay=float(acc[1]), az=float(acc[2]))

        if target_acc is not None:
            F_body, _ = target_acc.apply_to_model(self.model)
            F_full += F_body

        F_active = self.dof_mgr.condense_forces(F_full.ravel())
        if self.apply_bcs and len(self.fixed_dofs) > 0:
            F_active[self.fixed_dofs] = 0.0

        return F_active.astype(self.dtype)

    def solve_pcg(
        self,
        F: Optional[np.ndarray] = None,
        tol: float = 1e-6,
        maxiter: int = 1000,
        x0: Optional[np.ndarray] = None,
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Solve the heterogeneous coupled system via high-performance PCG.

        Returns:
            U_full: Full nodal displacement vector (n_nodes * 6,).
            info: Dictionary with convergence status, iterations, residual history.
        """
        start_time = time.time()
        b = self.assemble_rhs() if F is None else np.copy(F).astype(self.dtype)
        if self.apply_bcs and len(self.fixed_dofs) > 0:
            b[self.fixed_dofs] = 0.0

        b_norm = np.linalg.norm(b)
        if b_norm == 0.0:
            u_full = np.zeros(self.n_nodes * 6, dtype=self.dtype)
            return u_full, {"converged": True, "iterations": 0, "residual": 0.0, "time": 0.0}

        u = np.zeros(self.n_dofs, dtype=self.dtype) if x0 is None else np.copy(x0).astype(self.dtype)
        if self.apply_bcs and len(self.fixed_dofs) > 0:
            u[self.fixed_dofs] = 0.0

        r = b - self._matvec(u)
        if self.apply_bcs and len(self.fixed_dofs) > 0:
            r[self.fixed_dofs] = 0.0

        z = r * self.inv_diag_K
        if self.apply_bcs and len(self.fixed_dofs) > 0:
            z[self.fixed_dofs] = 0.0

        p = np.copy(z)
        rz_old = np.dot(r, z)

        res_history = [np.linalg.norm(r) / b_norm]
        converged = False
        k = 0

        for k in range(1, maxiter + 1):
            Ap = self._matvec(p)
            if self.apply_bcs and len(self.fixed_dofs) > 0:
                Ap[self.fixed_dofs] = 0.0

            pAp = np.dot(p, Ap)
            if abs(pAp) < 1e-30:
                break

            alpha = rz_old / pAp
            u += alpha * p
            r -= alpha * Ap
            if self.apply_bcs and len(self.fixed_dofs) > 0:
                r[self.fixed_dofs] = 0.0

            rel_res = np.linalg.norm(r) / b_norm
            res_history.append(rel_res)

            if callback is not None:
                callback(k, rel_res)

            if rel_res < tol:
                converged = True
                break

            z = r * self.inv_diag_K
            if self.apply_bcs and len(self.fixed_dofs) > 0:
                z[self.fixed_dofs] = 0.0

            rz_new = np.dot(r, z)
            if abs(rz_old) < 1e-30:
                break

            beta = rz_new / rz_old
            p = z + beta * p
            rz_old = rz_new

        elapsed = time.time() - start_time
        u_full = self.dof_mgr.expand_displacements(u)
        self.model.displacements = u_full

        info = {
            "converged": converged,
            "iterations": k,
            "final_residual": res_history[-1] if res_history else 0.0,
            "residual_history": res_history,
            "elapsed_seconds": elapsed,
        }
        return u_full, info


def solve_heterogeneous(
    model: FEAModel,
    tol: float = 1e-6,
    maxiter: int = 1000,
    device: str = "auto",
    precision: str = "fp64",
) -> np.ndarray:
    """
    High-level drop-in solver for heterogeneous beam-solid models.
    Solves via matrix-free GPU solids + multi-threaded CPU beams and stores
    results in model.displacements.
    """
    op = HeterogeneousOperator(model, apply_bcs=True, device=device, precision=precision)
    u_full, info = op.solve_pcg(tol=tol, maxiter=maxiter)
    return u_full


# Alias for backward compatibility
HeterogeneousAssembler = HeterogeneousOperator

