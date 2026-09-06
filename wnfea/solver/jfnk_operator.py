"""
Jacobian-Free Newton-Krylov (JFNK) Matrix-Free Operator for WNFEA.

Evaluates the directional derivative (Fréchet derivative) of the equilibrium
residual:
    J(U) * v ≈ [R(U + ε v) - R(U)] / ε
without ever forming or storing the global tangent stiffness matrix.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator

from ..model import FEAModel
from .residual import compute_equilibrium_residual, get_boundary_constraints


class MatrixFreeJFNKOperator(LinearOperator):
    """
    LinearOperator representing the action of the tangent Jacobian J(U) on a vector v:
        w = J(U) * v

    Guarantees zero tangent matrix allocation and minimal memory footprint.
    """

    def __init__(
        self,
        model: FEAModel,
        U: np.ndarray,
        F_ext: np.ndarray,
        load_factor: float = 1.0,
        R_current: np.ndarray | None = None,
        dof_mgr: DOFManager | None = None,
        eps_scale: float = 1.4901161193847656e-8,  # sqrt(eps_mach) for float64
    ):
        from .dof_manager import DOFManager
        self.model = model
        self.dof_mgr = dof_mgr if dof_mgr is not None else DOFManager(model)
        self.U = np.asarray(U, dtype=np.float64)
        self.F_ext = np.asarray(F_ext, dtype=np.float64)
        self.load_factor = float(load_factor)
        self.eps_scale = float(eps_scale)

        self.constrained_dofs, self.prescribed_vals = get_boundary_constraints(model, self.dof_mgr)
        self.is_constrained = np.zeros(len(self.U), dtype=bool)
        if len(self.constrained_dofs) > 0:
            self.is_constrained[self.constrained_dofs] = True

        if R_current is not None:
            self.R_current = np.asarray(R_current, dtype=np.float64)
        else:
            self.R_current = compute_equilibrium_residual(
                self.model, self.U, self.F_ext, self.load_factor,
                self.constrained_dofs, self.prescribed_vals,
                dof_mgr=self.dof_mgr
            )

        n = len(self.U)
        super().__init__(dtype=np.float64, shape=(n, n))

    def update_state(self, U: np.ndarray, R_current: np.ndarray | None = None) -> None:
        """Update the base state U around which the Jacobian is linearized."""
        self.U = np.asarray(U, dtype=np.float64)
        if R_current is not None:
            self.R_current = np.asarray(R_current, dtype=np.float64)
        else:
            self.R_current = compute_equilibrium_residual(
                self.model, self.U, self.F_ext, self.load_factor,
                self.constrained_dofs, self.prescribed_vals,
                dof_mgr=self.dof_mgr
            )

    def _matvec(self, v: np.ndarray) -> np.ndarray:
        """
        Matrix-vector multiplication w = J(U) * v via Fréchet finite differencing.
        """
        v = np.asarray(v, dtype=np.float64)
        v_norm = float(np.linalg.norm(v))
        if v_norm < 1e-14:
            return np.zeros_like(v)

        u_norm = float(np.linalg.norm(self.U))
        eps = self.eps_scale * (1.0 + u_norm) / v_norm

        # Zero out perturbation at constrained DOFs to prevent artificial boundary drift
        v_pert = v.copy()
        v_pert[self.is_constrained] = 0.0

        U_pert = self.U + eps * v_pert

        R_pert = compute_equilibrium_residual(
            self.model, U_pert, self.F_ext, self.load_factor,
            self.constrained_dofs, self.prescribed_vals,
            dof_mgr=self.dof_mgr
        )

        w = (R_pert - self.R_current) / eps

        # For constrained DOFs, set equation to identity (1.0 * v_i)
        w[self.is_constrained] = v[self.is_constrained]

        return w
