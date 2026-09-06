"""
Jacobian-Free Newton-Krylov (JFNK) Matrix-Free Operator for WNFEA.

Evaluates the directional derivative (Fréchet derivative) of the equilibrium
residual:
    J(U) * v ≈ [R(U + ε v) - R(U)] / ε
without ever forming or storing the global tangent stiffness matrix.

Mixed-Precision Support:
    When working_dtype=np.float32, the perturbation arithmetic and finite-difference
    evaluation run in FP32. The perturbation scale ε is adapted to sqrt(eps_mach)
    of the working precision. This reduces memory for the stored base state and
    gives FP32-accurate Jacobian-vector products suitable for use as an inner
    solver direction in iterative refinement.
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

    Mixed-Precision:
        Set working_dtype=np.float32 to run all finite-difference arithmetic
        in FP32. The eps_scale is automatically adapted to sqrt(eps_mach) of
        the working dtype.
    """

    def __init__(
        self,
        model: FEAModel,
        U: np.ndarray,
        F_ext: np.ndarray,
        load_factor: float = 1.0,
        R_current: np.ndarray | None = None,
        dof_mgr: DOFManager | None = None,
        eps_scale: float | None = None,
        working_dtype: np.dtype = np.float64,
    ):
        from .dof_manager import DOFManager
        self.model = model
        self.dof_mgr = dof_mgr if dof_mgr is not None else DOFManager(model)
        self.working_dtype = np.dtype(working_dtype)

        # Auto-select perturbation scale based on working precision
        if eps_scale is None:
            self.eps_scale = float(np.sqrt(np.finfo(self.working_dtype).eps))
        else:
            self.eps_scale = float(eps_scale)

        self.load_factor = float(load_factor)

        # Store base state in working_dtype for inner matvec
        self.U = np.asarray(U, dtype=self.working_dtype)
        self.F_ext = np.asarray(F_ext, dtype=self.working_dtype)

        self.constrained_dofs, self.prescribed_vals = get_boundary_constraints(model, self.dof_mgr)
        self.is_constrained = np.zeros(len(self.U), dtype=bool)
        if len(self.constrained_dofs) > 0:
            self.is_constrained[self.constrained_dofs] = True

        if R_current is not None:
            self.R_current = np.asarray(R_current, dtype=self.working_dtype)
        else:
            self.R_current = compute_equilibrium_residual(
                self.model, self.U.astype(np.float64), self.F_ext.astype(np.float64),
                self.load_factor,
                self.constrained_dofs, self.prescribed_vals,
                dof_mgr=self.dof_mgr
            ).astype(self.working_dtype)

        n = len(self.U)
        super().__init__(dtype=np.float64, shape=(n, n))

    def update_state(self, U: np.ndarray, R_current: np.ndarray | None = None) -> None:
        """Update the base state U around which the Jacobian is linearized."""
        self.U = np.asarray(U, dtype=self.working_dtype)
        if R_current is not None:
            self.R_current = np.asarray(R_current, dtype=self.working_dtype)
        else:
            self.R_current = compute_equilibrium_residual(
                self.model, self.U.astype(np.float64), self.F_ext.astype(np.float64),
                self.load_factor,
                self.constrained_dofs, self.prescribed_vals,
                dof_mgr=self.dof_mgr
            ).astype(self.working_dtype)

    def _matvec(self, v: np.ndarray) -> np.ndarray:
        """
        Matrix-vector multiplication w = J(U) * v via Fréchet finite differencing.

        The perturbation arithmetic runs in working_dtype. The residual evaluations
        call compute_equilibrium_residual in FP64 internally (element kernels always
        compute in FP64 for robustness), but we cast inputs/outputs to working_dtype
        for the finite difference.
        """
        v = np.asarray(v, dtype=self.working_dtype)
        v_norm = float(np.linalg.norm(v))
        if v_norm < 1e-14:
            return np.zeros(len(v), dtype=np.float64)

        u_norm = float(np.linalg.norm(self.U))
        eps = self.eps_scale * (1.0 + u_norm) / v_norm

        # Zero out perturbation at constrained DOFs to prevent artificial boundary drift
        v_pert = v.copy()
        v_pert[self.is_constrained] = 0.0

        U_pert = self.U + eps * v_pert

        # Evaluate perturbed residual: upcast to FP64 for element kernels, downcast result
        R_pert = compute_equilibrium_residual(
            self.model, U_pert.astype(np.float64), self.F_ext.astype(np.float64),
            self.load_factor,
            self.constrained_dofs, self.prescribed_vals,
            dof_mgr=self.dof_mgr
        ).astype(self.working_dtype)

        w = (R_pert - self.R_current) / eps

        # For constrained DOFs, set equation to identity (1.0 * v_i)
        w[self.is_constrained] = v[self.is_constrained]

        return w.astype(np.float64)
