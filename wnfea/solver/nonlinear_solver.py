"""
Non-Linear FEA Solver with Matrix-Free JFNK Inner Loop and AMG Preconditioning.

Solves non-linear geometric beam problems through:
1. Incremental load stepping (lambda in [0, 1]).
2. Newton-Raphson equilibrium iterations with Armijo backtracking line search.
3. Inner Jacobian-Free Newton-Krylov (JFNK) Preconditioned Conjugate Gradient (PCG).
4. Block-AMG preconditioning built on a compact base representation to preserve
   a minimal VRAM/memory footprint.
"""

from __future__ import annotations

import time
import numpy as np

from ..model import FEAModel
from .residual import (
    build_external_force_vector,
    compute_equilibrium_residual,
    get_boundary_constraints,
)
from .jfnk_operator import MatrixFreeJFNKOperator
from .amg_preconditioner import BlockBeamAMGPreconditioner
from .stress import compute_element_stresses


class NonLinearConvergenceError(Exception):
    """Raised when the non-linear solver fails to converge."""
    pass


def pcg_solve(
    A_op,
    b: np.ndarray,
    M_prec=None,
    tol: float = 1e-6,
    max_iter: int = 150,
) -> tuple[np.ndarray, int, float]:
    """
    Preconditioned Conjugate Gradient (PCG) solver for matrix-free linear operators.

    Args:
        A_op: LinearOperator implementing matvec w = A * v (e.g. MatrixFreeJFNKOperator).
        b: Right-hand-side vector (shape (n,)).
        M_prec: Preconditioner operator implementing matvec z = M^{-1} * r (e.g. AMG).
        tol: Relative residual convergence tolerance.
        max_iter: Maximum iterations.

    Returns:
        x: Solution vector.
        iters: Number of PCG iterations executed.
        res_norm: Final Euclidean residual norm.
    """
    b = np.asarray(b, dtype=np.float64)
    x = np.zeros_like(b)
    r = b.copy()

    norm_b = float(np.linalg.norm(b))
    if norm_b < 1e-15:
        return x, 0, 0.0

    if M_prec is not None:
        z = M_prec.matvec(r)
    else:
        z = r.copy()

    p = z.copy()
    rz_old = float(np.dot(r, z))

    for it in range(max_iter):
        Ap = A_op.matvec(p)
        pAp = float(np.dot(p, Ap))

        if abs(pAp) < 1e-20:
            # Breakdown or null direction
            break

        alpha = rz_old / pAp
        x += alpha * p
        r -= alpha * Ap

        res_norm = float(np.linalg.norm(r))
        if res_norm / norm_b < tol:
            return x, it + 1, res_norm

        if M_prec is not None:
            z = M_prec.matvec(r)
        else:
            z = r.copy()

        rz_new = float(np.dot(r, z))
        beta = rz_new / (rz_old + 1e-30)
        p = z + beta * p
        rz_old = rz_new

    return x, max_iter, float(np.linalg.norm(r))


def solve_nonlinear_jfnk(
    model: FEAModel,
    n_load_steps: int = 5,
    max_newton_iter: int = 25,
    tol_rel: float = 1e-5,
    tol_abs: float = 1e-5,
    use_amg: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """
    Solve the geometrically non-linear FEA model using matrix-free JFNK + AMG.

    Args:
        model: FEAModel with geometry, properties, mesh, supports, and loads defined.
        n_load_steps: Number of incremental load steps between 0 and 1.
        max_newton_iter: Maximum Newton iterations per load step.
        tol_rel: Relative residual tolerance for Newton convergence.
        tol_abs: Absolute residual tolerance for Newton convergence.
        use_amg: Whether to use Block Beam AMG preconditioning for the inner PCG loop.
        verbose: Print progress summary.

    Returns:
        U: Final converged global displacement vector (N*6,).
    """
    errors = model.validate_for_solving()
    if errors:
        raise ValueError("Cannot solve model:\n  " + "\n  ".join(errors))

    n_nodes = len(model.mesh_nodes)
    n_dofs = n_nodes * 6

    # Initial state
    U = np.zeros(n_dofs, dtype=np.float64)
    F_ext = build_external_force_vector(model)
    constrained_dofs, prescribed_vals = get_boundary_constraints(model)

    if verbose:
        print("=" * 65)
        print("  WNFEA Matrix-Free JFNK Non-Linear Solver (AMG-Preconditioned)")
        print("=" * 65)
        print(f"  DOFs: {n_dofs} ({n_nodes} nodes, {len(model.mesh_elements)} elements)")
        print(f"  Load steps: {n_load_steps}, Max Newton iters: {max_newton_iter}")
        print(f"  Preconditioner: {'Block Beam AMG' if use_amg else 'None (Diagonal)'}")
        print("-" * 65)

    # Build AMG Preconditioner once (lagged base representation)
    amg_prec = None
    if use_amg:
        t_amg0 = time.perf_counter()
        amg_prec = BlockBeamAMGPreconditioner(model)
        if verbose:
            print(f"  AMG hierarchy constructed in {time.perf_counter() - t_amg0:.3f} s "
                  f"({len(amg_prec.levels)} levels, coarsest={amg_prec.levels[-1].A.shape[0]} DOFs)")

    start_time = time.perf_counter()
    total_pcg_iters = 0

    load_steps = np.linspace(1.0 / n_load_steps, 1.0, n_load_steps)

    for step_idx, lam in enumerate(load_steps, 1):
        if verbose:
            print(f"\n[Load Step {step_idx}/{n_load_steps}] Load factor lambda = {lam:.3f}")

        # Compute initial residual for this load step
        R = compute_equilibrium_residual(
            model, U, F_ext, load_factor=lam,
            constrained_dofs=constrained_dofs, prescribed_vals=prescribed_vals
        )
        res_norm0 = float(np.linalg.norm(R))
        prev_res_norm = res_norm0

        if res_norm0 < tol_abs:
            if verbose:
                print(f"  Converged immediately: ||R|| = {res_norm0:.3e}")
            continue

        for newton_it in range(1, max_newton_iter + 1):
            res_norm = float(np.linalg.norm(R))
            rel_res = res_norm / (res_norm0 + 1e-20)

            if res_norm < tol_abs or rel_res < tol_rel:
                if verbose:
                    print(f"  --> Step {step_idx} converged at iter {newton_it - 1}: "
                          f"||R|| = {res_norm:.3e} (rel: {rel_res:.2e})")
                break

            # Adaptive inexact Newton tolerance (Eisenstat-Walker heuristic)
            eta = min(0.1, 0.5 * (res_norm / (prev_res_norm + 1e-20))**2)
            inner_tol = max(eta * res_norm, 1e-10)
            inner_tol_rel = inner_tol / (res_norm + 1e-20)

            # Construct Matrix-Free JFNK Operator around current state U
            J_op = MatrixFreeJFNKOperator(
                model, U, F_ext, load_factor=lam, R_current=R
            )

            # Solve J(U) * delta_U = -R via PCG
            rhs = -R
            delta_U, pcg_its, pcg_res = pcg_solve(
                J_op, rhs, M_prec=amg_prec, tol=inner_tol_rel, max_iter=100
            )
            total_pcg_iters += pcg_its
            prev_res_norm = res_norm

            # Backtracking Armijo Line Search
            alpha = 1.0
            U_trial = U + alpha * delta_U
            R_trial = compute_equilibrium_residual(
                model, U_trial, F_ext, load_factor=lam,
                constrained_dofs=constrained_dofs, prescribed_vals=prescribed_vals
            )
            trial_norm = float(np.linalg.norm(R_trial))

            line_search_steps = 0
            while trial_norm >= res_norm and alpha > 0.0625:
                alpha *= 0.5
                line_search_steps += 1
                U_trial = U + alpha * delta_U
                R_trial = compute_equilibrium_residual(
                    model, U_trial, F_ext, load_factor=lam,
                    constrained_dofs=constrained_dofs, prescribed_vals=prescribed_vals
                )
                trial_norm = float(np.linalg.norm(R_trial))

            U = U_trial
            R = R_trial

            if verbose:
                print(f"    Newton {newton_it:2d}: ||R|| = {trial_norm:.4e} "
                      f"(rel: {trial_norm / (res_norm0 + 1e-20):.2e}) | "
                      f"PCG iters: {pcg_its:2d} | step: {alpha:.3f}")
        else:
            raise NonLinearConvergenceError(
                f"Newton solver failed to converge at load step {step_idx} (lambda={lam:.3f}). "
                f"Final residual: {float(np.linalg.norm(R)):.4e}"
            )

    elapsed = time.perf_counter() - start_time
    if verbose:
        print("\n" + "=" * 65)
        print("  NON-LINEAR SOLVE CONVERGED SUCCESSFULLY")
        print(f"  Total solve time: {elapsed:.3f} s")
        print(f"  Total inner PCG iterations: {total_pcg_iters}")
        print(f"  Max displacement: {float(np.max(np.abs(U))):.6e} m")
        print("=" * 65)

    # Store results in model
    model.displacements = U
    model.element_results = compute_element_stresses(model)

    return U
