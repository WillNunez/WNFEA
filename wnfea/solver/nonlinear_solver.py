"""
Non-Linear FEA Solver with Matrix-Free JFNK Inner Loop and AMG Preconditioning.

Solves non-linear geometric beam problems through:
1. Incremental load stepping (lambda in [0, 1]).
2. Newton-Raphson equilibrium iterations with Armijo backtracking line search.
3. Inner Jacobian-Free Newton-Krylov (JFNK) Preconditioned Conjugate Gradient (PCG).
4. Block-AMG preconditioning built on a compact base representation to preserve
   a minimal VRAM/memory footprint.

Mixed-Precision Iterative Refinement:
    When use_mixed_precision=True, the inner PCG solve and AMG V-cycle run in
    FP32 while the outer Newton residual evaluation, convergence checks, line
    search, and solution accumulation remain in FP64. This halves the memory
    footprint of the inner solver while maintaining full double-precision
    accuracy through the iterative refinement mechanism:
        r = R(U)           (FP64 outer residual)
        delta_U = PCG(r)   (FP32 inner solve)
        U += alpha * dU    (FP64 accumulation)
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
from .mixed_precision import (
    PrecisionConfig,
    MIXED_FP32,
    TRI_PRECISION,
    FULL_FP64,
    cast_to_inner,
    cast_to_outer,
)


class NonLinearConvergenceError(Exception):
    """Raised when the non-linear solver fails to converge."""
    pass


def pcg_solve(
    A_op,
    b: np.ndarray,
    M_prec=None,
    tol: float = 1e-6,
    max_iter: int = 150,
    working_dtype: np.dtype = np.float64,
    stats: dict | None = None,
) -> tuple[np.ndarray, int, float]:
    """
    Preconditioned Conjugate Gradient (PCG) solver for matrix-free linear operators.

    Supports mixed-precision (FP32 working vectors) and tri-precision (adaptive
    FP16 preconditioner V-cycles for early iterations, switching to FP32 for
    polishing when residual is low).

    Args:
        A_op: LinearOperator implementing matvec w = A * v (e.g. MatrixFreeJFNKOperator).
        b: Right-hand-side vector (shape (n,)).
        M_prec: Preconditioner operator implementing matvec z = M^{-1} * r (e.g. AMG).
        tol: Relative residual convergence tolerance.
        max_iter: Maximum iterations.
        working_dtype: Precision for internal PCG vectors (np.float32 for mixed precision).
        stats: Optional dictionary to record iteration breakdown (fp16_iters, fp32_iters).

    Returns:
        x: Solution vector (float64).
        iters: Number of PCG iterations executed.
        res_norm: Final Euclidean residual norm.
    """
    wd = np.dtype(working_dtype)

    fp16_iters = 0
    fp32_iters = 0

    # Cast RHS and initialize working vectors in working_dtype
    b_work = np.asarray(b, dtype=wd)
    x = np.zeros_like(b_work)
    r = b_work.copy()

    norm_b = float(np.linalg.norm(r))
    if norm_b < 1e-15:
        if stats is not None:
            stats["fp16_iters"] = 0
            stats["fp32_iters"] = 0
        return x.astype(np.float64), 0, 0.0

    if M_prec is not None:
        if hasattr(M_prec, "apply_adaptive"):
            z = np.asarray(M_prec.apply_adaptive(r.astype(np.float64), current_res_rel=1.0), dtype=wd)
            if getattr(M_prec, "last_precision_used", "") == "float16":
                fp16_iters += 1
            else:
                fp32_iters += 1
        else:
            z = np.asarray(M_prec.matvec(r.astype(np.float64)), dtype=wd)
            fp32_iters += 1
    else:
        z = r.copy()

    p = z.copy()
    rz_old = float(np.dot(r, z))

    for it in range(max_iter):
        # A_op.matvec returns float64 from LinearOperator contract; cast to working
        Ap = np.asarray(A_op.matvec(p.astype(np.float64)), dtype=wd)
        pAp = float(np.dot(p, Ap))

        if abs(pAp) < 1e-20:
            # Breakdown or null direction
            break

        alpha = rz_old / pAp
        x += alpha * p
        r -= alpha * Ap

        res_norm = float(np.linalg.norm(r))
        rel_res = res_norm / (norm_b + 1e-30)
        if rel_res < tol:
            if stats is not None:
                stats["fp16_iters"] = fp16_iters
                stats["fp32_iters"] = fp32_iters
            return x.astype(np.float64), it + 1, res_norm

        if M_prec is not None:
            if hasattr(M_prec, "apply_adaptive"):
                z = np.asarray(M_prec.apply_adaptive(r.astype(np.float64), current_res_rel=rel_res), dtype=wd)
                if getattr(M_prec, "last_precision_used", "") == "float16":
                    fp16_iters += 1
                else:
                    fp32_iters += 1
            else:
                z = np.asarray(M_prec.matvec(r.astype(np.float64)), dtype=wd)
                fp32_iters += 1
        else:
            z = r.copy()

        rz_new = float(np.dot(r, z))
        beta = rz_new / (rz_old + 1e-30)
        p = z + beta * p
        rz_old = rz_new

    if stats is not None:
        stats["fp16_iters"] = fp16_iters
        stats["fp32_iters"] = fp32_iters
    return x.astype(np.float64), max_iter, float(np.linalg.norm(r))


def solve_nonlinear_jfnk(
    model: FEAModel,
    n_load_steps: int = 5,
    max_newton_iter: int = 25,
    tol_rel: float = 1e-5,
    tol_abs: float = 1e-5,
    use_amg: bool = True,
    use_mixed_precision: bool = False,
    use_tri_precision: bool = False,
    switch_tol: float = 1e-2,
    verbose: bool = True,
) -> np.ndarray:
    """
    Solve the geometrically non-linear FEA model using matrix-free JFNK + AMG.

    Supports three precision modes:
    1. Uniform FP64 (default): Pure double precision throughout.
    2. Mixed Precision (use_mixed_precision=True): FP32 inner solve (PCG/JFNK/AMG),
       FP64 outer residual and solution accumulation.
    3. Tri-Precision (use_tri_precision=True): FP16 early AMG V-cycles when
       residual is high, switching automatically to FP32 as residual converges,
       with FP64 outer equilibrium residual.

    Args:
        model: FEAModel with geometry, properties, mesh, supports, and loads defined.
        n_load_steps: Number of incremental load steps between 0 and 1.
        max_newton_iter: Maximum Newton iterations per load step.
        tol_rel: Relative residual tolerance for Newton convergence.
        tol_abs: Absolute residual tolerance for Newton convergence.
        use_amg: Whether to use Block Beam AMG preconditioning for the inner PCG loop.
        use_mixed_precision: Whether to use FP32 inner solve with FP64 outer residual.
        use_tri_precision: Whether to enable 3rd level casting (adaptive FP16 preconditioner).
        switch_tol: Relative residual threshold to switch preconditioner from FP16 to FP32.
        verbose: Print progress summary.

    Returns:
        U: Final converged global displacement vector (N*6,).
    """
    errors = model.validate_for_solving()
    if errors:
        raise ValueError("Cannot solve model:\n  " + "\n  ".join(errors))

    from .dof_manager import DOFManager
    dof_mgr = DOFManager(model)

    n_nodes = len(model.mesh_nodes)
    n_dofs = dof_mgr.total_active_dofs

    # Configure precision
    if use_tri_precision:
        use_mixed_precision = True
        prec_config = TRI_PRECISION
        prec_config.switch_tol = float(switch_tol)
        inner_dtype = np.float32
    elif use_mixed_precision:
        prec_config = MIXED_FP32
        inner_dtype = np.float32
    else:
        prec_config = FULL_FP64
        inner_dtype = np.float64

    # Initial state in active DOF space (always FP64 for outer accumulation)
    U = np.zeros(n_dofs, dtype=np.float64)
    F_ext = build_external_force_vector(model, dof_mgr)
    constrained_dofs, prescribed_vals = get_boundary_constraints(model, dof_mgr)

    if verbose:
        print("=" * 65)
        print("  WNFEA Matrix-Free JFNK Non-Linear Solver (AMG-Preconditioned)")
        print("=" * 65)
        n_elems = len(model.mesh_elements) if model.mesh_elements is not None else 0
        if getattr(model, "solid_elements", None) is not None:
            n_elems += len(model.solid_elements)
        print(f"  Active DOFs: {n_dofs} ({n_nodes} nodes, {n_elems} elements)")
        print(f"  Load steps: {n_load_steps}, Max Newton iters: {max_newton_iter}")
        print(f"  Preconditioner: {'Block Beam AMG' if use_amg else 'None (Diagonal)'}")
        print(f"  Precision: {prec_config.summary()}")
        print("-" * 65)

    # Build AMG Preconditioner once (lagged base representation)
    amg_prec = None
    if use_amg:
        t_amg0 = time.perf_counter()
        amg_prec = BlockBeamAMGPreconditioner(
            model, working_dtype=inner_dtype
        )
        if use_tri_precision:
            amg_prec.enable_tri_precision(switch_tol=switch_tol)

        if verbose:
            print(f"  AMG hierarchy constructed in {time.perf_counter() - t_amg0:.3f} s "
                  f"({len(amg_prec.levels)} levels, coarsest={amg_prec.levels[-1].A.shape[0]} DOFs)")
            if use_tri_precision:
                print(f"  AMG hierarchy configured for adaptive FP16 -> FP32 (switch at rel_res={switch_tol:.1e})")
            elif use_mixed_precision:
                print(f"  AMG hierarchy stored in {np.dtype(inner_dtype).name}")

    start_time = time.perf_counter()
    total_pcg_iters = 0

    load_steps = np.linspace(1.0 / n_load_steps, 1.0, n_load_steps)

    for step_idx, lam in enumerate(load_steps, 1):
        if verbose:
            print(f"\n[Load Step {step_idx}/{n_load_steps}] Load factor lambda = {lam:.3f}")

        # Compute initial residual for this load step (ALWAYS FP64)
        R = compute_equilibrium_residual(
            model, U, F_ext, load_factor=lam,
            constrained_dofs=constrained_dofs, prescribed_vals=prescribed_vals,
            dof_mgr=dof_mgr
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
            # In mixed precision, JFNK stores U and R_current in inner_dtype
            J_op = MatrixFreeJFNKOperator(
                model, U, F_ext, load_factor=lam, R_current=R, dof_mgr=dof_mgr,
                working_dtype=inner_dtype,
            )

            # Solve J(U) * delta_U = -R via PCG (inner solve in inner_dtype)
            rhs = -R
            pcg_stats = {}
            delta_U, pcg_its, pcg_res = pcg_solve(
                J_op, rhs, M_prec=amg_prec, tol=inner_tol_rel, max_iter=100,
                working_dtype=inner_dtype, stats=pcg_stats,
            )
            total_pcg_iters += pcg_its
            prev_res_norm = res_norm

            # delta_U is now FP64 (cast back inside pcg_solve)
            # Backtracking Armijo Line Search (FP64 outer residual)
            alpha = 1.0
            U_trial = U + alpha * delta_U
            R_trial = compute_equilibrium_residual(
                model, U_trial, F_ext, load_factor=lam,
                constrained_dofs=constrained_dofs, prescribed_vals=prescribed_vals,
                dof_mgr=dof_mgr
            )
            trial_norm = float(np.linalg.norm(R_trial))

            line_search_steps = 0
            while trial_norm >= res_norm and alpha > 0.0625:
                alpha *= 0.5
                line_search_steps += 1
                U_trial = U + alpha * delta_U
                R_trial = compute_equilibrium_residual(
                    model, U_trial, F_ext, load_factor=lam,
                    constrained_dofs=constrained_dofs, prescribed_vals=prescribed_vals,
                    dof_mgr=dof_mgr
                )
                trial_norm = float(np.linalg.norm(R_trial))

            U = U_trial
            R = R_trial

            if verbose:
                if use_tri_precision:
                    fp16_i = pcg_stats.get("fp16_iters", 0)
                    fp32_i = pcg_stats.get("fp32_iters", 0)
                    prec_str = f" ({fp16_i} fp16, {fp32_i} fp32)"
                else:
                    prec_str = ""
                print(f"    Newton {newton_it:2d}: ||R|| = {trial_norm:.4e} "
                      f"(rel: {trial_norm / (res_norm0 + 1e-20):.2e}) | "
                      f"PCG iters: {pcg_its:2d}{prec_str} | step: {alpha:.3f}")
        else:
            raise NonLinearConvergenceError(
                f"Newton solver failed to converge at load step {step_idx} (lambda={lam:.3f}). "
                f"Final residual: {float(np.linalg.norm(R)):.4e}"
            )

    elapsed = time.perf_counter() - start_time
    # Expand active displacement solution back to full nodal DOFs
    U_full = dof_mgr.expand_displacements(U)
    model.displacements = U_full

    if verbose:
        print("\n" + "=" * 65)
        print("  NON-LINEAR SOLVE CONVERGED SUCCESSFULLY")
        print(f"  Total solve time: {elapsed:.3f} s")
        print(f"  Total inner PCG iterations: {total_pcg_iters}")
        if use_tri_precision and amg_prec is not None:
            print(f"    - FP16 preconditioner V-cycles: {amg_prec.fp16_vcycle_count}")
            print(f"    - FP32 preconditioner V-cycles: {amg_prec.fp32_vcycle_count}")
        print(f"  Max displacement: {float(np.max(np.abs(U_full))):.6e} m")
        if use_tri_precision:
            print(f"  Precision: Tri-Precision (FP64 residual, FP32 Krylov, adaptive FP16->FP32 AMG)")
        elif use_mixed_precision:
            print(f"  Inner precision: {np.dtype(inner_dtype).name} | Outer precision: float64")
        print("=" * 65)

    if model.mesh_elements is not None and len(model.mesh_elements) > 0:
        model.element_results = compute_element_stresses(model)

    return U_full
