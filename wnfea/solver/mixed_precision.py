"""
Mixed-Precision Configuration and Utilities for WNFEA.

Provides precision management for iterative refinement: FP32 inner solve
(PCG + JFNK + AMG V-cycle) with FP64 outer residual evaluation. The outer
residual is always computed in FP64, guaranteeing convergence to full
double-precision accuracy while the computationally expensive inner solve
runs in FP32 to halve VRAM footprint and improve throughput.

Precision hierarchy:
    FP64 (outer): Newton residual R(U), convergence checks, line search,
                  solution accumulation U += alpha * delta_U
    FP32 (inner): PCG working vectors, JFNK Fréchet matvec, AMG V-cycle
                  hierarchy (P, R, A_c, smoother diagonals, coarse solve)
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class PrecisionConfig:
    """
    Configuration for mixed-precision and tri-precision iterative refinement.

    Attributes
    ----------
    inner_dtype : np.dtype
        Precision for inner solve (PCG vectors, JFNK matvec, fine AMG V-cycle).
        Default: np.float32.
    outer_dtype : np.dtype
        Precision for outer residual and solution accumulation.
        Default: np.float64.
    precond_coarse_dtype : np.dtype
        Precision for early/coarse preconditioner V-cycle iterations.
        Default: np.float16.
    switch_tol : float
        Relative residual threshold below which preconditioner switches from
        precond_coarse_dtype (FP16) to inner_dtype (FP32).
        Default: 1e-2.
    use_tri_precision : bool
        Whether to enable the 3rd level of precision (FP16 preconditioner).
        Default: False.
    enabled : bool
        Whether mixed precision is active. If False, everything runs in outer_dtype.
    """
    inner_dtype: np.dtype = np.float32
    outer_dtype: np.dtype = np.float64
    precond_coarse_dtype: np.dtype = np.float16
    switch_tol: float = 1e-2
    use_tri_precision: bool = False
    enabled: bool = True

    @property
    def inner_eps_scale(self) -> float:
        """sqrt(eps_mach) for the inner dtype, used as JFNK perturbation scale."""
        return float(np.sqrt(np.finfo(self.inner_dtype).eps))

    @property
    def outer_eps_scale(self) -> float:
        """sqrt(eps_mach) for the outer dtype."""
        return float(np.sqrt(np.finfo(self.outer_dtype).eps))

    @property
    def working_dtype(self) -> np.dtype:
        """The dtype that inner operations should use."""
        if self.enabled:
            return np.dtype(self.inner_dtype)
        return np.dtype(self.outer_dtype)

    def summary(self) -> str:
        """Human-readable summary of precision configuration."""
        if not self.enabled:
            return f"Uniform {np.dtype(self.outer_dtype).name} (mixed precision disabled)"
        inner_name = np.dtype(self.inner_dtype).name
        outer_name = np.dtype(self.outer_dtype).name
        if self.use_tri_precision:
            coarse_name = np.dtype(self.precond_coarse_dtype).name
            return (
                f"Tri-precision: outer={outer_name} (residual/accumulation), "
                f"inner={inner_name} (PCG/JFNK), "
                f"preconditioner={coarse_name}->{inner_name} (switch at rel_res={self.switch_tol:.1e})"
            )
        return (
            f"Mixed precision: inner={inner_name} (PCG/JFNK/AMG), "
            f"outer={outer_name} (residual/accumulation)"
        )


# Pre-built configurations
MIXED_FP32 = PrecisionConfig(
    inner_dtype=np.float32, outer_dtype=np.float64, enabled=True, use_tri_precision=False
)
TRI_PRECISION = PrecisionConfig(
    inner_dtype=np.float32,
    outer_dtype=np.float64,
    precond_coarse_dtype=np.float16,
    switch_tol=1e-2,
    use_tri_precision=True,
    enabled=True,
)
FULL_FP64 = PrecisionConfig(
    inner_dtype=np.float64, outer_dtype=np.float64, enabled=False, use_tri_precision=False
)


def cast_to_inner(arr: np.ndarray, config: PrecisionConfig) -> np.ndarray:
    """Cast array to inner working precision."""
    target = config.working_dtype
    if arr.dtype == target:
        return arr
    return arr.astype(target, copy=False)


def cast_to_outer(arr: np.ndarray, config: PrecisionConfig) -> np.ndarray:
    """Cast array to outer (accumulation) precision."""
    target = np.dtype(config.outer_dtype)
    if arr.dtype == target:
        return arr
    return arr.astype(target, copy=False)
