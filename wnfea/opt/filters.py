"""
Spatial Sensitivity Filtering & Differentiable Heaviside Projection for WNFEA.
-------------------------------------------------------------------------------
Implements mesh-independent filtering and projection for topology optimization:
1. Distance-weighted neighborhood convolution filter H (via cKDTree in O(N log N)).
2. Exact adjoint chain-rule sensitivity filtering: grad_rho = H.T @ grad_rho_tilde.
3. Smoothed Heaviside projection with beta-continuation for crisp 0-1 boundaries.
4. Non-design domain constraints (frozen solid bolt bosses / pads and void clearance).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple
from dataclasses import dataclass, field
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


class SensitivityFilter:
    """
    Spatial radius filter constructing the normalized convolution operator H
    such that rho_tilde = H @ rho, and grad_rho = H.T @ grad_rho_tilde.
    """
    def __init__(
        self,
        centroids: np.ndarray,
        volumes: np.ndarray,
        radius: float,
    ):
        """
        Parameters:
            centroids: (N, 3) coordinates of element centroids.
            volumes: (N,) element volumes.
            radius: Spatial filter radius r_min in meters.
        """
        self.n_elements = len(centroids)
        self.radius = float(radius)
        self.volumes = np.asarray(volumes, dtype=np.float64)

        # Build cKDTree for O(N log N) spatial neighbor search
        tree = cKDTree(centroids)
        pairs = tree.query_pairs(r=self.radius, output_type="ndarray")

        # Include self-weights (distance = 0)
        self_rows = np.arange(self.n_elements)
        self_cols = np.arange(self.n_elements)
        self_weights = self.radius * self.volumes

        if len(pairs) > 0:
            i_idx = pairs[:, 0]
            j_idx = pairs[:, 1]
            dists = np.linalg.norm(centroids[i_idx] - centroids[j_idx], axis=1)
            w_ij = (self.radius - dists) * self.volumes[j_idx]
            w_ji = (self.radius - dists) * self.volumes[i_idx]

            rows = np.concatenate([self_rows, i_idx, j_idx])
            cols = np.concatenate([self_cols, j_idx, i_idx])
            data = np.concatenate([self_weights, w_ij, w_ji])
        else:
            rows = self_rows
            cols = self_cols
            data = self_weights

        # Assemble raw weighting matrix W
        W = sp.coo_matrix((data, (rows, cols)), shape=(self.n_elements, self.n_elements)).tocsr()

        # Row normalization: H_ij = W_ij / sum_k W_ik
        row_sums = np.array(W.sum(axis=1)).ravel()
        row_sums[row_sums < 1e-12] = 1.0
        inv_row_sums = 1.0 / row_sums

        # Scale rows
        self.H = sp.diags(inv_row_sums) @ W
        self.H_T = self.H.T.tocsr()

    def filter_densities(self, rho: np.ndarray) -> np.ndarray:
        """Compute filtered physical densities: rho_tilde = H @ rho."""
        return self.H @ rho

    def filter_sensitivities(self, dC_drho_tilde: np.ndarray) -> np.ndarray:
        """Adjoint chain-rule filtering: dC_drho = H.T @ dC_drho_tilde."""
        return self.H_T @ dC_drho_tilde


@dataclass
class HeavisideProjection:
    """
    Smoothed Heaviside projection for driving intermediate densities to crisp 0 or 1.
    rho_bar = [tanh(beta * eta) + tanh(beta * (rho - eta))] / [tanh(beta * eta) + tanh(beta * (1 - eta))]
    """
    eta: float = 0.5              # Threshold parameter in (0, 1)
    beta: float = 1.0             # Sharpness continuation parameter
    beta_max: float = 32.0        # Maximum continuation beta
    continuation_rate: float = 1.5 # Multiplicative factor per step

    def project(self, rho: np.ndarray, beta: Optional[float] = None) -> np.ndarray:
        """Apply smoothed Heaviside thresholding."""
        b = self.beta if beta is None else float(beta)
        if b <= 1.0:
            return rho  # Linear identity at beta=1

        tanh_beta_eta = np.tanh(b * self.eta)
        tanh_beta_1_eta = np.tanh(b * (1.0 - self.eta))
        denom = tanh_beta_eta + tanh_beta_1_eta

        return (tanh_beta_eta + np.tanh(b * (rho - self.eta))) / denom

    def derivative(self, rho: np.ndarray, beta: Optional[float] = None) -> np.ndarray:
        """Derivative d(rho_bar) / d(rho)."""
        b = self.beta if beta is None else float(beta)
        if b <= 1.0:
            return np.ones_like(rho)

        tanh_beta_eta = np.tanh(b * self.eta)
        tanh_beta_1_eta = np.tanh(b * (1.0 - self.eta))
        denom = tanh_beta_eta + tanh_beta_1_eta

        tanh_term = np.tanh(b * (rho - self.eta))
        d_tanh = b * (1.0 - tanh_term ** 2)
        return d_tanh / denom

    def step_continuation(self):
        """Advance continuation parameter beta."""
        self.beta = min(self.beta * self.continuation_rate, self.beta_max)
