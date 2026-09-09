"""
Successive Refinement 1% Decay Boundary Engine for WNFEA.
----------------------------------------------------------
Implements the core tenet requested for accelerating large structural models:
Tracks spatial field perturbations between successive mesh refinement iterations:
    delta(r) = ||u^(k+1)(r) - u^(k)(r)|| / ||u^(k)(r)|| <= 0.01 (1%)
and dynamically sizes the minimal bounding sphere R_1% around localized stress
hotspots where field variables change by less than 1%.

Elements outside this 1% boundary sphere are excluded from subsequent refined
re-solves, saving massive wallclock time while preserving high-fidelity accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union, Tuple
import numpy as np
from scipy.spatial import cKDTree


@dataclass
class DecayBoundaryResult:
    """
    Telemetry and geometric parameters of the 1% successive refinement decay boundary.
    """
    radius: float                       # Minimal radius R_1% where delta(r) <= threshold
    center: np.ndarray                  # (3,) Hotspot center coordinate
    threshold: float                    # Perturbation tolerance (default: 0.01 = 1%)
    radii: np.ndarray                   # Evaluated radial shell boundaries
    relative_deltas: np.ndarray         # Relative perturbation delta(r) per shell
    converged: bool                     # True if perturbation dropped below threshold
    nominal_perturbation: float         # Peak perturbation at hotspot center (r -> 0)
    safety_factor: float = 1.0          # Applied margin multiplier to raw R_1%


def compute_successive_refinement_decay_radius(
    hotspot_center: Union[np.ndarray, Sequence[float]],
    coords_coarse: np.ndarray,
    disp_coarse: np.ndarray,
    coords_refined: np.ndarray,
    disp_refined: np.ndarray,
    threshold: float = 0.01,
    num_shells: int = 25,
    min_radius: float = 0.005,
    max_radius: Optional[float] = None,
    safety_factor: float = 1.10,
) -> DecayBoundaryResult:
    """
    Determine the Saint-Venant decay boundary R_1% where the relative difference
    between successive mesh refinements (pass k and pass k+1) drops below threshold.

    Parameters:
        hotspot_center: (3,) center of localized stress concentration.
        coords_coarse: (N_c, 3) nodal coordinates of coarse mesh (pass k).
        disp_coarse: (N_c, 3) or (3*N_c,) displacement vector of pass k.
        coords_refined: (N_r, 3) nodal coordinates of refined mesh (pass k+1).
        disp_refined: (N_r, 3) or (3*N_r,) displacement vector of pass k+1.
        threshold: Relative perturbation cutoff (default: 0.01 for 1%).
        num_shells: Number of concentric spherical evaluation shells.
        min_radius: Minimum allowable radius in meters.
        max_radius: Maximum search radius. If None, derived from domain bounding box.
        safety_factor: Safety multiplier applied to R_1% (default: 1.10 = 10% margin).

    Returns:
        DecayBoundaryResult containing the minimal bounding sphere radius and radial delta curve.
    """
    center = np.asarray(hotspot_center, dtype=np.float64).reshape(3)
    coords_c = np.asarray(coords_coarse, dtype=np.float64)
    coords_r = np.asarray(coords_refined, dtype=np.float64)

    # Format displacement vectors to (N, 3)
    u_c = np.asarray(disp_coarse, dtype=np.float64)
    if u_c.ndim == 1:
        u_c = u_c.reshape(-1, 3)

    u_r = np.asarray(disp_refined, dtype=np.float64)
    if u_r.ndim == 1:
        u_r = u_r.reshape(-1, 3)

    # Distance of refined nodes from hotspot center
    r_dist = np.linalg.norm(coords_r - center, axis=1)

    if max_radius is None:
        max_radius = float(np.max(r_dist))
    else:
        max_radius = float(max_radius)

    min_radius = float(min_radius)
    if max_radius <= min_radius:
        max_radius = min_radius * 2.0

    # Build k-d tree for fast spatial interpolation of coarse displacements at refined nodes
    tree_c = cKDTree(coords_c)
    # Query k=4 nearest coarse nodes for inverse distance weighting (IDW)
    k_neighbors = min(4, len(coords_c))
    dists, indices = tree_c.query(coords_r, k=k_neighbors)

    if k_neighbors == 1:
        u_c_interp = u_c[indices]
    else:
        # Inverse Distance Weighting interpolation
        eps = 1e-12
        weights = 1.0 / np.maximum(dists, eps)
        weights_sum = np.sum(weights, axis=1, keepdims=True)
        norm_weights = weights / weights_sum  # (N_r, k)
        u_c_interp = np.sum(norm_weights[:, :, None] * u_c[indices], axis=1)  # (N_r, 3)

    # Discrete radial shells starting from 0.0 to capture hotspot center
    shell_edges = np.linspace(0.0, max_radius, num_shells + 1)
    shell_radii = 0.5 * (shell_edges[:-1] + shell_edges[1:])
    relative_deltas = np.zeros(num_shells, dtype=np.float64)

    eps_reg = 1e-14
    for j in range(num_shells):
        r_low = shell_edges[j]
        r_high = shell_edges[j + 1]

        mask = (r_dist >= r_low) & (r_dist < r_high)
        if not np.any(mask):
            if j == 0 and len(coords_r) > 0:
                # Fallback to the nearest node to center
                nn_idx = int(np.argmin(r_dist))
                delta_nn = u_r[nn_idx] - u_c_interp[nn_idx]
                norm_nn = np.sqrt(np.sum(delta_nn ** 2))
                base_nn = np.sqrt(np.sum(u_r[nn_idx] ** 2)) + eps_reg
                relative_deltas[j] = norm_nn / base_nn
            else:
                relative_deltas[j] = relative_deltas[j - 1] if j > 0 else 0.0
            continue

        # L2 norm of displacement perturbation in this shell
        delta_u = u_r[mask] - u_c_interp[mask]
        norm_delta = np.sqrt(np.sum(delta_u ** 2))
        norm_base = np.sqrt(np.sum(u_r[mask] ** 2)) + eps_reg

        relative_deltas[j] = norm_delta / norm_base

    nominal_perturbation = float(np.max(relative_deltas[:max(1, num_shells // 3)]))

    # Find the smallest radius R where all shells beyond R have delta <= threshold
    converged = False
    chosen_radius = max_radius

    for j in range(num_shells):
        # Check if all subsequent shells remain below or at threshold
        if np.all(relative_deltas[j:] <= threshold):
            chosen_radius = float(shell_edges[j])
            converged = True
            break

    # If no shell had delta <= threshold, fall back to the shell with the minimum delta
    if not converged:
        min_idx = int(np.argmin(relative_deltas))
        chosen_radius = float(shell_edges[min_idx])

    # Apply safety margin and ensure >= min_radius
    bounded_radius = max(min_radius, chosen_radius * float(safety_factor))

    return DecayBoundaryResult(
        radius=float(bounded_radius),
        center=center,
        threshold=float(threshold),
        radii=shell_radii,
        relative_deltas=relative_deltas,
        converged=converged,
        nominal_perturbation=nominal_perturbation,
        safety_factor=float(safety_factor),
    )


def compute_stress_decay_radius_from_field(
    hotspot_center: Union[np.ndarray, Sequence[float]],
    elem_centroids: np.ndarray,
    elem_von_mises: np.ndarray,
    threshold: float = 0.01,
    num_shells: int = 25,
    min_radius: float = 0.005,
    max_radius: Optional[float] = None,
    safety_factor: float = 1.10,
) -> DecayBoundaryResult:
    """
    Determine the Saint-Venant decay radius from a single stress field
    where the radial stress gradient relative to the peak stress drops below 1%.

    Parameters:
        hotspot_center: (3,) location of peak stress.
        elem_centroids: (M, 3) element centroid coordinates.
        elem_von_mises: (M,) scalar von Mises stresses.
        threshold: Relative change tolerance (default: 0.01).
        num_shells: Number of radial evaluation shells.
        min_radius: Minimum allowable radius.
        max_radius: Maximum search radius.
        safety_factor: Safety factor multiplier.

    Returns:
        DecayBoundaryResult with decay radius and radial stress curve.
    """
    center = np.asarray(hotspot_center, dtype=np.float64).reshape(3)
    cents = np.asarray(elem_centroids, dtype=np.float64)
    stresses = np.asarray(elem_von_mises, dtype=np.float64)

    r_dist = np.linalg.norm(cents - center, axis=1)
    if max_radius is None:
        max_radius = float(np.max(r_dist))
    else:
        max_radius = float(max_radius)

    min_radius = float(min_radius)
    if max_radius <= min_radius:
        max_radius = min_radius * 2.0

    peak_stress = float(np.max(stresses)) if len(stresses) > 0 else 1.0
    shell_edges = np.linspace(min_radius, max_radius, num_shells + 1)
    shell_radii = 0.5 * (shell_edges[:-1] + shell_edges[1:])
    relative_deltas = np.zeros(num_shells, dtype=np.float64)

    # Average stress per shell
    shell_stresses = np.zeros(num_shells, dtype=np.float64)
    for j in range(num_shells):
        r_low = shell_edges[j]
        r_high = shell_edges[j + 1]
        mask = (r_dist >= r_low) & (r_dist < r_high)
        if np.any(mask):
            shell_stresses[j] = np.mean(stresses[mask])
        else:
            shell_stresses[j] = shell_stresses[j - 1] if j > 0 else peak_stress

    # Relative stress change between adjacent shells: |s_{j+1} - s_j| / peak_stress
    for j in range(num_shells - 1):
        rel_change = abs(shell_stresses[j + 1] - shell_stresses[j]) / max(peak_stress, 1e-12)
        relative_deltas[j] = rel_change
    relative_deltas[-1] = relative_deltas[-2] if num_shells > 1 else 0.0

    converged = False
    chosen_radius = max_radius
    for j in range(num_shells):
        if np.all(relative_deltas[j:] <= threshold):
            chosen_radius = float(shell_edges[j])
            converged = True
            break

    if not converged:
        min_idx = int(np.argmin(relative_deltas))
        chosen_radius = float(shell_edges[min_idx])

    bounded_radius = max(min_radius, chosen_radius * float(safety_factor))

    return DecayBoundaryResult(
        radius=float(bounded_radius),
        center=center,
        threshold=float(threshold),
        radii=shell_radii,
        relative_deltas=relative_deltas,
        converged=converged,
        nominal_perturbation=float(relative_deltas[0]) if num_shells > 0 else 0.0,
        safety_factor=float(safety_factor),
    )
