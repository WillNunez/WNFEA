"""
Result data structures for WNFEA post-processing.

Provides structured containers for per-element and model-level results
so that downstream consumers (GUI, export, reporting) have a clean API.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StressPoint:
    """
    Stress state at a single point on a beam cross-section.

    Attributes
    ----------
    description : str
        Label for the evaluation point (e.g. 'y_plus', 'z_minus').
    sigma_axial : float
        Normal stress from axial force (Pa).
    sigma_bending_y : float
        Normal stress from bending about local y (Pa).
    sigma_bending_z : float
        Normal stress from bending about local z (Pa).
    sigma_total : float
        Total normal stress σ_x (Pa).
    tau_torsion : float
        Shear stress from torsion (Pa).
    von_mises : float
        Von Mises equivalent stress (Pa).
    """
    description: str
    sigma_axial: float = 0.0
    sigma_bending_y: float = 0.0
    sigma_bending_z: float = 0.0
    sigma_total: float = 0.0
    tau_torsion: float = 0.0
    von_mises: float = 0.0


@dataclass
class ElementResult:
    """
    Results for a single beam element.

    Attributes
    ----------
    element_id : int
        Element index in the mesh.
    node1_id : int
        Mesh node index at element start.
    node2_id : int
        Mesh node index at element end.
    local_forces : np.ndarray
        12-component local force/moment vector (N, N·m).
    stress_points : list[StressPoint]
        Stresses at extreme-fiber evaluation points.
    max_von_mises : float
        Maximum von Mises stress across all evaluation points (Pa).
    """
    element_id: int
    node1_id: int = 0
    node2_id: int = 0
    local_forces: np.ndarray = field(default_factory=lambda: np.zeros(12))
    stress_points: list[StressPoint] = field(default_factory=list)
    max_von_mises: float = 0.0


@dataclass
class ModelResults:
    """
    Aggregated results for the entire model.

    Attributes
    ----------
    element_results : dict[int, ElementResult]
        Per-element results keyed by element ID.
    max_displacement : float
        Maximum nodal translational displacement magnitude (m).
    max_displacement_node : int
        Mesh node ID with the maximum displacement.
    max_von_mises : float
        Maximum von Mises stress across all elements (Pa).
    max_von_mises_element : int
        Element ID with the maximum von Mises stress.
    reaction_forces : np.ndarray | None
        Full reaction force vector (N*6,).
    safety_factor : float | None
        Minimum safety factor (yield_strength / max_von_mises), or None
        if max_von_mises is zero.
    """
    element_results: dict[int, ElementResult] = field(default_factory=dict)
    max_displacement: float = 0.0
    max_displacement_node: int = -1
    max_von_mises: float = 0.0
    max_von_mises_element: int = -1
    reaction_forces: np.ndarray | None = None
    safety_factor: float | None = None

    def summary(self) -> str:
        """Return a human-readable results summary."""
        lines = ["=== WNFEA Results Summary ==="]
        lines.append(f"  Elements analysed:     {len(self.element_results)}")
        lines.append(f"  Max displacement:      {self.max_displacement:.6e} m  (node {self.max_displacement_node})")
        lines.append(f"  Max Von Mises stress:  {self.max_von_mises:.3e} Pa  ({self.max_von_mises / 1e6:.3f} MPa)  (elem {self.max_von_mises_element})")
        if self.safety_factor is not None:
            lines.append(f"  Min safety factor:     {self.safety_factor:.3f}")
        else:
            lines.append(f"  Min safety factor:     N/A (zero stress)")
        return "\n".join(lines)
