"""
Result set container for WNFEA.

Provides convenience methods for extracting and formatting analysis results
from the FEAModel after solving.
"""

import numpy as np
from ..model import FEAModel


class ResultSet:
    """
    Convenience wrapper around a solved FEAModel's results.

    Provides methods for extracting nodal displacements, element forces,
    stresses, and computing summary metrics.
    """

    def __init__(self, model: FEAModel):
        if model.displacements is None:
            raise ValueError("Model has no results. Run the solver first.")
        if model.element_results is None:
            raise ValueError("Model has no element results. Run stress post-processing first.")
        self._model = model

    @property
    def n_nodes(self) -> int:
        return len(self._model.mesh_nodes)

    @property
    def n_elements(self) -> int:
        return len(self._model.mesh_elements)

    @property
    def displacements(self) -> np.ndarray:
        """Full DOF displacement vector."""
        return self._model.displacements

    def nodal_displacement(self, node_id: int) -> np.ndarray:
        """Get the 6-DOF displacement for a specific mesh node."""
        return self._model.displacements[node_id * 6 : (node_id + 1) * 6]

    def nodal_translation(self, node_id: int) -> np.ndarray:
        """Get the [ux, uy, uz] translation for a mesh node."""
        return self.nodal_displacement(node_id)[:3]

    def nodal_rotation(self, node_id: int) -> np.ndarray:
        """Get the [rx, ry, rz] rotation for a mesh node."""
        return self.nodal_displacement(node_id)[3:]

    def total_deflection(self, node_id: int) -> float:
        """Euclidean magnitude of the translational displacement at a node."""
        t = self.nodal_translation(node_id)
        return float(np.linalg.norm(t))

    def max_deflection(self) -> tuple[int, float]:
        """Find the node with the largest total deflection. Returns (node_id, deflection)."""
        max_node = 0
        max_val = 0.0
        for i in range(self.n_nodes):
            d = self.total_deflection(i)
            if d > max_val:
                max_val = d
                max_node = i
        return max_node, max_val

    def element_von_mises(self, elem_id: int) -> dict:
        """Get Von Mises stress dict for an element."""
        return self._model.element_results[elem_id]['von_mises']

    def element_forces(self, elem_id: int) -> np.ndarray:
        """Get local internal force vector for an element."""
        return self._model.element_results[elem_id]['local_forces']

    def max_von_mises(self) -> tuple[int, float]:
        """Find the element with the highest Von Mises stress. Returns (elem_id, stress)."""
        max_elem = 0
        max_val = 0.0
        for eid, res in self._model.element_results.items():
            vm = res['von_mises']['max']
            if vm > max_val:
                max_val = vm
                max_elem = eid
        return max_elem, max_val

    def safety_factor(self, material_name: str | None = None) -> float:
        """
        Compute the minimum safety factor across all elements.

        Uses yield_strength / max_von_mises. If material_name is None,
        uses the first material in the model.
        """
        if material_name is None:
            material_name = next(iter(self._model.materials))
        mat = self._model.materials[material_name]

        _, max_vm = self.max_von_mises()
        if max_vm < 1e-6:
            return 999.0
        return mat.yield_strength / max_vm

    def all_element_stresses(self) -> list[float]:
        """Get the max Von Mises stress for each element, in element order."""
        return [
            self._model.element_results[eid]['von_mises']['max']
            for eid in sorted(self._model.element_results.keys())
        ]

    def nodal_displacements_table(self) -> list[dict]:
        """Return a list of dicts suitable for tabular display of nodal displacements."""
        rows = []
        for i in range(self.n_nodes):
            d = self.nodal_displacement(i)
            rows.append({
                'Node ID': i,
                'Disp X (mm)': d[0] * 1e3,
                'Disp Y (mm)': d[1] * 1e3,
                'Disp Z (mm)': d[2] * 1e3,
                'Rot X (mrad)': d[3] * 1e3,
                'Rot Y (mrad)': d[4] * 1e3,
                'Rot Z (mrad)': d[5] * 1e3,
            })
        return rows

    def element_forces_table(self) -> list[dict]:
        """Return a list of dicts suitable for tabular display of element results."""
        rows = []
        for eid in sorted(self._model.element_results.keys()):
            f = self._model.element_results[eid]['local_forces']
            vm = self._model.element_results[eid]['von_mises']['max']
            conn = self._model.mesh_elements[eid]
            rows.append({
                'Element ID': eid,
                'Connectivity': f"{conn[0]} -> {conn[1]}",
                'Axial Fx (N)': f[6],
                'Torsion Mx (N-m)': f[9],
                'Bending My (N-m)': f[10],
                'Bending Mz (N-m)': f[11],
                'Max Von Mises (MPa)': vm / 1e6,
            })
        return rows

    def summary(self) -> str:
        """Human-readable results summary."""
        max_node, max_defl = self.max_deflection()
        max_elem, max_vm = self.max_von_mises()
        sf = self.safety_factor()
        lines = [
            "=== Analysis Results ===",
            f"  Nodes: {self.n_nodes}, Elements: {self.n_elements}",
            f"  Max Deflection: {max_defl * 1e3:.4f} mm at node {max_node}",
            f"  Max Von Mises:  {max_vm / 1e6:.3f} MPa at element {max_elem}",
            f"  Safety Factor:  {sf:.2f}",
        ]
        return "\n".join(lines)
