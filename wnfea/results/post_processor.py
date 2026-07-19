"""
Post-processor for WNFEA beam FEA results.

After the solver populates ``model.displacements``, the post-processor
computes per-element internal forces, stresses at extreme-fiber points,
von Mises equivalent stress, and model-level aggregated results.

The stress recovery mirrors the approach in the original ``main.py``
(``calculate_element_results_3d_beam``), refactored into the package
architecture.
"""

from __future__ import annotations

import numpy as np

from ..model import FEAModel
from ..solver.beam_solver import BeamSolver
from .result_data import StressPoint, ElementResult, ModelResults


class PostProcessError(Exception):
    """Raised when post-processing fails."""
    pass


class PostProcessor:
    """
    Recovers element forces, stresses, and model-level summary from a
    solved FEA model.

    Usage::

        pp = PostProcessor()
        results = pp.process(model)
        print(results.summary())
    """

    def process(self, model: FEAModel) -> ModelResults:
        """
        Compute all post-processing results.

        Parameters
        ----------
        model : FEAModel
            A model that has been solved (``model.displacements`` is not None).

        Returns
        -------
        ModelResults
            Aggregated results object.

        Raises
        ------
        PostProcessError
            If the model has not been solved.
        """
        if model.displacements is None:
            raise PostProcessError("Model has not been solved yet.")

        nodes = model.mesh_nodes
        elements = model.mesh_elements
        U = model.displacements

        element_results: dict[int, ElementResult] = {}
        global_max_vm = 0.0
        global_max_vm_elem = -1

        for elem_id in range(len(elements)):
            n1_idx, n2_idx = elements[elem_id]
            node1 = nodes[n1_idx]
            node2 = nodes[n2_idx]

            assignment = model.element_properties[elem_id]
            material = model.materials[assignment.material_name]
            section = model.sections[assignment.section_name]

            er = self._element_result(
                elem_id, n1_idx, n2_idx, node1, node2, U,
                material.youngs_modulus, material.shear_modulus,
                section.area, section.iy, section.iz, section.j,
                section.outer_radius, section.inner_radius,
            )
            element_results[elem_id] = er

            if er.max_von_mises > global_max_vm:
                global_max_vm = er.max_von_mises
                global_max_vm_elem = elem_id

        # Max displacement (translational only)
        disp_reshaped = U.reshape(-1, 6)
        disp_magnitudes = np.linalg.norm(disp_reshaped[:, :3], axis=1)
        max_disp_node = int(np.argmax(disp_magnitudes))
        max_disp = float(disp_magnitudes[max_disp_node])

        # Safety factor: min over all elements
        # Find the minimum yield strength across used materials
        min_yield = float("inf")
        for assignment in model.element_properties.values():
            mat = model.materials[assignment.material_name]
            if mat.yield_strength < min_yield:
                min_yield = mat.yield_strength

        safety_factor = None
        if global_max_vm > 1e-15 and min_yield < float("inf"):
            safety_factor = min_yield / global_max_vm

        # Store element results on model as well
        model.element_results = {
            eid: {
                "local_forces_moments": er.local_forces,
                "von_mises_stresses": {
                    sp.description: sp.von_mises for sp in er.stress_points
                },
                "stress_components": {
                    sp.description: (
                        sp.sigma_total, 0.0, 0.0,
                        sp.tau_torsion, 0.0, sp.tau_torsion
                    )
                    for sp in er.stress_points
                },
                "max_von_mises": er.max_von_mises,
            }
            for eid, er in element_results.items()
        }

        return ModelResults(
            element_results=element_results,
            max_displacement=max_disp,
            max_displacement_node=max_disp_node,
            max_von_mises=global_max_vm,
            max_von_mises_element=global_max_vm_elem,
            reaction_forces=getattr(model, "reaction_forces", None),
            safety_factor=safety_factor,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _element_result(
        self,
        elem_id: int,
        n1_idx: int, n2_idx: int,
        node1: np.ndarray, node2: np.ndarray,
        U: np.ndarray,
        E: float, G: float,
        A: float, Iy: float, Iz: float, J: float,
        outer_radius: float | None,
        inner_radius: float,
    ) -> ElementResult:
        """Compute forces and stresses for a single element."""
        vec = node2 - node1
        L = float(np.linalg.norm(vec))

        if L < 1e-12:
            return ElementResult(element_id=elem_id, node1_id=n1_idx, node2_id=n2_idx)

        # Build the rotation / transformation (reuse solver logic)
        R = BeamSolver._rotation_matrix_3x3(node1, node2)
        T = np.zeros((12, 12))
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[6:9, 6:9] = R
        T[9:12, 9:12] = R

        # Extract element global displacements (12 DOFs)
        dofs = np.concatenate([
            np.arange(n1_idx * 6, n1_idx * 6 + 6),
            np.arange(n2_idx * 6, n2_idx * 6 + 6),
        ])
        u_global = U[dofs]

        # Transform to local
        u_local = T @ u_global

        # Local stiffness
        Ke_local = BeamSolver._local_stiffness(E, G, A, Iy, Iz, J, node1, node2)

        # Local internal forces
        f_local = Ke_local @ u_local

        # Stress recovery at extreme-fiber points
        stress_points: list[StressPoint] = []
        max_vm = 0.0

        if outer_radius is not None and outer_radius > 0:
            # Evaluate at 4 extreme-fiber points on the cross-section
            eval_points = [
                ("y_plus",  outer_radius, 0.0),
                ("y_minus", -outer_radius, 0.0),
                ("z_plus",  0.0, outer_radius),
                ("z_minus", 0.0, -outer_radius),
            ]

            # Use forces at node 2 end (indices 6–11 in local vector)
            Fx2 = f_local[6]
            Mx2 = f_local[9]
            My2 = f_local[10]
            Mz2 = f_local[11]

            for desc, y_local, z_local in eval_points:
                sigma_axial = Fx2 / A if A > 0 else 0.0
                sigma_bending_z = (-Mz2 * y_local) / Iz if Iz > 0 else 0.0
                sigma_bending_y = (My2 * z_local) / Iy if Iy > 0 else 0.0
                sigma_total = sigma_axial + sigma_bending_z + sigma_bending_y

                r = np.sqrt(y_local**2 + z_local**2)
                tau_torsion = (abs(Mx2) * r) / J if J > 0 and r > 0 else 0.0

                # Von Mises: sqrt(σ² + 3·τ²)
                von_mises = float(np.sqrt(sigma_total**2 + 3 * tau_torsion**2))

                sp = StressPoint(
                    description=desc,
                    sigma_axial=sigma_axial,
                    sigma_bending_y=sigma_bending_y,
                    sigma_bending_z=sigma_bending_z,
                    sigma_total=sigma_total,
                    tau_torsion=tau_torsion,
                    von_mises=von_mises,
                )
                stress_points.append(sp)

                if von_mises > max_vm:
                    max_vm = von_mises

        return ElementResult(
            element_id=elem_id,
            node1_id=n1_idx,
            node2_id=n2_idx,
            local_forces=f_local,
            stress_points=stress_points,
            max_von_mises=max_vm,
        )
