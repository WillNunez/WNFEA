"""
Stress post-processing for WNFEA.

Computes element internal forces and Von Mises equivalent stresses at the
extreme fiber points of each beam cross-section.
"""

import numpy as np
from ..model import FEAModel
from .assembler import build_element_stiffness_3d_beam, build_transformation_matrix


def compute_element_stresses(model: FEAModel) -> dict:
    """
    Post-process solved displacements to compute element-level results.

    For each element, calculates:
      - Local internal forces and moments (12-component vector)
      - Von Mises equivalent stress at 4 extreme fiber points on the
        outer radius of the cross-section

    Args:
        model: FEAModel with displacements already computed.

    Returns:
        dict mapping element_id -> {
            'local_forces': np.ndarray (12,),
            'von_mises': {
                'y_plus': float, 'y_minus': float,
                'z_plus': float, 'z_minus': float,
                'max': float,
            }
        }
        Also stored in model.element_results.
    """
    if model.displacements is None:
        raise ValueError("No displacements computed. Run the solver first.")

    U = model.displacements
    results = {}

    for elem_id, (n1_idx, n2_idx) in enumerate(model.mesh_elements):
        node1 = model.mesh_nodes[n1_idx]
        node2 = model.mesh_nodes[n2_idx]

        assignment = model.element_properties[elem_id]
        mat = model.materials[assignment.material_name]
        sec = model.sections[assignment.section_name]

        E = mat.youngs_modulus
        G = mat.shear_modulus
        A = sec.area
        Iy = sec.iy
        Iz = sec.iz
        J = sec.j
        ro = sec.outer_radius if sec.outer_radius else 0.0

        L = float(np.linalg.norm(node2 - node1))
        if L < 1e-12:
            results[elem_id] = {
                'local_forces': np.zeros(12),
                'von_mises': {'y_plus': 0, 'y_minus': 0, 'z_plus': 0, 'z_minus': 0, 'max': 0},
            }
            continue

        # Build local stiffness and transformation
        Ke_local = build_element_stiffness_3d_beam(node1, node2, E, G, A, Iy, Iz, J)
        T = build_transformation_matrix(node1, node2)

        # Extract element global displacements and transform to local
        dofs = np.concatenate([
            np.arange(n1_idx * 6, n1_idx * 6 + 6),
            np.arange(n2_idx * 6, n2_idx * 6 + 6),
        ])
        u_global = U[dofs]
        u_local = T @ u_global

        # Local internal forces
        f_local = Ke_local @ u_local

        # Stress at node 2 end (critical section)
        Fx2 = f_local[6]
        Mx2 = f_local[9]
        My2 = f_local[10]
        Mz2 = f_local[11]

        # Evaluate Von Mises at 4 extreme fiber points
        vm_dict = {}
        points = {
            'y_plus': (ro, 0.0),
            'y_minus': (-ro, 0.0),
            'z_plus': (0.0, ro),
            'z_minus': (0.0, -ro),
        }

        for label, (yp, zp) in points.items():
            sigma_axial = Fx2 / A if A > 0 else 0
            sigma_bz = (-Mz2 * yp) / Iz if Iz > 0 else 0
            sigma_by = (My2 * zp) / Iy if Iy > 0 else 0
            sigma_normal = sigma_axial + sigma_bz + sigma_by

            tau_torsion = (abs(Mx2) * ro) / J if J > 0 and ro > 0 else 0

            vm = np.sqrt(sigma_normal**2 + 3 * tau_torsion**2)
            vm_dict[label] = float(vm)

        vm_dict['max'] = max(vm_dict.values()) if vm_dict else 0.0

        results[elem_id] = {
            'local_forces': f_local,
            'von_mises': vm_dict,
        }

    model.element_results = results
    return results
