"""
ParaView VTU Exporter for WNFEA.

Exports FEAModel geometry, meshes, nodal displacements, and element Cauchy/Von Mises
stress tensors into standard XML-based VTK UnstructuredGrid format (.vtu).
The exported file can be opened directly in open-source ParaView or any VTK-compliant
scientific visualization suite.

Supported element topologies:
- 10-node quadratic tetrahedron (C3D10) -> VTK_QUADRATIC_TETRA (cell type 24)
- 2-node 3D Euler-Bernoulli beam      -> VTK_LINE            (cell type 3)
- Hybrid / mixed beam-solid systems
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np

from ..model import FEAModel
from ..elements.c3d10 import element_stresses_c3d10


# VTK Cell Type Constants
VTK_LINE = 3
VTK_QUADRATIC_TETRA = 24


class VTUExporter:
    """
    Exports FEAModel instances to standard XML-based VTK UnstructuredGrid (.vtu) files.
    """

    def __init__(self, precision: int = 8):
        self.precision = precision

    def export(
        self,
        model: FEAModel,
        filepath: str | Path,
        compute_stresses: bool = True,
    ) -> str:
        """
        Export the given FEAModel to a .vtu file.

        Parameters
        ----------
        model : FEAModel
            The FEA model to export (solved or unsolved).
        filepath : str | Path
            Target destination path ending with .vtu.
        compute_stresses : bool
            Whether to compute and export Cauchy and Von Mises stresses if displacements exist.

        Returns
        -------
        out_path : str
            Absolute path to the exported .vtu file.
        """
        if model.mesh_nodes is None or len(model.mesh_nodes) == 0:
            raise ValueError("FEAModel contains no mesh nodes. Mesh model before exporting.")

        nodes = np.asarray(model.mesh_nodes, dtype=np.float64)
        n_points = len(nodes)

        # -------------------------------------------------------------------
        # 1. Gather Cells (Beams + Solids)
        # -------------------------------------------------------------------
        connectivity: list[int] = []
        offsets: list[int] = []
        cell_types: list[int] = []
        cell_categories: list[str] = []  # "beam" or "solid"
        cell_orig_ids: list[int] = []

        current_offset = 0

        # Beam elements
        if model.mesh_elements is not None and len(model.mesh_elements) > 0:
            for elem_id, (n1, n2) in enumerate(model.mesh_elements):
                connectivity.extend([int(n1), int(n2)])
                current_offset += 2
                offsets.append(current_offset)
                cell_types.append(VTK_LINE)
                cell_categories.append("beam")
                cell_orig_ids.append(elem_id)

        # Solid C3D10 elements
        if getattr(model, "solid_elements", None) is not None and len(model.solid_elements) > 0:
            for elem_id, elem_nodes in enumerate(model.solid_elements):
                connectivity.extend([int(n) for n in elem_nodes])
                current_offset += 10
                offsets.append(current_offset)
                cell_types.append(VTK_QUADRATIC_TETRA)
                cell_categories.append("solid")
                cell_orig_ids.append(elem_id)

        n_cells = len(offsets)
        if n_cells == 0:
            raise ValueError("FEAModel has mesh nodes but no beam or solid elements to export.")

        # -------------------------------------------------------------------
        # 2. Extract Point Data (Displacements, Rotations)
        # -------------------------------------------------------------------
        u_trans = np.zeros((n_points, 3), dtype=np.float64)
        u_rot = np.zeros((n_points, 3), dtype=np.float64)
        has_displacements = model.displacements is not None

        if has_displacements:
            U = np.asarray(model.displacements, dtype=np.float64).reshape(-1)
            if len(U) == n_points * 6:
                u_reshaped = U.reshape(n_points, 6)
                u_trans = u_reshaped[:, 0:3]
                u_rot = u_reshaped[:, 3:6]
            elif len(U) == n_points * 3:
                u_trans = U.reshape(n_points, 3)
            else:
                # Handle active DOFs or custom sizes via fallback
                dof_len = min(len(U), n_points * 6)
                u_temp = np.zeros((n_points, 6))
                u_temp.flat[:dof_len] = U[:dof_len]
                u_trans = u_temp[:, 0:3]
                u_rot = u_temp[:, 3:6]

        u_mag = np.linalg.norm(u_trans, axis=1)

        # -------------------------------------------------------------------
        # 3. Compute Cell Data (Von Mises Stress, Cauchy Tensors)
        # -------------------------------------------------------------------
        cell_vm = np.zeros(n_cells, dtype=np.float64)
        cell_stresses = np.zeros((n_cells, 6), dtype=np.float64)  # [sxx, syy, szz, sxy, syz, szx]
        nodal_vm_accum = np.zeros(n_points, dtype=np.float64)
        nodal_vm_counts = np.zeros(n_points, dtype=np.int64)

        if has_displacements and compute_stresses:
            for cell_idx, (cat, orig_id) in enumerate(zip(cell_categories, cell_orig_ids)):
                if cat == "solid":
                    elem_nodes = model.solid_elements[orig_id]
                    elem_coords = nodes[elem_nodes]

                    u_solid = np.zeros(30, dtype=np.float64)
                    for i, nid in enumerate(elem_nodes):
                        u_solid[3 * i : 3 * i + 3] = u_trans[nid]

                    mat_name = model.solid_materials.get(orig_id)
                    if not mat_name and model.materials:
                        mat_name = next(iter(model.materials))
                    mat = model.materials[mat_name]

                    gp_sig, gp_vm, max_vm = element_stresses_c3d10(
                        elem_coords, u_solid, mat.youngs_modulus, mat.poissons_ratio
                    )
                    # Average over Gauss points for cell values
                    cell_stresses[cell_idx, :] = np.mean(gp_sig, axis=0)
                    cell_vm[cell_idx] = float(np.mean(gp_vm))

                    for nid in elem_nodes:
                        nodal_vm_accum[nid] += cell_vm[cell_idx]
                        nodal_vm_counts[nid] += 1

                elif cat == "beam":
                    if model.element_results and orig_id in model.element_results:
                        er = model.element_results[orig_id]
                        vm = getattr(er, "max_von_mises", 0.0)
                        cell_vm[cell_idx] = vm
                        cell_stresses[cell_idx, 0] = getattr(er, "axial_stress", 0.0)

                    n1, n2 = model.mesh_elements[orig_id]
                    for nid in (n1, n2):
                        nodal_vm_accum[nid] += cell_vm[cell_idx]
                        nodal_vm_counts[nid] += 1

        # Smooth nodal Von Mises stress (for smooth contour maps in ParaView)
        nodal_vm = np.zeros(n_points, dtype=np.float64)
        mask = nodal_vm_counts > 0
        nodal_vm[mask] = nodal_vm_accum[mask] / nodal_vm_counts[mask]

        # -------------------------------------------------------------------
        # 4. Construct VTK XML UnstructuredGrid Document
        # -------------------------------------------------------------------
        lines: list[str] = [
            '<?xml version="1.0"?>',
            '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">',
            '  <UnstructuredGrid>',
            f'    <Piece NumberOfPoints="{n_points}" NumberOfCells="{n_cells}">',
        ]

        # Point Data
        lines.append('      <PointData>')
        lines.append(f'        <DataArray type="Float64" Name="Displacement" NumberOfComponents="3" format="ascii">')
        lines.append("          " + " ".join(f"{v:.8e}" for v in u_trans.flat))
        lines.append('        </DataArray>')

        lines.append(f'        <DataArray type="Float64" Name="DisplacementMagnitude" NumberOfComponents="1" format="ascii">')
        lines.append("          " + " ".join(f"{v:.8e}" for v in u_mag))
        lines.append('        </DataArray>')

        if has_displacements and np.any(u_rot != 0.0):
            lines.append(f'        <DataArray type="Float64" Name="Rotation" NumberOfComponents="3" format="ascii">')
            lines.append("          " + " ".join(f"{v:.8e}" for v in u_rot.flat))
            lines.append('        </DataArray>')

        if has_displacements and compute_stresses:
            lines.append(f'        <DataArray type="Float64" Name="VonMises_Nodal" NumberOfComponents="1" format="ascii">')
            lines.append("          " + " ".join(f"{v:.8e}" for v in nodal_vm))
            lines.append('        </DataArray>')
        lines.append('      </PointData>')

        # Cell Data
        lines.append('      <CellData>')
        if has_displacements and compute_stresses:
            lines.append(f'        <DataArray type="Float64" Name="VonMises" NumberOfComponents="1" format="ascii">')
            lines.append("          " + " ".join(f"{v:.8e}" for v in cell_vm))
            lines.append('        </DataArray>')

            lines.append(f'        <DataArray type="Float64" Name="StressTensor" NumberOfComponents="6" format="ascii">')
            lines.append("          " + " ".join(f"{v:.8e}" for v in cell_stresses.flat))
            lines.append('        </DataArray>')

        lines.append(f'        <DataArray type="Int64" Name="CellID" NumberOfComponents="1" format="ascii">')
        lines.append("          " + " ".join(str(i) for i in range(n_cells)))
        lines.append('        </DataArray>')
        lines.append('      </CellData>')

        # Points
        lines.append('      <Points>')
        lines.append('        <DataArray type="Float64" Name="Points" NumberOfComponents="3" format="ascii">')
        lines.append("          " + " ".join(f"{v:.8e}" for v in nodes.flat))
        lines.append('        </DataArray>')
        lines.append('      </Points>')

        # Cells
        lines.append('      <Cells>')
        lines.append('        <DataArray type="Int64" Name="connectivity" format="ascii">')
        lines.append("          " + " ".join(str(c) for c in connectivity))
        lines.append('        </DataArray>')

        lines.append('        <DataArray type="Int64" Name="offsets" format="ascii">')
        lines.append("          " + " ".join(str(o) for o in offsets))
        lines.append('        </DataArray>')

        lines.append('        <DataArray type="UInt8" Name="types" format="ascii">')
        lines.append("          " + " ".join(str(t) for t in cell_types))
        lines.append('        </DataArray>')
        lines.append('      </Cells>')

        # Footer
        lines.append('    </Piece>')
        lines.append('  </UnstructuredGrid>')
        lines.append('</VTKFile>')

        # Write to file
        out_path = Path(filepath).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        return str(out_path)


def export_vtu(
    model: FEAModel,
    filepath: str | Path,
    compute_stresses: bool = True,
) -> str:
    """Convenience function to export a FEAModel to a .vtu file."""
    exporter = VTUExporter()
    return exporter.export(model, filepath, compute_stresses=compute_stresses)
