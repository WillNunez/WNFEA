"""
Automated ParaView State & Multi-Topology Visualization Generator for WNFEA.
----------------------------------------------------------------------------
Provides unified ParaView export capabilities:
1. VTU Exporter for Hex8 Voxel Grids & Generative Topologies:
   - Cell type 12 (VTK_HEXAHEDRON) with density fields, nodal displacements, and von Mises stresses.
2. Automated ParaView State File (.pvsm / XML) Generator:
   - Sets up pipeline sources, WarpByVector deformation filters, and color palettes.
3. Standalone ParaView Python Macro Script:
   - Allows headless pvpython rendering or instant interactive GUI loading.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence
import numpy as np

from ..mesh.voxel_mesher import VoxelGrid


VTK_HEXAHEDRON = 12
VTK_QUADRATIC_TETRA = 24


def export_voxel_grid_vtu(
    grid: VoxelGrid,
    filepath: str,
    displacements: Optional[np.ndarray] = None,
    densities: Optional[np.ndarray] = None,
    von_mises: Optional[np.ndarray] = None,
    temperatures: Optional[np.ndarray] = None,
    heat_fluxes: Optional[np.ndarray] = None,
    fatigue_damage: Optional[np.ndarray] = None,
    log_fatigue_life: Optional[np.ndarray] = None,
    threshold: float = 0.05,
) -> str:
    """
    Export active cells of a VoxelGrid to an XML-based VTK UnstructuredGrid (.vtu) file.

    Parameters:
        grid: VoxelGrid.
        filepath: Target output file path ending in .vtu.
        displacements: Optional (N_nodes * 3,) or (N_nodes, 3) nodal displacements.
        densities: Optional (N_cells,) physical density array.
        von_mises: Optional (N_cells,) von Mises stress array.
        temperatures: Optional (N_nodes,) nodal temperature field.
        heat_fluxes: Optional (N_cells, 3) centroidal heat flux vectors.
        threshold: Minimum density threshold to filter void cells (default: 0.05).

    Returns:
        out_path: Absolute path to written .vtu file.
    """
    out_path = Path(filepath).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Filter active cells
    dens = densities if densities is not None else grid.volume_fractions
    mask = dens >= threshold
    active_cell_ids = np.where(mask)[0]

    if len(active_cell_ids) == 0:
        # Export all cells if none above threshold
        active_cell_ids = np.arange(grid.total_cells)

    active_elems = grid.elements[active_cell_ids]  # (M, 8)
    n_cells = len(active_elems)

    # Extract unique active nodes
    unique_nodes, inverse = np.unique(active_elems.ravel(), return_inverse=True)
    compact_elems = inverse.reshape(n_cells, 8)
    compact_nodes = grid.nodes[unique_nodes]
    n_nodes = len(compact_nodes)

    # 2. Build connectivity and offsets
    connectivity_str = " ".join(str(idx) for idx in compact_elems.ravel())
    offsets_str = " ".join(str((i + 1) * 8) for i in range(n_cells))
    types_str = " ".join(str(VTK_HEXAHEDRON) for _ in range(n_cells))

    # 3. Points array
    points_str = " ".join(f"{pt[0]:.6e} {pt[1]:.6e} {pt[2]:.6e}" for pt in compact_nodes)

    # 4. Point Data (Displacements and Temperatures)
    point_data_xml = []
    if displacements is not None:
        u_arr = np.asarray(displacements, dtype=np.float64)
        if len(u_arr) == grid.total_nodes * 3:
            u_3d = u_arr.reshape(-1, 3)[unique_nodes]
        elif len(u_arr) == grid.total_nodes * 6:
            u_3d = u_arr.reshape(-1, 6)[unique_nodes, :3]
        else:
            u_3d = np.zeros((n_nodes, 3), dtype=np.float64)

        u_str = " ".join(f"{u[0]:.6e} {u[1]:.6e} {u[2]:.6e}" for u in u_3d)
        mag_str = " ".join(f"{np.linalg.norm(u):.6e}" for u in u_3d)

        point_data_xml.append(
            f'<DataArray type="Float64" Name="Displacement" NumberOfComponents="3" format="ascii">\n{u_str}\n</DataArray>'
        )
        point_data_xml.append(
            f'<DataArray type="Float64" Name="Displacement_Magnitude" NumberOfComponents="1" format="ascii">\n{mag_str}\n</DataArray>'
        )

    if temperatures is not None:
        t_arr = np.asarray(temperatures, dtype=np.float64)[unique_nodes]
        t_str = " ".join(f"{t:.6e}" for t in t_arr)
        point_data_xml.append(
            f'<DataArray type="Float64" Name="Temperature" NumberOfComponents="1" format="ascii">\n{t_str}\n</DataArray>'
        )

    # 5. Cell Data (Density, VonMisesStress, HeatFlux)
    cell_data_xml = []
    if densities is not None:
        dens_str = " ".join(f"{float(dens[cid]):.4f}" for cid in active_cell_ids)
        cell_data_xml.append(
            f'<DataArray type="Float64" Name="Density" NumberOfComponents="1" format="ascii">\n{dens_str}\n</DataArray>'
        )

    if von_mises is not None:
        vm_str = " ".join(f"{float(von_mises[cid]):.6e}" for cid in active_cell_ids)
        cell_data_xml.append(
            f'<DataArray type="Float64" Name="VonMisesStress" NumberOfComponents="1" format="ascii">\n{vm_str}\n</DataArray>'
        )

    if heat_fluxes is not None:
        q_arr = np.asarray(heat_fluxes, dtype=np.float64)[active_cell_ids]
        q_str = " ".join(f"{q[0]:.6e} {q[1]:.6e} {q[2]:.6e}" for q in q_arr)
        q_mag = np.linalg.norm(q_arr, axis=1)
        q_mag_str = " ".join(f"{mag:.6e}" for mag in q_mag)
        cell_data_xml.append(
            f'<DataArray type="Float64" Name="HeatFlux" NumberOfComponents="3" format="ascii">\n{q_str}\n</DataArray>'
        )
        cell_data_xml.append(
            f'<DataArray type="Float64" Name="HeatFlux_Magnitude" NumberOfComponents="1" format="ascii">\n{q_mag_str}\n</DataArray>'
        )

    if fatigue_damage is not None:
        fd_arr = np.asarray(fatigue_damage, dtype=np.float64)[active_cell_ids]
        fd_str = " ".join(f"{float(val):.6e}" for val in fd_arr)
        cell_data_xml.append(
            f'<DataArray type="Float64" Name="FatigueDamage" NumberOfComponents="1" format="ascii">\n{fd_str}\n</DataArray>'
        )

    if log_fatigue_life is not None:
        lfl_arr = np.asarray(log_fatigue_life, dtype=np.float64)[active_cell_ids]
        lfl_str = " ".join(f"{float(val):.4f}" for val in lfl_arr)
        cell_data_xml.append(
            f'<DataArray type="Float64" Name="Log10_FatigueLife" NumberOfComponents="1" format="ascii">\n{lfl_str}\n</DataArray>'
        )

    # 6. Assemble XML content
    xml_content = f"""<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="{n_nodes}" NumberOfCells="{n_cells}">
      <PointData>
        {chr(10).join(point_data_xml)}
      </PointData>
      <CellData>
        {chr(10).join(cell_data_xml)}
      </CellData>
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">
          {points_str}
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int64" Name="connectivity" format="ascii">
          {connectivity_str}
        </DataArray>
        <DataArray type="Int64" Name="offsets" format="ascii">
          {offsets_str}
        </DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">
          {types_str}
        </DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""

    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write(xml_content)

    return str(out_path)


def generate_paraview_macro(
    vtu_filepath: str,
    output_py_path: str,
    warp_scale: float = 50.0,
    color_by: str = "VonMisesStress",
) -> str:
    """
    Generate a Python script for ParaView (runnable with pvpython or ParaView GUI macro)
    that sets up automatic WarpByVector and colormapping.
    """
    macro_path = Path(output_py_path).resolve()
    macro_path.parent.mkdir(parents=True, exist_ok=True)
    abs_vtu = Path(vtu_filepath).resolve().as_posix()

    script = f"""# ParaView automated visualization script generated by WNFEA
import paraview.simple as pvs

# 1. Load VTU file
reader = pvs.XMLUnstructuredGridReader(FileName=['{abs_vtu}'])
pvs.Show(reader)

# 2. Add WarpByVector filter
warp = pvs.WarpByVector(Input=reader)
warp.Vectors = ['POINTS', 'Displacement']
warp.ScaleFactor = {warp_scale}
warp_display = pvs.Show(warp)

# 3. Setup Colormap
pvs.ColorBy(warp_display, ('CELLS', '{color_by}'))
warp_display.SetScalarBarVisibility(pvs.GetActiveViewOrCreate('RenderView'), True)

# 4. Reset Camera and Render
view = pvs.GetActiveViewOrCreate('RenderView')
view.ResetCamera()
pvs.Render()
print("WNFEA ParaView state loaded successfully!")
"""

    with open(macro_path, "w", encoding="utf-8") as fp:
        fp.write(script)

    return str(macro_path)


def export_modal_analysis_vtu(
    grid: VoxelGrid,
    filepath: str,
    modal_result: any,
    densities: Optional[np.ndarray] = None,
    threshold: float = 0.05,
) -> str:
    """
    Export dynamic vibration eigenmodes to an XML-based VTK UnstructuredGrid (.vtu) file.
    Each mode is stored as a vector DataArray 'Mode_{i+1}_Disp' with corresponding scalar magnitude.
    """
    out_path = Path(filepath).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dens = densities if densities is not None else grid.volume_fractions
    mask = dens >= threshold
    active_cell_ids = np.where(mask)[0]
    if len(active_cell_ids) == 0:
        active_cell_ids = np.arange(grid.total_cells)

    active_elems = grid.elements[active_cell_ids]
    n_cells = len(active_elems)
    unique_nodes, inverse = np.unique(active_elems.ravel(), return_inverse=True)
    compact_elems = inverse.reshape(n_cells, 8)
    compact_nodes = grid.nodes[unique_nodes]
    n_nodes = len(compact_nodes)

    connectivity_str = " ".join(str(idx) for idx in compact_elems.ravel())
    offsets_str = " ".join(str((i + 1) * 8) for i in range(n_cells))
    types_str = " ".join(str(VTK_HEXAHEDRON) for _ in range(n_cells))
    points_str = " ".join(f"{pt[0]:.6e} {pt[1]:.6e} {pt[2]:.6e}" for pt in compact_nodes)

    point_data_xml = []
    num_modes = len(modal_result.frequencies_hz)
    for m_idx in range(num_modes):
        freq = modal_result.frequencies_hz[m_idx]
        mode_3d = modal_result.mode_shapes[m_idx][unique_nodes]  # (n_nodes, 3)

        u_str = " ".join(f"{u[0]:.6e} {u[1]:.6e} {u[2]:.6e}" for u in mode_3d)
        mag_str = " ".join(f"{np.linalg.norm(u):.6e}" for u in mode_3d)

        mode_name = f"Mode_{m_idx + 1}_{freq:.1f}Hz"
        point_data_xml.append(
            f'<DataArray type="Float64" Name="{mode_name}" NumberOfComponents="3" format="ascii">\n{u_str}\n</DataArray>'
        )
        point_data_xml.append(
            f'<DataArray type="Float64" Name="{mode_name}_Magnitude" NumberOfComponents="1" format="ascii">\n{mag_str}\n</DataArray>'
        )

    cell_data_xml = []
    dens_str = " ".join(f"{float(dens[cid]):.4f}" for cid in active_cell_ids)
    cell_data_xml.append(
        f'<DataArray type="Float64" Name="Density" NumberOfComponents="1" format="ascii">\n{dens_str}\n</DataArray>'
    )

    xml_content = f"""<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="{n_nodes}" NumberOfCells="{n_cells}">
      <PointData>
        {chr(10).join(point_data_xml)}
      </PointData>
      <CellData>
        {chr(10).join(cell_data_xml)}
      </CellData>
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">
          {points_str}
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int64" Name="connectivity" format="ascii">
          {connectivity_str}
        </DataArray>
        <DataArray type="Int64" Name="offsets" format="ascii">
          {offsets_str}
        </DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">
          {types_str}
        </DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"""

    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write(xml_content)

    return str(out_path)


def generate_modal_paraview_macro(
    vtu_filepath: str,
    output_py_path: str,
    mode_name: str,
    warp_scale: float = 20.0,
) -> str:
    """Generate a ParaView script configured to warp and animate a selected natural eigenmode."""
    macro_path = Path(output_py_path).resolve()
    macro_path.parent.mkdir(parents=True, exist_ok=True)
    abs_vtu = Path(vtu_filepath).resolve().as_posix()

    script = f"""# ParaView Modal Animation Script generated by WNFEA
import paraview.simple as pvs

reader = pvs.XMLUnstructuredGridReader(FileName=['{abs_vtu}'])
pvs.Show(reader)

warp = pvs.WarpByVector(Input=reader)
warp.Vectors = ['POINTS', '{mode_name}']
warp.ScaleFactor = {warp_scale}
warp_display = pvs.Show(warp)

pvs.ColorBy(warp_display, ('POINTS', '{mode_name}_Magnitude'))
warp_display.SetScalarBarVisibility(pvs.GetActiveViewOrCreate('RenderView'), True)

view = pvs.GetActiveViewOrCreate('RenderView')
view.ResetCamera()
pvs.Render()
print("WNFEA Modal State '{mode_name}' loaded successfully!")
"""

    with open(macro_path, "w", encoding="utf-8") as fp:
        fp.write(script)

    return str(macro_path)
