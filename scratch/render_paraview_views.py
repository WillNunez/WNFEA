"""
Publication-Quality ParaView Post-Processing Script for Solved C3D10 I-Beam.
Executed via pvpython.exe.
"""

import os
from pathlib import Path
import paraview.simple as pvs

results_dir = Path("results").resolve()
vtu_file = results_dir / "cantilever_ibeam_c3d10_solved.vtu"

print(f"Loading Solved FEA Model into ParaView: {vtu_file}...")

# 1. Read the VTU file
reader = pvs.XMLUnstructuredGridReader(registrationName="C3D10_IBeam_Solved", FileName=[str(vtu_file)])
reader.PointArrayStatus = ["Displacement", "DisplacementMagnitude", "VonMises_Nodal"]
reader.CellArrayStatus = ["VonMises", "StressTensor"]

# Active Render View
view = pvs.GetActiveViewOrCreate("RenderView")
view.ViewSize = [1920, 1080]
view.Background = [0.12, 0.14, 0.18]  # Dark slate gray
view.OrientationAxesVisibility = 1

# 2. Warp By Vector filter (Scale Factor = 15.0 for clear flexure across 4m span)
warp = pvs.WarpByVector(registrationName="Warp_Displacement", Input=reader)
warp.Vectors = ["POINTS", "Displacement"]
warp.ScaleFactor = 15.0

# 3. Create Display for Warped Geometry
warp_disp = pvs.Show(warp, view, "UnstructuredGridRepresentation")
warp_disp.Representation = "Surface"

# ---------------------------------------------------------------------------
# View 1: Isometric 3D View Colored by Von Mises Stress (0 to 90 MPa)
# ---------------------------------------------------------------------------
pvs.ColorBy(warp_disp, ("POINTS", "VonMises_Nodal"))
vm_lut = pvs.GetColorTransferFunction("VonMises_Nodal")
vm_lut.ApplyPreset("Turbo", True)
vm_lut.RescaleTransferFunction(0.0, 9.0e7)  # 0 to 90 MPa

warp_disp.SetScalarBarVisibility(view, True)
sb_vm = pvs.GetScalarBar(vm_lut, view)
sb_vm.Title = "Von Mises Stress (Pa)"
sb_vm.ComponentTitle = ""
sb_vm.TitleColor = [1.0, 1.0, 1.0]
sb_vm.LabelColor = [0.9, 0.9, 0.9]
sb_vm.Position = [0.85, 0.25]
sb_vm.ScalarBarLength = 0.50

# Optimal 3D isometric camera framing
view.CameraPosition = [7.2, 3.8, 2.6]
view.CameraFocalPoint = [2.0, 0.0, -0.05]
view.CameraViewUp = [0.0, 0.0, 1.0]
view.CameraParallelScale = 2.1
pvs.Render(view)

out_vm = results_dir / "paraview_ibeam_von_mises_deformed.png"
pvs.SaveScreenshot(str(out_vm), view, ImageResolution=[1920, 1080])
print(f"Saved View 1: {out_vm}")

# ---------------------------------------------------------------------------
# View 2: Close-up on Root Flange Clamped Bending Stresses
# ---------------------------------------------------------------------------
view.CameraPosition = [0.85, 0.95, 0.65]
view.CameraFocalPoint = [0.20, 0.0, 0.0]
view.CameraViewUp = [0.0, 0.0, 1.0]
view.CameraParallelScale = 0.35
pvs.Render(view)

out_root = results_dir / "paraview_ibeam_root_stress_closeup.png"
pvs.SaveScreenshot(str(out_root), view, ImageResolution=[1920, 1080])
print(f"Saved View 2: {out_root}")

# ---------------------------------------------------------------------------
# View 3: Full Cantilever Deflection Magnitude (0 to 15 mm)
# ---------------------------------------------------------------------------
view.CameraPosition = [5.8, -4.2, 2.2]
view.CameraFocalPoint = [2.0, 0.0, -0.10]
view.CameraViewUp = [0.0, 0.0, 1.0]
view.CameraParallelScale = 2.1

# Hide Von Mises scalar bar before showing displacement scalar bar
warp_disp.SetScalarBarVisibility(view, False)

pvs.ColorBy(warp_disp, ("POINTS", "DisplacementMagnitude"))
disp_lut = pvs.GetColorTransferFunction("DisplacementMagnitude")
disp_lut.ApplyPreset("Viridis", True)
disp_lut.RescaleTransferFunction(0.0, 0.015)  # 0 to 15 mm

disp_sb = pvs.GetScalarBar(disp_lut, view)
disp_sb.Title = "Displacement (m)"
disp_sb.ComponentTitle = ""
disp_sb.TitleColor = [1.0, 1.0, 1.0]
disp_sb.LabelColor = [0.9, 0.9, 0.9]
disp_sb.Position = [0.85, 0.25]
disp_sb.ScalarBarLength = 0.50
disp_sb.Visibility = 1

pvs.Render(view)
out_disp = results_dir / "paraview_ibeam_displacement_magnitude.png"
pvs.SaveScreenshot(str(out_disp), view, ImageResolution=[1920, 1080])
print(f"Saved View 3: {out_disp}")

# ---------------------------------------------------------------------------
# View 4: Longitudinal Cross-Sectional Slice (Web Centerline Y = 0)
# ---------------------------------------------------------------------------
disp_sb.Visibility = 0
slice1 = pvs.Slice(registrationName="Slice_Web_Centerline", Input=warp)
slice1.SliceType = "Plane"
slice1.SliceType.Origin = [2.0, 0.0, 0.0]
slice1.SliceType.Normal = [0.0, 1.0, 0.0]

pvs.Hide(warp, view)
slice_disp = pvs.Show(slice1, view, "GeometryRepresentation")
slice_disp.Representation = "Surface"
pvs.ColorBy(slice_disp, ("POINTS", "VonMises_Nodal"))
slice_disp.SetScalarBarVisibility(view, True)

sb_vm.Visibility = 1
view.CameraPosition = [2.0, -4.5, -0.05]
view.CameraFocalPoint = [2.0, 0.0, -0.05]
view.CameraViewUp = [0.0, 0.0, 1.0]
view.CameraParallelScale = 1.2
pvs.Render(view)

out_slice = results_dir / "paraview_ibeam_internal_slice.png"
pvs.SaveScreenshot(str(out_slice), view, ImageResolution=[1920, 1080])
print(f"Saved View 4: {out_slice}")

# Restore full warped view for saving state
pvs.Hide(slice1, view)
pvs.Show(warp, view, "UnstructuredGridRepresentation")
pvs.ColorBy(warp_disp, ("POINTS", "VonMises_Nodal"))
view.CameraPosition = [7.2, 3.8, 2.6]
view.CameraFocalPoint = [2.0, 0.0, -0.05]
view.CameraViewUp = [0.0, 0.0, 1.0]
pvs.Render(view)

# Save ParaView State File (.pvsm)
state_file = results_dir / "cantilever_ibeam_c3d10_postprocessed.pvsm"
pvs.SaveState(str(state_file))
print(f"Saved ParaView State: {state_file}")
print("ParaView post-processing completed successfully!")
