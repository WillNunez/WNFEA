import xml.etree.ElementTree as ET
import numpy as np

for fname in ["results/cantilever_ibeam_c3d10_solved.vtu", "results/cantilever_ibeam_200k_c3d10_solved.vtu"]:
    tree = ET.parse(fname)
    piece = tree.getroot().find(".//Piece")
    for da in piece.findall(".//PointData/DataArray"):
        if da.attrib["Name"] == "VonMises_Nodal":
            vm = np.array([float(x) for x in da.text.split()])
    for da in piece.findall(".//Points/DataArray"):
        if da.attrib["Name"] == "Points":
            pts = np.array([float(x) for x in da.text.split()]).reshape(-1, 3)

    idx_max = np.argmax(vm)
    idx_min = np.argmin(vm)
    root_mask = pts[:, 0] < 0.05
    tip_mask = pts[:, 0] > 3.95
    print(f"File: {fname}")
    print(f"  Max VM: {vm[idx_max]/1e6:.2f} MPa at Point: {pts[idx_max]}")
    print(f"  Min VM: {vm[idx_min]/1e6:.2f} MPa at Point: {pts[idx_min]}")
    print(f"  Average VM at Root (x=0): {np.mean(vm[root_mask])/1e6:.2f} MPa")
    print(f"  Average VM at Tip (x=4):  {np.mean(vm[tip_mask])/1e6:.2f} MPa")
