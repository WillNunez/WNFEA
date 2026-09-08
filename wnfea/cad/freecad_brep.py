"""
FreeCAD 1.1 B-Rep Feature Recognition and Automated Boundary Condition Engine.

Interfaces with the FreeCAD OpenCASCADE B-Rep kernel to:
1. Extract exact analytical cylindrical surface faces (GeomAbs_Cylinder).
2. Classify internal bolt/pin bores vs external bosses/pins.
3. Automatically match measured hole diameters against ISO 273 and ASME B18.2.8
   standard bolt clearance hole tables.
4. Synthesize automatic boundary conditions (fixed, pinned, sliding),
   bearing pressure load distributions, and kinematic spider couplings (RBE2).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import numpy as np

from ..boundary.conditions import SupportDef, LoadDef, DOFConstraint, DOFType
from ..boundary.coupling import RigidCoupling


# ============================================================================
# Standard Clearance Hole Databases (ISO 273 & ASME B18.2.8)
# ============================================================================

# Metric ISO 273 Clearance Hole Table: { Nominal: { 'close': d_close, 'medium': d_med, 'free': d_free } } (diameters in mm)
ISO_273_BOLT_TABLE: dict[str, dict[str, float]] = {
    "M2":   {"close": 2.2,  "medium": 2.4,  "free": 2.6},
    "M2.5": {"close": 2.7,  "medium": 2.9,  "free": 3.1},
    "M3":   {"close": 3.2,  "medium": 3.4,  "free": 3.6},
    "M4":   {"close": 4.3,  "medium": 4.5,  "free": 4.8},
    "M5":   {"close": 5.3,  "medium": 5.5,  "free": 5.8},
    "M6":   {"close": 6.4,  "medium": 6.6,  "free": 7.0},
    "M8":   {"close": 8.4,  "medium": 9.0,  "free": 10.0},
    "M10":  {"close": 10.5, "medium": 11.0, "free": 12.0},
    "M12":  {"close": 13.0, "medium": 13.5, "free": 14.5},
    "M14":  {"close": 15.0, "medium": 15.5, "free": 16.5},
    "M16":  {"close": 17.0, "medium": 17.5, "free": 18.5},
    "M18":  {"close": 19.0, "medium": 20.0, "free": 21.0},
    "M20":  {"close": 21.0, "medium": 22.0, "free": 24.0},
    "M22":  {"close": 23.0, "medium": 24.0, "free": 26.0},
    "M24":  {"close": 25.0, "medium": 26.0, "free": 28.0},
    "M27":  {"close": 28.0, "medium": 30.0, "free": 32.0},
    "M30":  {"close": 31.0, "medium": 33.0, "free": 35.0},
    "M36":  {"close": 37.0, "medium": 39.0, "free": 42.0},
}

# Unified / Imperial ASME B18.2.8 Clearance Hole Table (diameters in mm, converted from inches)
ASME_B18_BOLT_TABLE: dict[str, dict[str, float]] = {
    "#4":    {"close": 3.10, "normal": 3.26},
    "#6":    {"close": 3.66, "normal": 3.80},
    "#8":    {"close": 4.34, "normal": 4.50},
    "#10":   {"close": 4.98, "normal": 5.11},
    '1/4"':  {"close": 6.63, "normal": 6.76},
    '5/16"': {"close": 8.28, "normal": 8.43},
    '3/8"':  {"close": 9.93, "normal": 10.08},
    '7/16"': {"close": 11.71, "normal": 11.91},
    '1/2"':  {"close": 13.28, "normal": 13.49},
    '9/16"': {"close": 14.88, "normal": 15.08},
    '5/8"':  {"close": 16.46, "normal": 16.67},
    '3/4"':  {"close": 19.63, "normal": 19.84},
    '7/8"':  {"close": 22.81, "normal": 23.01},
    '1"':    {"close": 25.98, "normal": 26.19},
}


def match_standard_bolt(
    diameter_mm: float,
    tol_mm: float = 0.35,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    best_match = None
    min_diff = tol_mm

    # 1. Search Metric ISO 273
    for size, fits in ISO_273_BOLT_TABLE.items():
        for fit_name, d_std in fits.items():
            diff = abs(diameter_mm - d_std)
            if diff < min_diff:
                min_diff = diff
                best_match = (f"{size} (ISO 273 {fit_name.capitalize()})", size, fit_name)

    # 2. Search Imperial ASME B18.2.8
    for size, fits in ASME_B18_BOLT_TABLE.items():
        for fit_name, d_std in fits.items():
            diff = abs(diameter_mm - d_std)
            if diff < min_diff:
                min_diff = diff
                best_match = (f"{size} (ASME B18.2.8 {fit_name.capitalize()})", size, fit_name)

    return best_match if best_match else (None, None, None)


@dataclass
class CylindricalFeature:
    feature_id: int
    face_index: int
    is_internal: bool          # True = internal bore/hole; False = external pin/boss
    radius: float              # Cylinder radius in model units (mm)
    diameter: float            # 2 * radius
    axis: np.ndarray           # (3,) unit vector along cylinder axis
    center: np.ndarray         # (3,) reference center coordinate on cylinder axis
    length: float              # Axial span / depth
    area: float                # Surface area
    min_proj: float            # Minimum projection along axis
    max_proj: float            # Maximum projection along axis
    matched_standard: Optional[str] = None
    nominal_bolt_size: Optional[str] = None
    clearance_type: Optional[str] = None

    @property
    def is_bolt_hole(self) -> bool:
        return self.is_internal and self.nominal_bolt_size is not None

    def distance_to_axis(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
        v = pts - self.center
        proj = np.sum(v * self.axis, axis=1, keepdims=True)
        radial_vec = v - proj * self.axis
        return np.linalg.norm(radial_vec, axis=1)

    def projection_on_axis(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
        v = pts - self.center
        return np.sum(v * self.axis, axis=1)


class FreeCADBRepDetector:
    def __init__(self, freecad_path: Optional[str] = None):
        self.freecad_cmd = freecad_path or self.find_freecad_cmd()

    @staticmethod
    def find_freecad_cmd() -> Optional[str]:
        candidates = [
            r"C:\Program Files\FreeCAD 1.1\bin\python.exe",
            r"C:\Program Files\FreeCAD 1.0\bin\python.exe",
            r"C:\Program Files\FreeCAD\bin\python.exe",
            r"C:\Program Files\FreeCAD 1.1\bin\freecadcmd.exe",
            r"C:\Program Files\FreeCAD 1.0\bin\freecadcmd.exe",
            r"C:\Program Files\FreeCAD\bin\freecadcmd.exe",
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
        for name in ("freecadcmd", "FreeCADCmd", "freecad"):
            path = shutil.which(name)
            if path:
                return path
        return None

    def detect_cylinders(
        self,
        cad_filepath: Union[str, Path],
        unit_scale: float = 1.0,
    ) -> list[CylindricalFeature]:
        cad_path = Path(cad_filepath).resolve()
        if not cad_path.exists():
            raise FileNotFoundError(f"CAD file not found: {cad_path}")

        if not self.freecad_cmd:
            raise RuntimeError(
                r"FreeCAD 1.1 executable (freecadcmd.exe) not found. "
                r"Please verify FreeCAD installation at C:\Program Files\FreeCAD 1.1."
            )

        fc_bin = os.path.dirname(self.freecad_cmd)
        fc_root = os.path.dirname(fc_bin)
        fc_lib = os.path.join(fc_root, "lib")

        header = (
            "import os, sys, json\n"
            f"fc_bin = {repr(fc_bin)}\n"
            f"fc_lib = {repr(fc_lib)}\n"
            "if hasattr(os, 'add_dll_directory') and os.path.isdir(fc_bin):\n"
            "    try:\n"
            "        os.add_dll_directory(fc_bin)\n"
            "    except Exception:\n"
            "        pass\n"
            "for p in (fc_bin, fc_lib):\n"
            "    if os.path.isdir(p) and p not in sys.path:\n"
            "        sys.path.insert(0, p)\n"
            "\n"
            "import FreeCAD, Part\n\n"
        )

        body = '''shape = Part.Shape()
shape.read(sys.argv[1])

cylinders = []
for idx, f in enumerate(shape.Faces):
    surf_type = f.Surface.__class__.__name__
    if surf_type == "Cylinder":
        s = f.Surface
        rad = float(s.Radius)
        axis = [float(s.Axis.x), float(s.Axis.y), float(s.Axis.z)]
        center = [float(s.Center.x), float(s.Center.y), float(s.Center.z)]
        area = float(f.Area)

        try:
            u_mid = 0.5 * (f.ParameterRange[0] + f.ParameterRange[1])
            v_mid = 0.5 * (f.ParameterRange[2] + f.ParameterRange[3])
            pt = f.valueAt(u_mid, v_mid)
            norm = f.normalAt(u_mid, v_mid)
        except Exception:
            pt = f.valueAt(0.0, 0.0)
            norm = f.normalAt(0.0, 0.0)

        axis_v = s.Axis
        v_diff = pt - s.Center
        proj = v_diff - axis_v * (v_diff.dot(axis_v))
        is_internal = bool(norm.dot(proj) < 0.0)

        circ = 2.0 * 3.141592653589793 * rad
        length = area / circ if circ > 1e-12 else 0.0

        projs = []
        for v in f.Vertexes:
            v_p = v.Point - s.Center
            projs.append(v_p.dot(axis_v))
        min_proj = min(projs) if projs else -0.5 * length
        max_proj = max(projs) if projs else 0.5 * length

        cylinders.append({
            "face_idx": idx,
            "radius": rad,
            "axis": axis,
            "center": center,
            "area": area,
            "length": length,
            "min_proj": min_proj,
            "max_proj": max_proj,
            "is_internal": is_internal,
        })

print("__FREECAD_BREP_JSON_START__")
print(json.dumps(cylinders))
print("__FREECAD_BREP_JSON_END__")
'''
        script_content = header + body
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8") as tf:
            tf.write(script_content)
            script_path = tf.name

        try:
            res = subprocess.run(
                [self.freecad_cmd, script_path, str(cad_path)],
                capture_output=True,
                text=True,
                check=True,
            )
            stdout = res.stdout
        finally:
            if os.path.exists(script_path):
                os.remove(script_path)

        start_marker = "__FREECAD_BREP_JSON_START__"
        end_marker = "__FREECAD_BREP_JSON_END__"
        if start_marker not in stdout or end_marker not in stdout:
            raise RuntimeError(f"Failed to extract B-Rep cylinder metadata from FreeCAD: {stdout}")

        json_str = stdout.split(start_marker)[1].split(end_marker)[0].strip()
        data = json.loads(json_str)

        features: list[CylindricalFeature] = []
        for feat_id, item in enumerate(data):
            rad = item["radius"] * unit_scale
            diam = 2.0 * rad
            axis = np.array(item["axis"], dtype=np.float64)
            norm_a = np.linalg.norm(axis)
            if norm_a > 1e-12:
                axis /= norm_a
            center = np.array(item["center"], dtype=np.float64) * unit_scale

            std_name, nom_size, fit = match_standard_bolt(diam)

            feature = CylindricalFeature(
                feature_id=feat_id,
                face_index=item["face_idx"],
                is_internal=item["is_internal"],
                radius=rad,
                diameter=diam,
                axis=axis,
                center=center,
                length=item["length"] * unit_scale,
                area=item["area"] * (unit_scale**2),
                min_proj=item["min_proj"] * unit_scale,
                max_proj=item["max_proj"] * unit_scale,
                matched_standard=std_name,
                nominal_bolt_size=nom_size,
                clearance_type=fit,
            )
            features.append(feature)

        return features


# ============================================================================
# Automatic Boundary Condition & Load Synthesizers
# ============================================================================

def find_surface_nodes(
    feature: CylindricalFeature,
    mesh_nodes: np.ndarray,
    rad_tol: float = 0.08,
    axial_margin: float = 0.1,
) -> np.ndarray:
    nodes = np.asarray(mesh_nodes, dtype=np.float64)
    dists = feature.distance_to_axis(nodes)
    rad_err = np.abs(dists - feature.radius) / max(feature.radius, 1e-12)

    projs = feature.projection_on_axis(nodes)
    span = feature.max_proj - feature.min_proj
    axial_tol = max(span * axial_margin, 1e-4)

    mask = (
        (rad_err <= rad_tol) &
        (projs >= feature.min_proj - axial_tol) &
        (projs <= feature.max_proj + axial_tol)
    )
    return np.where(mask)[0]


def generate_bolt_supports(
    feature: CylindricalFeature,
    mesh_nodes: np.ndarray,
    bc_type: str = "fixed",
    rad_tol: float = 0.08,
) -> list[SupportDef]:
    surf_nodes = find_surface_nodes(feature, mesh_nodes, rad_tol=rad_tol)
    supports: list[SupportDef] = []
    lbl = f"Bolt_{feature.nominal_bolt_size or 'Hole'}_{feature.feature_id}"

    for nid in surf_nodes:
        sup = SupportDef(
            node_id=int(nid),
            is_geometry_node=False,
            ux=DOFConstraint(DOFType.FIXED),
            uy=DOFConstraint(DOFType.FIXED),
            uz=DOFConstraint(DOFType.FIXED),
            label=lbl,
        )
        supports.append(sup)

    return supports


def generate_bearing_loads(
    feature: CylindricalFeature,
    mesh_nodes: np.ndarray,
    total_force: float,
    force_direction: np.ndarray,
    rad_tol: float = 0.08,
) -> list[LoadDef]:
    f_dir = np.asarray(force_direction, dtype=np.float64)
    norm_f = np.linalg.norm(f_dir)
    if norm_f > 1e-12:
        f_dir /= norm_f

    surf_nodes = find_surface_nodes(feature, mesh_nodes, rad_tol=rad_tol)
    if len(surf_nodes) == 0:
        return []

    pts = mesh_nodes[surf_nodes]
    v = pts - feature.center
    proj = np.sum(v * feature.axis, axis=1, keepdims=True)
    radial_vecs = v - proj * feature.axis
    radial_lens = np.linalg.norm(radial_vecs, axis=1, keepdims=True)
    radial_dirs = radial_vecs / np.maximum(radial_lens, 1e-12)

    # Cosine weight: cos(theta) = radial_dir . f_dir
    cos_theta = np.sum(radial_dirs * f_dir, axis=1)
    contact_mask = cos_theta > 0.0

    if not np.any(contact_mask):
        return []

    active_indices = surf_nodes[contact_mask]
    active_weights = cos_theta[contact_mask]
    weight_sum = np.sum(active_weights)

    loads: list[LoadDef] = []
    lbl = f"BearingLoad_{feature.nominal_bolt_size or 'Hole'}_{feature.feature_id}"

    for nid, w in zip(active_indices, active_weights):
        f_mag = total_force * (w / weight_sum)
        f_vec = f_mag * f_dir
        loads.append(LoadDef(
            node_id=int(nid),
            is_geometry_node=False,
            fx=float(f_vec[0]),
            fy=float(f_vec[1]),
            fz=float(f_vec[2]),
            label=lbl,
        ))

    return loads


def generate_spider_coupling(
    feature: CylindricalFeature,
    mesh_nodes: np.ndarray,
    master_node_id: int,
    rad_tol: float = 0.08,
) -> RigidCoupling:
    surf_nodes = find_surface_nodes(feature, mesh_nodes, rad_tol=rad_tol)
    return RigidCoupling(
        master_node_id=int(master_node_id),
        slave_node_ids=[int(nid) for nid in surf_nodes],
        label=f"SpiderCoupling_{feature.nominal_bolt_size or 'Hole'}_{feature.feature_id}",
    )
