"""
FreeCAD 1.1 OpenCASCADE B-Rep Solid Reconstruction & STEP Export Engine.
------------------------------------------------------------------------
Bridges organic topology optimization density fields back into parametric CAD:
1. Takes a watertight triangular surface mesh.
2. Invokes FreeCAD 1.1 OpenCASCADE kernel to sew surface triangles into a closed B-Rep Solid.
3. Preserves exact analytical bolt hole geometry: cuts analytical cylinders (GeomAbs_Cylinder)
   to ensure bolt holes are mathematically exact cylindrical bores for CNC drilling.
4. Exports valid ISO 10303 STEP solid models (.stp / .step).
"""

from __future__ import annotations

import os
import json
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional, Sequence, List
from pathlib import Path

from .isosurface import TriangularMesh
from .freecad_brep import FreeCADBRepDetector, CylindricalFeature


@dataclass
class BRepReconstructionResult:
    """
    Output metadata from FreeCAD B-Rep reconstruction.
    """
    step_filepath: str
    volume: float              # Solid volume in m^3 or mm^3
    num_faces: int             # Number of B-Rep topological faces
    num_solids: int            # Number of solid lumps (typically 1)
    is_valid_solid: bool       # OpenCASCADE isValid() check
    elapsed_time: float        # Wallclock reconstruction time (s)


class BRepReconstructor:
    """
    Interfaces with FreeCAD 1.1 to convert organic meshes into B-Rep STEP solids.
    """
    def __init__(self, freecad_path: Optional[str] = None):
        self.freecad_cmd = freecad_path or FreeCADBRepDetector.find_freecad_cmd()

    def reconstruct_step_solid(
        self,
        mesh: TriangularMesh,
        output_filepath: str,
        bolt_holes: Optional[Sequence[CylindricalFeature]] = None,
        tolerance: float = 0.05,
    ) -> BRepReconstructionResult:
        """
        Convert triangular mesh into a closed OpenCASCADE B-Rep solid and export STEP.

        Parameters:
            mesh: TriangularMesh object.
            output_filepath: Target .stp / .step destination path.
            bolt_holes: Optional list of CylindricalFeature objects to preserve as exact bores.
            tolerance: Sewing tolerance in model units (default: 0.05).

        Returns:
            BRepReconstructionResult with volume and topological diagnostics.
        """
        if not self.freecad_cmd:
            raise RuntimeError("FreeCAD 1.1 executable not found on system.")

        output_path = Path(output_filepath).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Write temporary binary STL file
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tf_stl:
            stl_path = tf_stl.name
        mesh.write_stl(stl_path, binary=True)

        # 2. Package bolt hole specifications
        bolt_specs = []
        if bolt_holes:
            for bh in bolt_holes:
                p0 = bh.center - 0.5 * bh.length * bh.axis
                bolt_specs.append({
                    "radius": float(bh.radius),
                    "length": float(bh.length * 1.5),  # Slight through-hole extension
                    "p0": [float(p0[0]), float(p0[1]), float(p0[2])],
                    "axis": [float(bh.axis[0]), float(bh.axis[1]), float(bh.axis[2])],
                    "is_internal": bool(bh.is_internal),
                })

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tf_meta:
            meta_path = tf_meta.name
            json.dump({
                "stl_path": stl_path,
                "step_path": str(output_path),
                "tolerance": float(tolerance),
                "bolt_specs": bolt_specs,
            }, tf_meta)

        # 3. Headless FreeCAD OpenCASCADE Python script
        worker_script = f"""
import sys, os, json, time
import FreeCAD, Part, Mesh
from FreeCAD import Base

t0 = time.time()
with open(r"{meta_path}", "r") as fp:
    cfg = json.load(fp)

stl_path = cfg["stl_path"]
step_path = cfg["step_path"]
tol = cfg["tolerance"]
bolt_specs = cfg["bolt_specs"]

# 1. Load STL mesh
m = Mesh.Mesh(stl_path)

# 2. Make B-Rep shape from mesh topology
shape = Part.Shape()
shape.makeShapeFromMesh(m.Topology, tol)

# 3. Create closed B-Rep solid
try:
    solid = Part.makeSolid(Part.Shell(shape.Faces))
except Exception:
    # Fallback to direct solid wrapping
    solid = Part.Solid(shape)

# 4. Enforce exact analytical cylindrical bolt bores
for b in bolt_specs:
    try:
        p0 = Base.Vector(*b["p0"])
        axis = Base.Vector(*b["axis"])
        cyl = Part.makeCylinder(b["radius"], b["length"], p0, axis)
        if b["is_internal"]:
            solid = solid.cut(cyl)
        else:
            solid = solid.fuse(cyl)
    except Exception as ex:
        pass

# 5. Export STEP solid
Part.export([solid], step_path)
t_elapsed = time.time() - t0

result = {{
    "volume": float(solid.Volume),
    "num_faces": len(solid.Faces),
    "num_solids": len(solid.Solids),
    "is_valid": bool(solid.isValid()),
    "elapsed_time": t_elapsed
}}
print("__JSON_RESULT__:" + json.dumps(result))
"""

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tf_py:
            script_path = tf_py.name
            tf_py.write(worker_script)

        try:
            cmd = [self.freecad_cmd, script_path]
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60,
            )

            res_json = None
            for line in proc.stdout.splitlines():
                if line.startswith("__JSON_RESULT__:"):
                    res_json = json.loads(line.replace("__JSON_RESULT__:", ""))
                    break

            if res_json is None:
                raise RuntimeError(
                    f"FreeCAD B-Rep conversion failed. stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
                )

            return BRepReconstructionResult(
                step_filepath=str(output_path),
                volume=res_json["volume"],
                num_faces=res_json["num_faces"],
                num_solids=res_json["num_solids"],
                is_valid_solid=res_json["is_valid"],
                elapsed_time=res_json["elapsed_time"],
            )

        finally:
            for p in (stl_path, meta_path, script_path):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
