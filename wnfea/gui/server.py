"""
WNFEA Web GUI Server & REST API Engine.
---------------------------------------
Provides lightweight HTTP backend for the WNFEA Generative Design Web Studio.
Serves static UI assets and JSON REST endpoints for parameter inputs, materials,
manufacturability constraints (3-axis CNC), optimization execution, and 3D mesh streaming.
"""

from __future__ import annotations

import os
import sys
import json
import time
import threading
import tempfile
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from ..mesh.voxel_mesher import VoxelMesher, VoxelGrid
from ..opt.topology import TopologyOptimizer, TopologyConfig, OptimizationResult
from ..opt.machinability import CNCMillingConstraint
from ..cad.isosurface import extract_isosurface_mesh, TriangularMesh
from ..cad.brep_reconstruction import BRepReconstructor, CylindricalFeature
from ..opt.evolutionary import EvolutionaryTopologyOptimizer, EvolutionaryRunResult


MATERIALS_DB: Dict[str, Dict[str, Any]] = {
    "al6061_t6": {
        "id": "al6061_t6",
        "name": "Aluminum 6061-T6",
        "category": "Aluminum Alloy",
        "E": 68.9e9,          # Pa
        "nu": 0.33,
        "density": 2700.0,    # kg/m^3
        "yield_strength": 276e6,  # Pa (276 MPa)
        "uts": 310e6,         # Pa
        "machinability_rating": 0.90,
        "recommended_milling_axis": "bi-z",
        "default_cutter_radius": 0.003, # 3 mm
    },
    "al7075_t6": {
        "id": "al7075_t6",
        "name": "Aluminum 7075-T6",
        "category": "High-Strength Aerospace Aluminum",
        "E": 71.7e9,
        "nu": 0.33,
        "density": 2810.0,
        "yield_strength": 503e6,
        "uts": 572e6,
        "machinability_rating": 0.85,
        "recommended_milling_axis": "bi-z",
        "default_cutter_radius": 0.003,
    },
    "ti6al4v": {
        "id": "ti6al4v",
        "name": "Titanium Ti-6Al-4V (Grade 5)",
        "category": "Titanium Alloy",
        "E": 113.8e9,
        "nu": 0.34,
        "density": 4430.0,
        "yield_strength": 880e6,
        "uts": 950e6,
        "machinability_rating": 0.45,
        "recommended_milling_axis": "bi-z",
        "default_cutter_radius": 0.002,
    },
    "structural_steel": {
        "id": "structural_steel",
        "name": "Structural Steel A36",
        "category": "Ferrous Steel",
        "E": 200.0e9,
        "nu": 0.29,
        "density": 7850.0,
        "yield_strength": 250e6,
        "uts": 400e6,
        "machinability_rating": 0.70,
        "recommended_milling_axis": "bi-z",
        "default_cutter_radius": 0.004,
    },
    "stainless_316l": {
        "id": "stainless_316l",
        "name": "Stainless Steel 316L",
        "category": "Stainless Steel",
        "E": 193.0e9,
        "nu": 0.27,
        "density": 8000.0,
        "yield_strength": 290e6,
        "uts": 580e6,
        "machinability_rating": 0.50,
        "recommended_milling_axis": "bi-z",
        "default_cutter_radius": 0.003,
    },
}

PRESETS_DB: Dict[str, Dict[str, Any]] = {
    "model_car_chassis_300mm": {
        "id": "model_car_chassis_300mm",
        "name": "300mm Model Car Chassis (Double Wishbone 5g 6061 3-Axis)",
        "description": (
            "1/10 to 1/8 scale RC / autonomous competition model car chassis. "
            "Optimized for 5g tire grip acceleration across cornering, braking, bump, "
            "and torsional cases. Constrained for 3-axis top/bottom CNC vertical milling."
        ),
        "dimensions": {"lx": 0.300, "ly": 0.100, "lz": 0.036},  # 300 x 100 x 36 mm
        "resolution": {"nx": 40, "ny": 16, "nz": 8},
        "material_id": "al6061_t6",
        "target_volume_fraction": 0.28,
        "filter_radius": 0.008,
        "simp_penalty": 3.0,
        "max_iterations": 25,
        "cnc_milling": {
            "enabled": True,
            "axis": "bi-z",
            "cutter_radius": 0.003,
            "penalty_weight": 0.20,
        },
        "suspension_type": "double_wishbone",
        "load_cases": [
            {"name": "5g Lateral Cornering", "weight": 0.35, "g_level": 5.0, "type": "lateral"},
            {"name": "5g Longitudinal Braking", "weight": 0.25, "g_level": 5.0, "type": "braking"},
            {"name": "5g Vertical Bump", "weight": 0.25, "g_level": 5.0, "type": "bump"},
            {"name": "Torsional Rigidity", "weight": 0.15, "g_level": 2.5, "type": "torsion"},
        ],
    },
    "aerospace_bracket": {
        "id": "aerospace_bracket",
        "name": "Aerospace Angle Bracket (6061 3-Axis)",
        "description": "Standard aerospace structural mounting bracket under combined tension and shear.",
        "dimensions": {"lx": 0.150, "ly": 0.060, "lz": 0.040},
        "resolution": {"nx": 30, "ny": 15, "nz": 10},
        "material_id": "al6061_t6",
        "target_volume_fraction": 0.30,
        "filter_radius": 0.007,
        "simp_penalty": 3.0,
        "max_iterations": 20,
        "cnc_milling": {
            "enabled": True,
            "axis": "bi-z",
            "cutter_radius": 0.003,
            "penalty_weight": 0.15,
        },
        "suspension_type": "none",
        "load_cases": [
            {"name": "Vertical Shear", "weight": 0.60, "g_level": 3.0, "type": "shear"},
            {"name": "Lateral Bending", "weight": 0.40, "g_level": 2.0, "type": "bending"},
        ],
    },
}


class OptimizationJob:
    """
    Manages background execution of topology optimization and exports.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.is_running = False
        self.is_completed = False
        self.error_message: Optional[str] = None
        self.current_iteration = 0
        self.max_iterations = 0
        self.compliance_history: List[float] = []
        self.volume_history: List[float] = []
        self.change_history: List[float] = []
        self.status_text = "Idle"
        self.elapsed_time = 0.0

        # Current 3D mesh data (vertices and faces for Three.js)
        self.mesh_vertices: Optional[List[List[float]]] = None
        self.mesh_faces: Optional[List[List[int]]] = None
        self.final_mesh: Optional[TriangularMesh] = None
        self.opt_result: Optional[OptimizationResult] = None
        self.grid: Optional[VoxelGrid] = None
        self.config_params: Dict[str, Any] = {}

        # Export filepaths
        self.stl_path: Optional[str] = None
        self.step_path: Optional[str] = None
        self.vtu_path: Optional[str] = None
        self.candidates_5: List[Dict[str, Any]] = []

    def reset(self, config_params: Dict[str, Any]):
        with self.lock:
            self.is_running = True
            self.is_completed = False
            self.error_message = None
            self.current_iteration = 0
            self.max_iterations = config_params.get("max_iterations", 25)
            self.compliance_history.clear()
            self.volume_history.clear()
            self.change_history.clear()
            self.status_text = "Initializing FEA mesh & load cases..."
            self.elapsed_time = 0.0
            self.mesh_vertices = None
            self.mesh_faces = None
            self.final_mesh = None
            self.opt_result = None
            self.grid = None
            self.config_params = config_params
            self.stl_path = None
            self.step_path = None
            self.vtu_path = None
            self.candidates_5.clear()

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "is_running": self.is_running,
                "is_completed": self.is_completed,
                "error": self.error_message,
                "current_iteration": self.current_iteration,
                "max_iterations": self.max_iterations,
                "status_text": self.status_text,
                "elapsed_time": round(self.elapsed_time, 2),
                "compliance_history": list(self.compliance_history),
                "volume_history": list(self.volume_history),
                "change_history": list(self.change_history),
                "current_compliance": self.compliance_history[-1] if self.compliance_history else 0.0,
                "current_volume_fraction": self.volume_history[-1] if self.volume_history else 0.0,
                "has_mesh": self.mesh_vertices is not None,
                "has_stl": self.stl_path is not None and os.path.exists(self.stl_path),
                "has_step": self.step_path is not None and os.path.exists(self.step_path),
                "has_vtu": self.vtu_path is not None and os.path.exists(self.vtu_path),
                "has_candidates": len(self.candidates_5) > 0,
                "candidates": [
                    {k: v for k, v in c.items() if k not in ("vertices", "faces")}
                    for c in self.candidates_5
                ],
            }


GLOBAL_JOB = OptimizationJob()


def build_model_car_chassis_problem(
    lx: float = 0.300,
    ly: float = 0.100,
    lz: float = 0.036,
    nx: int = 40,
    ny: int = 16,
    nz: int = 8,
    mat_id: str = "al6061_t6",
    vehicle_mass: float = 2.0,  # kg
) -> Tuple[VoxelGrid, List[np.ndarray], List[float], List[int], List[int], List[int]]:
    """
    Constructs the 300mm Model Car Chassis FEA problem with double wishbone suspension hardpoints.
    Returns:
        (grid, load_cases, load_weights, fixed_dofs, passive_solid_elems, passive_void_elems)
    """
    grid = VoxelMesher.create_box_grid(bounds=(0.0, lx, -ly / 2.0, ly / 2.0, 0.0, lz), resolution=(nx, ny, nz))
    n_dofs = grid.total_nodes * 3

    # Suspension geometry parameters for 300mm chassis:
    # Wheelbase = 210 mm, Front overhang = 45 mm, Rear overhang = 45 mm
    x_front_axle = 0.055
    x_rear_axle = 0.245
    half_track_inboard = 0.038  # Chassis width at suspension pickup mounts
    z_lca = 0.006               # Lower Control Arm pivot height
    z_uca = 0.024               # Upper Control Arm pivot height
    delta_x_spread = 0.022      # Spread between front & rear wishbone pivots

    # Identify suspension pickup nodes
    front_left_uca = []
    front_left_lca = []
    front_right_uca = []
    front_right_lca = []
    rear_left_uca = []
    rear_left_lca = []
    rear_right_uca = []
    rear_right_lca = []

    for i, pt in enumerate(grid.nodes):
        x, y, z = pt
        # Front Left (+y)
        if abs(x - x_front_axle) <= delta_x_spread and abs(y - half_track_inboard) < 0.015:
            if abs(z - z_uca) < 0.008:
                front_left_uca.append(i)
            elif abs(z - z_lca) < 0.008:
                front_left_lca.append(i)
        # Front Right (-y)
        elif abs(x - x_front_axle) <= delta_x_spread and abs(y - (-half_track_inboard)) < 0.015:
            if abs(z - z_uca) < 0.008:
                front_right_uca.append(i)
            elif abs(z - z_lca) < 0.008:
                front_right_lca.append(i)
        # Rear Left (+y)
        elif abs(x - x_rear_axle) <= delta_x_spread and abs(y - half_track_inboard) < 0.015:
            if abs(z - z_uca) < 0.008:
                rear_left_uca.append(i)
            elif abs(z - z_lca) < 0.008:
                rear_left_lca.append(i)
        # Rear Right (-y)
        elif abs(x - x_rear_axle) <= delta_x_spread and abs(y - (-half_track_inboard)) < 0.015:
            if abs(z - z_uca) < 0.008:
                rear_right_uca.append(i)
            elif abs(z - z_lca) < 0.008:
                rear_right_lca.append(i)

    # Passive solid elements around suspension mounting lugs
    susp_nodes_all = set(
        front_left_uca + front_left_lca + front_right_uca + front_right_lca +
        rear_left_uca + rear_left_lca + rear_right_uca + rear_right_lca
    )
    passive_solid = []
    passive_void = []

    elem_centers = grid.nodes[grid.elements].mean(axis=1)  # (N, 3)

    for e in range(grid.total_cells):
        cx, cy, cz = elem_centers[e]
        # Check if element touches suspension pickup
        nodes_e = grid.elements[e]
        if any(n in susp_nodes_all for n in nodes_e):
            passive_solid.append(e)
            continue

        # Central electronics & drivetrain void tunnel:
        # Central electronics bay: x in [0.090, 0.210], |y| < 0.024, z in [0.008, 0.034]
        if (0.095 < cx < 0.205) and (abs(cy) < 0.022) and (0.008 < cz < 0.032):
            passive_void.append(e)

    # Base boundary condition:
    # When simulating cornering / braking / bump, react through the 4 wheels/suspension points.
    # To prevent rigid body motion, clamp rear suspension pivot DOFs
    fixed_dofs = []
    for n in (rear_left_lca + rear_right_lca):
        fixed_dofs.extend([3 * n + 0, 3 * n + 1, 3 * n + 2])
    for n in (rear_left_uca + rear_right_uca):
        fixed_dofs.extend([3 * n + 0, 3 * n + 1])

    # 5g Tire Grip Load Cases
    f_total_5g = vehicle_mass * (5.0 * 9.81)  # ~98.1 N total dynamic tire grip

    # 1. 5g Lateral Cornering Load (Side grip acting across front suspension)
    f_lat = np.zeros(n_dofs, dtype=np.float64)
    target_nodes_lat = front_left_lca + front_right_lca + front_left_uca + front_right_uca
    if target_nodes_lat:
        val_y = f_total_5g / len(target_nodes_lat)
        for n in target_nodes_lat:
            f_lat[3 * n + 1] = val_y

    # 2. 5g Longitudinal Braking Load (Front axle braking deceleration pushing rearward)
    f_brake = np.zeros(n_dofs, dtype=np.float64)
    target_nodes_brk = front_left_lca + front_right_lca
    if target_nodes_brk:
        val_x = -f_total_5g / len(target_nodes_brk)
        for n in target_nodes_brk:
            f_brake[3 * n + 0] = val_x

    # 3. 5g Vertical Bump (Kerb strike / high downforce compression on front suspension)
    f_bump = np.zeros(n_dofs, dtype=np.float64)
    target_nodes_bmp = front_left_uca + front_right_uca
    if target_nodes_bmp:
        val_z = f_total_5g / len(target_nodes_bmp)
        for n in target_nodes_bmp:
            f_bump[3 * n + 2] = val_z

    # 4. Torsional Rigidity Load Case (Opposite vertical forces at front left vs front right)
    f_tors = np.zeros(n_dofs, dtype=np.float64)
    if front_left_uca and front_right_uca:
        val_t = (f_total_5g * 0.5) / max(len(front_left_uca), 1)
        for n in front_left_uca:
            f_tors[3 * n + 2] = val_t
        for n in front_right_uca:
            f_tors[3 * n + 2] = -val_t

    load_cases = [f_lat, f_brake, f_bump, f_tors]
    load_weights = [0.35, 0.25, 0.25, 0.15]

    return grid, load_cases, load_weights, fixed_dofs, passive_solid, passive_void


def run_optimization_worker(config: Dict[str, Any]):
    """Background worker executing the topology optimization pipeline."""
    job = GLOBAL_JOB
    start_time = time.time()
    try:
        dim = config.get("dimensions", {"lx": 0.300, "ly": 0.100, "lz": 0.036})
        res = config.get("resolution", {"nx": 40, "ny": 16, "nz": 8})
        mat_id = config.get("material_id", "al6061_t6")
        mat_info = MATERIALS_DB.get(mat_id, MATERIALS_DB["al6061_t6"])
        v_target = float(config.get("target_volume_fraction", 0.28))
        r_filter = float(config.get("filter_radius", 0.008))
        p_simp = float(config.get("simp_penalty", 3.0))
        max_iter = int(config.get("max_iterations", 25))

        cnc_cfg = config.get("cnc_milling", {})
        cnc_enabled = cnc_cfg.get("enabled", True)
        cnc_axis = cnc_cfg.get("axis", "bi-z") if cnc_enabled else None
        cnc_weight = float(cnc_cfg.get("penalty_weight", 0.20)) if cnc_enabled else 0.0

        with job.lock:
            job.status_text = "Building FEA domain, suspension hardpoints, and 5g load cases..."

        grid, load_cases, load_weights, fixed_dofs, p_solid, p_void = build_model_car_chassis_problem(
            lx=dim["lx"],
            ly=dim["ly"],
            lz=dim["lz"],
            nx=res["nx"],
            ny=res["ny"],
            nz=res["nz"],
            mat_id=mat_id,
        )
        job.grid = grid

        top_config = TopologyConfig(
            target_volume_fraction=v_target,
            simp_penalty=p_simp,
            filter_radius=r_filter,
            max_iterations=max_iter,
            cnc_milling_axis=cnc_axis,
            cnc_penalty_weight=cnc_weight,
            enable_heaviside=True,
            heaviside_start_iter=8,
            verbose=False,
        )

        optimizer = TopologyOptimizer(
            grid=grid,
            forces=load_cases,
            fixed_dofs=fixed_dofs,
            config=top_config,
            load_weights=load_weights,
            E=mat_info["E"],
            nu=mat_info["nu"],
            passive_solid=p_solid,
            passive_void=p_void,
        )

        with job.lock:
            job.status_text = "Running Matrix-Free Topology Optimization..."

        # Hook into optimizer iteration loop or run
        opt_res = optimizer.optimize()

        with job.lock:
            job.opt_result = opt_res
            job.compliance_history = list(opt_res.compliance_history)
            job.volume_history = list(opt_res.volume_history)
            job.change_history = list(opt_res.change_history)
            job.current_iteration = opt_res.total_iterations
            job.status_text = "Extracting smoothed watertight 3D isosurface..."

        # Extract 3D Isosurface
        rho_field = opt_res.optimized_densities
        mesh = extract_isosurface_mesh(grid, rho_field, isovalue=0.50, smoothing_iters=5, smoothing_factor=0.35)

        # Simplify mesh vertices for web JSON transfer if large
        verts_list = mesh.vertices.tolist()
        faces_list = mesh.faces.tolist()

        # Save binary STL
        export_dir = os.path.join(tempfile.gettempdir(), "wnfea_gui_exports")
        os.makedirs(export_dir, exist_ok=True)
        stl_file = os.path.join(export_dir, f"model_car_chassis_{mat_id}_3axis.stl")
        mesh.write_stl(stl_file, binary=True)

        # Reconstruct B-Rep STEP solid if FreeCAD is present
        step_file = os.path.join(export_dir, f"model_car_chassis_{mat_id}_3axis.step")
        vtu_file = os.path.join(export_dir, f"model_car_chassis_{mat_id}_3axis.vtu")

        with job.lock:
            job.mesh_vertices = verts_list
            job.mesh_faces = faces_list
            job.final_mesh = mesh
            job.stl_path = stl_file
            job.step_path = step_file
            job.vtu_path = vtu_file
            job.status_text = "Generating FreeCAD OpenCASCADE STEP solid..."

        try:
            reconstructor = BRepReconstructor()
            if reconstructor.freecad_cmd:
                reconstructor.reconstruct_step_solid(mesh=mesh, output_filepath=step_file, tolerance=0.05)
        except Exception as e:
            # Non-fatal if FreeCAD fails on specific complex boundary
            pass

        with job.lock:
            job.is_running = False
            job.is_completed = True
            job.elapsed_time = time.time() - start_time
            job.status_text = f"Optimization complete in {job.elapsed_time:.1f}s. Structure ready for 3-axis CNC machining!"

    except Exception as e:
        with job.lock:
            job.is_running = False
            job.is_completed = False
            job.error_message = str(e)
            job.status_text = f"Error: {e}"
            job.elapsed_time = time.time() - start_time


def run_evolutionary_worker(config: Dict[str, Any]):
    """Background worker executing the 5-candidate evolutionary generative pipeline."""
    job = GLOBAL_JOB
    start_time = time.time()
    try:
        dim = config.get("dimensions", {"lx": 0.300, "ly": 0.100, "lz": 0.036})
        res = config.get("resolution", {"nx": 30, "ny": 12, "nz": 6})
        mat_id = config.get("material_id", "al6061_t6")
        mat_info = MATERIALS_DB.get(mat_id, MATERIALS_DB["al6061_t6"])

        with job.lock:
            job.status_text = "Initializing Evolutionary Problem & 5g Load Envelopes..."

        grid, load_cases, load_weights, fixed_dofs, p_solid, p_void = build_model_car_chassis_problem(
            lx=dim["lx"],
            ly=dim["ly"],
            lz=dim["lz"],
            nx=res["nx"],
            ny=res["ny"],
            nz=res["nz"],
            mat_id=mat_id,
        )
        job.grid = grid

        engine = EvolutionaryTopologyOptimizer(
            grid=grid,
            load_cases=load_cases,
            fixed_dofs=fixed_dofs,
            E=mat_info["E"],
            nu=mat_info["nu"],
            material_density=mat_info["density"],
            passive_solid=p_solid,
            passive_void=p_void,
            inner_iterations=12,
        )

        with job.lock:
            job.status_text = "Synthesizing 5 Diverse Archetypes via Quality-Diversity..."

        run_res = engine.generate_5_diverse_solutions()

        candidates_payload = []
        export_dir = os.path.join(tempfile.gettempdir(), "wnfea_gui_exports")
        os.makedirs(export_dir, exist_ok=True)

        for c in run_res.candidates:
            mesh = extract_isosurface_mesh(grid, c.densities, isovalue=0.50, smoothing_iters=4, smoothing_factor=0.35)
            stl_path = os.path.join(export_dir, f"candidate_{c.id}_{c.archetype_name.replace(' ', '_')}.stl")
            mesh.write_stl(stl_path, binary=True)

            candidates_payload.append({
                "id": c.id,
                "name": c.archetype_name,
                "mass_grams": c.mass_grams,
                "volume_fraction": round(c.achieved_volume_fraction, 4),
                "compliance": round(c.compliance, 6),
                "machinability": c.machinability_score,
                "torsion_score": c.torsional_stiffness_score,
                "seed": c.seed_morphology,
                "vertices": mesh.vertices.tolist(),
                "faces": mesh.faces.tolist(),
                "stl_path": stl_path,
            })

        with job.lock:
            job.candidates_5 = candidates_payload
            job.mesh_vertices = candidates_payload[0]["vertices"]
            job.mesh_faces = candidates_payload[0]["faces"]
            job.stl_path = candidates_payload[0]["stl_path"]
            job.is_running = False
            job.is_completed = True
            job.elapsed_time = time.time() - start_time
            job.status_text = f"5 Diverse Solutions successfully synthesized in {job.elapsed_time:.1f}s!"

    except Exception as e:
        with job.lock:
            job.is_running = False
            job.is_completed = False
            job.error_message = str(e)
            job.status_text = f"Evolutionary Error: {e}"
            job.elapsed_time = time.time() - start_time


class WNFEAHttpHandler(BaseHTTPRequestHandler):
    """
    HTTP Request Handler serving WNFEA Generative Design Web Studio.
    """
    def log_message(self, format, *args):
        # Silence routine access logs in console
        pass

    def _send_json(self, data: Any, status_code: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path in ("", "/"):
            # Serve index.html
            static_dir = os.path.join(os.path.dirname(__file__), "static")
            index_path = os.path.join(static_dir, "index.html")
            if os.path.exists(index_path):
                with open(index_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, "index.html not found")
            return

        if path == "/api/materials":
            self._send_json(MATERIALS_DB)
            return

        if path == "/api/presets":
            self._send_json(PRESETS_DB)
            return

        if path == "/api/status":
            self._send_json(GLOBAL_JOB.get_status())
            return

        if path == "/api/mesh":
            with GLOBAL_JOB.lock:
                if GLOBAL_JOB.mesh_vertices is None:
                    self._send_json({"error": "No mesh available yet"}, status_code=404)
                    return
                # Return mesh data
                data = {
                    "num_vertices": len(GLOBAL_JOB.mesh_vertices),
                    "num_faces": len(GLOBAL_JOB.mesh_faces),
                    "vertices": GLOBAL_JOB.mesh_vertices,
                    "faces": GLOBAL_JOB.mesh_faces,
                    "compliance": GLOBAL_JOB.compliance_history[-1] if GLOBAL_JOB.compliance_history else 0.0,
                    "volume_fraction": GLOBAL_JOB.volume_history[-1] if GLOBAL_JOB.volume_history else 0.0,
                }
            self._send_json(data)
            return

        if path == "/api/candidates":
            with GLOBAL_JOB.lock:
                data = [
                    {k: v for k, v in c.items() if k not in ("vertices", "faces")}
                    for c in GLOBAL_JOB.candidates_5
                ]
            self._send_json(data)
            return

        if path == "/api/candidate_mesh":
            cid = int(query.get("id", ["1"])[0])
            with GLOBAL_JOB.lock:
                cand = next((c for c in GLOBAL_JOB.candidates_5 if c["id"] == cid), None)
                if not cand or "vertices" not in cand:
                    self._send_json({"error": f"Candidate {cid} not found"}, status_code=404)
                    return
                data = {
                    "id": cand["id"],
                    "name": cand["name"],
                    "mass_grams": cand["mass_grams"],
                    "compliance": cand["compliance"],
                    "machinability": cand["machinability"],
                    "vertices": cand["vertices"],
                    "faces": cand["faces"],
                }
            self._send_json(data)
            return

        if path == "/api/export":
            fmt = query.get("format", ["stl"])[0].lower()
            with GLOBAL_JOB.lock:
                if fmt == "stl" and GLOBAL_JOB.stl_path and os.path.exists(GLOBAL_JOB.stl_path):
                    target_file = GLOBAL_JOB.stl_path
                    content_type = "application/sla"
                    filename = "model_car_chassis_6061_3axis.stl"
                elif fmt == "step" and GLOBAL_JOB.step_path and os.path.exists(GLOBAL_JOB.step_path):
                    target_file = GLOBAL_JOB.step_path
                    content_type = "application/step"
                    filename = "model_car_chassis_6061_3axis.step"
                elif fmt == "vtu" and GLOBAL_JOB.vtu_path and os.path.exists(GLOBAL_JOB.vtu_path):
                    target_file = GLOBAL_JOB.vtu_path
                    content_type = "application/octet-stream"
                    filename = "model_car_chassis_6061_3axis.vtu"
                else:
                    self._send_json({"error": f"Requested format '{fmt}' not available"}, status_code=404)
                    return

            with open(target_file, "rb") as fp:
                file_data = fp.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Disposition", f"attachment; filename={filename}")
            self.send_header("Content-Length", str(len(file_data)))
            self.end_headers()
            self.wfile.write(file_data)
            return

        self.send_error(404, f"Path {path} not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/optimize":
            if GLOBAL_JOB.is_running:
                self._send_json({"error": "An optimization job is already running"}, status_code=409)
                return

            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                config_data = json.loads(body.decode("utf-8")) if body else {}
            except Exception:
                config_data = {}

            # Use preset default if empty
            if not config_data:
                config_data = PRESETS_DB["model_car_chassis_300mm"]

            GLOBAL_JOB.reset(config_data)

            # Spawn background worker thread
            worker_thread = threading.Thread(
                target=run_optimization_worker,
                args=(config_data,),
                daemon=True,
            )
            worker_thread.start()

            self._send_json({
                "message": "Optimization job started successfully",
                "status": "running",
                "max_iterations": config_data.get("max_iterations", 25),
            })
            return

        if path == "/api/generate_5_solutions":
            if GLOBAL_JOB.is_running:
                self._send_json({"error": "An optimization job is already running"}, status_code=409)
                return

            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                config_data = json.loads(body.decode("utf-8")) if body else {}
            except Exception:
                config_data = {}

            if not config_data:
                config_data = PRESETS_DB["model_car_chassis_300mm"]

            GLOBAL_JOB.reset(config_data)

            worker_thread = threading.Thread(
                target=run_evolutionary_worker,
                args=(config_data,),
                daemon=True,
            )
            worker_thread.start()

            self._send_json({
                "message": "5-Candidate Evolutionary Generative Studio started successfully",
                "status": "running",
            })
            return

        self.send_error(404, f"POST path {path} not found")


class GUIServer:
    """
    Manages WNFEA GUI HTTP daemon.
    """
    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        self.host = host
        self.port = port
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    def start(self, blocking: bool = False):
        """Start the HTTP server."""
        try:
            self.server = HTTPServer((self.host, self.port), WNFEAHttpHandler)
        except OSError:
            # Try next port if busy
            self.port += 1
            self.server = HTTPServer((self.host, self.port), WNFEAHttpHandler)

        print(f"[WNFEA GUI] Serving interactive Generative Design Studio at: http://{self.host}:{self.port}/")

        if blocking:
            try:
                self.server.serve_forever()
            except KeyboardInterrupt:
                self.stop()
        else:
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()

    def stop(self):
        """Stop the HTTP server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None


def start_gui_server(host: str = "127.0.0.1", port: int = 8080, blocking: bool = True) -> GUIServer:
    """Convenience launcher for the WNFEA GUI."""
    server = GUIServer(host=host, port=port)
    server.start(blocking=blocking)
    return server


if __name__ == "__main__":
    start_gui_server(blocking=True)
