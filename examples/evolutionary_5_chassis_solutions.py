"""
WNFEA Evolutionary Generative Studio: 5 Distinct 300mm Model Car Chassis Solutions
-----------------------------------------------------------------------------------
Demonstrates the Autodesk Generative Design / Project Dreamcatcher paradigm:
1. Formulates the 300mm Aluminum 6061-T6 Model Car Chassis with double wishbone
   suspension hardpoints and 4 dynamic 5g tire grip load cases.
2. Runs the Bi-Level Evolutionary & Quality-Diversity Engine.
3. Automatically synthesizes and converges 5 distinct structural candidates:
   - Candidate 1: Torsional Rigidity Specialist (heavy X-truss bracing)
   - Candidate 2: 3-Axis CNC Production Champion (fast machining, zero undercuts)
   - Candidate 3: Ultra-Lightweight Minimalist (aggressive mass cut, V* <= 23%)
   - Candidate 4: Perimeter-Sill Impact Frame (reinforced side rails, open center)
   - Candidate 5: Balanced 5g All-Rounder (Pareto multi-load dynamic compromise)
4. Exports watertight binary STLs, STEP solids, and comparison telemetry.
"""

from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wnfea.gui.server import build_model_car_chassis_problem
from wnfea.opt.evolutionary import EvolutionaryTopologyOptimizer, EvolutionaryRunResult
from wnfea.cad.isosurface import extract_isosurface_mesh
from wnfea.cad.brep_reconstruction import BRepReconstructor
from wnfea.cad.freecad_brep import CylindricalFeature


def run_evolutionary_chassis_studio(
    output_dir: str = "output/model_car_chassis_5_solutions",
    resolution: tuple[int, int, int] = (30, 12, 6),
    inner_iterations: int = 15,
) -> dict:
    print("=" * 85)
    print("   WNFEA EVOLUTIONARY GENERATIVE STUDIO: 5 DIVERSE 300mm CHASSIS CANDIDATES")
    print("=" * 85)
    t0 = time.perf_counter()

    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Build Chassis Design Space with Double Wishbone Pickups
    print("\n[Stage 1/4] Formulating Chassis Design Envelope & 5g Tire Grip Envelope...")
    grid, load_cases, load_weights, fixed_dofs, p_solid, p_void = build_model_car_chassis_problem(
        lx=0.300,
        ly=0.100,
        lz=0.036,
        nx=resolution[0],
        ny=resolution[1],
        nz=resolution[2],
        mat_id="al6061_t6",
        vehicle_mass=2.0,
    )
    print(f"  Voxel Mesh: {resolution[0]}x{resolution[1]}x{resolution[2]} ({grid.total_cells} cells, {grid.total_nodes*3} DOFs)")
    print(f"  Hardpoints: 16 double wishbone pivot lugs preserved as solid non-design zones")
    print(f"  Clearance:  Central LiPo battery & 540 brushless motor bay preserved as void")

    # 2. Run Bi-Level Evolutionary Topology Engine
    print("\n[Stage 2/4] Executing Bi-Level Evolutionary & Quality-Diversity Loop...")
    engine = EvolutionaryTopologyOptimizer(
        grid=grid,
        load_cases=load_cases,
        fixed_dofs=fixed_dofs,
        E=68.9e9,
        nu=0.33,
        material_density=2700.0,
        passive_solid=p_solid,
        passive_void=p_void,
        inner_iterations=inner_iterations,
    )

    run_res = engine.generate_5_diverse_solutions()
    print(f"  Evaluated {run_res.total_evaluations} candidates in {run_res.total_time:.2f}s "
          f"({run_res.total_time/run_res.total_evaluations*1000:.1f} ms/candidate)")

    # 3. Print Comparison Table
    print("\n[Stage 3/4] Generative Design Comparison Matrix (5 Distinct Solutions):")
    print("-" * 115)
    print(f"{'ID':<3} | {'Archetype':<32} | {'Mass (g)':<9} | {'Mass Cut':<9} | {'Compliance (J)':<15} | {'Machinability':<14} | {'Seed Type':<15}")
    print("-" * 115)
    for c in run_res.summary_table:
        print(f"{c['candidate_id']:<3} | {c['archetype']:<32} | {c['mass_grams']:<9.1f} | {c['mass_reduction_percent']:<8.1f}% | {c['compliance_joules']:<15.6f} | {c['machinability_score_percent']:<13.1f}% | {c['seed_morphology']:<15}")
    print("-" * 115)

    # 4. Export Solid Meshes (STL & STEP) for All 5 Candidates
    print("\n[Stage 4/4] Exporting Watertight Solid Models for All 5 Candidates...")
    reconstructor = BRepReconstructor()
    candidate_artifacts = []

    for c in run_res.candidates:
        prefix = f"candidate_{c.id}_{c.archetype_name.lower().replace(' ', '_').replace('-', '_')}"
        stl_path = str(out_path / f"{prefix}.stl")
        step_path = str(out_path / f"{prefix}.step")

        # Extract smoothed watertight isosurface
        mesh = extract_isosurface_mesh(grid, c.densities, isovalue=0.50, smoothing_iters=4, smoothing_factor=0.35)
        mesh.write_stl(stl_path, binary=True)
        stl_size_kb = round(os.path.getsize(stl_path) / 1024, 1)

        # Attempt STEP reconstruction
        has_step = False
        try:
            if reconstructor.freecad_cmd:
                reconstructor.reconstruct_step_solid(mesh=mesh, output_filepath=step_path, tolerance=0.05)
                has_step = os.path.exists(step_path)
        except Exception:
            pass

        print(f"  Candidate {c.id}: [STL: {stl_size_kb} KB] -> {os.path.basename(stl_path)}")
        if has_step:
            print(f"               [STEP: {round(os.path.getsize(step_path)/1024, 1)} KB] -> {os.path.basename(step_path)}")

        candidate_artifacts.append({
            "candidate_id": c.id,
            "archetype": c.archetype_name,
            "stl_filepath": stl_path,
            "step_filepath": step_path if has_step else None,
            "mass_grams": c.mass_grams,
            "compliance_joules": c.compliance,
            "machinability_score_percent": c.machinability_score,
            "torsional_score": c.torsional_stiffness_score,
        })

    # Save summary JSON
    summary_path = str(out_path / "chassis_5_solutions_summary.json")
    summary_data = {
        "case_study": "300mm Model Car Chassis Evolutionary Generative Studio",
        "material": "Aluminum 6061-T6",
        "total_solve_time_seconds": round(time.perf_counter() - t0, 2),
        "diversity_matrix": run_res.diversity_matrix.tolist(),
        "candidates": candidate_artifacts,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n  Exported Master Summary JSON: {summary_path}")

    print("\n" + "=" * 85)
    print(f"   5 DIVERSE GENERATIVE CHASSIS CANDIDATES SUCCESSFULLY SYNTHESIZED ({summary_data['total_solve_time_seconds']}s)")
    print("=" * 85)
    return summary_data


if __name__ == "__main__":
    run_evolutionary_chassis_studio()
