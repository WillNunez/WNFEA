"""
WNFEA End-to-End Pipeline Test Suite
=====================================

Runs two tests:

1. **Cantilever beam (analytical validation)**
   - 2-node beam along the X axis, fixed at node 0, point load at node 1.
   - Compares tip deflection against the analytical Euler–Bernoulli formula
     δ = PL³ / 3EI  (should match to < 1% error).

2. **Tetrahedral truss (STEP file integration test)**
   - Parses ``test_truss.stp`` (4 nodes, 6 edges).
   - Assigns Structural Steel + hollow tube to all edges.
   - Meshes with 2 divisions per edge.
   - Fixes node 0, applies -1000 N in Y at node 3.
   - Verifies that the solver converges, stresses are > 0, reactions balance.

Usage::

    python test_pipeline.py
"""

import sys
import os
import numpy as np

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wnfea import (
    FEAModel, PipelineStage, PropertyAssignment,
    STEPParser,
    MaterialDef, get_preset_material, SectionDef, create_hollow_tube, create_solid_circle,
    SupportDef, LoadDef, DOFConstraint, DOFType, create_fixed_support,
    BeamMesher,
    BeamSolver,
    PostProcessor,
)
from wnfea.geometry.primitives import Point3D, GeometryNode, GeometryEdge


# ═══════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════

class TestResult:
    """Simple test outcome tracker."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors: list[str] = []

    def check(self, condition: bool, message: str):
        if condition:
            self.passed += 1
            print(f"  [PASS] {message}")
        else:
            self.failed += 1
            self.errors.append(message)
            print(f"  [FAIL] {message}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'=' * 60}")
        print(f"Results: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print("Failures:")
            for e in self.errors:
                print(f"  - {e}")
        print(f"{'=' * 60}")
        return self.failed == 0


# ═══════════════════════════════════════════════════════════════════
# Test 1: Cantilever Beam — Analytical Validation
# ═══════════════════════════════════════════════════════════════════

def test_cantilever_beam(tr: TestResult):
    print("\n" + "-" * 60)
    print("TEST 1: Cantilever Beam -- Analytical Validation")
    print("-" * 60)

    model = FEAModel()
    tr.check(model.stage == PipelineStage.EMPTY, "Initial stage is EMPTY")

    # Build geometry manually (no STEP file needed)
    L = 1.0  # 1 metre beam
    model.geometry_nodes = {
        0: GeometryNode(id=0, label="Fixed", point=Point3D(0.0, 0.0, 0.0)),
        1: GeometryNode(id=1, label="Free",  point=Point3D(L, 0.0, 0.0)),
    }
    model.geometry_edges = {
        0: GeometryEdge(id=0, label="Beam", start_node_id=0, end_node_id=1),
    }
    tr.check(model.stage == PipelineStage.GEOMETRY_LOADED, "Stage after geometry = GEOMETRY_LOADED")

    # Assign properties
    steel = get_preset_material("Structural Steel")
    model.materials[steel.name] = steel
    tr.check(len(steel.validate()) == 0, "Steel material validates OK")

    section = create_solid_circle("Test Rod", radius=0.05)
    model.sections[section.name] = section
    tr.check(len(section.validate()) == 0, "Section validates OK")

    model.edge_assignments[0] = PropertyAssignment(
        material_name=steel.name,
        section_name=section.name,
    )
    tr.check(model.stage == PipelineStage.PROPERTIES_ASSIGNED, "Stage after assignment = PROPERTIES_ASSIGNED")

    # Mesh (4 elements for better accuracy)
    mesher = BeamMesher(n_divisions=4)
    mesher.mesh(model)
    tr.check(model.mesh_nodes is not None, "Mesh nodes created")
    tr.check(model.mesh_elements is not None, "Mesh elements created")
    tr.check(len(model.mesh_nodes) == 5, f"Expected 5 mesh nodes, got {len(model.mesh_nodes)}")
    tr.check(len(model.mesh_elements) == 4, f"Expected 4 mesh elements, got {len(model.mesh_elements)}")
    tr.check(model.stage == PipelineStage.MESHED, "Stage after meshing = MESHED")

    # Boundary conditions
    model.supports.append(create_fixed_support(node_id=0, is_geometry_node=True, label="Wall"))
    P = -1000.0  # 1000 N downward (Y direction)
    model.loads.append(LoadDef(node_id=1, is_geometry_node=True, fy=P, label="Tip load"))
    tr.check(model.stage == PipelineStage.BCS_DEFINED, "Stage after BCs = BCS_DEFINED")

    # Solve
    solver = BeamSolver()
    solver.solve(model)
    tr.check(model.displacements is not None, "Displacements computed")
    tr.check(model.stage == PipelineStage.SOLVED, "Stage after solve = SOLVED")

    # Post-process
    pp = PostProcessor()
    results = pp.process(model)
    tr.check(results.max_von_mises > 0, f"Max VM stress = {results.max_von_mises:.3e} Pa (> 0)")

    # Analytical cantilever deflection: δ = PL³ / 3EI
    E = steel.youngs_modulus
    I = section.iy  # circular → Iy == Iz
    delta_analytical = abs(P) * L**3 / (3 * E * I)

    # Find the tip node mesh index
    tip_mesh_id = model.geometry_to_mesh_node_map[1]
    tip_disp_y = abs(model.displacements[tip_mesh_id * 6 + 1])

    error_pct = abs(tip_disp_y - delta_analytical) / delta_analytical * 100

    print(f"\n  Analytical deflection:  {delta_analytical:.6e} m")
    print(f"  FEA tip deflection:    {tip_disp_y:.6e} m")
    print(f"  Error:                 {error_pct:.4f}%")

    tr.check(error_pct < 1.0, f"Deflection error < 1% (got {error_pct:.4f}%)")

    # Verify reactions balance applied load
    if model.reaction_forces is not None:
        Ry = model.reaction_forces[0 * 6 + 1]  # Y reaction at node 0
        load_balance_err = abs(Ry - (-P)) / abs(P) * 100
        print(f"  Reaction Fy at fixed:  {Ry:.3f} N  (expected {-P:.3f} N)")
        print(f"  Reaction balance err:  {load_balance_err:.4f}%")
        tr.check(load_balance_err < 1.0, f"Reaction force balance < 1% (got {load_balance_err:.4f}%)")

    # Safety factor check
    tr.check(results.safety_factor is not None and results.safety_factor > 0,
             f"Safety factor = {results.safety_factor:.3f}")

    print(f"\n{results.summary()}")


# ═══════════════════════════════════════════════════════════════════
# Test 2: Tetrahedral Truss — STEP File Integration
# ═══════════════════════════════════════════════════════════════════

def test_truss_from_step(tr: TestResult):
    print("\n" + "-" * 60)
    print("TEST 2: Tetrahedral Truss -- STEP File Integration")
    print("-" * 60)

    step_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_truss.stp")
    if not os.path.exists(step_path):
        print(f"  SKIP: {step_path} not found")
        return

    model = FEAModel()

    # Stage 1: Parse geometry
    parser = STEPParser()
    parse_result = parser.parse(step_path)
    model.load_geometry(parse_result)

    tr.check(len(model.geometry_nodes) == 4, f"Expected 4 nodes, got {len(model.geometry_nodes)}")
    tr.check(len(model.geometry_edges) == 6, f"Expected 6 edges, got {len(model.geometry_edges)}")
    tr.check(model.stage == PipelineStage.GEOMETRY_LOADED, "Stage = GEOMETRY_LOADED")

    print(f"  Nodes: {len(model.geometry_nodes)}, Edges: {len(model.geometry_edges)}")
    for nid, node in model.geometry_nodes.items():
        print(f"    Node {nid}: ({node.point.x:.3f}, {node.point.y:.3f}, {node.point.z:.3f})")

    # Stage 2: Assign properties to all edges
    steel = get_preset_material("Structural Steel")
    model.materials[steel.name] = steel

    tube = create_hollow_tube("Standard Tube", outer_radius=0.05, inner_radius=0.04)
    model.sections[tube.name] = tube

    for eid in model.geometry_edges:
        model.edge_assignments[eid] = PropertyAssignment(
            material_name=steel.name,
            section_name=tube.name,
        )
    tr.check(model.stage == PipelineStage.PROPERTIES_ASSIGNED, "Stage = PROPERTIES_ASSIGNED")

    # Stage 3: Mesh with 2 divisions per edge
    mesher = BeamMesher(n_divisions=2)
    mesher.mesh(model)

    n_expected_nodes = 4 + 6 * 1  # 4 geometry + 6 interior (1 per edge with 2 div)
    n_expected_elems = 6 * 2      # 6 edges × 2 divisions

    tr.check(len(model.mesh_nodes) == n_expected_nodes,
             f"Expected {n_expected_nodes} mesh nodes, got {len(model.mesh_nodes)}")
    tr.check(len(model.mesh_elements) == n_expected_elems,
             f"Expected {n_expected_elems} mesh elements, got {len(model.mesh_elements)}")
    tr.check(model.stage == PipelineStage.MESHED, "Stage = MESHED")

    # Stage 4: Boundary conditions
    # Fix node 0 (all DOFs)
    model.supports.append(create_fixed_support(node_id=0, is_geometry_node=True, label="Ground"))
    # Also fix node 1 to prevent rigid body motion
    model.supports.append(create_fixed_support(node_id=1, is_geometry_node=True, label="Ground 2"))
    # Fix node 2 as well for a fully constrained base triangle
    model.supports.append(create_fixed_support(node_id=2, is_geometry_node=True, label="Ground 3"))

    # Apply load at node 3 (the apex)
    model.loads.append(LoadDef(node_id=3, is_geometry_node=True, fy=-1000.0, label="Apex load"))
    tr.check(model.stage == PipelineStage.BCS_DEFINED, "Stage = BCS_DEFINED")

    # Stage 5: Solve
    solver = BeamSolver()
    try:
        solver.solve(model)
        solve_ok = True
    except Exception as e:
        solve_ok = False
        print(f"  Solver error: {e}")
    tr.check(solve_ok, "Solver converged successfully")

    if not solve_ok:
        return

    tr.check(model.displacements is not None, "Displacements computed")
    tr.check(model.stage == PipelineStage.SOLVED, "Stage = SOLVED")

    # Post-process
    pp = PostProcessor()
    results = pp.process(model)

    tr.check(results.max_von_mises > 0, f"Max VM = {results.max_von_mises:.3e} Pa (> 0)")
    tr.check(results.max_displacement > 0, f"Max disp = {results.max_displacement:.6e} m (> 0)")

    # Verify reaction forces sum to applied load
    if model.reaction_forces is not None:
        total_Fy_reaction = 0.0
        for support in model.supports:
            mesh_id = model.geometry_to_mesh_node_map.get(support.node_id)
            if mesh_id is not None:
                total_Fy_reaction += model.reaction_forces[mesh_id * 6 + 1]

        expected_load = -1000.0
        balance_err = abs(total_Fy_reaction - (-expected_load))
        print(f"  Total Fy reaction: {total_Fy_reaction:.3f} N (expected {-expected_load:.3f} N)")
        print(f"  Balance error:     {balance_err:.6f} N")
        tr.check(balance_err < 1.0, f"Reaction force Y-balance error < 1 N (got {balance_err:.6f} N)")

    print(f"\n{results.summary()}")
    print(f"\n{model.summary()}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("         WNFEA -- Full Pipeline Test Suite")
    print("=" * 60)

    tr = TestResult()

    test_cantilever_beam(tr)
    test_truss_from_step(tr)

    all_ok = tr.summary()
    sys.exit(0 if all_ok else 1)
