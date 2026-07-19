"""End-to-end integration test for WNFEA backend pipeline."""
import sys
sys.path.insert(0, '.')

from wnfea.geometry.step_parser import STEPParser
from wnfea.model import FEAModel, PropertyAssignment
from wnfea.properties.materials import get_preset_material
from wnfea.properties.sections import create_hollow_tube
from wnfea.boundary.conditions import create_fixed_support, LoadDef
from wnfea.mesh.beam_mesher import mesh_model
from wnfea.solver.linear_static import solve_linear_static
from wnfea.solver.stress import compute_element_stresses
from wnfea.results.result_set import ResultSet

print("=" * 60)
print("WNFEA End-to-End Integration Test")
print("=" * 60)

# Stage 1: Import geometry
print("\n[1] Importing geometry from test_truss.stp...")
parser = STEPParser()
result = parser.parse("test_truss.stp")
model = FEAModel()
model.load_geometry(result)
print(f"    Nodes: {len(model.geometry_nodes)}, Edges: {len(model.geometry_edges)}")
assert len(model.geometry_nodes) == 4, f"Expected 4 nodes, got {len(model.geometry_nodes)}"
assert len(model.geometry_edges) == 6, f"Expected 6 edges, got {len(model.geometry_edges)}"
print("    PASS")

# Stage 2: Assign properties
print("\n[2] Assigning material and section properties...")
steel = get_preset_material("Structural Steel")
model.materials[steel.name] = steel
tube = create_hollow_tube("HT-100x80", 0.10, 0.08)
model.sections[tube.name] = tube
for eid in model.geometry_edges:
    model.edge_assignments[eid] = PropertyAssignment(
        material_name=steel.name,
        section_name=tube.name,
    )
print(f"    Material: {steel.name}, Section: {tube.name}")
print(f"    Assigned to {len(model.edge_assignments)} edges")
print("    PASS")

# Stage 3: Mesh
print("\n[3] Meshing (2 elements per edge)...")
mesh_model(model, elements_per_edge=2)
print(f"    Mesh nodes: {len(model.mesh_nodes)}")
print(f"    Mesh elements: {len(model.mesh_elements)}")
print(f"    Geometry-to-mesh map: {model.geometry_to_mesh_node_map}")
assert len(model.mesh_nodes) > 4, "Should have more nodes than geometry nodes"
assert len(model.mesh_elements) == 12, f"Expected 12 elements (6 edges * 2), got {len(model.mesh_elements)}"
print("    PASS")

# Stage 4: Boundary conditions
print("\n[4] Applying boundary conditions...")
# Fix node 0 (geometry node)
support = create_fixed_support(0, is_geometry_node=True, label="Fixed base")
model.supports.append(support)
# Apply downward force at node 3 (geometry node)
load = LoadDef(node_id=3, is_geometry_node=True, fy=-5000.0, label="Tip load")
model.loads.append(load)
print(f"    Support: {support}")
print(f"    Load: {load}")
print("    PASS")

# Stage 5: Solve
print("\n[5] Solving linear static system...")
U = solve_linear_static(model)
print(f"    DOF vector length: {len(U)}")
print(f"    Max displacement: {max(abs(U)):.6e}")
print("    PASS")

# Stage 6: Post-process stresses
print("\n[6] Computing element stresses...")
element_results = compute_element_stresses(model)
print(f"    Elements processed: {len(element_results)}")
print("    PASS")

# Stage 7: Results
print("\n[7] Extracting results...")
rs = ResultSet(model)
max_node, max_defl = rs.max_deflection()
max_elem, max_vm = rs.max_von_mises()
sf = rs.safety_factor()
print(f"    Max deflection: {max_defl * 1e3:.4f} mm at node {max_node}")
print(f"    Max Von Mises:  {max_vm / 1e6:.3f} MPa at element {max_elem}")
print(f"    Safety Factor:  {sf:.2f}")
print("    PASS")

print("\n" + rs.summary())
print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
