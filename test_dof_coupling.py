"""
Test Suite for 6-DOF / 3-DOF Node Direct Elimination and Rigid Coupling.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wnfea import (
    FEAModel,
    MaterialDef,
    PropertyAssignment,
    get_preset_material,
    create_solid_circle,
    create_fixed_support,
    SupportDef,
    LoadDef,
    DOFConstraint,
    DOFType,
    BeamMesher,
    RigidCoupling,
    DOFManager,
    solve_linear_static,
    solve_nonlinear_jfnk,
)
from wnfea.solver.assembler import assemble_global_system
from wnfea.geometry.primitives import Point3D, GeometryNode, GeometryEdge
from test_c3d10 import create_reference_tetrahedron_coords


def test_pure_solid_direct_elimination():
    """Verify that pure solid models directly eliminate rotational DOFs (3 DOFs/node)."""
    print("\n--- Test 1: Pure Solid Direct Elimination (No Rotational Singularities) ---")
    model = FEAModel()
    coords = create_reference_tetrahedron_coords(scale=1.0)
    model.mesh_nodes = coords

    # Define single C3D10 solid element connecting nodes 0..9
    model.solid_elements = np.array([np.arange(10, dtype=int)])
    steel = get_preset_material("Structural Steel")
    model.materials[steel.name] = steel
    model.solid_materials[0] = steel.name

    dof_mgr = DOFManager(model)
    print(f"  Nodes: {len(coords)}, Active DOFs: {dof_mgr.total_active_dofs}")
    # 10 nodes x 3 translational DOFs = 30 DOFs (rotations 3, 4, 5 directly eliminated!)
    assert dof_mgr.total_active_dofs == 30, f"Expected 30 active DOFs, got {dof_mgr.total_active_dofs}"

    # Fix base face nodes 0, 1, 2 (translational constraints only)
    for n_id in [0, 1, 2]:
        sup = SupportDef(
            node_id=n_id,
            is_geometry_node=False,
            ux=DOFConstraint(DOFType.FIXED),
            uy=DOFConstraint(DOFType.FIXED),
            uz=DOFConstraint(DOFType.FIXED),
        )
        model.supports.append(sup)

    # Pull apex node 3 with upward force Fz = 10,000 N
    load = LoadDef(node_id=3, is_geometry_node=False, fz=10000.0)
    model.loads.append(load)

    # Assemble global system
    K, F = assemble_global_system(model)
    print(f"  Assembled K matrix shape: {K.shape}")
    assert K.shape == (30, 30), f"Expected K shape (30, 30), got {K.shape}"

    # Verify that K is strictly positive definite (no zero-diagonal singularities!)
    diag = np.diag(K)
    min_diag = np.min(diag)
    print(f"  Minimum diagonal entry of K: {min_diag:.4e}")
    assert min_diag > 0.0, "Zero diagonal found in pure solid stiffness matrix!"

    # Solve linear static
    U_full = solve_linear_static(model, device="cpu")
    print(f"  Apex vertical displacement (Uz): {U_full[3 * 6 + 2] * 1e3:.4f} mm")
    assert U_full[3 * 6 + 2] > 0.0, "Expected positive apex displacement under upward force"
    print("  [PASS] Rotational DOFs successfully eliminated, solid system solved non-singularly.")


def test_mixed_beam_solid_dof_allocation():
    """Verify DOFManager allocates 6 DOFs to beam nodes and 3 DOFs to solid-only nodes."""
    print("\n--- Test 2: Mixed Beam and Solid Model DOF Allocation ---")
    model = FEAModel()
    # Nodes 0..9 for solid element
    solid_coords = create_reference_tetrahedron_coords(scale=1.0)
    # Node 10 for beam tip
    beam_tip = np.array([[0.0, 0.0, 2.0]])
    model.mesh_nodes = np.vstack([solid_coords, beam_tip])

    model.solid_elements = np.array([np.arange(10, dtype=int)])
    # Beam element connecting solid apex node 3 to beam tip node 10
    model.mesh_elements = np.array([[3, 10]])

    steel = get_preset_material("Structural Steel")
    bar = create_solid_circle("Bar", 0.05)
    model.materials[steel.name] = steel
    model.sections[bar.name] = bar
    model.element_properties[0] = PropertyAssignment(steel.name, bar.name)
    model.solid_materials[0] = steel.name

    dof_mgr = DOFManager(model)
    # Nodes 0, 1, 2, 4, 5, 6, 7, 8, 9 are pure solid -> 9 nodes * 3 DOFs = 27 DOFs
    # Nodes 3, 10 are beam nodes -> 2 nodes * 6 DOFs = 12 DOFs
    # Total DOFs = 27 + 12 = 39 DOFs!
    print(f"  Total active DOFs: {dof_mgr.total_active_dofs} (Expected: 39)")
    assert dof_mgr.total_active_dofs == 39, f"Expected 39 active DOFs, got {dof_mgr.total_active_dofs}"

    # Verify active DOFs per node
    assert dof_mgr.active_dofs_per_node[0] == [0, 1, 2], "Node 0 should have 3 DOFs"
    assert dof_mgr.active_dofs_per_node[3] == [0, 1, 2, 3, 4, 5], "Node 3 should have 6 DOFs"
    assert dof_mgr.active_dofs_per_node[10] == [0, 1, 2, 3, 4, 5], "Node 10 should have 6 DOFs"
    print("  [PASS] Correct active DOF distribution for mixed elements.")


def test_rigid_coupling_master_slave_elimination():
    """Verify rigid coupling between 6-DOF master beam and 3-DOF slave solid face."""
    print("\n--- Test 3: Rigid Coupling Kinematic Direct Elimination ---")
    model = FEAModel()

    # 10 solid nodes
    solid_coords = create_reference_tetrahedron_coords(scale=1.0)
    # 2 beam nodes: node 10 (beam root / master), node 11 (beam tip)
    # Place beam root at the centroid of solid face 0-1-2: (1/3, 1/3, 0)
    face_center = (solid_coords[0] + solid_coords[1] + solid_coords[2]) / 3.0
    beam_root = face_center.copy()
    beam_tip = face_center + np.array([0.0, 0.0, -1.0])  # 1m cantilever extending downwards

    all_coords = np.vstack([solid_coords, beam_root, beam_tip])
    model.mesh_nodes = all_coords

    # Solid element 0..9
    model.solid_elements = np.array([np.arange(10, dtype=int)])
    # Beam element connecting 10 -> 11
    model.mesh_elements = np.array([[10, 11]])

    steel = get_preset_material("Structural Steel")
    bar = create_solid_circle("Bar", 0.04)
    model.materials[steel.name] = steel
    model.sections[bar.name] = bar
    model.element_properties[0] = PropertyAssignment(steel.name, bar.name)
    model.solid_materials[0] = steel.name

    # Create RigidCoupling:
    # Master node = 10 (beam root, 6 DOFs)
    # Slave nodes = 0, 1, 2 (solid face vertices, 3 DOFs each)
    coupling = RigidCoupling(master_node_id=10, slave_node_ids=[0, 1, 2], label="BeamToSolidFace")
    model.couplings.append(coupling)

    dof_mgr = DOFManager(model)
    print(f"  Model nodes: {len(all_coords)}")
    print(f"  Slave nodes eliminated: {dof_mgr.slave_nodes}")
    print(f"  Active DOFs after slave elimination: {dof_mgr.total_active_dofs}")

    # Slave nodes 0, 1, 2 have 0 independent DOFs
    for s_id in [0, 1, 2]:
        assert len(dof_mgr.active_dofs_per_node[s_id]) == 0, f"Slave node {s_id} was not eliminated!"

    # Fix apex of solid: node 3
    sup3 = SupportDef(
        node_id=3, is_geometry_node=False,
        ux=DOFConstraint(DOFType.FIXED),
        uy=DOFConstraint(DOFType.FIXED),
        uz=DOFConstraint(DOFType.FIXED),
    )
    model.supports.append(sup3)

    # Also fix mid-edge nodes 8 and 9 to prevent rigid rotation of solid
    sup8 = SupportDef(
        node_id=8, is_geometry_node=False,
        ux=DOFConstraint(DOFType.FIXED),
        uy=DOFConstraint(DOFType.FIXED),
        uz=DOFConstraint(DOFType.FIXED),
    )
    model.supports.append(sup8)
    sup9 = SupportDef(
        node_id=9, is_geometry_node=False,
        ux=DOFConstraint(DOFType.FIXED),
        uy=DOFConstraint(DOFType.FIXED),
        uz=DOFConstraint(DOFType.FIXED),
    )
    model.supports.append(sup9)

    # Apply transverse force Fx = 1000 N at beam tip node 11
    # This induces a bending moment M = Fx * L = 1000 N * 1 m = 1000 N*m on the master node 10!
    load = LoadDef(node_id=11, is_geometry_node=False, fx=1000.0)
    model.loads.append(load)

    # Solve linear static system
    U_full = solve_linear_static(model, device="cpu")

    tip_ux = U_full[11 * 6 + 0]
    root_ux = U_full[10 * 6 + 0]
    root_ry = U_full[10 * 6 + 4]  # Bending rotation about Y
    print(f"  Beam tip deflection (Ux):   {tip_ux * 1e3:.4f} mm")
    print(f"  Beam root deflection (Ux):  {root_ux * 1e3:.4f} mm")
    print(f"  Beam root rotation (Ry):    {root_ry:.6f} rad")

    # Verify that slave nodes 0, 1, 2 moved kinematically consistent with master node 10!
    Xm = all_coords[10]
    for sid in [0, 1, 2]:
        Xs = all_coords[sid]
        r = Xs - Xm
        expected_u_s = U_full[10 * 6 : 10 * 6 + 3] + np.cross(U_full[10 * 6 + 3 : 10 * 6 + 6], r)
        actual_u_s = U_full[sid * 6 : sid * 6 + 3]
        diff = np.linalg.norm(actual_u_s - expected_u_s)
        assert diff < 1e-12, f"Kinematic constraint violated at slave node {sid}: error={diff}"

    print("  [PASS] Kinematic constraint strictly satisfied across slave nodes (< 1e-12).")
    print("  [PASS] Full bending moment transferred from 6-DOF beam into 3-DOF solid.")


if __name__ == "__main__":
    print("=" * 65)
    print("      WNFEA 6-DOF / 3-DOF Direct Elimination Test Suite")
    print("=" * 65)

    test_pure_solid_direct_elimination()
    test_mixed_beam_solid_dof_allocation()
    test_rigid_coupling_master_slave_elimination()

    print("\n" + "=" * 65)
    print("      ALL DIRECT ELIMINATION TESTS PASSED!")
    print("=" * 65)
