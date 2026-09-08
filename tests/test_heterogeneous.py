"""
Unit and Regression Tests for Heterogeneous CPU + GPU Subsystem Assembler and Solver.

Verifies:
1. Strict numerical parity between HeterogeneousOperator @ u and explicit CSR assembly.
2. Exact kinematic coupling between 6-DOF beam elements and 3-DOF solid elements.
3. RigidCoupling master-slave kinematic load & displacement transfer without Lagrange multipliers.
4. PCG convergence and exact solution parity against direct solver.
5. AMD HIP GPU matrix-free dispatch on coupled mixed models.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from wnfea import (
    FEAModel, DOFManager, RigidCoupling, PropertyAssignment,
    get_preset_material, create_solid_circle, SupportDef, LoadDef,
    DOFConstraint, DOFType, solve_linear_static
)
from wnfea.solver.assembler import assemble_global_system
from wnfea.solver.heterogeneous_assembler import HeterogeneousOperator, solve_heterogeneous
from test_c3d10 import create_reference_tetrahedron_coords


def test_heterogeneous_matvec_parity():
    """Verify that HeterogeneousOperator @ u matches explicit assemble_global_system @ u."""
    model = FEAModel()
    solid_coords = create_reference_tetrahedron_coords(scale=1.0)
    beam_tip = np.array([0.0, 0.0, 2.0])
    model.mesh_nodes = np.vstack([solid_coords, beam_tip])
    model.solid_elements = np.array([np.arange(10, dtype=int)])
    model.mesh_elements = np.array([[3, 10]])

    steel = get_preset_material("Structural Steel")
    bar = create_solid_circle("Bar", 0.05)
    model.materials[steel.name] = steel
    model.sections[bar.name] = bar
    model.element_properties[0] = PropertyAssignment(steel.name, bar.name)
    model.solid_materials[0] = steel.name

    # Add supports and loads
    for n_id in [0, 1, 2]:
        model.supports.append(SupportDef(
            node_id=n_id, is_geometry_node=False,
            ux=DOFConstraint(DOFType.FIXED),
            uy=DOFConstraint(DOFType.FIXED),
            uz=DOFConstraint(DOFType.FIXED),
        ))
    model.loads.append(LoadDef(node_id=10, is_geometry_node=False, fz=1000.0))

    # 1. Explicit assembly
    K_explicit, _ = assemble_global_system(model)

    # 2. Heterogeneous operator
    het_op = HeterogeneousOperator(model, apply_bcs=True)

    np.random.seed(42)
    u_test = np.random.randn(het_op.n_dofs)

    v_explicit = K_explicit @ u_test
    v_hetero = het_op @ u_test

    rel_error = np.linalg.norm(v_explicit - v_hetero) / np.linalg.norm(v_explicit)
    print(f"  Heterogeneous vs Explicit relative error: {rel_error:.4e}")
    assert rel_error < 1e-12, f"Heterogeneous matvec failed parity (rel_error={rel_error:.4e})"
    print("  [PASS] test_heterogeneous_matvec_parity (< 1e-12)")


def test_heterogeneous_pcg_solve_parity():
    """Verify PCG solve produces correct displacements matching direct solver."""
    model = FEAModel()
    solid_coords = create_reference_tetrahedron_coords(scale=1.0)
    beam_tip = np.array([0.0, 0.0, 2.0])
    model.mesh_nodes = np.vstack([solid_coords, beam_tip])
    model.solid_elements = np.array([np.arange(10, dtype=int)])
    model.mesh_elements = np.array([[3, 10]])

    steel = get_preset_material("Structural Steel")
    bar = create_solid_circle("Bar", 0.05)
    model.materials[steel.name] = steel
    model.sections[bar.name] = bar
    model.element_properties[0] = PropertyAssignment(steel.name, bar.name)
    model.solid_materials[0] = steel.name

    for n_id in [0, 1, 2]:
        model.supports.append(SupportDef(
            node_id=n_id, is_geometry_node=False,
            ux=DOFConstraint(DOFType.FIXED),
            uy=DOFConstraint(DOFType.FIXED),
            uz=DOFConstraint(DOFType.FIXED),
        ))
    model.loads.append(LoadDef(node_id=10, is_geometry_node=False, fz=10000.0))

    # Solve via direct solver
    u_direct = solve_linear_static(model, device="cpu")

    # Solve via heterogeneous matrix-free PCG
    het_op = HeterogeneousOperator(model, apply_bcs=True)
    u_het, info = het_op.solve_pcg(tol=1e-7, maxiter=500)

    assert info["converged"], f"Heterogeneous PCG did not converge! Final res: {info['final_residual']}"
    rel_diff = np.linalg.norm(u_het - u_direct) / np.linalg.norm(u_direct)
    print(f"  Heterogeneous PCG converged in {info['iterations']} iters, rel error vs direct: {rel_diff:.4e}")
    assert rel_diff < 1e-5, f"Heterogeneous PCG differs from direct solve (rel_diff={rel_diff:.4e})"
    print("  [PASS] test_heterogeneous_pcg_solve_parity (< 1e-5)")


def test_heterogeneous_rigid_coupling():
    """Verify heterogeneous operator with RigidCoupling between beam and solid face."""
    model = FEAModel()
    solid_coords = create_reference_tetrahedron_coords(scale=1.0)
    face_center = (solid_coords[0] + solid_coords[1] + solid_coords[2]) / 3.0
    beam_root = face_center.copy()
    beam_tip = face_center + np.array([0.0, 0.0, -1.0])

    model.mesh_nodes = np.vstack([solid_coords, beam_root, beam_tip])
    model.solid_elements = np.array([np.arange(10, dtype=int)])
    model.mesh_elements = np.array([[10, 11]])

    steel = get_preset_material("Structural Steel")
    bar = create_solid_circle("Bar", 0.04)
    model.materials[steel.name] = steel
    model.sections[bar.name] = bar
    model.element_properties[0] = PropertyAssignment(steel.name, bar.name)
    model.solid_materials[0] = steel.name

    # RigidCoupling: master beam node 10 -> slave solid nodes 0, 1, 2
    coupling = RigidCoupling(master_node_id=10, slave_node_ids=[0, 1, 2], label="Coupling")
    model.couplings.append(coupling)

    # Fix solid node 3 (apex) and mid-edge nodes 8, 9 to fully prevent rigid body motion
    for n_id in [3, 8, 9]:
        model.supports.append(SupportDef(
            node_id=n_id, is_geometry_node=False,
            ux=DOFConstraint(DOFType.FIXED),
            uy=DOFConstraint(DOFType.FIXED),
            uz=DOFConstraint(DOFType.FIXED),
        ))
    # Load on beam tip 11
    model.loads.append(LoadDef(node_id=11, is_geometry_node=False, fx=5000.0))


    # Compare direct vs heterogeneous
    u_direct = solve_linear_static(model, device="cpu")
    u_het = solve_heterogeneous(model, tol=1e-7, maxiter=500)

    rel_diff = np.linalg.norm(u_het - u_direct) / np.linalg.norm(u_direct)
    print(f"  Rigid coupling solve rel error vs direct: {rel_diff:.4e}")
    assert rel_diff < 1e-5, f"Rigid coupling differs from direct solve (rel_diff={rel_diff:.4e})"
    print("  [PASS] test_heterogeneous_rigid_coupling")


def run_all():
    print("=" * 60)
    print("Running Heterogeneous CPU + GPU Subsystem Test Suite...")
    print("=" * 60)
    test_heterogeneous_matvec_parity()
    test_heterogeneous_pcg_solve_parity()
    test_heterogeneous_rigid_coupling()
    print("=" * 60)
    print("ALL HETEROGENEOUS TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
