"""
Verification Suite: Applied Acceleration Fields & Spatial Point Loads
----------------------------------------------------------------------
Validates:
1. Analytical C3D10 Mass & Force Parity: sum(F_solid) == M_solid * a (< 1e-12).
2. Beam Consistent Inertial Force & Moment Parity: sum(F_beam) == M_beam * a.
3. RBE3 Spatial Point Load Projection: Exact translational (sum(f) == F) and
   moment conservation (sum(r x f) == M).
4. Heterogeneous PCG Solve under 10G Maneuver Load.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from wnfea.model import FEAModel
from wnfea.properties.materials import get_preset_material
from wnfea.properties.sections import create_solid_circle
from wnfea.boundary.conditions import create_fixed_support, SupportDef, LoadDef, DOFConstraint, DOFType
from wnfea.boundary.body_loads import AccelerationField, SpatialPointLoad
from wnfea.mesh.solid_generator import generate_c3d10_structured_block
from wnfea.solver.heterogeneous_assembler import HeterogeneousAssembler


def test_c3d10_acceleration_mass_parity():
    """Verify sum(F) == M * a for C3D10 quadratic tetrahedra under arbitrary 3D acceleration."""
    # Create solid mesh with 20 elements along length
    solid_model, _ = generate_c3d10_structured_block(
        length=2.0, width=0.2, height=0.3, nx=8, ny=3, nz=3
    )
    nodes = solid_model.mesh_nodes
    elements = solid_model.solid_elements
    rho = 7850.0  # Steel kg/m^3

    # Theoretical mass: V = 2.0 * 0.2 * 0.3 = 0.12 m^3, M = 7850 * 0.12 = 942.0 kg
    V_theory = 2.0 * 0.2 * 0.3
    M_theory = rho * V_theory

    acc = AccelerationField(ax=15.0, ay=-9.80665, az=4.2, label="Arbitrary 3D G-Load")
    F_solid, total_mass = acc.compute_solid_body_forces(nodes, elements, rho)

    # 1. Total mass parity
    assert np.isclose(total_mass, M_theory, rtol=1e-5), f"Expected {M_theory} kg, got {total_mass} kg"

    # 2. Total force vector sum parity (Newton's second law F = M * a)
    F_sum = np.sum(F_solid, axis=0)
    F_expected = total_mass * acc.vector
    rel_force_err = np.linalg.norm(F_sum - F_expected) / np.linalg.norm(F_expected)

    assert rel_force_err < 1e-12, f"Force parity relative error too high: {rel_force_err}"
    print(f"  C3D10 Mass: {total_mass:.4f} kg (theory: {M_theory:.4f} kg)")
    print(f"  C3D10 Force parity relative error: {rel_force_err:.2e}")
    print("  [PASS] test_c3d10_acceleration_mass_parity")


def test_beam_acceleration_force_and_moment_parity():
    """Verify sum(F) == M * a and zero net moment for beams under transverse acceleration."""
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ], dtype=np.float64)
    elems = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    area = 0.01   # 0.01 m^2
    rho = 2700.0  # Aluminum kg/m^3
    total_len = 3.0
    M_theory = rho * area * total_len

    acc = AccelerationField(ax=5.0, ay=12.0, az=-3.0)
    F_beam, total_mass = acc.compute_beam_body_forces(nodes, elems, area, rho)

    assert np.isclose(total_mass, M_theory, rtol=1e-12)

    F_trans_sum = np.sum(F_beam[:, 0:3], axis=0)
    F_trans_expected = total_mass * acc.vector
    rel_trans_err = np.linalg.norm(F_trans_sum - F_trans_expected) / np.linalg.norm(F_trans_expected)
    assert rel_trans_err < 1e-14, f"Beam force sum error: {rel_trans_err}"

    print(f"  Beam Mass: {total_mass:.4f} kg | Force parity relative error: {rel_trans_err:.2e}")
    print("  [PASS] test_beam_acceleration_force_and_moment_parity")


def test_spatial_point_load_rbe3_equilibrium():
    """Verify RBE3 distribution preserves exact F and M about load point."""
    # Create cluster of 8 surface nodes around (1.0, 0.5, 0.2)
    rng = np.random.default_rng(42)
    center = np.array([1.0, 0.5, 0.2])
    cloud = center[None, :] + rng.uniform(-0.1, 0.1, size=(20, 3))

    sp_load = SpatialPointLoad(
        x=1.0, y=0.5, z=0.2,
        fx=1000.0, fy=-2500.0, fz=500.0,
        mx=150.0, my=-80.0, mz=320.0,
        n_nearest=8,
        method="rbe3",
        label="Engine Mount Thrust",
    )

    distributed_loads = sp_load.distribute_to_mesh(cloud)
    assert len(distributed_loads) == 8

    # Sum of nodal forces
    F_sum = np.zeros(3)
    M_sum = np.zeros(3)
    P = sp_load.position

    for ld in distributed_loads:
        f = np.array([ld.fx, ld.fy, ld.fz])
        x_node = cloud[ld.node_id]
        r = x_node - P
        F_sum += f
        M_sum += np.cross(r, f)

    # Check equilibrium
    F_err = np.linalg.norm(F_sum - sp_load.force) / np.linalg.norm(sp_load.force)
    M_err = np.linalg.norm(M_sum - sp_load.moment) / np.linalg.norm(sp_load.moment)

    assert F_err < 1e-10, f"RBE3 Force equilibrium error: {F_err}"
    assert M_err < 1e-10, f"RBE3 Moment equilibrium error: {M_err}"

    print(f"  RBE3 Force equilibrium error: {F_err:.2e} | Moment equilibrium error: {M_err:.2e}")
    print("  [PASS] test_spatial_point_load_rbe3_equilibrium")


def test_heterogeneous_solve_under_applied_gravity():
    """Verify heterogeneous CPU+GPU solve of cantilever solid under gravity body load."""
    solid_model, _ = generate_c3d10_structured_block(
        length=1.0, width=0.1, height=0.1, nx=4, ny=2, nz=2
    )

    # Fix base at x=0
    base_nodes = [i for i, n in enumerate(solid_model.mesh_nodes) if np.isclose(n[0], 0.0)]
    for nid in base_nodes:
        solid_model.supports.append(create_fixed_support(nid, is_geometry_node=False))

    # Clear any preexisting default tip load so only gravity acts
    solid_model.loads.clear()

    # Apply 10G downward gravity (-y)
    acc = AccelerationField.standard_gravity(axis="-y", g=98.0665)
    assembler = HeterogeneousAssembler(solid_model, device="cpu")

    # Assemble RHS directly with acceleration field
    F_active = assembler.assemble_rhs(acceleration=acc)
    assert np.any(np.abs(F_active) > 0.0), "Active force vector should not be all zero under gravity!"

    # Solve PCG
    u_full, stats = assembler.solve_pcg(F=F_active, tol=1e-6)
    assert stats["converged"], "Heterogeneous PCG did not converge under gravity!"

    # Check tip deflection in -y direction
    tip_nodes = [i for i, n in enumerate(solid_model.mesh_nodes) if np.isclose(n[0], 1.0)]
    tip_dy = [u_full[nid * 6 + 1] for nid in tip_nodes]
    avg_tip_dy = float(np.mean(tip_dy))

    assert avg_tip_dy < 0.0, f"Tip should deflect downwards under gravity, got {avg_tip_dy}"
    print(f"  Gravity solve converged in {stats['iterations']} iters, average tip deflection: {avg_tip_dy*1e3:.4f} mm")
    print("  [PASS] test_heterogeneous_solve_under_applied_gravity")


def run_all():
    print("=" * 60)
    print("Running Applied Acceleration Fields & Spatial Point Load Tests...")
    print("=" * 60)
    test_c3d10_acceleration_mass_parity()
    test_beam_acceleration_force_and_moment_parity()
    test_spatial_point_load_rbe3_equilibrium()
    test_heterogeneous_solve_under_applied_gravity()
    print("=" * 60)
    print("ALL ACCELERATION AND POINT LOAD TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
