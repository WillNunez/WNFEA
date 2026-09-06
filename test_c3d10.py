"""
Verification Test Suite for 10-Node Quadratic Tetrahedral Solid Element (C3D10).
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wnfea.elements.c3d10 import (
    shape_functions_c3d10,
    jacobian_c3d10,
    b_matrix_c3d10,
    elasticity_matrix_3d,
    element_stiffness_c3d10,
    element_internal_forces_c3d10,
    element_stresses_c3d10,
    GAUSS_POINTS_C3D10,
)


def create_reference_tetrahedron_coords(scale: float = 1.0) -> np.ndarray:
    """
    Create coordinates for a 10-node straight-edged tetrahedron.
    Corners at (0,0,0), (1,0,0), (0,1,0), (0,0,1) scaled by 'scale'.
    Mid-edge nodes at exact midpoints.
    """
    # 4 corner nodes
    c1 = np.array([0.0, 0.0, 0.0]) * scale
    c2 = np.array([1.0, 0.0, 0.0]) * scale
    c3 = np.array([0.0, 1.0, 0.0]) * scale
    c4 = np.array([0.0, 0.0, 1.0]) * scale

    # 6 mid-edge nodes
    m5  = 0.5 * (c1 + c2)
    m6  = 0.5 * (c2 + c3)
    m7  = 0.5 * (c3 + c1)
    m8  = 0.5 * (c1 + c4)
    m9  = 0.5 * (c2 + c4)
    m10 = 0.5 * (c3 + c4)

    return np.array([c1, c2, c3, c4, m5, m6, m7, m8, m9, m10], dtype=np.float64)


def test_shape_functions_and_partition_of_unity():
    """Verify interpolation property N_i(node_j) = delta_ij and sum(N_i) = 1."""
    print("\n--- Test 1: C3D10 Shape Functions & Partition of Unity ---")
    natural_nodes = [
        (0.0, 0.0, 0.0),  # Node 1
        (1.0, 0.0, 0.0),  # Node 2
        (0.0, 1.0, 0.0),  # Node 3
        (0.0, 0.0, 1.0),  # Node 4
        (0.5, 0.0, 0.0),  # Node 5
        (0.5, 0.5, 0.0),  # Node 6
        (0.0, 0.5, 0.0),  # Node 7
        (0.0, 0.0, 0.5),  # Node 8
        (0.5, 0.0, 0.5),  # Node 9
        (0.0, 0.5, 0.5),  # Node 10
    ]

    # Kronecker delta check
    for j, (xi, eta, zeta) in enumerate(natural_nodes):
        N, _ = shape_functions_c3d10(xi, eta, zeta)
        for i in range(10):
            expected = 1.0 if i == j else 0.0
            assert abs(N[i] - expected) < 1e-14, f"Shape function N_{i+1} at node {j+1} failed: {N[i]} != {expected}"

    # Partition of unity at arbitrary internal test points
    np.random.seed(42)
    for _ in range(20):
        # Generate random point inside reference tetrahedron: L1+L2+L3+L4 = 1
        r = np.random.dirichlet(np.ones(4))
        xi, eta, zeta = r[1], r[2], r[3]
        N, dN = shape_functions_c3d10(xi, eta, zeta)
        sum_N = np.sum(N)
        assert abs(sum_N - 1.0) < 1e-14, f"Partition of unity violated: sum(N) = {sum_N}"
        # Sum of derivatives must be zero
        assert np.allclose(np.sum(dN, axis=1), 0.0, atol=1e-14), "Sum of shape function derivatives is not zero"

    print("  [PASS] Kronecker delta and partition of unity verified.")


def test_rigid_body_modes_and_eigenvalues():
    """Verify that Ke has exactly 6 zero eigenvalues and 24 positive eigenvalues."""
    print("\n--- Test 2: Rigid Body Modes & Stiffness Spectrum ---")
    E = 2.1e11
    nu = 0.3
    coords = create_reference_tetrahedron_coords(scale=1.5)

    Ke = element_stiffness_c3d10(coords, E, nu)

    # 1. Symmetry check
    asym = np.max(np.abs(Ke - Ke.T))
    print(f"  Maximum asymmetry of Ke: {asym:.4e}")
    assert asym < 1e-6, f"Ke is not symmetric (asymmetry = {asym})"

    # 2. Eigenvalue spectrum
    eigvals = np.linalg.eigvalsh(Ke)
    print("  Lowest 8 eigenvalues of Ke:")
    for idx, ev in enumerate(eigvals[:8]):
        print(f"    Mode {idx+1:2d}: {ev:.4e}")

    # First 6 must be zero within numerical precision relative to trace
    trace = np.trace(Ke)
    for i in range(6):
        assert abs(eigvals[i]) / trace < 1e-12, f"Rigid body eigenvalue {i+1} is non-zero: {eigvals[i]}"

    # Remaining 24 must be positive (strain modes)
    for i in range(6, 30):
        assert eigvals[i] > 1e-8 * trace, f"Deformational eigenvalue {i+1} is not positive: {eigvals[i]}"

    # 3. Explicit Rigid Body Displacements
    # Pure translations
    for t_dir in range(3):
        u_trans = np.zeros(30)
        for node in range(10):
            u_trans[3 * node + t_dir] = 1.0
        f_trans = Ke @ u_trans
        assert np.linalg.norm(f_trans) / trace < 1e-12, f"Translation along axis {t_dir} produced non-zero force"

    # Pure rotations
    # Rotation about Z-axis: ux = -Y, uy = X, uz = 0
    u_rot_z = np.zeros(30)
    for node in range(10):
        X, Y, Z = coords[node]
        u_rot_z[3 * node + 0] = -Y
        u_rot_z[3 * node + 1] = X
    f_rot_z = Ke @ u_rot_z
    assert np.linalg.norm(f_rot_z) / trace < 1e-12, "Rotation about Z produced non-zero force"

    print("  [PASS] Exactly 6 zero rigid body modes and 24 positive strain modes verified.")


def test_constant_strain_patch_test():
    """Verify that any linear displacement field produces exact constant strain and stress."""
    print("\n--- Test 3: Constant Strain Patch Test ---")
    E = 2.0e11
    nu = 0.28
    C = elasticity_matrix_3d(E, nu)

    # General curved/perturbed coordinates to test isoparametric mapping
    coords = create_reference_tetrahedron_coords(scale=2.0)
    # Slightly perturb mid-edge nodes (quadratic curvature)
    coords[4] += np.array([0.02, 0.01, -0.01])
    coords[7] += np.array([-0.01, 0.02, 0.01])

    # Prescribe arbitrary constant strain state:
    # eps = [eps_xx, eps_yy, eps_zz, gamma_xy, gamma_yz, gamma_zx]
    eps_target = np.array([1.2e-3, -0.5e-3, 0.8e-3, 0.6e-3, -0.4e-3, 0.3e-3], dtype=np.float64)
    sigma_target = C @ eps_target

    # Corresponding linear displacement field:
    # u = eps_xx*x + 0.5*gamma_xy*y + 0.5*gamma_zx*z
    # v = 0.5*gamma_xy*x + eps_yy*y + 0.5*gamma_yz*z
    # w = 0.5*gamma_zx*x + 0.5*gamma_yz*y + eps_zz*z
    u_e = np.zeros(30, dtype=np.float64)
    for i in range(10):
        x, y, z = coords[i]
        u_e[3 * i + 0] = eps_target[0] * x + 0.5 * eps_target[3] * y + 0.5 * eps_target[5] * z
        u_e[3 * i + 1] = 0.5 * eps_target[3] * x + eps_target[1] * y + 0.5 * eps_target[4] * z
        u_e[3 * i + 2] = 0.5 * eps_target[5] * x + 0.5 * eps_target[4] * y + eps_target[2] * z

    # Check strain and stress at all Gauss points
    gp_stresses, gp_vm, _ = element_stresses_c3d10(coords, u_e, E, nu)

    for gp_idx, ((xi, eta, zeta), _) in enumerate(zip(GAUSS_POINTS_C3D10, GAUSS_POINTS_C3D10)):
        B, _ = b_matrix_c3d10(coords, xi, eta, zeta)
        eps_gp = B @ u_e
        err_eps = np.max(np.abs(eps_gp - eps_target))
        assert err_eps < 1e-12, f"Gauss point {gp_idx} strain error: {err_eps}"

        err_sig = np.max(np.abs(gp_stresses[gp_idx] - sigma_target))
        assert err_sig < 1e-2, f"Gauss point {gp_idx} stress error: {err_sig}"

    print(f"  Target Von Mises stress: {gp_vm[0] / 1e6:.4f} MPa")
    print("  [PASS] Constant strain patch test passed with machine precision (< 1e-12).")


def test_matrix_free_internal_force_consistency():
    """Verify that element_internal_forces_c3d10 matches Ke @ u_e exactly."""
    print("\n--- Test 4: Matrix-Free Internal Force Consistency ---")
    E = 1.8e11
    nu = 0.33
    coords = create_reference_tetrahedron_coords(scale=1.2)

    Ke = element_stiffness_c3d10(coords, E, nu)

    np.random.seed(99)
    u_e = np.random.randn(30) * 1e-3

    f_assembled = Ke @ u_e
    f_matrix_free = element_internal_forces_c3d10(coords, u_e, E, nu)

    rel_err = np.linalg.norm(f_matrix_free - f_assembled) / np.linalg.norm(f_assembled)
    print(f"  Relative difference between Ke @ u_e and matrix-free f_int: {rel_err:.4e}")
    assert rel_err < 1e-14, f"Matrix-free internal force mismatch: {rel_err}"
    print("  [PASS] Matrix-free internal force vector exactly equals Ke @ u_e.")


if __name__ == "__main__":
    print("=" * 65)
    print("      WNFEA C3D10 Quadratic Tetrahedral Element Test Suite")
    print("=" * 65)

    test_shape_functions_and_partition_of_unity()
    test_rigid_body_modes_and_eigenvalues()
    test_constant_strain_patch_test()
    test_matrix_free_internal_force_consistency()

    print("\n" + "=" * 65)
    print("      ALL C3D10 ELEMENT FORMULATION TESTS PASSED!")
    print("=" * 65)
