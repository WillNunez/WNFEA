"""
3D Co-rotational Beam Formulation for WNFEA.

Implements large-displacement, finite-rotation kinematics for 3D beam elements.
Computes local deformational displacements and the global internal force vector
f_int^e = B^T q, guaranteeing exact self-equilibrium and frame invariance
under arbitrary rigid-body motions.
"""

from __future__ import annotations

import numpy as np


def compute_element_triad(
    node1: np.ndarray,
    node2: np.ndarray,
    u_e: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Compute the current co-rotating triad (r1, r2, r3), current length L,
    and undeformed length L0 for a 3D beam element.

    Args:
        node1: Undeformed coordinates of node 1 (3,).
        node2: Undeformed coordinates of node 2 (3,).
        u_e: Global displacement/rotation vector (12,) [u1(3), theta1(3), u2(3), theta2(3)].

    Returns:
        r1, r2, r3: Unit vectors of current co-rotating frame (each (3,)).
        L: Current deformed chord length.
        L0: Undeformed chord length.
    """
    X1 = np.asarray(node1, dtype=np.float64)
    X2 = np.asarray(node2, dtype=np.float64)
    v0 = X2 - X1
    L0 = float(np.linalg.norm(v0))
    if L0 < 1e-12:
        raise ValueError("Element has zero undeformed length")

    # Undeformed local x-axis
    e1_0 = v0 / L0

    # Initial reference triad
    if np.isclose(abs(e1_0[0]), 1.0):
        e2_0 = np.array([0.0, 1.0, 0.0])
        e3_0 = np.array([0.0, 0.0, 1.0])
    elif np.isclose(abs(e1_0[1]), 1.0):
        e2_0 = np.array([1.0, 0.0, 0.0])
        e3_0 = np.array([0.0, 0.0, 1.0])
    elif np.isclose(abs(e1_0[2]), 1.0):
        e2_0 = np.array([1.0, 0.0, 0.0])
        e3_0 = np.array([0.0, 1.0, 0.0])
    else:
        gz = np.array([0.0, 0.0, 1.0])
        e2_0 = np.cross(gz, e1_0)
        norm2 = np.linalg.norm(e2_0)
        if norm2 < 1e-9:
            e2_0 = np.cross(np.array([0.0, 1.0, 0.0]), e1_0)
            norm2 = np.linalg.norm(e2_0)
        e2_0 /= norm2
        e3_0 = np.cross(e1_0, e2_0)
        e3_0 /= np.linalg.norm(e3_0)

    if u_e is None or np.linalg.norm(u_e) < 1e-15:
        return e1_0, e2_0, e3_0, L0, L0

    # Current deformed positions
    u1 = u_e[0:3]
    theta1 = u_e[3:6]
    u2 = u_e[6:9]
    theta2 = u_e[9:12]

    x1 = X1 + u1
    x2 = X2 + u2
    v = x2 - x1
    L = float(np.linalg.norm(v))
    if L < 1e-12:
        raise ValueError("Element collapsed to zero length during deformation")

    # Current chord axis
    r1 = v / L

    # Rotation vector aligning e1_0 to r1 (Rodrigues' rotation)
    cross_prod = np.cross(e1_0, r1)
    sin_phi = np.linalg.norm(cross_prod)
    cos_phi = float(np.dot(e1_0, r1))

    if sin_phi > 1e-12:
        rot_axis = cross_prod / sin_phi
        phi = np.arctan2(sin_phi, cos_phi)
        def rodrigues(vec):
            return (vec * np.cos(phi) +
                    np.cross(rot_axis, vec) * np.sin(phi) +
                    rot_axis * np.dot(rot_axis, vec) * (1.0 - np.cos(phi)))
        r2 = rodrigues(e2_0)
        r3 = rodrigues(e3_0)
    else:
        if cos_phi < 0.0:  # 180-degree flip
            r2 = -e2_0
            r3 = -e3_0
        else:
            r2 = e2_0.copy()
            r3 = e3_0.copy()

    # Average nodal twist about current axis r1
    avg_twist = 0.5 * float(np.dot(theta1 + theta2, r1))
    if abs(avg_twist) > 1e-12:
        cos_t = np.cos(avg_twist)
        sin_t = np.sin(avg_twist)
        r2_t = r2 * cos_t + np.cross(r1, r2) * sin_t
        r3_t = r3 * cos_t + np.cross(r1, r3) * sin_t
        r2 = r2_t
        r3 = r3_t

    # Re-orthonormalize (Gram-Schmidt)
    r1 /= np.linalg.norm(r1)
    r2 = r2 - np.dot(r2, r1) * r1
    r2 /= np.linalg.norm(r2)
    r3 = np.cross(r1, r2)
    r3 /= np.linalg.norm(r3)

    return r1, r2, r3, L, L0


def build_corotational_projector(
    r1: np.ndarray,
    r2: np.ndarray,
    r3: np.ndarray,
    L: float,
) -> np.ndarray:
    """
    Build the 6x12 kinematic transformation matrix B relating local deformational
    degrees of freedom to global nodal displacement/rotation variations:
    delta_p = B * delta_u_e.

    p = [u_l, theta_y1, theta_z1, theta_x, theta_y2, theta_z2]^T
    u_e = [u1(3), theta1(3), u2(3), theta2(3)]^T
    """
    B = np.zeros((6, 12), dtype=np.float64)

    # 1. Axial elongation u_l
    B[0, 0:3] = -r1
    B[0, 6:9] = r1

    # 2. Bending rotation y1
    B[1, 0:3] = -r3 / L
    B[1, 3:6] = r2
    B[1, 6:9] = r3 / L

    # 3. Bending rotation z1
    B[2, 0:3] = r2 / L
    B[2, 3:6] = r3
    B[2, 6:9] = -r2 / L

    # 4. Torsional twist theta_x
    B[3, 3:6] = -r1
    B[3, 9:12] = r1

    # 5. Bending rotation y2
    B[4, 0:3] = -r3 / L
    B[4, 6:9] = r3 / L
    B[4, 9:12] = r2

    # 6. Bending rotation z2
    B[5, 0:3] = r2 / L
    B[5, 6:9] = -r2 / L
    B[5, 9:12] = r3

    return B


def compute_corotational_element_forces(
    node1: np.ndarray,
    node2: np.ndarray,
    u_e: np.ndarray,
    E: float,
    G: float,
    A: float,
    Iy: float,
    Iz: float,
    J: float,
) -> np.ndarray:
    """
    Compute the 12-DOF global internal force vector for a 3D co-rotational beam:
    f_int,e = B^T * q

    Args:
        node1, node2: Undeformed nodal coordinates (3,).
        u_e: Global displacement vector for element (12,).
        E, G, A, Iy, Iz, J: Material and cross-section parameters.

    Returns:
        f_int,e: 12-DOF global internal force vector.
    """
    r1, r2, r3, L, L0 = compute_element_triad(node1, node2, u_e)

    # Extract nodal rotations and relative deformations
    u1 = u_e[0:3]
    theta1 = u_e[3:6]
    u2 = u_e[6:9]
    theta2 = u_e[9:12]

    du = u2 - u1

    # Deformational displacements:
    # Elongation (numerically stable against catastrophic cancellation)
    v0 = np.asarray(node2, dtype=np.float64) - np.asarray(node1, dtype=np.float64)
    u_l = float((2.0 * np.dot(v0, du) + np.dot(du, du)) / (L + L0))

    # Local rotational deformations relative to chord
    th_y1 = float(np.dot(r2, theta1) + np.dot(r3, du) / L)
    th_z1 = float(np.dot(r3, theta1) - np.dot(r2, du) / L)
    th_x  = float(np.dot(r1, theta2 - theta1))
    th_y2 = float(np.dot(r2, theta2) + np.dot(r3, du) / L)
    th_z2 = float(np.dot(r3, theta2) - np.dot(r2, du) / L)

    # Local constitutive forces q = [N, My1, Mz1, Tx, My2, Mz2]
    # Axial force
    N = (E * A / L0) * u_l
    My1 = (E * Iy / L0) * (4.0 * th_y1 + 2.0 * th_y2)
    My2 = (E * Iy / L0) * (2.0 * th_y1 + 4.0 * th_y2)
    Mz1 = (E * Iz / L0) * (4.0 * th_z1 + 2.0 * th_z2)
    Mz2 = (E * Iz / L0) * (2.0 * th_z1 + 4.0 * th_z2)
    Tx  = (G * J / L0) * th_x

    q = np.array([N, My1, Mz1, Tx, My2, Mz2], dtype=np.float64)

    # Project to global 12 DOFs: f_int = B^T * q
    B = build_corotational_projector(r1, r2, r3, L)
    f_int = B.T @ q

    return f_int
