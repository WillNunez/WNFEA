"""
10-Node Quadratic Tetrahedral Solid Element (C3D10).

Industry-standard continuum solid formulation matching Abaqus C3D10.
Features:
- 10 nodes: 4 corner vertices and 6 mid-edge nodes
- 3 translational DOFs per node (ux, uy, uz), 30 DOFs per element
- Isoparametric quadratic shape functions in natural volume coordinates (L1, L2, L3, L4)
- Symmetric 4-point Gauss-Hammer numerical integration
- 3D isotropic elasticity constitutive tensor
- Both 30x30 stiffness matrix assembly and matrix-free internal force vector evaluation
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Quadrature Constants for Tetrahedral Elements (Hammer 4-Point Rule)
# ---------------------------------------------------------------------------
_ALPHA = (5.0 - np.sqrt(5.0)) / 20.0       # ≈ 0.1381966011250105
_BETA = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0  # ≈ 0.5854101966249685
_WEIGHT = 1.0 / 24.0

GAUSS_POINTS_C3D10 = (
    np.array([_ALPHA, _ALPHA, _ALPHA]),  # L = (BETA, ALPHA, ALPHA, ALPHA)
    np.array([_BETA, _ALPHA, _ALPHA]),   # L = (ALPHA, BETA, ALPHA, ALPHA)
    np.array([_ALPHA, _BETA, _ALPHA]),   # L = (ALPHA, ALPHA, BETA, ALPHA)
    np.array([_ALPHA, _ALPHA, _BETA]),   # L = (ALPHA, ALPHA, ALPHA, BETA)
)
GAUSS_WEIGHTS_C3D10 = (_WEIGHT, _WEIGHT, _WEIGHT, _WEIGHT)


def shape_functions_c3d10(xi: float, eta: float, zeta: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the 10 quadratic shape functions and their natural derivatives
    at natural coordinates (xi, eta, zeta).

    Volume coordinates:
        L1 = 1 - xi - eta - zeta
        L2 = xi
        L3 = eta
        L4 = zeta

    Corner nodes:
        N1 = L1*(2*L1 - 1)
        N2 = L2*(2*L2 - 1)
        N3 = L3*(2*L3 - 1)
        N4 = L4*(2*L4 - 1)

    Mid-edge nodes (Abaqus convention):
        N5  = 4*L1*L2  (edge 1-2)
        N6  = 4*L2*L3  (edge 2-3)
        N7  = 4*L3*L1  (edge 3-1)
        N8  = 4*L1*L4  (edge 1-4)
        N9  = 4*L2*L4  (edge 2-4)
        N10 = 4*L3*L4  (edge 3-4)

    Returns:
        N: (10,) array of shape function values.
        dN_dxi: (3, 10) array of derivatives [dN/dxi; dN/deta; dN/dzeta].
    """
    L1 = 1.0 - xi - eta - zeta
    L2 = xi
    L3 = eta
    L4 = zeta

    # Shape functions
    N = np.array([
        L1 * (2.0 * L1 - 1.0),
        L2 * (2.0 * L2 - 1.0),
        L3 * (2.0 * L3 - 1.0),
        L4 * (2.0 * L4 - 1.0),
        4.0 * L1 * L2,
        4.0 * L2 * L3,
        4.0 * L3 * L1,
        4.0 * L1 * L4,
        4.0 * L2 * L4,
        4.0 * L3 * L4,
    ], dtype=np.float64)

    # Derivatives w.r.t L1, L2, L3, L4:
    # dN_dL has shape (4, 10)
    dN_dL = np.array([
        # [dN/dL1, dN/dL2, dN/dL3, dN/dL4] for each shape function
        [4.0*L1 - 1.0, 0.0, 0.0, 0.0],          # N1
        [0.0, 4.0*L2 - 1.0, 0.0, 0.0],          # N2
        [0.0, 0.0, 4.0*L3 - 1.0, 0.0],          # N3
        [0.0, 0.0, 0.0, 4.0*L4 - 1.0],          # N4
        [4.0*L2, 4.0*L1, 0.0, 0.0],             # N5
        [0.0, 4.0*L3, 4.0*L2, 0.0],             # N6
        [4.0*L3, 0.0, 4.0*L1, 0.0],             # N7
        [4.0*L4, 0.0, 0.0, 4.0*L1],             # N8
        [0.0, 4.0*L4, 0.0, 4.0*L2],             # N9
        [0.0, 0.0, 4.0*L4, 4.0*L3],             # N10
    ], dtype=np.float64).T  # Shape: (4, 10)

    # Chain rule: d/dxi = d/dL2 - d/dL1, etc.
    # dL/dxi = [-1, 1, 0, 0], dL/deta = [-1, 0, 1, 0], dL/dzeta = [-1, 0, 0, 1]
    dN_dxi = np.zeros((3, 10), dtype=np.float64)
    dN_dxi[0, :] = -dN_dL[0, :] + dN_dL[1, :]  # dN/dxi
    dN_dxi[1, :] = -dN_dL[0, :] + dN_dL[2, :]  # dN/deta
    dN_dxi[2, :] = -dN_dL[0, :] + dN_dL[3, :]  # dN/dzeta

    return N, dN_dxi


def jacobian_c3d10(
    node_coords: np.ndarray,
    dN_dxi: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Compute the Jacobian matrix J, determinant det(J), and inverse J^{-1}
    for a 10-node tetrahedron at an integration point.

    Args:
        node_coords: (10, 3) array of nodal coordinates X, Y, Z.
        dN_dxi: (3, 10) array of natural shape function derivatives.

    Returns:
        J: (3, 3) Jacobian matrix [dx/dxi, dy/dxi, dz/dxi; ...].
        det_J: Determinant of J.
        inv_J: (3, 3) Inverse of J.
    """
    # J = dN_dxi @ node_coords  (shape (3, 3))
    J = dN_dxi @ node_coords

    det_J = float(np.linalg.det(J))
    if det_J <= 0.0:
        raise ValueError(
            f"Non-positive Jacobian determinant in C3D10 element (det_J = {det_J:.4e}). "
            "Element is inverted, degenerate, or has bad node numbering."
        )

    inv_J = np.linalg.inv(J)
    return J, det_J, inv_J


def b_matrix_c3d10(
    node_coords: np.ndarray,
    xi: float,
    eta: float,
    zeta: float,
) -> tuple[np.ndarray, float]:
    """
    Build the 6x30 strain-displacement matrix B in global Cartesian coordinates:
        epsilon = B * u_e
    where epsilon = [eps_xx, eps_yy, eps_zz, gamma_xy, gamma_yz, gamma_zx]^T.

    Args:
        node_coords: (10, 3) array of nodal coordinates.
        xi, eta, zeta: Natural coordinates in reference tetrahedron.

    Returns:
        B: (6, 30) strain-displacement matrix.
        det_J: Determinant of Jacobian at this point.
    """
    _, dN_dxi = shape_functions_c3d10(xi, eta, zeta)
    _, det_J, inv_J = jacobian_c3d10(node_coords, dN_dxi)

    # Cartesian shape function derivatives: dN_dx = inv_J @ dN_dxi (shape (3, 10))
    dN_dx = inv_J @ dN_dxi

    B = np.zeros((6, 30), dtype=np.float64)

    for i in range(10):
        dN_x = dN_dx[0, i]
        dN_y = dN_dx[1, i]
        dN_z = dN_dx[2, i]

        col = 3 * i
        # eps_xx = du/dx
        B[0, col + 0] = dN_x
        # eps_yy = dv/dy
        B[1, col + 1] = dN_y
        # eps_zz = dw/dz
        B[2, col + 2] = dN_z
        # gamma_xy = du/dy + dv/dx
        B[3, col + 0] = dN_y
        B[3, col + 1] = dN_x
        # gamma_yz = dv/dz + dw/dy
        B[4, col + 1] = dN_z
        B[4, col + 2] = dN_y
        # gamma_zx = dw/dx + du/dz
        B[5, col + 0] = dN_z
        B[5, col + 2] = dN_x

    return B, det_J


def elasticity_matrix_3d(E: float, nu: float) -> np.ndarray:
    """
    Construct the 6x6 3D isotropic constitutive matrix C relating engineering
    strain to Cauchy stress:
        sigma = C * epsilon
    """
    factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    lam = factor * nu
    two_mu = factor * (1.0 - 2.0 * nu)  # = 2*G
    diag_term = factor * (1.0 - nu)

    C = np.zeros((6, 6), dtype=np.float64)
    # Direct normal components
    C[0, 0] = C[1, 1] = C[2, 2] = diag_term
    C[0, 1] = C[1, 0] = lam
    C[0, 2] = C[2, 0] = lam
    C[1, 2] = C[2, 1] = lam

    # Shear components (G = E / (2*(1+nu)))
    G = E / (2.0 * (1.0 + nu))
    C[3, 3] = G
    C[4, 4] = G
    C[5, 5] = G

    return C


def element_stiffness_c3d10(
    node_coords: np.ndarray,
    E: float,
    nu: float,
) -> np.ndarray:
    """
    Compute the 30x30 elemental stiffness matrix for a C3D10 quadratic tetrahedron:
        K_e = ∫ B^T C B dΩ = ∑_g w_g B_g^T C B_g det(J_g)

    Args:
        node_coords: (10, 3) array of nodal coordinates.
        E: Young's modulus.
        nu: Poisson's ratio.

    Returns:
        Ke: (30, 30) symmetric element stiffness matrix.
    """
    coords = np.asarray(node_coords, dtype=np.float64)
    if coords.shape != (10, 3):
        raise ValueError(f"Expected node_coords of shape (10, 3), got {coords.shape}")

    C = elasticity_matrix_3d(E, nu)
    Ke = np.zeros((30, 30), dtype=np.float64)

    for (xi, eta, zeta), w in zip(GAUSS_POINTS_C3D10, GAUSS_WEIGHTS_C3D10):
        B, det_J = b_matrix_c3d10(coords, xi, eta, zeta)
        Ke += w * (B.T @ C @ B) * det_J

    # Enforce exact machine symmetry
    Ke = 0.5 * (Ke + Ke.T)
    return Ke


def element_internal_forces_c3d10(
    node_coords: np.ndarray,
    u_e: np.ndarray,
    E: float,
    nu: float,
) -> np.ndarray:
    """
    Compute the 30-DOF internal force vector for a C3D10 element:
        f_int = ∫ B^T σ dΩ = ∑_g w_g B_g^T σ_g det(J_g)
    Directly compatible with matrix-free JFNK residual evaluation.

    Args:
        node_coords: (10, 3) array of nodal coordinates.
        u_e: (30,) element nodal displacement vector.
        E, nu: Material elastic constants.

    Returns:
        f_int: (30,) element internal force vector.
    """
    coords = np.asarray(node_coords, dtype=np.float64)
    u_vec = np.asarray(u_e, dtype=np.float64)
    C = elasticity_matrix_3d(E, nu)

    f_int = np.zeros(30, dtype=np.float64)

    for (xi, eta, zeta), w in zip(GAUSS_POINTS_C3D10, GAUSS_WEIGHTS_C3D10):
        B, det_J = b_matrix_c3d10(coords, xi, eta, zeta)
        eps = B @ u_vec
        sigma = C @ eps
        f_int += w * (B.T @ sigma) * det_J

    return f_int


def element_stresses_c3d10(
    node_coords: np.ndarray,
    u_e: np.ndarray,
    E: float,
    nu: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Evaluate Cauchy stresses and equivalent Von Mises stress at integration points.

    Returns:
        gp_stresses: (4, 6) Cauchy stresses at the 4 Gauss points [sxx, syy, szz, sxy, syz, szx].
        gp_von_mises: (4,) Von Mises equivalent stress at each Gauss point.
        max_von_mises: Maximum Von Mises stress within the element.
    """
    coords = np.asarray(node_coords, dtype=np.float64)
    u_vec = np.asarray(u_e, dtype=np.float64)
    C = elasticity_matrix_3d(E, nu)

    gp_stresses = np.zeros((4, 6), dtype=np.float64)
    gp_vm = np.zeros(4, dtype=np.float64)

    for idx, ((xi, eta, zeta), _) in enumerate(zip(GAUSS_POINTS_C3D10, GAUSS_WEIGHTS_C3D10)):
        B, _ = b_matrix_c3d10(coords, xi, eta, zeta)
        eps = B @ u_vec
        sig = C @ eps
        gp_stresses[idx, :] = sig

        sxx, syy, szz, sxy, syz, szx = sig
        vm = np.sqrt(0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2 +
                            6.0 * (sxy**2 + syz**2 + szx**2)))
        gp_vm[idx] = vm

    return gp_stresses, gp_vm, float(np.max(gp_vm))
