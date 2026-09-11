"""
Numerical Asymptotic Homogenization Engine for WNFEA (Phase 20).
-----------------------------------------------------------------
Evaluates the macroscale effective elasticity tensor C^eff_ijkl and engineering
elastic constants (E_x, E_y, E_z, G_xy, nu_xy) of periodic porous cellular
structures (such as TPMS Gyroid, Schwarz P, Diamond lattices) using numerical
homogenization under periodic boundary conditions (PBC).

Formulation:
1. Unit cell domain Y = [0, Lx] x [0, Ly] x [0, Lz] with periodic boundary conditions.
2. 6 Canonical macroscopic unit strain load cases eps_bar^(k) for k in {1..6}.
3. Periodic boundary conditions enforced directly via periodic index reduction:
      node_id_periodic(i, j, k) = (i % Nx) + (j % Ny)*Nx + (k % Nz)*Nx*Ny
   eliminating slave DOFs exactly with zero penalty errors.
4. Microscale displacement fluctuations chi^(k) solved from:
      K_periodic * chi^(k) = -sum_e V_e * B_e^T * C_e * eps_bar^(k)
5. Symmetric positive-semidefinite effective elasticity tensor:
      C^eff_ij = 1/|Y| * [ sum_e V_e * (eps_bar^(i))^T * C_e * eps_bar^(j) - (chi^(i))^T * K_p * chi^(j) ]
6. Extraction of engineering constants, Zener anisotropy index A_Z, and Gibson-Ashby scaling.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve, cg

from ..opt.tpms_lattice import TPMSConfig, evaluate_graded_tpms_field


@dataclass
class HomogenizationResult:
    """
    Results from asymptotic numerical homogenization of a periodic micro-structure.
    """
    C_effective: np.ndarray           # (6, 6) Effective elasticity tensor (Pa) in Voigt [xx, yy, zz, yz, zx, xy]
    S_effective: np.ndarray           # (6, 6) Effective compliance tensor (1/Pa)
    E_x: float                        # Effective Young's modulus along X (Pa)
    E_y: float                        # Effective Young's modulus along Y (Pa)
    E_z: float                        # Effective Young's modulus along Z (Pa)
    G_yz: float                       # Effective shear modulus YZ (Pa)
    G_zx: float                       # Effective shear modulus ZX (Pa)
    G_xy: float                       # Effective shear modulus XY (Pa)
    nu_xy: float                      # Effective Poisson's ratio
    nu_yz: float                      # Effective Poisson's ratio
    nu_zx: float                      # Effective Poisson's ratio
    relative_density: float           # Actual volume fraction of solid material in [0, 1]
    anisotropy_ratio: float           # Zener anisotropy index A_Z = 2*C44 / (C11 - C12)
    cubic_symmetry_residual: float    # Measure of deviation from ideal cubic symmetry
    solve_time: float                 # Solution wallclock time in seconds


def compute_isotropic_elasticity_matrix(E: float, nu: float) -> np.ndarray:
    """Compute standard 6x6 isotropic elasticity matrix C0 in Voigt notation."""
    c_val = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    C0 = c_val * np.array([
        [1.0 - nu, nu, nu, 0.0, 0.0, 0.0],
        [nu, 1.0 - nu, nu, 0.0, 0.0, 0.0],
        [nu, nu, 1.0 - nu, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, (1.0 - 2.0 * nu) * 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, (1.0 - 2.0 * nu) * 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, (1.0 - 2.0 * nu) * 0.5],
    ], dtype=np.float64)
    return C0


def solve_periodic_homogenization(
    element_densities: np.ndarray,
    cell_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    base_E: float = 210e9,
    base_nu: float = 0.3,
    simp_penalty: float = 1.0,
    rho_min: float = 1e-4,
) -> HomogenizationResult:
    """
    Compute the homogenized effective elasticity tensor C^eff for a 3D periodic voxel micro-structure.

    Parameters:
        element_densities: 3D array of shape (Nx, Ny, Nz) or 1D array of shape (Nx*Ny*Nz,)
                           specifying element relative densities in [0, 1].
        cell_size: Macroscale dimensions (Lx, Ly, Lz) of the periodic unit cell in meters.
        base_E: Young's modulus of solid material in Pa.
        base_nu: Poisson's ratio of solid material.
        simp_penalty: SIMP interpolation exponent p (default: 1.0 for linear cut-cell).
        rho_min: Minimum stiffness threshold for void elements (default: 1e-4).

    Returns:
        HomogenizationResult instance.
    """
    start_time = time.time()

    dens = np.asarray(element_densities, dtype=np.float64)
    if dens.ndim == 3:
        Nx, Ny, Nz = dens.shape
        rho_elems = dens.ravel(order="F")  # column-major or consistent flattening
    elif dens.ndim == 1:
        # Assume cubic grid
        n_elem = len(dens)
        n_side = int(round(n_elem ** (1.0 / 3.0)))
        if n_side ** 3 != n_elem:
            raise ValueError(f"1D element_densities length {n_elem} must be a perfect cube (e.g. 8x8x8).")
        Nx, Ny, Nz = n_side, n_side, n_side
        rho_elems = dens
    else:
        raise ValueError(f"Unsupported element_densities shape: {dens.shape}")

    Lx, Ly, Lz = cell_size
    hx, hy, hz = Lx / Nx, Ly / Ny, Lz / Nz
    vol_cell = hx * hy * hz
    vol_total = Lx * Ly * Lz

    # Periodic nodal mapping
    n_nodes_p = Nx * Ny * Nz
    n_dofs = 3 * n_nodes_p

    def p_node_id(i: int, j: int, k: int) -> int:
        return (i % Nx) + (j % Ny) * Nx + (k % Nz) * (Nx * Ny)

    # 1. Reference Hex8 element B-matrix and base stiffness
    C0 = compute_isotropic_elasticity_matrix(base_E, base_nu)

    inv_hx, inv_hy, inv_hz = 1.0 / hx, 1.0 / hy, 1.0 / hz
    dNdxi = np.array([
        [-1, 1, 1, -1, -1, 1, 1, -1],
        [-1, -1, 1, 1, -1, -1, 1, 1],
        [-1, -1, -1, -1, 1, 1, 1, 1],
    ], dtype=np.float64) * 0.125
    dNdX = np.array([
        dNdxi[0] * inv_hx,
        dNdxi[1] * inv_hy,
        dNdxi[2] * inv_hz,
    ])

    B0 = np.zeros((6, 24), dtype=np.float64)
    for a in range(8):
        B0[0, 3*a]   = dNdX[0, a]
        B0[1, 3*a+1] = dNdX[1, a]
        B0[2, 3*a+2] = dNdX[2, a]
        B0[3, 3*a+1] = dNdX[2, a]
        B0[3, 3*a+2] = dNdX[1, a]
        B0[4, 3*a]   = dNdX[2, a]
        B0[4, 3*a+2] = dNdX[0, a]
        B0[5, 3*a]   = dNdX[1, a]
        B0[5, 3*a+1] = dNdX[0, a]

    Ke0 = vol_cell * (B0.T @ C0 @ B0)  # (24, 24)

    # 2. Assemble periodic global stiffness matrix K_p and 6 load vectors F_p
    K_p = lil_matrix((n_dofs, n_dofs), dtype=np.float64)
    F_p = np.zeros((n_dofs, 6), dtype=np.float64)
    unit_strains = np.eye(6)

    elem_idx = 0
    macro_energy = np.zeros((6, 6), dtype=np.float64)
    total_solid_vol = 0.0

    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                rho_e = float(rho_elems[elem_idx])
                elem_idx += 1
                total_solid_vol += rho_e * vol_cell

                # SIMP penalization
                factor = max(rho_e ** simp_penalty, rho_min)
                Ke_e = factor * Ke0
                Ce_e = factor * C0

                # 8 corner nodes under periodic wrap
                corners = [
                    p_node_id(i, j, k),
                    p_node_id(i + 1, j, k),
                    p_node_id(i + 1, j + 1, k),
                    p_node_id(i, j + 1, k),
                    p_node_id(i, j, k + 1),
                    p_node_id(i + 1, j, k + 1),
                    p_node_id(i + 1, j + 1, k + 1),
                    p_node_id(i, j + 1, k + 1),
                ]

                edofs = np.zeros(24, dtype=int)
                for a in range(8):
                    edofs[3*a : 3*a+3] = [3*corners[a], 3*corners[a]+1, 3*corners[a]+2]

                # Accumulate element stiffness
                for r in range(24):
                    for c in range(24):
                        K_p[edofs[r], edofs[c]] += Ke_e[r, c]

                # Accumulate macroscopic energy and microscopic load vectors
                for s in range(6):
                    fe = vol_cell * (B0.T @ (Ce_e @ unit_strains[s]))
                    F_p[edofs, s] -= fe
                    for t in range(6):
                        macro_energy[s, t] += vol_cell * (unit_strains[s] @ (Ce_e @ unit_strains[t]))

    K_csr = K_p.tocsr()

    # 3. Solve microscale characteristic fluctuations chi^(s) for s = 1..6
    # Fix master node 0 DOFs (0, 1, 2) to prevent rigid body translation
    free_dofs = np.arange(3, n_dofs)
    K_sub = K_csr[free_dofs, :][:, free_dofs]

    chi = np.zeros((n_dofs, 6), dtype=np.float64)
    for s in range(6):
        rhs = F_p[free_dofs, s]
        chi[free_dofs, s] = spsolve(K_sub, rhs)

    # 4. Evaluate homogenized elasticity tensor:
    # C_eff[i, j] = 1/|Y| * [ macro_energy[i, j] - chi^(i)^T * K * chi^(j) ]
    C_eff = np.zeros((6, 6), dtype=np.float64)
    for i in range(6):
        for j in range(6):
            term_fluct = chi[:, i] @ (K_csr @ chi[:, j])
            C_eff[i, j] = (macro_energy[i, j] - term_fluct) / vol_total

    # Enforce strict symmetry
    C_eff = 0.5 * (C_eff + C_eff.T)

    # 5. Extract compliance tensor S_eff and engineering elastic constants
    S_eff = np.linalg.pinv(C_eff)

    Ex = 1.0 / max(S_eff[0, 0], 1e-30)
    Ey = 1.0 / max(S_eff[1, 1], 1e-30)
    Ez = 1.0 / max(S_eff[2, 2], 1e-30)

    Gyz = 1.0 / max(S_eff[3, 3], 1e-30)
    Gzx = 1.0 / max(S_eff[4, 4], 1e-30)
    Gxy = 1.0 / max(S_eff[5, 5], 1e-30)

    nu_xy = -S_eff[1, 0] * Ex
    nu_yz = -S_eff[2, 1] * Ey
    nu_zx = -S_eff[0, 2] * Ez

    actual_density = total_solid_vol / vol_total

    # Zener Anisotropy index A_Z = 2 * C44 / (C11 - C12)
    c11_mean = (C_eff[0, 0] + C_eff[1, 1] + C_eff[2, 2]) / 3.0
    c12_mean = (C_eff[0, 1] + C_eff[1, 2] + C_eff[0, 2]) / 3.0
    c44_mean = (C_eff[3, 3] + C_eff[4, 4] + C_eff[5, 5]) / 3.0
    diff = max(c11_mean - c12_mean, 1e-12)
    anisotropy_ratio = 2.0 * c44_mean / diff

    # Measure cubic symmetry residual
    cubic_residual = (
        abs(C_eff[0, 0] - C_eff[1, 1]) + abs(C_eff[1, 1] - C_eff[2, 2]) +
        abs(C_eff[0, 1] - C_eff[1, 2]) + abs(C_eff[1, 2] - C_eff[0, 2]) +
        abs(C_eff[3, 3] - C_eff[4, 4]) + abs(C_eff[4, 4] - C_eff[5, 5])
    ) / max(C_eff[0, 0], 1e-12)

    total_time = time.time() - start_time

    return HomogenizationResult(
        C_effective=C_eff,
        S_effective=S_eff,
        E_x=float(Ex),
        E_y=float(Ey),
        E_z=float(Ez),
        G_yz=float(Gyz),
        G_zx=float(Gzx),
        G_xy=float(Gxy),
        nu_xy=float(nu_xy),
        nu_yz=float(nu_yz),
        nu_zx=float(nu_zx),
        relative_density=float(actual_density),
        anisotropy_ratio=float(anisotropy_ratio),
        cubic_symmetry_residual=float(cubic_residual),
        solve_time=total_time,
    )


def homogenize_tpms_unit_cell(
    config: TPMSConfig,
    resolution: int = 12,
    base_E: float = 210e9,
    base_nu: float = 0.3,
) -> HomogenizationResult:
    """
    Convenience function: generates a high-resolution voxel unit cell of the given TPMS
    and evaluates its effective elasticity tensor via asymptotic homogenization.

    Parameters:
        config: TPMS configuration (morphology, mode, target relative density).
        resolution: Number of voxel elements per axis (e.g. 10 or 12).
        base_E: Base material Young's modulus (Pa).
        base_nu: Base material Poisson's ratio.

    Returns:
        HomogenizationResult instance.
    """
    N = resolution
    Lx, Ly, Lz = config.cell_size

    # Evaluate cell centroid coordinates
    xs = (np.arange(N) + 0.5) * (Lx / N)
    ys = (np.arange(N) + 0.5) * (Ly / N)
    zs = (np.arange(N) + 0.5) * (Lz / N)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # Evaluate signed level-set distance field
    phi = evaluate_graded_tpms_field(pts, config, density_field=config.relative_density)

    # Binary solid mask: 1.0 if solid (phi <= 0), 0.0 if void
    # Use continuous approximation near boundary for smoothed volume fractions
    char_len = min(Lx, Ly, Lz) / N
    solid_frac = 1.0 / (1.0 + np.exp(np.clip(phi / (0.25 * char_len), -30.0, 30.0)))
    densities_3d = solid_frac.reshape(N, N, N)

    result = solve_periodic_homogenization(
        element_densities=densities_3d,
        cell_size=config.cell_size,
        base_E=base_E,
        base_nu=base_nu,
        simp_penalty=1.0,
    )

    return result
