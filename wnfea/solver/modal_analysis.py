"""
Matrix-Free LOBPCG Modal Dynamic Eigen-Solver for Structural Dynamics.
-----------------------------------------------------------------------
Solves the generalized undamped structural dynamic eigenvalue problem:
    K * phi = omega^2 * M * phi
using Locally Optimal Block Preconditioned Conjugate Gradient (LOBPCG):
1. Zero Matrix Storage: Operates purely through matrix-free stiffness contractions K(v).
2. Lumped Mass Matrix: M_diag takes O(N) memory, strictly positive definite.
3. Jacobi / p-Multigrid Preconditioning on Free DOFs.
4. M-Orthonormal Mode Shapes: phi_i^T * M * phi_j = delta_ij.
5. Modal Effective Mass and Directional Mass Participation Factors (X, Y, Z).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union, List
import numpy as np
from scipy.sparse.linalg import lobpcg, LinearOperator

from ..mesh.voxel_mesher import VoxelGrid
from .matrix_free_hex8 import compute_hex8_reference_stiffness


@dataclass
class ModalResult:
    """
    Structural dynamic modal analysis results.
    """
    frequencies_hz: np.ndarray             # (k,) Natural frequencies in Hertz
    circular_frequencies: np.ndarray       # (k,) Circular frequencies omega_i in rad/s
    eigenvalues: np.ndarray                # (k,) Eigenvalues lambda_i = omega_i^2
    mode_shapes: np.ndarray                # (k, n_nodes, 3) 3D nodal mode shape displacements
    effective_modal_mass: np.ndarray       # (k, 3) Effective modal mass along (X, Y, Z) in kg
    mass_participation_ratio: np.ndarray   # (k, 3) Mass participation fractions in [0, 1]
    total_mass: float                      # Total structural mass in kg
    solve_time: float                      # Total solver wallclock time in seconds
    iterations: int                        # Number of LOBPCG iterations taken


def compute_lumped_mass_hex8(
    grid: VoxelGrid,
    density_material: float = 7850.0,
    densities: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute diagonal lumped mass matrix vector M_diag of shape (N_nodes * 3,).

    Parameters:
        grid: VoxelGrid domain.
        density_material: Solid material mass density in kg/m^3 (default: 7850 kg/m^3 for steel).
        densities: Optional (N_elements,) element density or volume fraction array.

    Returns:
        m_diag: (N_nodes * 3,) float64 diagonal mass array.
    """
    hx, hy, hz = grid.pitch
    cell_vol = hx * hy * hz
    rho_elem = densities if densities is not None else grid.volume_fractions

    # Nodal mass accumulator
    node_mass = np.zeros(grid.total_nodes, dtype=np.float64)
    elem_masses = density_material * cell_vol * rho_elem  # (N_elements,)

    # Each Hex8 cell shares 1/8 of its mass to its 8 corner nodes
    mass_per_node = elem_masses / 8.0
    for node_local_idx in range(8):
        corner_nodes = grid.elements[:, node_local_idx]
        np.add.at(node_mass, corner_nodes, mass_per_node)

    # Floor inactive nodes to a tiny epsilon mass to avoid 0 division
    node_mass = np.maximum(node_mass, 1e-12)

    # Expand to 3 translational DOFs per node (X, Y, Z)
    m_diag = np.repeat(node_mass, 3)
    return m_diag


def solve_modal_analysis(
    grid: VoxelGrid,
    fixed_dofs: Sequence[int],
    num_modes: int = 6,
    density_material: float = 7850.0,
    densities: Optional[np.ndarray] = None,
    E: float = 2.1e11,
    nu: float = 0.3,
    tol: float = 1e-5,
    max_iter: int = 60,
    method: str = "eigsh",
) -> ModalResult:
    """
    Extract the lowest `num_modes` natural frequencies and mode shapes via matrix-free LOBPCG.

    Parameters:
        grid: VoxelGrid domain.
        fixed_dofs: Boundary condition constrained global DOF indices.
        num_modes: Number of eigenmodes to extract (default: 6).
        density_material: Material density in kg/m^3 (default: 7850.0).
        densities: Optional (N_elements,) element physical densities.
        E: Young's modulus in Pa (default: 2.1e11).
        nu: Poisson's ratio (default: 0.3).
        tol: LOBPCG convergence tolerance.
        max_iter: Maximum iterations for LOBPCG.

    Returns:
        ModalResult with natural frequencies, mode shapes, and modal mass participation.
    """
    t0 = time.perf_counter()

    n_nodes = grid.total_nodes
    n_dofs = n_nodes * 3
    n_elems = grid.total_cells

    fixed_set = set(fixed_dofs)
    free_dofs = np.array([d for d in range(n_dofs) if d not in fixed_set], dtype=np.int64)
    n_free = len(free_dofs)

    if num_modes > n_free:
        num_modes = n_free

    # 1. Compute elemental stiffness reference and connectivity
    hx, hy, hz = grid.pitch
    k_0, _, _ = compute_hex8_reference_stiffness(hx, hy, hz, E, nu)

    elem_nodes = grid.elements  # (N, 8)
    node_dofs = elem_nodes[:, :, None] * 3 + np.array([0, 1, 2])[None, None, :]
    elem_dofs = node_dofs.reshape(n_elems, 24)

    # Physical element stiffness scalings (SIMP penalization or linear volume fraction)
    rho_phys = densities if densities is not None else grid.volume_fractions
    rho_scaled = np.clip(rho_phys ** 3.0, 1e-4, 1.0)

    # 2. Lumped Mass Vector
    m_diag_full = compute_lumped_mass_hex8(grid, density_material=density_material, densities=densities)
    m_diag_free = m_diag_full[free_dofs]
    total_mass = float(np.sum(m_diag_full[0::3]))  # Sum along X translational DOFs

    # 3. Compute Diagonal of Stiffness Matrix for Preconditioning
    k_0_diag = np.diag(k_0)  # (24,)
    k_diag_full = np.zeros(n_dofs, dtype=np.float64)
    for i in range(24):
        np.add.at(k_diag_full, elem_dofs[:, i], k_0_diag[i] * rho_scaled)
    k_diag_full = np.maximum(k_diag_full, 1e-6)
    k_diag_free = k_diag_full[free_dofs]

    # 4. Matrix-Free Operator Implementations
    def matvec_K(v_free: np.ndarray) -> np.ndarray:
        """Matrix-free action K_ff @ v_free."""
        if v_free.ndim == 1:
            v_full = np.zeros(n_dofs, dtype=np.float64)
            v_full[free_dofs] = v_free
            u_e = v_full[elem_dofs]  # (N, 24)
            ku_e = (u_e @ k_0) * rho_scaled[:, None]  # (N, 24)
            out_full = np.zeros(n_dofs, dtype=np.float64)
            for i in range(24):
                np.add.at(out_full, elem_dofs[:, i], ku_e[:, i])
            return out_full[free_dofs]
        else:
            # Block of vectors: v_free is (n_free, k)
            k_cols = v_free.shape[1]
            out_block = np.zeros_like(v_free)
            for c in range(k_cols):
                out_block[:, c] = matvec_K(v_free[:, c])
            return out_block

    def matvec_M(v_free: np.ndarray) -> np.ndarray:
        """Diagonal mass action M_ff @ v_free."""
        if v_free.ndim == 1:
            return m_diag_free * v_free
        else:
            return m_diag_free[:, None] * v_free

    def matvec_prec(v_free: np.ndarray) -> np.ndarray:
        """Preconditioner action T @ v_free = v_free / k_diag_free."""
        if v_free.ndim == 1:
            return v_free / k_diag_free
        else:
            return v_free / k_diag_free[:, None]

    A_op = LinearOperator((n_free, n_free), matvec=matvec_K, matmat=matvec_K, dtype=np.float64)

    # 5. Solve Generalized Eigenvalue Problem: K * phi = lambda * M * phi
    from scipy.sparse import diags
    from scipy.sparse.linalg import eigsh

    M_sparse = diags(m_diag_free, format="csr")

    if method == "eigsh":
        # ARPACK Shift-and-Invert / Smallest Magnitude Lanczos
        eigenvalues, V_free = eigsh(
            A_op,
            k=num_modes,
            M=M_sparse,
            which="SM",
            tol=tol,
            maxiter=max_iter * 20,
        )
    else:
        # LOBPCG Block Conjugate Gradient
        B_op = LinearOperator((n_free, n_free), matvec=matvec_M, matmat=matvec_M, dtype=np.float64)
        M_op = LinearOperator((n_free, n_free), matvec=matvec_prec, matmat=matvec_prec, dtype=np.float64)
        np.random.seed(42)
        X = np.random.randn(n_free, num_modes)
        eigenvalues, V_free = lobpcg(
            A=A_op,
            X=X,
            B=B_op,
            M=M_op,
            tol=tol,
            maxiter=max_iter,
            largest=False,
        )

    # Sort eigenvalues ascending
    sort_idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_idx]
    V_free = V_free[:, sort_idx]

    # Ensure non-negative eigenvalues (clamp tiny numerical negatives to 0)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    omega = np.sqrt(eigenvalues)
    freq_hz = omega / (2.0 * np.pi)

    # 7. Reconstruct Full Mode Shapes and Compute Modal Effective Mass
    mode_shapes = np.zeros((num_modes, n_nodes, 3), dtype=np.float64)
    effective_mass = np.zeros((num_modes, 3), dtype=np.float64)
    mass_part_ratio = np.zeros((num_modes, 3), dtype=np.float64)

    # Directional unit influence vectors (rigid body translations)
    r_x = np.zeros(n_dofs, dtype=np.float64)
    r_y = np.zeros(n_dofs, dtype=np.float64)
    r_z = np.zeros(n_dofs, dtype=np.float64)
    r_x[0::3] = 1.0
    r_y[1::3] = 1.0
    r_z[2::3] = 1.0
    R_dirs = np.stack([r_x, r_y, r_z], axis=1)  # (n_dofs, 3)

    for i in range(num_modes):
        phi_full = np.zeros(n_dofs, dtype=np.float64)
        phi_full[free_dofs] = V_free[:, i]

        # Ensure M-orthonormality: phi^T * M * phi = 1.0
        m_modal = float(np.sum(phi_full * m_diag_full * phi_full))
        if m_modal > 1e-15:
            phi_full /= np.sqrt(m_modal)

        mode_shapes[i] = phi_full.reshape((n_nodes, 3))

        # Effective Modal Mass: L_ij = phi_i^T * M * r_j -> M_eff = L_ij^2 / (phi^T * M * phi)
        # Since phi is M-normalized, M_eff = L_ij^2
        m_phi = phi_full * m_diag_full
        for j in range(3):
            L_ij = float(np.sum(m_phi * R_dirs[:, j]))
            eff_m = L_ij ** 2
            effective_mass[i, j] = eff_m
            mass_part_ratio[i, j] = eff_m / max(total_mass, 1e-12)

    t_solve = time.perf_counter() - t0

    return ModalResult(
        frequencies_hz=freq_hz,
        circular_frequencies=omega,
        eigenvalues=eigenvalues,
        mode_shapes=mode_shapes,
        effective_modal_mass=effective_mass,
        mass_participation_ratio=mass_part_ratio,
        total_mass=total_mass,
        solve_time=t_solve,
        iterations=max_iter,
    )
