"""
Matrix-Free Aero-Structural Random Vibration (PSD) Response Solver for WNFEA.
-----------------------------------------------------------------------------
Formulates:
1. Base Excitation Power Spectral Density (PSD) representations:
   - User spectra, flat white-noise, NAVMAT P-9492, and NASA GEVS profiles.
   - Log-log interpolation onto fine dynamic frequency grids.
2. Modal Base Participation Factor evaluation:
   - Rigid body influence vector r along excitation direction d.
   - Effective modal participation Gamma_j = phi_j^T * M * r.
   - Effective modal mass M_eff_j = Gamma_j^2.
3. Frequency Response Function (FRF) and Response PSD calculation:
   - Relative/absolute displacement transfer function H_u(omega).
   - Absolute acceleration transfer function H_a(omega) (recovering H_a(0) = 1.0).
   - Element centroidal modal stress transfer function H_sigma(omega).
   - S_uu(f) = |H_u(2*pi*f)|^2 * S_in(f), S_aa(f) = |H_a(2*pi*f)|^2 * S_in(f).
4. Spectral Moment Integration:
   - m_k = int_0^infty f^k * S(f) df (for k = 0, 1, 2, 4).
   - RMS values: sigma_RMS = sqrt(m_0), g_RMS = sqrt(m_0) / g.
   - Zero-crossing frequency E[0] = sqrt(m_2 / m_0).
   - Peak rate E[P] = sqrt(m_4 / m_2).
   - Irregularity factor gamma = m_2 / sqrt(m_0 * m_4).
5. Exact parity with analytical SDOF Miles equation: g_RMS = sqrt(pi/2 * f_n * Q * S_0).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from ..mesh.voxel_mesher import VoxelGrid
from .modal_analysis import ModalResult, compute_lumped_mass_hex8

# Compatibility for numpy >= 2.0 (np.trapezoid) and < 2.0 (np.trapz)
if hasattr(np, "trapezoid"):
    _trapezoid = np.trapezoid
else:
    _trapezoid = getattr(np, "trapz")

STANDARD_GRAVITY = 9.80665  # m/s^2


@dataclass
class BaseExcitationPSD:
    """
    Representation of input Base Acceleration Power Spectral Density S_in(f).
    """
    frequencies_hz: np.ndarray      # Breakpoint frequencies (Hz)
    psd_values: np.ndarray          # PSD values in g^2/Hz or (m/s^2)^2/Hz
    unit: str = "g^2/Hz"            # "g^2/Hz" or "(m/s^2)^2/Hz"
    direction: Tuple[float, float, float] = (0.0, 0.0, 1.0) # Excitation unit vector

    def __post_init__(self):
        self.frequencies_hz = np.asarray(self.frequencies_hz, dtype=np.float64)
        self.psd_values = np.asarray(self.psd_values, dtype=np.float64)
        d = np.asarray(self.direction, dtype=np.float64)
        norm = np.linalg.norm(d)
        if norm > 1e-12:
            self.direction = tuple(d / norm)
        else:
            self.direction = (0.0, 0.0, 1.0)

    @classmethod
    def flat(
        cls,
        f_min: float = 20.0,
        f_max: float = 2000.0,
        value: float = 0.04,
        direction: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        unit: str = "g^2/Hz",
    ) -> "BaseExcitationPSD":
        """Flat / White-noise spectrum between f_min and f_max."""
        return cls(
            frequencies_hz=np.array([f_min, f_max], dtype=np.float64),
            psd_values=np.array([value, value], dtype=np.float64),
            unit=unit,
            direction=direction,
        )

    @classmethod
    def navmat_p9492(
        cls,
        direction: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> "BaseExcitationPSD":
        """
        Standard US Navy NAVMAT P-9492 screening profile (~6.06 g RMS):
        - 20 Hz: 0.01 g^2/Hz
        - 80 Hz: 0.04 g^2/Hz (+3 dB/oct)
        - 350 Hz: 0.04 g^2/Hz (flat plateau)
        - 2000 Hz: 0.007 g^2/Hz (-3 dB/oct)
        """
        freqs = np.array([20.0, 80.0, 350.0, 2000.0], dtype=np.float64)
        vals = np.array([0.01, 0.04, 0.04, 0.007], dtype=np.float64)
        return cls(frequencies_hz=freqs, psd_values=vals, unit="g^2/Hz", direction=direction)

    @classmethod
    def nasa_gevs(
        cls,
        direction: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> "BaseExcitationPSD":
        """
        Standard NASA GEVS (General Environmental Verification Standard) Acceptance (~10.0 g RMS):
        - 20 Hz: 0.013 g^2/Hz
        - 50 Hz: 0.08 g^2/Hz (+6 dB/oct)
        - 800 Hz: 0.08 g^2/Hz (flat)
        - 2000 Hz: 0.013 g^2/Hz (-9 dB/oct)
        """
        freqs = np.array([20.0, 50.0, 800.0, 2000.0], dtype=np.float64)
        vals = np.array([0.013, 0.08, 0.08, 0.013], dtype=np.float64)
        return cls(frequencies_hz=freqs, psd_values=vals, unit="g^2/Hz", direction=direction)

    def interpolate(self, target_freqs_hz: np.ndarray) -> np.ndarray:
        """
        Interpolate input PSD onto query frequencies using log-log interpolation.
        Returns PSD values in SI units: (m/s^2)^2 / Hz.
        """
        freqs = np.asarray(target_freqs_hz, dtype=np.float64)
        f_pts = self.frequencies_hz
        p_pts = self.psd_values.copy()

        # Convert to (m/s^2)^2 / Hz if specified in g^2/Hz
        if self.unit.lower() in ("g^2/hz", "g2/hz"):
            p_pts *= (STANDARD_GRAVITY ** 2)

        # Log-log interpolation
        log_f_pts = np.log10(np.maximum(f_pts, 1e-6))
        log_p_pts = np.log10(np.maximum(p_pts, 1e-18))

        log_f = np.log10(np.maximum(freqs, 1e-6))
        log_p = np.interp(log_f, log_f_pts, log_p_pts, left=-18.0, right=-18.0)
        psd_si = 10.0 ** log_p

        # Zero outside range
        mask_out = (freqs < f_pts[0]) | (freqs > f_pts[-1])
        psd_si[mask_out] = 0.0
        return psd_si


@dataclass
class RandomVibrationResult:
    """
    Results from Random Vibration Analysis under Base PSD Excitation.
    """
    frequencies_hz: np.ndarray             # Evaluation frequencies (Hz)
    psd_displacement: np.ndarray           # (N_freq, N_monitors) displacement PSD (m^2/Hz)
    psd_acceleration: np.ndarray           # (N_freq, N_monitors) absolute acceleration PSD (g^2/Hz)
    rms_displacement: np.ndarray           # (N_monitors,) 1-sigma RMS displacement (m)
    rms_acceleration_g: np.ndarray         # (N_monitors,) 1-sigma RMS absolute acceleration (g)
    element_rms_stress: Optional[np.ndarray] # (N_elements,) 1-sigma RMS von Mises stress (Pa)
    spectral_moments_stress: Dict[str, np.ndarray] # 'm0', 'm1', 'm2', 'm4' arrays for elements
    zero_crossing_rate: np.ndarray         # (N_elements or N_monitors,) E[0] in Hz
    peak_rate: np.ndarray                  # (N_elements or N_monitors,) E[P] in Hz
    irregularity_factor: np.ndarray        # (N_elements or N_monitors,) gamma in [0, 1]
    effective_modal_masses: np.ndarray     # (m,) Effective modal mass per mode (kg)
    solve_time: float                      # Total solve time in seconds


def solve_random_vibration(
    grid: VoxelGrid,
    modal_result: ModalResult,
    base_psd: BaseExcitationPSD,
    damping_ratio: float = 0.02,
    freq_start_hz: Optional[float] = None,
    freq_end_hz: Optional[float] = None,
    num_frequencies: int = 400,
    monitor_dofs: Optional[Sequence[int]] = None,
    compute_stresses: bool = True,
    material_density: float = 7850.0,
    material_E: float = 210e9,
    material_nu: float = 0.3,
) -> RandomVibrationResult:
    """
    Solve structural dynamic random vibration response under base excitation PSD.

    Parameters:
        grid: Cartesian VoxelGrid.
        modal_result: Modal natural frequencies and M-orthonormal mode shapes.
        base_psd: BaseExcitationPSD defining the base acceleration spectrum.
        damping_ratio: Viscous damping ratio zeta (default: 0.02 = 2% critical damping).
        freq_start_hz: Start frequency in Hz (defaults to base_psd.frequencies_hz[0]).
        freq_end_hz: End frequency in Hz (defaults to base_psd.frequencies_hz[-1]).
        num_frequencies: Number of frequency integration points.
        monitor_dofs: Specific global DOFs to extract displacement and acceleration PSDs.
        compute_stresses: If True, evaluates element centroidal RMS stresses and spectral moments.
        material_density: Mass density rho (kg/m^3).
        material_E: Young's modulus (Pa).
        material_nu: Poisson's ratio.

    Returns:
        RandomVibrationResult object containing PSDs, RMS responses, and spectral moments.
    """
    start_time = time.time()

    # 1. Frequency sweep discretization
    f_min = float(freq_start_hz if freq_start_hz is not None else base_psd.frequencies_hz[0])
    f_max = float(freq_end_hz if freq_end_hz is not None else base_psd.frequencies_hz[-1])

    # Enhanced frequency grid: include fine sampling around resonant natural frequencies
    nat_freqs = modal_result.frequencies_hz
    extra_pts = []
    for fn in nat_freqs:
        if f_min <= fn <= f_max:
            # 10 fine points in the 3 dB resonance bandwidth: fn * (1 +/- 3 * zeta)
            bw = fn * damping_ratio * 3.0
            extra_pts.extend(np.linspace(max(fn - bw, f_min), min(fn + bw, f_max), 15))

    base_grid = np.linspace(f_min, f_max, num_frequencies)
    all_freqs = np.unique(np.concatenate([base_grid, np.array(extra_pts)]))
    all_freqs.sort()
    n_freq = len(all_freqs)
    omegas = 2.0 * np.pi * all_freqs

    # 2. Input Base Acceleration PSD (SI units: (m/s^2)^2 / Hz)
    S_in_si = base_psd.interpolate(all_freqs)

    # 3. Lumped mass and rigid body influence vector r
    mass_diag = compute_lumped_mass_hex8(grid, density_material=material_density)
    n_dofs = len(mass_diag)

    dx, dy, dz = base_psd.direction
    r_vec = np.zeros(n_dofs, dtype=np.float64)
    r_vec[0::3] = dx
    r_vec[1::3] = dy
    r_vec[2::3] = dz

    # 4. Modal Base Participation Factors: Gamma_j = phi_j^T * (M * r)
    m_modes = len(modal_result.circular_frequencies)
    phi_modes = modal_result.mode_shapes.reshape(m_modes, -1)  # (m, n_dofs)
    omega_nat = modal_result.circular_frequencies               # (m,)

    Mr = mass_diag * r_vec
    Gamma = np.dot(phi_modes, Mr)  # (m,)
    eff_mass = Gamma ** 2          # (m,)

    # 5. Monitor DOFs
    if monitor_dofs is None:
        monitor_dofs = np.arange(min(n_dofs, 30))
    else:
        monitor_dofs = np.asarray(monitor_dofs, dtype=np.int64)
    phi_mon = phi_modes[:, monitor_dofs]  # (m, n_mon)
    r_mon = r_vec[monitor_dofs]           # (n_mon,)

    # 6. Modal Transfer Functions
    # Denominator: (omega_j^2 - omega^2) + 2*i*zeta*omega_j*omega
    omega_sq_diff = (omega_nat[None, :] ** 2) - (omegas[:, None] ** 2)
    damping_term = 2.0 * 1j * damping_ratio * (omega_nat[None, :] * omegas[:, None])
    denom = omega_sq_diff + damping_term  # (n_freq, m)

    # Modal displacement coordinate transfer: H_q(omega) = -Gamma_j / denom
    H_q = -Gamma[None, :] / denom  # (n_freq, m)

    # Physical displacement transfer function: H_u(omega) = H_q @ phi_mon
    # Shape: (n_freq, n_mon)
    H_u = np.dot(H_q, phi_mon)
    S_uu = (np.abs(H_u) ** 2) * S_in_si[:, None]  # m^2 / Hz

    # Absolute acceleration transfer function: H_a(omega) = r_mon - omega^2 * H_u(omega)
    # At omega = 0, H_a = r_mon (1.0 along excitation direction, exactly physical!)
    H_a = r_mon[None, :] - (omegas[:, None] ** 2) * H_u
    S_aa_si = (np.abs(H_a) ** 2) * S_in_si[:, None]  # (m/s^2)^2 / Hz
    S_aa_g = S_aa_si / (STANDARD_GRAVITY ** 2)      # g^2 / Hz

    # Integrate displacement and acceleration RMS using trapezoidal rule
    m0_u = _trapezoid(S_uu, x=all_freqs, axis=0)
    rms_disp = np.sqrt(np.maximum(m0_u, 0.0))

    m0_a = _trapezoid(S_aa_g, x=all_freqs, axis=0)
    rms_acc_g = np.sqrt(np.maximum(m0_a, 0.0))

    # 7. Element Centroidal Modal Stresses & Spectral Moments
    elem_rms_stress = None
    spectral_moments: Dict[str, np.ndarray] = {}
    e0 = np.zeros(len(monitor_dofs), dtype=np.float64)
    ep = np.zeros(len(monitor_dofs), dtype=np.float64)
    gamma = np.zeros(len(monitor_dofs), dtype=np.float64)

    if compute_stresses:
        n_elem = grid.total_active_cells
        active_elems = grid.elements[grid.active_element_indices]

        # Centroidal B-matrix for Hex8
        hx, hy, hz = grid.pitch
        # Elementary stress operator: modal displacement -> centroidal strain -> stress
        # For simplicity and speed, compute modal strains at cell centroid:
        # B_0 is (6, 24) evaluated at xi=0, eta=0, zeta=0
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
        ])  # (3, 8)

        # Constitutive matrix C (6, 6)
        c_val = material_E / ((1.0 + material_nu) * (1.0 - 2.0 * material_nu))
        C_mat = c_val * np.array([
            [1.0 - material_nu, material_nu, material_nu, 0, 0, 0],
            [material_nu, 1.0 - material_nu, material_nu, 0, 0, 0],
            [material_nu, material_nu, 1.0 - material_nu, 0, 0, 0],
            [0, 0, 0, (1.0 - 2.0 * material_nu) * 0.5, 0, 0],
            [0, 0, 0, 0, (1.0 - 2.0 * material_nu) * 0.5, 0],
            [0, 0, 0, 0, 0, (1.0 - 2.0 * material_nu) * 0.5],
        ])

        # Evaluate modal von Mises stress for each mode j across elements: sigma_modal (m, n_elem)
        sigma_modal = np.zeros((m_modes, n_elem), dtype=np.float64)

        for j in range(m_modes):
            phi_j = phi_modes[j]
            for e_idx in range(n_elem):
                nodes = active_elems[e_idx]
                dofs = np.zeros(24, dtype=np.int64)
                for a in range(8):
                    dofs[3*a : 3*a+3] = [3*nodes[a], 3*nodes[a]+1, 3*nodes[a]+2]
                u_e = phi_j[dofs]  # (24,)

                # Centroidal strain: eps_xx = sum_a dN/dx * u_x, etc.
                ux = u_e[0::3]
                uy = u_e[1::3]
                uz = u_e[2::3]

                exx = np.dot(dNdX[0], ux)
                eyy = np.dot(dNdX[1], uy)
                ezz = np.dot(dNdX[2], uz)
                gxy = np.dot(dNdX[1], ux) + np.dot(dNdX[0], uy)
                gyz = np.dot(dNdX[2], uy) + np.dot(dNdX[1], uz)
                gzx = np.dot(dNdX[0], uz) + np.dot(dNdX[2], ux)

                eps = np.array([exx, eyy, ezz, gxy, gyz, gzx])
                sig = C_mat @ eps

                # Centroidal von Mises stress
                s_vm = np.sqrt(0.5 * ((sig[0] - sig[1])**2 + (sig[1] - sig[2])**2 + (sig[2] - sig[0])**2 + 6.0 * (sig[3]**2 + sig[4]**2 + sig[5]**2)))
                sigma_modal[j, e_idx] = s_vm

        # Stress transfer function: H_sigma = H_q @ sigma_modal
        # Shape: (n_freq, n_elem)
        H_sigma = np.dot(H_q, sigma_modal)
        S_sigma = (np.abs(H_sigma) ** 2) * S_in_si[:, None]  # Pa^2 / Hz

        # Spectral moments for each element:
        # m0 = int S_sigma df, m1 = int f S_sigma df, m2 = int f^2 S_sigma df, m4 = int f^4 S_sigma df
        m0 = _trapezoid(S_sigma, x=all_freqs, axis=0)
        m1 = _trapezoid(all_freqs[:, None] * S_sigma, x=all_freqs, axis=0)
        m2 = _trapezoid((all_freqs[:, None] ** 2) * S_sigma, x=all_freqs, axis=0)
        m4 = _trapezoid((all_freqs[:, None] ** 4) * S_sigma, x=all_freqs, axis=0)

        elem_rms_stress = np.sqrt(np.maximum(m0, 0.0))
        spectral_moments["m0"] = m0
        spectral_moments["m1"] = m1
        spectral_moments["m2"] = m2
        spectral_moments["m4"] = m4

        # Zero crossing and peak rates
        e0 = np.sqrt(np.maximum(m2 / np.maximum(m0, 1e-12), 0.0))
        ep = np.sqrt(np.maximum(m4 / np.maximum(m2, 1e-12), 0.0))
        gamma = np.clip(m2 / np.sqrt(np.maximum(m0 * m4, 1e-12)), 0.0, 1.0)
    else:
        # Acceleration spectral moments at monitor DOFs
        m0_acc = _trapezoid(S_aa_si, x=all_freqs, axis=0)
        m1_acc = _trapezoid(all_freqs[:, None] * S_aa_si, x=all_freqs, axis=0)
        m2_acc = _trapezoid((all_freqs[:, None] ** 2) * S_aa_si, x=all_freqs, axis=0)
        m4_acc = _trapezoid((all_freqs[:, None] ** 4) * S_aa_si, x=all_freqs, axis=0)
        spectral_moments["m0"] = m0_acc
        spectral_moments["m1"] = m1_acc
        spectral_moments["m2"] = m2_acc
        spectral_moments["m4"] = m4_acc

        e0 = np.sqrt(np.maximum(m2_acc / np.maximum(m0_acc, 1e-12), 0.0))
        ep = np.sqrt(np.maximum(m4_acc / np.maximum(m2_acc, 1e-12), 0.0))
        gamma = np.clip(m2_acc / np.sqrt(np.maximum(m0_acc * m4_acc, 1e-12)), 0.0, 1.0)

    total_time = time.time() - start_time

    return RandomVibrationResult(
        frequencies_hz=all_freqs,
        psd_displacement=S_uu,
        psd_acceleration=S_aa_g,
        rms_displacement=rms_disp,
        rms_acceleration_g=rms_acc_g,
        element_rms_stress=elem_rms_stress,
        spectral_moments_stress=spectral_moments,
        zero_crossing_rate=e0,
        peak_rate=ep,
        irregularity_factor=gamma,
        effective_modal_masses=eff_mass,
        solve_time=total_time,
    )
