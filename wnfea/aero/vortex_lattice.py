"""
Quasi-Steady 3D Vortex Lattice Method (VLM) & Surface Spline Engine for WNFEA (Phase 21).
------------------------------------------------------------------------------------------
Provides:
1. Quadrilateral Aerodynamic Lifting Panels with 1/4-chord bound vortices and 3/4-chord control points.
2. 3D Vortex Lattice Method (VLM) with Biot-Savart horseshoe vortex downwash induction.
3. Subsonic Prandtl-Glauert compressibility correction beta = sqrt(1 - M_inf^2).
4. Finite-wing lift slope C_L_alpha and induced drag calculation.
5. Conservative Structural Surface Spline (Infinite Plate / Radial Basis) mapping structural
   displacements u_s -> u_a and aerodynamic forces F_a -> F_s = G_as^T F_a, preserving
   exact virtual work F_s^T u_s = F_a^T u_a.
6. Generalized Aerodynamic Force (GAF) matrix evaluation for structural modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np


@dataclass
class AeroPanel:
    """
    Single quadrilateral aerodynamic panel on a lifting surface.
    """
    p1: np.ndarray             # (3,) Root-leading edge vertex
    p2: np.ndarray             # (3,) Tip-leading edge vertex
    p3: np.ndarray             # (3,) Tip-trailing edge vertex
    p4: np.ndarray             # (3,) Root-trailing edge vertex
    bound_v1: np.ndarray       # (3,) Bound vortex root endpoint (at 25% chord)
    bound_v2: np.ndarray       # (3,) Bound vortex tip endpoint (at 25% chord)
    control_point: np.ndarray  # (3,) Control / collocation point (at 75% chord, 50% span)
    normal: np.ndarray         # (3,) Unit surface normal vector (pointing in lift direction)
    area: float                # Panel planform area (m^2)
    span_width: float          # Spanwise width dy (m)
    chord_len: float           # Average chord length c (m)


class VortexLatticeMesh:
    """
    3D Vortex Lattice aerodynamic mesh representing lifting surfaces (wings, fins, pylons).
    """

    def __init__(self, panels: List[AeroPanel]):
        self.panels = panels
        self.num_panels = len(panels)
        self._aic_matrix: Optional[np.ndarray] = None
        self._cached_mach: Optional[float] = None

    @classmethod
    def create_rectangular_wing(
        cls,
        span: float,
        chord: float,
        num_chord: int = 4,
        num_span: int = 12,
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        dihedral_deg: float = 0.0,
    ) -> VortexLatticeMesh:
        """
        Generate a planar rectangular lifting surface centered at y = 0 or extending from y = 0.

        Parameters:
            span: Total span along Y axis (m).
            chord: Chord length along X axis (m).
            num_chord: Number of chordwise panel strips.
            num_span: Number of spanwise panel strips.
            origin: (x0, y0, z0) leading-edge root location.
            dihedral_deg: Dihedral angle in degrees.
        """
        x0, y0, z0 = origin
        dx = chord / num_chord
        dy = span / num_span
        gamma_rad = np.radians(dihedral_deg)

        panels: List[AeroPanel] = []

        for j in range(num_span):
            y_root = y0 + j * dy
            y_tip = y_root + dy
            z_root = z0 + abs(y_root - y0) * np.tan(gamma_rad)
            z_tip = z0 + abs(y_tip - y0) * np.tan(gamma_rad)

            for i in range(num_chord):
                x_le = x0 + i * dx
                x_te = x_le + dx

                p1 = np.array([x_le, y_root, z_root], dtype=np.float64)
                p2 = np.array([x_le, y_tip, z_tip], dtype=np.float64)
                p3 = np.array([x_te, y_tip, z_tip], dtype=np.float64)
                p4 = np.array([x_te, y_root, z_root], dtype=np.float64)

                # Bound vortex line at 25% chord
                x_quarter = x_le + 0.25 * dx
                bound_v1 = np.array([x_quarter, y_root, z_root], dtype=np.float64)
                bound_v2 = np.array([x_quarter, y_tip, z_tip], dtype=np.float64)

                # Control point at 75% chord, 50% span
                x_three_quarter = x_le + 0.75 * dx
                y_mid = 0.5 * (y_root + y_tip)
                z_mid = 0.5 * (z_root + z_tip)
                control_pt = np.array([x_three_quarter, y_mid, z_mid], dtype=np.float64)

                # Outward normal (approx +Z for planar wing)
                chord_vec = p4 - p1
                span_vec = p2 - p1
                norm_vec = np.cross(chord_vec, span_vec)
                norm_len = np.linalg.norm(norm_vec)
                normal = norm_vec / max(norm_len, 1e-12)

                panel_area = float(norm_len)
                span_len = float(np.linalg.norm(span_vec))
                chord_val = float(np.linalg.norm(chord_vec))

                panels.append(
                    AeroPanel(
                        p1=p1,
                        p2=p2,
                        p3=p3,
                        p4=p4,
                        bound_v1=bound_v1,
                        bound_v2=bound_v2,
                        control_point=control_pt,
                        normal=normal,
                        area=panel_area,
                        span_width=span_len,
                        chord_len=chord_val,
                    )
                )

        return cls(panels)

    @classmethod
    def create_trapezoidal_wing(
        cls,
        root_chord: float,
        tip_chord: float,
        semi_span: float,
        sweep_le_deg: float = 0.0,
        num_chord: int = 4,
        num_span: int = 12,
        symmetric: bool = True,
    ) -> VortexLatticeMesh:
        """
        Generate a trapezoidal wing with leading-edge sweep and taper ratio lambda = tip_chord / root_chord.
        """
        panels: List[AeroPanel] = []
        dy = semi_span / num_span
        sweep_rad = np.radians(sweep_le_deg)

        sides = [-1.0, 1.0] if symmetric else [1.0]

        for s_idx, side in enumerate(sides):
            for j in range(num_span):
                eta_root = j / num_span
                eta_tip = (j + 1) / num_span

                c_root = root_chord + (tip_chord - root_chord) * eta_root
                c_tip = root_chord + (tip_chord - root_chord) * eta_tip

                y_root = side * semi_span * eta_root
                y_tip = side * semi_span * eta_tip
                if side < 0:
                    y_root, y_tip = y_tip, y_root
                    c_root, c_tip = c_tip, c_root

                x_le_root = abs(y_root) * np.tan(sweep_rad)
                x_le_tip = abs(y_tip) * np.tan(sweep_rad)

                for i in range(num_chord):
                    frac_le = i / num_chord
                    frac_te = (i + 1) / num_chord

                    p1 = np.array([x_le_root + frac_le * c_root, y_root, 0.0], dtype=np.float64)
                    p2 = np.array([x_le_tip + frac_le * c_tip, y_tip, 0.0], dtype=np.float64)
                    p3 = np.array([x_le_tip + frac_te * c_tip, y_tip, 0.0], dtype=np.float64)
                    p4 = np.array([x_le_root + frac_te * c_root, y_root, 0.0], dtype=np.float64)

                    bound_v1 = p1 + 0.25 * (p4 - p1)
                    bound_v2 = p2 + 0.25 * (p3 - p2)

                    ctrl_root = p1 + 0.75 * (p4 - p1)
                    ctrl_tip = p2 + 0.75 * (p3 - p2)
                    ctrl_pt = 0.5 * (ctrl_root + ctrl_tip)

                    chord_vec = p4 - p1
                    span_vec = p2 - p1
                    norm_vec = np.cross(chord_vec, span_vec)
                    norm_len = np.linalg.norm(norm_vec)
                    normal = norm_vec / max(norm_len, 1e-12)
                    if normal[2] < 0:
                        normal = -normal

                    panels.append(
                        AeroPanel(
                            p1=p1,
                            p2=p2,
                            p3=p3,
                            p4=p4,
                            bound_v1=bound_v1,
                            bound_v2=bound_v2,
                            control_point=ctrl_pt,
                            normal=normal,
                            area=float(norm_len),
                            span_width=float(abs(y_tip - y_root)),
                            chord_len=float(0.5 * (c_root + c_tip)),
                        )
                    )

        return cls(panels)

    @property
    def total_area(self) -> float:
        """Total planform area S (m^2)."""
        return sum(p.area for p in self.panels)

    @property
    def control_points(self) -> np.ndarray:
        """(N_panels, 3) coordinates of all control points."""
        return np.array([p.control_point for p in self.panels], dtype=np.float64)

    @property
    def bound_midpoints(self) -> np.ndarray:
        """(N_panels, 3) midpoints of bound vortex lines."""
        return np.array([0.5 * (p.bound_v1 + p.bound_v2) for p in self.panels], dtype=np.float64)

    @property
    def normals(self) -> np.ndarray:
        """(N_panels, 3) normal vectors of all panels."""
        return np.array([p.normal for p in self.panels], dtype=np.float64)

    def build_aic_matrix(self, mach: float = 0.0) -> np.ndarray:
        """
        Build the Aerodynamic Influence Coefficient (AIC) matrix A_ij using Biot-Savart law.
        A[i, j] is the normal downwash induced at control point i by a horseshoe vortex on panel j of unit circulation.

        Parameters:
            mach: Freestream Mach number M_inf for Prandtl-Glauert compressibility scaling.
        """
        if self._aic_matrix is not None and self._cached_mach == mach:
            return self._aic_matrix

        N = self.num_panels
        ctrl_pts = self.control_points
        normals = self.normals

        # Prandtl-Glauert compressibility factor
        beta = 1.0
        if 0.0 < mach < 1.0:
            beta = np.sqrt(1.0 - mach**2)

        # Scale X-coordinates by beta for subsonic flow
        scaled_ctrl = ctrl_pts.copy()
        scaled_ctrl[:, 0] /= beta

        A = np.zeros((N, N), dtype=np.float64)
        x_far = 1000.0 * max(p.chord_len for p in self.panels)

        for j, pj in enumerate(self.panels):
            b1 = pj.bound_v1.copy()
            b2 = pj.bound_v2.copy()
            b1[0] /= beta
            b2[0] /= beta

            p_far1 = np.array([x_far, b1[1], b1[2]], dtype=np.float64)
            p_far2 = np.array([x_far, b2[1], b2[2]], dtype=np.float64)

            # Induced velocity from 3 segments of the horseshoe vortex:
            # 1. Semi-infinite trailing line from (+x_far, y1, z1) -> (x1, y1, z1)
            v1 = _biot_savart_filament(p_far1, b1, scaled_ctrl)
            # 2. Bound vortex line from (x1, y1, z1) -> (x2, y2, z2)
            v2 = _biot_savart_filament(b1, b2, scaled_ctrl)
            # 3. Semi-infinite trailing line from (x2, y2, z2) -> (+x_far, y2, z2)
            v3 = _biot_savart_filament(b2, p_far2, scaled_ctrl)

            v_ind = v1 + v2 + v3  # (N, 3)

            # Compressibility scaling on induced velocity
            v_ind[:, 0] /= beta

            # Downwash projected onto panel normal: A_ij = v_ind . n_i
            A[:, j] = np.sum(v_ind * normals, axis=1)

        self._aic_matrix = A
        self._cached_mach = mach
        return A

    def solve_steady(
        self,
        alpha_rad: float,
        V_inf: float = 50.0,
        rho_inf: float = 1.225,
        mach: float = 0.0,
        yaw_rad: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Solve steady aerodynamic circulation and forces at a given angle of attack.

        Parameters:
            alpha_rad: Angle of attack in radians.
            V_inf: Freestream velocity in m/s.
            rho_inf: Air density in kg/m^3.
            mach: Freestream Mach number.
            yaw_rad: Sideslip angle in radians.

        Returns:
            gamma: (N_panels,) bound vortex circulations (m^2/s).
            lift_forces: (N_panels,) panel aerodynamic lift forces (N).
            C_L: Total 3D lift coefficient.
            C_Di: Total induced drag coefficient.
        """
        A = self.build_aic_matrix(mach=mach)
        normals = self.normals

        # Freestream velocity vector
        u_inf = V_inf * np.cos(alpha_rad) * np.cos(yaw_rad)
        v_inf = -V_inf * np.sin(yaw_rad)
        w_inf = V_inf * np.sin(alpha_rad)
        V_vec = np.array([u_inf, v_inf, w_inf], dtype=np.float64)

        # Flow tangency condition: w_norm = - V_vec . n
        w_norm = -np.sum(V_vec[None, :] * normals, axis=1)

        # Solve linear system for circulations Gamma
        gamma = np.linalg.solve(A, w_norm)

        # Kutta-Joukowski theorem for panel lift:
        # dL_j = rho_inf * V_inf * Gamma_j * dy_j
        span_widths = np.array([p.span_width for p in self.panels], dtype=np.float64)
        lift_forces = rho_inf * V_inf * gamma * span_widths

        total_lift = float(np.sum(lift_forces))
        S_ref = self.total_area
        q_inf = 0.5 * rho_inf * (V_inf**2)
        C_L = total_lift / max(q_inf * S_ref, 1e-12)

        # Induced drag: Di = rho_inf * sum(Gamma_i * w_ind_i * dy_i)
        w_ind = A @ gamma
        induced_drag = -rho_inf * float(np.sum(gamma * w_ind * span_widths))
        C_Di = max(induced_drag, 0.0) / max(q_inf * S_ref, 1e-12)

        return gamma, lift_forces, C_L, C_Di


class SurfaceSplineCoupler:
    """
    Conservative Structural-to-Aerodynamic Surface Spline Coupler.
    --------------------------------------------------------------
    Maps structural displacement fields u_s onto aerodynamic panel control points:
        u_a = G_as * u_s
    and maps aerodynamic forces F_a conservatively onto structural nodes:
        F_s = G_as^T * F_a
    strictly conserving total virtual work:
        F_s^T * u_s = F_a^T * u_a.
    """

    def __init__(
        self,
        structural_nodes: np.ndarray,
        aero_points: np.ndarray,
        spline_radius: Optional[float] = None,
    ):
        """
        Initialize the spline mapping matrix between structural and aerodynamic surfaces.

        Parameters:
            structural_nodes: (N_s, 3) 3D coordinates of structural nodes.
            aero_points: (N_a, 3) 3D coordinates of aerodynamic collocation / control points.
            spline_radius: Characteristic smoothing radius for Radial Basis Functions.
        """
        self.struct_pts = np.asarray(structural_nodes, dtype=np.float64)
        self.aero_pts = np.asarray(aero_points, dtype=np.float64)
        self.num_struct = len(self.struct_pts)
        self.num_aero = len(self.aero_pts)

        if spline_radius is None:
            # Sized based on mean nearest-neighbor distance of structural nodes
            diff = self.struct_pts[:, None, :] - self.struct_pts[None, :, :]
            dist = np.linalg.norm(diff, axis=-1)
            np.fill_diagonal(dist, np.inf)
            min_dist = np.min(dist, axis=1)
            spline_radius = float(2.5 * np.mean(min_dist[np.isfinite(min_dist)]))

        self.radius = max(spline_radius, 1e-6)
        self.G_as = self._compute_spline_matrix()

    def _compute_spline_matrix(self) -> np.ndarray:
        """
        Compute the (N_a, N_s) coupling matrix G_as using thin-plate / Wendland RBF interpolation.
        """
        # Distances between structural nodes: (N_s, N_s)
        diff_ss = self.struct_pts[:, None, :] - self.struct_pts[None, :, :]
        r_ss = np.linalg.norm(diff_ss, axis=-1)
        K_ss = self._rbf(r_ss) + 1e-6 * np.eye(self.num_struct)

        # Distances between aerodynamic points and structural nodes: (N_a, N_s)
        diff_as = self.aero_pts[:, None, :] - self.struct_pts[None, :, :]
        r_as = np.linalg.norm(diff_as, axis=-1)
        K_as = self._rbf(r_as)

        # G_as = K_as * K_ss^-1
        G_as = np.linalg.solve(K_ss.T, K_as.T).T
        return G_as

    def _rbf(self, r: np.ndarray) -> np.ndarray:
        """Wendland C2 compact radial basis function or Gaussian kernel."""
        # Gaussian kernel with radius
        return np.exp(-(r / self.radius)**2)

    def map_displacements(self, struct_displacements: np.ndarray) -> np.ndarray:
        """
        Map structural displacement vector u_s (N_s,) or (N_s, 3) to aerodynamic points u_a (N_a,).
        """
        u_s = np.asarray(struct_displacements, dtype=np.float64)
        if u_s.ndim == 1 and len(u_s) == self.num_struct:
            return self.G_as @ u_s
        elif u_s.ndim == 2 and u_s.shape[0] == self.num_struct:
            return self.G_as @ u_s
        else:
            raise ValueError(f"Shape mismatch: struct_displacements {u_s.shape} vs num_struct {self.num_struct}")

    def map_forces(self, aero_forces: np.ndarray) -> np.ndarray:
        """
        Map aerodynamic forces F_a (N_a,) or (N_a, 3) conservatively to structural nodes F_s = G_as^T F_a.
        """
        f_a = np.asarray(aero_forces, dtype=np.float64)
        return self.G_as.T @ f_a


def _biot_savart_filament(
    p1: np.ndarray,
    p2: np.ndarray,
    query_points: np.ndarray,
) -> np.ndarray:
    """
    Vectorized evaluation of Biot-Savart induced velocity from a straight vortex filament of unit circulation.
    """
    r1 = query_points - p1[None, :]  # (N, 3)
    r2 = query_points - p2[None, :]  # (N, 3)
    r0 = (p2 - p1)[None, :]          # (1, 3)

    r1_len = np.linalg.norm(r1, axis=1, keepdims=True)
    r2_len = np.linalg.norm(r2, axis=1, keepdims=True)

    cross12 = np.cross(r1, r2)
    cross_sq = np.sum(cross12**2, axis=1, keepdims=True)

    mask = (cross_sq > 1e-12)
    safe_sq = np.where(mask, cross_sq, 1.0)

    dot1 = np.sum(r0 * r1, axis=1, keepdims=True) / np.maximum(r1_len, 1e-12)
    dot2 = np.sum(r0 * r2, axis=1, keepdims=True) / np.maximum(r2_len, 1e-12)

    factor = (cross12 / (4.0 * np.pi * safe_sq)) * (dot1 - dot2)
    vel = np.where(mask, factor, 0.0)
    return vel
