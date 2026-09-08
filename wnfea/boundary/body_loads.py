"""
Body Loads and Applied Acceleration Fields for WNFEA.
------------------------------------------------------
Implements:
1. Applied uniform/directional acceleration fields (gravity, inertia relief, dynamic G-loads):
   - C3D10 quadratic solids: Exact consistent Hammer 4-point Gauss quadrature body forces
     fe,i = rho * integral(N_i * a dOmega), conserving exact total mass (sum(fe) = M * a).
   - 3D Euler-Bernoulli/Timoshenko beams: Consistent translational (1/2 m*a) and transverse
     moment (+- 1/12 m*L*(t x a)) distribution.
2. Concentrated point loads:
   - Direct nodal loads (node_id).
   - Spatial point loads (x, y, z): Projected onto nearest node or distributed across
     neighboring surface/volume nodes via RBE3 interpolation conserving exact F and M.
"""

from dataclasses import dataclass, field
from typing import Optional, Union
import numpy as np
from .conditions import LoadDef


# Hammer 4-point Gauss quadrature constants for tetrahedra (degree 2 exact)
HAMMER_ALPHA = 0.1381966011250105
HAMMER_BETA  = 0.5854101966249685
HAMMER_WEIGHT = 1.0 / 24.0

GAUSS_XI   = np.array([HAMMER_ALPHA, HAMMER_BETA,  HAMMER_ALPHA, HAMMER_ALPHA], dtype=np.float64)
GAUSS_ETA  = np.array([HAMMER_ALPHA, HAMMER_ALPHA, HAMMER_BETA,  HAMMER_ALPHA], dtype=np.float64)
GAUSS_ZETA = np.array([HAMMER_ALPHA, HAMMER_ALPHA, HAMMER_ALPHA, HAMMER_BETA],  dtype=np.float64)

def _compute_c3d10_gauss_shape_functions() -> np.ndarray:
    N = np.zeros((4, 10), dtype=np.float64)
    for g in range(4):
        xi, eta, zeta = GAUSS_XI[g], GAUSS_ETA[g], GAUSS_ZETA[g]
        L1 = 1.0 - xi - eta - zeta
        L2 = xi
        L3 = eta
        L4 = zeta

        # Corner nodes
        N[g, 0] = L1 * (2.0 * L1 - 1.0)
        N[g, 1] = L2 * (2.0 * L2 - 1.0)
        N[g, 2] = L3 * (2.0 * L3 - 1.0)
        N[g, 3] = L4 * (2.0 * L4 - 1.0)

        # Mid-edge nodes
        N[g, 4] = 4.0 * L1 * L2
        N[g, 5] = 4.0 * L2 * L3
        N[g, 6] = 4.0 * L3 * L1
        N[g, 7] = 4.0 * L1 * L4
        N[g, 8] = 4.0 * L2 * L4
        N[g, 9] = 4.0 * L3 * L4
    return N

C3D10_GAUSS_N = _compute_c3d10_gauss_shape_functions()

def _compute_c3d10_gauss_derivatives() -> np.ndarray:
    dN_dxi = np.zeros((4, 3, 10), dtype=np.float64)
    for g in range(4):
        xi, eta, zeta = GAUSS_XI[g], GAUSS_ETA[g], GAUSS_ZETA[g]
        L1 = 1.0 - xi - eta - zeta
        L2 = xi
        L3 = eta
        L4 = zeta

        dN_dL = np.array([
            [4.0*L1 - 1.0, 0.0, 0.0, 0.0, 4.0*L2, 0.0, 4.0*L3, 4.0*L4, 0.0, 0.0],
            [0.0, 4.0*L2 - 1.0, 0.0, 0.0, 4.0*L1, 4.0*L3, 0.0, 0.0, 4.0*L4, 0.0],
            [0.0, 0.0, 4.0*L3 - 1.0, 0.0, 0.0, 4.0*L2, 4.0*L1, 0.0, 0.0, 4.0*L4],
            [0.0, 0.0, 0.0, 4.0*L4 - 1.0, 0.0, 0.0, 0.0, 4.0*L1, 4.0*L2, 4.0*L3]
        ], dtype=np.float64)

        dN_dxi[g, 0, :] = dN_dL[1, :] - dN_dL[0, :]
        dN_dxi[g, 1, :] = dN_dL[2, :] - dN_dL[0, :]
        dN_dxi[g, 2, :] = dN_dL[3, :] - dN_dL[0, :]
    return dN_dxi

C3D10_GAUSS_DN_DXI = _compute_c3d10_gauss_derivatives()


@dataclass
class AccelerationField:
    """
    Applied uniform linear acceleration field (e.g. gravity, G-load, inertia relief).

    Attributes:
        ax, ay, az: Acceleration vector components in m/s^2.
        label: Human readable label.
    """
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0
    label: str = ""

    @property
    def vector(self) -> np.ndarray:
        return np.array([self.ax, self.ay, self.az], dtype=np.float64)

    @classmethod
    def standard_gravity(cls, axis: str = "-y", g: float = 9.80665) -> "AccelerationField":
        a = [0.0, 0.0, 0.0]
        sign = -1.0 if axis.startswith("-") else 1.0
        axis_char = axis.lstrip("+-").lower()
        idx = {"x": 0, "y": 1, "z": 2}[axis_char]
        a[idx] = sign * g
        return cls(ax=a[0], ay=a[1], az=a[2], label=f"Standard Gravity ({axis})")

    def compute_solid_body_forces(
        self,
        nodes: np.ndarray,
        solid_elements: np.ndarray,
        density: Union[float, dict[int, float], np.ndarray],
    ) -> tuple[np.ndarray, float]:
        n_nodes = len(nodes)
        n_elems = len(solid_elements)
        a_vec = self.vector

        F_solid = np.zeros((n_nodes, 3), dtype=np.float64)
        total_mass = 0.0

        if n_elems == 0 or np.linalg.norm(a_vec) < 1e-15:
            return F_solid, total_mass

        if isinstance(density, (int, float)):
            rho_arr = np.full(n_elems, float(density), dtype=np.float64)
        elif isinstance(density, dict):
            rho_arr = np.array([float(density.get(i, 7850.0)) for i in range(n_elems)], dtype=np.float64)
        else:
            rho_arr = np.asarray(density, dtype=np.float64)

        X_e = nodes[solid_elements]

        det_J = np.zeros((4, n_elems), dtype=np.float64)
        for g in range(4):
            J_g = np.einsum("ik,mkj->mij", C3D10_GAUSS_DN_DXI[g], X_e)
            det_J[g] = np.linalg.det(J_g)

        V_e = np.sum(det_J, axis=0) * HAMMER_WEIGHT
        m_e = rho_arr * V_e
        total_mass = float(np.sum(m_e))

        N_weights = np.einsum("gm,gi->mi", det_J, C3D10_GAUSS_N) * HAMMER_WEIGHT
        f_elem = rho_arr[:, None, None] * a_vec[None, None, :] * N_weights[:, :, None]

        for i in range(10):
            node_ids = solid_elements[:, i]
            np.add.at(F_solid, node_ids, f_elem[:, i, :])

        return F_solid, total_mass

    def compute_beam_body_forces(
        self,
        nodes: np.ndarray,
        beam_elements: np.ndarray,
        areas: Union[float, np.ndarray, dict[int, float]],
        density: Union[float, np.ndarray, dict[int, float]],
    ) -> tuple[np.ndarray, float]:
        n_nodes = len(nodes)
        n_elems = len(beam_elements)
        a_vec = self.vector

        F_beam = np.zeros((n_nodes, 6), dtype=np.float64)
        total_mass = 0.0

        if n_elems == 0 or np.linalg.norm(a_vec) < 1e-15:
            return F_beam, total_mass

        if isinstance(areas, (int, float)):
            A_arr = np.full(n_elems, float(areas), dtype=np.float64)
        elif isinstance(areas, dict):
            A_arr = np.array([float(areas.get(i, 1e-4)) for i in range(n_elems)], dtype=np.float64)
        else:
            A_arr = np.asarray(areas, dtype=np.float64)

        if isinstance(density, (int, float)):
            rho_arr = np.full(n_elems, float(density), dtype=np.float64)
        elif isinstance(density, dict):
            rho_arr = np.array([float(density.get(i, 7850.0)) for i in range(n_elems)], dtype=np.float64)
        else:
            rho_arr = np.asarray(density, dtype=np.float64)

        n1 = beam_elements[:, 0]
        n2 = beam_elements[:, 1]
        x1 = nodes[n1]
        x2 = nodes[n2]
        d = x2 - x1
        L = np.linalg.norm(d, axis=1)

        m_e = rho_arr * A_arr * L
        total_mass = float(np.sum(m_e))

        f_trans = 0.5 * m_e[:, None] * a_vec[None, :]
        safe_L = np.where(L > 1e-12, L, 1.0)[:, None]
        t_hat = d / safe_L

        t_cross_a = np.cross(t_hat, a_vec[None, :])
        M1 = (1.0 / 12.0) * (m_e * L)[:, None] * t_cross_a

        np.add.at(F_beam[:, 0:3], n1, f_trans)
        np.add.at(F_beam[:, 0:3], n2, f_trans)
        np.add.at(F_beam[:, 3:6], n1, M1)
        np.add.at(F_beam[:, 3:6], n2, -M1)

        return F_beam, total_mass

    def apply_to_model(
        self,
        model,
        default_solid_density: float = 7850.0,
        default_beam_density: float = 7850.0,
    ) -> tuple[np.ndarray, float]:
        nodes = model.mesh_nodes
        if nodes is None or len(nodes) == 0:
            return np.zeros((0, 6)), 0.0

        n_nodes = len(nodes)
        F_total = np.zeros((n_nodes, 6), dtype=np.float64)
        total_mass = 0.0

        if model.solid_elements is not None and len(model.solid_elements) > 0:
            densities = []
            for i in range(len(model.solid_elements)):
                mat_name = model.solid_materials.get(i)
                rho = default_solid_density
                if mat_name and hasattr(model, "materials") and mat_name in model.materials:
                    rho = getattr(model.materials[mat_name], "density", default_solid_density)
                densities.append(rho)

            F_sol, m_sol = self.compute_solid_body_forces(
                nodes, model.solid_elements, np.array(densities, dtype=np.float64)
            )
            F_total[:, :3] += F_sol
            total_mass += m_sol

        if model.mesh_elements is not None and len(model.mesh_elements) > 0:
            areas = []
            rhos = []
            for i in range(len(model.mesh_elements)):
                prop = model.element_properties.get(i)
                A = 1e-4
                rho = default_beam_density
                if prop:
                    if hasattr(prop, "section") and prop.section:
                        A = getattr(prop.section, "area", 1e-4)
                    if hasattr(prop, "material") and prop.material:
                        rho = getattr(prop.material, "density", default_beam_density)
                areas.append(A)
                rhos.append(rho)

            F_bm, m_bm = self.compute_beam_body_forces(
                nodes, model.mesh_elements, np.array(areas), np.array(rhos)
            )
            F_total += F_bm
            total_mass += m_bm

        return F_total, total_mass


@dataclass
class SpatialPointLoad:
    """
    Concentrated point load applied at an arbitrary 3D spatial coordinate (x, y, z).
    """
    x: float
    y: float
    z: float
    fx: float = 0.0
    fy: float = 0.0
    fz: float = 0.0
    mx: float = 0.0
    my: float = 0.0
    mz: float = 0.0
    radius: Optional[float] = None
    n_nearest: int = 8
    method: str = "rbe3"
    label: str = ""

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @property
    def force(self) -> np.ndarray:
        return np.array([self.fx, self.fy, self.fz], dtype=np.float64)

    @property
    def moment(self) -> np.ndarray:
        return np.array([self.mx, self.my, self.mz], dtype=np.float64)

    def distribute_to_mesh(
        self,
        mesh_nodes: np.ndarray,
        candidate_node_indices: Optional[np.ndarray] = None,
    ) -> list[LoadDef]:
        if candidate_node_indices is None:
            candidate_indices = np.arange(len(mesh_nodes))
        else:
            candidate_indices = np.asarray(candidate_node_indices, dtype=np.int64)

        if len(candidate_indices) == 0:
            raise ValueError("No candidate mesh nodes provided to distribute spatial load.")

        P = self.position
        F = self.force
        M = self.moment

        cand_coords = mesh_nodes[candidate_indices]
        dists = np.linalg.norm(cand_coords - P[None, :], axis=1)

        if self.method == "nearest" or len(candidate_indices) == 1:
            best_idx = candidate_indices[np.argmin(dists)]
            return [LoadDef(
                node_id=int(best_idx),
                is_geometry_node=False,
                fx=float(F[0]), fy=float(F[1]), fz=float(F[2]),
                mx=float(M[0]), my=float(M[1]), mz=float(M[2]),
                label=self.label or "SpatialPointLoad (Nearest)",
            )]

        if self.radius is not None and self.radius > 0.0:
            mask = dists <= self.radius
            if not np.any(mask):
                k = min(self.n_nearest, len(candidate_indices))
                chosen_local = np.argsort(dists)[:k]
            else:
                chosen_local = np.where(mask)[0]
        else:
            k = min(self.n_nearest, len(candidate_indices))
            chosen_local = np.argsort(dists)[:k]

        target_nodes = candidate_indices[chosen_local]
        target_coords = mesh_nodes[target_nodes]
        target_dists = dists[chosen_local]

        eps = 1e-6
        inv_d = 1.0 / (target_dists + eps)
        weights = inv_d / np.sum(inv_d)

        centroid = np.sum(weights[:, None] * target_coords, axis=0)
        r_i = target_coords - centroid[None, :]

        M_net = M + np.cross(P - centroid, F)

        I_tensor = np.zeros((3, 3), dtype=np.float64)
        for i in range(len(target_nodes)):
            r = r_i[i]
            w = weights[i]
            r2 = float(np.dot(r, r))
            I_tensor += w * (r2 * np.eye(3) - np.outer(r, r))

        try:
            omega = np.linalg.solve(I_tensor + 1e-12 * np.trace(I_tensor) * np.eye(3), M_net)
        except np.linalg.LinAlgError:
            omega = np.zeros(3, dtype=np.float64)

        f_trans = weights[:, None] * F[None, :]
        f_moment = weights[:, None] * np.cross(omega[None, :], r_i)
        f_total = f_trans + f_moment

        loads: list[LoadDef] = []
        for idx, nid in enumerate(target_nodes):
            loads.append(LoadDef(
                node_id=int(nid),
                is_geometry_node=False,
                fx=float(f_total[idx, 0]),
                fy=float(f_total[idx, 1]),
                fz=float(f_total[idx, 2]),
                mx=0.0, my=0.0, mz=0.0,
                label=f"{self.label} (RBE3 {idx+1}/{len(target_nodes)})",
            ))

        return loads
