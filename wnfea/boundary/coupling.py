"""
Kinematic Multi-Point Constraints and Couplings for WNFEA.

Implements rigid coupling between 6-DOF master nodes (e.g. beam frames) and
3-DOF slave nodes (e.g. solid continuum surface faces) via direct kinematic
elimination.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class RigidCoupling:
    """
    Rigid kinematic coupling between a 6-DOF master node and a cluster of
    3-DOF slave nodes.

    Displacement relationship for each slave node i:
        u_s,i = u_m + theta_m x (X_s,i - X_m)
              = [I_3x3,  -skew(X_s,i - X_m)] * [u_m; theta_m]

    Force/moment relationship:
        F_m = sum_i F_s,i
        M_m = sum_i (X_s,i - X_m) x F_s,i
    """
    master_node_id: int
    slave_node_ids: list[int]
    label: str = ""

    def compute_coupling_matrices(
        self,
        master_coord: np.ndarray,
        slave_coords: list[np.ndarray],
    ) -> list[np.ndarray]:
        """
        Compute the 3x6 transformation matrix T_i for each slave node:
            u_s,i = T_i * [u_m; theta_m]

        where:
            T_i = [I_3x3,  -skew(r_i)]
            r_i = X_s,i - X_m
        """
        Xm = np.asarray(master_coord, dtype=np.float64)
        T_matrices: list[np.ndarray] = []

        for Xs in slave_coords:
            r = np.asarray(Xs, dtype=np.float64) - Xm
            rx, ry, rz = r

            # Skew-symmetric cross product matrix: skew(r) * v = r x v
            # -skew(r) = [ 0,  rz, -ry;
            #             -rz,  0,  rx;
            #              ry, -rx,  0 ]
            neg_skew_r = np.array([
                [0.0, rz, -ry],
                [-rz, 0.0, rx],
                [ry, -rx, 0.0],
            ], dtype=np.float64)

            T_i = np.zeros((3, 6), dtype=np.float64)
            T_i[0:3, 0:3] = np.eye(3)
            T_i[0:3, 3:6] = neg_skew_r

            T_matrices.append(T_i)

        return T_matrices
