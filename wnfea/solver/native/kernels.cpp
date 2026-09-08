// WNFEA High-Performance Native Element Kernels (AVX / SIMD)
// Optimized for contiguous, coalesced memory access

#include <cmath>
#include <cstring>
#include <vector>

#if defined(_MSC_VER) || defined(__MINGW32__)
#define WNFEA_EXPORT __declspec(dllexport)
#else
#define WNFEA_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

// ============================================================================
// Co-Rotational 3D Beam Internal Forces Kernel
// ============================================================================
WNFEA_EXPORT void compute_beam_internal_forces_native(
    const double* __restrict nodes,
    const int* __restrict elements,
    const double* __restrict props,
    const double* __restrict e_ref,
    const double* __restrict L0_arr,
    const double* __restrict U,
    double* __restrict F_int,
    int n_elements,
    int n_nodes
) {
    for (int e = 0; e < n_elements; ++e) {
        const int n1 = elements[e * 2];
        const int n2 = elements[e * 2 + 1];

        // Coalesced loads of node coordinates
        const double X1[3] = { nodes[n1 * 3], nodes[n1 * 3 + 1], nodes[n1 * 3 + 2] };
        const double X2[3] = { nodes[n2 * 3], nodes[n2 * 3 + 1], nodes[n2 * 3 + 2] };

        // Initial chord vector & length
        const double v0[3] = { X2[0] - X1[0], X2[1] - X1[1], X2[2] - X1[2] };
        const double L0 = L0_arr[e];

        // Reference triad
        const double* e1_0 = &e_ref[e * 9 + 0];
        const double* e2_0 = &e_ref[e * 9 + 3];
        const double* e3_0 = &e_ref[e * 9 + 6];

        // Displacements
        const double u1[3]  = { U[n1 * 6 + 0], U[n1 * 6 + 1], U[n1 * 6 + 2] };
        const double th1[3] = { U[n1 * 6 + 3], U[n1 * 6 + 4], U[n1 * 6 + 5] };
        const double u2[3]  = { U[n2 * 6 + 0], U[n2 * 6 + 1], U[n2 * 6 + 2] };
        const double th2[3] = { U[n2 * 6 + 3], U[n2 * 6 + 4], U[n2 * 6 + 5] };

        // Current chord vector & length
        const double du[3] = { u2[0] - u1[0], u2[1] - u1[1], u2[2] - u1[2] };
        const double v[3] = { v0[0] + du[0], v0[1] + du[1], v0[2] + du[2] };
        const double L = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        const double inv_L = 1.0 / L;
        const double r1[3] = { v[0] * inv_L, v[1] * inv_L, v[2] * inv_L };

        // Cross product e1_0 x r1
        const double cr[3] = {
            e1_0[1] * r1[2] - e1_0[2] * r1[1],
            e1_0[2] * r1[0] - e1_0[0] * r1[2],
            e1_0[0] * r1[1] - e1_0[1] * r1[0]
        };
        const double sin_phi = std::sqrt(cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]);
        const double cos_phi = e1_0[0] * r1[0] + e1_0[1] * r1[1] + e1_0[2] * r1[2];

        double r2[3] = { e2_0[0], e2_0[1], e2_0[2] };
        double r3[3] = { e3_0[0], e3_0[1], e3_0[2] };

        if (sin_phi > 1e-12) {
            const double inv_sin = 1.0 / sin_phi;
            const double ax[3] = { cr[0] * inv_sin, cr[1] * inv_sin, cr[2] * inv_sin };
            const double phi = std::atan2(sin_phi, cos_phi);
            const double cp = std::cos(phi);
            const double sp = std::sin(phi);

            // ax x e2_0
            const double ax_x_e2[3] = {
                ax[1] * e2_0[2] - ax[2] * e2_0[1],
                ax[2] * e2_0[0] - ax[0] * e2_0[2],
                ax[0] * e2_0[1] - ax[1] * e2_0[0]
            };
            const double ax_dot_e2 = ax[0] * e2_0[0] + ax[1] * e2_0[1] + ax[2] * e2_0[2];
            r2[0] = e2_0[0] * cp + ax_x_e2[0] * sp + ax[0] * ax_dot_e2 * (1.0 - cp);
            r2[1] = e2_0[1] * cp + ax_x_e2[1] * sp + ax[1] * ax_dot_e2 * (1.0 - cp);
            r2[2] = e2_0[2] * cp + ax_x_e2[2] * sp + ax[2] * ax_dot_e2 * (1.0 - cp);

            // ax x e3_0
            const double ax_x_e3[3] = {
                ax[1] * e3_0[2] - ax[2] * e3_0[1],
                ax[2] * e3_0[0] - ax[0] * e3_0[2],
                ax[0] * e3_0[1] - ax[1] * e3_0[0]
            };
            const double ax_dot_e3 = ax[0] * e3_0[0] + ax[1] * e3_0[1] + ax[2] * e3_0[2];
            r3[0] = e3_0[0] * cp + ax_x_e3[0] * sp + ax[0] * ax_dot_e3 * (1.0 - cp);
            r3[1] = e3_0[1] * cp + ax_x_e3[1] * sp + ax[1] * ax_dot_e3 * (1.0 - cp);
            r3[2] = e3_0[2] * cp + ax_x_e3[2] * sp + ax[2] * ax_dot_e3 * (1.0 - cp);
        }

        // Torsional twist around r1
        const double avg_twist = 0.5 * ((th1[0] + th2[0]) * r1[0] + (th1[1] + th2[1]) * r1[1] + (th1[2] + th2[2]) * r1[2]);
        const double ct = std::cos(avg_twist);
        const double st = std::sin(avg_twist);

        const double r1_x_r2[3] = {
            r1[1] * r2[2] - r1[2] * r2[1],
            r1[2] * r2[0] - r1[0] * r2[2],
            r1[0] * r2[1] - r1[1] * r2[0]
        };
        const double r1_x_r3[3] = {
            r1[1] * r3[2] - r1[2] * r3[1],
            r1[2] * r3[0] - r1[0] * r3[2],
            r1[0] * r3[1] - r1[1] * r3[0]
        };
        r2[0] = r2[0] * ct + r1_x_r2[0] * st;
        r2[1] = r2[1] * ct + r1_x_r2[1] * st;
        r2[2] = r2[2] * ct + r1_x_r2[2] * st;
        r3[0] = r3[0] * ct + r1_x_r3[0] * st;
        r3[1] = r3[1] * ct + r1_x_r3[1] * st;
        r3[2] = r3[2] * ct + r1_x_r3[2] * st;

        // Gram-Schmidt
        const double r2_dot_r1 = r2[0] * r1[0] + r2[1] * r1[1] + r2[2] * r1[2];
        r2[0] -= r2_dot_r1 * r1[0];
        r2[1] -= r2_dot_r1 * r1[1];
        r2[2] -= r2_dot_r1 * r1[2];
        const double norm_r2 = std::sqrt(r2[0] * r2[0] + r2[1] * r2[1] + r2[2] * r2[2]);
        const double inv_norm_r2 = 1.0 / norm_r2;
        r2[0] *= inv_norm_r2; r2[1] *= inv_norm_r2; r2[2] *= inv_norm_r2;

        r3[0] = r1[1] * r2[2] - r1[2] * r2[1];
        r3[1] = r1[2] * r2[0] - r1[0] * r2[2];
        r3[2] = r1[0] * r2[1] - r1[1] * r2[0];
        const double norm_r3 = std::sqrt(r3[0] * r3[0] + r3[1] * r3[1] + r3[2] * r3[2]);
        const double inv_norm_r3 = 1.0 / norm_r3;
        r3[0] *= inv_norm_r3; r3[1] *= inv_norm_r3; r3[2] *= inv_norm_r3;

        // Relative deformations
        const double v0_dot_du = v0[0] * du[0] + v0[1] * du[1] + v0[2] * du[2];
        const double du_dot_du = du[0] * du[0] + du[1] * du[1] + du[2] * du[2];
        const double u_l = (2.0 * v0_dot_du + du_dot_du) / (L + L0);

        const double r3_dot_du = r3[0] * du[0] + r3[1] * du[1] + r3[2] * du[2];
        const double r2_dot_du = r2[0] * du[0] + r2[1] * du[1] + r2[2] * du[2];

        const double th_y1 = (r2[0] * th1[0] + r2[1] * th1[1] + r2[2] * th1[2]) + r3_dot_du * inv_L;
        const double th_z1 = (r3[0] * th1[0] + r3[1] * th1[1] + r3[2] * th1[2]) - r2_dot_du * inv_L;
        const double th_x  = r1[0] * (th2[0] - th1[0]) + r1[1] * (th2[1] - th1[1]) + r1[2] * (th2[2] - th1[2]);
        const double th_y2 = (r2[0] * th2[0] + r2[1] * th2[1] + r2[2] * th2[2]) + r3_dot_du * inv_L;
        const double th_z2 = (r3[0] * th2[0] + r3[1] * th2[1] + r3[2] * th2[2]) - r2_dot_du * inv_L;

        // Properties: [E_mod, G_mod, A_area, Iy, Iz, J]
        const double E_mod  = props[e * 6 + 0];
        const double G_mod  = props[e * 6 + 1];
        const double A_area = props[e * 6 + 2];
        const double Iy     = props[e * 6 + 3];
        const double Iz     = props[e * 6 + 4];
        const double J      = props[e * 6 + 5];

        const double N   = (E_mod * A_area / L0) * u_l;
        const double My1 = (E_mod * Iy / L0) * (4.0 * th_y1 + 2.0 * th_y2);
        const double My2 = (E_mod * Iy / L0) * (2.0 * th_y1 + 4.0 * th_y2);
        const double Mz1 = (E_mod * Iz / L0) * (4.0 * th_z1 + 2.0 * th_z2);
        const double Mz2 = (E_mod * Iz / L0) * (2.0 * th_z1 + 4.0 * th_z2);
        const double Tx  = (G_mod * J / L0) * th_x;

        // Direct forces
        const double shear_z = (My1 + My2) * inv_L;
        const double shear_y = (Mz1 + Mz2) * inv_L;

        const double f1[3] = {
            -N * r1[0] - shear_z * r3[0] + shear_y * r2[0],
            -N * r1[1] - shear_z * r3[1] + shear_y * r2[1],
            -N * r1[2] - shear_z * r3[2] + shear_y * r2[2]
        };
        const double m1[3] = {
            -Tx * r1[0] + My1 * r2[0] + Mz1 * r3[0],
            -Tx * r1[1] + My1 * r2[1] + Mz1 * r3[1],
            -Tx * r1[2] + My1 * r2[2] + Mz1 * r3[2]
        };
        const double m2[3] = {
            Tx * r1[0] + My2 * r2[0] + Mz2 * r3[0],
            Tx * r1[1] + My2 * r2[1] + Mz2 * r3[1],
            Tx * r1[2] + My2 * r2[2] + Mz2 * r3[2]
        };

        // Coalesced accumulation into F_int
        F_int[n1 * 6 + 0] += f1[0];
        F_int[n1 * 6 + 1] += f1[1];
        F_int[n1 * 6 + 2] += f1[2];
        F_int[n1 * 6 + 3] += m1[0];
        F_int[n1 * 6 + 4] += m1[1];
        F_int[n1 * 6 + 5] += m1[2];

        F_int[n2 * 6 + 0] -= f1[0];
        F_int[n2 * 6 + 1] -= f1[1];
        F_int[n2 * 6 + 2] -= f1[2];
        F_int[n2 * 6 + 3] += m2[0];
        F_int[n2 * 6 + 4] += m2[1];
        F_int[n2 * 6 + 5] += m2[2];
    }
}


// ============================================================================
// C3D10 10-Node Quadratic Tetrahedral Element Internal Forces Kernel
// ============================================================================

// Precomputed Gauss quadrature points & weights (Hammer 4-point rule)
static const double ALPHA = 0.1381966011250105;
static const double BETA  = 0.5854101966249685;
static const double WEIGHT = 1.0 / 24.0;

static const double GAUSS_XI[4]   = { ALPHA, BETA, ALPHA, ALPHA };
static const double GAUSS_ETA[4]  = { ALPHA, ALPHA, BETA, ALPHA };
static const double GAUSS_ZETA[4] = { ALPHA, ALPHA, ALPHA, BETA };

// Evaluate natural derivatives dN/dxi (3 x 10) for C3D10
static inline void evaluate_dN_dxi_c3d10(double xi, double eta, double zeta, double dN_dxi[3][10]) {
    const double L1 = 1.0 - xi - eta - zeta;
    const double L2 = xi;
    const double L3 = eta;
    const double L4 = zeta;

    // dN/dL (4 x 10)
    // [0]=dL1, [1]=dL2, [2]=dL3, [3]=dL4
    const double dN_dL[4][10] = {
        { 4.0*L1 - 1.0, 0.0, 0.0, 0.0, 4.0*L2, 0.0, 4.0*L3, 4.0*L4, 0.0, 0.0 },
        { 0.0, 4.0*L2 - 1.0, 0.0, 0.0, 4.0*L1, 4.0*L3, 0.0, 0.0, 4.0*L4, 0.0 },
        { 0.0, 0.0, 4.0*L3 - 1.0, 0.0, 0.0, 4.0*L2, 4.0*L1, 0.0, 0.0, 4.0*L4 },
        { 0.0, 0.0, 0.0, 4.0*L4 - 1.0, 0.0, 0.0, 0.0, 4.0*L1, 4.0*L2, 4.0*L3 }
    };

    // Chain rule: d/dxi = d/dL2 - d/dL1, d/deta = d/dL3 - d/dL1, d/dzeta = d/dL4 - d/dL1
    for (int i = 0; i < 10; ++i) {
        dN_dxi[0][i] = dN_dL[1][i] - dN_dL[0][i];
        dN_dxi[1][i] = dN_dL[2][i] - dN_dL[0][i];
        dN_dxi[2][i] = dN_dL[3][i] - dN_dL[0][i];
    }
}

WNFEA_EXPORT void compute_c3d10_internal_forces_native(
    const double* __restrict nodes,
    const int* __restrict solid_elements,
    const double* __restrict props,      // [E_solids * 2] (E, nu)
    const double* __restrict U,          // [N * 6] (or active mapping)
    double* __restrict F_int,            // [N * 6]
    int n_solids,
    int n_nodes
) {
    // Precompute shape function derivatives at the 4 Gauss points
    double dN_dxi_gp[4][3][10];
    for (int g = 0; g < 4; ++g) {
        evaluate_dN_dxi_c3d10(GAUSS_XI[g], GAUSS_ETA[g], GAUSS_ZETA[g], dN_dxi_gp[g]);
    }

    for (int e = 0; e < n_solids; ++e) {
        const int* elem_nodes = &solid_elements[e * 10];
        const double E_mod = props[e * 2 + 0];
        const double nu    = props[e * 2 + 1];

        // Isotropic elasticity constants
        const double factor = E_mod / ((1.0 + nu) * (1.0 - 2.0 * nu));
        const double c11 = factor * (1.0 - nu);
        const double c12 = factor * nu;
        const double c44 = factor * 0.5 * (1.0 - 2.0 * nu);

        // Coalesced loads of 10 nodal coordinates and displacements
        double coords[10][3];
        double u_elem[10][3];
        for (int i = 0; i < 10; ++i) {
            const int nid = elem_nodes[i];
            coords[i][0] = nodes[nid * 3 + 0];
            coords[i][1] = nodes[nid * 3 + 1];
            coords[i][2] = nodes[nid * 3 + 2];

            u_elem[i][0] = U[nid * 6 + 0];
            u_elem[i][1] = U[nid * 6 + 1];
            u_elem[i][2] = U[nid * 6 + 2];
        }

        double f_elem[10][3] = { {0.0} };

        // Loop over 4 Gauss points
        for (int g = 0; g < 4; ++g) {
            const auto& dN_dxi = dN_dxi_gp[g];

            // Compute Jacobian J = sum_i coords[i] (x) dN_dxi[:, i]
            double J[3][3] = { {0.0} };
            for (int i = 0; i < 10; ++i) {
                J[0][0] += dN_dxi[0][i] * coords[i][0];
                J[0][1] += dN_dxi[0][i] * coords[i][1];
                J[0][2] += dN_dxi[0][i] * coords[i][2];

                J[1][0] += dN_dxi[1][i] * coords[i][0];
                J[1][1] += dN_dxi[1][i] * coords[i][1];
                J[1][2] += dN_dxi[1][i] * coords[i][2];

                J[2][0] += dN_dxi[2][i] * coords[i][0];
                J[2][1] += dN_dxi[2][i] * coords[i][1];
                J[2][2] += dN_dxi[2][i] * coords[i][2];
            }

            // Invert 3x3 Jacobian
            const double c00 = J[1][1] * J[2][2] - J[1][2] * J[2][1];
            const double c01 = J[1][2] * J[2][0] - J[1][0] * J[2][2];
            const double c02 = J[1][0] * J[2][1] - J[1][1] * J[2][0];

            const double detJ = J[0][0] * c00 + J[0][1] * c01 + J[0][2] * c02;
            if (std::abs(detJ) < 1e-15) continue;
            const double inv_detJ = 1.0 / detJ;

            const double invJ[3][3] = {
                { c00 * inv_detJ, (J[0][2]*J[2][1] - J[0][1]*J[2][2]) * inv_detJ, (J[0][1]*J[1][2] - J[0][2]*J[1][1]) * inv_detJ },
                { c01 * inv_detJ, (J[0][0]*J[2][2] - J[0][2]*J[2][0]) * inv_detJ, (J[0][2]*J[1][0] - J[0][0]*J[1][2]) * inv_detJ },
                { c02 * inv_detJ, (J[0][1]*J[2][0] - J[0][0]*J[2][1]) * inv_detJ, (J[0][0]*J[1][1] - J[0][1]*J[1][0]) * inv_detJ }
            };

            // Cartesian derivatives: dN_dX = invJ * dN_dxi (3 x 10)
            double dN_dX[3][10];
            for (int i = 0; i < 10; ++i) {
                dN_dX[0][i] = invJ[0][0] * dN_dxi[0][i] + invJ[0][1] * dN_dxi[1][i] + invJ[0][2] * dN_dxi[2][i];
                dN_dX[1][i] = invJ[1][0] * dN_dxi[0][i] + invJ[1][1] * dN_dxi[1][i] + invJ[1][2] * dN_dxi[2][i];
                dN_dX[2][i] = invJ[2][0] * dN_dxi[0][i] + invJ[2][1] * dN_dxi[1][i] + invJ[2][2] * dN_dxi[2][i];
            }

            // Strains: eps = B * u
            double eps[6] = { 0.0 };
            for (int i = 0; i < 10; ++i) {
                const double ux = u_elem[i][0];
                const double uy = u_elem[i][1];
                const double uz = u_elem[i][2];

                eps[0] += dN_dX[0][i] * ux;                     // eps_xx
                eps[1] += dN_dX[1][i] * uy;                     // eps_yy
                eps[2] += dN_dX[2][i] * uz;                     // eps_zz
                eps[3] += dN_dX[1][i] * ux + dN_dX[0][i] * uy;  // gamma_xy
                eps[4] += dN_dX[2][i] * uy + dN_dX[1][i] * uz;  // gamma_yz
                eps[5] += dN_dX[2][i] * ux + dN_dX[0][i] * uz;  // gamma_zx
            }

            // Stress: sigma = C * eps
            const double sig[6] = {
                c11 * eps[0] + c12 * eps[1] + c12 * eps[2],
                c12 * eps[0] + c11 * eps[1] + c12 * eps[2],
                c12 * eps[0] + c12 * eps[1] + c11 * eps[2],
                c44 * eps[3],
                c44 * eps[4],
                c44 * eps[5]
            };

            const double w_detJ = WEIGHT * detJ;

            // Nodal forces: f_i += w * detJ * B_i^T * sigma
            for (int i = 0; i < 10; ++i) {
                const double dNx = dN_dX[0][i];
                const double dNy = dN_dX[1][i];
                const double dNz = dN_dX[2][i];

                f_elem[i][0] += w_detJ * (dNx * sig[0] + dNy * sig[3] + dNz * sig[5]);
                f_elem[i][1] += w_detJ * (dNy * sig[1] + dNx * sig[3] + dNz * sig[4]);
                f_elem[i][2] += w_detJ * (dNz * sig[2] + dNy * sig[4] + dNx * sig[5]);
            }
        }

        // Scatter accumulate into F_int
        for (int i = 0; i < 10; ++i) {
            const int nid = elem_nodes[i];
            F_int[nid * 6 + 0] += f_elem[i][0];
            F_int[nid * 6 + 1] += f_elem[i][1];
            F_int[nid * 6 + 2] += f_elem[i][2];
        }
    }
}

// ============================================================================
// C3D10 Exact Stiffness Diagonal Kernel (for Jacobi / Multigrid Preconditioners)
// ============================================================================
WNFEA_EXPORT void compute_c3d10_diagonal_native(
    const double* __restrict nodes,
    const int* __restrict solid_elements,
    const double* __restrict props,      // [E_solids * 2] (E, nu)
    double* __restrict diag_K,           // [n_nodes * 3]
    int n_solids,
    int n_nodes
) {
    double dN_dxi_gp[4][3][10];
    for (int g = 0; g < 4; ++g) {
        evaluate_dN_dxi_c3d10(GAUSS_XI[g], GAUSS_ETA[g], GAUSS_ZETA[g], dN_dxi_gp[g]);
    }

    for (int e = 0; e < n_solids; ++e) {
        const int* elem_nodes = &solid_elements[e * 10];
        const double E_mod = props[e * 2 + 0];
        const double nu    = props[e * 2 + 1];

        // Isotropic elasticity constants
        const double factor = E_mod / ((1.0 + nu) * (1.0 - 2.0 * nu));
        const double c11 = factor * (1.0 - nu);
        const double c44 = factor * 0.5 * (1.0 - 2.0 * nu);

        double coords[10][3];
        for (int i = 0; i < 10; ++i) {
            const int nid = elem_nodes[i];
            coords[i][0] = nodes[nid * 3 + 0];
            coords[i][1] = nodes[nid * 3 + 1];
            coords[i][2] = nodes[nid * 3 + 2];
        }

        for (int g = 0; g < 4; ++g) {
            const auto& dN_dxi = dN_dxi_gp[g];

            double J[3][3] = { {0.0} };
            for (int i = 0; i < 10; ++i) {
                J[0][0] += dN_dxi[0][i] * coords[i][0];
                J[0][1] += dN_dxi[0][i] * coords[i][1];
                J[0][2] += dN_dxi[0][i] * coords[i][2];

                J[1][0] += dN_dxi[1][i] * coords[i][0];
                J[1][1] += dN_dxi[1][i] * coords[i][1];
                J[1][2] += dN_dxi[1][i] * coords[i][2];

                J[2][0] += dN_dxi[2][i] * coords[i][0];
                J[2][1] += dN_dxi[2][i] * coords[i][1];
                J[2][2] += dN_dxi[2][i] * coords[i][2];
            }

            const double c00 = J[1][1] * J[2][2] - J[1][2] * J[2][1];
            const double c01 = J[1][2] * J[2][0] - J[1][0] * J[2][2];
            const double c02 = J[1][0] * J[2][1] - J[1][1] * J[2][0];

            const double detJ = J[0][0] * c00 + J[0][1] * c01 + J[0][2] * c02;
            if (std::abs(detJ) < 1e-15) continue;
            const double inv_detJ = 1.0 / detJ;

            const double invJ[3][3] = {
                { c00 * inv_detJ, (J[0][2]*J[2][1] - J[0][1]*J[2][2]) * inv_detJ, (J[0][1]*J[1][2] - J[0][2]*J[1][1]) * inv_detJ },
                { c01 * inv_detJ, (J[0][0]*J[2][2] - J[0][2]*J[2][0]) * inv_detJ, (J[0][2]*J[1][0] - J[0][0]*J[1][2]) * inv_detJ },
                { c02 * inv_detJ, (J[0][1]*J[2][0] - J[0][0]*J[2][1]) * inv_detJ, (J[0][0]*J[1][1] - J[0][1]*J[1][0]) * inv_detJ }
            };

            const double w_detJ = WEIGHT * detJ;

            for (int i = 0; i < 10; ++i) {
                const double dNx = invJ[0][0] * dN_dxi[0][i] + invJ[0][1] * dN_dxi[1][i] + invJ[0][2] * dN_dxi[2][i];
                const double dNy = invJ[1][0] * dN_dxi[0][i] + invJ[1][1] * dN_dxi[1][i] + invJ[1][2] * dN_dxi[2][i];
                const double dNz = invJ[2][0] * dN_dxi[0][i] + invJ[2][1] * dN_dxi[1][i] + invJ[2][2] * dN_dxi[2][i];

                const double dNx2 = dNx * dNx;
                const double dNy2 = dNy * dNy;
                const double dNz2 = dNz * dNz;

                const double kxx = w_detJ * (c11 * dNx2 + c44 * (dNy2 + dNz2));
                const double kyy = w_detJ * (c11 * dNy2 + c44 * (dNx2 + dNz2));
                const double kzz = w_detJ * (c11 * dNz2 + c44 * (dNx2 + dNy2));

                const int nid = elem_nodes[i];
                diag_K[nid * 3 + 0] += kxx;
                diag_K[nid * 3 + 1] += kyy;
                diag_K[nid * 3 + 2] += kzz;
            }
        }
    }
}

} // extern "C"
