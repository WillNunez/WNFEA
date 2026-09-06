// WNFEA Architecture-Tuned HIP Kernels for AMD Radeon RX 7800 XT (gfx1101 / RDNA 3)
// Optimized for Wave32 wavefronts, 30 WGPs, 128 KB VGPRs/SIMD, and 4 MB L2 cache

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

#if defined(_MSC_VER) || defined(__MINGW32__)
#ifdef __HIP_DEVICE_COMPILE__
#define WNFEA_EXPORT
#else
#define WNFEA_EXPORT __declspec(dllexport)
#endif
#else
#define WNFEA_EXPORT __attribute__((visibility("default")))
#endif

// ============================================================================
// Atomic Add Helpers (FP32 & FP64)
// ============================================================================
__device__ inline double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ inline float atomicAddFloat(float* address, float val) {
    return atomicAdd(address, val);
}

// ============================================================================
// Co-Rotational 3D Beam Kernel
// Tile size: 256 threads (8 Wave32 waves)
// Launch bounds: (256, 2) -> limits VGPRs <= 64, ensures >= 2 blocks/SIMD (50% occupancy)
// Grid-stride loop ensures uniform load across all 30 WGPs and eliminates tail latency
// ============================================================================
__launch_bounds__(256, 2)
__global__ void beam_internal_forces_kernel(
    const double* __restrict__ nodes,      // [n_nodes * 3]
    const int*    __restrict__ elements,   // [n_elements * 2]
    const double* __restrict__ props,      // [n_elements * 6] (E, G, A, Iy, Iz, J)
    const double* __restrict__ e_ref,      // [n_elements * 9]
    const double* __restrict__ L0_arr,     // [n_elements]
    const double* __restrict__ U,          // [n_nodes * 6]
    double*       __restrict__ F_int,      // [n_nodes * 6]
    int n_elements
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int e = tid; e < n_elements; e += stride) {
        const int n1 = elements[e * 2];
        const int n2 = elements[e * 2 + 1];

        // Coalesced loads of node coordinates
        const double X1[3] = { nodes[n1 * 3],     nodes[n1 * 3 + 1], nodes[n1 * 3 + 2] };
        const double X2[3] = { nodes[n2 * 3],     nodes[n2 * 3 + 1], nodes[n2 * 3 + 2] };

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
        const double L = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        const double inv_L = 1.0 / L;
        const double r1[3] = { v[0] * inv_L, v[1] * inv_L, v[2] * inv_L };

        // Cross product e1_0 x r1
        const double cr[3] = {
            e1_0[1] * r1[2] - e1_0[2] * r1[1],
            e1_0[2] * r1[0] - e1_0[0] * r1[2],
            e1_0[0] * r1[1] - e1_0[1] * r1[0]
        };
        const double sin_phi = sqrt(cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]);
        const double cos_phi = e1_0[0] * r1[0] + e1_0[1] * r1[1] + e1_0[2] * r1[2];

        double r2[3] = { e2_0[0], e2_0[1], e2_0[2] };
        double r3[3] = { e3_0[0], e3_0[1], e3_0[2] };

        if (sin_phi > 1e-12) {
            const double inv_sin = 1.0 / sin_phi;
            const double ax[3] = { cr[0] * inv_sin, cr[1] * inv_sin, cr[2] * inv_sin };
            const double phi = atan2(sin_phi, cos_phi);
            const double cp = cos(phi);
            const double sp = sin(phi);

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
        const double ct = cos(avg_twist);
        const double st = sin(avg_twist);

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
        const double norm_r2 = sqrt(r2[0] * r2[0] + r2[1] * r2[1] + r2[2] * r2[2]);
        const double inv_norm_r2 = 1.0 / norm_r2;
        r2[0] *= inv_norm_r2; r2[1] *= inv_norm_r2; r2[2] *= inv_norm_r2;

        r3[0] = r1[1] * r2[2] - r1[2] * r2[1];
        r3[1] = r1[2] * r2[0] - r1[0] * r2[2];
        r3[2] = r1[0] * r2[1] - r1[1] * r2[0];
        const double norm_r3 = sqrt(r3[0] * r3[0] + r3[1] * r3[1] + r3[2] * r3[2]);
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

        // Atomic accumulation into global F_int
        atomicAddDouble(&F_int[n1 * 6 + 0],  f1[0]);
        atomicAddDouble(&F_int[n1 * 6 + 1],  f1[1]);
        atomicAddDouble(&F_int[n1 * 6 + 2],  f1[2]);
        atomicAddDouble(&F_int[n1 * 6 + 3],  m1[0]);
        atomicAddDouble(&F_int[n1 * 6 + 4],  m1[1]);
        atomicAddDouble(&F_int[n1 * 6 + 5],  m1[2]);

        atomicAddDouble(&F_int[n2 * 6 + 0], -f1[0]);
        atomicAddDouble(&F_int[n2 * 6 + 1], -f1[1]);
        atomicAddDouble(&F_int[n2 * 6 + 2], -f1[2]);
        atomicAddDouble(&F_int[n2 * 6 + 3],  m2[0]);
        atomicAddDouble(&F_int[n2 * 6 + 4],  m2[1]);
        atomicAddDouble(&F_int[n2 * 6 + 5],  m2[2]);
    }
}


// ============================================================================
// C3D10 10-Node Quadratic Tetrahedral Element Kernel
// Tile size: 128 threads (4 Wave32 waves)
// Launch bounds: (128, 2)
// Fits active working set (128 elements * 30 DOFs * 8B ~ 30.7 KB) inside 32 KB L1 vector cache
// ============================================================================

// Constant Gauss quadrature parameters (Hammer 4-point rule)
__constant__ double C_GAUSS_XI[4]   = { 0.1381966011250105, 0.5854101966249685, 0.1381966011250105, 0.1381966011250105 };
__constant__ double C_GAUSS_ETA[4]  = { 0.1381966011250105, 0.1381966011250105, 0.5854101966249685, 0.1381966011250105 };
__constant__ double C_GAUSS_ZETA[4] = { 0.1381966011250105, 0.1381966011250105, 0.1381966011250105, 0.5854101966249685 };
static const double C_WEIGHT = 1.0 / 24.0;

// Helper: evaluate single node natural derivative dN/dxi for node i in 0..9
__device__ inline void get_node_dN_dxi(int node_idx, double xi, double eta, double zeta, double dN[3]) {
    const double L1 = 1.0 - xi - eta - zeta;
    const double L2 = xi;
    const double L3 = eta;
    const double L4 = zeta;

    double dL1 = 0.0, dL2 = 0.0, dL3 = 0.0, dL4 = 0.0;
    switch (node_idx) {
        case 0: dL1 = 4.0 * L1 - 1.0; break;
        case 1: dL2 = 4.0 * L2 - 1.0; break;
        case 2: dL3 = 4.0 * L3 - 1.0; break;
        case 3: dL4 = 4.0 * L4 - 1.0; break;
        case 4: dL1 = 4.0 * L2; dL2 = 4.0 * L1; break;
        case 5: dL2 = 4.0 * L3; dL3 = 4.0 * L2; break;
        case 6: dL1 = 4.0 * L3; dL3 = 4.0 * L1; break;
        case 7: dL1 = 4.0 * L4; dL4 = 4.0 * L1; break;
        case 8: dL2 = 4.0 * L4; dL4 = 4.0 * L2; break;
        case 9: dL3 = 4.0 * L4; dL4 = 4.0 * L3; break;
    }
    // Chain rule: d/dxi = d/dL2 - d/dL1, d/deta = d/dL3 - d/dL1, d/dzeta = d/dL4 - d/dL1
    dN[0] = dL2 - dL1;
    dN[1] = dL3 - dL1;
    dN[2] = dL4 - dL1;
}

#define C3D10_BLOCK_SIZE 32

__launch_bounds__(C3D10_BLOCK_SIZE, 2)
__global__ void c3d10_internal_forces_kernel(
    const double* __restrict__ nodes,          // [n_nodes * 3]
    const int*    __restrict__ solid_elements, // [n_solids * 10]
    const double* __restrict__ props,          // [n_solids * 2] (E, nu)
    const double* __restrict__ U,              // [n_nodes * 6]
    double*       __restrict__ F_int,          // [n_nodes * 6]
    int n_solids
) {
    __shared__ double s_coords[C3D10_BLOCK_SIZE][10][3];
    __shared__ double s_u[C3D10_BLOCK_SIZE][10][3];
    __shared__ double s_f[C3D10_BLOCK_SIZE][10][3];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int t = threadIdx.x;

    for (int e = tid; e < n_solids; e += stride) {
        const int* elem_nodes = &solid_elements[e * 10];
        const double E_mod = props[e * 2 + 0];
        const double nu    = props[e * 2 + 1];

        // Isotropic elasticity constants
        const double factor = E_mod / ((1.0 + nu) * (1.0 - 2.0 * nu));
        const double c11 = factor * (1.0 - nu);
        const double c12 = factor * nu;
        const double c44 = factor * 0.5 * (1.0 - 2.0 * nu);

        // Load into shared memory (LDS)
        #pragma unroll
        for (int i = 0; i < 10; ++i) {
            const int nid = elem_nodes[i];
            s_coords[t][i][0] = nodes[nid * 3 + 0];
            s_coords[t][i][1] = nodes[nid * 3 + 1];
            s_coords[t][i][2] = nodes[nid * 3 + 2];

            s_u[t][i][0] = U[nid * 6 + 0];
            s_u[t][i][1] = U[nid * 6 + 1];
            s_u[t][i][2] = U[nid * 6 + 2];

            s_f[t][i][0] = 0.0;
            s_f[t][i][1] = 0.0;
            s_f[t][i][2] = 0.0;
        }

        // 4 Gauss points evaluated sequentially with on-the-fly shape derivatives
        #pragma nounroll
        for (int g = 0; g < 4; ++g) {
            const double xi   = C_GAUSS_XI[g];
            const double eta  = C_GAUSS_ETA[g];
            const double zeta = C_GAUSS_ZETA[g];

            // 1. Compute Jacobian J = sum_i coords[i] (x) dN_dxi[:, i]
            double J[3][3] = { {0.0} };
            for (int i = 0; i < 10; ++i) {
                double dN[3];
                get_node_dN_dxi(i, xi, eta, zeta, dN);
                J[0][0] += dN[0] * s_coords[t][i][0];
                J[0][1] += dN[1] * s_coords[t][i][0];
                J[0][2] += dN[2] * s_coords[t][i][0];

                J[1][0] += dN[0] * s_coords[t][i][1];
                J[1][1] += dN[1] * s_coords[t][i][1];
                J[1][2] += dN[2] * s_coords[t][i][1];

                J[2][0] += dN[0] * s_coords[t][i][2];
                J[2][1] += dN[1] * s_coords[t][i][2];
                J[2][2] += dN[2] * s_coords[t][i][2];
            }

            // Invert 3x3 Jacobian
            const double c00 = J[1][1] * J[2][2] - J[1][2] * J[2][1];
            const double c01 = J[1][2] * J[2][0] - J[1][0] * J[2][2];
            const double c02 = J[1][0] * J[2][1] - J[1][1] * J[2][0];

            const double detJ = J[0][0] * c00 + J[0][1] * c01 + J[0][2] * c02;
            if (fabs(detJ) < 1e-15) continue;
            const double inv_detJ = 1.0 / detJ;

            const double invJ[3][3] = {
                { c00 * inv_detJ, (J[0][2]*J[2][1] - J[0][1]*J[2][2]) * inv_detJ, (J[0][1]*J[1][2] - J[0][2]*J[1][1]) * inv_detJ },
                { c01 * inv_detJ, (J[0][0]*J[2][2] - J[0][2]*J[2][0]) * inv_detJ, (J[0][2]*J[1][0] - J[0][0]*J[1][2]) * inv_detJ },
                { c02 * inv_detJ, (J[0][1]*J[2][0] - J[0][0]*J[2][1]) * inv_detJ, (J[0][0]*J[1][1] - J[0][1]*J[1][0]) * inv_detJ }
            };

            // 2. Compute strains: eps = sum_i B_i * u_i on the fly
            double eps[6] = { 0.0 };
            for (int i = 0; i < 10; ++i) {
                double dN[3];
                get_node_dN_dxi(i, xi, eta, zeta, dN);
                const double dNx = invJ[0][0] * dN[0] + invJ[0][1] * dN[1] + invJ[0][2] * dN[2];
                const double dNy = invJ[1][0] * dN[0] + invJ[1][1] * dN[1] + invJ[1][2] * dN[2];
                const double dNz = invJ[2][0] * dN[0] + invJ[2][1] * dN[1] + invJ[2][2] * dN[2];

                const double ux = s_u[t][i][0];
                const double uy = s_u[t][i][1];
                const double uz = s_u[t][i][2];

                eps[0] += dNx * ux;
                eps[1] += dNy * uy;
                eps[2] += dNz * uz;
                eps[3] += dNy * ux + dNx * uy;
                eps[4] += dNz * uy + dNy * uz;
                eps[5] += dNz * ux + dNx * uz;
            }

            // 3. Stress: sigma = C * eps
            const double sig[6] = {
                c11 * eps[0] + c12 * eps[1] + c12 * eps[2],
                c12 * eps[0] + c11 * eps[1] + c12 * eps[2],
                c12 * eps[0] + c12 * eps[1] + c11 * eps[2],
                c44 * eps[3],
                c44 * eps[4],
                c44 * eps[5]
            };

            const double w_detJ = C_WEIGHT * detJ;

            // 4. Nodal forces: f_i += w * detJ * B_i^T * sigma on the fly
            for (int i = 0; i < 10; ++i) {
                double dN[3];
                get_node_dN_dxi(i, xi, eta, zeta, dN);
                const double dNx = invJ[0][0] * dN[0] + invJ[0][1] * dN[1] + invJ[0][2] * dN[2];
                const double dNy = invJ[1][0] * dN[0] + invJ[1][1] * dN[1] + invJ[1][2] * dN[2];
                const double dNz = invJ[2][0] * dN[0] + invJ[2][1] * dN[1] + invJ[2][2] * dN[2];

                s_f[t][i][0] += w_detJ * (dNx * sig[0] + dNy * sig[3] + dNz * sig[5]);
                s_f[t][i][1] += w_detJ * (dNy * sig[1] + dNx * sig[3] + dNz * sig[4]);
                s_f[t][i][2] += w_detJ * (dNz * sig[2] + dNy * sig[4] + dNx * sig[5]);
            }
        }

        // Scatter accumulate into F_int with atomic additions
        for (int i = 0; i < 10; ++i) {
            const int nid = elem_nodes[i];
            atomicAddDouble(&F_int[nid * 6 + 0], s_f[t][i][0]);
            atomicAddDouble(&F_int[nid * 6 + 1], s_f[t][i][1]);
            atomicAddDouble(&F_int[nid * 6 + 2], s_f[t][i][2]);
        }
    }
}


// ============================================================================
// Wave32-Tiled Sparse Matrix-Vector Multiplication (CSR SpMV)
// Tile size: 256 threads (8 Wave32 waves per block)
// ============================================================================
__launch_bounds__(256, 4)
__global__ void spmv_csr_kernel(
    int num_rows,
    const int*    __restrict__ row_ptr,
    const int*    __restrict__ col_idx,
    const double* __restrict__ values,
    const double* __restrict__ x,
    double*       __restrict__ y,
    double alpha,
    double beta
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int r = row; r < num_rows; r += stride) {
        int start = row_ptr[r];
        int end   = row_ptr[r + 1];
        double sum = 0.0;
        for (int j = start; j < end; ++j) {
            sum += values[j] * x[col_idx[j]];
        }
        if (beta == 0.0) {
            y[r] = alpha * sum;
        } else {
            y[r] = alpha * sum + beta * y[r];
        }
    }
}


// ============================================================================
// Exported C Interface for Host & Python Dispatch
// ============================================================================
extern "C" {

struct HipDeviceSummary {
    char name[128];
    char arch[32];
    int multiProcessorCount;       // 30 on RX 7800 XT
    int warpSize;                  // 32
    int maxThreadsPerBlock;        // 1024
    int maxThreadsPerMP;           // 2048
    int regsPerBlock;              // 196608
    int sharedMemPerBlock;         // 65536
    int l2CacheSize;               // 4194304
    int optimalBeamBlockSize;      // 256
    int optimalSolidBlockSize;     // 128
    int optimalMinGridSize;        // 60 (30 WGP * 2)
    int optimalMaxGridSize;        // 480 (30 WGP * 16)
};

WNFEA_EXPORT int hip_get_device_summary(HipDeviceSummary* summary) {
    if (!summary) return -1;
    hipDeviceProp_t prop;
    hipError_t err = hipGetDeviceProperties(&prop, 0);
    if (err != hipSuccess) return (int)err;

    snprintf(summary->name, sizeof(summary->name), "%s", prop.name);
    snprintf(summary->arch, sizeof(summary->arch), "%s", prop.gcnArchName);
    summary->multiProcessorCount   = prop.multiProcessorCount;
    summary->warpSize              = prop.warpSize;
    summary->maxThreadsPerBlock    = prop.maxThreadsPerBlock;
    summary->maxThreadsPerMP       = prop.maxThreadsPerMultiProcessor;
    summary->regsPerBlock          = prop.regsPerBlock;
    summary->sharedMemPerBlock     = (int)prop.sharedMemPerBlock;
    summary->l2CacheSize           = prop.l2CacheSize;

    // Architecture-specific optimal configurations:
    // Wave32, 30 WGPs, 128 KB VGPR file, 4 MB L2 cache
    summary->optimalBeamBlockSize  = 256; // 8 Wave32 waves -> <= 64 VGPRs -> 50% occupancy
    summary->optimalSolidBlockSize = C3D10_BLOCK_SIZE; // 1 Wave32 wave (32 threads) -> 0 spills, fits LDS
    summary->optimalMinGridSize    = prop.multiProcessorCount * 2;  // 60 blocks: saturates all 30 WGPs
    summary->optimalMaxGridSize    = prop.multiProcessorCount * 16; // 480 blocks: bounds working set in 4MB L2

    return 0;
}

// Compute 3D Beam internal forces on AMD GPU
WNFEA_EXPORT int hip_compute_beam_forces(
    const double* h_nodes,      // [n_nodes * 3]
    const int*    h_elements,   // [n_elements * 2]
    const double* h_props,      // [n_elements * 6]
    const double* h_e_ref,      // [n_elements * 9]
    const double* h_L0,         // [n_elements]
    const double* h_U,          // [n_nodes * 6]
    double*       h_F_int,      // [n_nodes * 6] (output)
    int n_elements,
    int n_nodes,
    int block_size,             // 0 = use default 256
    int grid_size               // 0 = use auto-tuned grid
) {
    if (block_size <= 0) block_size = 256;
    if (grid_size <= 0) {
        int desired = (n_elements + block_size - 1) / block_size;
        if (desired < 60) desired = 60;
        if (desired > 480) desired = 480;
        grid_size = desired;
    }

    double *d_nodes = nullptr, *d_props = nullptr, *d_e_ref = nullptr, *d_L0 = nullptr, *d_U = nullptr, *d_F = nullptr;
    int *d_elements = nullptr;

    size_t sz_nodes = n_nodes * 3 * sizeof(double);
    size_t sz_elem  = n_elements * 2 * sizeof(int);
    size_t sz_props = n_elements * 6 * sizeof(double);
    size_t sz_eref  = n_elements * 9 * sizeof(double);
    size_t sz_L0    = n_elements * sizeof(double);
    size_t sz_dof   = n_nodes * 6 * sizeof(double);

    hipMalloc(&d_nodes, sz_nodes);
    hipMalloc(&d_elements, sz_elem);
    hipMalloc(&d_props, sz_props);
    hipMalloc(&d_e_ref, sz_eref);
    hipMalloc(&d_L0, sz_L0);
    hipMalloc(&d_U, sz_dof);
    hipMalloc(&d_F, sz_dof);

    hipMemcpy(d_nodes, h_nodes, sz_nodes, hipMemcpyHostToDevice);
    hipMemcpy(d_elements, h_elements, sz_elem, hipMemcpyHostToDevice);
    hipMemcpy(d_props, h_props, sz_props, hipMemcpyHostToDevice);
    hipMemcpy(d_e_ref, h_e_ref, sz_eref, hipMemcpyHostToDevice);
    hipMemcpy(d_L0, h_L0, sz_L0, hipMemcpyHostToDevice);
    hipMemcpy(d_U, h_U, sz_dof, hipMemcpyHostToDevice);
    hipMemset(d_F, 0, sz_dof);

    hipLaunchKernelGGL(beam_internal_forces_kernel, dim3(grid_size), dim3(block_size), 0, 0,
                       d_nodes, d_elements, d_props, d_e_ref, d_L0, d_U, d_F, n_elements);
    hipDeviceSynchronize();

    hipMemcpy(h_F_int, d_F, sz_dof, hipMemcpyDeviceToHost);

    hipFree(d_nodes);
    hipFree(d_elements);
    hipFree(d_props);
    hipFree(d_e_ref);
    hipFree(d_L0);
    hipFree(d_U);
    hipFree(d_F);

    return 0;
}

// Compute C3D10 solid element internal forces on AMD GPU
WNFEA_EXPORT int hip_compute_c3d10_forces(
    const double* h_nodes,          // [n_nodes * 3]
    const int*    h_solid_elements, // [n_solids * 10]
    const double* h_props,          // [n_solids * 2] (E, nu)
    const double* h_U,              // [n_nodes * 6]
    double*       h_F_int,          // [n_nodes * 6] (output)
    int n_solids,
    int n_nodes,
    int block_size,                 // 0 = use default C3D10_BLOCK_SIZE (32)
    int grid_size                   // 0 = use auto-tuned grid
) {
    if (block_size <= 0) block_size = C3D10_BLOCK_SIZE;
    if (grid_size <= 0) {
        int desired = (n_solids + block_size - 1) / block_size;
        if (desired < 60) desired = 60;
        if (desired > 480) desired = 480;
        grid_size = desired;
    }

    double *d_nodes = nullptr, *d_props = nullptr, *d_U = nullptr, *d_F = nullptr;
    int *d_solid_elements = nullptr;

    size_t sz_nodes = n_nodes * 3 * sizeof(double);
    size_t sz_elem  = n_solids * 10 * sizeof(int);
    size_t sz_props = n_solids * 2 * sizeof(double);
    size_t sz_dof   = n_nodes * 6 * sizeof(double);

    hipMalloc(&d_nodes, sz_nodes);
    hipMalloc(&d_solid_elements, sz_elem);
    hipMalloc(&d_props, sz_props);
    hipMalloc(&d_U, sz_dof);
    hipMalloc(&d_F, sz_dof);

    hipMemcpy(d_nodes, h_nodes, sz_nodes, hipMemcpyHostToDevice);
    hipMemcpy(d_solid_elements, h_solid_elements, sz_elem, hipMemcpyHostToDevice);
    hipMemcpy(d_props, h_props, sz_props, hipMemcpyHostToDevice);
    hipMemcpy(d_U, h_U, sz_dof, hipMemcpyHostToDevice);
    hipMemset(d_F, 0, sz_dof);

    hipLaunchKernelGGL(c3d10_internal_forces_kernel, dim3(grid_size), dim3(block_size), 0, 0,
                       d_nodes, d_solid_elements, d_props, d_U, d_F, n_solids);
    hipDeviceSynchronize();

    hipMemcpy(h_F_int, d_F, sz_dof, hipMemcpyDeviceToHost);

    hipFree(d_nodes);
    hipFree(d_solid_elements);
    hipFree(d_props);
    hipFree(d_U);
    hipFree(d_F);

    return 0;
}

// CSR SpMV on AMD GPU
WNFEA_EXPORT int hip_spmv_csr_solve(
    int num_rows,
    const int*    h_row_ptr,
    const int*    h_col_idx,
    const double* h_values,
    const double* h_x,
    double*       h_y,
    double alpha,
    double beta,
    int nnz,
    int block_size,
    int grid_size
) {
    if (block_size <= 0) block_size = 256;
    if (grid_size <= 0) {
        int desired = (num_rows + block_size - 1) / block_size;
        if (desired < 60) desired = 60;
        if (desired > 480) desired = 480;
        grid_size = desired;
    }

    int *d_row_ptr = nullptr, *d_col_idx = nullptr;
    double *d_values = nullptr, *d_x = nullptr, *d_y = nullptr;

    hipMalloc(&d_row_ptr, (num_rows + 1) * sizeof(int));
    hipMalloc(&d_col_idx, nnz * sizeof(int));
    hipMalloc(&d_values,  nnz * sizeof(double));
    hipMalloc(&d_x,       num_rows * sizeof(double));
    hipMalloc(&d_y,       num_rows * sizeof(double));

    hipMemcpy(d_row_ptr, h_row_ptr, (num_rows + 1) * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_col_idx, h_col_idx, nnz * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_values,  h_values,  nnz * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_x,       h_x,       num_rows * sizeof(double), hipMemcpyHostToDevice);
    if (beta != 0.0) {
        hipMemcpy(d_y, h_y, num_rows * sizeof(double), hipMemcpyHostToDevice);
    }

    hipLaunchKernelGGL(spmv_csr_kernel, dim3(grid_size), dim3(block_size), 0, 0,
                       num_rows, d_row_ptr, d_col_idx, d_values, d_x, d_y, alpha, beta);
    hipDeviceSynchronize();

    hipMemcpy(h_y, d_y, num_rows * sizeof(double), hipMemcpyDeviceToHost);

    hipFree(d_row_ptr);
    hipFree(d_col_idx);
    hipFree(d_values);
    hipFree(d_x);
    hipFree(d_y);

    return 0;
}

} // extern "C"
