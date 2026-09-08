// WNFEA Architecture-Tuned HIP Kernels for AMD Radeon RX 7800 XT (gfx1101 / RDNA 3)
// Optimized for Wave32 wavefronts, 30 WGPs, 128 KB VGPRs/SIMD, and 4 MB L2 cache

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>

#if defined(_MSC_VER) || defined(__MINGW32__)
#ifdef __HIP_DEVICE_COMPILE__
#define WNFEA_EXPORT
#else
#define WNFEA_EXPORT __declspec(dllexport)
#endif
#else
#define WNFEA_EXPORT __attribute__((visibility("default")))
#endif

#define HIP_CHECK(call) do { \
    hipError_t _err = call; \
    if (_err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_err)); \
        return (int)_err; \
    } \
} while (0)

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
                const auto xi_c = s_coords[t][i][0];
                const auto yi_c = s_coords[t][i][1];
                const auto zi_c = s_coords[t][i][2];

                J[0][0] += dN[0] * xi_c;
                J[0][1] += dN[0] * yi_c;
                J[0][2] += dN[0] * zi_c;

                J[1][0] += dN[1] * xi_c;
                J[1][1] += dN[1] * yi_c;
                J[1][2] += dN[1] * zi_c;

                J[2][0] += dN[2] * xi_c;
                J[2][1] += dN[2] * yi_c;
                J[2][2] += dN[2] * zi_c;
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
// Vector-4 CSR SpMV (4 threads per row, sub-warp coalesced memory access)
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
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int subwarp_id = tid >> 2;
    int lane = tid & 3;
    int total_subwarps = (blockDim.x * gridDim.x) >> 2;

    for (int r = subwarp_id; r < num_rows; r += total_subwarps) {
        int start = row_ptr[r];
        int end   = row_ptr[r + 1];
        double sum = 0.0;
        for (int j = start + lane; j < end; j += 4) {
            sum += values[j] * x[col_idx[j]];
        }
        sum += __shfl_down(sum, 2, 32);
        sum += __shfl_down(sum, 1, 32);

        if (lane == 0) {
            if (beta == 0.0) {
                y[r] = alpha * sum;
            } else {
                y[r] = alpha * sum + beta * y[r];
            }
        }
    }
}

// ============================================================================
// GPU Algebraic Multigrid (AMG) V-Cycle Kernels (Vector-4 Sub-Warp Optimized)
// Native Wave32 execution: Jacobi smoothing, defect, prolongation & restriction
// ============================================================================

// 0. Vectorized Scale-Multiply for Zero-Initial Guess Down-Sweep Smoother
// x_out[r] = omega * inv_diag[r] * b[r] (full coalesced 624 GB/s bandwidth)
__launch_bounds__(256, 4)
__global__ void vec_scale_mul_f64_kernel(
    int n,
    double omega,
    const double* __restrict__ inv_diag,
    const double* __restrict__ b,
    double*       __restrict__ x
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        x[i] = omega * inv_diag[i] * b[i];
    }
}

__launch_bounds__(256, 4)
__global__ void vec_scale_mul_fp32_kernel(
    int n,
    float omega,
    const float* __restrict__ inv_diag,
    const float* __restrict__ b,
    float*       __restrict__ x
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        x[i] = omega * inv_diag[i] * b[i];
    }
}

__launch_bounds__(256, 4)
__global__ void vec_scale_mul_fp16_kernel(
    int n,
    float omega,
    const __half* __restrict__ inv_diag,
    const __half* __restrict__ b,
    __half*       __restrict__ x
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float val = omega * __half2float(inv_diag[i]) * __half2float(b[i]);
        x[i] = __float2half(val);
    }
}

// 1. Damped Jacobi Smoother (FP64, Vector-4)
// x_out[r] = (x_in ? x_in[r] : 0) + omega * inv_diag[r] * (b[r] - A * x_in)
__launch_bounds__(256, 4)
__global__ void jacobi_smooth_csr_kernel(
    int num_rows,
    const int*    __restrict__ row_ptr,
    const int*    __restrict__ col_idx,
    const double* __restrict__ values,
    const double* __restrict__ inv_diag,
    const double* __restrict__ b,
    const double* __restrict__ x_in,
    double*       __restrict__ x_out,
    double omega
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int subwarp_id = tid >> 2;
    int lane = tid & 3;
    int total_subwarps = (blockDim.x * gridDim.x) >> 2;

    for (int r = subwarp_id; r < num_rows; r += total_subwarps) {
        int start = row_ptr[r];
        int end   = row_ptr[r + 1];

        if (x_in != nullptr) {
            double sum = 0.0;
            for (int j = start + lane; j < end; j += 4) {
                sum += values[j] * x_in[col_idx[j]];
            }
            sum += __shfl_down(sum, 2, 32);
            sum += __shfl_down(sum, 1, 32);

            if (lane == 0) {
                double res = b[r] - sum;
                x_out[r] = x_in[r] + omega * inv_diag[r] * res;
            }
        } else {
            if (lane == 0) {
                x_out[r] = omega * inv_diag[r] * b[r];
            }
        }
    }
}

// Damped Jacobi Smoother (FP32, Vector-4)
__launch_bounds__(256, 4)
__global__ void jacobi_smooth_csr_fp32_kernel(
    int num_rows,
    const int*   __restrict__ row_ptr,
    const int*   __restrict__ col_idx,
    const float* __restrict__ values,
    const float* __restrict__ inv_diag,
    const float* __restrict__ b,
    const float* __restrict__ x_in,
    float*       __restrict__ x_out,
    float omega
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int subwarp_id = tid >> 2;
    int lane = tid & 3;
    int total_subwarps = (blockDim.x * gridDim.x) >> 2;

    for (int r = subwarp_id; r < num_rows; r += total_subwarps) {
        int start = row_ptr[r];
        int end   = row_ptr[r + 1];

        if (x_in != nullptr) {
            float sum = 0.0f;
            for (int j = start + lane; j < end; j += 4) {
                sum += values[j] * x_in[col_idx[j]];
            }
            sum += __shfl_down(sum, 2, 32);
            sum += __shfl_down(sum, 1, 32);

            if (lane == 0) {
                float res = b[r] - sum;
                x_out[r] = x_in[r] + omega * inv_diag[r] * res;
            }
        } else {
            if (lane == 0) {
                x_out[r] = omega * inv_diag[r] * b[r];
            }
        }
    }
}

// Damped Jacobi Smoother (FP16 storage, FP32 accumulation, Vector-4)
__launch_bounds__(256, 4)
__global__ void jacobi_smooth_csr_fp16_kernel(
    int num_rows,
    const int*    __restrict__ row_ptr,
    const int*    __restrict__ col_idx,
    const __half* __restrict__ values,
    const __half* __restrict__ inv_diag,
    const __half* __restrict__ b,
    const __half* __restrict__ x_in,
    __half*       __restrict__ x_out,
    float omega
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int subwarp_id = tid >> 2;
    int lane = tid & 3;
    int total_subwarps = (blockDim.x * gridDim.x) >> 2;

    for (int r = subwarp_id; r < num_rows; r += total_subwarps) {
        int start = row_ptr[r];
        int end   = row_ptr[r + 1];

        if (x_in != nullptr) {
            float sum = 0.0f;
            for (int j = start + lane; j < end; j += 4) {
                sum += __half2float(values[j]) * __half2float(x_in[col_idx[j]]);
            }
            sum += __shfl_down(sum, 2, 32);
            sum += __shfl_down(sum, 1, 32);

            if (lane == 0) {
                float res = __half2float(b[r]) - sum;
                float x_new = __half2float(x_in[r]) + omega * __half2float(inv_diag[r]) * res;
                x_out[r] = __float2half(x_new);
            }
        } else {
            if (lane == 0) {
                float x_new = omega * __half2float(inv_diag[r]) * __half2float(b[r]);
                x_out[r] = __float2half(x_new);
            }
        }
    }
}

// 2. Defect Computation res = b - A * x (Vector-4)
__launch_bounds__(256, 4)
__global__ void defect_csr_kernel(
    int num_rows,
    const int*    __restrict__ row_ptr,
    const int*    __restrict__ col_idx,
    const double* __restrict__ values,
    const double* __restrict__ b,
    const double* __restrict__ x,
    double*       __restrict__ res
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int subwarp_id = tid >> 2;
    int lane = tid & 3;
    int total_subwarps = (blockDim.x * gridDim.x) >> 2;

    for (int r = subwarp_id; r < num_rows; r += total_subwarps) {
        int start = row_ptr[r];
        int end   = row_ptr[r + 1];
        double sum = 0.0;
        for (int j = start + lane; j < end; j += 4) {
            sum += values[j] * x[col_idx[j]];
        }
        sum += __shfl_down(sum, 2, 32);
        sum += __shfl_down(sum, 1, 32);

        if (lane == 0) {
            res[r] = b[r] - sum;
        }
    }
}

__launch_bounds__(256, 4)
__global__ void defect_csr_fp32_kernel(
    int num_rows,
    const int*   __restrict__ row_ptr,
    const int*   __restrict__ col_idx,
    const float* __restrict__ values,
    const float* __restrict__ b,
    const float* __restrict__ x,
    float*       __restrict__ res
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int subwarp_id = tid >> 2;
    int lane = tid & 3;
    int total_subwarps = (blockDim.x * gridDim.x) >> 2;

    for (int r = subwarp_id; r < num_rows; r += total_subwarps) {
        int start = row_ptr[r];
        int end   = row_ptr[r + 1];
        float sum = 0.0f;
        for (int j = start + lane; j < end; j += 4) {
            sum += values[j] * x[col_idx[j]];
        }
        sum += __shfl_down(sum, 2, 32);
        sum += __shfl_down(sum, 1, 32);

        if (lane == 0) {
            res[r] = b[r] - sum;
        }
    }
}

__launch_bounds__(256, 4)
__global__ void defect_csr_fp16_kernel(
    int num_rows,
    const int*    __restrict__ row_ptr,
    const int*    __restrict__ col_idx,
    const __half* __restrict__ values,
    const __half* __restrict__ b,
    const __half* __restrict__ x,
    __half*       __restrict__ res
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int subwarp_id = tid >> 2;
    int lane = tid & 3;
    int total_subwarps = (blockDim.x * gridDim.x) >> 2;

    for (int r = subwarp_id; r < num_rows; r += total_subwarps) {
        int start = row_ptr[r];
        int end   = row_ptr[r + 1];
        float sum = 0.0f;
        for (int j = start + lane; j < end; j += 4) {
            sum += __half2float(values[j]) * __half2float(x[col_idx[j]]);
        }
        sum += __shfl_down(sum, 2, 32);
        sum += __shfl_down(sum, 1, 32);

        if (lane == 0) {
            float r_val = __half2float(b[r]) - sum;
            res[r] = __float2half(r_val);
        }
    }
}

// 3. Fused Prolongation + Accumulate: x += P * e_coarse (Scalar per row, unroll 6)
// Each row of P has <= 6 nonzeros (rigid body modes per node).
// 1 thread per row eliminates shuffle instructions and delivers 4x higher warp throughput.
__launch_bounds__(256, 4)
__global__ void prolongation_add_kernel(
    int num_rows,
    const int*    __restrict__ p_row_ptr,
    const int*    __restrict__ p_col_idx,
    const double* __restrict__ p_values,
    const double* __restrict__ e_coarse,
    double*       __restrict__ x
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (; r < num_rows; r += stride) {
        int start = p_row_ptr[r];
        int end   = p_row_ptr[r + 1];
        double sum = 0.0;
        #pragma unroll 6
        for (int j = start; j < end; ++j) {
            sum += p_values[j] * e_coarse[p_col_idx[j]];
        }
        x[r] += sum;
    }
}

__launch_bounds__(256, 4)
__global__ void prolongation_add_fp32_kernel(
    int num_rows,
    const int*   __restrict__ p_row_ptr,
    const int*   __restrict__ p_col_idx,
    const float* __restrict__ p_values,
    const float* __restrict__ e_coarse,
    float*       __restrict__ x
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (; r < num_rows; r += stride) {
        int start = p_row_ptr[r];
        int end   = p_row_ptr[r + 1];
        float sum = 0.0f;
        #pragma unroll 6
        for (int j = start; j < end; ++j) {
            sum += p_values[j] * e_coarse[p_col_idx[j]];
        }
        x[r] += sum;
    }
}

__launch_bounds__(256, 4)
__global__ void prolongation_add_fp16_kernel(
    int num_rows,
    const int*    __restrict__ p_row_ptr,
    const int*    __restrict__ p_col_idx,
    const __half* __restrict__ p_values,
    const __half* __restrict__ e_coarse,
    __half*       __restrict__ x
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (; r < num_rows; r += stride) {
        int start = p_row_ptr[r];
        int end   = p_row_ptr[r + 1];
        float sum = 0.0f;
        #pragma unroll 6
        for (int j = start; j < end; ++j) {
            sum += __half2float(p_values[j]) * __half2float(e_coarse[p_col_idx[j]]);
        }
        float x_new = __half2float(x[r]) + sum;
        x[r] = __float2half(x_new);
    }
}

// 4. SpMV in FP32 & FP16 (for restriction R * res, Vector-4)
__launch_bounds__(256, 4)
__global__ void spmv_csr_fp32_kernel(
    int num_rows,
    const int*   __restrict__ row_ptr,
    const int*   __restrict__ col_idx,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float*       __restrict__ y,
    float alpha,
    float beta
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int subwarp_id = tid >> 2;
    int lane = tid & 3;
    int total_subwarps = (blockDim.x * gridDim.x) >> 2;

    for (int r = subwarp_id; r < num_rows; r += total_subwarps) {
        int start = row_ptr[r];
        int end   = row_ptr[r + 1];
        float sum = 0.0f;
        for (int j = start + lane; j < end; j += 4) {
            sum += values[j] * x[col_idx[j]];
        }
        sum += __shfl_down(sum, 2, 32);
        sum += __shfl_down(sum, 1, 32);

        if (lane == 0) {
            if (beta == 0.0f) {
                y[r] = alpha * sum;
            } else {
                y[r] = alpha * sum + beta * y[r];
            }
        }
    }
}

__launch_bounds__(256, 4)
__global__ void spmv_csr_fp16_kernel(
    int num_rows,
    const int*    __restrict__ row_ptr,
    const int*    __restrict__ col_idx,
    const __half* __restrict__ values,
    const __half* __restrict__ x,
    __half*       __restrict__ y
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int subwarp_id = tid >> 2;
    int lane = tid & 3;
    int total_subwarps = (blockDim.x * gridDim.x) >> 2;

    for (int r = subwarp_id; r < num_rows; r += total_subwarps) {
        int start = row_ptr[r];
        int end   = row_ptr[r + 1];
        float sum = 0.0f;
        for (int j = start + lane; j < end; j += 4) {
            sum += __half2float(values[j]) * __half2float(x[col_idx[j]]);
        }
        sum += __shfl_down(sum, 2, 32);
        sum += __shfl_down(sum, 1, 32);

        if (lane == 0) {
            y[r] = __float2half(sum);
        }
    }
}

// 5. Symmetric Diagonal Equilibration & Precision Casting
__launch_bounds__(256, 4)
__global__ void vec_pointwise_mult_kernel(
    int n,
    const double* __restrict__ in,
    const double* __restrict__ diag,
    double*       __restrict__ out
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int idx = i; idx < n; idx += stride) {
        out[idx] = in[idx] * diag[idx];
    }
}

__launch_bounds__(256, 4)
__global__ void cast_double_to_float_kernel(int n, const double* __restrict__ in, float* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int idx = i; idx < n; idx += stride) {
        out[idx] = (float)in[idx];
    }
}

__launch_bounds__(256, 4)
__global__ void cast_float_to_double_kernel(int n, const float* __restrict__ in, double* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int idx = i; idx < n; idx += stride) {
        out[idx] = (double)in[idx];
    }
}

__launch_bounds__(256, 4)
__global__ void cast_double_to_half_kernel(int n, const double* __restrict__ in, __half* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int idx = i; idx < n; idx += stride) {
        out[idx] = __float2half((float)in[idx]);
    }
}

__launch_bounds__(256, 4)
__global__ void cast_half_to_double_kernel(int n, const __half* __restrict__ in, double* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int idx = i; idx < n; idx += stride) {
        out[idx] = (double)__half2float(in[idx]);
    }
}

// 6. Wave32 Parallel Reduction Dot Product & BLAS-1 Vector Kernels
__launch_bounds__(256, 4)
__global__ void vec_dot_kernel(
    int n,
    const double* __restrict__ x,
    const double* __restrict__ y,
    double*       __restrict__ result
) {
    __shared__ double s_wave[8];
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wave_id = tid / 32;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    double sum = 0.0;
    for (int i = idx; i < n; i += stride) {
        sum += x[i] * y[i];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down(sum, offset, 32);
    }

    if (lane == 0) {
        s_wave[wave_id] = sum;
    }
    __syncthreads();

    if (wave_id == 0) {
        double w_sum = (lane < 8) ? s_wave[lane] : 0.0;
        #pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            w_sum += __shfl_down(w_sum, offset, 32);
        }
        if (lane == 0) {
            atomicAdd(result, w_sum);
        }
    }
}

__launch_bounds__(256, 4)
__global__ void vec_axpy_kernel(
    int n,
    double alpha,
    const double* __restrict__ x,
    double*       __restrict__ y
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        y[i] += alpha * x[i];
    }
}

__launch_bounds__(256, 4)
__global__ void vec_xpay_kernel(
    int n,
    const double* __restrict__ x,
    double beta,
    double*       __restrict__ y
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        y[i] = x[i] + beta * y[i];
    }
}

// 7. Fused PCG Solution/Residual Update & Norm Reduction Kernel
__launch_bounds__(256, 4)
__global__ void vec_pcg_update_and_norm_kernel(
    int n,
    double alpha,
    const double* __restrict__ p,
    const double* __restrict__ q,
    double*       __restrict__ u,
    double*       __restrict__ r,
    double*       __restrict__ r_norm_sq
) {
    __shared__ double s_wave[8];
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wave_id = tid / 32;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    double sum = 0.0;
    for (int i = idx; i < n; i += stride) {
        double pi = p[i];
        double qi = q[i];
        u[i] += alpha * pi;
        double ri = r[i] - alpha * qi;
        r[i] = ri;
        sum += ri * ri;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down(sum, offset, 32);
    }

    if (lane == 0) {
        s_wave[wave_id] = sum;
    }
    __syncthreads();

    if (wave_id == 0) {
        double w_sum = (lane < 8) ? s_wave[lane] : 0.0;
        #pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            w_sum += __shfl_down(w_sum, offset, 32);
        }
        if (lane == 0) {
            atomicAdd(r_norm_sq, w_sum);
        }
    }
}

// Device-Scalar Direction Update Kernel: p = z + beta * p
// Reads d_rho and d_rho_prev directly on device, eliminates host-device roundtrip latency
__launch_bounds__(256, 4)
__global__ void vec_update_p_device_kernel(
    int n,
    const double* __restrict__ z,
    double*       __restrict__ p,
    const double* __restrict__ d_rho,
    const double* __restrict__ d_rho_prev,
    int iter
) {
    __shared__ double s_beta;
    if (threadIdx.x == 0) {
        if (iter == 0) {
            s_beta = 0.0;
        } else {
            double rho_prev = *d_rho_prev;
            s_beta = (fabs(rho_prev) > 1e-30) ? (*d_rho / rho_prev) : 0.0;
        }
    }
    __syncthreads();

    double beta = s_beta;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if (iter == 0) {
        for (int i = idx; i < n; i += stride) {
            p[i] = z[i];
        }
    } else {
        for (int i = idx; i < n; i += stride) {
            p[i] = z[i] + beta * p[i];
        }
    }
}

// Device-Scalar PCG Solution/Residual Update & Norm Reduction Kernel
// Reads d_rho and d_gamma directly on device, computes alpha = rho / gamma in registers
// Updates u += alpha * p, r -= alpha * q, reduces ||r||^2 to d_r_sq, and saves *d_rho_prev = *d_rho
__launch_bounds__(256, 4)
__global__ void vec_pcg_update_device_kernel(
    int n,
    const double* __restrict__ p,
    const double* __restrict__ q,
    double*       __restrict__ u,
    double*       __restrict__ r,
    const double* __restrict__ d_rho,
    const double* __restrict__ d_gamma,
    double*       __restrict__ d_r_sq,
    double*       __restrict__ d_rho_prev,
    float*        __restrict__ d_r_f32 = nullptr
) {
    __shared__ double s_alpha;
    __shared__ double s_wave[8];

    int tid = threadIdx.x;
    int lane = tid % 32;
    int wave_id = tid / 32;

    if (tid == 0) {
        double gamma = *d_gamma;
        s_alpha = (fabs(gamma) > 1e-30) ? (*d_rho / gamma) : 0.0;
        if (d_rho_prev != nullptr && blockIdx.x == 0) {
            *d_rho_prev = *d_rho;
        }
    }
    __syncthreads();

    double alpha = s_alpha;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    double sum = 0.0;
    for (int i = idx; i < n; i += stride) {
        double pi = p[i];
        double qi = q[i];
        double ui = u[i] + alpha * pi;
        double ri = r[i] - alpha * qi;
        u[i] = ui;
        r[i] = ri;
        if (d_r_f32 != nullptr) {
            d_r_f32[i] = (float)ri;
        }
        sum += ri * ri;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down(sum, offset, 32);
    }

    if (lane == 0) {
        s_wave[wave_id] = sum;
    }
    __syncthreads();

    if (wave_id == 0) {
        double w_sum = (lane < 8) ? s_wave[lane] : 0.0;
        #pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            w_sum += __shfl_down(w_sum, offset, 32);
        }
        if (lane == 0) {
            atomicAdd(d_r_sq, w_sum);
        }
    }
}

// Fused SpMV and Inner Product: q = A * p and gamma = p^T q in ONE kernel
__launch_bounds__(256, 4)
__global__ void spmv_csr_dot_kernel(
    int num_rows,
    const int*    __restrict__ row_ptr,
    const int*    __restrict__ col_idx,
    const double* __restrict__ values,
    const double* __restrict__ x,
    double*       __restrict__ y,
    double*       __restrict__ d_dot
) {
    __shared__ double s_wave[8];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int subwarp_id = tid >> 2;
    int lane = tid & 3;
    int wave_id = threadIdx.x / 32;
    int wave_lane = threadIdx.x % 32;
    int total_subwarps = (blockDim.x * gridDim.x) >> 2;

    double dot_thread = 0.0;

    for (int r = subwarp_id; r < num_rows; r += total_subwarps) {
        int start = row_ptr[r];
        int end   = row_ptr[r + 1];
        double sum = 0.0;
        for (int j = start + lane; j < end; j += 4) {
            sum += values[j] * x[col_idx[j]];
        }
        sum += __shfl_down(sum, 2, 32);
        sum += __shfl_down(sum, 1, 32);

        if (lane == 0) {
            y[r] = sum;
            dot_thread += x[r] * sum;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        dot_thread += __shfl_down(dot_thread, offset, 32);
    }

    if (wave_lane == 0) {
        s_wave[wave_id] = dot_thread;
    }
    __syncthreads();

    if (wave_id == 0) {
        double w_sum = (wave_lane < 8) ? s_wave[wave_lane] : 0.0;
        #pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            w_sum += __shfl_down(w_sum, offset, 32);
        }
        if (wave_lane == 0 && d_dot != nullptr) {
            atomicAdd(d_dot, w_sum);
        }
    }
}

// ============================================================================
// BSR 6x6 (Block Compressed Sparse Row) GPU Kernels
// Optimized for RDNA 3 / Wave32 execution:
// - Eliminates 97.2% of column index memory traffic
// - Vectorized loads with shared-read nodal vector x
// ============================================================================

__launch_bounds__(256, 4)
__global__ void spmv_bsr6x6_fp32_kernel(
    int num_nodes,
    const int*   __restrict__ b_row_ptr,
    const int*   __restrict__ b_col_idx,
    const float* __restrict__ b_values, // [nnz_blocks * 36]
    const float* __restrict__ x,        // [num_nodes * 6]
    float*       __restrict__ y,        // [num_nodes * 6]
    float alpha,
    float beta
) {
    int total_dofs = num_nodes * 6;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int r = idx; r < total_dofs; r += stride) {
        int br = r / 6;
        int k  = r % 6; // intra-block row index: 0..5

        int b_start = b_row_ptr[br];
        int b_end   = b_row_ptr[br + 1];
        float sum = 0.0f;

        for (int b = b_start; b < b_end; ++b) {
            int bc = b_col_idx[b];
            int v_idx = b * 36 + k * 6;
            int x_idx = bc * 6;

            sum += b_values[v_idx + 0] * x[x_idx + 0]
                 + b_values[v_idx + 1] * x[x_idx + 1]
                 + b_values[v_idx + 2] * x[x_idx + 2]
                 + b_values[v_idx + 3] * x[x_idx + 3]
                 + b_values[v_idx + 4] * x[x_idx + 4]
                 + b_values[v_idx + 5] * x[x_idx + 5];
        }

        if (beta == 0.0f) {
            y[r] = alpha * sum;
        } else {
            y[r] = alpha * sum + beta * y[r];
        }
    }
}

__launch_bounds__(256, 4)
__global__ void spmv_bsr6x6_fp64_kernel(
    int num_nodes,
    const int*    __restrict__ b_row_ptr,
    const int*    __restrict__ b_col_idx,
    const double* __restrict__ b_values, // [nnz_blocks * 36]
    const double* __restrict__ x,        // [num_nodes * 6]
    double*       __restrict__ y,        // [num_nodes * 6]
    double alpha,
    double beta
) {
    int total_dofs = num_nodes * 6;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int r = idx; r < total_dofs; r += stride) {
        int br = r / 6;
        int k  = r % 6;

        int b_start = b_row_ptr[br];
        int b_end   = b_row_ptr[br + 1];
        double sum = 0.0;

        for (int b = b_start; b < b_end; ++b) {
            int bc = b_col_idx[b];
            int v_idx = b * 36 + k * 6;
            int x_idx = bc * 6;

            sum += b_values[v_idx + 0] * x[x_idx + 0]
                 + b_values[v_idx + 1] * x[x_idx + 1]
                 + b_values[v_idx + 2] * x[x_idx + 2]
                 + b_values[v_idx + 3] * x[x_idx + 3]
                 + b_values[v_idx + 4] * x[x_idx + 4]
                 + b_values[v_idx + 5] * x[x_idx + 5];
        }

        if (beta == 0.0) {
            y[r] = alpha * sum;
        } else {
            y[r] = alpha * sum + beta * y[r];
        }
    }
}

// Fused BSR 6x6 SpMV + Inner Product Reduction (q = A * p, gamma = p^T q)
__launch_bounds__(256, 4)
__global__ void spmv_bsr6x6_dot_fp32_kernel(
    int num_nodes,
    const int*   __restrict__ b_row_ptr,
    const int*   __restrict__ b_col_idx,
    const float* __restrict__ b_values,
    const float* __restrict__ x,
    float*       __restrict__ y,
    double*      __restrict__ d_dot
) {
    __shared__ double s_wave[8];
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wave_id = tid / 32;

    int total_dofs = num_nodes * 6;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    double dot_thread = 0.0;

    for (int r = idx; r < total_dofs; r += stride) {
        int br = r / 6;
        int k  = r % 6;

        int b_start = b_row_ptr[br];
        int b_end   = b_row_ptr[br + 1];
        float sum = 0.0f;

        for (int b = b_start; b < b_end; ++b) {
            int bc = b_col_idx[b];
            int v_idx = b * 36 + k * 6;
            int x_idx = bc * 6;

            sum += b_values[v_idx + 0] * x[x_idx + 0]
                 + b_values[v_idx + 1] * x[x_idx + 1]
                 + b_values[v_idx + 2] * x[x_idx + 2]
                 + b_values[v_idx + 3] * x[x_idx + 3]
                 + b_values[v_idx + 4] * x[x_idx + 4]
                 + b_values[v_idx + 5] * x[x_idx + 5];
        }

        y[r] = sum;
        dot_thread += (double)x[r] * (double)sum;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        dot_thread += __shfl_down(dot_thread, offset, 32);
    }

    if (lane == 0) {
        s_wave[wave_id] = dot_thread;
    }
    __syncthreads();

    if (wave_id == 0) {
        double w_sum = (lane < 8) ? s_wave[lane] : 0.0;
        #pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            w_sum += __shfl_down(w_sum, offset, 32);
        }
        if (lane == 0 && d_dot != nullptr) {
            atomicAdd(d_dot, w_sum);
        }
    }
}


// Fused Cast and Inner Product: converts z_f32 to z_f64 AND computes rho = r_f64^T z_f64 in ONE kernel
__launch_bounds__(256, 4)
__global__ void cast_and_dot_fp32_to_double_kernel(
    int n,
    const float*  __restrict__ z_f32,
    const double* __restrict__ r_f64,
    double*       __restrict__ z_f64,
    double*       __restrict__ d_rho
) {
    __shared__ double s_wave[8];
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wave_id = tid / 32;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    double sum = 0.0;
    for (int i = idx; i < n; i += stride) {
        double zi = (double)z_f32[i];
        z_f64[i] = zi;
        sum += r_f64[i] * zi;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down(sum, offset, 32);
    }

    if (lane == 0) {
        s_wave[wave_id] = sum;
    }
    __syncthreads();

    if (wave_id == 0) {
        double w_sum = (lane < 8) ? s_wave[lane] : 0.0;
        #pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            w_sum += __shfl_down(w_sum, offset, 32);
        }
        if (lane == 0 && d_rho != nullptr) {
            atomicAdd(d_rho, w_sum);
        }
    }
}

// 8. 100% GPU Coarse Grid Triangular Cholesky Solvers (Zero PCIe Round-Trips)
__launch_bounds__(32, 1)
__global__ void coarse_triangular_solve_f64_kernel(
    int m,
    const double* __restrict__ L,
    const double* __restrict__ b,
    double*       __restrict__ x
) {
    if (threadIdx.x == 0) {
        double y[128];
        for (int i = 0; i < m; ++i) {
            double s = b[i];
            for (int j = 0; j < i; ++j) {
                s -= L[i * m + j] * y[j];
            }
            y[i] = s / L[i * m + i];
        }
        for (int i = m - 1; i >= 0; --i) {
            double s = y[i];
            for (int j = i + 1; j < m; ++j) {
                s -= L[j * m + i] * x[j];
            }
            x[i] = s / L[i * m + i];
        }
    }
}

__launch_bounds__(32, 1)
__global__ void coarse_triangular_solve_f32_kernel(
    int m,
    const double* __restrict__ L,
    const float*  __restrict__ b,
    float*        __restrict__ x
) {
    if (threadIdx.x == 0) {
        double y[128];
        for (int i = 0; i < m; ++i) {
            double s = (double)b[i];
            for (int j = 0; j < i; ++j) {
                s -= L[i * m + j] * y[j];
            }
            y[i] = s / L[i * m + i];
        }
        for (int i = m - 1; i >= 0; --i) {
            double s = y[i];
            for (int j = i + 1; j < m; ++j) {
                s -= L[j * m + i] * y[j];
            }
            x[i] = (float)(s / L[i * m + i]);
        }
    }
}

__launch_bounds__(32, 1)
__global__ void coarse_triangular_solve_f16_kernel(
    int m,
    const double* __restrict__ L,
    const __half* __restrict__ b,
    __half*       __restrict__ x
) {
    if (threadIdx.x == 0) {
        double y[128];
        for (int i = 0; i < m; ++i) {
            double s = (double)__half2float(b[i]);
            for (int j = 0; j < i; ++j) {
                s -= L[i * m + j] * y[j];
            }
            y[i] = s / L[i * m + i];
        }
        for (int i = m - 1; i >= 0; --i) {
            double s = y[i];
            for (int j = i + 1; j < m; ++j) {
                s -= L[j * m + i] * y[j];
            }
            x[i] = __float2half((float)(s / L[i * m + i]));
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

// ============================================================================
// Resident GPU AMG Preconditioner Data Structures & C API
// Keeps all level matrices and scratch vectors in VRAM on the AMD GPU
// ============================================================================

struct HipCSR {
    int num_rows = 0;
    int num_cols = 0;
    int nnz = 0;
    int* d_row_ptr = nullptr;
    int* d_col_idx = nullptr;
    double* d_values_f64 = nullptr;
    float*  d_values_f32 = nullptr;
    __half* d_values_f16 = nullptr;

    void free() {
        if (d_row_ptr)    { hipFree(d_row_ptr);    d_row_ptr = nullptr; }
        if (d_col_idx)    { hipFree(d_col_idx);    d_col_idx = nullptr; }
        if (d_values_f64) { hipFree(d_values_f64); d_values_f64 = nullptr; }
        if (d_values_f32) { hipFree(d_values_f32); d_values_f32 = nullptr; }
        if (d_values_f16) { hipFree(d_values_f16); d_values_f16 = nullptr; }
    }
};

struct HipAMGLevelData {
    int fine_rows = 0;
    int coarse_rows = 0;
    HipCSR A;
    double* d_inv_diag_f64 = nullptr;
    float*  d_inv_diag_f32 = nullptr;
    __half* d_inv_diag_f16 = nullptr;
    HipCSR P;
    HipCSR R;

    // Persistent work vectors in GPU VRAM (allocated once at setup)
    double* d_b_f64 = nullptr;
    double* d_x_f64 = nullptr;
    double* d_x_temp_f64 = nullptr;
    double* d_res_f64 = nullptr;

    float*  d_b_f32 = nullptr;
    float*  d_x_f32 = nullptr;
    float*  d_x_temp_f32 = nullptr;
    float*  d_res_f32 = nullptr;

    __half* d_b_f16 = nullptr;
    __half* d_x_f16 = nullptr;
    __half* d_x_temp_f16 = nullptr;
    __half* d_res_f16 = nullptr;

    void free() {
        A.free();
        P.free();
        R.free();
        if (d_inv_diag_f64) { hipFree(d_inv_diag_f64); d_inv_diag_f64 = nullptr; }
        if (d_inv_diag_f32) { hipFree(d_inv_diag_f32); d_inv_diag_f32 = nullptr; }
        if (d_inv_diag_f16) { hipFree(d_inv_diag_f16); d_inv_diag_f16 = nullptr; }

        if (d_b_f64)        { hipFree(d_b_f64);        d_b_f64 = nullptr; }
        if (d_x_f64)        { hipFree(d_x_f64);        d_x_f64 = nullptr; }
        if (d_x_temp_f64)   { hipFree(d_x_temp_f64);   d_x_temp_f64 = nullptr; }
        if (d_res_f64)      { hipFree(d_res_f64);      d_res_f64 = nullptr; }

        if (d_b_f32)        { hipFree(d_b_f32);        d_b_f32 = nullptr; }
        if (d_x_f32)        { hipFree(d_x_f32);        d_x_f32 = nullptr; }
        if (d_x_temp_f32)   { hipFree(d_x_temp_f32);   d_x_temp_f32 = nullptr; }
        if (d_res_f32)      { hipFree(d_res_f32);      d_res_f32 = nullptr; }

        if (d_b_f16)        { hipFree(d_b_f16);        d_b_f16 = nullptr; }
        if (d_x_f16)        { hipFree(d_x_f16);        d_x_f16 = nullptr; }
        if (d_x_temp_f16)   { hipFree(d_x_temp_f16);   d_x_temp_f16 = nullptr; }
        if (d_res_f16)      { hipFree(d_res_f16);      d_res_f16 = nullptr; }
    }
};

struct HipAMGSolver {
    int num_levels = 0;
    int fine_size = 0;
    double* d_fine_inv_sqrt_d_f64 = nullptr;
    double* d_work_in_f64 = nullptr;
    double* d_work_out_f64 = nullptr;
    std::vector<HipAMGLevelData> levels;
    int coarse_sweeps = 4;

    // Coarsest level direct solve on CPU/GPU
    int coarsest_size = 0;
    double* d_coarse_L = nullptr;
    std::vector<double> h_coarse_L;
    std::vector<double> h_coarse_r;
    std::vector<double> h_coarse_x;
    std::vector<double> h_coarse_y;

    // 100% Resident GPU PCG Buffers
    double* d_pcg_u = nullptr;
    double* d_pcg_r = nullptr;
    double* d_pcg_p = nullptr;
    double* d_pcg_q = nullptr;
    double* d_pcg_z = nullptr;
    double* d_pcg_scalar = nullptr;
    double* d_pcg_rho = nullptr;
    double* d_pcg_rho_prev = nullptr;
    double* d_pcg_gamma = nullptr;
    double* d_pcg_r_sq = nullptr;

    void free() {
        for (auto& lvl : levels) {
            lvl.free();
        }
        levels.clear();
        if (d_coarse_L)            { hipFree(d_coarse_L);            d_coarse_L = nullptr; }
        if (d_fine_inv_sqrt_d_f64) { hipFree(d_fine_inv_sqrt_d_f64); d_fine_inv_sqrt_d_f64 = nullptr; }
        if (d_work_in_f64)         { hipFree(d_work_in_f64);         d_work_in_f64 = nullptr; }
        if (d_work_out_f64)        { hipFree(d_work_out_f64);        d_work_out_f64 = nullptr; }

        if (d_pcg_u)               { hipFree(d_pcg_u);               d_pcg_u = nullptr; }
        if (d_pcg_r)               { hipFree(d_pcg_r);               d_pcg_r = nullptr; }
        if (d_pcg_p)               { hipFree(d_pcg_p);               d_pcg_p = nullptr; }
        if (d_pcg_q)               { hipFree(d_pcg_q);               d_pcg_q = nullptr; }
        if (d_pcg_z)               { hipFree(d_pcg_z);               d_pcg_z = nullptr; }
        if (d_pcg_scalar)          { hipFree(d_pcg_scalar);          d_pcg_scalar = nullptr; }
        if (d_pcg_rho)             { hipFree(d_pcg_rho);             d_pcg_rho = nullptr; }
        if (d_pcg_rho_prev)        { hipFree(d_pcg_rho_prev);        d_pcg_rho_prev = nullptr; }
        if (d_pcg_gamma)           { hipFree(d_pcg_gamma);           d_pcg_gamma = nullptr; }
        if (d_pcg_r_sq)            { hipFree(d_pcg_r_sq);            d_pcg_r_sq = nullptr; }
    }
};

inline void solve_coarse_cholesky_internal(HipAMGSolver* solver, const double* r, double* x) {
    int m = solver->coarsest_size;
    const double* L = solver->h_coarse_L.data();
    double* y = solver->h_coarse_y.data();

    // Forward substitution: L * y = r
    for (int i = 0; i < m; ++i) {
        double s = r[i];
        for (int j = 0; j < i; ++j) {
            s -= L[i * m + j] * y[j];
        }
        y[i] = s / L[i * m + i];
    }
    // Backward substitution: L^T * x = y
    for (int i = m - 1; i >= 0; --i) {
        double s = y[i];
        for (int j = i + 1; j < m; ++j) {
            s -= L[j * m + i] * x[j];
        }
        x[i] = s / L[i * m + i];
    }
}

WNFEA_EXPORT void* hip_amg_create(int num_levels, int fine_size, const double* h_inv_sqrt_d) {
    if (num_levels <= 0 || fine_size <= 0) return nullptr;

    HipAMGSolver* solver = new HipAMGSolver();
    solver->num_levels = num_levels;
    solver->fine_size = fine_size;
    solver->levels.resize(num_levels);

    hipMalloc(&solver->d_fine_inv_sqrt_d_f64, fine_size * sizeof(double));
    hipMalloc(&solver->d_work_in_f64, fine_size * sizeof(double));
    hipMalloc(&solver->d_work_out_f64, fine_size * sizeof(double));

    hipMalloc(&solver->d_pcg_u, fine_size * sizeof(double));
    hipMalloc(&solver->d_pcg_r, fine_size * sizeof(double));
    hipMalloc(&solver->d_pcg_p, fine_size * sizeof(double));
    hipMalloc(&solver->d_pcg_q, fine_size * sizeof(double));
    hipMalloc(&solver->d_pcg_z, fine_size * sizeof(double));
    hipMalloc(&solver->d_pcg_scalar, sizeof(double));
    hipMalloc(&solver->d_pcg_rho, sizeof(double));
    hipMalloc(&solver->d_pcg_rho_prev, sizeof(double));
    hipMalloc(&solver->d_pcg_gamma, sizeof(double));
    hipMalloc(&solver->d_pcg_r_sq, sizeof(double));

    if (h_inv_sqrt_d != nullptr) {
        hipMemcpy(solver->d_fine_inv_sqrt_d_f64, h_inv_sqrt_d, fine_size * sizeof(double), hipMemcpyHostToDevice);
    } else {
        std::vector<double> ones(fine_size, 1.0);
        hipMemcpy(solver->d_fine_inv_sqrt_d_f64, ones.data(), fine_size * sizeof(double), hipMemcpyHostToDevice);
    }

    return (void*)solver;
}

WNFEA_EXPORT int hip_amg_set_level(
    void* handle,
    int level_idx,
    int fine_rows,
    int coarse_rows,
    // Matrix A
    int a_nnz,
    const int* h_a_row_ptr,
    const int* h_a_col_idx,
    const double* h_a_values,
    const double* h_inv_diag,
    // Matrix P (null if coarsest)
    int p_nnz,
    const int* h_p_row_ptr,
    const int* h_p_col_idx,
    const double* h_p_values,
    // Matrix R (null if coarsest)
    int r_nnz,
    const int* h_r_row_ptr,
    const int* h_r_col_idx,
    const double* h_r_values
) {
    if (!handle) return -1;
    HipAMGSolver* solver = (HipAMGSolver*)handle;
    if (level_idx < 0 || level_idx >= solver->num_levels) return -2;

    auto& lvl = solver->levels[level_idx];
    lvl.fine_rows = fine_rows;
    lvl.coarse_rows = coarse_rows;

    // 1. Matrix A
    lvl.A.num_rows = fine_rows;
    lvl.A.num_cols = fine_rows;
    lvl.A.nnz = a_nnz;
    HIP_CHECK(hipMalloc(&lvl.A.d_row_ptr, (fine_rows + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&lvl.A.d_col_idx, a_nnz * sizeof(int)));
    HIP_CHECK(hipMalloc(&lvl.A.d_values_f64, a_nnz * sizeof(double)));
    HIP_CHECK(hipMalloc(&lvl.A.d_values_f32, a_nnz * sizeof(float)));
    HIP_CHECK(hipMalloc(&lvl.A.d_values_f16, a_nnz * sizeof(__half)));

    HIP_CHECK(hipMemcpy(lvl.A.d_row_ptr, h_a_row_ptr, (fine_rows + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(lvl.A.d_col_idx, h_a_col_idx, a_nnz * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(lvl.A.d_values_f64, h_a_values, a_nnz * sizeof(double), hipMemcpyHostToDevice));

    std::vector<float> a_f32(a_nnz);
    std::vector<__half> a_f16(a_nnz);
    for (int i = 0; i < a_nnz; ++i) {
        a_f32[i] = (float)h_a_values[i];
        a_f16[i] = __float2half((float)h_a_values[i]);
    }
    HIP_CHECK(hipMemcpy(lvl.A.d_values_f32, a_f32.data(), a_nnz * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(lvl.A.d_values_f16, a_f16.data(), a_nnz * sizeof(__half), hipMemcpyHostToDevice));

    // Inv Diag
    HIP_CHECK(hipMalloc(&lvl.d_inv_diag_f64, fine_rows * sizeof(double)));
    HIP_CHECK(hipMalloc(&lvl.d_inv_diag_f32, fine_rows * sizeof(float)));
    HIP_CHECK(hipMalloc(&lvl.d_inv_diag_f16, fine_rows * sizeof(__half)));
    HIP_CHECK(hipMemcpy(lvl.d_inv_diag_f64, h_inv_diag, fine_rows * sizeof(double), hipMemcpyHostToDevice));

    std::vector<float> diag_f32(fine_rows);
    std::vector<__half> diag_f16(fine_rows);
    for (int i = 0; i < fine_rows; ++i) {
        diag_f32[i] = (float)h_inv_diag[i];
        diag_f16[i] = __float2half((float)h_inv_diag[i]);
    }
    HIP_CHECK(hipMemcpy(lvl.d_inv_diag_f32, diag_f32.data(), fine_rows * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(lvl.d_inv_diag_f16, diag_f16.data(), fine_rows * sizeof(__half), hipMemcpyHostToDevice));

    // 2. Matrix P (if not coarsest)
    if (p_nnz > 0 && h_p_row_ptr && h_p_col_idx && h_p_values) {
        lvl.P.num_rows = fine_rows;
        lvl.P.num_cols = coarse_rows;
        lvl.P.nnz = p_nnz;
        HIP_CHECK(hipMalloc(&lvl.P.d_row_ptr, (fine_rows + 1) * sizeof(int)));
        HIP_CHECK(hipMalloc(&lvl.P.d_col_idx, p_nnz * sizeof(int)));
        HIP_CHECK(hipMalloc(&lvl.P.d_values_f64, p_nnz * sizeof(double)));
        HIP_CHECK(hipMalloc(&lvl.P.d_values_f32, p_nnz * sizeof(float)));
        HIP_CHECK(hipMalloc(&lvl.P.d_values_f16, p_nnz * sizeof(__half)));

        HIP_CHECK(hipMemcpy(lvl.P.d_row_ptr, h_p_row_ptr, (fine_rows + 1) * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(lvl.P.d_col_idx, h_p_col_idx, p_nnz * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(lvl.P.d_values_f64, h_p_values, p_nnz * sizeof(double), hipMemcpyHostToDevice));

        std::vector<float> p_f32(p_nnz);
        std::vector<__half> p_f16(p_nnz);
        for (int i = 0; i < p_nnz; ++i) {
            p_f32[i] = (float)h_p_values[i];
            p_f16[i] = __float2half((float)h_p_values[i]);
        }
        HIP_CHECK(hipMemcpy(lvl.P.d_values_f32, p_f32.data(), p_nnz * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(lvl.P.d_values_f16, p_f16.data(), p_nnz * sizeof(__half), hipMemcpyHostToDevice));
    }

    // 3. Matrix R (if not coarsest)
    if (r_nnz > 0 && h_r_row_ptr && h_r_col_idx && h_r_values) {
        lvl.R.num_rows = coarse_rows;
        lvl.R.num_cols = fine_rows;
        lvl.R.nnz = r_nnz;
        HIP_CHECK(hipMalloc(&lvl.R.d_row_ptr, (coarse_rows + 1) * sizeof(int)));
        HIP_CHECK(hipMalloc(&lvl.R.d_col_idx, r_nnz * sizeof(int)));
        HIP_CHECK(hipMalloc(&lvl.R.d_values_f64, r_nnz * sizeof(double)));
        HIP_CHECK(hipMalloc(&lvl.R.d_values_f32, r_nnz * sizeof(float)));
        HIP_CHECK(hipMalloc(&lvl.R.d_values_f16, r_nnz * sizeof(__half)));

        HIP_CHECK(hipMemcpy(lvl.R.d_row_ptr, h_r_row_ptr, (coarse_rows + 1) * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(lvl.R.d_col_idx, h_r_col_idx, r_nnz * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(lvl.R.d_values_f64, h_r_values, r_nnz * sizeof(double), hipMemcpyHostToDevice));

        std::vector<float> r_f32(r_nnz);
        std::vector<__half> r_f16(r_nnz);
        for (int i = 0; i < r_nnz; ++i) {
            r_f32[i] = (float)h_r_values[i];
            r_f16[i] = __float2half((float)h_r_values[i]);
        }
        HIP_CHECK(hipMemcpy(lvl.R.d_values_f32, r_f32.data(), r_nnz * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(lvl.R.d_values_f16, r_f16.data(), r_nnz * sizeof(__half), hipMemcpyHostToDevice));
    }

    // 4. Pre-allocate work vectors in GPU VRAM
    HIP_CHECK(hipMalloc(&lvl.d_b_f64,      fine_rows * sizeof(double)));
    HIP_CHECK(hipMalloc(&lvl.d_x_f64,      fine_rows * sizeof(double)));
    HIP_CHECK(hipMalloc(&lvl.d_x_temp_f64, fine_rows * sizeof(double)));
    HIP_CHECK(hipMalloc(&lvl.d_res_f64,    fine_rows * sizeof(double)));

    HIP_CHECK(hipMalloc(&lvl.d_b_f32,      fine_rows * sizeof(float)));
    HIP_CHECK(hipMalloc(&lvl.d_x_f32,      fine_rows * sizeof(float)));
    HIP_CHECK(hipMalloc(&lvl.d_x_temp_f32, fine_rows * sizeof(float)));
    HIP_CHECK(hipMalloc(&lvl.d_res_f32,    fine_rows * sizeof(float)));

    HIP_CHECK(hipMalloc(&lvl.d_b_f16,      fine_rows * sizeof(__half)));
    HIP_CHECK(hipMalloc(&lvl.d_x_f16,      fine_rows * sizeof(__half)));
    HIP_CHECK(hipMalloc(&lvl.d_x_temp_f16, fine_rows * sizeof(__half)));
    HIP_CHECK(hipMalloc(&lvl.d_res_f16,    fine_rows * sizeof(__half)));

    return 0;
}

WNFEA_EXPORT int hip_amg_set_coarse_cholesky(
    void* handle,
    int coarse_size,
    const double* h_L_dense
) {
    if (!handle || coarse_size <= 0 || !h_L_dense) return -1;
    HipAMGSolver* solver = (HipAMGSolver*)handle;
    solver->coarsest_size = coarse_size;
    solver->h_coarse_L.assign(h_L_dense, h_L_dense + coarse_size * coarse_size);
    solver->h_coarse_r.resize(coarse_size);
    solver->h_coarse_x.resize(coarse_size);
    solver->h_coarse_y.resize(coarse_size);

    if (solver->d_coarse_L) { hipFree(solver->d_coarse_L); solver->d_coarse_L = nullptr; }
    HIP_CHECK(hipMalloc(&solver->d_coarse_L, coarse_size * coarse_size * sizeof(double)));
    HIP_CHECK(hipMemcpy(solver->d_coarse_L, h_L_dense, coarse_size * coarse_size * sizeof(double), hipMemcpyHostToDevice));
    return 0;
}

static int hip_amg_apply_pure_vcycle_device(
    HipAMGSolver* solver,
    const double* d_r,
    double* d_z,
    int precision_mode,
    double* d_rho = nullptr
) {
    int fine_size = solver->fine_size;
    int num_levels = solver->num_levels;
    int block_size = 256;
    int grid_0_1d = std::min(std::max((fine_size + block_size - 1) / block_size, 60), 480);
    int c_sweeps = solver->coarse_sweeps;
    if (c_sweeps <= 0) c_sweeps = 4;

    // MODE 0: FP64 V-Cycle
    if (precision_mode == 0) {
        if (d_r != nullptr && d_r != solver->levels[0].d_b_f64) {
            HIP_CHECK(hipMemcpyAsync(solver->levels[0].d_b_f64, d_r, fine_size * sizeof(double), hipMemcpyDeviceToDevice));
        }

        // Downward sweep
        for (int l = 0; l < num_levels - 1; ++l) {
            auto& lvl = solver->levels[l];
            int grid_l = std::min(std::max((lvl.fine_rows * 4 + block_size - 1) / block_size, 60), 1920);
            int grid_c = std::min(std::max((lvl.coarse_rows * 4 + block_size - 1) / block_size, 60), 1920);

            // Pre-smooth: x_0 = 0 -> output to lvl.d_x_f64
            int grid_vec = std::min(std::max((lvl.fine_rows + block_size - 1) / block_size, 60), 480);
            hipLaunchKernelGGL(vec_scale_mul_f64_kernel, dim3(grid_vec), dim3(block_size), 0, 0,
                               lvl.fine_rows, 0.67, lvl.d_inv_diag_f64, lvl.d_b_f64, lvl.d_x_f64);

            // Defect: res_l = b_l - A_l * x_l
            hipLaunchKernelGGL(defect_csr_kernel, dim3(grid_l), dim3(block_size), 0, 0,
                               lvl.fine_rows, lvl.A.d_row_ptr, lvl.A.d_col_idx, lvl.A.d_values_f64,
                               lvl.d_b_f64, lvl.d_x_f64, lvl.d_res_f64);

            // Restriction: r_{l+1} = R_l * res_l -> stored in solver->levels[l+1].d_b_f64
            hipLaunchKernelGGL(spmv_csr_kernel, dim3(grid_c), dim3(block_size), 0, 0,
                               lvl.coarse_rows, lvl.R.d_row_ptr, lvl.R.d_col_idx, lvl.R.d_values_f64,
                               lvl.d_res_f64, solver->levels[l + 1].d_b_f64, 1.0, 0.0);
        }

        // Coarsest solve
        int last = num_levels - 1;
        auto& coarsest = solver->levels[last];
        int c_size = coarsest.fine_rows;

        if (solver->coarsest_size == c_size && solver->d_coarse_L != nullptr && c_size <= 128) {
            hipLaunchKernelGGL(coarse_triangular_solve_f64_kernel, dim3(1), dim3(32), 0, 0,
                               c_size, solver->d_coarse_L, coarsest.d_b_f64, coarsest.d_x_f64);
        } else if (solver->coarsest_size == c_size && !solver->h_coarse_L.empty() && (int)solver->h_coarse_L.size() == c_size * c_size) {
            HIP_CHECK(hipMemcpy(solver->h_coarse_r.data(), coarsest.d_b_f64, c_size * sizeof(double), hipMemcpyDeviceToHost));
            solve_coarse_cholesky_internal(solver, solver->h_coarse_r.data(), solver->h_coarse_x.data());
            HIP_CHECK(hipMemcpy(coarsest.d_x_f64, solver->h_coarse_x.data(), c_size * sizeof(double), hipMemcpyHostToDevice));
        } else {
            int grid_last = std::min(std::max((c_size * 4 + block_size - 1) / block_size, 60), 480);
            for (int it = 0; it < c_sweeps; ++it) {
                hipLaunchKernelGGL(jacobi_smooth_csr_kernel, dim3(grid_last), dim3(block_size), 0, 0,
                                   coarsest.fine_rows, coarsest.A.d_row_ptr, coarsest.A.d_col_idx, coarsest.A.d_values_f64,
                                   coarsest.d_inv_diag_f64, coarsest.d_b_f64, (it == 0 ? (const double*)nullptr : coarsest.d_x_f64),
                                   coarsest.d_x_temp_f64, 0.67);
                std::swap(coarsest.d_x_f64, coarsest.d_x_temp_f64);
            }
        }

        // Upward sweep
        for (int l = num_levels - 2; l >= 0; --l) {
            auto& lvl = solver->levels[l];
            int grid_l = std::min(std::max((lvl.fine_rows * 4 + block_size - 1) / block_size, 60), 1920);
            const double* e_coarse = solver->levels[l + 1].d_x_f64;

            int grid_p = std::min(std::max((lvl.fine_rows + block_size - 1) / block_size, 60), 480);
            hipLaunchKernelGGL(prolongation_add_kernel, dim3(grid_p), dim3(block_size), 0, 0,
                               lvl.fine_rows, lvl.P.d_row_ptr, lvl.P.d_col_idx, lvl.P.d_values_f64,
                               e_coarse, lvl.d_x_f64);

            hipLaunchKernelGGL(jacobi_smooth_csr_kernel, dim3(grid_l), dim3(block_size), 0, 0,
                               lvl.fine_rows, lvl.A.d_row_ptr, lvl.A.d_col_idx, lvl.A.d_values_f64,
                               lvl.d_inv_diag_f64, lvl.d_b_f64, lvl.d_x_f64, lvl.d_x_temp_f64, 0.67);

            std::swap(lvl.d_x_f64, lvl.d_x_temp_f64);
        }

        if (d_z != solver->levels[0].d_x_f64) {
            HIP_CHECK(hipMemcpyAsync(d_z, solver->levels[0].d_x_f64, fine_size * sizeof(double), hipMemcpyDeviceToDevice));
        }
        if (d_rho != nullptr) {
            hipMemsetAsync(d_rho, 0, sizeof(double));
            hipLaunchKernelGGL(vec_dot_kernel, dim3(60), dim3(block_size), 0, 0,
                               fine_size, d_z, solver->d_pcg_r, d_rho);
        }
        return 0;
    }

    // MODE 1: FP32 V-Cycle (2x memory bandwidth acceleration)
    if (precision_mode == 1) {
        if (d_r != nullptr) {
            hipLaunchKernelGGL(cast_double_to_float_kernel, dim3(grid_0_1d), dim3(block_size), 0, 0,
                               fine_size, d_r, solver->levels[0].d_b_f32);
        }

        // Downward sweep
        for (int l = 0; l < num_levels - 1; ++l) {
            auto& lvl = solver->levels[l];
            int grid_l = std::min(std::max((lvl.fine_rows * 4 + block_size - 1) / block_size, 60), 1920);
            int grid_c = std::min(std::max((lvl.coarse_rows * 4 + block_size - 1) / block_size, 60), 1920);

            int grid_vec = std::min(std::max((lvl.fine_rows + block_size - 1) / block_size, 60), 480);
            hipLaunchKernelGGL(vec_scale_mul_fp32_kernel, dim3(grid_vec), dim3(block_size), 0, 0,
                               lvl.fine_rows, 0.67f, lvl.d_inv_diag_f32, lvl.d_b_f32, lvl.d_x_f32);

            hipLaunchKernelGGL(defect_csr_fp32_kernel, dim3(grid_l), dim3(block_size), 0, 0,
                               lvl.fine_rows, lvl.A.d_row_ptr, lvl.A.d_col_idx, lvl.A.d_values_f32,
                               lvl.d_b_f32, lvl.d_x_f32, lvl.d_res_f32);

            hipLaunchKernelGGL(spmv_csr_fp32_kernel, dim3(grid_c), dim3(block_size), 0, 0,
                               lvl.coarse_rows, lvl.R.d_row_ptr, lvl.R.d_col_idx, lvl.R.d_values_f32,
                               lvl.d_res_f32, solver->levels[l + 1].d_b_f32, 1.0f, 0.0f);
        }

        // Coarsest solve
        int last = num_levels - 1;
        auto& coarsest = solver->levels[last];
        int c_size = coarsest.fine_rows;

        if (solver->coarsest_size == c_size && solver->d_coarse_L != nullptr && c_size <= 128) {
            hipLaunchKernelGGL(coarse_triangular_solve_f32_kernel, dim3(1), dim3(32), 0, 0,
                               c_size, solver->d_coarse_L, coarsest.d_b_f32, coarsest.d_x_f32);
        } else if (solver->coarsest_size == c_size && !solver->h_coarse_L.empty() && (int)solver->h_coarse_L.size() == c_size * c_size) {
            std::vector<float> h_r_f32(c_size);
            HIP_CHECK(hipMemcpy(h_r_f32.data(), coarsest.d_b_f32, c_size * sizeof(float), hipMemcpyDeviceToHost));
            for (int i = 0; i < c_size; ++i) solver->h_coarse_r[i] = (double)h_r_f32[i];
            solve_coarse_cholesky_internal(solver, solver->h_coarse_r.data(), solver->h_coarse_x.data());
            std::vector<float> h_x_f32(c_size);
            for (int i = 0; i < c_size; ++i) h_x_f32[i] = (float)solver->h_coarse_x[i];
            HIP_CHECK(hipMemcpy(coarsest.d_x_f32, h_x_f32.data(), c_size * sizeof(float), hipMemcpyHostToDevice));
        } else {
            int grid_last = std::min(std::max((c_size * 4 + block_size - 1) / block_size, 60), 480);
            for (int it = 0; it < c_sweeps; ++it) {
                hipLaunchKernelGGL(jacobi_smooth_csr_fp32_kernel, dim3(grid_last), dim3(block_size), 0, 0,
                                   coarsest.fine_rows, coarsest.A.d_row_ptr, coarsest.A.d_col_idx, coarsest.A.d_values_f32,
                                   coarsest.d_inv_diag_f32, coarsest.d_b_f32, (it == 0 ? (const float*)nullptr : coarsest.d_x_f32),
                                   coarsest.d_x_temp_f32, 0.67f);
                std::swap(coarsest.d_x_f32, coarsest.d_x_temp_f32);
            }
        }

        // Upward sweep
        for (int l = num_levels - 2; l >= 0; --l) {
            auto& lvl = solver->levels[l];
            int grid_l = std::min(std::max((lvl.fine_rows * 4 + block_size - 1) / block_size, 60), 1920);
            const float* e_coarse = solver->levels[l + 1].d_x_f32;

            int grid_p = std::min(std::max((lvl.fine_rows + block_size - 1) / block_size, 60), 480);
            hipLaunchKernelGGL(prolongation_add_fp32_kernel, dim3(grid_p), dim3(block_size), 0, 0,
                               lvl.fine_rows, lvl.P.d_row_ptr, lvl.P.d_col_idx, lvl.P.d_values_f32,
                               e_coarse, lvl.d_x_f32);

            hipLaunchKernelGGL(jacobi_smooth_csr_fp32_kernel, dim3(grid_l), dim3(block_size), 0, 0,
                               lvl.fine_rows, lvl.A.d_row_ptr, lvl.A.d_col_idx, lvl.A.d_values_f32,
                               lvl.d_inv_diag_f32, lvl.d_b_f32, lvl.d_x_f32, lvl.d_x_temp_f32, 0.67f);

            std::swap(lvl.d_x_f32, lvl.d_x_temp_f32);
        }

        // Fused Cast and Dot Product: converts z_f32 to z_f64 and computes rho = r^T z
        if (d_rho != nullptr) {
            hipMemsetAsync(d_rho, 0, sizeof(double));
            hipLaunchKernelGGL(cast_and_dot_fp32_to_double_kernel, dim3(grid_0_1d), dim3(block_size), 0, 0,
                               fine_size, solver->levels[0].d_x_f32, solver->d_pcg_r, d_z, d_rho);
        } else {
            hipLaunchKernelGGL(cast_float_to_double_kernel, dim3(grid_0_1d), dim3(block_size), 0, 0,
                               fine_size, solver->levels[0].d_x_f32, d_z);
        }
        return 0;
    }

    // MODE 2: FP16 V-Cycle (4x memory bandwidth acceleration)
    if (precision_mode == 2) {
        if (d_r != nullptr) {
            hipLaunchKernelGGL(cast_double_to_half_kernel, dim3(grid_0_1d), dim3(block_size), 0, 0,
                               fine_size, d_r, solver->levels[0].d_b_f16);
        }

        // Downward sweep
        for (int l = 0; l < num_levels - 1; ++l) {
            auto& lvl = solver->levels[l];
            int grid_l = std::min(std::max((lvl.fine_rows * 4 + block_size - 1) / block_size, 60), 1920);
            int grid_c = std::min(std::max((lvl.coarse_rows * 4 + block_size - 1) / block_size, 60), 1920);

            int grid_vec = std::min(std::max((lvl.fine_rows + block_size - 1) / block_size, 60), 480);
            hipLaunchKernelGGL(vec_scale_mul_fp16_kernel, dim3(grid_vec), dim3(block_size), 0, 0,
                               lvl.fine_rows, 0.67f, lvl.d_inv_diag_f16, lvl.d_b_f16, lvl.d_x_f16);

            hipLaunchKernelGGL(defect_csr_fp16_kernel, dim3(grid_l), dim3(block_size), 0, 0,
                               lvl.fine_rows, lvl.A.d_row_ptr, lvl.A.d_col_idx, lvl.A.d_values_f16,
                               lvl.d_b_f16, lvl.d_x_f16, lvl.d_res_f16);

            hipLaunchKernelGGL(spmv_csr_fp16_kernel, dim3(grid_c), dim3(block_size), 0, 0,
                               lvl.coarse_rows, lvl.R.d_row_ptr, lvl.R.d_col_idx, lvl.R.d_values_f16,
                               lvl.d_res_f16, solver->levels[l + 1].d_b_f16);
        }

        // Coarsest solve
        int last = num_levels - 1;
        auto& coarsest = solver->levels[last];
        int c_size = coarsest.fine_rows;

        if (solver->coarsest_size == c_size && solver->d_coarse_L != nullptr && c_size <= 128) {
            hipLaunchKernelGGL(coarse_triangular_solve_f16_kernel, dim3(1), dim3(32), 0, 0,
                               c_size, solver->d_coarse_L, coarsest.d_b_f16, coarsest.d_x_f16);
        } else if (solver->coarsest_size == c_size && !solver->h_coarse_L.empty() && (int)solver->h_coarse_L.size() == c_size * c_size) {
            std::vector<__half> h_r_f16(c_size);
            HIP_CHECK(hipMemcpy(h_r_f16.data(), coarsest.d_b_f16, c_size * sizeof(__half), hipMemcpyDeviceToHost));
            for (int i = 0; i < c_size; ++i) solver->h_coarse_r[i] = (double)__half2float(h_r_f16[i]);
            solve_coarse_cholesky_internal(solver, solver->h_coarse_r.data(), solver->h_coarse_x.data());
            std::vector<__half> h_x_f16(c_size);
            for (int i = 0; i < c_size; ++i) h_x_f16[i] = __float2half((float)solver->h_coarse_x[i]);
            HIP_CHECK(hipMemcpy(coarsest.d_x_f16, h_x_f16.data(), c_size * sizeof(__half), hipMemcpyHostToDevice));
        } else {
            int grid_last = std::min(std::max((c_size * 4 + block_size - 1) / block_size, 60), 480);
            for (int it = 0; it < c_sweeps; ++it) {
                hipLaunchKernelGGL(jacobi_smooth_csr_fp16_kernel, dim3(grid_last), dim3(block_size), 0, 0,
                                   coarsest.fine_rows, coarsest.A.d_row_ptr, coarsest.A.d_col_idx, coarsest.A.d_values_f16,
                                   coarsest.d_inv_diag_f16, coarsest.d_b_f16, (it == 0 ? (const __half*)nullptr : coarsest.d_x_f16),
                                   coarsest.d_x_temp_f16, 0.67f);
                std::swap(coarsest.d_x_f16, coarsest.d_x_temp_f16);
            }
        }

        // Upward sweep
        for (int l = num_levels - 2; l >= 0; --l) {
            auto& lvl = solver->levels[l];
            int grid_l = std::min(std::max((lvl.fine_rows * 4 + block_size - 1) / block_size, 60), 1920);
            const __half* e_coarse = solver->levels[l + 1].d_x_f16;

            int grid_p = std::min(std::max((lvl.fine_rows + block_size - 1) / block_size, 60), 480);
            hipLaunchKernelGGL(prolongation_add_fp16_kernel, dim3(grid_p), dim3(block_size), 0, 0,
                               lvl.fine_rows, lvl.P.d_row_ptr, lvl.P.d_col_idx, lvl.P.d_values_f16,
                               e_coarse, lvl.d_x_f16);

            hipLaunchKernelGGL(jacobi_smooth_csr_fp16_kernel, dim3(grid_l), dim3(block_size), 0, 0,
                               lvl.fine_rows, lvl.A.d_row_ptr, lvl.A.d_col_idx, lvl.A.d_values_f16,
                               lvl.d_inv_diag_f16, lvl.d_b_f16, lvl.d_x_f16, lvl.d_x_temp_f16, 0.67f);

            std::swap(lvl.d_x_f16, lvl.d_x_temp_f16);
        }

        // Cast back to FP64 in d_z
        hipLaunchKernelGGL(cast_half_to_double_kernel, dim3(grid_0_1d), dim3(block_size), 0, 0,
                           fine_size, solver->levels[0].d_x_f16, d_z);
        if (d_rho != nullptr) {
            hipMemsetAsync(d_rho, 0, sizeof(double));
            hipLaunchKernelGGL(vec_dot_kernel, dim3(60), dim3(block_size), 0, 0,
                               fine_size, d_z, solver->d_pcg_r, d_rho);
        }
        return 0;
    }

    return -3; // Unknown precision mode
}

WNFEA_EXPORT void hip_amg_set_coarse_sweeps(void* handle, int sweeps) {
    if (!handle || sweeps <= 0) return;
    HipAMGSolver* solver = (HipAMGSolver*)handle;
    solver->coarse_sweeps = sweeps;
}

WNFEA_EXPORT int hip_amg_apply_vcycle_device(
    void* handle,
    const double* d_r,
    double* d_z,
    int precision_mode
) {
    if (!handle || !d_r || !d_z) return -1;
    HipAMGSolver* solver = (HipAMGSolver*)handle;
    int fine_size = solver->fine_size;
    int block_size = 256;
    int grid_0_1d = std::min(std::max((fine_size + block_size - 1) / block_size, 60), 480);

    // 1. Initial equilibration: r_equil = D^{-1/2} * r -> stored in d_pcg_r
    hipLaunchKernelGGL(vec_pointwise_mult_kernel, dim3(grid_0_1d), dim3(block_size), 0, 0,
                       fine_size, d_r, solver->d_fine_inv_sqrt_d_f64, solver->d_pcg_r);

    // 2. Pure V-Cycle on equilibrated system
    int err = hip_amg_apply_pure_vcycle_device(solver, solver->d_pcg_r, solver->d_pcg_z, precision_mode);
    if (err != 0) return err;

    // 3. Post-equilibrate: z = D^{-1/2} * z_equil
    hipLaunchKernelGGL(vec_pointwise_mult_kernel, dim3(grid_0_1d), dim3(block_size), 0, 0,
                       fine_size, solver->d_pcg_z, solver->d_fine_inv_sqrt_d_f64, d_z);

    HIP_CHECK(hipDeviceSynchronize());
    return 0;
}

static inline double gpu_dot(int n, const double* d_x, const double* d_y, double* d_scalar, int grid, int block) {
    hipMemsetAsync(d_scalar, 0, sizeof(double));
    hipLaunchKernelGGL(vec_dot_kernel, dim3(grid), dim3(block), 0, 0, n, d_x, d_y, d_scalar);
    double h_val = 0.0;
    hipMemcpy(&h_val, d_scalar, sizeof(double), hipMemcpyDeviceToHost);
    return h_val;
}

WNFEA_EXPORT int hip_amg_solve_pcg_device(
    void* handle,
    const double* d_b,
    double* d_u,
    double rtol,
    int max_iter,
    int precision_mode,
    int check_interval,
    int* iters_out,
    double* res_out
) {
    if (!handle || !d_b || !d_u) return -1;
    HipAMGSolver* solver = (HipAMGSolver*)handle;
    int n = solver->fine_size;
    int block_size = 256;
    int grid_1d = std::min(std::max((n + block_size - 1) / block_size, 60), 480);
    int grid_v4 = std::min(std::max((n * 4 + block_size - 1) / block_size, 60), 1920);
    int grid_dot = 60;
    if (check_interval <= 0) check_interval = 1;

    double* d_r = solver->d_pcg_r;
    double* d_p = solver->d_pcg_p;
    double* d_q = solver->d_pcg_q;
    double* d_z = solver->d_pcg_z;
    double* d_scalar = solver->d_pcg_scalar;
    double* d_rho = solver->d_pcg_rho;
    double* d_rho_prev = solver->d_pcg_rho_prev;
    double* d_gamma = solver->d_pcg_gamma;
    double* d_r_sq = solver->d_pcg_r_sq;

    // 1. Initial equilibration: r_0 = \bar{b} = D^{-1/2} b
    hipLaunchKernelGGL(vec_pointwise_mult_kernel, dim3(grid_1d), dim3(block_size), 0, 0,
                       n, d_b, solver->d_fine_inv_sqrt_d_f64, d_r);

    // Initial solution: \bar{u} = 0
    HIP_CHECK(hipMemsetAsync(d_u, 0, n * sizeof(double)));

    // Norm of \bar{b}
    double r0_sq = gpu_dot(n, d_r, d_r, d_scalar, grid_dot, block_size);
    double r0 = std::sqrt(r0_sq);

    if (r0 <= 1e-30) {
        if (iters_out) *iters_out = 0;
        if (res_out) *res_out = 0.0;
        return 0;
    }

    double tol_sq = (rtol * rtol) * r0_sq;
    int iters = 0;

    // If FP32 V-Cycle, prepare initial d_b_f32 from d_r
    if (precision_mode == 1 && solver->levels[0].d_b_f32) {
        hipLaunchKernelGGL(cast_double_to_float_kernel, dim3(grid_1d), dim3(block_size), 0, 0,
                           n, d_r, solver->levels[0].d_b_f32);
    }

    for (iters = 0; iters < max_iter; ++iters) {
        // Step 1: Preconditioner + Fused Dot Product (z = M^{-1} r, rho = r^T z)
        // Pass d_r = nullptr on iters > 0 in FP32 because d_b_f32 is already populated by previous vec_pcg_update
        const double* r_in = (iters == 0 || precision_mode != 1) ? d_r : nullptr;
        int err = hip_amg_apply_pure_vcycle_device(solver, r_in, d_z, precision_mode, d_rho);
        if (err != 0) return err;

        // Step 2: Update direction: p = z + beta * p (computes beta = rho / rho_prev directly on device)
        hipLaunchKernelGGL(vec_update_p_device_kernel, dim3(grid_1d), dim3(block_size), 0, 0,
                           n, d_z, d_p, d_rho, d_rho_prev, iters);

        // Step 3: Fused SpMV and Inner Product: q = A * p AND gamma = p^T q in ONE single kernel
        hipMemsetAsync(d_gamma, 0, sizeof(double));
        hipLaunchKernelGGL(spmv_csr_dot_kernel, dim3(grid_v4), dim3(block_size), 0, 0,
                           n, solver->levels[0].A.d_row_ptr, solver->levels[0].A.d_col_idx,
                           solver->levels[0].A.d_values_f64, d_p, d_q, d_gamma);

        // Step 4: Fused Vector Update: u += alpha*p, r -= alpha*q, ||r||^2, *d_rho_prev = *d_rho,
        // and directly write d_b_f32 = (float)r (zero-overhead stream store for next V-cycle!)
        hipMemsetAsync(d_r_sq, 0, sizeof(double));
        float* d_b_f32_next = (precision_mode == 1) ? solver->levels[0].d_b_f32 : nullptr;
        hipLaunchKernelGGL(vec_pcg_update_device_kernel, dim3(grid_1d), dim3(block_size), 0, 0,
                           n, d_p, d_q, d_u, d_r, d_rho, d_gamma, d_r_sq, d_rho_prev, d_b_f32_next);

        // Step 5: Periodic host-device sync for convergence check
        if ((iters + 1) % check_interval == 0 || iters == max_iter - 1) {
            double r_sq = 0.0;
            HIP_CHECK(hipMemcpy(&r_sq, d_r_sq, sizeof(double), hipMemcpyDeviceToHost));

            if (r_sq <= tol_sq) {
                iters++;
                if (res_out) *res_out = std::sqrt(r_sq) / r0;
                break;
            }
        }
    }

    if (iters_out) *iters_out = iters;
    if (res_out && iters >= max_iter) {
        double r_sq = gpu_dot(n, d_r, d_r, d_scalar, grid_dot, block_size);
        *res_out = std::sqrt(r_sq) / r0;
    }

    // Post-equilibrate solution: u = D^{-1/2} * \bar{u}
    hipLaunchKernelGGL(vec_pointwise_mult_kernel, dim3(grid_1d), dim3(block_size), 0, 0,
                       n, d_u, solver->d_fine_inv_sqrt_d_f64, d_u);

    HIP_CHECK(hipDeviceSynchronize());
    return 0;
}

WNFEA_EXPORT int hip_amg_solve_pcg(
    void* handle,
    const double* h_b,
    double* h_u,
    double rtol,
    int max_iter,
    int precision_mode,
    int check_interval,
    int* iters_out,
    double* res_out
) {
    if (!handle || !h_b || !h_u) return -1;
    HipAMGSolver* solver = (HipAMGSolver*)handle;
    int fine_size = solver->fine_size;

    // Copy RHS to GPU work buffer
    HIP_CHECK(hipMemcpy(solver->d_work_in_f64, h_b, fine_size * sizeof(double), hipMemcpyHostToDevice));

    // Execute 100% GPU resident PCG solve
    int err = hip_amg_solve_pcg_device(handle, solver->d_work_in_f64, solver->d_pcg_u,
                                       rtol, max_iter, precision_mode, check_interval, iters_out, res_out);
    if (err != 0) return err;

    // Copy final converged solution back to host
    HIP_CHECK(hipMemcpy(h_u, solver->d_pcg_u, fine_size * sizeof(double), hipMemcpyDeviceToHost));
    return 0;
}

WNFEA_EXPORT int hip_amg_apply_vcycle(
    void* handle,
    const double* h_r,
    double* h_z,
    int precision_mode
) {
    if (!handle || !h_r || !h_z) return -1;
    HipAMGSolver* solver = (HipAMGSolver*)handle;
    int fine_size = solver->fine_size;

    // Copy input residual to device
    HIP_CHECK(hipMemcpy(solver->d_work_in_f64, h_r, fine_size * sizeof(double), hipMemcpyHostToDevice));

    // Execute V-Cycle directly in GPU VRAM
    int err = hip_amg_apply_vcycle_device(handle, solver->d_work_in_f64, solver->d_work_out_f64, precision_mode);
    if (err != 0) return err;

    // Copy solution vector back to host
    HIP_CHECK(hipMemcpy(h_z, solver->d_work_out_f64, fine_size * sizeof(double), hipMemcpyDeviceToHost));
    return 0;
}

WNFEA_EXPORT void hip_amg_destroy(void* handle) {
    if (!handle) return;
    HipAMGSolver* solver = (HipAMGSolver*)handle;
    solver->free();
    delete solver;
}

// ============================================================================
// Native GPU Sparse Matrix Assembly for 3D Beam Elements
// ============================================================================

__device__ inline void mult_Rt_M_R(const double R[3][3], const double M[3][3], double out[3][3]) {
    double temp[3][3];
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            temp[i][j] = R[0][i] * M[0][j] + R[1][i] * M[1][j] + R[2][i] * M[2][j];
        }
    }
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            out[i][j] = temp[i][0] * R[0][j] + temp[i][1] * R[1][j] + temp[i][2] * R[2][j];
        }
    }
}

__launch_bounds__(256, 2)
__global__ void beam3d_assemble_stiffness_csr_kernel(
    int num_elements,
    const double* __restrict__ nodes,
    const int*    __restrict__ elements,
    const double* __restrict__ props,
    const int*    __restrict__ csr_row_ptr,
    const int*    __restrict__ csr_col_idx,
    double*       __restrict__ csr_values,
    double*       __restrict__ out_elem_vals,
    int direct_scatter
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int e = tid; e < num_elements; e += stride) {
        int n1 = elements[e * 2];
        int n2 = elements[e * 2 + 1];

        double x1 = nodes[n1 * 3], y1 = nodes[n1 * 3 + 1], z1 = nodes[n1 * 3 + 2];
        double x2 = nodes[n2 * 3], y2 = nodes[n2 * 3 + 1], z2 = nodes[n2 * 3 + 2];

        double dx = x2 - x1;
        double dy = y2 - y1;
        double dz = z2 - z1;
        double L = sqrt(dx * dx + dy * dy + dz * dz);
        if (L < 1e-12) continue;

        double E  = props[e * 6 + 0];
        double G  = props[e * 6 + 1];
        double A  = props[e * 6 + 2];
        double Iy = props[e * 6 + 3];
        double Iz = props[e * 6 + 4];
        double J  = props[e * 6 + 5];

        double lx[3] = { dx / L, dy / L, dz / L };
        double ly[3] = { 0.0, 0.0, 0.0 };
        double lz[3] = { 0.0, 0.0, 0.0 };

        if (fabs(fabs(dx) - L) < 1e-9) {
            lx[0] = (dx > 0) ? 1.0 : -1.0; lx[1] = 0.0; lx[2] = 0.0;
            ly[0] = 0.0; ly[1] = 1.0; ly[2] = 0.0;
            lz[0] = 0.0; lz[1] = 0.0; lz[2] = 1.0;
        } else if (fabs(fabs(dy) - L) < 1e-9) {
            lx[0] = 0.0; lx[1] = (dy > 0) ? 1.0 : -1.0; lx[2] = 0.0;
            ly[0] = 1.0; ly[1] = 0.0; ly[2] = 0.0;
            lz[0] = 0.0; lz[1] = 0.0; lz[2] = 1.0;
        } else if (fabs(fabs(dz) - L) < 1e-9) {
            lx[0] = 0.0; lx[1] = 0.0; lx[2] = (dz > 0) ? 1.0 : -1.0;
            ly[0] = 1.0; ly[1] = 0.0; ly[2] = 0.0;
            lz[0] = 0.0; lz[1] = 1.0; lz[2] = 0.0;
        } else {
            ly[0] = -lx[1];
            ly[1] = lx[0];
            ly[2] = 0.0;
            double n_ly = sqrt(ly[0] * ly[0] + ly[1] * ly[1] + ly[2] * ly[2]);
            if (n_ly < 1e-9) {
                ly[0] = lx[2];
                ly[1] = 0.0;
                ly[2] = -lx[0];
                n_ly = sqrt(ly[0] * ly[0] + ly[1] * ly[1] + ly[2] * ly[2]);
            }
            double inv_nly = 1.0 / n_ly;
            ly[0] *= inv_nly; ly[1] *= inv_nly; ly[2] *= inv_nly;

            lz[0] = lx[1] * ly[2] - lx[2] * ly[1];
            lz[1] = lx[2] * ly[0] - lx[0] * ly[2];
            lz[2] = lx[0] * ly[1] - lx[1] * ly[0];
            double n_lz = sqrt(lz[0] * lz[0] + lz[1] * lz[1] + lz[2] * lz[2]);
            if (n_lz > 1e-12) {
                double inv_nlz = 1.0 / n_lz;
                lz[0] *= inv_nlz; lz[1] *= inv_nlz; lz[2] *= inv_nlz;
            }
        }

        double R[3][3] = {
            { lx[0], lx[1], lx[2] },
            { ly[0], ly[1], ly[2] },
            { lz[0], lz[1], lz[2] }
        };

        double inv_L = 1.0 / L;
        double inv_L2 = inv_L * inv_L;
        double inv_L3 = inv_L2 * inv_L;

        double ax  = E * A * inv_L;
        double tor = G * J * inv_L;
        double bz1 = 12.0 * E * Iz * inv_L3;
        double bz2 =  6.0 * E * Iz * inv_L2;
        double bz3 =  4.0 * E * Iz * inv_L;
        double bz4 =  2.0 * E * Iz * inv_L;

        double by1 = 12.0 * E * Iy * inv_L3;
        double by2 =  6.0 * E * Iy * inv_L2;
        double by3 =  4.0 * E * Iy * inv_L;
        double by4 =  2.0 * E * Iy * inv_L;

        double M00[3][3] = { { ax, 0.0, 0.0 }, { 0.0, bz1, 0.0 }, { 0.0, 0.0, by1 } };
        double M01[3][3] = { { 0.0, 0.0, 0.0 }, { 0.0, 0.0, bz2 }, { 0.0, -by2, 0.0 } };
        double M02[3][3] = { { -ax, 0.0, 0.0 }, { 0.0, -bz1, 0.0 }, { 0.0, 0.0, -by1 } };
        double M03[3][3] = { { 0.0, 0.0, 0.0 }, { 0.0, 0.0, bz2 }, { 0.0, -by2, 0.0 } };

        double M11[3][3] = { { tor, 0.0, 0.0 }, { 0.0, by3, 0.0 }, { 0.0, 0.0, bz3 } };
        double M12[3][3] = { { 0.0, 0.0, 0.0 }, { 0.0, 0.0, by2 }, { 0.0, -bz2, 0.0 } };
        double M13[3][3] = { { -tor, 0.0, 0.0 }, { 0.0, by4, 0.0 }, { 0.0, 0.0, bz4 } };

        double M22[3][3] = { { ax, 0.0, 0.0 }, { 0.0, bz1, 0.0 }, { 0.0, 0.0, by1 } };
        double M23[3][3] = { { 0.0, 0.0, 0.0 }, { 0.0, 0.0, -bz2 }, { 0.0, by2, 0.0 } };
        double M33[3][3] = { { tor, 0.0, 0.0 }, { 0.0, by3, 0.0 }, { 0.0, 0.0, bz3 } };

        double Kg[12][12];
        double B00[3][3], B01[3][3], B02[3][3], B03[3][3];
        double B11[3][3], B12[3][3], B13[3][3];
        double B22[3][3], B23[3][3], B33[3][3];

        mult_Rt_M_R(R, M00, B00);
        mult_Rt_M_R(R, M01, B01);
        mult_Rt_M_R(R, M02, B02);
        mult_Rt_M_R(R, M03, B03);

        mult_Rt_M_R(R, M11, B11);
        mult_Rt_M_R(R, M12, B12);
        mult_Rt_M_R(R, M13, B13);

        mult_Rt_M_R(R, M22, B22);
        mult_Rt_M_R(R, M23, B23);
        mult_Rt_M_R(R, M33, B33);

        #pragma unroll
        for (int r = 0; r < 3; ++r) {
            #pragma unroll
            for (int c = 0; c < 3; ++c) {
                Kg[r + 0][c + 0] = B00[r][c];
                Kg[r + 0][c + 3] = B01[r][c];
                Kg[r + 0][c + 6] = B02[r][c];
                Kg[r + 0][c + 9] = B03[r][c];

                Kg[r + 3][c + 0] = B01[c][r];
                Kg[r + 3][c + 3] = B11[r][c];
                Kg[r + 3][c + 6] = B12[r][c];
                Kg[r + 3][c + 9] = B13[r][c];

                Kg[r + 6][c + 0] = B02[c][r];
                Kg[r + 6][c + 3] = B12[c][r];
                Kg[r + 6][c + 6] = B22[r][c];
                Kg[r + 6][c + 9] = B23[r][c];

                Kg[r + 9][c + 0] = B03[c][r];
                Kg[r + 9][c + 3] = B13[c][r];
                Kg[r + 9][c + 6] = B23[c][r];
                Kg[r + 9][c + 9] = B33[r][c];
            }
        }

        if (out_elem_vals != nullptr) {
            #pragma unroll
            for (int r = 0; r < 12; ++r) {
                #pragma unroll
                for (int c = 0; c < 12; ++c) {
                    out_elem_vals[e * 144 + r * 12 + c] = Kg[r][c];
                }
            }
        }

        if (direct_scatter && csr_row_ptr != nullptr && csr_col_idx != nullptr && csr_values != nullptr) {
            int elem_dofs[12];
            #pragma unroll
            for (int d = 0; d < 6; ++d) {
                elem_dofs[d]     = n1 * 6 + d;
                elem_dofs[6 + d] = n2 * 6 + d;
            }

            for (int r = 0; r < 12; ++r) {
                int row = elem_dofs[r];
                int row_start = csr_row_ptr[row];
                int row_end   = csr_row_ptr[row + 1];

                for (int c = 0; c < 12; ++c) {
                    int col = elem_dofs[c];
                    double val = Kg[r][c];
                    if (fabs(val) < 1e-30) continue;

                    int low = row_start;
                    int high = row_end - 1;
                    while (low <= high) {
                        int mid = (low + high) >> 1;
                        int mid_col = csr_col_idx[mid];
                        if (mid_col == col) {
                            atomicAddDouble(&csr_values[mid], val);
                            break;
                        } else if (mid_col < col) {
                            low = mid + 1;
                        } else {
                            high = mid - 1;
                        }
                    }
                }
            }
        }
    }
}

__global__ void apply_dirichlet_bc_rows_kernel(
    int num_fixed,
    const int* __restrict__ fixed_dofs,
    const int* __restrict__ csr_row_ptr,
    const int* __restrict__ csr_col_idx,
    double*    __restrict__ csr_values,
    double*    __restrict__ F
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < num_fixed; i += stride) {
        int dof = fixed_dofs[i];
        if (F != nullptr) {
            F[dof] = 0.0;
        }
        int start = csr_row_ptr[dof];
        int end   = csr_row_ptr[dof + 1];
        for (int k = start; k < end; ++k) {
            if (csr_col_idx[k] == dof) {
                csr_values[k] = 1.0;
            } else {
                csr_values[k] = 0.0;
            }
        }
    }
}

__global__ void apply_dirichlet_bc_cols_kernel(
    int n_dofs,
    const unsigned char* __restrict__ is_fixed_mask,
    const int* __restrict__ csr_row_ptr,
    const int* __restrict__ csr_col_idx,
    double*    __restrict__ csr_values
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_dofs) return;

    if (is_fixed_mask[r]) return; // Handled by row kernel

    int start = csr_row_ptr[r];
    int end   = csr_row_ptr[r + 1];
    for (int k = start; k < end; ++k) {
        int col = csr_col_idx[k];
        if (is_fixed_mask[col]) {
            csr_values[k] = 0.0;
        }
    }
}

WNFEA_EXPORT int hip_assemble_beam_system_csr_device(
    int num_nodes,
    int num_elements,
    const double* d_nodes,
    const int* d_elements,
    const double* d_props,
    int nnz,
    const int* d_csr_row_ptr,
    const int* d_csr_col_idx,
    double* d_csr_values,
    int num_fixed,
    const int* d_fixed_dofs,
    const unsigned char* d_is_fixed_mask,
    double* d_F,
    double* d_out_elem_vals,
    int direct_scatter
) {
    if (num_elements <= 0) return 0;
    int block_size = 256;
    int grid_size = std::min(std::max((num_elements + block_size - 1) / block_size, 60), 1920);

    if (direct_scatter && d_csr_values != nullptr) {
        HIP_CHECK(hipMemsetAsync(d_csr_values, 0, nnz * sizeof(double)));
    }

    hipLaunchKernelGGL(beam3d_assemble_stiffness_csr_kernel, dim3(grid_size), dim3(block_size), 0, 0,
                       num_elements, d_nodes, d_elements, d_props,
                       d_csr_row_ptr, d_csr_col_idx, d_csr_values, d_out_elem_vals, direct_scatter);

    if (direct_scatter && num_fixed > 0 && d_fixed_dofs != nullptr && d_is_fixed_mask != nullptr) {
        int grid_fixed = std::min(std::max((num_fixed + block_size - 1) / block_size, 1), 480);
        hipLaunchKernelGGL(apply_dirichlet_bc_rows_kernel, dim3(grid_fixed), dim3(block_size), 0, 0,
                           num_fixed, d_fixed_dofs, d_csr_row_ptr, d_csr_col_idx, d_csr_values, d_F);

        int n_dofs = num_nodes * 6;
        int grid_cols = std::min(std::max((n_dofs + block_size - 1) / block_size, 60), 1920);
        hipLaunchKernelGGL(apply_dirichlet_bc_cols_kernel, dim3(grid_cols), dim3(block_size), 0, 0,
                           n_dofs, d_is_fixed_mask, d_csr_row_ptr, d_csr_col_idx, d_csr_values);
    }

    HIP_CHECK(hipDeviceSynchronize());
    return 0;
}

WNFEA_EXPORT int hip_assemble_beam_system_csr(
    int num_nodes,
    int num_elements,
    const double* h_nodes,
    const int* h_elements,
    const double* h_props,
    int nnz,
    const int* h_csr_row_ptr,
    const int* h_csr_col_idx,
    double* h_csr_values,
    int num_fixed,
    const int* h_fixed_dofs,
    double* h_F,
    double* h_out_elem_vals,
    int direct_scatter
) {
    if (num_elements <= 0) return 0;
    int n_dofs = num_nodes * 6;

    double* d_nodes = nullptr;
    int* d_elements = nullptr;
    double* d_props = nullptr;
    int* d_csr_row_ptr = nullptr;
    int* d_csr_col_idx = nullptr;
    double* d_csr_values = nullptr;
    int* d_fixed_dofs = nullptr;
    unsigned char* d_is_fixed_mask = nullptr;
    double* d_F = nullptr;
    double* d_out_elem_vals = nullptr;

    HIP_CHECK(hipMalloc(&d_nodes, num_nodes * 3 * sizeof(double)));
    HIP_CHECK(hipMemcpy(d_nodes, h_nodes, num_nodes * 3 * sizeof(double), hipMemcpyHostToDevice));

    HIP_CHECK(hipMalloc(&d_elements, num_elements * 2 * sizeof(int)));
    HIP_CHECK(hipMemcpy(d_elements, h_elements, num_elements * 2 * sizeof(int), hipMemcpyHostToDevice));

    HIP_CHECK(hipMalloc(&d_props, num_elements * 6 * sizeof(double)));
    HIP_CHECK(hipMemcpy(d_props, h_props, num_elements * 6 * sizeof(double), hipMemcpyHostToDevice));

    if (direct_scatter && h_csr_row_ptr && h_csr_col_idx && h_csr_values) {
        HIP_CHECK(hipMalloc(&d_csr_row_ptr, (n_dofs + 1) * sizeof(int)));
        HIP_CHECK(hipMemcpy(d_csr_row_ptr, h_csr_row_ptr, (n_dofs + 1) * sizeof(int), hipMemcpyHostToDevice));

        HIP_CHECK(hipMalloc(&d_csr_col_idx, nnz * sizeof(int)));
        HIP_CHECK(hipMemcpy(d_csr_col_idx, h_csr_col_idx, nnz * sizeof(int), hipMemcpyHostToDevice));

        HIP_CHECK(hipMalloc(&d_csr_values, nnz * sizeof(double)));
    }

    if (h_out_elem_vals != nullptr) {
        HIP_CHECK(hipMalloc(&d_out_elem_vals, num_elements * 144 * sizeof(double)));
    }

    std::vector<unsigned char> h_mask;
    if (direct_scatter && num_fixed > 0 && h_fixed_dofs != nullptr) {
        HIP_CHECK(hipMalloc(&d_fixed_dofs, num_fixed * sizeof(int)));
        HIP_CHECK(hipMemcpy(d_fixed_dofs, h_fixed_dofs, num_fixed * sizeof(int), hipMemcpyHostToDevice));

        h_mask.assign(n_dofs, 0);
        for (int i = 0; i < num_fixed; ++i) {
            int dof = h_fixed_dofs[i];
            if (dof >= 0 && dof < n_dofs) h_mask[dof] = 1;
        }
        HIP_CHECK(hipMalloc(&d_is_fixed_mask, n_dofs * sizeof(unsigned char)));
        HIP_CHECK(hipMemcpy(d_is_fixed_mask, h_mask.data(), n_dofs * sizeof(unsigned char), hipMemcpyHostToDevice));

        if (h_F != nullptr) {
            HIP_CHECK(hipMalloc(&d_F, n_dofs * sizeof(double)));
            HIP_CHECK(hipMemcpy(d_F, h_F, n_dofs * sizeof(double), hipMemcpyHostToDevice));
        }
    }

    int err = hip_assemble_beam_system_csr_device(
        num_nodes, num_elements, d_nodes, d_elements, d_props,
        nnz, d_csr_row_ptr, d_csr_col_idx, d_csr_values,
        num_fixed, d_fixed_dofs, d_is_fixed_mask, d_F,
        d_out_elem_vals, direct_scatter
    );

    if (err == 0) {
        if (direct_scatter && h_csr_values && d_csr_values) {
            HIP_CHECK(hipMemcpy(h_csr_values, d_csr_values, nnz * sizeof(double), hipMemcpyDeviceToHost));
        }
        if (direct_scatter && h_F && d_F) {
            HIP_CHECK(hipMemcpy(h_F, d_F, n_dofs * sizeof(double), hipMemcpyDeviceToHost));
        }
        if (h_out_elem_vals && d_out_elem_vals) {
            HIP_CHECK(hipMemcpy(h_out_elem_vals, d_out_elem_vals, num_elements * 144 * sizeof(double), hipMemcpyDeviceToHost));
        }
    }

    if (d_nodes) hipFree(d_nodes);
    if (d_elements) hipFree(d_elements);
    if (d_props) hipFree(d_props);
    if (d_csr_row_ptr) hipFree(d_csr_row_ptr);
    if (d_csr_col_idx) hipFree(d_csr_col_idx);
    if (d_csr_values) hipFree(d_csr_values);
    if (d_fixed_dofs) hipFree(d_fixed_dofs);
    if (d_is_fixed_mask) hipFree(d_is_fixed_mask);
    if (d_F) hipFree(d_F);
    if (d_out_elem_vals) hipFree(d_out_elem_vals);

    return err;
}

// ============================================================================
// BSR 6x6 Exported C Interfaces
// ============================================================================

WNFEA_EXPORT int hip_spmv_bsr6x6_fp32(
    int num_nodes,
    int nnz_blocks,
    const int*   h_b_row_ptr,
    const int*   h_b_col_idx,
    const float* h_b_values,
    const float* h_x,
    float*       h_y,
    float alpha,
    float beta,
    int block_size,
    int grid_size
) {
    if (num_nodes <= 0 || nnz_blocks <= 0 || !h_b_row_ptr || !h_b_col_idx || !h_b_values || !h_x || !h_y) {
        return -1;
    }
    int total_dofs = num_nodes * 6;
    if (block_size <= 0) block_size = 256;
    if (grid_size <= 0) {
        int desired = (total_dofs + block_size - 1) / block_size;
        if (desired < 60) desired = 60;
        if (desired > 1920) desired = 1920;
        grid_size = desired;
    }

    int *d_b_row_ptr = nullptr, *d_b_col_idx = nullptr;
    float *d_b_values = nullptr, *d_x = nullptr, *d_y = nullptr;

    HIP_CHECK(hipMalloc(&d_b_row_ptr, (num_nodes + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_b_col_idx, nnz_blocks * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_b_values,  nnz_blocks * 36 * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_x,         total_dofs * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_y,         total_dofs * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_b_row_ptr, h_b_row_ptr, (num_nodes + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b_col_idx, h_b_col_idx, nnz_blocks * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b_values,  h_b_values,  nnz_blocks * 36 * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x,         h_x,         total_dofs * sizeof(float), hipMemcpyHostToDevice));
    if (beta != 0.0f) {
        HIP_CHECK(hipMemcpy(d_y, h_y, total_dofs * sizeof(float), hipMemcpyHostToDevice));
    }

    hipLaunchKernelGGL(spmv_bsr6x6_fp32_kernel, dim3(grid_size), dim3(block_size), 0, 0,
                       num_nodes, d_b_row_ptr, d_b_col_idx, d_b_values, d_x, d_y, alpha, beta);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_y, d_y, total_dofs * sizeof(float), hipMemcpyDeviceToHost));

    hipFree(d_b_row_ptr);
    hipFree(d_b_col_idx);
    hipFree(d_b_values);
    hipFree(d_x);
    hipFree(d_y);

    return 0;
}

WNFEA_EXPORT int hip_spmv_bsr6x6_fp64(
    int num_nodes,
    int nnz_blocks,
    const int*    h_b_row_ptr,
    const int*    h_b_col_idx,
    const double* h_b_values,
    const double* h_x,
    double*       h_y,
    double alpha,
    double beta,
    int block_size,
    int grid_size
) {
    if (num_nodes <= 0 || nnz_blocks <= 0 || !h_b_row_ptr || !h_b_col_idx || !h_b_values || !h_x || !h_y) {
        return -1;
    }
    int total_dofs = num_nodes * 6;
    if (block_size <= 0) block_size = 256;
    if (grid_size <= 0) {
        int desired = (total_dofs + block_size - 1) / block_size;
        if (desired < 60) desired = 60;
        if (desired > 1920) desired = 1920;
        grid_size = desired;
    }

    int *d_b_row_ptr = nullptr, *d_b_col_idx = nullptr;
    double *d_b_values = nullptr, *d_x = nullptr, *d_y = nullptr;

    HIP_CHECK(hipMalloc(&d_b_row_ptr, (num_nodes + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_b_col_idx, nnz_blocks * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_b_values,  nnz_blocks * 36 * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_x,         total_dofs * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_y,         total_dofs * sizeof(double)));

    HIP_CHECK(hipMemcpy(d_b_row_ptr, h_b_row_ptr, (num_nodes + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b_col_idx, h_b_col_idx, nnz_blocks * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b_values,  h_b_values,  nnz_blocks * 36 * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x,         h_x,         total_dofs * sizeof(double), hipMemcpyHostToDevice));
    if (beta != 0.0) {
        HIP_CHECK(hipMemcpy(d_y, h_y, total_dofs * sizeof(double), hipMemcpyHostToDevice));
    }

    hipLaunchKernelGGL(spmv_bsr6x6_fp64_kernel, dim3(grid_size), dim3(block_size), 0, 0,
                       num_nodes, d_b_row_ptr, d_b_col_idx, d_b_values, d_x, d_y, alpha, beta);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_y, d_y, total_dofs * sizeof(double), hipMemcpyDeviceToHost));

    hipFree(d_b_row_ptr);
    hipFree(d_b_col_idx);
    hipFree(d_b_values);
    hipFree(d_x);
    hipFree(d_y);

    return 0;
}

WNFEA_EXPORT int hip_spmv_bsr6x6_benchmark(
    int num_nodes,
    int nnz_blocks,
    const int*   h_b_row_ptr,
    const int*   h_b_col_idx,
    const float* h_b_values,
    int num_repeats,
    double* out_time_ms,
    double* out_bandwidth_gbs,
    double* out_gflops
) {
    if (num_nodes <= 0 || nnz_blocks <= 0 || num_repeats <= 0) return -1;
    int total_dofs = num_nodes * 6;
    int block_size = 256;
    int grid_size = std::min(std::max((total_dofs + block_size - 1) / block_size, 60), 1920);

    int *d_b_row_ptr = nullptr, *d_b_col_idx = nullptr;
    float *d_b_values = nullptr, *d_x = nullptr, *d_y = nullptr;

    HIP_CHECK(hipMalloc(&d_b_row_ptr, (num_nodes + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_b_col_idx, nnz_blocks * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_b_values,  nnz_blocks * 36 * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_x,         total_dofs * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_y,         total_dofs * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_b_row_ptr, h_b_row_ptr, (num_nodes + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b_col_idx, h_b_col_idx, nnz_blocks * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b_values,  h_b_values,  nnz_blocks * 36 * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_x, 1, total_dofs * sizeof(float)));
    HIP_CHECK(hipMemset(d_y, 0, total_dofs * sizeof(float)));

    // Warmup
    hipLaunchKernelGGL(spmv_bsr6x6_fp32_kernel, dim3(grid_size), dim3(block_size), 0, 0,
                       num_nodes, d_b_row_ptr, d_b_col_idx, d_b_values, d_x, d_y, 1.0f, 0.0f);
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start));
    for (int it = 0; it < num_repeats; ++it) {
        hipLaunchKernelGGL(spmv_bsr6x6_fp32_kernel, dim3(grid_size), dim3(block_size), 0, 0,
                           num_nodes, d_b_row_ptr, d_b_col_idx, d_b_values, d_x, d_y, 1.0f, 0.0f);
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    double avg_ms = (double)ms / num_repeats;

    if (out_time_ms) *out_time_ms = avg_ms;

    double bytes_per_pass = (double)nnz_blocks * 36.0 * 4.0
                          + (double)nnz_blocks * 4.0
                          + (double)(num_nodes + 1) * 4.0
                          + (double)total_dofs * 4.0 * 2.0;

    double bw_gbs = (bytes_per_pass / (avg_ms * 1e-3)) / 1e9;
    if (out_bandwidth_gbs) *out_bandwidth_gbs = bw_gbs;

    double flops = (double)nnz_blocks * 36.0 * 2.0;
    double gf = (flops / (avg_ms * 1e-3)) / 1e9;
    if (out_gflops) *out_gflops = gf;

    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipFree(d_b_row_ptr);
    hipFree(d_b_col_idx);
    hipFree(d_b_values);
    hipFree(d_x);
    hipFree(d_y);

    return 0;
}


// ============================================================================
// Matrix-Free C3D10 Continuum Solid 3-DOF Operator (Wave32 AMD RDNA 3 Tuning)
// Evaluates v = K @ u on-the-fly without global matrix assembly
// Supports pure 3-DOF per node, tri-precision (FP32/FP64) and symmetric Dirichlet BCs
// ============================================================================

__constant__ float C_GAUSS_XI_F32[4]   = { 0.1381966011250105f, 0.5854101966249685f, 0.1381966011250105f, 0.1381966011250105f };
__constant__ float C_GAUSS_ETA_F32[4]  = { 0.1381966011250105f, 0.1381966011250105f, 0.5854101966249685f, 0.1381966011250105f };
__constant__ float C_GAUSS_ZETA_F32[4] = { 0.1381966011250105f, 0.1381966011250105f, 0.1381966011250105f, 0.5854101966249685f };
static const float C_WEIGHT_F32 = 1.0f / 24.0f;

__device__ inline void get_node_dN_dxi_fp32(int node_idx, float xi, float eta, float zeta, float dN[3]) {
    const float L1 = 1.0f - xi - eta - zeta;
    const float L2 = xi;
    const float L3 = eta;
    const float L4 = zeta;

    float dL1 = 0.0f, dL2 = 0.0f, dL3 = 0.0f, dL4 = 0.0f;
    switch (node_idx) {
        case 0: dL1 = 4.0f * L1 - 1.0f; break;
        case 1: dL2 = 4.0f * L2 - 1.0f; break;
        case 2: dL3 = 4.0f * L3 - 1.0f; break;
        case 3: dL4 = 4.0f * L4 - 1.0f; break;
        case 4: dL1 = 4.0f * L2; dL2 = 4.0f * L1; break;
        case 5: dL2 = 4.0f * L3; dL3 = 4.0f * L2; break;
        case 6: dL1 = 4.0f * L3; dL3 = 4.0f * L1; break;
        case 7: dL1 = 4.0f * L4; dL4 = 4.0f * L1; break;
        case 8: dL2 = 4.0f * L4; dL4 = 4.0f * L2; break;
        case 9: dL3 = 4.0f * L4; dL4 = 4.0f * L3; break;
    }
    dN[0] = dL2 - dL1;
    dN[1] = dL3 - dL1;
    dN[2] = dL4 - dL1;
}

__launch_bounds__(32, 2)
__global__ void c3d10_matrix_free_3dof_fp64_kernel(
    const double*  __restrict__ nodes,          // [n_nodes * 3]
    const int*     __restrict__ solid_elements, // [n_solids * 10]
    const double*  __restrict__ props,          // [n_solids * 2] (E, nu)
    const double*  __restrict__ u,              // [n_nodes * 3]
    double*        __restrict__ v,              // [n_nodes * 3]
    const uint8_t* __restrict__ is_fixed_mask,  // [n_nodes * 3]
    int n_solids
) {
    __shared__ double s_coords[32][10][3];
    __shared__ double s_u[32][10][3];
    __shared__ double s_f[32][10][3];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int t = threadIdx.x;

    for (int e = tid; e < n_solids; e += stride) {
        const int* elem_nodes = &solid_elements[e * 10];
        const double E_mod = props[e * 2 + 0];
        const double nu    = props[e * 2 + 1];

        const double factor = E_mod / ((1.0 + nu) * (1.0 - 2.0 * nu));
        const double c11 = factor * (1.0 - nu);
        const double c12 = factor * nu;
        const double c44 = factor * 0.5 * (1.0 - 2.0 * nu);

        #pragma unroll
        for (int i = 0; i < 10; ++i) {
            const int nid = elem_nodes[i];
            s_coords[t][i][0] = nodes[nid * 3 + 0];
            s_coords[t][i][1] = nodes[nid * 3 + 1];
            s_coords[t][i][2] = nodes[nid * 3 + 2];

            if (is_fixed_mask) {
                s_u[t][i][0] = is_fixed_mask[nid * 3 + 0] ? 0.0 : u[nid * 3 + 0];
                s_u[t][i][1] = is_fixed_mask[nid * 3 + 1] ? 0.0 : u[nid * 3 + 1];
                s_u[t][i][2] = is_fixed_mask[nid * 3 + 2] ? 0.0 : u[nid * 3 + 2];
            } else {
                s_u[t][i][0] = u[nid * 3 + 0];
                s_u[t][i][1] = u[nid * 3 + 1];
                s_u[t][i][2] = u[nid * 3 + 2];
            }

            s_f[t][i][0] = 0.0;
            s_f[t][i][1] = 0.0;
            s_f[t][i][2] = 0.0;
        }

        #pragma nounroll
        for (int g = 0; g < 4; ++g) {
            const double xi   = C_GAUSS_XI[g];
            const double eta  = C_GAUSS_ETA[g];
            const double zeta = C_GAUSS_ZETA[g];

            double J[3][3] = { {0.0} };
            for (int i = 0; i < 10; ++i) {
                double dN[3];
                get_node_dN_dxi(i, xi, eta, zeta, dN);
                const auto xi_c = s_coords[t][i][0];
                const auto yi_c = s_coords[t][i][1];
                const auto zi_c = s_coords[t][i][2];

                J[0][0] += dN[0] * xi_c;
                J[0][1] += dN[0] * yi_c;
                J[0][2] += dN[0] * zi_c;

                J[1][0] += dN[1] * xi_c;
                J[1][1] += dN[1] * yi_c;
                J[1][2] += dN[1] * zi_c;

                J[2][0] += dN[2] * xi_c;
                J[2][1] += dN[2] * yi_c;
                J[2][2] += dN[2] * zi_c;
            }

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

            const double sig[6] = {
                c11 * eps[0] + c12 * eps[1] + c12 * eps[2],
                c12 * eps[0] + c11 * eps[1] + c12 * eps[2],
                c12 * eps[0] + c12 * eps[1] + c11 * eps[2],
                c44 * eps[3],
                c44 * eps[4],
                c44 * eps[5]
            };

            const double w_detJ = C_WEIGHT * detJ;

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

        for (int i = 0; i < 10; ++i) {
            const int nid = elem_nodes[i];
            atomicAddDouble(&v[nid * 3 + 0], s_f[t][i][0]);
            atomicAddDouble(&v[nid * 3 + 1], s_f[t][i][1]);
            atomicAddDouble(&v[nid * 3 + 2], s_f[t][i][2]);
        }
    }
}

__launch_bounds__(32, 2)
__global__ void c3d10_matrix_free_3dof_fp32_kernel(
    const float*   __restrict__ nodes,          // [n_nodes * 3]
    const int*     __restrict__ solid_elements, // [n_solids * 10]
    const float*   __restrict__ props,          // [n_solids * 2] (E, nu)
    const float*   __restrict__ u,              // [n_nodes * 3]
    float*         __restrict__ v,              // [n_nodes * 3]
    const uint8_t* __restrict__ is_fixed_mask,  // [n_nodes * 3]
    int n_solids
) {
    __shared__ float s_coords[32][10][3];
    __shared__ float s_u[32][10][3];
    __shared__ float s_f[32][10][3];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int t = threadIdx.x;

    for (int e = tid; e < n_solids; e += stride) {
        const int* elem_nodes = &solid_elements[e * 10];
        const float E_mod = props[e * 2 + 0];
        const float nu    = props[e * 2 + 1];

        const float factor = E_mod / ((1.0f + nu) * (1.0f - 2.0f * nu));
        const float c11 = factor * (1.0f - nu);
        const float c12 = factor * nu;
        const float c44 = factor * 0.5f * (1.0f - 2.0f * nu);

        #pragma unroll
        for (int i = 0; i < 10; ++i) {
            const int nid = elem_nodes[i];
            s_coords[t][i][0] = nodes[nid * 3 + 0];
            s_coords[t][i][1] = nodes[nid * 3 + 1];
            s_coords[t][i][2] = nodes[nid * 3 + 2];

            if (is_fixed_mask) {
                s_u[t][i][0] = is_fixed_mask[nid * 3 + 0] ? 0.0f : u[nid * 3 + 0];
                s_u[t][i][1] = is_fixed_mask[nid * 3 + 1] ? 0.0f : u[nid * 3 + 1];
                s_u[t][i][2] = is_fixed_mask[nid * 3 + 2] ? 0.0f : u[nid * 3 + 2];
            } else {
                s_u[t][i][0] = u[nid * 3 + 0];
                s_u[t][i][1] = u[nid * 3 + 1];
                s_u[t][i][2] = u[nid * 3 + 2];
            }

            s_f[t][i][0] = 0.0f;
            s_f[t][i][1] = 0.0f;
            s_f[t][i][2] = 0.0f;
        }

        #pragma nounroll
        for (int g = 0; g < 4; ++g) {
            const float xi   = C_GAUSS_XI_F32[g];
            const float eta  = C_GAUSS_ETA_F32[g];
            const float zeta = C_GAUSS_ZETA_F32[g];

            float J[3][3] = { {0.0f} };
            for (int i = 0; i < 10; ++i) {
                float dN[3];
                get_node_dN_dxi_fp32(i, xi, eta, zeta, dN);
                const auto xi_c = s_coords[t][i][0];
                const auto yi_c = s_coords[t][i][1];
                const auto zi_c = s_coords[t][i][2];

                J[0][0] += dN[0] * xi_c;
                J[0][1] += dN[0] * yi_c;
                J[0][2] += dN[0] * zi_c;

                J[1][0] += dN[1] * xi_c;
                J[1][1] += dN[1] * yi_c;
                J[1][2] += dN[1] * zi_c;

                J[2][0] += dN[2] * xi_c;
                J[2][1] += dN[2] * yi_c;
                J[2][2] += dN[2] * zi_c;
            }

            const float c00 = J[1][1] * J[2][2] - J[1][2] * J[2][1];
            const float c01 = J[1][2] * J[2][0] - J[1][0] * J[2][2];
            const float c02 = J[1][0] * J[2][1] - J[1][1] * J[2][0];

            const float detJ = J[0][0] * c00 + J[0][1] * c01 + J[0][2] * c02;
            if (fabsf(detJ) < 1e-15f) continue;
            const float inv_detJ = 1.0f / detJ;

            const float invJ[3][3] = {
                { c00 * inv_detJ, (J[0][2]*J[2][1] - J[0][1]*J[2][2]) * inv_detJ, (J[0][1]*J[1][2] - J[0][2]*J[1][1]) * inv_detJ },
                { c01 * inv_detJ, (J[0][0]*J[2][2] - J[0][2]*J[2][0]) * inv_detJ, (J[0][2]*J[1][0] - J[0][0]*J[1][2]) * inv_detJ },
                { c02 * inv_detJ, (J[0][1]*J[2][0] - J[0][0]*J[2][1]) * inv_detJ, (J[0][0]*J[1][1] - J[0][1]*J[1][0]) * inv_detJ }
            };

            float eps[6] = { 0.0f };
            for (int i = 0; i < 10; ++i) {
                float dN[3];
                get_node_dN_dxi_fp32(i, xi, eta, zeta, dN);
                const float dNx = invJ[0][0] * dN[0] + invJ[0][1] * dN[1] + invJ[0][2] * dN[2];
                const float dNy = invJ[1][0] * dN[0] + invJ[1][1] * dN[1] + invJ[1][2] * dN[2];
                const float dNz = invJ[2][0] * dN[0] + invJ[2][1] * dN[1] + invJ[2][2] * dN[2];

                const float ux = s_u[t][i][0];
                const float uy = s_u[t][i][1];
                const float uz = s_u[t][i][2];

                eps[0] += dNx * ux;
                eps[1] += dNy * uy;
                eps[2] += dNz * uz;
                eps[3] += dNy * ux + dNx * uy;
                eps[4] += dNz * uy + dNy * uz;
                eps[5] += dNz * ux + dNx * uz;
            }

            const float sig[6] = {
                c11 * eps[0] + c12 * eps[1] + c12 * eps[2],
                c12 * eps[0] + c11 * eps[1] + c12 * eps[2],
                c12 * eps[0] + c12 * eps[1] + c11 * eps[2],
                c44 * eps[3],
                c44 * eps[4],
                c44 * eps[5]
            };

            const float w_detJ = C_WEIGHT_F32 * detJ;

            for (int i = 0; i < 10; ++i) {
                float dN[3];
                get_node_dN_dxi_fp32(i, xi, eta, zeta, dN);
                const float dNx = invJ[0][0] * dN[0] + invJ[0][1] * dN[1] + invJ[0][2] * dN[2];
                const float dNy = invJ[1][0] * dN[0] + invJ[1][1] * dN[1] + invJ[1][2] * dN[2];
                const float dNz = invJ[2][0] * dN[0] + invJ[2][1] * dN[1] + invJ[2][2] * dN[2];

                s_f[t][i][0] += w_detJ * (dNx * sig[0] + dNy * sig[3] + dNz * sig[5]);
                s_f[t][i][1] += w_detJ * (dNy * sig[1] + dNx * sig[3] + dNz * sig[4]);
                s_f[t][i][2] += w_detJ * (dNz * sig[2] + dNy * sig[4] + dNx * sig[5]);
            }
        }

        for (int i = 0; i < 10; ++i) {
            const int nid = elem_nodes[i];
            atomicAddFloat(&v[nid * 3 + 0], s_f[t][i][0]);
            atomicAddFloat(&v[nid * 3 + 1], s_f[t][i][1]);
            atomicAddFloat(&v[nid * 3 + 2], s_f[t][i][2]);
        }
    }
}

__global__ void c3d10_enforce_fixed_bcs_fp64_kernel(
    const double* __restrict__ u,
    double*       __restrict__ v,
    const int*    __restrict__ fixed_dofs,
    int n_fixed_dofs
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_fixed_dofs) {
        int dof = fixed_dofs[tid];
        v[dof] = u[dof];
    }
}

__global__ void c3d10_enforce_fixed_bcs_fp32_kernel(
    const float* __restrict__ u,
    float*       __restrict__ v,
    const int*   __restrict__ fixed_dofs,
    int n_fixed_dofs
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_fixed_dofs) {
        int dof = fixed_dofs[tid];
        v[dof] = u[dof];
    }
}

WNFEA_EXPORT int hip_c3d10_matrix_free_matvec_fp64(
    const double* h_nodes,
    const int*    h_elements,
    const double* h_props,
    const double* h_u,
    double*       h_v,
    const int*    h_fixed_dofs,
    int n_fixed_dofs,
    int n_nodes,
    int n_solids
) {
    if (!h_nodes || !h_elements || !h_props || !h_u || !h_v || n_nodes <= 0 || n_solids <= 0) return -1;
    int total_dofs = n_nodes * 3;

    double *d_nodes = nullptr, *d_props = nullptr, *d_u = nullptr, *d_v = nullptr;
    int *d_elements = nullptr, *d_fixed_dofs = nullptr;
    uint8_t* d_is_fixed_mask = nullptr;

    HIP_CHECK(hipMalloc(&d_nodes,    n_nodes * 3 * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_elements, n_solids * 10 * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_props,    n_solids * 2 * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_u,        total_dofs * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_v,        total_dofs * sizeof(double)));

    HIP_CHECK(hipMemcpy(d_nodes,    h_nodes,    n_nodes * 3 * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_elements, h_elements, n_solids * 10 * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_props,    h_props,    n_solids * 2 * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_u,        h_u,        total_dofs * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_v, 0,                 total_dofs * sizeof(double)));

    if (n_fixed_dofs > 0 && h_fixed_dofs) {
        std::vector<uint8_t> h_mask(total_dofs, 0);
        for (int i = 0; i < n_fixed_dofs; ++i) {
            int d = h_fixed_dofs[i];
            if (d >= 0 && d < total_dofs) h_mask[d] = 1;
        }
        HIP_CHECK(hipMalloc(&d_is_fixed_mask, total_dofs * sizeof(uint8_t)));
        HIP_CHECK(hipMemcpy(d_is_fixed_mask, h_mask.data(), total_dofs * sizeof(uint8_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMalloc(&d_fixed_dofs, n_fixed_dofs * sizeof(int)));
        HIP_CHECK(hipMemcpy(d_fixed_dofs, h_fixed_dofs, n_fixed_dofs * sizeof(int), hipMemcpyHostToDevice));
    }

    int block_size = 32;
    int grid_size = std::min(std::max((n_solids + block_size - 1) / block_size, 60), 1920);

    hipLaunchKernelGGL(c3d10_matrix_free_3dof_fp64_kernel, dim3(grid_size), dim3(block_size), 0, 0,
                       d_nodes, d_elements, d_props, d_u, d_v, d_is_fixed_mask, n_solids);

    if (n_fixed_dofs > 0 && d_fixed_dofs) {
        int bc_threads = 128;
        int bc_blocks = (n_fixed_dofs + bc_threads - 1) / bc_threads;
        hipLaunchKernelGGL(c3d10_enforce_fixed_bcs_fp64_kernel, dim3(bc_blocks), dim3(bc_threads), 0, 0,
                           d_u, d_v, d_fixed_dofs, n_fixed_dofs);
    }

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_v, d_v, total_dofs * sizeof(double), hipMemcpyDeviceToHost));

    hipFree(d_nodes);
    hipFree(d_elements);
    hipFree(d_props);
    hipFree(d_u);
    hipFree(d_v);
    if (d_is_fixed_mask) hipFree(d_is_fixed_mask);
    if (d_fixed_dofs) hipFree(d_fixed_dofs);

    return 0;
}

WNFEA_EXPORT int hip_c3d10_matrix_free_matvec_fp32(
    const float*  h_nodes,
    const int*    h_elements,
    const float*  h_props,
    const float*  h_u,
    float*        h_v,
    const int*    h_fixed_dofs,
    int n_fixed_dofs,
    int n_nodes,
    int n_solids
) {
    if (!h_nodes || !h_elements || !h_props || !h_u || !h_v || n_nodes <= 0 || n_solids <= 0) return -1;
    int total_dofs = n_nodes * 3;

    float *d_nodes = nullptr, *d_props = nullptr, *d_u = nullptr, *d_v = nullptr;
    int *d_elements = nullptr, *d_fixed_dofs = nullptr;
    uint8_t* d_is_fixed_mask = nullptr;

    HIP_CHECK(hipMalloc(&d_nodes,    n_nodes * 3 * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_elements, n_solids * 10 * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_props,    n_solids * 2 * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_u,        total_dofs * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_v,        total_dofs * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_nodes,    h_nodes,    n_nodes * 3 * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_elements, h_elements, n_solids * 10 * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_props,    h_props,    n_solids * 2 * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_u,        h_u,        total_dofs * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_v, 0,                 total_dofs * sizeof(float)));

    if (n_fixed_dofs > 0 && h_fixed_dofs) {
        std::vector<uint8_t> h_mask(total_dofs, 0);
        for (int i = 0; i < n_fixed_dofs; ++i) {
            int d = h_fixed_dofs[i];
            if (d >= 0 && d < total_dofs) h_mask[d] = 1;
        }
        HIP_CHECK(hipMalloc(&d_is_fixed_mask, total_dofs * sizeof(uint8_t)));
        HIP_CHECK(hipMemcpy(d_is_fixed_mask, h_mask.data(), total_dofs * sizeof(uint8_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMalloc(&d_fixed_dofs, n_fixed_dofs * sizeof(int)));
        HIP_CHECK(hipMemcpy(d_fixed_dofs, h_fixed_dofs, n_fixed_dofs * sizeof(int), hipMemcpyHostToDevice));
    }

    int block_size = 32;
    int grid_size = std::min(std::max((n_solids + block_size - 1) / block_size, 60), 1920);

    hipLaunchKernelGGL(c3d10_matrix_free_3dof_fp32_kernel, dim3(grid_size), dim3(block_size), 0, 0,
                       d_nodes, d_elements, d_props, d_u, d_v, d_is_fixed_mask, n_solids);

    if (n_fixed_dofs > 0 && d_fixed_dofs) {
        int bc_threads = 128;
        int bc_blocks = (n_fixed_dofs + bc_threads - 1) / bc_threads;
        hipLaunchKernelGGL(c3d10_enforce_fixed_bcs_fp32_kernel, dim3(bc_blocks), dim3(bc_threads), 0, 0,
                           d_u, d_v, d_fixed_dofs, n_fixed_dofs);
    }

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_v, d_v, total_dofs * sizeof(float), hipMemcpyDeviceToHost));

    hipFree(d_nodes);
    hipFree(d_elements);
    hipFree(d_props);
    hipFree(d_u);
    hipFree(d_v);
    if (d_is_fixed_mask) hipFree(d_is_fixed_mask);
    if (d_fixed_dofs) hipFree(d_fixed_dofs);

    return 0;
}

} // extern "C"
