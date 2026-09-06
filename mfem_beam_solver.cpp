#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
    // 1. Parse command-line options.
    const char *device_config = "cpu";
    const char *input_file = "K_F_input.bin";
    const char *output_file = "U_output.bin";

    OptionsParser args(argc, argv);
    args.AddOption(&device_config, "-d", "--device",
                   "Device configuration string, e.g. cpu, openmp, cuda, hip.");
    args.AddOption(&input_file, "-i", "--input",
                   "Input binary file containing stiffness matrix and force vector.");
    args.AddOption(&output_file, "-o", "--output",
                   "Output binary file for displacement vector.");
    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(cout);
        return 1;
    }

    // 2. Enable hardware devices.
    Device device(device_config);
    device.Print();

    // 3. Read the binary file containing K and F.
    ifstream in(input_file, ios::binary);
    if (!in.is_open())
    {
        cerr << "Error: Could not open input file " << input_file << endl;
        return 2;
    }

    int nrows = 0, ncols = 0, nnz = 0;
    in.read(reinterpret_cast<char*>(&nrows), sizeof(int));
    in.read(reinterpret_cast<char*>(&ncols), sizeof(int));
    in.read(reinterpret_cast<char*>(&nnz), sizeof(int));

    if (nrows <= 0 || ncols <= 0 || nnz <= 0)
    {
        cerr << "Error: Invalid matrix dimensions (nrows=" << nrows 
             << ", ncols=" << ncols << ", nnz=" << nnz << ")" << endl;
        return 3;
    }

    cout << "Reading sparse matrix of size " << nrows << " x " << ncols 
         << " with " << nnz << " nonzeros..." << endl;

    // Read CSR arrays: i, j, a
    vector<int> h_i(nrows + 1);
    vector<int> h_j(nnz);
    vector<real_t> h_a(nnz);

    in.read(reinterpret_cast<char*>(h_i.data()), (nrows + 1) * sizeof(int));
    in.read(reinterpret_cast<char*>(h_j.data()), nnz * sizeof(int));
    in.read(reinterpret_cast<char*>(h_a.data()), nnz * sizeof(real_t));

    // Read force vector F
    vector<real_t> h_F(nrows);
    in.read(reinterpret_cast<char*>(h_F.data()), nrows * sizeof(real_t));

    if (!in.good())
    {
        cerr << "Error: Failed to read complete matrix/vector data from " << input_file << endl;
        return 4;
    }
    in.close();

    // 4. Construct MFEM SparseMatrix and Vectors.
    // Allocate raw arrays and transfer ownership to SparseMatrix.
    int *i_ptr = new int[nrows + 1];
    int *j_ptr = new int[nnz];
    real_t *a_ptr = new real_t[nnz];

    std::copy(h_i.begin(), h_i.end(), i_ptr);
    std::copy(h_j.begin(), h_j.end(), j_ptr);
    std::copy(h_a.begin(), h_a.end(), a_ptr);

    // Reconstruct K
    SparseMatrix K(i_ptr, j_ptr, a_ptr, nrows, ncols);
    K.UseGPUSparse(false);

    // Reconstruct F and U
    Vector F(nrows);
    F.UseDevice(true);
    {
        real_t *F_data = F.HostWrite();
        std::copy(h_F.begin(), h_F.end(), F_data);
    }

    Vector U(nrows);
    U.UseDevice(true);
    U = 0.0; // Initial guess

    // 5. Set up the Conjugate Gradient solver.
    CGSolver cg;
    cg.SetRelTol(1e-12);
    cg.SetMaxIter(10000);
    cg.SetPrintLevel(1);
    cg.SetOperator(K);

#ifdef MFEM_USE_MPI
    // Hypre BoomerAMG GPU-Accelerated Preconditioner
    // Offloaded to AMD Radeon RX 7800 XT (gfx1101) via HIP
    HypreParMatrix *A_par = new HypreParMatrix(MPI_COMM_WORLD, nrows, ...);
    HypreBoomerAMG *amg = new HypreBoomerAMG(*A_par);

    // 1. Parallel Modified Independent Set (PMIS) coarsening
    amg->SetCoarseningType(8);

    // 2. Extended+i interpolation (GPU/SIMD optimized)
    amg->SetInterpolationType(14);

    // 3. Truncate deep graph hierarchy
    amg->SetMaxLevels(4);

    // 4. Aggressive weak connection dropping
    amg->SetMaxRowSum(0.9);

    cg.SetPreconditioner(*amg);
#else
    // Serial/Threaded HIP device fallback: Jacobi diagonal smoother
    DSmoother prec(K);
    cg.SetPreconditioner(prec);
#endif

    // 6. Solve the system
    cout << "Solving system of equations..." << endl;
    cg.Mult(F, U);
    cout << "System solved." << endl;

    // 7. Write U to the binary output file
    ofstream out(output_file, ios::binary);
    if (!out.is_open())
    {
        cerr << "Error: Could not open output file " << output_file << endl;
        return 5;
    }

    const real_t *U_data = U.HostRead();
    out.write(reinterpret_cast<const char*>(U_data), nrows * sizeof(real_t));
    out.close();

    cout << "Displacement vector written to " << output_file << endl;

    return 0;
}

#ifdef MFEM_USE_HIP
#include <hipblas/hipblas.h>

extern "C" {
hipblasStatus_t hipblasDgetrfBatched(hipblasHandle_t handle,
                                     const int       n,
                                     double* const   A[],
                                     const int       lda,
                                     int*            ipiv,
                                     int*            info,
                                     const int       batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDgetrsBatched(hipblasHandle_t          handle,
                                     const hipblasOperation_t trans,
                                     const int                n,
                                     const int                nrhs,
                                     double* const            A[],
                                     const int                lda,
                                     const int*               ipiv,
                                     double* const            B[],
                                     const int                ldb,
                                     int*                     info,
                                     const int                batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t hipblasDgetriBatched(hipblasHandle_t handle,
                                     const int       n,
                                     double* const   A[],
                                     const int       lda,
                                     int*            ipiv,
                                     double* const   C[],
                                     const int       ldc,
                                     int*            info,
                                     const int       batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;
}
}
#endif
