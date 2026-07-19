cmake_minimum_required(VERSION 3.12)
project(mfem_beam_solver CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the MFEM package in the build directory
find_package(MFEM REQUIRED PATHS "C:/Users/wsnun/Documents/Local_Programs/Open_Source_FEA/mfem-4.9/mfem-4.9/build_hip" "C:/Users/wsnun/Documents/Local_Programs/Open_Source_FEA/mfem-4.9/mfem-4.9/build")

add_executable(mfem_beam_solver mfem_beam_solver.cpp)

# Link against MFEM and ROCm solvers/blas libraries
target_link_libraries(mfem_beam_solver PRIVATE 
    mfem
    "C:/Program Files/AMD/ROCm/7.1/lib/rocblas.lib"
    "C:/Program Files/AMD/ROCm/7.1/lib/rocsolver.lib"
    "C:/Program Files/AMD/ROCm/7.1/lib/hipsolver.lib"
)

# Force linker to include the entire mfem.lib archive (prevents stripping of GPU kernels on Windows)
target_link_options(mfem_beam_solver PRIVATE "SHELL:-Xlinker /WHOLEARCHIVE:C:/Users/wsnun/Documents/Local_Programs/Open_Source_FEA/mfem-4.9/mfem-4.9/build_hip/mfem.lib")
