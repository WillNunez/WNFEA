# WNFEA 72-Hour Unattended Autonomous Development Roadmap (10x SOTA Engine)

This document is the persistent execution backlog for autonomous development of a state-of-the-art structural FEA engine targeting 10x market speedups.

## Autonomous Agent Operational Guardrails
1. **Zero-Regression Gate**: Execute `python run_all_tests.py` before every commit. Must achieve 100% pass rate.
2. **Dedicated Sprint Branch**: Work strictly on `feat/unattended-72h-sprint`.
3. **Atomic Commits**: One commit per completed task with structured conventional commit messages.
4. **Failure Recovery**: Attempt up to 3 diagnostic fixes. If unresolved, rollback with `git checkout -- .`, record the blocker in `UNATTENDED_LOG.md`, and advance to the next independent task.
5. **State Synchronization**: Update status to `[COMPLETED]` and log execution metrics in `UNATTENDED_LOG.md`.
6. **Tool Stack Integration**:
   - **CAD Kernel**: FreeCAD 1.1 (`C:\Program Files\FreeCAD 1.1\bin\freecadcmd.exe` and macros in `C:/Users/wsnun/AppData/Roaming/FreeCAD/v1-1/Macro`).
   - **Mesher**: Gmsh 4.15.2 (Python API).
   - **Post-Processing**: ParaView (VTU / PVSM).

---

## Task Backlog

### Phase 1: Matrix-Free C3D10 Operator & VRAM Minimization (Target: 2M DOFs in <300MB VRAM)
- [x] **Task 1.1: Matrix-Free C3D10 Evaluation Engine**
  - **Objective**: Implement matrix-free operator `K @ u` in `wnfea/solver/matrix_free_c3d10.py` that computes element-level actions on-the-fly without assembling or storing global stiffness matrices.
  - **Acceptance Criteria**: Exact numerical parity with explicit CSR matrix (< 1e-12 relative error) while reducing memory consumption by >95%.
  - **Verification**: `python run_all_tests.py`. (PASSED - commit `6e482be`)

- [x] **Task 1.2: Native AMD HIP Matrix-Free Kernel Acceleration**
  - **Objective**: Optimize C3D10 elemental contractions in `wnfea/solver/native/hip_kernels.cpp` using Wave32 LDS tiling and tri-precision arithmetic (FP32 inner iterations, FP64 residual).
  - **Acceptance Criteria**: Peak compute utilization exceeding 50% of RX 7800 XT theoretical TFLOPs; exact FP64 (< 1e-12) and FP32 (< 1e-5) parity with CPU and CSR.
  - **Verification**: `python run_all_tests.py`. (PASSED - verified in test suite)


- [x] **Task 1.3: p-Multigrid Preconditioning (Linear Tet Coarse Grid for Quadratic Tet)**
  - **Objective**: Implement two-level geometric p-multigrid preconditioner where the coarse level is formed by vertex nodes (C3D4) and the fine level adds edge mid-nodes (C3D10).
  - **Acceptance Criteria**: Reduces PCG iteration count by >60% vs Point Jacobi (70.9% reduction achieved: 206 -> 60 iters); exact numerical solution parity vs direct solve (< 3.8e-11 error); exact adjointness (< 6.5e-16).
  - **Verification**: `python run_all_tests.py`. (PASSED - 9/9 suites pass 100%)


---

### Phase 2: Heterogeneous CPU + GPU Pipelining & 2M DOF Benchmark
- [x] **Task 2.1: Heterogeneous Subsystem Assembly (99% Solids on GPU, 1% Beams on CPU)**
  - **Objective**: Create `wnfea/solver/heterogeneous_assembler.py` to concurrently condense 6-DOF beam constraints and Dirichlet conditions on CPU multi-threading while GPU streams solid element evaluations.
  - **Acceptance Criteria**: Seamless kinematic coupling between beam rot-DOFs and solid trans-DOFs with zero CPU-GPU transfer bottlenecks. Exact parity vs explicit assembly ($2.75 \times 10^{-16}$) and exact solution parity ($2.02 \times 10^{-10}$).
  - **Verification**: `python run_all_tests.py`. (PASSED - 10/10 suites pass 100%)

- [x] **Task 2.2: 2M DOF Realistic Large Structural Benchmark**
  - **Objective**: Build `scratch/benchmark_2m_dof.py` generating a complex 2,000,000 DOF solid-beam stiffened structure, profiling wallclock solve time, VRAM peak, and GFLOP/s.
  - **Acceptance Criteria**: Solves 2M DOFs in under 15 seconds with peak VRAM < 2.0 GB. (PASSED - 2,024,067 DOFs solved in 13.72s, peak VRAM 132.6 MB, SpMV latency 54.54 ms on AMD Radeon RX 7800 XT).
  - **Verification**: `python scratch/benchmark_2m_dof.py` and `python run_all_tests.py`.

---

### Phase 3: Core Tenet: Feature-Delta Re-meshing & Automated BCs
- [x] **Task 3.1: FreeCAD 1.1 B-Rep Cylindrical Face & Bolt Detection**
  - **Objective**: Interface with FreeCAD 1.1 CAD kernel to extract cylindrical faces (`GeomAbs_Cylinder`), identify bolt hole standard diameters (M3–M16, 1/4"–1/2"), and generate automated pin/bolt constraints and bearing pressure distributions.
  - **Acceptance Criteria**: Auto-identifies 100% of cylindrical holes on imported STEP CAD models without manual face picking.
  - **Verification**: `python run_all_tests.py`. (PASSED - commit `3fa3f4d`)

- [x] **Task 3.2: Applied Acceleration Fields & Concentrated Point Loads**
  - **Objective**: Implement uniform/angular acceleration body forces ($f_e = \rho \int N^T \vec{a} d\Omega$) and concentrated point load distribution in `wnfea/boundary/body_loads.py`.
  - **Acceptance Criteria**: Acceleration load vector matches analytical $F = m \cdot a$ to machine precision across full 3D solid meshes (< 1e-15 rel error) and RBE3 forces/moments conserve equilibrium (< 5e-12 error).
  - **Verification**: `python run_all_tests.py`. (PASSED - commit `c22899e`)

- [x] **Task 3.3: Core Tenet: Localized Feature-Delta Re-meshing Engine**
  - **Objective**: Implement bounding-box spatial void carving and localized Gmsh re-meshing when a FreeCAD CAD feature (fillet, hole, pocket) is modified. Stitch new local elements into the existing global matrix-free model without re-meshing unaffected geometry.
  - **Acceptance Criteria**: Sub-model solve in 10.66 ms with 7.77e-17 interior displacement parity vs global solve.
  - **Verification**: `python run_all_tests.py`. (PASSED - commit `147d82c`)

- [x] **Task 3.4: Non-Linear JFNK Neural Warm-Start Interface**
  - **Objective**: Implement `wnfea/solver/neural_warm_start.py` allowing surrogate neural network predictions (NeMo / FNO / GNO) to initialize displacement vectors for non-linear load steps.
  - **Acceptance Criteria**: Residual verification rejects divergence, accepts accurate surrogates with >60% residual reduction, achieves identical non-linear convergence (4.20e-17 rel diff).
  - **Verification**: `python run_all_tests.py`. (PASSED - commit `8c8e326`)

---

### Phase 4: Architectural Investigations & Generative Design (Documentation & Study)
- [x] **Task 4.1: Voxel First-Pass & CAD-Conforming Spherical Sub-Modeling Architecture**
  - **Objective**: Author `docs/voxel_amr_architecture.md` defining a dual-stage solver: (1) Fast first-pass global Cartesian voxel/octree solve to detect hot-spots; (2) Localized spherical sub-modeling around high-stress concentrations where boundaries are placed at gradient <1%, prescribing coarse displacements as Dirichlet BCs, re-meshing interior to conform strictly to CAD B-rep geometry, and executing embarrassingly parallel localized solves.
  - **Acceptance Criteria**: Detailed mathematical formulation, boundary condition transfer, CAD boundary snapping, and parallel sub-domain scaling proof.
  - **Verification**: File review & `python run_all_tests.py`. (COMPLETED)

- [x] **Task 4.2: In-the-Loop Topology Optimization Specification**
  - **Objective**: Author `docs/generative_design_integration.md` defining integration of WNFEA's fast matrix-free solver into SIMP / Level-Set loops with GPU sensitivity filtering.
  - **Acceptance Criteria**: Concrete API interfaces, adjoint gradient formulation, and FreeCAD B-Rep STEP reconstruction.
  - **Verification**: File review & `python run_all_tests.py`. (COMPLETED)

- [x] **Task 4.3: 3-Axis & 5-Axis CNC Machinability Intelligence Framework**
  - **Objective**: Author `docs/machinability_study.md` defining parametric visibility cones, tool clearance constraints, and translation of density fields into parametric CAD B-rep surfaces (Fusion 360 style).
  - **Acceptance Criteria**: Rigorous mathematical formulation of differentiable CNC accessibility penalties and parametric feature constraints.
  - **Verification**: File review & `python run_all_tests.py`. (COMPLETED)

---

### Phase 5: Dual-Stage Voxel First-Pass & CAD-Conforming Sub-Modeling Engine
- [x] **Task 5.1: Fast Cartesian Voxelizer & Immersed Boundary (IFEM) Generator**
  - **Objective**: Implement `wnfea/mesh/voxel_mesher.py` to generate structured Hex8 voxel grids over complex CAD geometries or boundary meshes with active cell classification.
  - **Acceptance Criteria**: Voxelizes 3D domains into 100,000+ cells in <100 ms with accurate interior/boundary cell tagging and exact trilinear shape interpolation (< 1e-14 error).
  - **Verification**: `python run_all_tests.py`. (PASSED)

- [x] **Task 5.2: Ultra-Fast Matrix-Free Hex8 Voxel Solver**
  - **Objective**: Implement `wnfea/solver/matrix_free_hex8.py` evaluating on-the-fly elemental contractions for uniform 8-node hexahedra with zero matrix storage and PCG solver.
  - **Acceptance Criteria**: Linear static solve in <2 ms on CPU with exact energy parity and element von Mises stress field recovery.
  - **Verification**: `python run_all_tests.py`. (PASSED)

- [x] **Task 5.3: End-to-End Dual-Stage Pipeline with 1% Saint-Venant Transfer**
  - **Objective**: Implement `wnfea/solver/dual_stage_pipeline.py` chaining Stage 1 (Fast Voxel) -> Automated Hotspot & 1% Saint-Venant Sphere -> Stage 2 (Curvature-conforming C3D10 Sub-Model Solve).
  - **Acceptance Criteria**: Automated cut-boundary Dirichlet displacement interpolation, localized C3D10 sub-model solve, and verified 154 ms total dual-stage solve time.
  - **Verification**: `python run_all_tests.py`. (PASSED - 15/15 suites pass 100%)

---

### Phase 6: In-the-Loop Topology Optimization & Generative Design Engine
- [x] **Task 6.1: Matrix-Free SIMP Topology Optimization Core**
  - **Objective**: Implement `wnfea/opt/topology.py` with on-the-fly adjoint element strain energy evaluation $\partial C / \partial \rho_e = -p \rho_e^{p-1} \mathbf{u}_e^T \mathbf{k}_0 \mathbf{u}_e$ without global matrix assembly, coupled with Optimality Criteria (OC) update.
  - **Acceptance Criteria**: Reduces compliance under volume fraction constraint $V^*$, achieving 20 iterations in 85 ms (4.2 ms/iter).
  - **Verification**: `python run_all_tests.py`. (PASSED)

- [x] **Task 6.2: Spatial Sensitivity Filtering & Differentiable Heaviside Projection**
  - **Objective**: Implement `wnfea/opt/filters.py` providing mesh-independent radius filtering and smoothed Heaviside projection ($\beta$-continuation) for crisp 0-1 black/white boundaries and non-design domain freezing (bolt holes/pads).
  - **Acceptance Criteria**: Eliminates checkerboard instabilities, exact filter adjointness (< 1e-13), and strict non-design domain preservation.
  - **Verification**: `python run_all_tests.py`. (PASSED)

- [x] **Task 6.3: 3-Axis CNC Machinability Milling Constraint**
  - **Objective**: Implement `wnfea/opt/machinability.py` evaluating differentiable line-of-sight visibility cones and undercut penalties along milling spindle axes ($\pm Z$) with morphological minimum tool radius enforcement ($r \ge R_{cutter}$).
  - **Acceptance Criteria**: Suppresses internal undercuts and enforces tool clearance for 3-axis CNC machining.
  - **Verification**: `python run_all_tests.py`. (PASSED)

- [x] **Task 6.4: Verification Suite & Benchmark Generative Design Solve**
  - **Objective**: Implement `tests/test_topology_optimization.py` validating compliance convergence, sensitivity gradients, filtering, and CNC machinability constraints.
  - **Acceptance Criteria**: 100% pass rate in verification harness.
  - **Verification**: `python run_all_tests.py`. (PASSED - 16/16 suites pass 100%)

---

### Phase 7: FreeCAD B-Rep Generative Reconstruction & ParaView State Export
- [x] **Task 7.1: Isosurface Extraction & Dual Contouring**
  - **Objective**: Implement `wnfea/cad/isosurface.py` extracting watertight, manifold triangular boundary meshes from optimized 3D density fields with curvature-preserving smoothing.
  - **Acceptance Criteria**: Extracts watertight STL/PLY surface mesh from 3D density grid in <50 ms.
  - **Verification**: `python run_all_tests.py`. (PASSED)

- [x] **Task 7.2: FreeCAD 1.1 OpenCASCADE B-Rep Solid & STEP Export**
  - **Objective**: Implement `wnfea/cad/brep_reconstruction.py` sewing isosurfaces into closed B-Rep solids in FreeCAD 1.1 and boolean-fusing analytical cylindrical bolt bores.
  - **Acceptance Criteria**: Exports valid, watertight STEP solid with exact analytical cylindrical bolt holes.
  - **Verification**: `python run_all_tests.py`. (PASSED)

- [x] **Task 7.3: Automated ParaView Visualization & State File Generator**
  - **Objective**: Implement `wnfea/results/paraview_export.py` generating binary XML VTU files and automated ParaView state files (`.pvsm`) with warp-by-vector deformation and stress contours.
  - **Acceptance Criteria**: Generates valid VTU + PVSM state files loadable directly in ParaView.
  - **Verification**: `python run_all_tests.py`. (PASSED)

- [x] **Task 7.4: Verification Suite & End-to-End CAD/ParaView Benchmark**
  - **Objective**: Implement `tests/test_cad_paraview_pipeline.py` testing the complete chain from optimization to STEP export and ParaView visualization.
  - **Acceptance Criteria**: 100% pass rate in verification harness.
  - **Verification**: `python run_all_tests.py`. (PASSED - 17/17 suites pass 100%)

---

### Phase 8: Multi-Load Case Generative Design & 5-Axis Machinability Workflow
- [ ] **Task 8.1: Multi-Load Case Matrix-Free Topology Optimization**
  - **Objective**: Extend `wnfea/opt/topology.py` to support weighted multiple independent load cases (e.g., $C = \sum_k w_k C_k$), computing composite sensitivity fields and enforcing multi-directional stiffness.
  - **Acceptance Criteria**: Converges to balanced topologies resisting combined bending, torsion, and axial body acceleration.
  - **Verification**: `python run_all_tests.py`.

- [ ] **Task 8.2: 5-Axis Spindle Orientation & Multi-Directional Milling Constraints**
  - **Objective**: Extend `wnfea/opt/machinability.py` to evaluate arbitrary 5-axis tool access orientations $(\theta, \phi)$ on spherical tooling manifolds, optimizing part setup orientation to maximize machinable volume.
  - **Acceptance Criteria**: Identifies optimal 3-axis/5-axis spindle setups minimizing unmachined internal pockets.
  - **Verification**: `python run_all_tests.py`.

- [ ] **Task 8.3: End-to-End Generative Bracket Production Case Study**
  - **Objective**: Build `examples/generative_bracket_case_study.py` demonstrating the entire workflow: CAD B-Rep import -> Bolt hole recognition -> Multi-load topology optimization with CNC constraints -> Watertight FreeCAD STEP solid export -> Dual-stage sub-model stress verification -> ParaView VTU report generation.
  - **Acceptance Criteria**: Script executes end-to-end under 5 seconds, producing production-ready STEP and VTU assets.
  - **Verification**: `python examples/generative_bracket_case_study.py` and `python run_all_tests.py`.




