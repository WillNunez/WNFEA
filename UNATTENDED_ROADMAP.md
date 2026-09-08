# WNFEA 72-Hour Unattended Autonomous Development Roadmap (10x SOTA Engine)

This document is the persistent execution backlog for autonomous development of a state-of-the-art structural FEA engine targeting 10x market speedups.

## Autonomous Agent Operational Guardrails
1. **Zero-Regression Gate**: Execute `python run_all_tests.py` before every commit. Must achieve 100% pass rate.
2. **Dedicated Sprint Branch**: Work strictly on `feat/unattended-72h-sprint`.
3. **Atomic Commits**: One commit per completed task with structured conventional commit messages.
4. **Failure Recovery**: Attempt up to 3 diagnostic fixes. If unresolved, rollback with `git checkout -- .`, record the blocker in `UNATTENDED_LOG.md`, and advance to the next independent task.
5. **State Synchronization**: Update status to `[COMPLETED]` and log execution metrics in `UNATTENDED_LOG.md`.

---

## Task Backlog

### Phase 1: Matrix-Free C3D10 Operator & VRAM Minimization (Target: 2M DOFs in <300MB VRAM)
- [x] **Task 1.1: Matrix-Free C3D10 Evaluation Engine**
  - **Objective**: Implement matrix-free operator `K @ u` in `wnfea/solver/matrix_free_c3d10.py` that computes element-level actions on-the-fly without assembling or storing global stiffness matrices.
  - **Acceptance Criteria**: Exact numerical parity with explicit CSR matrix (< 1e-12 relative error) while reducing memory consumption by >95%.
  - **Verification**: `python run_all_tests.py`.

- [ ] **Task 1.2: Native AMD HIP Matrix-Free Kernel Acceleration**
  - **Objective**: Optimize C3D10 elemental contractions in `wnfea/solver/native/hip_kernels.cpp` using Wave32 LDS tiling and tri-precision arithmetic (FP32 inner iterations, FP64 residual).
  - **Acceptance Criteria**: Peak compute utilization exceeding 50% of RX 7800 XT theoretical TFLOPs.
  - **Verification**: `python run_all_tests.py`.

- [ ] **Task 1.3: p-Multigrid Preconditioning (Linear Tet Coarse Grid for Quadratic Tet)**
  - **Objective**: Implement two-level geometric p-multigrid preconditioner where the coarse level is formed by vertex nodes (C3D4) and the fine level adds edge mid-nodes (C3D10).
  - **Acceptance Criteria**: Reduces PCG iteration count from ~300 to <30 iterations on 2M DOF solid problems.
  - **Verification**: `python run_all_tests.py`.

---

### Phase 2: Heterogeneous CPU + GPU Pipelining & 2M DOF Benchmark
- [ ] **Task 2.1: Heterogeneous Subsystem Assembly (99% Solids on GPU, 1% Beams on CPU)**
  - **Objective**: Create `wnfea/solver/heterogeneous_assembler.py` to concurrently condense 6-DOF beam constraints and Dirichlet conditions on CPU multi-threading while GPU streams solid element evaluations.
  - **Acceptance Criteria**: Seamless kinematic coupling between beam rot-DOFs and solid trans-DOFs with zero CPU-GPU transfer bottlenecks.
  - **Verification**: `python run_all_tests.py`.

- [ ] **Task 2.2: 2M DOF Realistic Large Structural Benchmark**
  - **Objective**: Build `scratch/benchmark_2m_dof.py` generating a complex 2,000,000 DOF solid-beam stiffened structure, profiling wallclock solve time, VRAM peak, and GFLOP/s.
  - **Acceptance Criteria**: Solves 2M DOFs in under 15 seconds with peak VRAM < 2.0 GB.
  - **Verification**: `python scratch/benchmark_2m_dof.py` and `python run_all_tests.py`.

---

### Phase 3: Fast Workflow Acceleration & Neural Warm-Start
- [ ] **Task 3.1: Automatic Boundary Condition & Contact Detection**
  - **Objective**: Implement `wnfea/boundary/auto_boundary_conditions.py` to analyze surface geometry and automatically classify bolt holes, base support planes, and gravitational body forces.
  - **Acceptance Criteria**: Automatically tags 100% of standard cylindrical support features on STEP models without manual picking.
  - **Verification**: `python run_all_tests.py`.

- [ ] **Task 3.2: Non-Linear JFNK Neural Warm-Start Interface**
  - **Objective**: Implement `wnfea/solver/neural_warm_start.py` allowing surrogate neural network predictions (NeMo / FNO / GNO) to initialize the displacement vector $u_0$.
  - **Acceptance Criteria**: Solvers accept warm-start priors and verify residual norm reduction, falling back to cold start if prior is degraded.
  - **Verification**: `python run_all_tests.py`.

- [ ] **Task 3.3: Localized Feature-Delta Remeshing**
  - **Objective**: Implement spatial bounding-box delta remeshing so minor CAD changes only re-mesh modified topological regions rather than the entire structure.
  - **Acceptance Criteria**: Remeshing a modified feature takes <10% of full-model meshing time.
  - **Verification**: `python run_all_tests.py`.

---

### Phase 4: Architectural Investigations & Generative Design (Documentation & Study)
- [ ] **Task 4.1: Voxel / Octree AMR vs Conformal C3D10 Architectural Study**
  - **Objective**: Author `docs/voxel_amr_architecture.md` evaluating Ansys Discovery-style cut-cell Cartesian AMR vs C3D10 conformal meshes for high-throughput GPU solves.
  - **Acceptance Criteria**: Detailed mathematical formulation, roofline comparison, and implementation roadmap.
  - **Verification**: File review & `python run_all_tests.py`.

- [ ] **Task 4.2: In-the-Loop Topology Optimization Specification**
  - **Objective**: Author `docs/generative_design_integration.md` defining integration of WNFEA's fast matrix-free solver into SIMP / Level-Set loops with GPU sensitivity filtering.
  - **Acceptance Criteria**: Concrete API interfaces and gradient formulation.
  - **Verification**: File review & `python run_all_tests.py`.

- [ ] **Task 4.3: 3-Axis & 5-Axis CNC Machinability Intelligence Framework**
  - **Objective**: Author `docs/machinability_study.md` defining parametric visibility cones, tool clearance constraints, and translation of density fields into parametric CAD B-rep surfaces (Fusion 360 style).
  - **Acceptance Criteria**: Rigorous mathematical formulation of differentiable CNC accessibility penalties and parametric feature constraints.
  - **Verification**: File review & `python run_all_tests.py`.
