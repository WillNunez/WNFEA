# WNFEA 72-Hour Unattended Autonomous Development Roadmap

This document serves as the persistent state machine and execution backlog for autonomous development on WNFEA.

## Autonomous Agent Execution Rules
1. **Never commit without 100% test pass**: Run `python run_all_tests.py` before every commit.
2. **Dedicated branch**: Work strictly on `feat/unattended-72h-sprint`.
3. **Atomic commits**: One commit per completed task with a descriptive conventional commit message.
4. **Failure recovery**: If tests fail after a task implementation, attempt up to 3 diagnostic fixes. If still failing, revert with `git checkout -- .`, mark the task `[BLOCKED]` in this roadmap, record the issue in `UNATTENDED_LOG.md`, and proceed to the next independent task.
5. **State logging**: After completing any task, update the status here to `[COMPLETED]` and append a detailed entry into `UNATTENDED_LOG.md`.

---

## Task Backlog

### Phase 1: BSR 6x6 & AMG Preconditioning Integration
- [ ] **Task 1.1: Integrate RCM Bandwidth Reduction into Assembler Pipeline**
  - **Objective**: Connect `wnfea/mesh/rcm.py` into `wnfea/solver/assembler.py` and `wnfea/model.py` so models can optionally enable RCM reordering before matrix assembly.
  - **Target Files**: `wnfea/solver/assembler.py`, `wnfea/model.py`
  - **Acceptance Criteria**: `FEAModel(enable_rcm=True)` reduces matrix profile bandwidth while preserving exact solution displacements after inverse permutation.
  - **Verification**: `python -m unittest tests/test_bsr_rcm.py` and `python run_all_tests.py`.

- [ ] **Task 1.2: BSR 6x6 Block-Diagonal Jacobi & Chebyshev Preconditioner**
  - **Objective**: Implement block-diagonal inverted 6x6 Jacobi preconditioner kernel in `wnfea/solver/bsr_matrix.py` and `wnfea/solver/fast_kernels.py`.
  - **Target Files**: `wnfea/solver/bsr_matrix.py`, `wnfea/solver/fast_kernels.py`
  - **Acceptance Criteria**: Preconditioned SpMV / PCG with BSR 6x6 converges in significantly fewer iterations than standard diagonal scaling on structural frame systems.
  - **Verification**: `python run_all_tests.py`.

- [ ] **Task 1.3: Automated Regression Suite for Integrated RCM Assembly**
  - **Objective**: Add dedicated test suite verifying end-to-end model assembly and solve with and without RCM.
  - **Target Files**: `tests/test_rcm_integration.py`, `run_all_tests.py`
  - **Acceptance Criteria**: All degrees of freedom match within 1e-12 between standard ordering and RCM ordering.
  - **Verification**: `python tests/test_rcm_integration.py` and `python run_all_tests.py`.

---

### Phase 2: Solid Mechanics & Gmsh C3D10 Pipeline Integration
- [ ] **Task 2.1: Gmsh 2nd-Order C3D10 Mesh Generation Integration**
  - **Objective**: Enhance `wnfea/mesh/gmsh_mesher.py` to support tetrahedral 2nd-order C3D10 element generation from CAD STEP files.
  - **Target Files**: `wnfea/mesh/gmsh_mesher.py`, `wnfea/mesh/__init__.py`
  - **Acceptance Criteria**: Gmsh generates 10-node tetrahedra when requested with mid-side nodes placed on curved geometric boundaries.
  - **Verification**: `python run_all_tests.py`.

- [ ] **Task 2.2: Solid Solver Routing in FEAModel**
  - **Objective**: Update `wnfea/model.py` `solve()` method to automatically detect solid elements and route assembly and boundary condition imposition to `wnfea/solver/solid_solver.py`.
  - **Target Files**: `wnfea/model.py`, `wnfea/solver/solid_solver.py`
  - **Acceptance Criteria**: A model containing C3D10 solid elements can be solved via `model.solve()` with valid displacement field and Von Mises stress tensor.
  - **Verification**: `python run_all_tests.py`.

- [ ] **Task 2.3: End-to-End Solid Pipeline Regression Test**
  - **Objective**: Create an automated test verifying CAD STEP import -> C3D10 meshing -> PCG solve -> Von Mises calculation.
  - **Target Files**: `tests/test_solid_cad_pipeline.py`, `run_all_tests.py`
  - **Acceptance Criteria**: Solution converges and matches analytical cantilever solid deflection within 5%.
  - **Verification**: `python tests/test_solid_cad_pipeline.py` and `python run_all_tests.py`.

---

### Phase 3: Hardware Monitor Telemetry & Desktop App Integration
- [ ] **Task 3.1: Headless & Telemetry Unit Tests for Hardware Monitor**
  - **Objective**: Create headless test suite for `gui_hardware_monitor.py` verifying metric polling, ROCm/GPU fallback, and bottleneck analysis without opening a blocking window.
  - **Target Files**: `tests/test_hardware_monitor.py`, `gui_hardware_monitor.py`, `run_all_tests.py`
  - **Acceptance Criteria**: Test polls CPU, RAM, GPU metrics and validates calculation logic without UI deadlocks.
  - **Verification**: `python tests/test_hardware_monitor.py` and `python run_all_tests.py`.

- [ ] **Task 3.2: Desktop App Hardware Monitor Action**
  - **Objective**: Add a menu bar item / toolbar button in `desktop_app.py` to launch the hardware monitor as a standalone detached subprocess.
  - **Target Files**: `desktop_app.py`
  - **Acceptance Criteria**: Clicking "Hardware Monitor" spawns `gui_hardware_monitor.py` without blocking the main PySide6 UI thread.
  - **Verification**: `python run_all_tests.py`.

- [ ] **Task 3.3: Graceful Driver & Platform Fallbacks**
  - **Objective**: Ensure all solver components and monitors degrade gracefully when AMD HIP/ROCm or PySide6 are unavailable.
  - **Target Files**: `wnfea/solver/fast_kernels.py`, `wnfea/solver/amg_preconditioner.py`, `gui_hardware_monitor.py`
  - **Acceptance Criteria**: Code operates in pure CPU mode without throwing unhandled exceptions when GPU libraries are absent.
  - **Verification**: `python run_all_tests.py`.

---

### Phase 4: Non-Linear JFNK Solvers & Benchmarking
- [ ] **Task 4.1: Non-Linear Large Deformation Formulations for 3D Solids**
  - **Objective**: Extend matrix-free JFNK solver to handle geometric nonlinearity in 3D tetrahedral solid elements.
  - **Target Files**: `wnfea/solver/nonlinear_jfnk.py`, `wnfea/elements/c3d10.py`
  - **Acceptance Criteria**: Large-deflection solid test converges across multiple load steps with quadratic asymptotic rate.
  - **Verification**: `python run_all_tests.py`.

- [ ] **Task 4.2: Automated Scalability & Roofline Benchmark Suite**
  - **Objective**: Create `wnfea/benchmark/benchmark_suite.py` that automatically records throughput (GFLOP/s, GB/s, solve time) across varying DOF scales (10k, 50k, 100k, 250k) and exports JSON/CSV metrics.
  - **Target Files**: `wnfea/benchmark/benchmark_suite.py`
  - **Acceptance Criteria**: Generates structured benchmark report in `results/benchmark_metrics.json` without failing when GPU is stressed.
  - **Verification**: `python run_all_tests.py`.

---

### Phase 5: CI/CD Automation & Documentation
- [ ] **Task 5.1: GitHub Actions Automated CI Workflow**
  - **Objective**: Create `.github/workflows/ci.yml` running `python run_all_tests.py` on all PRs and pushes.
  - **Target Files**: `.github/workflows/ci.yml`
  - **Acceptance Criteria**: GitHub Actions workflow file is syntactically valid and executes tests on Windows and Ubuntu runners.
  - **Verification**: Validate syntax and verify with `python run_all_tests.py`.

- [ ] **Task 5.2: Comprehensive Architecture & API Documentation**
  - **Objective**: Update `README.md` with complete solver architecture overview, BSR 6x6 benchmarks, RCM ordering instructions, and telemetry monitoring guide.
  - **Target Files**: `README.md`
  - **Acceptance Criteria**: Complete, up-to-date documentation with clean GitHub-flavored markdown.
  - **Verification**: Inspect markdown formatting and verify `python run_all_tests.py`.
