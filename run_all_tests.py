#!/usr/bin/env python3
"""
WNFEA Autonomous Verification Gate and Test Suite Runner
-------------------------------------------------------
Executes all core regression and verification suites with strict per-test timeouts
to catch GPU driver hangs, infinite loops, or computational regressions.
"""

import sys
import os
import time
import subprocess

TEST_SUITES = [
    ("BSR 6x6 and RCM Bandwidth Reduction", [sys.executable, os.path.join("tests", "test_bsr_rcm.py")]),
    ("C3D10 Solid Benchmark and PCG Solver", [sys.executable, os.path.join("tests", "test_solid_bench.py")]),
    ("End-to-End Pipeline Validation", [sys.executable, "test_pipeline.py"]),
    ("C3D10 Element Formulation and Patch Test", [sys.executable, "test_c3d10.py"]),
    ("6-DOF / 3-DOF Direct Elimination and Coupling", [sys.executable, "test_dof_coupling.py"]),
    ("AMD HIP GPU vs CPU Parity", [sys.executable, "test_gpu.py"]),
    ("Non-Linear JFNK and AMG Large Deflection", [sys.executable, "test_nonlinear_jfnk.py"]),
]

TIMEOUT_SECONDS = 90  # 90-second safety timeout per suite

def run_suite(name, cmd):
    print("\n" + "=" * 70)
    print(f" RUNNING: {name}")
    print(f" COMMAND: {' '.join(cmd)}")
    print("=" * 70)
    
    start_time = time.time()
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=TIMEOUT_SECONDS
        )
        elapsed = time.time() - start_time
        print(proc.stdout)
        if proc.returncode == 0:
            print(f"--> [PASS] {name} completed in {elapsed:.2f}s")
            return True, elapsed, None
        else:
            print(f"--> [FAIL] {name} exited with code {proc.returncode} in {elapsed:.2f}s")
            return False, elapsed, f"Exit code {proc.returncode}"
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"--> [TIMEOUT] {name} exceeded {TIMEOUT_SECONDS}s timeout!")
        return False, elapsed, "Timeout exceeded"
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"--> [ERROR] {name} encountered execution exception: {e}")
        return False, elapsed, str(e)

def main():
    print("=" * 70)
    print("   WNFEA AUTOMATED VERIFICATION HARNESS")
    print(f"   Python: {sys.executable}")
    print(f"   Working Dir: {os.getcwd()}")
    print("=" * 70)
    
    total_start = time.time()
    results = []
    
    for name, cmd in TEST_SUITES:
        passed, elapsed, err = run_suite(name, cmd)
        results.append((name, passed, elapsed, err))
    
    total_elapsed = time.time() - total_start
    all_passed = all(r[1] for r in results)
    
    print("\n" + "=" * 70)
    print("   TEST RUN SUMMARY")
    print("=" * 70)
    for name, passed, elapsed, err in results:
        status_str = "PASS" if passed else "FAIL"
        err_str = f" ({err})" if err else ""
        print(f"  [{status_str:4s}] {name:<48s} {elapsed:6.2f}s{err_str}")
    
    print("-" * 70)
    passed_count = sum(1 for r in results if r[1])
    total_count = len(results)
    print(f"Total: {passed_count}/{total_count} passed in {total_elapsed:.2f}s")
    print("=" * 70)
    
    if all_passed:
        print("\nALL SUITES PASSED (100% PASS RATE) -- VERIFICATION GATE CLEARED.\n")
        sys.exit(0)
    else:
        print(f"\nVERIFICATION GATE FAILED ({total_count - passed_count} SUITE(S) FAILED).\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
