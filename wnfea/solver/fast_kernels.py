"""
Fast Native & Batched SIMD Element Kernels for WNFEA.

Implements high-performance internal force evaluations for co-rotational beam
and C3D10 solid elements with:
1. Native compiled C++/AVX-FMA acceleration (compiled via ROCm clang++).
2. Fully vectorized NumPy batch SIMD fallback (broadcasting with np.add.at).
3. Coalesced memory access layout to maximize cache line utilization and
   prevent memory thrashing.
"""

from __future__ import annotations

import os
import ctypes
from dataclasses import dataclass
import numpy as np

from ..model import FEAModel


@dataclass
class BatchModelData:
    """Contiguous, coalesced pre-allocated memory buffers for a FEAModel."""
    nodes: np.ndarray             # (N, 3) float64 contiguous
    # Beams
    has_beams: bool
    beam_elements: np.ndarray     # (E_beam, 2) int32 contiguous
    beam_props: np.ndarray        # (E_beam, 6) float64 contiguous
    beam_e_ref: np.ndarray        # (E_beam, 9) float64 contiguous (e1_0, e2_0, e3_0)
    beam_L0: np.ndarray           # (E_beam,) float64 contiguous
    n_beams: int
    # Solids
    has_solids: bool
    solid_elements: np.ndarray    # (E_solid, 10) int32 contiguous
    solid_props: np.ndarray       # (E_solid, 2) float64 contiguous (E, nu)
    n_solids: int
    n_nodes: int


# Global DLL handles
_NATIVE_LIB = None
_DLL_LOAD_ATTEMPTED = False

_HIP_LIB = None
_HIP_LOAD_ATTEMPTED = False
_HIP_DEVICE_SUMMARY = None


class HipDeviceSummary(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 128),
        ("arch", ctypes.c_char * 32),
        ("multiProcessorCount", ctypes.c_int),
        ("warpSize", ctypes.c_int),
        ("maxThreadsPerBlock", ctypes.c_int),
        ("maxThreadsPerMP", ctypes.c_int),
        ("regsPerBlock", ctypes.c_int),
        ("sharedMemPerBlock", ctypes.c_int),
        ("l2CacheSize", ctypes.c_int),
        ("optimalBeamBlockSize", ctypes.c_int),
        ("optimalSolidBlockSize", ctypes.c_int),
        ("optimalMinGridSize", ctypes.c_int),
        ("optimalMaxGridSize", ctypes.c_int),
    ]


def _get_native_lib():
    """Load or return the compiled native CPU kernel library (AVX2/FMA)."""
    global _NATIVE_LIB, _DLL_LOAD_ATTEMPTED
    if _DLL_LOAD_ATTEMPTED:
        return _NATIVE_LIB

    _DLL_LOAD_ATTEMPTED = True
    dll_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "native")
    dll_path = os.path.join(dll_dir, "wnfea_kernels.dll")

    if os.path.exists(dll_path):
        try:
            lib = ctypes.cdll.LoadLibrary(dll_path)
            lib.compute_beam_internal_forces_native.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int
            ]
            lib.compute_c3d10_internal_forces_native.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_int, ctypes.c_int
            ]
            if hasattr(lib, "compute_c3d10_diagonal_native"):
                lib.compute_c3d10_diagonal_native.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_int, ctypes.c_int
                ]
            _NATIVE_LIB = lib
        except Exception:
            _NATIVE_LIB = None
    return _NATIVE_LIB


def _get_hip_lib():
    """Load or return the compiled architecture-tuned HIP GPU kernel library."""
    global _HIP_LIB, _HIP_LOAD_ATTEMPTED
    if _HIP_LOAD_ATTEMPTED:
        return _HIP_LIB

    _HIP_LOAD_ATTEMPTED = True
    dll_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "native")
    dll_path = os.path.join(dll_dir, "wnfea_hip_kernels.dll")

    if os.path.exists(dll_path):
        try:
            lib = ctypes.cdll.LoadLibrary(dll_path)
            lib.hip_get_device_summary.argtypes = [ctypes.POINTER(HipDeviceSummary)]
            lib.hip_get_device_summary.restype = ctypes.c_int

            lib.hip_compute_beam_forces.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
            ]
            lib.hip_compute_beam_forces.restype = ctypes.c_int

            lib.hip_compute_c3d10_forces.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
            ]
            lib.hip_compute_c3d10_forces.restype = ctypes.c_int

            lib.hip_spmv_csr_solve.argtypes = [
                ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double, ctypes.c_double,
                ctypes.c_int, ctypes.c_int, ctypes.c_int
            ]
            lib.hip_spmv_csr_solve.restype = ctypes.c_int

            # Resident GPU AMG Preconditioner
            lib.hip_amg_create.argtypes = [
                ctypes.c_int, ctypes.c_int, ctypes.c_void_p
            ]
            lib.hip_amg_create.restype = ctypes.c_void_p

            lib.hip_amg_set_level.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ]
            lib.hip_amg_set_level.restype = ctypes.c_int

            lib.hip_amg_set_coarse_cholesky.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p
            ]
            lib.hip_amg_set_coarse_cholesky.restype = ctypes.c_int

            lib.hip_amg_apply_vcycle.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int
            ]
            lib.hip_amg_apply_vcycle.restype = ctypes.c_int

            lib.hip_amg_apply_vcycle_device.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int
            ]
            lib.hip_amg_apply_vcycle_device.restype = ctypes.c_int

            lib.hip_amg_set_coarse_sweeps.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.hip_amg_set_coarse_sweeps.restype = None

            lib.hip_amg_solve_pcg.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)
            ]
            lib.hip_amg_solve_pcg.restype = ctypes.c_int

            lib.hip_amg_solve_pcg_device.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)
            ]
            lib.hip_amg_solve_pcg_device.restype = ctypes.c_int

            lib.hip_amg_destroy.argtypes = [ctypes.c_void_p]
            lib.hip_amg_destroy.restype = None

            lib.hip_assemble_beam_system_csr.argtypes = [
                ctypes.c_int, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int
            ]
            lib.hip_assemble_beam_system_csr.restype = ctypes.c_int

            lib.hip_spmv_bsr6x6_fp32.argtypes = [
                ctypes.c_int, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_float, ctypes.c_float,
                ctypes.c_int, ctypes.c_int
            ]
            lib.hip_spmv_bsr6x6_fp32.restype = ctypes.c_int

            lib.hip_spmv_bsr6x6_fp64.argtypes = [
                ctypes.c_int, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_double, ctypes.c_double,
                ctypes.c_int, ctypes.c_int
            ]
            lib.hip_spmv_bsr6x6_fp64.restype = ctypes.c_int

            lib.hip_spmv_bsr6x6_benchmark.argtypes = [
                ctypes.c_int, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
            ]
            lib.hip_spmv_bsr6x6_benchmark.restype = ctypes.c_int

            # Matrix-Free C3D10 3-DOF GPU Kernels
            lib.hip_c3d10_matrix_free_matvec_fp64.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int
            ]
            lib.hip_c3d10_matrix_free_matvec_fp64.restype = ctypes.c_int

            lib.hip_c3d10_matrix_free_matvec_fp32.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int
            ]
            lib.hip_c3d10_matrix_free_matvec_fp32.restype = ctypes.c_int

            _HIP_LIB = lib
        except Exception:
            _HIP_LIB = None
    return _HIP_LIB


def get_hip_device_summary() -> HipDeviceSummary | None:
    """Returns detailed hardware & cache profile of the active AMD GPU."""
    global _HIP_DEVICE_SUMMARY
    if _HIP_DEVICE_SUMMARY is not None:
        return _HIP_DEVICE_SUMMARY

    lib = _get_hip_lib()
    if lib is None:
        return None

    summary = HipDeviceSummary()
    err = lib.hip_get_device_summary(ctypes.byref(summary))
    if err == 0:
        _HIP_DEVICE_SUMMARY = summary
        return summary
    return None


def is_hip_available() -> bool:
    """Returns True if AMD HIP GPU runtime and kernels are available."""
    return get_hip_device_summary() is not None


def get_optimal_launch_config(n_items: int, item_type: str = "beam") -> tuple[int, int]:
    """
    Computes optimal (block_size, grid_size) to maximize occupancy and fit
    registers and L1/L2 caches of the AMD GPU.
    """
    summary = get_hip_device_summary()
    if summary is None:
        block_size = 256 if item_type == "beam" else 32
        min_grid = 60
        max_grid = 480
    else:
        block_size = summary.optimalBeamBlockSize if item_type == "beam" else summary.optimalSolidBlockSize
        min_grid = summary.optimalMinGridSize
        max_grid = summary.optimalMaxGridSize

    desired = (n_items + block_size - 1) // block_size
    grid_size = min(max(desired, min_grid), max_grid)
    return block_size, grid_size


def prepare_batch_data(model: FEAModel) -> BatchModelData:
    """
    Pack and coalesce model geometry, connectivity, and properties into
    contiguous 64-byte aligned memory arrays.
    """
    nodes = np.ascontiguousarray(model.mesh_nodes, dtype=np.float64)
    n_nodes = len(nodes)

    # 1. Beams
    has_beams = model.mesh_elements is not None and len(model.mesh_elements) > 0
    if has_beams:
        beam_elements = np.ascontiguousarray(model.mesh_elements, dtype=np.int32)
        n_beams = len(beam_elements)
        beam_props = np.zeros((n_beams, 6), dtype=np.float64)

        for eid in range(n_beams):
            asgn = model.element_properties[eid]
            mat = model.materials[asgn.material_name]
            sec = model.sections[asgn.section_name]
            beam_props[eid] = [
                mat.youngs_modulus, mat.shear_modulus, sec.area, sec.iy, sec.iz, sec.j
            ]
        beam_props = np.ascontiguousarray(beam_props)

        # Precompute initial reference triads and lengths
        n1 = beam_elements[:, 0]
        n2 = beam_elements[:, 1]
        X1 = nodes[n1]
        X2 = nodes[n2]
        v0 = X2 - X1
        beam_L0 = np.ascontiguousarray(np.linalg.norm(v0, axis=1), dtype=np.float64)
        e1_0 = v0 / np.maximum(beam_L0[:, None], 1e-15)

        beam_e_ref = np.zeros((n_beams, 9), dtype=np.float64)
        for i in range(n_beams):
            ref = np.array([0., 0., 1.]) if abs(e1_0[i, 2]) < 0.9 else np.array([0., 1., 0.])
            e2 = np.cross(ref, e1_0[i])
            e2 /= np.linalg.norm(e2)
            e3 = np.cross(e1_0[i], e2)
            e3 /= np.linalg.norm(e3)
            beam_e_ref[i, 0:3] = e1_0[i]
            beam_e_ref[i, 3:6] = e2
            beam_e_ref[i, 6:9] = e3
        beam_e_ref = np.ascontiguousarray(beam_e_ref)
    else:
        beam_elements = np.empty((0, 2), dtype=np.int32)
        beam_props = np.empty((0, 6), dtype=np.float64)
        beam_e_ref = np.empty((0, 9), dtype=np.float64)
        beam_L0 = np.empty(0, dtype=np.float64)
        n_beams = 0

    # 2. Solids (C3D10)
    has_solids = getattr(model, "solid_elements", None) is not None and len(model.solid_elements) > 0
    if has_solids:
        solid_elements = np.ascontiguousarray(model.solid_elements, dtype=np.int32)
        n_solids = len(solid_elements)
        solid_props = np.zeros((n_solids, 2), dtype=np.float64)
        for eid in range(n_solids):
            if hasattr(model, "solid_materials") and eid in model.solid_materials:
                mat_name = model.solid_materials[eid]
            elif hasattr(model, "solid_element_properties") and eid in model.solid_element_properties:
                asgn = model.solid_element_properties[eid]
                mat_name = asgn.material_name if hasattr(asgn, "material_name") else str(asgn)
            elif model.materials:
                mat_name = next(iter(model.materials.keys()))
            else:
                mat_name = None

            if mat_name and mat_name in model.materials:
                mat = model.materials[mat_name]
                solid_props[eid] = [mat.youngs_modulus, mat.poissons_ratio]
            else:
                solid_props[eid] = [2.1e11, 0.3] # Default steel
        solid_props = np.ascontiguousarray(solid_props)
    else:
        solid_elements = np.empty((0, 10), dtype=np.int32)
        solid_props = np.empty((0, 2), dtype=np.float64)
        n_solids = 0

    batch = BatchModelData(
        nodes=nodes,
        has_beams=has_beams,
        beam_elements=beam_elements,
        beam_props=beam_props,
        beam_e_ref=beam_e_ref,
        beam_L0=beam_L0,
        n_beams=n_beams,
        has_solids=has_solids,
        solid_elements=solid_elements,
        solid_props=solid_props,
        n_solids=n_solids,
        n_nodes=n_nodes,
    )
    return batch


def compute_internal_forces_fast(
    model: FEAModel,
    U_full: np.ndarray,
    F_int: np.ndarray | None = None,
    device: str = "cpu",
    block_size: int = 0,
    grid_size: int = 0,
) -> np.ndarray:
    """
    High-performance internal force vector evaluation with coalesced memory access.
    Supports AMD GPU execution via HIP, native AVX2 CPU execution, and vectorized NumPy.

    Args:
        model: FEAModel instance.
        U_full: Displacements vector (n_nodes * 6).
        F_int: Output force vector (optional, allocated if None).
        device: "cpu", "hip", or "gpu".
        block_size: Custom HIP block size (0 = auto-tuned to GPU registers/cache).
        grid_size: Custom HIP grid size (0 = auto-tuned to WGP count/L2 cache).
    """
    if not hasattr(model, "_batch_data") or model._batch_data is None:
        model._batch_data = prepare_batch_data(model)

    b: BatchModelData = model._batch_data
    if F_int is None:
        F_int = np.zeros(b.n_nodes * 6, dtype=np.float64)
    else:
        F_int.fill(0.0)

    # 1. AMD GPU Execution (HIP)
    if device.lower() in ("hip", "gpu"):
        hip_lib = _get_hip_lib()
        if hip_lib is not None:
            if b.has_beams:
                b_size, g_size = (block_size, grid_size) if (block_size > 0 and grid_size > 0) else get_optimal_launch_config(b.n_beams, "beam")
                hip_lib.hip_compute_beam_forces(
                    b.nodes.ctypes.data,
                    b.beam_elements.ctypes.data,
                    b.beam_props.ctypes.data,
                    b.beam_e_ref.ctypes.data,
                    b.beam_L0.ctypes.data,
                    U_full.ctypes.data,
                    F_int.ctypes.data,
                    b.n_beams,
                    b.n_nodes,
                    b_size,
                    g_size,
                )
            if b.has_solids:
                b_size, g_size = (block_size, grid_size) if (block_size > 0 and grid_size > 0) else get_optimal_launch_config(b.n_solids, "solid")
                hip_lib.hip_compute_c3d10_forces(
                    b.nodes.ctypes.data,
                    b.solid_elements.ctypes.data,
                    b.solid_props.ctypes.data,
                    U_full.ctypes.data,
                    F_int.ctypes.data,
                    b.n_solids,
                    b.n_nodes,
                    b_size,
                    g_size,
                )
            return F_int

    # 2. Native CPU execution (AVX2 / FMA)
    lib = _get_native_lib()

    if lib is not None:
        if b.has_beams:
            lib.compute_beam_internal_forces_native(
                b.nodes.ctypes.data,
                b.beam_elements.ctypes.data,
                b.beam_props.ctypes.data,
                b.beam_e_ref.ctypes.data,
                b.beam_L0.ctypes.data,
                U_full.ctypes.data,
                F_int.ctypes.data,
                b.n_beams,
                b.n_nodes,
            )
        if b.has_solids:
            lib.compute_c3d10_internal_forces_native(
                b.nodes.ctypes.data,
                b.solid_elements.ctypes.data,
                b.solid_props.ctypes.data,
                U_full.ctypes.data,
                F_int.ctypes.data,
                b.n_solids,
                b.n_nodes,
            )
        return F_int

    # 2. Vectorized NumPy Fallback
    if b.has_beams:
        n1 = b.beam_elements[:, 0]
        n2 = b.beam_elements[:, 1]
        X1 = b.nodes[n1]
        X2 = b.nodes[n2]
        v0 = X2 - X1
        L0 = b.beam_L0

        e1_0 = b.beam_e_ref[:, 0:3]
        e2_0 = b.beam_e_ref[:, 3:6]
        e3_0 = b.beam_e_ref[:, 6:9]

        U_nodes = U_full.reshape(-1, 6)
        u1 = U_nodes[n1, :3]
        th1 = U_nodes[n1, 3:6]
        u2 = U_nodes[n2, :3]
        th2 = U_nodes[n2, 3:6]

        du = u2 - u1
        v = v0 + du
        L = np.linalg.norm(v, axis=1)
        r1 = v / np.maximum(L[:, None], 1e-15)

        cross = np.cross(e1_0, r1)
        sin_phi = np.linalg.norm(cross, axis=1)
        cos_phi = np.sum(e1_0 * r1, axis=1)

        r2 = e2_0.copy()
        r3 = e3_0.copy()
        mask = sin_phi > 1e-12
        if np.any(mask):
            ax = cross[mask] / sin_phi[mask, None]
            phi = np.arctan2(sin_phi[mask], cos_phi[mask])
            cp = np.cos(phi)[:, None]
            sp = np.sin(phi)[:, None]
            ax_dot_e2 = np.sum(ax * e2_0[mask], axis=1, keepdims=True)
            ax_dot_e3 = np.sum(ax * e3_0[mask], axis=1, keepdims=True)
            r2[mask] = e2_0[mask] * cp + np.cross(ax, e2_0[mask]) * sp + ax * ax_dot_e2 * (1.0 - cp)
            r3[mask] = e3_0[mask] * cp + np.cross(ax, e3_0[mask]) * sp + ax * ax_dot_e3 * (1.0 - cp)

        avg_twist = 0.5 * np.sum((th1 + th2) * r1, axis=1)
        ct = np.cos(avg_twist)[:, None]
        st = np.sin(avg_twist)[:, None]
        r2_t = r2 * ct + np.cross(r1, r2) * st
        r3_t = r3 * ct + np.cross(r1, r3) * st
        r2 = r2_t
        r3 = r3_t

        r1 = r1 / np.linalg.norm(r1, axis=1, keepdims=True)
        r2 = r2 - np.sum(r2 * r1, axis=1, keepdims=True) * r1
        r2 = r2 / np.linalg.norm(r2, axis=1, keepdims=True)
        r3 = np.cross(r1, r2)
        r3 = r3 / np.linalg.norm(r3, axis=1, keepdims=True)

        u_l = (2.0 * np.sum(v0 * du, axis=1) + np.sum(du * du, axis=1)) / (L + L0)
        th_y1 = np.sum(r2 * th1, axis=1) + np.sum(r3 * du, axis=1) / L
        th_z1 = np.sum(r3 * th1, axis=1) - np.sum(r2 * du, axis=1) / L
        th_x  = np.sum(r1 * (th2 - th1), axis=1)
        th_y2 = np.sum(r2 * th2, axis=1) + np.sum(r3 * du, axis=1) / L
        th_z2 = np.sum(r3 * th2, axis=1) - np.sum(r2 * du, axis=1) / L

        E_mod = b.beam_props[:, 0]
        G_mod = b.beam_props[:, 1]
        A_area = b.beam_props[:, 2]
        Iy = b.beam_props[:, 3]
        Iz = b.beam_props[:, 4]
        J = b.beam_props[:, 5]

        N   = (E_mod * A_area / L0) * u_l
        My1 = (E_mod * Iy / L0) * (4.0 * th_y1 + 2.0 * th_y2)
        My2 = (E_mod * Iy / L0) * (2.0 * th_y1 + 4.0 * th_y2)
        Mz1 = (E_mod * Iz / L0) * (4.0 * th_z1 + 2.0 * th_z2)
        Mz2 = (E_mod * Iz / L0) * (2.0 * th_z1 + 4.0 * th_z2)
        Tx  = (G_mod * J / L0) * th_x

        inv_L = 1.0 / L
        f1 = -N[:, None] * r1 - ((My1 + My2) * inv_L)[:, None] * r3 + ((Mz1 + Mz2) * inv_L)[:, None] * r2
        m1 = -Tx[:, None] * r1 + My1[:, None] * r2 + Mz1[:, None] * r3
        f2 = -f1
        m2 = Tx[:, None] * r1 + My2[:, None] * r2 + Mz2[:, None] * r3

        f_nodes = np.zeros((b.n_nodes, 6), dtype=np.float64)
        np.add.at(f_nodes, n1, np.hstack([f1, m1]))
        np.add.at(f_nodes, n2, np.hstack([f2, m2]))
        F_int += f_nodes.ravel()

    return F_int


def spmv_hip(
    A_csr,
    x: np.ndarray,
    y: np.ndarray | None = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    block_size: int = 0,
    grid_size: int = 0,
) -> np.ndarray:
    """
    Wave32-tiled sparse matrix-vector multiplication (CSR SpMV) on AMD GPU.

    Args:
        A_csr: scipy.sparse.csr_matrix.
        x: Input dense vector (length = A_csr.shape[1]).
        y: In-out dense vector (length = A_csr.shape[0]).
        alpha: Scalar multiplier for A*x.
        beta: Scalar multiplier for y.
        block_size: Custom block size (default 256).
        grid_size: Custom grid size (default auto-tuned to 30 WGPs).
    """
    hip_lib = _get_hip_lib()
    if hip_lib is None:
        raise RuntimeError("HIP kernel library (wnfea_hip_kernels.dll) is not available.")

    num_rows = A_csr.shape[0]
    if y is None:
        y = np.zeros(num_rows, dtype=np.float64)
    elif beta == 0.0:
        y.fill(0.0)

    x_contig = np.ascontiguousarray(x, dtype=np.float64)
    y_contig = np.ascontiguousarray(y, dtype=np.float64)
    row_ptr = np.ascontiguousarray(A_csr.indptr, dtype=np.int32)
    col_idx = np.ascontiguousarray(A_csr.indices, dtype=np.int32)
    values = np.ascontiguousarray(A_csr.data, dtype=np.float64)

    err = hip_lib.hip_spmv_csr_solve(
        num_rows,
        row_ptr.ctypes.data,
        col_idx.ctypes.data,
        values.ctypes.data,
        x_contig.ctypes.data,
        y_contig.ctypes.data,
        alpha,
        beta,
        A_csr.nnz,
        block_size,
        grid_size,
    )
    if err != 0:
        raise RuntimeError(f"hip_spmv_csr_solve failed with code {err}")

    if y is not y_contig:
        np.copyto(y, y_contig)
    return y


def hip_solve_pcg_amg(
    amg_preconditioner,
    b: np.ndarray,
    rtol: float = 1e-6,
    max_iter: int = 500,
    precision_mode: int | None = None,
    check_interval: int = 1,
    coarse_sweeps: int | None = None,
) -> tuple[np.ndarray, int, float]:
    """
    Execute 100% GPU-Resident Preconditioned Conjugate Gradient (PCG) on AMD GPU.
    The outer Krylov search vectors (u, r, p, q) stay resident in GPU VRAM,
    eliminating all PCIe bus synchronization latency.

    Args:
        amg_preconditioner: An instance of BlockBeamAMGPreconditioner with gpu_handle != None.
        b: Right-hand-side vector (numpy array).
        rtol: Relative convergence tolerance (e.g. 1e-6).
        max_iter: Maximum PCG iterations.
        precision_mode: 0 for FP64 V-cycle, 1 for FP32 V-cycle, 2 for FP16 V-cycle.
                        If None, uses amg_preconditioner.working_dtype.
        check_interval: Number of iterations between host-device residual syncs (default: 1).
                        Set to e.g. 4 for pure back-to-back GPU execution.
        coarse_sweeps: Number of smoothing sweeps on coarsest AMG level.

    Returns:
        (u, iters, final_rel_residual)
    """
    hip_lib = _get_hip_lib()
    if hip_lib is None:
        raise RuntimeError("HIP kernel library (wnfea_hip_kernels.dll) is not available.")

    if getattr(amg_preconditioner, "gpu_handle", None) is None:
        raise ValueError("Provided AMG preconditioner does not have an active GPU handle.")

    if coarse_sweeps is not None and coarse_sweeps > 0:
        hip_lib.hip_amg_set_coarse_sweeps(amg_preconditioner.gpu_handle, ctypes.c_int(coarse_sweeps))

    if precision_mode is None:
        if getattr(amg_preconditioner, "use_tri_precision", False):
            precision_mode = 2  # FP16
        elif amg_preconditioner.working_dtype == np.float32:
            precision_mode = 1  # FP32
        else:
            precision_mode = 0  # FP64

    b_contig = np.ascontiguousarray(b, dtype=np.float64)
    u_out = np.zeros_like(b_contig)
    iters_out = ctypes.c_int(0)
    res_out = ctypes.c_double(0.0)

    err = hip_lib.hip_amg_solve_pcg(
        amg_preconditioner.gpu_handle,
        b_contig.ctypes.data,
        u_out.ctypes.data,
        ctypes.c_double(rtol),
        ctypes.c_int(max_iter),
        ctypes.c_int(precision_mode),
        ctypes.c_int(check_interval),
        ctypes.byref(iters_out),
        ctypes.byref(res_out),
    )
    if err != 0:
        raise RuntimeError(f"hip_amg_solve_pcg failed with error code {err}")

    return u_out, iters_out.value, res_out.value


def hip_assemble_beam_system_csr(
    nodes: np.ndarray,
    elements: np.ndarray,
    props: np.ndarray,
    csr_row_ptr: np.ndarray | None = None,
    csr_col_idx: np.ndarray | None = None,
    csr_values: np.ndarray | None = None,
    fixed_dofs: np.ndarray | None = None,
    F: np.ndarray | None = None,
    direct_scatter: bool = True,
    compute_elem_vals: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Assemble the global 3D beam stiffness matrix using native GPU HIP kernels.

    Args:
        nodes: [N, 3] float64 node coordinates.
        elements: [E, 2] int32 element node indices.
        props: [E, 6] float64 element properties (E, G, A, Iy, Iz, J).
        csr_row_ptr: [N*6 + 1] int32 CSR row pointer (optional for direct scatter).
        csr_col_idx: [NNZ] int32 CSR column index (optional for direct scatter).
        csr_values: [NNZ] float64 CSR value buffer (will be populated on GPU).
        fixed_dofs: [num_fixed] int32 indices of constrained DOFs.
        F: [N*6] float64 right-hand side force vector.
        direct_scatter: Whether to scatter directly into csr_values on GPU.
        compute_elem_vals: Whether to compute and return raw [E, 144] element matrices.

    Returns:
        (csr_values, F, elem_vals)
    """
    hip_lib = _get_hip_lib()
    if hip_lib is None or not hasattr(hip_lib, "hip_assemble_beam_system_csr"):
        raise RuntimeError("hip_assemble_beam_system_csr not available in HIP DLL.")

    num_nodes = len(nodes)
    num_elements = len(elements)

    nodes_c = np.ascontiguousarray(nodes, dtype=np.float64)
    elements_c = np.ascontiguousarray(elements, dtype=np.int32)
    props_c = np.ascontiguousarray(props, dtype=np.float64)

    nnz = len(csr_col_idx) if csr_col_idx is not None else 0
    row_ptr_data = csr_row_ptr.ctypes.data if csr_row_ptr is not None else None
    col_idx_data = csr_col_idx.ctypes.data if csr_col_idx is not None else None

    if direct_scatter:
        if csr_values is None:
            csr_values = np.zeros(nnz, dtype=np.float64)
        csr_values_c = np.ascontiguousarray(csr_values, dtype=np.float64)
        csr_values_data = csr_values_c.ctypes.data
    else:
        csr_values_c = None
        csr_values_data = None

    if fixed_dofs is not None and len(fixed_dofs) > 0:
        fixed_c = np.ascontiguousarray(fixed_dofs, dtype=np.int32)
        fixed_data = fixed_c.ctypes.data
        num_fixed = len(fixed_c)
    else:
        fixed_data = None
        num_fixed = 0

    if F is not None:
        F_c = np.ascontiguousarray(F, dtype=np.float64)
        F_data = F_c.ctypes.data
    else:
        F_c = None
        F_data = None

    if compute_elem_vals:
        elem_vals = np.empty((num_elements, 144), dtype=np.float64)
        elem_vals_data = elem_vals.ctypes.data
    else:
        elem_vals = None
        elem_vals_data = None

    err = hip_lib.hip_assemble_beam_system_csr(
        ctypes.c_int(num_nodes),
        ctypes.c_int(num_elements),
        nodes_c.ctypes.data,
        elements_c.ctypes.data,
        props_c.ctypes.data,
        ctypes.c_int(nnz),
        row_ptr_data,
        col_idx_data,
        csr_values_data,
        ctypes.c_int(num_fixed),
        fixed_data,
        F_data,
        elem_vals_data,
        ctypes.c_int(1 if direct_scatter else 0),
    )

    if err != 0:
        raise RuntimeError(f"hip_assemble_beam_system_csr failed with error code {err}")

    return csr_values_c, F_c, elem_vals


def hip_spmv_bsr6x6(
    K_bsr,
    x: np.ndarray,
    y: np.ndarray | None = None,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> np.ndarray:
    """
    Execute BSR 6x6 SpMV on AMD GPU: y = alpha * K_bsr * x + beta * y.
    """
    hip_lib = _get_hip_lib()
    if hip_lib is None:
        from .bsr_matrix import bsr6x6_spmv_cpu
        y_cpu = bsr6x6_spmv_cpu(K_bsr, x, y)
        if alpha != 1.0 or beta != 0.0:
            return alpha * y_cpu + beta * (y if y is not None else 0.0)
        return y_cpu

    n_nodes = K_bsr.n_nodes
    nnz_blocks = K_bsr.nnz_blocks
    total_dofs = n_nodes * 6

    is_fp32 = (K_bsr.dtype == np.float32)
    dtype = np.float32 if is_fp32 else np.float64

    b_row_ptr_c = np.ascontiguousarray(K_bsr.block_row_ptr, dtype=np.int32)
    b_col_idx_c = np.ascontiguousarray(K_bsr.block_col_idx, dtype=np.int32)
    b_values_c = np.ascontiguousarray(K_bsr.values, dtype=dtype)

    x_c = np.ascontiguousarray(x, dtype=dtype)
    if y is None:
        y_c = np.zeros(total_dofs, dtype=dtype)
    else:
        y_c = np.ascontiguousarray(y, dtype=dtype)

    if is_fp32:
        err = hip_lib.hip_spmv_bsr6x6_fp32(
            ctypes.c_int(n_nodes),
            ctypes.c_int(nnz_blocks),
            b_row_ptr_c.ctypes.data,
            b_col_idx_c.ctypes.data,
            b_values_c.ctypes.data,
            x_c.ctypes.data,
            y_c.ctypes.data,
            ctypes.c_float(alpha),
            ctypes.c_float(beta),
            ctypes.c_int(256),
            ctypes.c_int(0),
        )
    else:
        err = hip_lib.hip_spmv_bsr6x6_fp64(
            ctypes.c_int(n_nodes),
            ctypes.c_int(nnz_blocks),
            b_row_ptr_c.ctypes.data,
            b_col_idx_c.ctypes.data,
            b_values_c.ctypes.data,
            x_c.ctypes.data,
            y_c.ctypes.data,
            ctypes.c_double(alpha),
            ctypes.c_double(beta),
            ctypes.c_int(256),
            ctypes.c_int(0),
        )

    if err != 0:
        raise RuntimeError(f"hip_spmv_bsr6x6 failed with error code {err}")

    return y_c


def hip_spmv_bsr6x6_benchmark(
    K_bsr,
    num_repeats: int = 100,
) -> tuple[float, float, float]:
    """
    Direct in-VRAM benchmark of BSR 6x6 SpMV throughput on AMD GPU.
    Returns (avg_time_ms, effective_bandwidth_gbs, throughput_gflops).
    """
    hip_lib = _get_hip_lib()
    if hip_lib is None:
        raise RuntimeError("HIP library not available for GPU benchmark.")

    n_nodes = K_bsr.n_nodes
    nnz_blocks = K_bsr.nnz_blocks

    b_row_ptr_c = np.ascontiguousarray(K_bsr.block_row_ptr, dtype=np.int32)
    b_col_idx_c = np.ascontiguousarray(K_bsr.block_col_idx, dtype=np.int32)
    b_values_c = np.ascontiguousarray(K_bsr.values, dtype=np.float32)

    out_time_ms = ctypes.c_double(0.0)
    out_bw = ctypes.c_double(0.0)
    out_gf = ctypes.c_double(0.0)

    err = hip_lib.hip_spmv_bsr6x6_benchmark(
        ctypes.c_int(n_nodes),
        ctypes.c_int(nnz_blocks),
        b_row_ptr_c.ctypes.data,
        b_col_idx_c.ctypes.data,
        b_values_c.ctypes.data,
        ctypes.c_int(num_repeats),
        ctypes.byref(out_time_ms),
        ctypes.byref(out_bw),
        ctypes.byref(out_gf),
    )

    if err != 0:
        raise RuntimeError(f"hip_spmv_bsr6x6_benchmark failed with error code {err}")

    return float(out_time_ms.value), float(out_bw.value), float(out_gf.value)


def hip_c3d10_matrix_free_matvec(
    nodes: np.ndarray,
    elements: np.ndarray,
    props: np.ndarray,
    u: np.ndarray,
    fixed_dofs: np.ndarray | None = None,
    precision: str = "fp64",
) -> np.ndarray | None:
    """
    Execute Matrix-Free C3D10 3-DOF matrix-vector product v = K @ u directly on AMD GPU.
    
    Parameters
    ----------
    nodes : (n_nodes, 3) float64 array of nodal coordinates.
    elements : (n_solids, 10) int32 array of C3D10 connectivity.
    props : (n_solids, 2) float64 array of material properties [E, nu].
    u : (n_nodes * 3,) input vector.
    fixed_dofs : Optional 1D int32 array of fixed Dirichlet DOFs.
    precision : "fp64" (double precision) or "fp32" (single precision).

    Returns
    -------
    v : (n_nodes * 3,) output product vector, or None if HIP is unavailable.
    """
    hip_lib = _get_hip_lib()
    if hip_lib is None:
        return None

    n_nodes = nodes.shape[0]
    n_solids = elements.shape[0]
    total_dofs = n_nodes * 3

    if fixed_dofs is not None and len(fixed_dofs) > 0:
        h_fixed = np.ascontiguousarray(fixed_dofs, dtype=np.int32)
        n_fixed = len(h_fixed)
        p_fixed = h_fixed.ctypes.data
    else:
        p_fixed = None
        n_fixed = 0

    if precision == "fp32":
        h_nodes = np.ascontiguousarray(nodes, dtype=np.float32)
        h_elems = np.ascontiguousarray(elements, dtype=np.int32)
        h_props = np.ascontiguousarray(props, dtype=np.float32)
        h_u = np.ascontiguousarray(u, dtype=np.float32)
        h_v = np.zeros(total_dofs, dtype=np.float32)

        err = hip_lib.hip_c3d10_matrix_free_matvec_fp32(
            h_nodes.ctypes.data,
            h_elems.ctypes.data,
            h_props.ctypes.data,
            h_u.ctypes.data,
            h_v.ctypes.data,
            p_fixed,
            ctypes.c_int(n_fixed),
            ctypes.c_int(n_nodes),
            ctypes.c_int(n_solids),
        )
        if err != 0:
            return None
        return h_v.astype(u.dtype)
    else:
        h_nodes = np.ascontiguousarray(nodes, dtype=np.float64)
        h_elems = np.ascontiguousarray(elements, dtype=np.int32)
        h_props = np.ascontiguousarray(props, dtype=np.float64)
        h_u = np.ascontiguousarray(u, dtype=np.float64)
        h_v = np.zeros(total_dofs, dtype=np.float64)

        err = hip_lib.hip_c3d10_matrix_free_matvec_fp64(
            h_nodes.ctypes.data,
            h_elems.ctypes.data,
            h_props.ctypes.data,
            h_u.ctypes.data,
            h_v.ctypes.data,
            p_fixed,
            ctypes.c_int(n_fixed),
            ctypes.c_int(n_nodes),
            ctypes.c_int(n_solids),
        )
        if err != 0:
            return None
        return h_v


def compute_c3d10_diagonal_fast(
    nodes: np.ndarray,
    elements: np.ndarray,
    props: np.ndarray,
    diag_out: np.ndarray | None = None,
) -> np.ndarray:
    """
    High-performance exact stiffness diagonal computation for C3D10 solid elements.
    Uses native C++ SIMD kernel when available, with scalar/chunked fallback.

    Args:
        nodes: (N, 3) float64 array of nodal coordinates.
        elements: (E, 10) int32 array of element node indices.
        props: (E, 2) float64 array of element material properties [E, nu].
        diag_out: (N * 3,) float64 array (optional, allocated if None).

    Returns:
        diag: (N * 3,) exact stiffness diagonal.
    """
    n_nodes = len(nodes)
    n_solids = len(elements)
    total_dofs = n_nodes * 3

    if diag_out is None:
        diag = np.zeros(total_dofs, dtype=np.float64)
    else:
        diag = diag_out
        diag.fill(0.0)

    lib = _get_native_lib()
    if lib is not None and hasattr(lib, "compute_c3d10_diagonal_native"):
        h_nodes = np.ascontiguousarray(nodes, dtype=np.float64)
        h_elems = np.ascontiguousarray(elements, dtype=np.int32)
        h_props = np.ascontiguousarray(props, dtype=np.float64)
        h_diag = np.ascontiguousarray(diag, dtype=np.float64)

        lib.compute_c3d10_diagonal_native(
            h_nodes.ctypes.data,
            h_elems.ctypes.data,
            h_props.ctypes.data,
            h_diag.ctypes.data,
            ctypes.c_int(n_solids),
            ctypes.c_int(n_nodes),
        )
        if diag_out is not None and diag_out is not h_diag:
            diag_out[:] = h_diag
        return h_diag

    # Fallback: elemental loop
    from ..elements.c3d10 import element_stiffness_c3d10
    elem_dofs = np.zeros((n_solids, 30), dtype=np.int64)
    for i in range(10):
        elem_dofs[:, i * 3 + 0] = elements[:, i] * 3 + 0
        elem_dofs[:, i * 3 + 1] = elements[:, i] * 3 + 1
        elem_dofs[:, i * 3 + 2] = elements[:, i] * 3 + 2

    for e_idx in range(n_solids):
        coords = nodes[elements[e_idx]]
        Ke = element_stiffness_c3d10(coords, props[e_idx, 0], props[e_idx, 1])
        np.add.at(diag, elem_dofs[e_idx], np.diag(Ke))

    return diag
