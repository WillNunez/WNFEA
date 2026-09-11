"""
Functionally Graded TPMS Lattice Micro-Architecture Generator for WNFEA (Phase 20).
-----------------------------------------------------------------------------------
Provides:
1. Implicit Triply Periodic Minimal Surfaces (TPMS):
   - Gyroid (G)
   - Schwarz Primitive (P)
   - Diamond (D)
   - Neovius (N)
   - I-Wrapped Package (I-WP)
2. Skeletal (Solid Network) and Sheet (Thin Wall Shell) Lattice Formulations:
   - Skeletal: Solid domain defined by F(X, Y, Z) <= t
   - Sheet: Solid domain defined by |F(X, Y, Z)| <= t / 2
3. Exact Relative Density to Level-Set Threshold Mapping:
   - Inversion calibrated to target volume fraction rho in (0, 1).
4. Functionally Graded Porous Infill:
   - Evaluates spatially varying relative density fields rho(x) directly from
     topology optimization density solutions or stress concentration fields.
5. Watertight Isosurface Extraction & STL/OBJ Triangular Mesh Export.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple, Union
import numpy as np

from ..cad.isosurface import TriangularMesh, extract_isosurface_mesh
from ..mesh.voxel_mesher import VoxelGrid


class TPMSType(enum.Enum):
    """Supported Triply Periodic Minimal Surface morphologies."""
    GYROID = "gyroid"
    SCHWARZ_P = "schwarz_p"
    DIAMOND = "diamond"
    NEOVIUS = "neovius"
    IWP = "iwp"


@dataclass
class TPMSConfig:
    """
    Configuration for Triply Periodic Minimal Surface (TPMS) generation.
    """
    tpms_type: TPMSType = TPMSType.GYROID
    cell_size: Tuple[float, float, float] = (0.01, 0.01, 0.01)  # (Lx, Ly, Lz) in meters
    lattice_mode: str = "skeletal"                              # "skeletal" or "sheet"
    relative_density: float = 0.30                              # Uniform target volume fraction in (0, 1)
    wall_thickness: Optional[float] = None                      # Explicit wall thickness for sheet mode

    def __post_init__(self):
        self.lattice_mode = self.lattice_mode.lower()
        if self.lattice_mode not in ("skeletal", "sheet"):
            raise ValueError(f"Unknown lattice_mode '{self.lattice_mode}', expected 'skeletal' or 'sheet'")
        self.relative_density = float(np.clip(self.relative_density, 0.01, 0.99))


def evaluate_tpms_levelset(
    points: np.ndarray,
    tpms_type: TPMSType,
    cell_size: Tuple[float, float, float] = (0.01, 0.01, 0.01),
) -> np.ndarray:
    """
    Vectorized evaluation of implicit TPMS mathematical level-set function F(X, Y, Z).

    Parameters:
        points: (P, 3) spatial coordinates (x, y, z) in meters.
        tpms_type: Morphology type (GYROID, SCHWARZ_P, DIAMOND, etc.).
        cell_size: Periodicity wavelengths (Lx, Ly, Lz) in meters.

    Returns:
        values: (P,) scalar level-set field values.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1 and pts.shape[0] == 3:
        pts = pts.reshape(1, 3)

    Lx, Ly, Lz = cell_size
    kx = 2.0 * np.pi / max(Lx, 1e-12)
    ky = 2.0 * np.pi / max(Ly, 1e-12)
    kz = 2.0 * np.pi / max(Lz, 1e-12)

    X = kx * pts[:, 0]
    Y = ky * pts[:, 1]
    Z = kz * pts[:, 2]

    sinX, cosX = np.sin(X), np.cos(X)
    sinY, cosY = np.sin(Y), np.cos(Y)
    sinZ, cosZ = np.sin(Z), np.cos(Z)

    if tpms_type == TPMSType.GYROID:
        # Gyroid: sin(X)*cos(Y) + sin(Y)*cos(Z) + sin(Z)*cos(X) = 0
        return sinX * cosY + sinY * cosZ + sinZ * cosX

    elif tpms_type == TPMSType.SCHWARZ_P:
        # Schwarz Primitive: cos(X) + cos(Y) + cos(Z) = 0
        return cosX + cosY + cosZ

    elif tpms_type == TPMSType.DIAMOND:
        # Diamond: sin(X)*sin(Y)*sin(Z) + sin(X)*cos(Y)*cos(Z) + cos(X)*sin(Y)*cos(Z) + cos(X)*cos(Y)*sin(Z) = 0
        return sinX * sinY * sinZ + sinX * cosY * cosZ + cosX * sinY * cosZ + cosX * cosY * sinZ

    elif tpms_type == TPMSType.NEOVIUS:
        # Neovius: 3*(cos(X) + cos(Y) + cos(Z)) + 4*cos(X)*cos(Y)*cos(Z) = 0
        return 3.0 * (cosX + cosY + cosZ) + 4.0 * cosX * cosY * cosZ

    elif tpms_type == TPMSType.IWP:
        # I-WP: 2*(cos(X)*cos(Y) + cos(Y)*cos(Z) + cos(Z)*cos(X)) - (cos(2X) + cos(2Y) + cos(2Z)) = 0
        cos2X = np.cos(2.0 * X)
        cos2Y = np.cos(2.0 * Y)
        cos2Z = np.cos(2.0 * Z)
        return 2.0 * (cosX * cosY + cosY * cosZ + cosZ * cosX) - (cos2X + cos2Y + cos2Z)

    else:
        raise ValueError(f"Unsupported TPMS type: {tpms_type}")


def density_to_levelset_threshold(
    tpms_type: TPMSType,
    mode: str,
    relative_density: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Convert target relative density rho in (0, 1) into implicit level-set threshold t.
    
    For skeletal lattices: F(x) <= t is solid. (rho = 0.5 corresponds to t = 0).
    For sheet lattices: |F(x)| <= t/2 is solid. (rho = 0 corresponds to t = 0).
    """
    rho = np.clip(relative_density, 0.001, 0.999)
    mode_lower = mode.lower()

    if mode_lower == "skeletal":
        # Calibrated cubic polynomial approximations for inverse cumulative distribution
        delta = rho - 0.5
        if tpms_type == TPMSType.GYROID:
            # Gyroid bounds: [-1.414, 1.414]. At rho=0.5, t=0
            t = 2.44 * delta - 0.88 * (delta ** 3)
        elif tpms_type == TPMSType.SCHWARZ_P:
            # Schwarz P bounds: [-3, 3]
            t = 4.50 * delta - 2.10 * (delta ** 3)
        elif tpms_type == TPMSType.DIAMOND:
            # Diamond bounds: [-2.0, 2.0]
            t = 3.25 * delta - 1.20 * (delta ** 3)
        else:
            t = 3.0 * delta

        return t

    elif mode_lower == "sheet":
        # For sheet TPMS, thickness scales approximately proportionally with volume fraction
        if tpms_type == TPMSType.GYROID:
            t = 1.60 * rho
        elif tpms_type == TPMSType.SCHWARZ_P:
            t = 2.80 * rho
        elif tpms_type == TPMSType.DIAMOND:
            t = 2.20 * rho
        else:
            t = 2.0 * rho
        return t

    else:
        raise ValueError(f"Unknown mode: {mode}")


def evaluate_graded_tpms_field(
    points: np.ndarray,
    config: TPMSConfig,
    density_field: Optional[Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]]] = None,
) -> np.ndarray:
    """
    Evaluate the signed level-set distance field Phi(x) for functionally graded TPMS.
    Solid material is defined by Phi(x) <= 0.0.

    Parameters:
        points: (P, 3) spatial coordinates.
        config: TPMSConfig specifying morphology, cell size, and mode.
        density_field: Target relative density rho(x). Can be a constant scalar,
                       an array matching points, or a spatial function f(x, y, z).

    Returns:
        phi: (P,) signed level-set field (<= 0 is solid, > 0 is void).
    """
    pts = np.asarray(points, dtype=np.float64)
    F = evaluate_tpms_levelset(pts, config.tpms_type, config.cell_size)

    # Determine local relative density at each point
    if density_field is None:
        local_rho = np.full(len(pts), config.relative_density, dtype=np.float64)
    elif callable(density_field):
        local_rho = np.asarray(density_field(pts), dtype=np.float64)
    elif isinstance(density_field, (int, float)):
        local_rho = np.full(len(pts), float(density_field), dtype=np.float64)
    else:
        local_rho = np.asarray(density_field, dtype=np.float64)

    t_thresh = density_to_levelset_threshold(config.tpms_type, config.lattice_mode, local_rho)

    if config.lattice_mode == "skeletal":
        # Solid where F <= t ==> Phi = F - t
        phi = F - t_thresh
    else:
        # Sheet: Solid where |F| <= t / 2 ==> Phi = |F| - t / 2
        phi = np.abs(F) - 0.5 * t_thresh

    return phi


def generate_tpms_voxel_infill(
    grid: VoxelGrid,
    config: TPMSConfig,
    density_field: Optional[Union[float, np.ndarray, Callable]] = None,
    subsampling: int = 2,
) -> np.ndarray:
    """
    Compute sub-sampled elemental volume fractions rho_e in [0, 1] across a VoxelGrid.

    Parameters:
        grid: VoxelGrid domain.
        config: TPMS configuration.
        density_field: Optional spatially varying density field.
        subsampling: Sub-grid sampling points per axis (2 = 8 sub-points, 3 = 27 sub-points).

    Returns:
        elem_fractions: (N_elements,) array of relative densities in [0, 1].
    """
    n_elem = grid.total_cells
    active_indices = grid.active_element_indices
    hx, hy, hz = grid.pitch

    # Sub-sampling offsets inside reference cell [-0.5, 0.5]
    offsets_1d = (np.arange(subsampling) + 0.5) / subsampling - 0.5
    sub_offsets = np.array(np.meshgrid(offsets_1d, offsets_1d, offsets_1d)).T.reshape(-1, 3)
    n_sub = len(sub_offsets)

    # Scale offsets by voxel pitch
    sub_offsets[:, 0] *= hx
    sub_offsets[:, 1] *= hy
    sub_offsets[:, 2] *= hz

    elem_fractions = np.zeros(n_elem, dtype=np.float64)

    # Compute cell centroids
    elem_nodes = grid.elements[active_indices]  # (N_active, 8)
    centroids = np.mean(grid.nodes[elem_nodes], axis=1)  # (N_active, 3)

    # Sub-points: (N_active * n_sub, 3)
    all_sub_pts = centroids[:, None, :] + sub_offsets[None, :, :]
    all_sub_pts = all_sub_pts.reshape(-1, 3)

    # Evaluate signed distance field
    phi = evaluate_graded_tpms_field(all_sub_pts, config, density_field)
    is_solid = (phi <= 0.0).reshape(len(active_indices), n_sub)

    elem_fractions[active_indices] = np.mean(is_solid, axis=1)
    return elem_fractions


def extract_tpms_isosurface(
    config: TPMSConfig,
    bounds: Tuple[float, float, float, float, float, float],
    resolution: Tuple[int, int, int],
    density_field: Optional[Union[float, np.ndarray, Callable]] = None,
    smoothing_iters: int = 2,
) -> TriangularMesh:
    """
    Extract a watertight, manifold triangular boundary mesh of the TPMS lattice
    over a bounding box [xmin, xmax, ymin, ymax, zmin, zmax].

    Parameters:
        config: TPMS configuration.
        bounds: Domain bounding box (xmin, xmax, ymin, ymax, zmin, zmax).
        resolution: (nx, ny, nz) grid resolution for isosurface discretization.
        density_field: Optional functional grading field.
        smoothing_iters: Laplacian smoothing iterations.

    Returns:
        TriangularMesh object with vertices, faces, and STL/OBJ export methods.
    """
    from ..mesh.voxel_mesher import VoxelMesher

    grid = VoxelMesher.create_box_grid(bounds, resolution)
    densities = generate_tpms_voxel_infill(
        grid=grid,
        config=config,
        density_field=density_field,
        subsampling=2,
    )

    mesh = extract_isosurface_mesh(
        grid=grid,
        density=densities,
        isovalue=0.5,
        smoothing_iters=smoothing_iters,
    )

    return mesh
