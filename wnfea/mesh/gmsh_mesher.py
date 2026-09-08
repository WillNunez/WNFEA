"""
Gmsh STEP Meshing Harness for WNFEA.

Interfaces with open-source Gmsh (OpenCASCADE kernel) to import STEP CAD files
(.stp / .step) and generate 3D continuum meshes composed of 10-node quadratic
tetrahedral elements (C3D10).

Node ordering from Gmsh Type 11 is automatically permuted to match the industry-standard
Abaqus C3D10 and VTK_QUADRATIC_TETRA specifications:
  Vertices:  0, 1, 2, 3
  Mid-edges: 4 (0-1), 5 (1-2), 6 (2-0), 7 (0-3), 8 (1-3), 9 (2-3)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable
import numpy as np

from ..model import FEAModel
from ..properties.materials import MaterialDef


class GmshError(Exception):
    """Raised when Gmsh meshing or CAD import fails."""
    pass


class GmshMesher:
    """
    Harness for importing STEP CAD models and meshing with Gmsh into
    C3D10 10-node quadratic tetrahedral elements.

    Parameters
    ----------
    mesh_size_max : float | None
        Maximum element characteristic length (Mesh.MeshSizeMax).
    mesh_size_min : float | None
        Minimum element characteristic length (Mesh.MeshSizeMin).
    element_order : int
        Polynomial order of elements (default 2 for quadratic C3D10).
    algorithm : int
        3D meshing algorithm (1: Delaunay, 4: Frontal, 10: HXT parallel).
    optimize : bool
        Whether to run 3D mesh quality optimization (Netgen optimizer).
    verbose : bool
        If True, displays Gmsh stdout/stderr output.
    """

    def __init__(
        self,
        mesh_size_max: float | None = None,
        mesh_size_min: float | None = None,
        element_order: int = 2,
        algorithm: int = 1,
        optimize: bool = True,
        verbose: bool = False,
    ):
        self.mesh_size_max = mesh_size_max
        self.mesh_size_min = mesh_size_min
        self.element_order = element_order
        self.algorithm = algorithm
        self.optimize = optimize
        self.verbose = verbose

    def _init_gmsh(self) -> None:
        """Initialize Gmsh runtime environment."""
        try:
            import gmsh
        except ImportError as e:
            raise GmshError(
                "Gmsh Python API is not installed. Install via `pip install gmsh`."
            ) from e

        if not gmsh.isInitialized():
            try:
                gmsh.initialize(interruptible=False)
            except TypeError:
                gmsh.initialize()

        # Set output verbosity
        gmsh.option.setNumber("General.Terminal", 1 if self.verbose else 0)

    def _configure_options(self) -> None:
        """Apply mesh generation options."""
        import gmsh

        gmsh.option.setNumber("Mesh.ElementOrder", self.element_order)
        gmsh.option.setNumber("Mesh.Algorithm3D", self.algorithm)

        if self.mesh_size_max is not None:
            gmsh.option.setNumber("Mesh.MeshSizeMax", float(self.mesh_size_max))
        if self.mesh_size_min is not None:
            gmsh.option.setNumber("Mesh.MeshSizeMin", float(self.mesh_size_min))

        if self.element_order == 2:
            # Generate second-order curvilinear nodes
            gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)

    def mesh_file(
        self,
        step_path: str | Path,
        model: FEAModel | None = None,
        material_name: str = "Steel",
    ) -> tuple[np.ndarray, np.ndarray, FEAModel]:
        """
        Import a STEP CAD file and generate a 3D C3D10 tetrahedral mesh.

        Parameters
        ----------
        step_path : str | Path
            Path to the .stp or .step file.
        model : FEAModel | None
            Optional FEAModel to populate. If None, a new FEAModel is created.
        material_name : str
            Material name to assign to the solid elements.

        Returns
        -------
        nodes : np.ndarray
            Nodal coordinates array of shape (N, 3).
        elements : np.ndarray
            C3D10 element connectivity array of shape (M, 10).
        model : FEAModel
            The populated FEAModel instance.
        """
        step_str = str(Path(step_path).resolve())
        if not os.path.exists(step_str):
            raise FileNotFoundError(f"STEP file not found: {step_str}")

        import gmsh

        self._init_gmsh()
        try:
            gmsh.clear()
            gmsh.model.add("STEP_Solid")

            # Open file via Gmsh
            gmsh.open(step_str)
            self._configure_options()

            # Generate 3D volume mesh
            gmsh.model.mesh.generate(3)

            if self.optimize:
                try:
                    gmsh.model.mesh.optimize("Netgen")
                except Exception:
                    # Fallback to default optimizer if Netgen not compiled in
                    gmsh.model.mesh.optimize("")

            if self.element_order > 1:
                gmsh.model.mesh.setOrder(self.element_order)

            nodes, elements = self._extract_mesh()
        finally:
            gmsh.finalize()

        if model is None:
            model = FEAModel()

        self._populate_model(model, nodes, elements, material_name, source_file=step_str)
        return nodes, elements, model

    def mesh_cad(
        self,
        cad_builder: Callable[[object], None],
        model: FEAModel | None = None,
        material_name: str = "Steel",
    ) -> tuple[np.ndarray, np.ndarray, FEAModel]:
        """
        Generate a 3D C3D10 tetrahedral mesh from programmatic OpenCASCADE CAD geometry.

        Parameters
        ----------
        cad_builder : Callable[[gmsh.model.occ], None]
            Function that accepts gmsh.model.occ and creates solid geometry.
        model : FEAModel | None
            Optional FEAModel to populate.
        material_name : str
            Material name to assign to the solid elements.
        """
        import gmsh

        self._init_gmsh()
        try:
            gmsh.clear()
            gmsh.model.add("OCC_Solid")

            # User CAD construction
            cad_builder(gmsh.model.occ)
            gmsh.model.occ.synchronize()

            self._configure_options()
            gmsh.model.mesh.generate(3)

            if self.optimize:
                try:
                    gmsh.model.mesh.optimize("Netgen")
                except Exception:
                    gmsh.model.mesh.optimize("")

            if self.element_order > 1:
                gmsh.model.mesh.setOrder(self.element_order)

            nodes, elements = self._extract_mesh()
        finally:
            gmsh.finalize()

        if model is None:
            model = FEAModel()

        self._populate_model(model, nodes, elements, material_name, source_file="CAD_Builder")
        return nodes, elements, model

    def _extract_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract mesh nodes and permute 10-node tetrahedra into Abaqus C3D10 convention.
        """
        import gmsh

        # Extract all nodes
        node_tags, node_coords_flat, _ = gmsh.model.mesh.getNodes()
        if len(node_tags) == 0:
            raise GmshError("Gmsh generated 0 nodes. Ensure the CAD model contains solid 3D volumes.")

        # Reshape coordinates to (N, 3)
        coords = np.array(node_coords_flat, dtype=np.float64).reshape(-1, 3)

        # Build contiguous 0-based indexing map
        tag_to_idx = {int(tag): idx for idx, tag in enumerate(node_tags)}

        # Extract 3D solid elements
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=3)

        c3d10_list: list[np.ndarray] = []

        for etype, tags, ntags in zip(elem_types, elem_tags, elem_node_tags):
            # Gmsh Type 11 is 10-node 2nd-order tetrahedron
            if etype == 11:
                n_elem = len(tags)
                raw_conn = np.array([tag_to_idx[int(t)] for t in ntags], dtype=np.int64).reshape(n_elem, 10)

                # Permute Gmsh Type 11 mid-edge nodes (8 <-> 9) to match Abaqus C3D10 / VTK convention:
                # Gmsh: Node 8 is edge 2-3, Node 9 is edge 1-3.
                # Abaqus / VTK: Node 8 is edge 1-3, Node 9 is edge 2-3.
                permuted_conn = raw_conn.copy()
                permuted_conn[:, [8, 9]] = permuted_conn[:, [9, 8]]
                c3d10_list.append(permuted_conn)

        if not c3d10_list:
            raise GmshError(
                f"No 10-node tetrahedral elements (Gmsh type 11) generated. Found element types: {elem_types}"
            )

        elements = np.vstack(c3d10_list)
        return coords, elements

    def _populate_model(
        self,
        model: FEAModel,
        nodes: np.ndarray,
        elements: np.ndarray,
        material_name: str,
        source_file: str,
    ) -> None:
        """Store nodes and solid elements onto the FEAModel."""
        model.clear_mesh()
        model.mesh_nodes = nodes
        model.solid_elements = elements
        model.source_file = source_file

        # Assign solid material for each element
        for elem_id in range(len(elements)):
            model.solid_materials[elem_id] = material_name

        # Ensure material exists in model
        if material_name not in model.materials:
            # Default structural steel
            model.materials[material_name] = MaterialDef(
                name=material_name,
                youngs_modulus=210.0e9,
                poissons_ratio=0.3,
                yield_strength=250.0e6,
            )

    @staticmethod
    def get_nodes_in_box(
        nodes: np.ndarray,
        xmin: float = -np.inf,
        xmax: float = np.inf,
        ymin: float = -np.inf,
        ymax: float = np.inf,
        zmin: float = -np.inf,
        zmax: float = np.inf,
        tol: float = 1e-5,
    ) -> np.ndarray:
        """
        Find indices of nodes lying inside or on a 3D bounding box.
        Useful for applying boundary conditions and loads to CAD regions.
        """
        mask = (
            (nodes[:, 0] >= xmin - tol) & (nodes[:, 0] <= xmax + tol) &
            (nodes[:, 1] >= ymin - tol) & (nodes[:, 1] <= ymax + tol) &
            (nodes[:, 2] >= zmin - tol) & (nodes[:, 2] <= zmax + tol)
        )
        return np.where(mask)[0]

    @staticmethod
    def get_nodes_on_plane(
        nodes: np.ndarray,
        axis: str = "x",
        coord: float = 0.0,
        tol: float = 1e-5,
    ) -> np.ndarray:
        """
        Find indices of nodes lying on a coordinate plane (e.g. x=0 or z=L).
        """
        ax_map = {"x": 0, "y": 1, "z": 2}
        col = ax_map[axis.lower()]
        mask = np.abs(nodes[:, col] - coord) <= tol
        return np.where(mask)[0]
