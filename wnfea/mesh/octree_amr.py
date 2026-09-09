"""
Hierarchical 1-Irregular Octree Adaptive Mesh Refinement (AMR) for WNFEA.
-------------------------------------------------------------------------
Implements localized geometric mesh refinement around stress hotspots:
1. 1-Irregular (2:1 Balanced) Octree Hierarchy:
   - Eliminates adjacent level jumps > 1 to maintain optimal element aspect ratios.
2. Hanging-Node Multi-Point Constraint (MPC) Elimination:
   - Automatically detects edge hanging vertices (w = 0.5, 0.5) and face hanging vertices (w = 0.25, 0.25, 0.25, 0.25).
   - Generates sparse constraint prolongation matrix C: u_all = C @ u_true, enforcing exact C^0 inter-element continuity.
3. Stress-Driven Adaptive Refinement:
   - Identifies high-stress elements from voxel / FEA solves and refines localized zones by 2x, 4x, or 8x.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, List, Dict, Tuple, Set
import numpy as np
from scipy.sparse import csr_matrix

from .voxel_mesher import VoxelGrid


@dataclass
class OctreeCell:
    """Represents a single cell in the adaptive octree hierarchy."""
    cell_id: int
    level: int
    bounds: Tuple[float, float, float, float, float, float]  # (xmin, xmax, ymin, ymax, zmin, zmax)
    centroid: np.ndarray                                     # (3,)
    is_leaf: bool = True
    parent_id: Optional[int] = None
    children_ids: Optional[List[int]] = None
    corner_node_ids: Optional[np.ndarray] = None             # (8,) global node IDs when leaf


@dataclass
class AMRMesh:
    """Conforming adaptive mesh with hanging-node MPC constraint structure."""
    nodes: np.ndarray                       # (N_nodes, 3) All nodal coordinates (including hanging)
    elements: np.ndarray                    # (N_leaf_elems, 8) Hex8 element connectivity
    leaf_levels: np.ndarray                 # (N_leaf_elems,) Refinement level of each leaf cell
    independent_node_indices: np.ndarray    # (N_true,) Nodes that are independent
    hanging_node_indices: np.ndarray        # (N_hanging,) Nodes that are constrained
    constraint_matrix: csr_matrix           # (N_nodes, N_true) C such that u = C @ u_true

    @property
    def total_nodes(self) -> int:
        return len(self.nodes)

    @property
    def total_true_nodes(self) -> int:
        return len(self.independent_node_indices)

    @property
    def total_elements(self) -> int:
        return len(self.elements)


class OctreeAMRMesher:
    """
    Constructs and refines adaptive octree meshes with 1-irregular balancing
    and automatic linear constraint elimination.
    """
    def __init__(self, root_grid: VoxelGrid):
        self.root_grid = root_grid
        self.cells: List[OctreeCell] = []
        self._init_from_grid(root_grid)

    def _init_from_grid(self, grid: VoxelGrid):
        """Initialize level 0 octree cells from root VoxelGrid."""
        nx, ny, nz = grid.resolution
        hx, hy, hz = grid.pitch
        xmin, _, ymin, _, zmin, _ = grid.bounds

        self.cells = []
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    cid = grid.cell_id(i, j, k)
                    x0 = xmin + i * hx
                    y0 = ymin + j * hy
                    z0 = zmin + k * hz
                    b = (x0, x0 + hx, y0, y0 + hy, z0, z0 + hz)
                    c = np.array([x0 + 0.5 * hx, y0 + 0.5 * hy, z0 + 0.5 * hz])
                    cell = OctreeCell(
                        cell_id=cid,
                        level=0,
                        bounds=b,
                        centroid=c,
                        is_leaf=True,
                        parent_id=None,
                        children_ids=None,
                        corner_node_ids=grid.elements[cid].copy(),
                    )
                    self.cells.append(cell)

    def get_leaf_cells(self) -> List[OctreeCell]:
        """Return all currently active leaf cells."""
        return [c for c in self.cells if c.is_leaf]

    def refine_cells(self, cell_ids_to_refine: Sequence[int], enforce_2to1_balance: bool = True) -> int:
        """
        Subdivide designated leaf cells into 8 child sub-cells each.
        Optionally balances the octree so adjacent cells never differ by more than 1 level.

        Returns:
            num_refined: Number of cells subdivided.
        """
        target_ids = set(cell_ids_to_refine)
        leaf_map = {c.cell_id: c for c in self.cells if c.is_leaf}

        # Filter to valid leaf cells
        candidates = [cid for cid in target_ids if cid in leaf_map]

        if enforce_2to1_balance:
            # 2:1 balancing: find any neighbors with level jump > 1 and mark them
            added = True
            while added:
                added = False
                current_marked = set(candidates)
                all_leaves = [c for c in self.cells if c.is_leaf]
                for leaf in all_leaves:
                    if leaf.cell_id in current_marked:
                        continue
                    # Check spatial adjacency against marked cells
                    for marked_id in list(current_marked):
                        marked_cell = self.cells[marked_id]
                        if leaf.level < marked_cell.level:
                            # Distance between centroids
                            max_dist = np.max(np.abs(leaf.centroid - marked_cell.centroid))
                            leaf_h = (leaf.bounds[1] - leaf.bounds[0])
                            if max_dist <= 1.05 * leaf_h:
                                candidates.append(leaf.cell_id)
                                current_marked.add(leaf.cell_id)
                                added = True
                                break

        # Execute subdivision
        subdivided_count = 0
        for cid in candidates:
            parent = self.cells[cid]
            if not parent.is_leaf:
                continue

            parent.is_leaf = False
            x0, x1, y0, y1, z0, z1 = parent.bounds
            hx = (x1 - x0) / 2.0
            hy = (y1 - y0) / 2.0
            hz = (z1 - z0) / 2.0

            child_ids = []
            for ck in range(2):
                for cj in range(2):
                    for ci in range(2):
                        child_cid = len(self.cells)
                        cx0 = x0 + ci * hx
                        cy0 = y0 + cj * hy
                        cz0 = z0 + ck * hz
                        child_bounds = (cx0, cx0 + hx, cy0, cy0 + hy, cz0, cz0 + hz)
                        child_cent = np.array([cx0 + 0.5 * hx, cy0 + 0.5 * hy, cz0 + 0.5 * hz])

                        child_cell = OctreeCell(
                            cell_id=child_cid,
                            level=parent.level + 1,
                            bounds=child_bounds,
                            centroid=child_cent,
                            is_leaf=True,
                            parent_id=parent.cell_id,
                            children_ids=None,
                        )
                        self.cells.append(child_cell)
                        child_ids.append(child_cid)

            parent.children_ids = child_ids
            subdivided_count += 1

        return subdivided_count

    def build_conforming_mesh(self) -> AMRMesh:
        """
        Assemble all active leaf cells into a unified global node mesh,
        detecting hanging nodes and generating the MPC constraint matrix C.
        """
        leaves = self.get_leaf_cells()
        n_leaves = len(leaves)

        # 1. Generate 8 corner coordinates for every leaf
        # Standard Hex8 node order:
        # Bottom: (0,0,0), (1,0,0), (1,1,0), (0,1,0)
        # Top:    (0,0,1), (1,0,1), (1,1,1), (0,1,1)
        corners_per_cell = np.zeros((n_leaves, 8, 3), dtype=np.float64)
        levels = np.zeros(n_leaves, dtype=np.int32)

        for idx, leaf in enumerate(leaves):
            x0, x1, y0, y1, z0, z1 = leaf.bounds
            levels[idx] = leaf.level
            corners_per_cell[idx] = np.array([
                [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
            ], dtype=np.float64)

        # 2. Merge duplicate vertices to construct global node array
        raw_nodes = corners_per_cell.reshape(-1, 3)
        snapped = np.round(raw_nodes, decimals=7)
        unique_nodes, inverse_idx = np.unique(snapped, axis=0, return_inverse=True)
        elements = inverse_idx.reshape(n_leaves, 8)
        n_nodes = len(unique_nodes)

        # Store corner node IDs on leaves
        for idx, leaf in enumerate(leaves):
            leaf.corner_node_ids = elements[idx]

        # 3. Detect Hanging Nodes (Edge and Face midpoints of coarse cells)
        # Any node that lies strictly inside a coarse edge or quad face of another cell is hanging.
        hanging_map: Dict[int, Dict[int, float]] = {}  # hanging_nid -> {coarse_nid: weight}

        # Build lookup of coarse edges and faces for level L cells
        for leaf in leaves:
            l = leaf.level
            c_nodes = leaf.corner_node_ids
            # Check against finer cells: if a node lies on one of leaf's 12 edges
            edges = [
                (c_nodes[0], c_nodes[1]), (c_nodes[1], c_nodes[2]), (c_nodes[2], c_nodes[3]), (c_nodes[3], c_nodes[0]),
                (c_nodes[4], c_nodes[5]), (c_nodes[5], c_nodes[6]), (c_nodes[6], c_nodes[7]), (c_nodes[7], c_nodes[4]),
                (c_nodes[0], c_nodes[4]), (c_nodes[1], c_nodes[5]), (c_nodes[2], c_nodes[6]), (c_nodes[3], c_nodes[7]),
            ]
            for n1, n2 in edges:
                p1 = unique_nodes[n1]
                p2 = unique_nodes[n2]
                mid = 0.5 * (p1 + p2)
                # Check if there is a node at mid
                mid_snapped = np.round(mid, decimals=7)
                match = np.where(np.all(np.abs(snapped - mid_snapped) < 1e-6, axis=1))[0]
                if len(match) > 0:
                    mid_nid = inverse_idx[match[0]]
                    if mid_nid != n1 and mid_nid != n2:
                        hanging_map[mid_nid] = {n1: 0.5, n2: 0.5}

            # Check 6 faces for face midpoints
            faces = [
                (c_nodes[0], c_nodes[1], c_nodes[2], c_nodes[3]),  # bottom
                (c_nodes[4], c_nodes[5], c_nodes[6], c_nodes[7]),  # top
                (c_nodes[0], c_nodes[1], c_nodes[5], c_nodes[4]),  # front
                (c_nodes[3], c_nodes[2], c_nodes[6], c_nodes[7]),  # back
                (c_nodes[0], c_nodes[3], c_nodes[7], c_nodes[4]),  # left
                (c_nodes[1], c_nodes[2], c_nodes[6], c_nodes[5]),  # right
            ]
            for f_nodes in faces:
                pts = unique_nodes[list(f_nodes)]
                f_mid = np.mean(pts, axis=0)
                f_mid_snapped = np.round(f_mid, decimals=7)
                match = np.where(np.all(np.abs(snapped - f_mid_snapped) < 1e-6, axis=1))[0]
                if len(match) > 0:
                    mid_nid = inverse_idx[match[0]]
                    if mid_nid not in f_nodes:
                        hanging_map[mid_nid] = {fn: 0.25 for fn in f_nodes}

        hanging_nodes = sorted(list(hanging_map.keys()))
        hanging_set = set(hanging_nodes)
        independent_nodes = [nid for nid in range(n_nodes) if nid not in hanging_set]

        # Map independent nodes to column indices in C
        indep_col_map = {nid: col for col, nid in enumerate(independent_nodes)}
        n_true = len(independent_nodes)

        # 4. Construct Sparse Constraint Matrix C: u_all = C @ u_true
        row_ind = []
        col_ind = []
        data_vals = []

        # Independent nodes: identity mapping
        for nid in independent_nodes:
            row_ind.append(nid)
            col_ind.append(indep_col_map[nid])
            data_vals.append(1.0)

        # Hanging nodes: linear combination of parent independent nodes
        for h_nid, weights in hanging_map.items():
            for parent_nid, w in weights.items():
                # If parent is itself independent
                if parent_nid in indep_col_map:
                    row_ind.append(h_nid)
                    col_ind.append(indep_col_map[parent_nid])
                    data_vals.append(float(w))
                else:
                    # Chained hanging node (rare in 1-irregular, but handle gracefully)
                    if parent_nid in hanging_map:
                        for grand_nid, gw in hanging_map[parent_nid].items():
                            if grand_nid in indep_col_map:
                                row_ind.append(h_nid)
                                col_ind.append(indep_col_map[grand_nid])
                                data_vals.append(float(w * gw))

        C_matrix = csr_matrix(
            (data_vals, (row_ind, col_ind)),
            shape=(n_nodes, n_true),
            dtype=np.float64,
        )

        return AMRMesh(
            nodes=unique_nodes,
            elements=elements,
            leaf_levels=levels,
            independent_node_indices=np.array(independent_nodes, dtype=np.int64),
            hanging_node_indices=np.array(hanging_nodes, dtype=np.int64),
            constraint_matrix=C_matrix,
        )
