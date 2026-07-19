"""
STEP file parser for WNFEA.

Parses a subset of ISO 10303-21 (STEP) files to extract:
  - CARTESIAN_POINT → node coordinates
  - VERTEX_POINT    → named vertices referencing cartesian points
  - EDGE_CURVE      → edges connecting two vertices (used for beam meshing)
  - ADVANCED_FACE / FACE_BOUND / EDGE_LOOP → surface/shell faces (visualization only)

This is a lightweight parser targeting wireframe and simple B-rep STEP files.
It does NOT implement a full STEP schema interpreter.
"""

import re
from pathlib import Path
from .primitives import Point3D, GeometryNode, GeometryEdge, GeometryFace


class STEPParseError(Exception):
    """Raised when the STEP file cannot be parsed."""
    pass


class STEPParser:
    """
    Parses a STEP file and returns geometry primitives.
    
    Usage:
        parser = STEPParser()
        result = parser.parse("path/to/file.stp")
        nodes = result['nodes']       # dict[int, GeometryNode]
        edges = result['edges']       # dict[int, GeometryEdge]
        faces = result['faces']       # dict[int, GeometryFace]
    """

    def __init__(self):
        self._raw_entities: dict[int, tuple[str, str]] = {}  # id → (entity_type, param_string)
        self._cartesian_points: dict[int, Point3D] = {}
        self._vertex_points: dict[int, int] = {}  # vertex_id → cartesian_point_id
        self._nodes: dict[int, GeometryNode] = {}
        self._edges: dict[int, GeometryEdge] = {}
        self._faces: dict[int, GeometryFace] = {}

    def parse(self, filepath: str | Path) -> dict:
        """
        Parse a STEP file and return extracted geometry.

        Args:
            filepath: Path to the .stp / .step file.

        Returns:
            dict with keys 'nodes', 'edges', 'faces', 'source_file'.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise STEPParseError(f"File not found: {filepath}")

        text = filepath.read_text(encoding='utf-8', errors='replace')

        self._raw_entities.clear()
        self._cartesian_points.clear()
        self._vertex_points.clear()
        self._nodes.clear()
        self._edges.clear()
        self._faces.clear()

        self._extract_entities(text)
        self._parse_cartesian_points()
        self._parse_vertex_points()
        self._build_nodes()
        self._parse_edge_curves()
        self._parse_faces()

        return {
            'nodes': dict(self._nodes),
            'edges': dict(self._edges),
            'faces': dict(self._faces),
            'source_file': str(filepath),
        }

    def _extract_entities(self, text: str):
        """Extract all entity lines from the DATA section of the STEP file."""
        # Find DATA section
        data_match = re.search(r'DATA\s*;(.*?)ENDSEC\s*;', text, re.DOTALL)
        if not data_match:
            raise STEPParseError("Could not find DATA section in STEP file.")

        data_section = data_match.group(1)

        # Join multi-line entities into single lines
        # STEP entities can span multiple lines and end with ';'
        lines = data_section.replace('\n', ' ').replace('\r', ' ')

        # Match entity pattern: #ID = ENTITY_TYPE(params);
        entity_pattern = re.compile(r'#(\d+)\s*=\s*(\w+)\s*\(([^;]*)\)\s*;')
        for match in entity_pattern.finditer(lines):
            entity_id = int(match.group(1))
            entity_type = match.group(2).upper()
            params = match.group(3).strip()
            self._raw_entities[entity_id] = (entity_type, params)

    def _parse_cartesian_points(self):
        """Parse CARTESIAN_POINT entities into Point3D objects."""
        for eid, (etype, params) in self._raw_entities.items():
            if etype == 'CARTESIAN_POINT':
                # Format: 'label', (x, y, z)
                coord_match = re.search(r'\(\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)', params)
                if coord_match:
                    x = float(coord_match.group(1))
                    y = float(coord_match.group(2))
                    z = float(coord_match.group(3))
                    self._cartesian_points[eid] = Point3D(x, y, z)

    def _parse_vertex_points(self):
        """Parse VERTEX_POINT entities to map vertex IDs to cartesian point IDs."""
        for eid, (etype, params) in self._raw_entities.items():
            if etype == 'VERTEX_POINT':
                # Format: 'label', #point_ref
                ref_match = re.search(r'#(\d+)', params)
                if ref_match:
                    self._vertex_points[eid] = int(ref_match.group(1))

    def _build_nodes(self):
        """Build GeometryNode objects from vertex points."""
        node_counter = 0
        for vertex_id, point_id in sorted(self._vertex_points.items()):
            if point_id in self._cartesian_points:
                point = self._cartesian_points[point_id]
                # Try to extract the label from the cartesian point's raw params
                label = self._extract_label(point_id)
                if not label:
                    label = f"Node {node_counter}"
                self._nodes[node_counter] = GeometryNode(
                    id=node_counter,
                    label=label,
                    point=point
                )
                node_counter += 1

        # Store mapping from STEP vertex entity ID → our node ID for edge building
        self._vertex_to_node_id = {}
        for i, vertex_id in enumerate(sorted(self._vertex_points.keys())):
            self._vertex_to_node_id[vertex_id] = i

    def _parse_edge_curves(self):
        """Parse EDGE_CURVE entities to build GeometryEdge objects."""
        edge_counter = 0
        for eid, (etype, params) in self._raw_entities.items():
            if etype == 'EDGE_CURVE':
                # Format: 'label', #start_vertex, #end_vertex, #curve_ref, .T./.F.
                refs = re.findall(r'#(\d+)', params)
                if len(refs) >= 2:
                    start_vertex_id = int(refs[0])
                    end_vertex_id = int(refs[1])

                    start_node = self._vertex_to_node_id.get(start_vertex_id)
                    end_node = self._vertex_to_node_id.get(end_vertex_id)

                    if start_node is not None and end_node is not None:
                        label = self._extract_label_from_params(params)
                        if not label:
                            label = f"Edge {edge_counter}"
                        self._edges[edge_counter] = GeometryEdge(
                            id=edge_counter,
                            label=label,
                            start_node_id=start_node,
                            end_node_id=end_node
                        )
                        edge_counter += 1

    def _parse_faces(self):
        """
        Parse ADVANCED_FACE → FACE_BOUND → EDGE_LOOP entities to build
        GeometryFace objects. These are for VISUALIZATION ONLY.
        """
        # Build edge loop → list of vertex IDs
        edge_loops: dict[int, list[int]] = {}
        for eid, (etype, params) in self._raw_entities.items():
            if etype == 'EDGE_LOOP':
                # Edge loops reference oriented edges, which reference edge curves
                refs = re.findall(r'#(\d+)', params)
                node_ids = set()
                for ref_id in refs:
                    ref_id = int(ref_id)
                    # Check if this is an ORIENTED_EDGE → get its EDGE_CURVE
                    if ref_id in self._raw_entities:
                        oe_type, oe_params = self._raw_entities[ref_id]
                        if oe_type == 'ORIENTED_EDGE':
                            oe_refs = re.findall(r'#(\d+)', oe_params)
                            for oe_ref in oe_refs:
                                oe_ref = int(oe_ref)
                                if oe_ref in self._raw_entities:
                                    ec_type, ec_params = self._raw_entities[oe_ref]
                                    if ec_type == 'EDGE_CURVE':
                                        ec_vrefs = re.findall(r'#(\d+)', ec_params)
                                        for vr in ec_vrefs[:2]:
                                            vr = int(vr)
                                            if vr in self._vertex_to_node_id:
                                                node_ids.add(self._vertex_to_node_id[vr])
                edge_loops[eid] = sorted(node_ids)

        # Build face bounds → edge loop mapping
        face_bounds: dict[int, int] = {}
        for eid, (etype, params) in self._raw_entities.items():
            if etype in ('FACE_BOUND', 'FACE_OUTER_BOUND'):
                refs = re.findall(r'#(\d+)', params)
                for ref_id in refs:
                    ref_id = int(ref_id)
                    if ref_id in edge_loops:
                        face_bounds[eid] = ref_id
                        break

        # Build advanced faces
        face_counter = 0
        for eid, (etype, params) in self._raw_entities.items():
            if etype == 'ADVANCED_FACE':
                refs = re.findall(r'#(\d+)', params)
                all_node_ids = set()
                for ref_id in refs:
                    ref_id = int(ref_id)
                    if ref_id in face_bounds:
                        loop_id = face_bounds[ref_id]
                        all_node_ids.update(edge_loops.get(loop_id, []))

                if all_node_ids:
                    label = self._extract_label_from_params(params)
                    if not label:
                        label = f"Face {face_counter}"
                    self._faces[face_counter] = GeometryFace(
                        id=face_counter,
                        label=label,
                        bounding_node_ids=sorted(all_node_ids)
                    )
                    face_counter += 1

    def _extract_label(self, entity_id: int) -> str:
        """Extract the label string from an entity's raw params."""
        if entity_id in self._raw_entities:
            _, params = self._raw_entities[entity_id]
            return self._extract_label_from_params(params)
        return ""

    def _extract_label_from_params(self, params: str) -> str:
        """Extract a quoted label string from entity parameters."""
        label_match = re.match(r"\s*'([^']*)'", params)
        if label_match:
            return label_match.group(1).strip()
        return ""
