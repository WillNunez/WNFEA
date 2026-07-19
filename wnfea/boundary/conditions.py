"""
Boundary condition definitions for WNFEA.

Supports per-DOF constraint specification:
  - FREE: unconstrained degree of freedom
  - FIXED: locked at zero displacement
  - PRESCRIBED: locked at a user-specified displacement value

Loads are defined as point forces and moments on specific nodes.
Both geometry nodes (pre-mesh) and mesh nodes (post-mesh) can be targets.
"""

from dataclasses import dataclass, field
from enum import Enum


class DOFType(Enum):
    """Constraint type for a single degree of freedom."""
    FREE = "free"
    FIXED = "fixed"
    PRESCRIBED = "prescribed"


@dataclass
class DOFConstraint:
    """
    Constraint definition for a single degree of freedom.

    Attributes:
        dof_type: FREE, FIXED, or PRESCRIBED.
        value: The prescribed displacement/rotation value (only used when PRESCRIBED).
    """
    dof_type: DOFType = DOFType.FREE
    value: float = 0.0

    def __repr__(self):
        if self.dof_type == DOFType.PRESCRIBED:
            return f"DOFConstraint({self.dof_type.value}, value={self.value})"
        return f"DOFConstraint({self.dof_type.value})"


@dataclass
class SupportDef:
    """
    Support (boundary condition) applied to a node.
    
    Each of the 6 DOFs (Ux, Uy, Uz, Rx, Ry, Rz) is independently defined
    as FREE, FIXED, or PRESCRIBED with a displacement/rotation value.

    Attributes:
        node_id: The node this support applies to.
        is_geometry_node: True if node_id refers to a geometry node, False for mesh node.
        ux, uy, uz: Translational DOF constraints.
        rx, ry, rz: Rotational DOF constraints.
        label: Optional human-readable label.
    """
    node_id: int
    is_geometry_node: bool = True
    ux: DOFConstraint = field(default_factory=DOFConstraint)
    uy: DOFConstraint = field(default_factory=DOFConstraint)
    uz: DOFConstraint = field(default_factory=DOFConstraint)
    rx: DOFConstraint = field(default_factory=DOFConstraint)
    ry: DOFConstraint = field(default_factory=DOFConstraint)
    rz: DOFConstraint = field(default_factory=DOFConstraint)
    label: str = ""

    @property
    def constraints(self) -> list[DOFConstraint]:
        """Return all 6 DOF constraints in order [Ux, Uy, Uz, Rx, Ry, Rz]."""
        return [self.ux, self.uy, self.uz, self.rx, self.ry, self.rz]

    @property
    def has_any_constraint(self) -> bool:
        """True if at least one DOF is not FREE."""
        return any(c.dof_type != DOFType.FREE for c in self.constraints)

    @property
    def fixed_dof_indices(self) -> list[int]:
        """Return local DOF indices (0-5) that are FIXED (zero displacement)."""
        return [i for i, c in enumerate(self.constraints) if c.dof_type == DOFType.FIXED]

    @property
    def prescribed_dofs(self) -> dict[int, float]:
        """Return {local_dof_index: prescribed_value} for PRESCRIBED DOFs."""
        return {
            i: c.value
            for i, c in enumerate(self.constraints)
            if c.dof_type == DOFType.PRESCRIBED
        }

    def __repr__(self):
        dof_labels = ['Ux', 'Uy', 'Uz', 'Rx', 'Ry', 'Rz']
        parts = []
        for label, c in zip(dof_labels, self.constraints):
            if c.dof_type == DOFType.FIXED:
                parts.append(f"{label}=FIXED")
            elif c.dof_type == DOFType.PRESCRIBED:
                parts.append(f"{label}={c.value}")
        if not parts:
            parts.append("all FREE")
        target = "geom" if self.is_geometry_node else "mesh"
        return f"SupportDef({target}_node={self.node_id}, {', '.join(parts)})"


@dataclass
class LoadDef:
    """
    Point load applied to a node.

    Attributes:
        node_id: The node this load applies to.
        is_geometry_node: True if node_id refers to a geometry node, False for mesh node.
        fx, fy, fz: Force components (N).
        mx, my, mz: Moment components (N-m).
        label: Optional human-readable label.
    """
    node_id: int
    is_geometry_node: bool = True
    fx: float = 0.0
    fy: float = 0.0
    fz: float = 0.0
    mx: float = 0.0
    my: float = 0.0
    mz: float = 0.0
    label: str = ""

    @property
    def force_vector(self) -> list[float]:
        """Return the 6-component load vector [Fx, Fy, Fz, Mx, My, Mz]."""
        return [self.fx, self.fy, self.fz, self.mx, self.my, self.mz]

    @property
    def has_any_load(self) -> bool:
        """True if at least one force/moment component is nonzero."""
        return any(abs(v) > 1e-12 for v in self.force_vector)

    def __repr__(self):
        parts = []
        if abs(self.fx) > 1e-12: parts.append(f"Fx={self.fx}")
        if abs(self.fy) > 1e-12: parts.append(f"Fy={self.fy}")
        if abs(self.fz) > 1e-12: parts.append(f"Fz={self.fz}")
        if abs(self.mx) > 1e-12: parts.append(f"Mx={self.mx}")
        if abs(self.my) > 1e-12: parts.append(f"My={self.my}")
        if abs(self.mz) > 1e-12: parts.append(f"Mz={self.mz}")
        target = "geom" if self.is_geometry_node else "mesh"
        return f"LoadDef({target}_node={self.node_id}, {', '.join(parts) or 'zero'})"


def create_fixed_support(node_id: int, is_geometry_node: bool = True, label: str = "") -> SupportDef:
    """Convenience: create a fully fixed support (all 6 DOFs locked at zero)."""
    fixed = DOFConstraint(DOFType.FIXED, 0.0)
    return SupportDef(
        node_id=node_id,
        is_geometry_node=is_geometry_node,
        ux=DOFConstraint(DOFType.FIXED),
        uy=DOFConstraint(DOFType.FIXED),
        uz=DOFConstraint(DOFType.FIXED),
        rx=DOFConstraint(DOFType.FIXED),
        ry=DOFConstraint(DOFType.FIXED),
        rz=DOFConstraint(DOFType.FIXED),
        label=label,
    )
