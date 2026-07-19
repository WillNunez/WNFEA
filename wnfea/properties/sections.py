"""
Beam cross-section profile definitions for WNFEA.

Computes the section properties (A, Iy, Iz, J) needed by the beam
element stiffness matrix formulation. Currently supports:
  - Hollow circular tube
  - Solid circular rod

Architecture is designed to be extended with rectangular, I-beam, etc.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class SectionDef:
    """
    Base beam cross-section definition.

    Attributes:
        name: Human-readable section profile name.
        area: Cross-sectional area A (m²).
        iy: Second moment of area about local y-axis (m⁴).
        iz: Second moment of area about local z-axis (m⁴).
        j: Polar moment of area / torsion constant (m⁴).
        outer_radius: Outer radius for tube visualization (m). May be None for non-circular.
        inner_radius: Inner radius for tube visualization (m). 0.0 for solid.
    """
    name: str
    area: float
    iy: float
    iz: float
    j: float
    outer_radius: float | None = None
    inner_radius: float = 0.0

    def validate(self) -> list[str]:
        """Return a list of validation error strings. Empty list means valid."""
        errors = []
        if self.area <= 0:
            errors.append(f"Area must be positive, got {self.area}")
        if self.iy <= 0:
            errors.append(f"Iy must be positive, got {self.iy}")
        if self.iz <= 0:
            errors.append(f"Iz must be positive, got {self.iz}")
        if self.j <= 0:
            errors.append(f"J must be positive, got {self.j}")
        return errors

    def to_dict(self) -> dict:
        """Convert to the dict format expected by the solver functions."""
        return {
            "Area": self.area,
            "Iy": self.iy,
            "Iz": self.iz,
            "J": self.j,
            "outer_radius": self.outer_radius,
            "inner_radius": self.inner_radius,
        }

    def __repr__(self):
        return (
            f"SectionDef('{self.name}', A={self.area:.6e}, "
            f"Iy={self.iy:.6e}, Iz={self.iz:.6e}, J={self.j:.6e})"
        )


def create_hollow_tube(name: str, outer_radius: float, inner_radius: float) -> SectionDef:
    """
    Create a hollow circular tube cross-section.

    Args:
        name: Profile name.
        outer_radius: Outer radius r_o (m).
        inner_radius: Inner radius r_i (m). Must be < outer_radius.

    Returns:
        A SectionDef with computed properties.

    Raises:
        ValueError: If radii are invalid.
    """
    if outer_radius <= 0:
        raise ValueError(f"Outer radius must be positive, got {outer_radius}")
    if inner_radius < 0:
        raise ValueError(f"Inner radius must be non-negative, got {inner_radius}")
    if inner_radius >= outer_radius:
        raise ValueError(
            f"Inner radius ({inner_radius}) must be less than outer radius ({outer_radius})"
        )

    A = np.pi * (outer_radius**2 - inner_radius**2)
    Iy = np.pi / 4 * (outer_radius**4 - inner_radius**4)
    Iz = Iy  # Symmetric circular section
    J = 2 * Iy  # Polar moment for thin/thick-walled tube

    return SectionDef(
        name=name,
        area=A,
        iy=Iy,
        iz=Iz,
        j=J,
        outer_radius=outer_radius,
        inner_radius=inner_radius,
    )


def create_solid_circle(name: str, radius: float) -> SectionDef:
    """
    Create a solid circular rod cross-section.

    Args:
        name: Profile name.
        radius: Section radius r (m).

    Returns:
        A SectionDef with computed properties.

    Raises:
        ValueError: If radius is invalid.
    """
    if radius <= 0:
        raise ValueError(f"Radius must be positive, got {radius}")

    A = np.pi * radius**2
    Iy = np.pi / 4 * radius**4
    Iz = Iy
    J = 2 * Iy

    return SectionDef(
        name=name,
        area=A,
        iy=Iy,
        iz=Iz,
        j=J,
        outer_radius=radius,
        inner_radius=0.0,
    )
