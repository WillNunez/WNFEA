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


def create_square_tube(name: str, width: float, thickness: float) -> SectionDef:
    """Create a square tube cross-section."""
    if width <= 0:
        raise ValueError(f"Width must be positive, got {width}")
    if thickness <= 0:
        raise ValueError(f"Thickness must be positive, got {thickness}")
    if thickness >= width / 2:
        raise ValueError(f"Thickness ({thickness}) must be less than half of width ({width/2})")
    
    w = width - 2.0 * thickness
    A = width**2 - w**2
    Iy = (1.0 / 12.0) * (width**4 - w**4)
    Iz = Iy
    J = (width - thickness)**3 * thickness
    
    return SectionDef(
        name=name,
        area=A,
        iy=Iy,
        iz=Iz,
        j=J,
        outer_radius=width / 2.0,
        inner_radius=w / 2.0,
    )


def create_rectangular_tube(name: str, width: float, height: float, thickness: float) -> SectionDef:
    """Create a rectangular tube cross-section."""
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive")
    if thickness <= 0:
        raise ValueError("Thickness must be positive")
    min_dim = min(width, height)
    if thickness >= min_dim / 2:
        raise ValueError(f"Thickness ({thickness}) must be less than half of minimum dimension ({min_dim/2})")
        
    w = width - 2.0 * thickness
    h = height - 2.0 * thickness
    A = width * height - w * h
    Iy = (1.0 / 12.0) * (width * height**3 - w * h**3)
    Iz = (1.0 / 12.0) * (height * width**3 - h * w**3)
    J = (2.0 * thickness * (width - thickness)**2 * (height - thickness)**2) / ((width - thickness) + (height - thickness))
    
    return SectionDef(
        name=name,
        area=A,
        iy=Iy,
        iz=Iz,
        j=J,
        outer_radius=width / 2.0,
        inner_radius=w / 2.0,
    )


def create_i_beam(name: str, height: float, width: float, flange_thickness: float, web_thickness: float) -> SectionDef:
    """Create an I-beam cross-section."""
    if height <= 0 or width <= 0 or flange_thickness <= 0 or web_thickness <= 0:
        raise ValueError("Dimensions must be positive")
    if flange_thickness >= height / 2:
        raise ValueError("Flange thickness must be less than half of height")
    if web_thickness >= width:
        raise ValueError("Web thickness must be less than flange width")
        
    A = 2.0 * (width * flange_thickness) + web_thickness * (height - 2.0 * flange_thickness)
    Iy = (1.0 / 12.0) * width * height**3 - (1.0 / 12.0) * (width - web_thickness) * (height - 2.0 * flange_thickness)**3
    Iz = 2.0 * ((1.0 / 12.0) * flange_thickness * width**3) + (1.0 / 12.0) * (height - 2.0 * flange_thickness) * web_thickness**3
    J = (1.0 / 3.0) * (2.0 * width * flange_thickness**3 + (height - 2.0 * flange_thickness) * web_thickness**3)
    
    return SectionDef(
        name=name,
        area=A,
        iy=Iy,
        iz=Iz,
        j=J,
        outer_radius=width / 2.0,
        inner_radius=0.0,
    )


def create_rectangular_bar(name: str, width: float, height: float) -> SectionDef:
    """Create a solid rectangular bar cross-section."""
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive")
        
    A = width * height
    Iy = (1.0 / 12.0) * width * height**3
    Iz = (1.0 / 12.0) * height * width**3
    a = max(width, height)
    b = min(width, height)
    J = a * b**3 * (1.0/3.0 - 0.21 * (b/a) * (1.0 - b**4 / (12.0 * a**4)))
    
    return SectionDef(
        name=name,
        area=A,
        iy=Iy,
        iz=Iz,
        j=J,
        outer_radius=width / 2.0,
        inner_radius=0.0,
    )


def create_general_section(
    name: str, area: float, iy: float, iz: float, j: float, outer_radius: float, inner_radius: float = 0.0
) -> SectionDef:
    """Create a custom general cross-section with validation."""
    sect = SectionDef(name, area, iy, iz, j, outer_radius, inner_radius)
    errors = sect.validate()
    if errors:
        raise ValueError("Invalid general section:\n  " + "\n  ".join(errors))
    return sect

