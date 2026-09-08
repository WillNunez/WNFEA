"""
Material definitions for beam finite elements.

Provides an isotropic material dataclass and a preset library of common
engineering materials (Structural Steel, Aluminum 6061-T6, Titanium Grade 5).
"""

from dataclasses import dataclass


@dataclass
class MaterialDef:
    """
    Isotropic material definition.

    Attributes:
        name: Human-readable material name.
        youngs_modulus: Young's Modulus E (Pa).
        poissons_ratio: Poisson's Ratio ν (dimensionless, 0 < ν < 0.5).
        yield_strength: Yield strength σ_y (Pa). Used for safety factor calculation.
    """
    name: str
    youngs_modulus: float  # Pa
    poissons_ratio: float  # dimensionless
    yield_strength: float  # Pa
    density: float = 7850.0  # kg/m^3

    @property
    def shear_modulus(self) -> float:
        """Shear modulus G derived from E and ν."""
        return self.youngs_modulus / (2.0 * (1.0 + self.poissons_ratio))

    def validate(self) -> list[str]:
        """Return a list of validation error strings. Empty list means valid."""
        errors = []
        if self.youngs_modulus <= 0:
            errors.append(f"Young's Modulus must be positive, got {self.youngs_modulus}")
        if not (0.0 < self.poissons_ratio < 0.5):
            errors.append(f"Poisson's Ratio must be in (0, 0.5), got {self.poissons_ratio}")
        if self.yield_strength <= 0:
            errors.append(f"Yield Strength must be positive, got {self.yield_strength}")
        if self.density <= 0:
            errors.append(f"Density must be positive, got {self.density}")
        return errors

    def __repr__(self):
        return (
            f"MaterialDef('{self.name}', E={self.youngs_modulus:.3e} Pa, "
            f"nu={self.poissons_ratio:.3f}, Sy={self.yield_strength:.3e} Pa, rho={self.density:.1f} kg/m^3)"
        )


# --- Preset Material Library ---

MATERIAL_PRESETS: dict[str, MaterialDef] = {
    "Structural Steel": MaterialDef(
        name="Structural Steel",
        youngs_modulus=200e9,
        poissons_ratio=0.27,
        yield_strength=250e6,
        density=7850.0,
    ),
    "Aluminum 6061-T6": MaterialDef(
        name="Aluminum 6061-T6",
        youngs_modulus=68.9e9,
        poissons_ratio=0.33,
        yield_strength=276e6,
        density=2700.0,
    ),
    "Titanium Grade 5": MaterialDef(
        name="Titanium Grade 5",
        youngs_modulus=114e9,
        poissons_ratio=0.34,
        yield_strength=880e6,
        density=4430.0,
    ),
}


def get_preset_material(name: str) -> MaterialDef:
    """
    Retrieve a preset material by name.

    Args:
        name: One of the keys in MATERIAL_PRESETS.

    Returns:
        A copy of the MaterialDef.

    Raises:
        KeyError: If the preset name is not found.
    """
    if name not in MATERIAL_PRESETS:
        available = ", ".join(MATERIAL_PRESETS.keys())
        raise KeyError(f"Unknown material preset '{name}'. Available: {available}")
    preset = MATERIAL_PRESETS[name]
    return MaterialDef(
        name=preset.name,
        youngs_modulus=preset.youngs_modulus,
        poissons_ratio=preset.poissons_ratio,
        yield_strength=preset.yield_strength,
    )


def create_custom_material(
    name: str,
    youngs_modulus: float,
    poissons_ratio: float,
    yield_strength: float,
) -> MaterialDef:
    """
    Create a custom isotropic material with validation.

    Args:
        name: Material name.
        youngs_modulus: E in Pa.
        poissons_ratio: ν (dimensionless).
        yield_strength: σ_y in Pa.

    Returns:
        A validated MaterialDef.

    Raises:
        ValueError: If any parameter is out of physical range.
    """
    mat = MaterialDef(name, youngs_modulus, poissons_ratio, yield_strength)
    errors = mat.validate()
    if errors:
        raise ValueError("Invalid material definition:\n  " + "\n  ".join(errors))
    return mat
