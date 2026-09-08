"""
WNFEA CAD Kernel Interface and B-Rep Feature Recognition Package.
"""

from .freecad_brep import (
    CylindricalFeature,
    FreeCADBRepDetector,
    match_standard_bolt,
    generate_bolt_supports,
    generate_bearing_loads,
    generate_spider_coupling,
)

__all__ = [
    "CylindricalFeature",
    "FreeCADBRepDetector",
    "match_standard_bolt",
    "generate_bolt_supports",
    "generate_bearing_loads",
    "generate_spider_coupling",
]
