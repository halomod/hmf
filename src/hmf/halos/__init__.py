"""
Provides descriptions of internal halo properties.

See ``halomod`` for more extended quantities in this regard.
"""

from . import mass_definitions
from .mass_definitions import MassDefinition

__all__ = [
    "MassDefinition",
    "mass_definitions",
]
