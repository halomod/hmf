"""
Provides descriptions of internal halo properties.

See ``halomod`` for more extended quantities in this regard.
"""

from . import mass_definitions
from .mass_definitions import BaseMassDefinition, MassDefinition

__all__ = [
    "BaseMassDefinition",
    "MassDefinition",  # the old name for BaseMassDefinition
    "mass_definitions",
]
