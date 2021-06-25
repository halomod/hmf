"""A package for producing halo mass functions under Spherical Collapse."""
try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass

from ._internals import (
    Component,
    Framework,
    cached_quantity,
    get_base_component,
    get_base_components,
    get_mdl,
    parameter,
)
from .alternatives import wdm
from .cosmology import Cosmology, GrowthFactor, cosmo, growth_factor
from .density_field import CAMB, Transfer, filters, halofit, transfer, transfer_models
from .halos import mass_definitions
from .helpers import functional, get_best_param_order, get_hmf, sample
from .mass_function import MassFunction, fitting_functions, hmf, integrate_hmf
