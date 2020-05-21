try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass

from .alternatives import wdm
from .cosmology import cosmo, growth_factor, Cosmology, GrowthFactor
from .density_field import filters, halofit, transfer, transfer_models, Transfer, CAMB
from .halos import mass_definitions
from .helpers import functional, sample, get_hmf, get_best_param_order
from .mass_function import fitting_functions, hmf, integrate_hmf, MassFunction
from ._internals import Component, Framework, cached_quantity, parameter
