"""
Module dealing with cosmological models.

The main class is `Cosmology`, which is a framework wrapping the astropy
cosmology classes, while converting it to a :class:`hmf._framework.Framework`
for use in this package.

Also provided in the namespace are the pre-defined cosmologies from `astropy`:
`WMAP5`, `WMAP7`, `WMAP9`, `Planck13` and `Planck15`, which may be used as arguments to the
Cosmology framework. All custom subclasses of :class:`astropy.cosmology.FLRW`
may be used as inputs.
"""

import astropy.units as u
import deprecation
import sys
from astropy.cosmology import FLRW, WMAP5, WMAP7, WMAP9, Planck13, Planck15  # noqa

from .. import __version__
from .._internals import _cache, _framework


@deprecation.deprecated(
    deprecated_in="3.1.3",
    removed_in="4.0",
    current_version=__version__,
    details="Use the cosmology.fromAstropy function in COLOSSUS instead",
)
def astropy_to_colossus(cosmo: FLRW, name: str = "custom", **kwargs):
    """Convert an astropy cosmology to a COLOSSUS cosmology"""
    try:
        from colossus.cosmology import cosmology

        return cosmology.fromAstropy(astropy_cosmo=cosmo, cosmo_name=name, **kwargs)
    except ImportError:  # pragma: nocover
        raise ImportError(
            "Cannot convert to COLOSSUS cosmology without installing COLOSSUS!"
        )


class Cosmology(_framework.Framework):
    """
    Basic Cosmology object.

    This class thinly wraps cosmology objects from the astropy package. The full
    functionality of the astropy cosmology objects are available in the
    :attr:`cosmo` attribute. What the class adds to the existing astropy
    implementation is the specification of the cosmological parameters
    as `parameter` inputs to an over-arching Framework.

    In particular, while any instance of a subclass of :class:`astropy.cosmology.FLRW`
    may be passed as the base cosmology, the specific parameters can be updated
    individually by passing them through the `cosmo_params` dictionary
    (both in the constructor and the :meth:`update` method.

    This dictionary is kept in memory and so adding a different parameter on a later
    update will *update* the dictionary, rather than replacing it.

    To read a standard documented list of parameters, use ``Cosmology.parameter_info()``.
    If you want to just see the plain list of available parameters, use ``Cosmology.get_all_parameters()``.
    To see the actual defaults for each parameter, use ``Cosmology.get_all_parameter_defaults()``.
    """

    def __init__(self, cosmo_model=Planck15, cosmo_params=None):
        # Call Framework init
        super().__init__()

        # Set all given parameters
        self.cosmo_model = cosmo_model
        self.cosmo_params = cosmo_params or {}

    @_cache.parameter("model")
    def cosmo_model(self, val):
        """
        The basis for the cosmology -- see astropy documentation. Can be a custom
        subclass. Defaults to Planck15.

        :type: instance of `astropy.cosmology.FLRW` subclass
        """
        if isinstance(val, str):
            return get_cosmo(val)

        if not isinstance(val, FLRW):
            raise ValueError(
                "cosmo_model must be an instance of astropy.cosmology.FLRW"
            )
        else:
            return val

    @_cache.parameter("param")
    def cosmo_params(self, val):
        """
        Parameters for the cosmology that deviate from the base cosmology passed.
        This is useful for repeated updates of a single parameter (leaving others
        the same). Default is the empty dict. The parameters passed must match
        the allowed parameters of `cosmo_model`. For the basic class this is

        :Tcmb0: Temperature of the CMB at z=0
        :Neff: Number of massless neutrino species
        :m_nu: Mass of neutrino species (list)
        :H0: The hubble constant at z=0
        :Om0: The normalised matter density at z=0

        :type: dict
        """
        return val

    # ===========================================================================
    # DERIVED PROPERTIES AND FUNCTIONS
    # ===========================================================================
    @_cache.cached_quantity
    def cosmo(self):
        """
        Cosmographic object (:class:`astropy.cosmology.FLRW` object), with custom
        cosmology from :attr:`~.cosmo_params` applied.
        """
        return self.cosmo_model.clone(**self.cosmo_params)

    @_cache.cached_quantity
    def mean_density0(self):
        """
        Mean density of universe at z=0, [Msun h^2 / Mpc**3]
        """
        return (
            (self.cosmo.Om0 * self.cosmo.critical_density0 / self.cosmo.h ** 2)
            .to(u.Msun / u.Mpc ** 3)
            .value
        )


def get_cosmo(name):
    """
    Returns a FLRW cosmology given a string (must be one defined in this module).

    Parameters
    ----------
    name : str
        The class name of the appropriate model
    """
    if isinstance(getattr(sys.modules[__name__], name), FLRW):
        return getattr(sys.modules[__name__], name)
    else:
        raise ValueError("%s is not a valid cosmology" % name)
