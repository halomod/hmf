"""
Module dealing with cosmological models.

The main class is `Cosmology`, which is a framework wrapping the astropy
cosmology classes and adding some structural support for this package.

Also provided in the namespace are the pre-defined cosmologies from astropy,
WMAP5, WMAP7, WMAP9 and Planck13, which may be used as arguments to the
Cosmology framework. All custom subclasses of :class:`astropy.cosmology.FLRW`
may be used as inputs.
"""

from _cache import parameter, cached_property
from astropy.cosmology import Planck13, FLRW, WMAP5, WMAP7, WMAP9
# from types import MethodType
from _framework import Framework
import sys
from astropy.units import MsolMass, Mpc
class Cosmology(Framework):
    """
    Basic Cosmology object.

    This class provides a cosmology, basically wrapping cosmology objects from
    the astropy package. The full functionality of the astropy cosmology objects
    are available in the :attr:`cosmo` attribute.

    This class patches in the baryon density as a parameter (to be included in a
    later version of astropy, therefore deprecated here), and also some structural
    support for the rest of the :module:`hmf` package. In particular, while any
    instance of a subclass of :class:`astropy.cosmology.FLRW` may be passed as
    the base cosmology, the specific parameters can be updated individually by
    passing them through the `cosmo_params` dictionary (both in the constructor
    and the :method:`update` method. This dictionary is kept in memory and so
    adding a different parameter on a later update will *update* the dictionary,
    rather than replacing it.

    Parameters
    ----------
    cosmo_model : instance of `astropy.cosmology.FLRW`, optional
        The basis for the cosmology -- see astropy documentation. Can be a custom
        subclass. Defaults to Planck13.

    cosmo_params : dict, optional
        Parameters for the cosmology that deviate from the base cosmology passed.
        This is useful for repeated updates of a single parameter (leaving others
        the same). Default is the empty dict. The parameters passed must match
        the allowed parameters of `cosmo_model`. For the basic class this is

        :w: The dark-energy equation of state
        :Tcmb0: Temperature of the CMB at z=0
        :Neff: Number of massless neutrino species
        :H0: The hubble constant at z=0
        :On0: The normalised density of neutrinos at z = 0
        :Ode0: The normalised density of dark energy at z=0
        :Om0: The normalised matter density at z=0

    """
    def __init__(self, cosmo_model=Planck13, cosmo_params=None):
        # Call Cosmology init
        super(Cosmology, self).__init__()

        # Set all given parameters
        self.cosmo_model = cosmo_model
        self.cosmo_params = cosmo_params or {}

    #===========================================================================
    # Parameters
    #===========================================================================
    @parameter
    def cosmo_model(self, val):
        """:class:`~astropy.cosmology.FLRW` instance"""
        if isinstance(val, basestring):
            cosmo = get_cosmo(val)
            return cosmo

        if not isinstance(val, FLRW):
                raise ValueError("cosmo_model must be an instance of astropy.cosmology.FLRW")
        else:
            return val

    @parameter
    def cosmo_params(self, val):
        return val

    #===========================================================================
    # DERIVED PROPERTIES AND FUNCTIONS
    #===========================================================================
    @cached_property("cosmo_params", "cosmo_model")
    def cosmo(self):
        return self.cosmo_model.clone(**self.cosmo_params)

    @cached_property("cosmo")
    def mean_density0(self):
        """
        Mean density of universe at z=0, units [Msun h^2 / Mpc**3]
        """
        # fixme: why the *1e6??
        return (self.cosmo.Om0 * self.cosmo.critical_density0 / self.cosmo.h ** 2).to(MsolMass/Mpc**3).value * 1e6


def get_cosmo(name):
    """
    Returns a cosmology.

    Parameters
    ----------
    name : str
        The class name of the appropriate model
    """
    if isinstance(getattr(sys.modules[__name__], name), FLRW):
        return getattr(sys.modules[__name__], name)
    else:
        raise ValueError("%s is not a valid cosmology" % name)
