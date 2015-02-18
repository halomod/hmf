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
import astropy.units as u
from _framework import Framework
import sys
h_unit = u.def_unit("h")

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
    base_cosmo : instance of `astropy.cosmology.FLRW`, optional
        The basis for the cosmology -- see astropy documentation. Can be a custom
        subclass. Defaults to Planck13.
        
    Ob0 : float, depcrecated, optional
        Baryon density at redshift zero. This will be removed when astropy includes
        it as standard in the cosmology classes. Defaults to 0.05
        
    cosmo_params : dict, optional
        Parameters for the cosmology that deviate from the base cosmology passed.
        This is useful for repeated updates of a single parameter (leaving others
        the same). Default is the empty dict. The parameters passed must match
        the allowed parameters of `base_cosmo`. For the basic class this is
        
        :w: The dark-energy equation of state
        :Tcmb0: Temperature of the CMB at z=0
        :Neff: Number of massless neutrino species
        :H0: The hubble constant at z=0
        :On0: The normalised density of neutrinos at z = 0
        :Ode0: The normalised density of dark energy at z=0
        :Om0: The normalised matter density at z=0     

    """
    def __init__(self, base_cosmo=Planck13, Ob0=0.05, cosmo_params=None):
        # Call Cosmology init
        super(Cosmology, self).__init__()

        # Set all given parameters
        self.base_cosmo = base_cosmo
        self.cosmo_params = cosmo_params or {}
        self.cosmo.Ob0 = Ob0  # deprecated when astropy includes Ob0

        # An additional unchangeable parameter for the h unit
        self._hunit = h_unit
    def update(self, **kwargs):
        """
        Update the class optimally with given arguments.
        
        Accepts any argument that the constructor takes
        """
        # # All the following is deprecated when astropy includes Ob0
        Ob0 = self.cosmo.Ob0
        if "Ob0" in kwargs:
            self.cosmo.Ob0 = kwargs.pop("Ob0")
        elif "base_cosmo" in kwargs:
            setattr(self, "base_cosmo", kwargs.pop("base_cosmo"))
            self.cosmo.Ob0 = Ob0
        elif "cosmo_params" in kwargs:
            setattr(self, "cosmo_params", kwargs.pop("cosmo_params"))
            self.cosmo.Ob0 = Ob0

        super(Cosmology, self).update(**kwargs)


    #===========================================================================
    # Parameters
    #===========================================================================
    @parameter
    def base_cosmo(self, val):
        """:class:`~astropy.cosmology.FLRW` instance"""
        if isinstance(val, basestring):
            cosmo = get_cosmo(val)
            return cosmo

        if not isinstance(val, FLRW):
                raise ValueError("base_cosmo must be an instance of astropy.cosmology.FLRW")
        else:
            # monkey-patch on the clone method
#            val.clone = MethodType(clone, val)
            return val

    @parameter
    def cosmo_params(self, val):
        return val

    #===========================================================================
    # DERIVED PROPERTIES AND FUNCTIONS
    #===========================================================================
    @cached_property("cosmo_params", "base_cosmo")
    def cosmo(self):
        return clone(self.base_cosmo, **self.cosmo_params)

    @cached_property("cosmo")
    def mean_density0(self):
        # fixme: why the *1e6??
        return h_unit ** 2 * (self.cosmo.Om0 * self.cosmo.critical_density0 / self.cosmo.h ** 2).to(u.MsolMass / u.Mpc ** 3) * 1e6


#===============================================================================
# Clone function for Cosmology, from astropy (future version)
#===============================================================================
def clone(self, **kwargs):
    """ Returns a copy of an FLRW object, potentially with some changes.
    
    This is shamelessly copied from the dev version of astropy and thus
    is already deprecated in principle.
    
    Returns
    -------
    newcos : Subclass of FLRW
        A new instance of this class with the specified changes.
    
    Notes
    -----
    This assumes that the values of all constructor arguments
    are available as properties, which is true of all the provided
    subclasses but may not be true of user-provided ones. You can't
    change the type of class, so this can't be used to change between
    flat and non-flat. If no modifications are requested, then
    a reference to this object is returned.
    
    Examples
    --------
    To make a copy of the Planck13 cosmology with a different Omega_m
    and a new name:
    
        >>> from astropy.cosmology import Planck13
        >>> newcos = Planck13.clone(name="Modified Planck 2013", Om0=0.35)
    """
    # Quick return check, taking advantage of the
    # immutability of cosmological objects
    if len(kwargs) == 0:
        return self

    # Get constructor arguments
    import inspect
    arglist = inspect.getargspec(self.__init__).args

    # Build the dictionary of values used to construct this
    # object. This -assumes- every argument to __init__ has a
    # property. This is true of all the classes we provide, but
    # maybe a user won't do that. So at least try to have a useful
    # error message.
    argdict = {}
    for arg in arglist[1:]:  # Skip self, which should always be first
        try:
            val = getattr(self, arg)
            argdict[arg] = val
        except AttributeError:
            # We didn't find a property -- complain usefully
            errstr = "Object did not have property corresponding "\
            "to constructor argument '%s'; perhaps it is a "\
            "user provided subclass that does not do so"
            raise AttributeError(errstr % arg)

    # Now substitute in new arguments
    for newarg in kwargs:
        if newarg not in argdict:
            errstr = "User provided argument '%s' not found in "\
            "constructor for this object"
            raise AttributeError(errstr % newarg)
        argdict[newarg] = kwargs[newarg]

    return self.__class__(**argdict)

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
    return getattr(sys.modules[name], name)(**kwargs)
#===============================================================================
# Some Extra Cosmology classes
#===============================================================================
