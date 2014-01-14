class Cosmology(object):
    """
    A class that nicely deals with cosmological parameters.
    
    Most cosmological parameters are merely input and are made available as
    attributes in the class. However, more complicated relations such as 
    the interrelation of omegab, omegac, omegam, omegav for example are dealt
    with in a more robust manner. 
    
    The secondary purpose of this class is to provide simple mappings of the
    parameters to common python cosmology libraries (for now just `cosmolopy`
    and `pycamb`). It has been found by the authors that using more than one
    library becomes confusing in terms of naming all the parameters, so this 
    class helps deal with that.
    
    .. note :: Currently, only several combinations of the density parameters
            are valid:
    
            1. ``omegab`` and ``omegac``
            #. ``omegam`` 
            #. ``omegab_h2`` and ``omegac_h2``
            #. None of them
    
            To this one may add ``omegav`` (dark-energy density) at will. More
            combinations will be added in the future. 
    
    Parameters
    ----------
    
    default : str, {``None``, ``"planck1_base"``}
        Defines a set of default parameters, based on a published set from WMAP
        or Planck. These defaults are applied in a smart way, so as not to 
        override any user-set parameters. 
        
        Current options are
        
        1. ``None``
        #. ``"planck1_base"``: The cosmology of first-year PLANCK mission (with no lensing or WP)
    
    force_flat : bool, default ``False``
        If ``True``, enforces a flat cosmology :math:`(\Omega_m+\Omega_\lambda=1)`.
        This will modify ``omegav`` only, never ``omegam``.
        
    \*\*kwargs : 
        The list of available keyword arguments is as follows:
        
        1. ``sigma_8``: The normalisation. Mass variance in top-hat spheres with :math:`R=8Mpc h^{-1}`
        #. ``n``: The spectral index
        #. ``w``: The dark-energy equation of state
        #. ``cs2_lam``: The constant comoving sound speed of dark energy
        #. ``t_cmb``: Temperature of the CMB
        #. ``y_he``: Helium fraction
        #. ``N_nu``: Number of massless neutrino species
        #. ``N_nu_massive``:Number of massive neutrino species
        #. ``z_reion``: Redshift of reionization
        #. ``tau``: Optical depth at reionization
        #. ``delta_c``: The critical overdensity for collapse
        #. ``h``: The hubble parameter
        #. ``H0``: The hubble constant
        #. ``omegan``: The normalised density of neutrinos
        #. ``omegam``: The normalised density of matter
        #. ``omegav``: The normalised density of dark energy
        #. ``omegab``: The normalised baryon density
        #. ``omegac``: The normalised CDM density
        #. ``omegab_h2``: The normalised baryon density by ``h**2``
        #. ``omegac_h2``: The normalised CDM density by ``h**2``
        
        .. note :: The reason these are implemented as `kwargs` rather than the
                usual arguments, is because the code can't tell *a priori* which
                combination of density parameters the user will input.
                 
    """
    # A dictionary of bounds for each parameter
    # This also forms a list of all parameters possible
    # Note that just because a parameter is within these bounds doesn't mean
    # it will actually work in say CAMB.
    _bounds = {"sigma_8":[0.1, 10],
              "n":[-3, 4],
              "w":[-1.5, 0],
              "cs2_lam":[-1, 2],
              "t_cmb":[0, 10.0],
              "y_he":[0, 1],
              "N_nu":[1, 10],
              "N_nu_massive":[0, 3],
              "z_reion":[2, 1000],
              "tau":[0, 1],
              "omegan":[0, 1],
              "h":[0.05, 5],
              "H0":[5, 500],
              "omegab":[0.0001, 1],
              "omegac":[0, 2],
              "omegav":[0, 2],
              "omegam":[0.0001, 3],
              "omegab_h2":[0.0001, 1],
              "omegac_h2":[0, 2]}

    def __init__(self, default=None, force_flat=False, **kwargs):

        # Map the 'default' cosmology to its dictionary
        if default == "planck1_base":
            self.__base = dict(planck1_base, **extras)

        # Set some simple parameters
        self.force_flat = force_flat
        self.crit_dens = 27.755e10

        #=======================================================================
        # Check values in kwargs
        #=======================================================================
        for k in kwargs:
            if k not in Cosmology._bounds:
                raise ValueError(k + " is not a valid parameter for Cosmology")

        #=======================================================================
        # First set the "easy" values (no dependence on anything else
        #=======================================================================
        easy_params = ["sigma_8", "n", 'w', 'cs2_lam', 't_cmb', 'y_he', "N_nu",
                       "z_reion", "tau", "omegan", 'delta_c', "N_nu_massive"]
        for p in easy_params:
            if p in kwargs:
                self.__dict__.update({p:kwargs.pop(p)})
            elif default is not None:
                self.__dict__.update({p:self.__base[p]})


        #=======================================================================
        # Now the hard parameters (multi-dependent)
        #=======================================================================
        ################### h/H0 ###############################################
        if "h" in kwargs and "H0" in kwargs:
            if kwargs['h'] != kwargs["H0"] / 100.0:
                print "h and H0 specified inconsistently, using h"



        if "H0" in kwargs:
            self.H0 = kwargs.pop("H0")
            self.h = self.H0 / 100.0

        if "h" in kwargs:
            self.h = kwargs.pop("h")
            self.H0 = 100 * self.h

        if not hasattr(self, "h") and default is not None:
            self.H0 = self.__base["H0"]
            self.h = self.H0 / 100.0

        ################### The omegas #########################################
        if "omegav" in kwargs:
            self.omegav = kwargs.pop("omegav")

        if len(kwargs) == 0:
            if self.force_flat and hasattr(self, "omegav"):
                    self.omegam = 1 - self.omegav
                    self.omegak = 0.0
            elif default is not None:
                self.omegab_h2 = self.__base["omegab_h2"]
                self.omegac_h2 = self.__base["omegac_h2"]
                self.omegab = self.omegab_h2 / self.h ** 2
                self.omegac = self.omegac_h2 / self.h ** 2
                self.omegam = self.omegab + self.omegac


        elif "omegab" in kwargs and "omegac" in kwargs and len(kwargs) == 2:
            self.omegab = kwargs["omegab"]
            self.omegac = kwargs["omegac"]
            self.omegam = self.omegab + self.omegac
            if hasattr(self, "h"):
                self.omegab_h2 = self.omegab * self.h ** 2
                self.omegac_h2 = self.omegac * self.h ** 2

        elif "omegam" in kwargs and len(kwargs) == 1:
            self.omegam = kwargs["omegam"]

        elif "omegab_h2" in kwargs and "omegac_h2" in kwargs and len(kwargs) == 2:
            if not hasattr(self, 'h'):
                raise AttributeError("You need to specify h as well")
            self.omegab_h2 = kwargs["omegab_h2"]
            self.omegac_h2 = kwargs["omegac_h2"]
            self.omegab = self.omegab_h2 / self.h ** 2
            self.omegac = self.omegac_h2 / self.h ** 2
            self.omegam = self.omegab + self.omegac

        else:
            raise AttributeError("your input omegaXXX arguments were invalid" + str(kwargs))

        if hasattr(self, "omegam"):
            self.mean_dens = self.crit_dens * self.omegam
            if self.force_flat:
                self.omegav = 1 - self.omegam
                self.omegak = 0.0
            elif default is not None and not hasattr(self, "omegav"):
                self.omegav = self.__base["omegav"]

            if hasattr(self, "omegav") and not self.force_flat:
                self.omegak = 1 - self.omegav - self.omegam

        # Check all their values
        for k, v in Cosmology._bounds.iteritems():
            if k in self.__dict__:
                self._check_bounds(k, v[0], v[1])


    def pycamb_dict(self):
        """
        Collect parameters into a dictionary suitable for pycamb.
        
        Returns
        -------
        dict
            Dictionary of values appropriate for pycamb
        """
        map = {"w":"w_lam",
               "t_cmb":"TCMB",
               "y_he":"yhe",
               # "tau":"reion__optical_depth",
               "z_reion":"reion__redshift",
               "N_nu":"Num_Nu_massless",
               "omegab":"omegab",
               "omegac":"omegac",
               "H0":"H0",
               "omegav":"omegav",
               "omegak":"omegak",
               "omegan":"omegan",
               "cs2_lam":"cs2_lam",
               "n":"scalar_index",
               "N_nu_massive":"Num_Nu_massive"
               }

        return_dict = {}
        for k, v in self.__dict__.iteritems():
            if k in map:
                return_dict.update({map[k]: v})

        return return_dict

    def cosmolopy_dict(self):
        """
        Collect parameters into a dictionary suitable for cosmolopy.
        
        Returns
        -------
        dict
            Dictionary of values appropriate for cosmolopy
        """
        map = {"tau":"tau",
               "z_reion":"z_reion",
               "omegab":"omega_b_0",
               "h":"h",
               "omegav":"omega_lambda_0",
               "omegak":"omega_k_0",
               "sigma_8":"sigma_8",
               "omegam":"omega_M_0",
               "n":"n",
               "omegan":"omega_n_0",
               "N_nu_massive":"N_nu",
               "w":"w"}
        return_dict = {}
        for k, v in self.__dict__.iteritems():
            if k in map:
                return_dict.update({map[k]: v})

        return return_dict

    def _check_bounds(self, item, low=None, high=None):
        if low is not None and high is not None:
            if self.__dict__[item] < low or self.__dict__[item] > high:
                raise ValueError(item + " must be between " + str(low) + " and " + str(high))
        elif low is not None:
            if self.__dict__[item] < low:
                raise ValueError(item + " must be less than " + str(low))
        elif high is not None:
            if self.__dict__[item] > high:
                raise ValueError(item + " must be greater than " + str(high))

#===============================================================================
# SOME BASE COSMOLOGIES
#===============================================================================
# The extras dict has common parameter defaults between all bases
extras = {"w"   :-1,
          "omegan"  : 0.0,
          'cs2_lam' : 1,
          't_cmb'    : 2.725,
          'y_he'     : 0.24,
          'N_nu'    : 3.04,
          "delta_c" : 1.686,
          "N_nu_massive":0.0,
          }

# # Base Planck (no extra things like lensing and WP)
planck1_base = {"omegab_h2"   : 0.022068,
                "omegac_h2"   : 0.12029,
                "omegav"   : 0.6825,
                "H0"       : 67.11,
                'z_reion': 11.35,
                'tau': 0.0925,
                "sigma_8":0.8344,
                "n":0.9624,
                }
