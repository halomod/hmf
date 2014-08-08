"""
This module contains a single class, `Transfer`, which provides methods to 
calculate the transfer function, matter power spectrum and several other 
related quantities. 
"""
import numpy as np
from cosmo import Cosmology
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import cosmolopy as cp
import scipy.integrate as integ
from _cache import cached_property, parameter
import sys
from halofit import _get_spec, halofit
# import cosmolopy.density as cden
import tools
try:
    import pycamb
    HAVE_PYCAMB = True
except ImportError:
    HAVE_PYCAMB = False


#===============================================================================
# Transfer Function Getting Routines
#===============================================================================
def get_transfer(name, t):
    """
    A function that chooses the correct Profile class and returns it
    """
    try:
        return getattr(sys.modules[__name__], name)(t)
    except AttributeError:
        raise AttributeError(str(name) + "  is not a valid GetTransfer class")

_allfits = ["CAMB", "FromFile", "EH", "BBKS", "BondEfs"]

class GetTransfer(object):
    def __init__(self, t):
        self.t = t

    def lnt(self, lnk):
        pass

class CAMB(GetTransfer):
    option_defaults = {"Scalar_initial_condition":1,
                       "lAccuracyBoost":1,
                       "AccuracyBoost":1,
                       "w_perturb":False,
                       "transfer__k_per_logint":11,
                       "transfer__kmax":5,
                       "ThreadNum":0}
    def _check_low_k(self, lnk, lnT):
        """
        Check convergence of transfer function at low k.
        
        Unfortunately, some versions of CAMB produce a transfer which has a
        turn-up at low k, which is what we seek to cut out here.
        
        Parameters
        ----------
        lnk : array_like
            Value of log(k)
            
        lnT : array_like
            Value of log(transfer)
        """

        start = 0
        for i in range(len(lnk) - 1):
            if abs((lnT[i + 1] - lnT[i]) / (lnk[i + 1] - lnk[i])) < 0.01:
                start = i
                break
        lnT = lnT[start:-1]
        lnk = lnk[start:-1]

        return lnk, lnT

    def lnt(self, lnk):
        """
        Generate transfer function with CAMB
        
        .. note :: This should not be called by the user!
        """
        self.t.transfer_options.update(self.option_defaults)

        cdict = dict(self.t.pycamb_dict,
                     **self.t.transfer_options)
        T = pycamb.transfers(**cdict)[1]
        T = np.log(T[[0, 6], :, 0])

        lnkout, lnT = self._check_low_k(T[0, :], T[1, :])

        return spline(lnkout, lnT, k=1)(lnk)

class FromFile(CAMB):
    def lnt(self, lnk):
        """
        Import the transfer function from file.
        
        The format should be the same as CAMB output, or a simple 2-column file.
        """
        try:
            T = np.log(np.genfromtxt(self.t.transfer_options["fname"])[:, [0, 6]].T)
        except IndexError:
            T = np.log(np.genfromtxt(self.t.transfer_options["fname"])[:, [0, 1]].T)

        lnkout, lnT = self._check_low_k(T[0, :], T[1, :])
        return spline(lnkout, lnT, k=1)(lnk)


class EH(GetTransfer):
    def lnt(self, lnk):
        """
        Eisenstein-Hu transfer function
        """

        T = np.log(cp.perturbation.transfer_function_EH(np.exp(lnk) * self.t.h,
                                    **self.t.cosmolopy_dict)[1])
        return T

class BBKS(GetTransfer):
    def lnt(self, lnk):
        """
        BBKS transfer function.
        """
        Gamma = self.t.omegam * self.t.h
        q = np.exp(lnk) / Gamma * np.exp(self.t.omegab + np.sqrt(2 * self.t.h) *
                               self.t.omegab / self.t.omegam)
        return np.log((np.log(1.0 + 2.34 * q) / (2.34 * q) *
                (1 + 3.89 * q + (16.1 * q) ** 2 + (5.47 * q) ** 3 +
                 (6.71 * q) ** 4) ** (-0.25)))

class BondEfs(GetTransfer):
    def lnt(self, lnk):
        """
        Bond and Efstathiou transfer function.
        """

        omegah2 = 1.0 / (self.t.omegam * self.t.h ** 2)

        a = 6.4 * omegah2
        b = 3.0 * omegah2
        c = 1.7 * omegah2
        nu = 1.13
        k = np.exp(lnk)
        return np.log((1 + (a * k + (b * k) ** 1.5 + (c * k) ** 2) ** nu) ** (-1 / nu))


class Transfer(Cosmology):
    '''
    Neatly deals with different transfer functions and their routines.
    
    The purpose of this class is to calculate transfer functions, power spectra
    and several tightly associated quantities using many of the available fits
    from the literature. 
        
    Importantly, it contains the means to calculate the transfer function using the
    popular CAMB code, the Eisenstein-Hu fit (1998), the BBKS fit or the Bond and
    Efstathiou fit (1984). Furthermore, it can calculate non-linear corrections
    using the halofit model (with updated parameters from Takahashi2012).
    
    The primary feature of this class is to wrap all the methods into a unified
    interface. On top of this, the class implements optimized updates of 
    parameters which is useful in, for example, MCMC code which covers a
    large parameter-space. Calling the `nonlinear_power` does not re-evaluate
    the entire transfer function, rather it just calculates the corrections, 
    improving performance.
    
    To update parameters optimally, use the update() method. 
    All output quantities are calculated only when needed (but stored after 
    first calculation for quick access).
    
    
    Parameters
    ----------
    lnk_min : float
        Defines min log wavenumber, *k* [units :math:`h Mpc^{-1}`]. 
        
    lnk_max : float
        Defines max log wavenumber, *k* [units :math:`h Mpc^{-1}`].
     
    dlnk : float
        Defines log interval between wavenumbers
        
    z : float, optional, default ``0.0``
        The redshift of the analysis.
                   
    wdm_mass : float, optional, default ``None``
        The warm dark matter particle size in *keV*, or ``None`` for CDM.
                                                                          
    transfer_fit : str, { ``"CAMB"``, ``"EH"``, ``"bbks"``, ``"bond_efs"``} 
        Defines which transfer function fit to use. If not defined from the
        listed options, it will be treated as a filename to be read in. In this
        case the file must contain a transfer function in CAMB output format. 
           
    Scalar_initial_condition : int, {1,2,3,4,5}
        (CAMB-only) Initial scalar perturbation mode (adiabatic=1, CDM iso=2, 
        Baryon iso=3,neutrino density iso =4, neutrino velocity iso = 5) 
        
    lAccuracyBoost : float, optional, default ``1.0``
        (CAMB-only) Larger to keep more terms in the hierarchy evolution
    
    AccuracyBoost : float, optional, default ``1.0``
        (CAMB-only) Increase accuracy_boost to decrease time steps, use more k 
        values,  etc.Decrease to speed up at cost of worse accuracy. 
        Suggest 0.8 to 3.
        
    w_perturb : bool, optional, default ``False``
        (CAMB-only) 
    
    transfer__k_per_logint : int, optional, default ``11``
        (CAMB-only) Number of wavenumbers estimated per log interval by CAMB
        Default of 11 gets best performance for requisite accuracy of mass function.
        
    transfer__kmax : float, optional, default ``0.25``
        (CAMB-only) Maximum value of the wavenumber.
        Default of 0.25 is high enough for requisite accuracy of mass function.
        
    ThreadNum : int, optional, default ``0``
        (CAMB-only) Number of threads to use for calculation of transfer 
        function by CAMB. Default 0 automatically determines the number.
                       
    kwargs : keywords
        The ``**kwargs`` take any cosmological parameters desired, which are 
        input to the `hmf.cosmo.Cosmology` class. `hmf.Perturbations` uses a 
        default parameter set from the first-year PLANCK mission, with optional 
        modifications by the user. Here is a list of parameters currently 
        available (and their defaults in `Transfer`):       
                 
        :sigma_8: [0.8344] The normalisation. Mass variance in top-hat spheres 
            with :math:`R=8Mpc h^{-1}`   
        :n: [0.9624] The spectral index 
        :w: [-1] The dark-energy equation of state
        :cs2_lam: [1] The constant comoving sound speed of dark energy
        :t_cmb: [2.725] Temperature of the CMB
        :y_he: [0.24] Helium fraction
        :N_nu: [3.04] Number of massless neutrino species
        :N_nu_massive: [0] Number of massive neutrino species
        :delta_c: [1.686] The critical overdensity for collapse
        :H0: [67.11] The hubble constant
        :h: [``H0/100.0``] The hubble parameter
        :omegan: [0] The normalised density of neutrinos
        :omegab_h2: [0.022068] The normalised baryon density by ``h**2``
        :omegac_h2: [0.12029] The normalised CDM density by ``h**2``
        :omegav: [0.6825] The normalised density of dark energy
        :omegab: [``omegab_h2/h**2``] The normalised baryon density
        :omegac: [``omegac_h2/h**2``] The normalised CDM density     
        :force_flat: [False] Whether to force the cosmology to be flat (affects only ``omegav``)
        :default: [``"planck1_base"``] A default set of cosmological parameters
    '''

    def __init__(self, z=0.0, lnk_min=np.log(1e-8),
                 lnk_max=np.log(2e4), dlnk=0.05,
                 wdm_mass=None, transfer_fit=CAMB,
                 transfer_options={}, **kwargs):
        '''
        Initialises some parameters
        '''
        # Call Cosmology init
        super(Transfer, self).__init__(**kwargs)

        # Set all given parameters
        self.lnk_min = lnk_min
        self.lnk_max = lnk_max
        self.dlnk = dlnk
        self.wdm_mass = wdm_mass
        self.z = z
        self.transfer_fit = transfer_fit
        self.transfer_options = transfer_options

    def update(self, **kwargs):
        """
        Update the class optimally with given arguments.
        
        Accepts any argument that the constructor takes
        """
        # Cosmology arguments are treated differently
        cosmo_kw = {k:v for k, v in kwargs.iteritems() if hasattr(Cosmology, k)}
        if cosmo_kw:
            self.cosmo_update(**cosmo_kw)
            # Remove cosmo from kwargs
            for k in cosmo_kw:
                del kwargs[k]

        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            del kwargs[k]

        if kwargs:
            raise ValueError("Invalid arguments: %s" % kwargs)

    #===========================================================================
    # Parameters
    #===========================================================================

    @parameter
    def transfer_options(self, val):
#         for v in val:
#             if v not in ['Scalar_initial_condition', 'scalar_amp', 'lAccuracyBoost',
#                          'AccuracyBoost', 'w_perturb', 'transfer__k_per_logint',
#                          'transfer__kmax', 'ThreadNum']:
#                 raise ValueError("%s not a valid camb option" % v)
        return val

    @parameter
    def lnk_min(self, val):
        return val

    @parameter
    def lnk_max(self, val):
        return val

    @parameter
    def dlnk(self, val):
        return val

    @parameter
    def z(self, val):
        try:
            val = float(val)
        except ValueError:
            raise ValueError("z must be a number (", val, ")")

        if val < 0:
            raise ValueError("z must be > 0 (", val, ")")

        return val

    @parameter
    def wdm_mass(self, val):
        if val is None:
            return val
        try:
            val = float(val)
        except ValueError:
            raise ValueError("wdm_mass must be a number (", val, ")")

        if val <= 0:
            raise ValueError("wdm_mass must be > 0 (", val, ")")
        return val

    @parameter
    def transfer_fit(self, val):
        if not HAVE_PYCAMB and val == "CAMB":
            raise ValueError("You cannot use the CAMB transfer since pycamb isn't installed")
        return val


    #===========================================================================
    # # ---- DERIVED PROPERTIES AND FUNCTIONS ---------------
    #===========================================================================
    @cached_property("lnk_min", "lnk_max", "dlnk")
    def lnk(self):
        return np.arange(self.lnk_min, self.lnk_max, self.dlnk)

    @cached_property("lnk", "pycamb_dict", "camb_options")
    def _unnormalised_lnT(self):
        """
        The un-normalised transfer function
        
        This wraps the individual transfer_fit methods to provide unified access.
        """
        try:
            return self.transfer_fit(self).lnt(self.lnk)
        except:
            return get_transfer(self.transfer_fit, self).lnt(self.lnk)

    @cached_property("n", "lnk", "_unnormalised_lnT")
    def _unnormalised_lnP(self):
        """
        Un-normalised CDM log power at :math:`z=0` [units :math:`Mpc^3/h^3`]
        """
        return self.n * self.lnk + 2 * self._unnormalised_lnT

    @cached_property("sigma_8", "_unnormalised_lnP", "lnk", "mean_dens")
    def _lnP_cdm_0(self):
        """
        Normalised CDM log power at z=0 [units :math:`Mpc^3/h^3`]
        """
        return tools.normalize(self.sigma_8,
                               self._unnormalised_lnP,
                               self.lnk, self.mean_dens)[0]

    @cached_property("sigma_8", "_unnormalised_lnT", "lnk", "mean_dens")
    def _lnT_cdm(self):
        """
        Normalised CDM log transfer function
        """
        return tools.normalize(self.sigma_8,
                               self._unnormalised_lnT,
                               self.lnk, self.mean_dens)

    @cached_property("wdm_mass", "_lnP_cdm_0", "lnk", "h", "omegac")
    def _lnP_0(self):
        """
        Normalised log power at :math:`z=0` (for CDM/WDM)
        """
        if self.wdm_mass is not None:
            print "doing this coz wdm_mass is ", self.wdm_mass
            return tools.wdm_transfer(self.wdm_mass, self._lnP_cdm_0,
                                      self.lnk, self.h, self.omegac)
        else:
            return self._lnP_cdm_0

    @cached_property("z", "omegam", "omegav", "omegak")
    def growth(self):
        r"""
        The growth factor :math:`d(z)`
        
        This is calculated (see Lukic 2007) as
        
        .. math:: d(z) = \frac{D^+(z)}{D^+(z=0)}
                
        where
        
        .. math:: D^+(z) = \frac{5\Omega_m}{2}\frac{H(z)}{H_0}\int_z^{\infty}{\frac{(1+z')dz'}{[H(z')/H_0]^3}}
        
        and
        
        .. math:: H(z) = H_0\sqrt{\Omega_m (1+z)^3 + (1-\Omega_m)}
        
        """
        if self.z > 0:
            return tools.growth_factor(self.z, self.cosmolopy_dict)
        else:
            return 1.0

    @cached_property("growth", "_lnP_0")
    def power(self):
        """
        Normalised log power spectrum [units :math:`Mpc^3/h^3`]
        """
        return 2 * np.log(self.growth) + self._lnP_0


    @cached_property("wdm_mass", "_lnT_cdm", "lnk", "h", "omegac")
    def transfer(self):
        """
        Normalised log transfer function for CDM/WDM
        """
        return tools.wdm_transfer(self.wdm_mass, self._lnT_cdm,
                                  self.lnk, self.h, self.omegac)

    @cached_property("lnk", "power")
    def delta_k(self):
        r"""
        Dimensionless power spectrum, :math:`\Delta_k = \frac{k^3 P(k)}{2\pi^2}`
        """
        return 3 * self.lnk + self.power - np.log(2 * np.pi ** 2)

    @cached_property("lnk", "nonlinear_delta_k")
    def nonlinear_power(self):
        """
        Non-linear log power [units :math:`Mpc^3/h^3`]
        
        Non-linear corrections come from HALOFIT (Smith2003) with updated
        parameters from Takahashi2012. 
        
        This code was heavily influenced by the HaloFit class from the 
        `chomp` python package by Christopher Morrison, Ryan Scranton 
        and Michael Schneider (https://code.google.com/p/chomp/). It has 
        been modified to improve its integration with this package.        
        """
        return -3 * self.lnk + self.nonlinear_delta_k + np.log(2 * np.pi ** 2)

    @cached_property("delta_k", "lnk", "z", "omegam", "omegav", "omegak", "omegan", 'w')
    def nonlinear_delta_k(self):
        r"""
        Dimensionless nonlinear power spectrum, :math:`\Delta_k = \frac{k^3 P_{\rm nl}(k)}{2\pi^2}`
        """
        rknl, rneff, rncur = _get_spec(self.lnk, self.delta_k, self.sigma_8)
        mask = np.exp(self.lnk) > 0.005
        plin = np.exp(self.delta_k[mask])
        k = np.exp(self.lnk[mask])
        pnl = halofit(k, self.z, self.omegam, self.omegav, self.w, self.omegan,
                      rneff, rncur, rknl, plin)
        nonlinear_delta_k = np.exp(self.delta_k)
        nonlinear_delta_k[mask] = pnl
        nonlinear_delta_k = np.log(nonlinear_delta_k)
        return nonlinear_delta_k

#====
