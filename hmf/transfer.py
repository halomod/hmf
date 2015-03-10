"""
This module contains a single class, `Transfer`, which provides methods to 
calculate the transfer function, matter power spectrum and several other 
related quantities. 
"""
import numpy as np
from cosmo import Cosmology
from _cache import cached_property, parameter
from halofit import _get_spec, halofit
from numpy import issubclass_
import astropy.units as u
# from tools import h_unit
from growth_factor import GrowthFactor
import transfer_models as tm
import tools
from _framework import get_model
from filters import TopHat

try:
    import pycamb
    HAVE_PYCAMB = True
except ImportError:
    HAVE_PYCAMB = False

class Transfer(Cosmology):
    '''
    Neatly deals with different transfer functions.
    
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
                       
    takahashi : bool, default ``True``
        Whether to use updated HALOFIT coefficients from Takahashi+12
        
    wdm_model : WDM subclass or string
        The WDM transfer function model to use
        
    kwargs : keywords
        The ``**kwargs`` take any cosmological parameters desired, which are 
        input to the `hmf.cosmo.Cosmology` class.
    '''

    def __init__(self, sigma_8=0.8, n=1.0, z=0.0, lnk_min=np.log(1e-8),
                 lnk_max=np.log(2e4), dlnk=0.05, transfer_fit=tm.CAMB,
                 transfer_options=None, takahashi=True, growth_model=GrowthFactor,
                 _growth_params=None, **kwargs):
        # Note the parameters that have empty dicts as defaults must be specified
        # as None, or the defaults themselves are updated!

        # Call Cosmology init
        super(Transfer, self).__init__(**kwargs)

        # Set all given parameters
        self.n = n
        self.sigma_8 = sigma_8
        self.growth_model = growth_model
        self._growth_params = _growth_params or {}
        self.lnk_min = lnk_min
        self.lnk_max = lnk_max
        self.dlnk = dlnk
        self.z = z
        self.transfer_fit = transfer_fit
        self.transfer_options = transfer_options or {}
        self.takahashi = takahashi


    #===========================================================================
    # Parameters
    #===========================================================================

    @parameter
    def growth_model(self, val):
        if not issubclass_(val, GrowthFactor) and not isinstance(val, basestring):
            raise ValueError("growth_model must be a GrowthFactor or string, got %s" % type(val))
        return val

    @parameter
    def _growth_params(self, val):
        return val

    @parameter
    def transfer_options(self, val):
#         for v in val:
#             if v not in ['Scalar_initial_condition', 'scalar_amp', 'lAccuracyBoost',
#                          'AccuracyBoost', 'w_perturb', 'transfer__k_per_logint',
#                          'transfer__kmax', 'ThreadNum']:
#                 raise ValueError("%s not a valid camb option" % v)
        return val

    @parameter
    def sigma_8(self, val):
        if val < 0.1 or val > 10:
            raise ValueError("sigma_8 out of bounds, %s" % val)
        return val

    @parameter
    def n(self, val):
        if val < -3 or val > 4:
            raise ValueError("n out of bounds, %s" % val)
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
    def takahashi(self, val):
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
    def transfer_fit(self, val):
        if not HAVE_PYCAMB and (val == "CAMB" or val == tm.CAMB):
            raise ValueError("You cannot use the CAMB transfer since pycamb isn't installed")
        if not (issubclass_(val, tm.Transfer) or isinstance(val, basestring)):
            raise ValueError("transfer_fit must be string or Transfer subclass")
        return val


    #===========================================================================
    # DERIVED PROPERTIES AND FUNCTIONS
    #===========================================================================
    @cached_property("lnk_min", "lnk_max", "dlnk")
    def k(self):
        return np.exp(np.arange(self.lnk_min, self.lnk_max, self.dlnk)) * self._hunit / u.Mpc

    @cached_property("k", "cosmo", "transfer_options", "transfer_fit")
    def _unnormalised_lnT(self):
        """
        The un-normalised transfer function
        
        This wraps the individual transfer_fit methods to provide unified access.
        """
        if issubclass_(self.transfer_fit, tm.Transfer):
            return self.transfer_fit(self.cosmo, **self.transfer_options).lnt(np.log(self.k.value))
        elif isinstance(self.transfer_fit, basestring):
            return get_model(self.transfer_fit, "hmf.transfer_models", cosmo=self.cosmo,
                             **self.transfer_options).lnt(np.log(self.k.value))

    @cached_property("n", "k", "_unnormalised_lnT")
    def _unnormalised_power(self):
        """
        Un-normalised CDM power at :math:`z=0` [units :math:`Mpc^3/h^3`]
        """
        return self.k.value ** self.n * np.exp(self._unnormalised_lnT) ** 2 * u.Mpc ** 3 / self._hunit ** 3

    @cached_property("mean_density0", "k", "_unnormalised_power", "sigma_8")
    def _normalisation(self):
        filter = TopHat(self.mean_density0, None, self.k, self._unnormalised_power)
        sigma_8 = filter.sigma(8.0 * u.Mpc / self._hunit)[0]

        # Calculate the normalization factor
        return self.sigma_8 / sigma_8

    @cached_property("_normalisation", "_unnormalised_power")
    def _power0(self):
        """
        Normalised power spectrum at z=0 [units :math:`Mpc^3/h^3`]
        """
        return self._normalisation ** 2 * self._unnormalised_power

#     @cached_property("sigma_8", "_unnormalised_lnT", "lnk", "mean_density0")
#     def _lnT(self):
#         """
#         Normalised CDM log transfer function
#         """
#         return tools.normalize(self.sigma_8,
#                                self._unnormalised_lnT,
#                                self.lnk, self.mean_density0)[0]

    @cached_property("cosmo", "growth_model", "_growth_params")
    def _growth(self):
        if issubclass_(self.growth_model, GrowthFactor):
            return self.growth_model(self.cosmo, **self._growth_params)
        else:
            return get_model(self.growth_model, "hmf.growth_factor", cosmo=self.cosmo,
                             **self._growth_params)

    @cached_property("z", "_growth")
    def growth_factor(self):
        r"""
        The growth factor :math:`d(z)`
        
        This is calculated (see Lukic 2007) as
        
        .. math:: d(z) = \frac{D^+(z)}{D^+(z=0)}
                
        where
        
        .. math:: D^+(z) = \frac{5\Omega_m}{2}\frac{H(z)}{H_0}\int_z^{\infty}{\frac{(1+z')dz'}{[H(z')/H_0]^3}}        
        """
        if self.z > 0:
            return self._growth.growth_factor(self.z)
        else:
            return 1.0

    @cached_property("growth", "_power0")
    def power(self):
        """
        Normalised log power spectrum [units :math:`Mpc^3/h^3`]
        """
        return self.growth ** 2 * self._power0

    @cached_property("k", "power")
    def delta_k(self):
        r"""
        Dimensionless power spectrum, :math:`\Delta_k = \frac{k^3 P(k)}{2\pi^2}`
        """
        return self.k ** 3 * self.power / (2 * np.pi ** 2)

    @cached_property("k", "nonlinear_delta_k")
    def nonlinear_power(self):
        """
        Non-linear log power [units :math:`Mpc^3/h^3`]
        
        Non-linear corrections come from HALOFIT (Smith2003) with updated
        parameters from Takahashi2012. 
        """
        return self.k ** -3 * self.nonlinear_delta_k * (2 * np.pi ** 2)

    @cached_property("delta_k", "k", "z", "sigma_8", "cosmo", 'takahashi')
    def nonlinear_delta_k(self):
        r"""
        Dimensionless nonlinear power spectrum, :math:`\Delta_k = \frac{k^3 P_{\rm nl}(k)}{2\pi^2}`
        """
        rknl, rneff, rncur = _get_spec(self.k, self.delta_k, self.sigma_8)
        mask = self.k.value > 0.005
        plin = self.delta_k[mask]
        k = self.k[mask]
        pnl = halofit(k.value, self.z, self.cosmo, rneff, rncur, rknl, plin, self.takahashi)
        nonlinear_delta_k = self.delta_k
        nonlinear_delta_k[mask] = pnl
        return nonlinear_delta_k




