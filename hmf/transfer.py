"""
Module providing a framework for transfer functions.

This module contains a single class, `Transfer`, which provides methods to
calculate the transfer function, matter power spectrum and several other
related quantities.
"""
import numpy as np
import cosmo
from _cache import cached_property, parameter
from halofit import halofit as _hfit
import growth_factor as gf
import transfer_models as tm
from _framework import get_model
import filters

try:
    import pycamb
    HAVE_PYCAMB = True
except ImportError:
    HAVE_PYCAMB = False

class Transfer(cosmo.Cosmology):
    '''
    Neatly deals with different transfer functions.

    The purpose of this :class:`hmf._frameworks.Framework` is to calculate
    transfer functions, power spectra and several tightly associated
    quantities given a basic model for the transfer function.

    Included are non-linear corrections using the halofit model
    (with updated parameters from Takahashi2012).

    As in all frameworks, to update parameters optimally, use the
    update() method. All output quantities are calculated only when needed
    (but stored after first calculation for quick access).

    Parameters
    ----------
    sigma_8 : float, optional
        RMS linear density fluctations in spheres of radius 8 Mpc/h

    n : float, optional
        Spectral index of fluctations

    z : float, optional
        Redshift.

    lnk_min : float, optional
        Defines min log wavenumber, *k* [units :math:`h Mpc^{-1}`].

    lnk_max : float, optional
        Defines max log wavenumber, *k* [units :math:`h Mpc^{-1}`].

    dlnk : float
        Defines log interval between wavenumbers

    transfer_model : str or :class:`hmf.transfer_models.TransferComponent` subclass, optional
        Defines which transfer function model to use. Built-in available models
        are found in the :mod:`hmf.transfer_models` module. Default is CAMB if installed,
        otherwise EH.

    transfer_params : dict
        Relevant parameters of the `transfer_model`.

    takahashi : bool, optional
        Whether to use updated HALOFIT coefficients from Takahashi+12

    growth_model : str or `hmf.growth_factor.GrowthFactor` subclass, optional
        The model to use to calculate the growth function/growth rate.

    growth_params : dict
        Relevant parameters of the `growth_model`.

    kwargs : keywords
        The ``**kwargs`` take any cosmological parameters desired, which are
        input to the `hmf.cosmo.Cosmology` class.
    '''

    def __init__(self, sigma_8=0.8344, n=0.9624, z=0.0, lnk_min=np.log(1e-8),
                 lnk_max=np.log(2e4), dlnk=0.05, transfer_model=tm.CAMB if HAVE_PYCAMB else tm.EH,
                 transfer_params=None, takahashi=True, growth_model=gf.GrowthFactor,
                 growth_params=None, **kwargs):
        # Note the parameters that have empty dicts as defaults must be specified
        # as None, or the defaults themselves are updated!

        # Call Cosmology init
        super(Transfer, self).__init__(**kwargs)

        # Set all given parameters
        self.n = n
        self.sigma_8 = sigma_8
        self.growth_model = growth_model
        self.growth_params = growth_params or {}
        self.lnk_min = lnk_min
        self.lnk_max = lnk_max
        self.dlnk = dlnk
        self.z = z
        self.transfer_model = transfer_model
        self.transfer_params = transfer_params or {}
        self.takahashi = takahashi


    #===========================================================================
    # Parameters
    #===========================================================================

    @parameter
    def growth_model(self, val):
        if not np.issubclass_(val, gf.GrowthFactor) and not isinstance(val, basestring):
            raise ValueError("growth_model must be a GrowthFactor or string, got %s" % type(val))
        return val

    @parameter
    def growth_params(self, val):
        return val

    @parameter
    def transfer_params(self, val):
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
    def transfer_model(self, val):
        if not HAVE_PYCAMB and (val == "CAMB" or val == tm.CAMB):
            raise ValueError("You cannot use the CAMB transfer since pycamb isn't installed")
        if not (np.issubclass_(val, tm.Transfer) or isinstance(val, basestring)):
            raise ValueError("transfer_model must be string or Transfer subclass")
        return val


    #===========================================================================
    # DERIVED PROPERTIES AND FUNCTIONS
    #===========================================================================
    @cached_property("lnk_min", "lnk_max", "dlnk")
    def k(self):
        "Wavenumbers, [h/Mpc]"
        return np.exp(np.arange(self.lnk_min, self.lnk_max, self.dlnk))

    @cached_property("k", "cosmo", "transfer_params", "transfer_model")
    def _unnormalised_lnT(self):
        """
        The un-normalised transfer function.
        """
        if np.issubclass_(self.transfer_model, tm.Transfer):
            return self.transfer_model(self.cosmo, **self.transfer_params).lnt(np.log(self.k))
        elif isinstance(self.transfer_model, basestring):
            return get_model(self.transfer_model, "hmf.transfer_models", cosmo=self.cosmo,
                             **self.transfer_params).lnt(np.log(self.k))

    @cached_property("n", "k", "_unnormalised_lnT")
    def _unnormalised_power(self):
        """
        Un-normalised CDM power at :math:`z=0` [units :math:`Mpc^3/h^3`]
        """
        return self.k ** self.n * np.exp(self._unnormalised_lnT) ** 2


    @cached_property("mean_density0", "k", "_unnormalised_power")
    def _unn_sig8(self):
        # Always use a TopHat for sigma_8
        filter = filters.TopHat(self.k, self._unnormalised_power)
        return filter.sigma(8.0)[0]

    @cached_property("_unn_sig8", "sigma_8")
    def _normalisation(self):
        # Calculate the normalization factor
        return self.sigma_8 / self._unn_sig8

    @cached_property("_normalisation", "_unnormalised_power")
    def _power0(self):
        """
        Normalised power spectrum at z=0 [units :math:`Mpc^3/h^3`]
        """
        return self._normalisation ** 2 * self._unnormalised_power

    @cached_property("sigma_8", "_unnormalised_lnT", "lnk", "mean_density0")
    def _transfer(self):
        """
        Normalised CDM log transfer function
        """
        return self._normalisation * np.exp(self._unnormalised_lnT)

    @cached_property("cosmo", "growth_model", "growth_params")
    def growth(self):
        "The growth model class"
        if np.issubclass_(self.growth_model, gf.GrowthFactor):
            return self.growth_model(self.cosmo, **self.growth_params)
        else:
            return get_model(self.growth_model, "hmf.growth_factor", cosmo=self.cosmo,
                             **self._growth_params)

    @cached_property("z", "growth")
    def growth_factor(self):
        r"""
        The growth factor
        """
        if self.z > 0:
            return self.growth.growth_factor(self.z)
        else:
            return 1.0

    @cached_property("growth_factor", "_power0")
    def power(self):
        """
        Normalised log power spectrum [units :math:`Mpc^3/h^3`]
        """
        return self.growth_factor ** 2 * self._power0

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

        Non-linear corrections come from HALOFIT.
        """
        return self.k ** -3 * self.nonlinear_delta_k * (2 * np.pi ** 2)

    @cached_property("delta_k", "k", "z", "sigma_8", "cosmo", 'takahashi')
    def nonlinear_delta_k(self):
        r"""
        Dimensionless nonlinear power spectrum, :math:`\Delta_k = \frac{k^3 P_{\rm nl}(k)}{2\pi^2}`
        """
        # rknl, rneff, rncur = _get_spec(self.k, self.delta_k, self.sigma_8)
        # mask = self.k > 0.005
        # plin = self.delta_k[mask]
        # k = self.k[mask]
        # pnl = halofit(k, self.z, self.cosmo, rneff, rncur, rknl, plin, self.takahashi)
        # nonlinear_delta_k = self.delta_k.copy()
        # nonlinear_delta_k[mask] = pnl
        return _hfit(self.k,self.delta_k,self.sigma_8,self.z,self.cosmo)
