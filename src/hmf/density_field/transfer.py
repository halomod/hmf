"""
Module providing a framework for transfer functions.

This module contains a single class, `Transfer`, which provides methods to
calculate the transfer function, matter power spectrum and several other
related quantities.
"""
import numpy as np

from .._internals._cache import cached_quantity, parameter
from .._internals._framework import get_mdl
from ..cosmology import cosmo
from ..density_field import filters
from ..density_field import transfer_models as tm
from .halofit import halofit as _hfit
from .transfer_models import HAVE_CAMB


class Transfer(cosmo.Cosmology):
    """
    A transfer function framework.

    The purpose of this :class:`hmf._frameworks.Framework` is to calculate
    transfer functions, power spectra and several tightly associated
    quantities given a basic model for the transfer function.

    As in all frameworks, to update parameters optimally, use the
    :meth:`update` method. All output quantities are calculated only when needed
    (but stored after first calculation for quick access).

    In addition to the parameters directly passed to this class, others are available
    which are passed on to its superclass. To read a standard documented list of (all)
    parameters, use ``Transfer.parameter_info()``. If you want to just see the plain
    list of available parameters, use ``Transfer.get_all_parameters()``.To see the
    actual defaults for each parameter, use ``Transfer.get_all_parameter_defaults()``.

    By default, the `growth_model` is :class:`~growth_factor.GrowthFactor`. However, if
    using a wCDM cosmology and camb is installed, it will default to
    :class:`~growth_factor.CambGrowth`.
    """

    def __init__(
        self,
        sigma_8=0.8159,
        n=0.9667,
        z=0.0,
        lnk_min=np.log(1e-8),  # noqa: B008
        lnk_max=np.log(2e4),  # noqa: B008
        dlnk=0.05,
        transfer_model=tm.CAMB if HAVE_CAMB else tm.EH,
        transfer_params=None,
        takahashi=True,
        growth_model=None,
        growth_params=None,
        use_splined_growth=False,
        **kwargs,
    ):

        # Call Cosmology init
        super().__init__(**kwargs)

        # Set all given parameters
        self.n = n
        self.sigma_8 = sigma_8
        self.growth_params = growth_params or {}
        self.use_splined_growth = use_splined_growth
        self.lnk_min = lnk_min
        self.lnk_max = lnk_max
        self.dlnk = dlnk
        self.z = z
        self.transfer_model = transfer_model
        self.transfer_params = transfer_params or {}
        self.takahashi = takahashi

        # Growth model has a more complicated default.
        # We set it here so that "None" is not a relevant option for self.growth_model
        # (and it can't be explicitly updated to None).
        if growth_model is None:
            if hasattr(self.cosmo, "w0") and HAVE_CAMB:
                self.growth_model = "CambGrowth"
            else:
                self.growth_model = "GrowthFactor"
        else:
            self.growth_model = growth_model

    # ===========================================================================
    # Parameters
    # ===========================================================================
    def validate(self):
        super().validate()
        assert (
            self.lnk_min < self.lnk_max
        ), f"lnk_min >= lnk_max: {self.lnk_min}, {self.lnk_max}"
        assert len(self.k) > 1, f"len(k) < 2: {len(self.k)}"

    @parameter("model")
    def growth_model(self, val):
        """
        The model to use to calculate the growth function/growth rate.

        :type: `hmf.growth_factor._GrowthFactor` subclass
        """
        return get_mdl(val, "_GrowthFactor")

    @parameter("param")
    def growth_params(self, val):
        """
        Relevant parameters of the :attr:`growth_model`.

        :type: dict
        """
        return val

    @parameter("model")
    def transfer_model(self, val):
        """
        Defines which transfer function model to use.

        Built-in available models are found in the :mod:`hmf.transfer_models` module.
        Default is CAMB if installed, otherwise EH.

        :type: str or :class:`hmf.transfer_models.TransferComponent` subclass, optional
        """
        if not HAVE_CAMB and val in ["CAMB", tm.CAMB]:
            raise ValueError(
                "You cannot use the CAMB transfer since pycamb isn't installed"
            )
        return get_mdl(val, "TransferComponent")

    @parameter("param")
    def transfer_params(self, val):
        """
        Relevant parameters of the `transfer_model`.

        :type: dict
        """
        return val

    @parameter("param")
    def sigma_8(self, val):
        """
        RMS linear density fluctuations in spheres of radius 8 Mpc/h

        :type: float
        """
        if val < 0.1 or val > 10:
            raise ValueError("sigma_8 out of bounds, %s" % val)
        return val

    @parameter("param")
    def n(self, val):
        """
        Spectral index of fluctuations

        Must be greater than -3 and less than 4.

        :type: float
        """
        if val < -3 or val > 4:
            raise ValueError("n out of bounds, %s" % val)
        return val

    @parameter("res")
    def lnk_min(self, val):
        """
        Minimum (natural) log wave-number, :attr:`k` [h/Mpc].

        :type: float
        """
        return val

    @parameter("res")
    def lnk_max(self, val):
        """
        Maximum (natural) log wave-number, :attr:`k` [h/Mpc].

        :type: float
        """
        return val

    @parameter("res")
    def dlnk(self, val):
        """
        Step-size of log wave-numbers

        :type: float
        """
        return val

    @parameter("switch")
    def takahashi(self, val):
        """
        Whether to use updated HALOFIT coefficients from Takahashi+12.

        If False, use the original coefficients from Smith+2003.

        :type: bool
        """
        return bool(val)

    @parameter("param")
    def z(self, val):
        """
        Redshift.

        Must be greater than 0.

        :type: float
        """
        try:
            val = float(val)
        except ValueError:
            raise ValueError("z must be a number (", val, ")")

        if val < 0:
            raise ValueError("z must be > 0 (", val, ")")

        return val

    # ===========================================================================
    # DERIVED PROPERTIES AND FUNCTIONS
    # ===========================================================================
    @cached_quantity
    def k(self):
        """Wavenumbers, [h/Mpc]"""
        return np.exp(np.arange(self.lnk_min, self.lnk_max, self.dlnk))

    @cached_quantity
    def transfer(self):
        """
        The instantiated transfer model
        """
        return self.transfer_model(self.cosmo, **self.transfer_params)

    @cached_quantity
    def _unnormalised_lnT(self):
        """
        The un-normalised transfer function.
        """
        return self.transfer.lnt(np.log(self.k))

    @cached_quantity
    def _unnormalised_power(self):
        """
        Un-normalised CDM power at :math:`z=0` [units :math:`Mpc^3/h^3`]
        """
        return self.k ** self.n * np.exp(self._unnormalised_lnT) ** 2

    @cached_quantity
    def _unn_sig8(self):
        # Always use a TopHat for sigma_8, and always use full k-range
        if self.lnk_min > -15 or self.lnk_max < 9:
            lnk = np.arange(-8, 8, self.dlnk)
            t = self.transfer.lnt(lnk)
            p = np.exp(lnk) ** self.n * np.exp(t) ** 2
            filt = filters.TopHat(np.exp(lnk), p)
        else:
            filt = filters.TopHat(self.k, self._unnormalised_power)

        return filt.sigma(8.0)[0]

    @cached_quantity
    def _normalisation(self):
        # Calculate the normalization factor
        return self.sigma_8 / self._unn_sig8

    @cached_quantity
    def _power0(self):
        """
        Normalised power spectrum at z=0 [units :math:`Mpc^3/h^3`]
        """
        return self._normalisation ** 2 * self._unnormalised_power

    @cached_quantity
    def transfer_function(self):
        """Normalised CDM log transfer function."""
        return self._normalisation * np.exp(self._unnormalised_lnT)

    @cached_quantity
    def growth(self):
        """The instantiated growth model."""
        return self.growth_model(self.cosmo, **self.growth_params)

    @cached_quantity
    def _growth_factor_fn(self):
        """Function that efficiently returns the growth factor."""
        return self.growth.growth_factor_fn()

    @cached_quantity
    def growth_factor(self):
        r"""The growth factor."""
        if self.use_splined_growth:
            return self._growth_factor_fn(self.z)
        else:
            return self.growth.growth_factor(self.z)

    @cached_quantity
    def power(self):
        """Normalised log power spectrum [units :math:`Mpc^3/h^3`]."""
        return self.growth_factor ** 2 * self._power0

    @cached_quantity
    def delta_k(self):
        r"""
        Dimensionless power spectrum, :math:`\Delta_k = \frac{k^3 P(k)}{2\pi^2}`.
        """
        return self.k ** 3 * self.power / (2 * np.pi ** 2)

    @cached_quantity
    def nonlinear_power(self):
        """
        Non-linear log power [units :math:`Mpc^3/h^3`].

        Non-linear corrections come from HALOFIT.
        """
        return self.k ** -3 * self.nonlinear_delta_k * (2 * np.pi ** 2)

    @cached_quantity
    def nonlinear_delta_k(self):
        r"""
        Dimensionless nonlinear power spectrum.

        .. math:: \Delta_k = \frac{k^3 P_{\rm nl}(k)}{2\pi^2}
        """
        return _hfit(
            self.k, self.delta_k, self.sigma_8, self.z, self.cosmo, self.takahashi
        )
