"""
The primary module for user-interaction with the :mod:`hmf` package.

The module contains a single class, `MassFunction`, which wraps almost all the
functionality of :mod:`hmf` in an easy-to-use way.
"""

import copy
import numpy as np
import warnings
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.optimize import minimize
from typing import Any, Dict, Optional, Union

from .._internals._cache import cached_quantity, parameter
from .._internals._framework import get_mdl
from ..density_field import transfer
from ..density_field.filters import Filter, TopHat
from ..halos.mass_definitions import MassDefinition as md
from ..halos.mass_definitions import SOGeneric, SOMean
from . import fitting_functions as ff
from .integrate_hmf import hmf_integral_gtm as int_gtm


class MassFunction(transfer.Transfer):
    """
    An object containing all relevant quantities for the mass function.

    The purpose of this class is to calculate many quantities associated with
    the dark matter halo mass function (HMF). The class is initialized to form a
    cosmology and takes in various options as to how to calculate all
    further quantities.

    Most outputs are provided as ``@cached_quantity`` attributes for ease of
    access.

    Contains an :meth:`~update` method which can be passed arguments to update, in the
    most optimal manner. All output quantities are calculated only when needed
    (but stored after first calculation for quick access).

    In addition to the parameters directly passed to this class, others are available
    which are passed on to its superclass (:class:`Transfer`).
    To read a standard documented list of (all) available
    parameters, use :func:`~parameter_info`. If you want to just see the plain
    list of available parameters, use :func:`~get_all_parameters`. To see the
    actual defaults for each parameter, use :func:`get_all_parameter_defaults`.

    Parameters
    ----------
    Mmin
        The log10 of the minimum halo mass to compute quantities for (units Msun/h).
    Mmax
        The log10 of the minimum halo mass to compute quantities for (units Msun/h).
    dlog10m
        The resolution of the mass array (logarithmically spaced).
    hmf_model
        The HMF fitting function to use.
    hmf_params
        Parameters specific to the chosen HMF.
    mdef_model
        The mass definition model to use. By default, use the mass definition in which
        the chosen HMF was measured. If that does not exist, use ``SOMean(200)``. If set,
        this must be compatible with the halo definition used in measuring the chosen
        HMF -- unless ``disable_mass_conversion`` is set to False, in which case the
        HMF is automatically translated to a new mass definition.
    mdef_params
        Parameters specific to the mass definition model. Typically "overdensity" or
        "linking_length".
    delta_c
        The critical overdensity for collapse.
    filter_model
        A model to compute the

    Examples
    --------
    Since all parameters have reasonable defaults, the most obvious thing to do is

    >>> h = MassFunction()
    >>> h.dndm

    Many different parameters may be passed, both models and parameters of those models.
    For instance:

    >>> h = MassFunction(z=1.0,Mmin=8,hmf_model="SMT")
    >>> h.dndm

    Once instantiated, changing parameters should be done through the :meth:`update`
    method:

    >>> h.update(z=2)
    >>> h.dndm
    """

    #: Switch to turn off exceptions for mdef's not matching hmf_model
    ERROR_ON_BAD_MDEF = True

    def __init__(
        self,
        Mmin: float = 10.0,
        Mmax: float = 15.0,
        dlog10m: float = 0.01,
        hmf_model: Union[str, ff.FittingFunction] = ff.Tinker08,
        hmf_params: Optional[Dict[str, Any]] = None,
        mdef_model: Union[None, str, md] = None,
        mdef_params: Union[dict, None] = None,
        delta_c: float = 1.686,
        filter_model: Union[str, Filter] = TopHat,
        filter_params: Union[dict, None] = None,
        disable_mass_conversion: bool = True,
        **transfer_kwargs,
    ):
        # Call super init MUST BE DONE FIRST.
        super().__init__(**transfer_kwargs)

        # Set all given parameters.
        self.hmf_model = hmf_model
        self.Mmin = Mmin
        self.Mmax = Mmax
        self.dlog10m = dlog10m
        self.mdef_model = mdef_model
        self.mdef_params = mdef_params or {}
        self.delta_c = delta_c
        self.hmf_params = hmf_params or {}
        self.filter_model = filter_model
        self.filter_params = filter_params or {}
        self.disable_mass_conversion = disable_mass_conversion

    # ===========================================================================
    # PARAMETERS
    # ===========================================================================
    def validate(self):
        super().validate()
        assert self.Mmin < self.Mmax, f"Mmin > Mmax: {self.Mmin}, {self.Mmax}"
        assert len(self.m) > 0, "mass vector has length zero!"

        # Check whether the hmf component validates.
        self.hmf

    @parameter("res")
    def Mmin(self, val):
        r"""
        Minimum mass at which to perform analysis [units :math:`\log_{10}M_\odot h^{-1}`].

        :type: float
        """
        return val

    @parameter("res")
    def Mmax(self, val):
        r"""
        Maximum mass at which to perform analysis [units :math:`\log_{10}M_\odot h^{-1}`].

        :type: float
        """
        return val

    @parameter("res")
    def dlog10m(self, val) -> float:
        """
        log10 interval between mass bins

        :type: float
        """
        return val

    @parameter("switch")
    def disable_mass_conversion(self, val) -> bool:
        """Disable converting mass function from builtin definition to that provided.

        :type: bool
        """
        return bool(val)

    @parameter("model")
    def filter_model(self, val):
        """
        A model for the window/filter function.

        :type: :class:`hmf.filters.Filter` subclass
        """
        return get_mdl(val, "Filter")

    @parameter("param")
    def filter_params(self, val):
        """
        Model parameters for `filter_model`.

        :type: dict
        """
        return val

    @parameter("param")
    def delta_c(self, val):
        r"""
        The critical overdensity for collapse, :math:`\delta_c`.

        :type: float
        """
        try:
            val = float(val)
        except ValueError:
            raise ValueError("delta_c must be a number: ", val)

        if val <= 0:
            raise ValueError("delta_c must be > 0 (", val, ")")
        if val > 10.0:
            raise ValueError("delta_c must be < 10.0 (", val, ")")

        return val

    @parameter("model")
    def hmf_model(self, val):
        r"""
        A model to use as the fitting function :math:`f(\sigma)`

        :type: str or `hmf.fitting_functions.FittingFunction` subclass
        """
        if val is None:
            return val
        return get_mdl(val, "FittingFunction")

    @parameter("param")
    def hmf_params(self, val):
        """
        Model parameters for `hmf_model`.

        :type: dict
        """
        return val

    @parameter("model")
    def mdef_model(self, val):
        """
        A model to use as the mass definition.

        :type: str or :class:`hmf.halos.mass_definitions.MassDefinition` subclass
        """
        if val is None or (isinstance(val, str) and val.lower() == "none"):
            return None
        return get_mdl(val, "MassDefinition")

    @parameter("param")
    def mdef_params(self, val):
        """
        Model parameters for `mdef_model`.
        :type: dict
        """
        return val

    # --------------------------------  PROPERTIES ------------------------------
    @cached_quantity
    def mean_density(self):
        """Mean density of universe at redshift z."""
        return self.mean_density0 * (1 + self.z) ** 3

    @cached_quantity
    def mdef(self) -> md:
        """The halo mass-definition model instance.

        Default mass definition is the one the chosen hmf model was measured with.
        """
        if self.mdef_model is None:
            # Get the default from the mass function definition
            mdef = self.hmf_model.get_measured_mdef()

            # Generic SO definitions, like in Tinker08, which can natively support
            # any SO definition, also provide a "preferred" definition to use as
            # the default.
            if isinstance(mdef, SOGeneric):
                mdef = mdef.preferred

            # Update the actual parameters if the user has supplied any explicitly.
            if self.mdef_params:
                mdef.params.update(self.mdef_params)

            if mdef is None:
                # Some mass functions don't have any set mass definition (eg. PS)
                mdef = SOMean(**self.mdef_params)
        else:
            mdef = self.mdef_model(**self.mdef_params)

            # Note we need to do the != in this order so that SOGeneric can compare.
            if self.hmf_model.get_measured_mdef() != mdef:
                if self.disable_mass_conversion and self.ERROR_ON_BAD_MDEF:
                    raise ValueError(
                        f"Your input mass definition '{mdef}' does not match the mass "
                        f"definition in which the hmf fit {self.hmf_model.__name__} was "
                        f"measured: '{self.hmf_model.get_measured_mdef()}' Either allow "
                        f"automatic mass conversion by setting `disable_mass_conversion=False, "
                        "or use the correct mass definition."
                    )
                extra_msg = (
                    "The mass function will be "
                    "converted to your input definition, "
                    "but note that some properties do not survive the conversion, eg. "
                    "the integral of the hmf over mass yielding the total mean density."
                )

                warnings.warn(
                    f"Your input mass definition '{mdef}' does not match the mass "
                    f"definition in which the hmf fit {self.hmf_model.__name__} was measured:"
                    f"'{self.hmf_model.get_measured_mdef()}'. {extra_msg if not self.disable_mass_conversion else ''}"
                )

        return mdef

    @cached_quantity
    def hmf(self):
        """Instantiated model for the hmf fitting function."""
        return self.hmf_model(
            m=self.m,
            nu2=self.nu,
            z=self.z,
            mass_definition=self.mdef,
            cosmo=self.cosmo,
            delta_c=self.delta_c,
            n_eff=self.n_eff,
            **self.hmf_params,
        )

    @cached_quantity
    def filter(self):
        """Instantiated model for filter/window functions.

        Note that this filter is *not* normalised -- i.e. the output of `filter.sigma(8)`
        will not be the input `sigma_8`.
        """
        return self.filter_model(self.k, self._unnormalised_power, **self.filter_params)

    @cached_quantity
    def halo_overdensity_mean(self):
        """The halo overdensity with respect to the mean background."""
        return self.mdef.halo_overdensity_mean(self.z, self.cosmo)

    @cached_quantity
    def halo_overdensity_crit(self):
        """The halo overdensity with respect to the critical density."""
        return self.mdef.halo_overdensity_crit(self.z, self.cosmo)

    @cached_quantity
    def normalised_filter(self):
        """A normalised filter, such that filter.sigma(8) == sigma8"""
        return self.filter_model(self.k, self.power, **self.filter_params)

    @cached_quantity
    def m(self):
        """Halo masses (defined via ``mdef``)."""
        return 10 ** np.arange(self.Mmin, self.Mmax, self.dlog10m)

    @cached_quantity
    def _unn_sigma0(self):
        """Un-normalised mass variance at z=0."""
        return self.filter.sigma(self.radii)

    @cached_quantity
    def _sigma_0(self):
        r"""The normalised mass variance at z=0 :math:`\sigma`."""
        return self._normalisation * self._unn_sigma0

    @cached_quantity
    def radii(self):
        """The radii corresponding to the masses `m`.

        Note that these are not the halo radii -- they are the radii containing mass
        m given a purely background density.
        """
        return self.filter.mass_to_radius(self.m, self.mean_density0)

    @cached_quantity
    def _dlnsdlnm(self):
        r"""
        The value of :math:`\left|\frac{\d \ln \sigma}{\d \ln m}\right|`, ``len=len(m)``

        Notes
        -----
        .. math:: frac{d\ln\sigma}{d\ln m} = \frac{3}{2\sigma^2\pi^2R^4}\int_0^\infty \frac{dW^2(kR)}{dM}\frac{P(k)}{k^2}dk
        """
        return 0.5 * self.filter.dlnss_dlnm(self.radii)

    @cached_quantity
    def sigma(self):
        """The sqrt of the mass variance at `z`, ``len=len(m)``."""
        return self._sigma_0 * self.growth_factor

    @cached_quantity
    def nu(self):
        r"""
        The parameter :math:`\nu = \left(\frac{\delta_c}{\sigma}\right)^2`, ``len=len(m)``
        """
        return (self.delta_c / self.sigma) ** 2

    @cached_quantity
    def mass_nonlinear(self):
        """The nonlinear mass, nu(Mstar) = 1."""
        if self.nu.min() > 1 or self.nu.max() < 1:
            warnings.warn("Nonlinear mass outside mass range")
            if self.nu.min() > 1:
                startr = np.log(self.radii.min())
            else:
                startr = np.log(self.radii.max())

            model = (
                lambda lnr: (
                    self.filter.sigma(np.exp(lnr))
                    * self._normalisation
                    * self.growth_factor
                    - self.delta_c
                )
                ** 2
            )

            res = minimize(
                model,
                [
                    startr,
                ],
            )

            if res.success:
                r = np.exp(res.x[0])
                return self.filter.radius_to_mass(r, self.mean_density0)
            else:
                warnings.warn("Minimization failed :(")
                return 0
        else:
            nu = spline(self.nu, self.m, k=5)
            return nu(1)

    @cached_quantity
    def lnsigma(self):
        """Natural log of inverse mass variance, ``len=len(m)``."""
        return np.log(1 / self.sigma)

    @cached_quantity
    def n_eff(self):
        """
        Effective spectral index at scale of halo radius, ``len=len(m)``

        Notes
        -----
        This function, and any derived quantities, can show small non-physical
        'wiggles' at the 0.1% level, if too coarse a grid in ln(k) is used. If
        applications are sensitive at this level, please use a very fine k-space
        grid.

        Uses eq. 42 in Lukic et. al 2007.
        """
        return -3.0 * (2.0 * self._dlnsdlnm + 1.0)

    @cached_quantity
    def fsigma(self):
        r"""
        The multiplicity function, :math:`f(\sigma)`, for `hmf_model`. ``len=len(m)``
        """
        return self.hmf.fsigma

    @cached_quantity
    def dndm(self):
        r"""
        The number density of haloes, ``len=len(m)`` [units :math:`h^4 M_\odot^{-1} Mpc^{-3}`]
        """
        # if self.z2 is None:  # #This is normally the case
        dndm = self.fsigma * self.mean_density0 * np.abs(self._dlnsdlnm) / self.m ** 2
        if isinstance(self.hmf, ff.Behroozi):
            ngtm_tinker = self._gtm(dndm)
            dndm = self.hmf._modify_dndm(self.m, dndm, self.z, ngtm_tinker)

        # Alter the mass definition
        if (
            self.hmf.measured_mass_definition is not None
            and self.hmf.measured_mass_definition != self.mdef
            and not self.disable_mass_conversion
        ):
            # this uses NFW, but we can change that in halomod.
            m_meas = self.mdef.change_definition(
                self.m,
                self.hmf.measured_mass_definition,
            )[0]

            dndm *= self.m / m_meas

        return dndm

    @cached_quantity
    def dndlnm(self):
        r"""
        The differential mass function in terms of natural log of `m`, ``len=len(m)`` [units :math:`h^3 Mpc^{-3}`]
        """
        return self.m * self.dndm

    @cached_quantity
    def dndlog10m(self):
        r"""
        The differential mass function in terms of log of `m`, ``len=len(m)`` [units :math:`h^3 Mpc^{-3}`]
        """
        return self.m * self.dndm * np.log(10)

    def _gtm(self, dndm, mass_density=False):
        """
        Calculate number or mass density above mass thresholds in `m`

        This function is here, separate from the properties, due to its need
        of being passed ``dndm`` in the case of the :class:`~fitting_functions.Behroozi`
        fit only, in which case an infinite recursion would occur otherwise.

        Parameters
        ----------
        dndm : array_like, ``len(self.m)``
            Should usually just be exactly :attr:`dndm`, except in Behroozi fit.

        mass_density : bool, ``False``
            Whether to get the mass density, or number density.
        """
        # Get required local variables
        size = len(dndm)
        m = self.m
        # If the highest mass is very low, we try calculating it to higher masses
        # The dlog10m is NOT CHANGED, so the input needs to be finely spaced.
        # If the top value of dndm is NaN, don't try calculating higher masses.
        if m[-1] < 10 ** 16.5 and not np.isnan(dndm[-1]) and not dndm[-1] == 0:
            # ff.Behroozi function won't work here.
            if not isinstance(self.hmf, ff.Behroozi):
                new_mf = copy.deepcopy(self)
                new_mf.update(Mmin=np.log10(self.m[-1]) + self.dlog10m, Mmax=18)
                dndm = np.concatenate((dndm, new_mf.dndm))

                m = np.concatenate((m, new_mf.m))

        ngtm = int_gtm(m[dndm > 0], dndm[dndm > 0], mass_density)

        # We need to set ngtm back in the original length vector with nans where
        # they were originally
        if len(ngtm) < len(m):  # Will happen if some dndlnm are NaN
            ngtm_temp = np.zeros(len(dndm))
            # ngtm_temp[:] = np.nan
            ngtm_temp[dndm > 0] = ngtm
            ngtm = ngtm_temp

        # Since ngtm may have been extended, we cut it back
        return ngtm[:size]

    @cached_quantity
    def ngtm(self):
        r"""
        The cumulative mass function above `m`, ``len=len(m)`` [units :math:`h^3 Mpc^{-3}`]

        In the case that `m` does not extend to sufficiently high masses, this
        routine will auto-generate ``dndm`` for an extended mass range.

        In the case of the ff.Behroozi fit, it is impossible to auto-extend the mass
        range except by the power-law fit, thus one should be careful to supply
        appropriate mass ranges in this case.
        """
        return self._gtm(self.dndm)

    @cached_quantity
    def rho_gtm(self):
        r"""
        Mass density in haloes `>m`, ``len=len(m)`` [units :math:`M_\odot h^2 Mpc^{-3}`]

        In the case that `m` does not extend to sufficiently high masses, this
        routine will auto-generate ``dndm`` for an extended mass range.

        In the case of the ff.Behroozi fit, it is impossible to auto-extend the mass
        range except by the power-law fit, thus one should be careful to supply
        appropriate mass ranges in this case.
        """
        return self._gtm(self.dndm, mass_density=True)

    @cached_quantity
    def rho_ltm(self):
        r"""
        Mass density in haloes `<m`, ``len=len(m)`` [units :math:`M_\odot h^2 Mpc^{-3}`]

        .. note :: As of v1.6.2, this assumes that the entire mass density of
                   halos is encoded by the ``mean_density0`` parameter (ie. all
                   mass is found in halos). This is not explicitly true of all
                   fitting functions (eg. Warren), in which case the definition
                   of this property is somewhat inconsistent, but will still
                   work.

        In the case that `m` does not extend to sufficiently high masses, this
        routine will auto-generate ``dndm`` for an extended mass range.

        In the case of the ff.Behroozi fit, it is impossible to auto-extend the mass
        range except by the power-law fit, thus one should be careful to supply
        appropriate mass ranges in this case.
        """
        return self.mean_density0 - self.rho_gtm

    @cached_quantity
    def how_big(self):
        r"""
        Size of simulation volume in which to expect one halo of mass m (with 95% probability), `
        `len=len(m)`` [units :math:`Mpch^{-1}`]
        """
        return (0.366362 / self.ngtm) ** (1.0 / 3.0)
