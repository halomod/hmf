"""
Module defining the growth factor `Component`.

The primary class, :class:`GrowthFactor`, executes a full
numerical calculation in standard flat LambdaCDM. Simplifications
which may be more efficient, or extensions to alternate cosmologies,
may be implemented.
"""

import numpy as np
from astropy import cosmology
from cached_property import cached_property
from scipy.interpolate import InterpolatedUnivariateSpline as _spline
from typing import Union

from .._internals._framework import Component as Cmpt
from .._internals._framework import pluggable
from .._internals._utils import inherit_docstrings as _inherit

try:
    import camb

    HAVE_CAMB = True
except ImportError:  # pragma: nocover
    HAVE_CAMB = False


@pluggable
class _GrowthFactor(Cmpt):
    r"""
    General class for a growth factor calculation.
    """
    supported_cosmos = (cosmology.LambdaCDM,)

    def __init__(self, cosmo, **model_parameters):
        self.cosmo = cosmo
        if not isinstance(self.cosmo, self.supported_cosmos):
            raise ValueError(
                f"Cosmology of type {type(self.cosmo)} not supported by "
                f"{self.__class__.__name__}. Supported cosmologies: "
                f"{self.supported_cosmos}."
            )
        super(_GrowthFactor, self).__init__(**model_parameters)


class GrowthFactor(_GrowthFactor):
    r"""
    Growth factor calculation, using a numerical integral, following [1]_.

    Parameters
    ----------
    cosmo : ``astropy.cosmology.FLRW`` instance
        Cosmological model.
    \*\*model_parameters : unpack-dict
        Parameters specific to this model. In this case, available
        parameters are as follows.To see their default values, check
        the :attr:`_defaults` class attribute.

        :dlna: Step-size in log-space for scale-factor integration
        :amin: Minimum scale-factor (i.e.e maximum redshift) to integrate to.
               Only used for :meth:`growth_factor_fn`.

    References
    ----------
    .. [1] Lukic et. al., ApJ, 2007, http://adsabs.harvard.edu/abs/2007ApJ...671.1160L
    """
    _defaults = {"dlna": 0.01, "amin": 1e-8}

    def __init__(self, *args, **kwargs):
        super(GrowthFactor, self).__init__(*args, **kwargs)

    @cached_property
    def _lna(self):
        lna = np.arange(np.log(self.params["amin"]), 0, self.params["dlna"])
        if lna[-1] != 0:
            lna = np.concatenate((lna, [0]))
        return lna

    @cached_property
    def _zvec(self):
        return 1.0 / np.exp(self._lna) - 1.0

    @cached_property
    def integral(self):
        a = np.exp(self._lna)
        return _spline(
            a, 2.5 * self.cosmo.Om0 / (a * self.cosmo.efunc(self._zvec)) ** 3
        ).antiderivative()

    def _d_plus(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""
        Finds the factor :math:`D^+(a)`, from Lukic et. al. 2007, eq. 8.

        Parameters
        ----------
        z
            The redshift

        Returns
        -------
        dplus
            The un-normalised growth factor -- same type as ``z``.
        """
        a_min = np.exp(self._lna).min()
        a = 1 / (1 + z)

        if np.any(z < 0):
            raise ValueError("Redshifts <0 not supported")
        if np.any(a < a_min):
            raise ValueError(
                f"Cannot compute integral for z > {1/a_min - 1}. Set amin lower."
            )

        return (self.integral(a) - self.integral(a_min)) * self.cosmo.efunc(z)

    @cached_property
    def _d_plus0(self) -> float:
        return self._d_plus(0.0)

    def growth_factor(self, z):
        r"""
        Calculate :math:`d(a) = D^+(a)/D^+(a=1)`, from Lukic et. al. 2007, eq. 7.

        Parameters
        ----------
        z : float
            The redshift

        Returns
        -------
        float
            The normalised growth factor.
        """
        return self._d_plus(z) / self._d_plus0

    def growth_factor_fn(self, zmin=0.0, inverse=False):
        """
        Calculate :math:`d(a) = D^+(a)/D^+(a=1)`, from Lukic et. al. 2007, eq. 7.

        Returns a function G(z).

        Parameters
        ----------
        zmin : float, optional
            The minimum redshift of the function. Default 0.0
        inverse: bool, optional
            Whether to return the inverse relationship [z(g)]. Default False.

        Returns
        -------
        callable
            The normalised growth factor as a function of redshift, or
            redshift as a function of growth factor if ``inverse`` is True.
        """

        if not inverse:
            return self.growth_factor

        z = np.sort(self._zvec)[::-1]
        gf = self.growth_factor(z)
        return _spline(gf, z)

    def growth_rate(self, z):
        """
        Growth rate, dln(d)/dln(a) from Hamilton 2000 eq. 4

        Parameters
        ----------
        z : float
            The redshift
        """
        return (
            -1
            - self.cosmo.Om(z) / 2
            + self.cosmo.Ode(z)
            + 5 * self.cosmo.Om(z) / (2 * self.growth_factor(z))
        )

    def growth_rate_fn(self, zmin=0):
        """
        Growth rate, dln(d)/dln(a) from Hamilton 2000 eq. 4, as callable.

        Parameters
        ----------
        zmin : float, optional
            The minimum redshift of the function. Default 0.0

        Returns
        -------
        callable
            The normalised growth rate as a function of redshift.
        """
        gfn = self.growth_factor_fn(zmin)

        return lambda z: (
            -1
            - self.cosmo.Om(z) / 2
            + self.cosmo.Ode(z)
            + 5 * self.cosmo.Om(z) / (2 * gfn(z))
        )


@_inherit
class GenMFGrowth(GrowthFactor):
    r"""
    Port of growth factor routines found in the ``genmf`` code.

    Parameters
    ----------
    cosmo : ``astropy.cosmology.FLRW`` instance
        Cosmological model.
    \*\*model_parameters : unpack-dict
        Parameters specific to this model. In this case, available
        parameters are as follows.To see their default values, check
        the :attr:`_defaults` class attribute.

        :dz: Step-size for redshift integration
        :zmax: Maximum redshift to integrate to. Only used for :meth:`growth_factor_fn`.
    """

    _defaults = {"dz": 0.01, "zmax": 1000.0}

    @cached_property
    def _lna(self):
        return 1 / (1 + self._zvec)

    @cached_property
    def _zvec(self):
        return np.arange(0, self.params["zmax"], self.params["dz"])

    def _d_plus(self, z):
        """
        This is not implemented in this class. It is not
        required to calculate :meth:`growth_factor`.
        """
        raise NotImplementedError()  # pragma: nocover

    def _general_case(self, w, x):
        x = np.atleast_1d(x)
        xn_vec = np.linspace(0, x.max(), 1000)

        func = _spline(xn_vec, (xn_vec / (xn_vec ** 3 + 2)) ** 1.5)

        g = np.array([func.integral(0, y) for y in x])
        return ((x ** 3.0 + 2.0) ** 0.5) * (g / x ** 1.5)

    def growth_factor(self, z):
        """
        The growth factor, :math:`d(a) = D^+(a)/D^+(a=1)`.

        This uses an approximation only valid in closed or
        flat cosmologies, ported from ``genmf``.

        Parameters
        ----------
        z : array_like
            Redshift.

        Returns
        -------
        gf : array_like
            The growth factor at `z`.
        """
        a = 1 / (1 + z)
        w = 1 / self.cosmo.Om0 - 1.0
        s = 1 - self.cosmo.Ok0
        if (s > 1 or self.cosmo.Om0 < 0 or (s != 1 and self.cosmo.Ode0 > 0)) and np.abs(
            s - 1.0
        ) > 1.0e-10:
            raise ValueError("Cannot cope with this cosmology!")

        if self.cosmo.Om0 == 1:
            return a
        elif self.cosmo.Ode0 > 0:
            xn = (2.0 * w) ** (1.0 / 3)
            aofxn = self._general_case(w, xn)
            x = a * xn
            aofx = self._general_case(w, x)
            return aofx / aofxn
        else:
            dn = (
                1
                + 3 / w
                + (3 * ((1 + w) ** 0.5) / w ** 1.5) * np.log((1 + w) ** 0.5 - w ** 0.5)
            )
            x = w * a
            return (
                1
                + 3 / x
                + (3 * ((1 + x) ** 0.5) / x ** 1.5) * np.log((1 + x) ** 0.5 - x ** 0.5)
            ) / dn


@_inherit
class Carroll1992(GrowthFactor):
    r"""
    Analytic approximation for the growth factor from Carroll et al. 1992.

    Adapted from chomp project.

    Parameters
    ----------
    cosmo : ``astropy.cosmology.FLRW`` instance
        Cosmological model.
    \*\*model_parameters : unpack-dict
        Parameters specific to this model. In this case, available
        parameters are as follows.To see their default values, check
        the :attr:`_defaults` class attribute.

        :dz: Step-size for redshift spline
        :zmax: Maximum redshift of spline. Only used for :meth:`growth_factor_fn`, when `inverse=True`.
    """

    _defaults = {"dz": 0.01, "zmax": 1000.0}

    @cached_property
    def _lna(self):
        return 1 / (1 + self._zvec)

    @cached_property
    def _zvec(self):
        return np.arange(0, self.params["zmax"], self.params["dz"])

    def _d_plus(self, z):
        """
        Calculate un-normalised growth factor as a function
        of redshift. Note that the `getvec` argument is not
        used in this function.
        """
        a = 1 / (1 + z)

        om = self.cosmo.Om0 / a ** 3
        denom = self.cosmo.Ode0 + om
        Omega_m = om / denom
        Omega_L = self.cosmo.Ode0 / denom
        coeff = 5.0 * Omega_m / (2.0 / a)
        term1 = Omega_m ** (4.0 / 7.0)
        term3 = (1.0 + 0.5 * Omega_m) * (1.0 + Omega_L / 70.0)
        return coeff / (term1 - Omega_L + term3)


if HAVE_CAMB:

    @_inherit
    class CambGrowth(_GrowthFactor):
        """
        Uses CAMB to generate the growth factor, at k/h = 1.0. This class is recommended
        if the cosmology is not LambdaCDM (but instead wCDM), as it correctly deals with
        the growth in this case. However, it standard LCDM is used, other classes are
        preferred, as this class needs to re-calculate the transfer function.
        """

        supported_cosmos = (cosmology.LambdaCDM, cosmology.w0waCDM, cosmology.wCDM)

        def __init__(self, *args, **kwargs):
            super(CambGrowth, self).__init__(*args, **kwargs)

            # Save the CAMB object properly for use
            # Set the cosmology
            self.p = camb.CAMBparams()

            if self.cosmo.Ob0 is None:
                raise ValueError(
                    "If using CAMB, the baryon density must be set explicitly in the cosmology."
                )

            if self.cosmo.Tcmb0.value == 0:
                raise ValueError(
                    "If using CAMB, the CMB temperature must be set explicitly in the cosmology."
                )

            self.p.set_cosmology(
                H0=self.cosmo.H0.value,
                ombh2=self.cosmo.Ob0 * self.cosmo.h ** 2,
                omch2=(self.cosmo.Om0 - self.cosmo.Ob0) * self.cosmo.h ** 2,
                omk=self.cosmo.Ok0,
                nnu=self.cosmo.Neff,
                standard_neutrino_neff=self.cosmo.Neff,
                TCMB=self.cosmo.Tcmb0.value,
            )

            self.p.WantTransfer = True

            # Set the DE equation of state. We only support constant w.
            if hasattr(self.cosmo, "w0"):
                self.p.set_dark_energy(w=self.cosmo.w0)

        @cached_property
        def _camb_transfers(self):
            return camb.get_transfer_functions(self.p)

        @cached_property
        def _t0(self):
            """The Transfer function at z=0."""
            return self._camb_transfers.get_redshift_evolution(1.0, 0.0, ["delta_tot"])[
                0
            ][0]

        def growth_factor(self, z):
            """
            Calculate :math:`d(a) = D^+(a)/D^+(a=1)`.

            Parameters
            ----------
            z : float
                The redshift

            Returns
            -------
            float
                The normalised growth factor.
            """
            growth = (
                self._camb_transfers.get_redshift_evolution(
                    1.0, z, ["delta_tot"]
                ).flatten()
                / self._t0
            )
            if len(growth) == 1:
                return growth[0]
            else:
                return growth
