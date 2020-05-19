"""
Module defining the growth factor `Component`.

The primary class, :class:`GrowthFactor`, executes a full
numerical calculation in standard flat LambdaCDM. Simplifications
which may be more efficient, or extensions to alternate cosmologies,
may be implemented.
"""

import numpy as np
from scipy import integrate as intg
from .._internals._framework import Component as Cmpt
from scipy.interpolate import InterpolatedUnivariateSpline as _spline
from .._internals._utils import inherit_docstrings as _inherit
import warnings

try:
    import camb

    HAVE_CAMB = True
except ImportError:
    HAVE_CAMB = False


class _GrowthFactor(Cmpt):
    r"""
    General class for a growth factor calculation.
    """

    def __init__(self, cosmo, **model_parameters):
        self.cosmo = cosmo
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

        if hasattr(self.cosmo, "w0"):
            warnings.warn(
                "Using this growth factor model in wCDM can lead to inaccuracy. Try using CambGrowth."
            )

    def _d_plus(self, z, getvec=False):
        r"""
        Finds the factor :math:`D^+(a)`, from Lukic et. al. 2007, eq. 8.

        Parameters
        ----------
        z : float
            The redshift
        getvec : bool, optional
            Whether to treat `z` as a maximum redshift and return a whole vector
            of values up to `z`. In this case, the minimum scale factor and the
            step size are defined in :attr:`_defaults` and can be over-ridden
            at instantiation.

        Returns
        -------
        dplus : float
            The un-normalised growth factor.
        """

        a_upper = 1.0 / (1.0 + z)

        lna = np.arange(
            np.log(self.params["amin"]), np.log(a_upper), self.params["dlna"]
        )
        lna = np.hstack((lna, np.log(a_upper)))

        self._zvec = 1.0 / np.exp(lna) - 1.0

        integrand = 1.0 / (np.exp(lna) * self.cosmo.efunc(self._zvec)) ** 3

        if not getvec:
            integral = intg.simps(np.exp(lna) * integrand, x=lna, even="avg")
            dplus = 5.0 * self.cosmo.Om0 * self.cosmo.efunc(z) * integral / 2.0
        else:
            integral = intg.cumtrapz(np.exp(lna) * integrand, x=lna, initial=0.0)
            dplus = 5.0 * self.cosmo.Om0 * self.cosmo.efunc(self._zvec) * integral / 2.0

        return dplus

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
        return self._d_plus(z) / self._d_plus(0.0)

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
        dp = self._d_plus(0.0, True)
        growth = dp / dp[-1]
        if not inverse:
            s = _spline(self._zvec[::-1], growth[::-1])
        else:
            s = _spline(growth, self._zvec)
        return s

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

    def _d_plus(self, z, getvec=False):
        """
        This is not implemented in this class. It is not
        required to calculate :meth:`growth_factor`.
        """
        raise NotImplementedError()

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

    def growth_factor_fn(self, zmin=0.0, inverse=False):
        r"""
        Return the growth factor as a callable function.

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
        else:
            self._zvec = np.arange(zmin, self.params["zmax"], self.params["dz"])
            gf = self.growth_factor(self._zvec)
            return _spline(gf[::-1], self._zvec[::-1])


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

    def _d_plus(self, z, getvec=False):
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

    def growth_factor(self, z):
        """
        The growth factor, :math:`d(a) = D^+(a)/D^+(a=1)`.

        Parameters
        ----------
        z : array_like
            Redshift.

        Returns
        -------
        gf : array_like
            The growth factor at `z`.
        """

        return self._d_plus(z) / self._d_plus(0.0)

    def growth_factor_fn(self, zmin=0.0, inverse=False):
        """
        Return the growth factor as a callable function.

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
        else:
            self._zvec = np.arange(zmin, self.params["zmax"], self.params["dz"])
            gf = self.growth_factor(self._zvec)
            return _spline(gf[::-1], self._zvec[::-1])


if HAVE_CAMB:

    @_inherit
    class CambGrowth(_GrowthFactor):
        """
        Uses CAMB to generate the growth factor, at k/h = 1.0. This class is recommended
        if the cosmology is not LambdaCDM (but instead wCDM), as it correctly deals with
        the growth in this case. However, it standard LCDM is used, other classes are
        preferred, as this class needs to re-calculate the transfer function.
        """

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

            # Now find the z=0 transfer
            self._camb_transfers = camb.get_transfer_functions(self.p)
            self._t0 = self._camb_transfers.get_redshift_evolution(
                1.0, 0.0, ["delta_tot"]
            )[0][0]

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
