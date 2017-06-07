'''
Module defining the growth factor `Component`.

The primary class, :class:`GrowthFactor`, executes a full
numerical calculation in standard flat LambdaCDM. Simplifications
which may be more efficient, or extensions to alternate cosmologies,
may be implemented.
'''

import numpy as np
from scipy import integrate as intg
from ._framework import Component as Cmpt
from scipy.interpolate import InterpolatedUnivariateSpline as _spline
from ._utils import inherit_docstrings as _inherit


class GrowthFactor(Cmpt):
    r"""
    General class for a growth factor calculation.

    Each of the methods in this class is defined using a numerical
    integral, following [1]_.

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
    _defaults = {"dlna":0.01, "amin":1e-8}

    def __init__(self, cosmo, **model_parameters):
        self.cosmo = cosmo
        super(GrowthFactor, self).__init__(**model_parameters)

    def _d_plus(self, z, getvec=False):
        """
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

        lna = np.arange(np.log(self.params["amin"]), np.log(a_upper), self.params['dlna'])
        lna = np.hstack((lna,np.log(a_upper)))

        self._zvec = 1.0 / np.exp(lna) - 1.0

        integrand = 1.0 / (np.exp(lna) * self.cosmo.efunc(self._zvec)) ** 3

        if not getvec:
            integral = intg.simps(np.exp(lna) * integrand, x=lna,even="avg")
            dplus = 5.0 * self.cosmo.Om0 * self.cosmo.efunc(z) * integral / 2.0
        else:
            integral = intg.cumtrapz(np.exp(lna) * integrand, x=lna, initial=0.0)
            dplus = 5.0 * self.cosmo.Om0 * self.cosmo.efunc(self._zvec) * integral / 2.0

        return dplus

    def growth_factor(self, z):
        """
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
        growth = self._d_plus(z)/self._d_plus(0.0)
        return growth

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
        growth = self._d_plus(zmin, True)/self._d_plus(0.0)
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
        return (-1 - self.cosmo.Om(z) / 2 + self.cosmo.Ode(z) +
                5 * self.cosmo.Om(z) / (2 * self.growth_factor(z)))


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

        return lambda z: (-1 - self.cosmo.Om(z) / 2 + self.cosmo.Ode(z) +
                          5 * self.cosmo.Om(z) / (2 * gfn(z)))

@_inherit
class GenMFGrowth(GrowthFactor):
    """
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
    _defaults = {"dz":0.01, "zmax":1000.0}

    def _d_plus(self, z, getvec=False):
        """
        This is not implemented in this class. It is not
        required to calculate :meth:`growth_factor`.
        """
        raise NotImplementedError()

    def _general_case(self, w, x):
        x = np.atleast_1d(x)
        xn_vec = np.linspace(0, x.max(), 1000)

        func = _spline(xn_vec,(xn_vec / (xn_vec ** 3 + 2)) ** 1.5)

        g = np.array([func.integral(0,y) for y in x])
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
        if (s > 1 or self.cosmo.Om0 < 0 or (s != 1 and self.cosmo.Ode0 > 0)):
            if np.abs(s - 1.0) > 1.e-10:
                raise ValueError('Cannot cope with this cosmology!')

        if self.cosmo.Om0 == 1:
            return a
        elif self.cosmo.Ode0 > 0:
            xn = (2.0 * w) ** (1.0 / 3)
            aofxn = self._general_case(w, xn)
            x = a * xn
            aofx = self._general_case(w, x)
            return aofx / aofxn
        else:
            dn = 1 + 3 / w + (3 * ((1 + w) ** 0.5) / w ** 1.5) * np.log((1 + w) ** 0.5 - w ** 0.5)
            x = w * a
            return (1 + 3 / x + (3 * ((1 + x) ** 0.5) / x ** 1.5) * np.log((1 + x) ** 0.5 - x ** 0.5)) / dn

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
            self._zvec = np.arange(zmin, self.params['zmax'], self.params['dz'])
            gf = self.growth_factor(self._zvec)
            return _spline(gf[::-1], self._zvec[::-1])



@_inherit
class Carroll1992(GrowthFactor):
    """
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
    _defaults = {"dz":0.01, "zmax":1000.0}

    def _d_plus(self, z, getvec=False):
        """
        Calculate un-normalised growth factor as a function
        of redshift. Note that the `getvec` argument is not
        used in this function.
        """
        a = 1 / (1 + z)

        om = self.cosmo.Om0/a ** 3
        denom = self.cosmo.Ode0 + om
        Omega_m = om/denom
        Omega_L = self.cosmo.Ode0/denom
        coeff = 5.*Omega_m/(2./a)
        term1 = Omega_m**(4./7.)
        term3 = (1. + 0.5*Omega_m)*(1. + Omega_L/70.)
        return coeff/(term1 - Omega_L + term3)

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

        return self._d_plus(z)/self._d_plus(0.0)

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
            self._zvec = np.arange(zmin, self.params['zmax'], self.params['dz'])
            gf = self.growth_factor(self._zvec)
            return _spline(gf[::-1], self._zvec[::-1])
