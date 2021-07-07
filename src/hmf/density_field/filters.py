"""A module containing various smoothing filter Component models."""

import collections
import numpy as np
import scipy.integrate as intg
import warnings
from scipy.interpolate import InterpolatedUnivariateSpline as _spline

from .._internals import _framework, _utils


@_framework.pluggable
class Filter(_framework.Component):
    r"""
    Base class for Filter components.

    Filters handle the calculation of the mass variance from the power
    spectrum, via a window function. Subclasses of :class:`~Filer` implement
    specific window functions.

    The general design is to specify all quantities in terms of length scales,
    rather than equivalent masses, but conversion methods are provided.

    Any explicit subclass need only specify `k_space`, `mass_to_radius`,
    `radius_to_mass` and `dw_dlnkr`, with `dlnr_dlnm` optional if the
    mass assignment is unconventional.

    Parameters
    ----------
    k : array_like
        Wavenumbers at which the power spectrum is defined.
    power : array_like
        The power spectrum at `k`.
    \*\*model_parameters : unpacked-dict
        As for any :class:`hmf._framework.Component` subclass, any particular
        parameters of the model may be passed to the constructor. Allowed
        parameters are found in the :attr:`~._defaults` attribute.

    Notes
    -----
    Besides the raw filter function itself, two quantities are of primary interest:
    firstly the mass variance (see :meth:`sigma`), which appears in many cosmological
    applications, and secondly its logarithmic derivative with mass, which appears in
    the Press-Schechter formalism for the halo mass function.

    To remain extensible and general, the methodology in this class is to calculate the
    latter quantity as

    .. math:: \frac{d\ln\sigma}{d\ln m} = \frac{1}{2} \frac{d \ln \sigma^2}{d\ln R}
              \frac{d\ln R}{d\ln m}.

    Each of the quantities on the right can be separately calculated, improving
    extensibility.

    The factor :math:`\frac{d\ln R}{d\ln m}` is typically 1/3, but this is not
    necessarily the case for window functions of arbitrary shape.
    """

    def __init__(self, k, power, **model_parameters):
        self.k = k
        self.power = power

        super(Filter, self).__init__(**model_parameters)

    def real_space(self, R, r):
        r"""
        Filter definition in real space.

        Parameters
        ----------
        R : float
            The smoothing scale

        r : array_like
            The radial co-ordinate
        """
        pass

    def k_space(self, kr):
        r"""
        Fourier-transform of the real-space filter.

        Parameters
        ----------
        kr : array_like
            The scales at which to return the filter function

        Returns
        -------
        w : array_like
            The filter in fourier space, ``len(kr)``
        """
        pass

    def mass_to_radius(self, m, rho_mean):
        r"""
        Return radius of a region of space given its mass.

        Parameters
        ----------
        m : array_like
            Masses
        rho_mean : float
            Mean density of the Universe.

        Returns
        -------
        r : array_like
            The corresponding radii to m

        Notes
        -----
        The units of `m` don't matter as long as they are consistent with `rho_mean`.
        """
        pass

    def radius_to_mass(self, r, rho_mean):
        r"""
        Return mass of a region of space given its radius

        Parameters
        ----------
        r : array_like
            Radii
        rho_mean : float
            Mean density of the Universe.

        Returns
        -------
        m : float or array of floats
            The corresponding masses to r

        Notes
        -----
        The units of `r` don't matter as long as they are consistent with `rho_mean`.
        """
        pass

    def dw_dlnkr(self, kr):
        r"""
        The derivative of the (fourier-transformed) filter with :math:`\ln(kr)`.

        Parameters
        ----------
        kr : array_like
            Scale(s) at which the derivative is evaluated.

        Notes
        -----
        In terms of :math:`\frac{dw^2}{dm}`, which is a commonly used
        quantity, this has the relationship

        .. math:: w\frac{dw}{d\ln r} = \frac{2}{r}\frac{dw^2}{dm}\frac{dm}{dr}.
        """
        pass

    def dlnss_dlnr(self, r):
        r"""
        The derivative of the mass variance with radius.

        Parameters
        ----------
        r : array_like
            Radii

        Returns
        -------
        dlnss_dlnr : array_like
            The derivative of the the mass variance with radius.

        Notes
        -----
        Given a prescription for how radius grows with mass (typically with
        a log-slope of 1/3, and set in :meth:`dlnr_dlnm`), this specifies
        the quantity :math:`\frac{d \ln \sigma^2}{d\ln m}`.

        The general formula is

        .. math:: \frac{d\ln \sigma^2}{d\ln R} = \frac{1}{\pi^2\sigma^2} \int_0^\infty
                  W(kR) \frac{dW(kR)}{d\ln(kR)} P(k)k^2 dk
        """
        dlnk = np.log(self.k[1] / self.k[0])
        s = self.sigma(r)
        rk = np.outer(r, self.k)

        rest = self.power * self.k ** 3
        w = self.k_space(rk)
        dw = self.dw_dlnkr(rk)
        integ = w * dw * rest
        return intg.simps(integ, dx=dlnk, axis=-1) / (np.pi ** 2 * s ** 2)

    def dlnr_dlnm(self, r):
        r"""
        The derivative of log radius with log mass.

        For the usual :math:`m\propto r^3` mass assignment, this is just 1/3.

        Parameters
        ----------
        r : array_like
            Radii.
        """
        return 1.0 / 3.0

    def dlnss_dlnm(self, r):
        r"""
        The logarithmic slope of mass variance with mass.

        This is an important quantity, and is used directly to calculate
        :math:`\frac{dn}{dm}`.

        Parameters
        ----------
        r : array_like
            Radii.
        """
        return self.dlnss_dlnr(r) * self.dlnr_dlnm(r)

    def sigma(self, r, order=0, rk=None):
        r"""
        Calculate the nth-moment of the smoothed density field, :math:`\sigma_n(r)`.

        .. note :: This is not :math:`\sigma_n^2(r)`!

        Parameters
        ----------
        r : float or array_like
            The radii of the spheres at which to calculate the nth moment.

        order : int, optional
            The order of the moment. Default 0 corresponds to common
            mass variance.

        Returns
        -------
        sigma : array_like
            The square root of the moment at `r`.

        Notes
        -----
        The general definition for the nth-moment of the smoothed density field is (see
        Bardeen et al. 1986, Eq 4.6c)

        .. math:: \sigma^2_n(R) = \frac{1}{2\pi^2} \int_0^\infty dk\ k^{2(1+n)}
                  P(k) W^2(kR)
        """
        if rk is None:
            rk = np.outer(r, self.k)

        dlnk = np.log(self.k[1] / self.k[0])

        # we multiply by k because our steps are in logk.
        rest = self.power * self.k ** (3 + order * 2)
        integ = rest * self.k_space(rk) ** 2
        sigma = (0.5 / np.pi ** 2) * intg.simps(integ, dx=dlnk, axis=-1)
        return np.sqrt(sigma)

    def nu(self, r, delta_c=1.686):
        r"""
        Peak height, :math:`\frac{\delta_c^2}{\sigma^2(r)}`.

        Parameters
        ----------
        r : array_like
            Radii

        delta_c : float, optional
            Critical overdensity for collapse.
        """
        return (delta_c / self.sigma(r)) ** 2


@_utils.inherit_docstrings
class TopHat(Filter):
    r"""
    Real-space top-hat window function.

    This class is based on :class:`~Filter`, which can be consulted for
    details of how to instantiate it.

    Notes
    -----
    This filter specifically implements the real-space filter:

    .. math:: F(r) = H(R-r)

    for a filter size of *R*, where *H* is the Heaviside step-function.

    Its fourier-transform is

    .. math:: W(x=kR) = 3\frac{\sin x - x\cos x}{x^3}.

    Furthermore the mass-assignment is

    .. math:: m(R) = \frac{4\pi}{3} R^3 \bar{\rho}

    and the derivative of the window function is

    .. math:: \frac{dW}{d\ln x}(x=kR) = \frac{1}{x^3}[9x\cos x + 3(x^2-3)\sin x].
    """

    def real_space(self, R, r):
        a = np.where(r < R, 1, 0)
        return np.where(r == R, 0.5, a)

    def k_space(self, kr):
        return np.where(kr > 1.4e-6, (3 / kr ** 3) * (np.sin(kr) - kr * np.cos(kr)), 1)

    def mass_to_radius(self, m, rho_mean):
        return (3.0 * m / (4.0 * np.pi * rho_mean)) ** (1.0 / 3.0)

    def radius_to_mass(self, r, rho_mean):
        return 4 * np.pi * r ** 3 * rho_mean / 3

    def dw_dlnkr(self, kr):
        return np.where(
            kr > 1e-3,
            (9 * kr * np.cos(kr) + 3 * (kr ** 2 - 3) * np.sin(kr)) / kr ** 3,
            0,
        )


@_utils.inherit_docstrings
class Gaussian(Filter):
    r"""
    Gaussian window function.

    This class is based on :class:`~Filter`, which can be consulted for
    details of how to instantiate it.

    Notes
    -----
    The real-space filter is

    .. math:: F(r) = \frac{\exp(-r^2/2R^2)}{R^3 (2\pi)^{3/2}}

    for a filter scale of *R*.

    The fourier-transform of the filter is

    .. math:: W(x=kR) = \exp(-x^2/2).

    The mass-assignment is

    .. math:: m(R) = R^3(2\pi)^{3/2}\bar{\rho},

    and the derivative of the window function is

    .. math:: \frac{dW}{d\ln x}(x=kR) = -xW(x).
    """

    def real_space(self, R, r):
        return np.exp(-(r ** 2) / 2 / R ** 2) / (2 * np.pi) ** 1.5 / R ** 3

    def k_space(self, kr):
        return np.exp(-(kr ** 2) / 2.0)

    def mass_to_radius(self, m, rho_mean):
        return (m / rho_mean) ** (1.0 / 3.0) / np.sqrt(2 * np.pi)

    def radius_to_mass(self, r, rho_mean):
        return (2 * np.pi) ** 1.5 * r ** 3 * rho_mean

    def dw_dlnkr(self, kr):
        return -(kr ** 2) * self.k_space(kr)


@_utils.inherit_docstrings
class SharpK(Filter):
    r"""
    Fourier-space top-hat window function

    This class is based on :class:`~Filter`, which can be consulted for
    details of how to instantiate it.

    Notes
    -----
    The real-space filter is

    .. math:: F(r) = \frac{\sin(r/R) -(r/R)\cos(r/R)}{2\pi^2r^3},

    for a filter scale of *R*.

    The fourier-transform of the filter is

    .. math:: W(x=kR) = H(kR-1)

    where *H* is the Heaviside step-function. The mass-assignment is

    .. math:: m(R) = \frac{4\pi}{3}[cR]^3\bar{\rho},

    where *c* is a free parameter, typically c~2.5. The derivative of the window
    function is

    .. math:: \frac{dW}{d\ln x}(x=kR) = \delta_D(x-1),

    where :math:`\delta_D` is the Dirac delta. Furthermore, the derivative of the mass
    variance takes a particularly simple form in this filter:

    .. math:: \frac{d\ln \sigma^2}{d \ln R} = -\frac{P(1/R)}{2\pi^2\sigma^2(R)R^3}.
    """

    _defaults = {"c": 2.5}

    def k_space(self, kr):
        a = np.where(kr > 1, 0, 1)
        return np.where(kr == 1, 0.5, a)

    def real_space(self, R, r):
        return (np.sin(r / R) - (r / R) * np.cos(r / R)) / (2 * np.pi ** 2 * r ** 3)

    def dw_dlnkr(self, kr):
        return np.where(kr == 1, 1.0, 0.0)

    def dlnss_dlnr(self, r):
        sigma = self.sigma(r)
        power = _spline(self.k, self.power)(1 / r)
        return -power / (2 * np.pi ** 2 * sigma ** 2 * r ** 3)

    def mass_to_radius(self, m, rho_mean):
        return (1.0 / self.params["c"]) * (3.0 * m / (4.0 * np.pi * rho_mean)) ** (
            1.0 / 3.0
        )

    def radius_to_mass(self, r, rho_mean):
        return 4 * np.pi * (self.params["c"] * r) ** 3 * rho_mean / 3

    def sigma(self, r, order=0):
        if not isinstance(r, collections.Iterable):
            r = np.atleast_1d(r)

        if self.k.max() < 1 / r.min():
            warnings.warn("Warning: Maximum r*k less than 1!")

        # # Need to re-define this because the integral needs to go exactly kr=1
        # # or else the function 'jitters'
        sigma = np.zeros(len(r))
        power = _spline(self.k, self.power)
        for i, rr in enumerate(r):
            k = np.logspace(
                np.log10(self.k[0]),
                min(np.log10(self.k.max()), np.log10(1.0 / rr)),
                max(100, len(self.k) - i),
            )

            p = power(k)
            dlnk = np.log(k[1] / k[0])
            integ = p * k ** (3 + 2 * order)
            sigma[i] = (0.5 / (np.pi ** 2)) * intg.simps(integ, dx=dlnk)

        return np.sqrt(sigma)


@_utils.inherit_docstrings
class SharpKEllipsoid(SharpK):
    """
    Fourier-space top-hat window function with ellipsoidal correction

    See Schneider, Smith, Reed 2013.

    Refer to :class:`~Filter` for more details.
    """

    _defaults = {"c": 2.0}

    def xm(self, g, v):
        """
        Peak of the distribution of x.

        Here, x is the sum of the eigenvalues of the inertia tensor (?) of an
        ellipsoidal peak, divided by the second spectral moment.

        Equation A6. in Schneider et al. 2013
        """
        top = 3 * (1 - g ** 2) + (1.1 - 0.9 * g ** 4) * np.exp(
            -g * (1 - g ** 2) * (g * v / 2) ** 2
        )
        bot = (3 * (1 - g ** 2) + 0.45 + (g * v / 2) ** 2) ** 0.5 + g * v / 2
        return g * v + top / bot

    def em(self, xm):
        """
        The average ellipticity of a patch as a function of peak tensor
        """
        return 1 / np.sqrt(5 * xm ** 2 + 6)

    def pm(self, xm):
        """
        The average prolateness of a patch as a function of peak tensor
        """
        return 30.0 / (5 * xm ** 2 + 6) ** 2

    def a3a1(self, e, p):
        """
        The short:long axis ratio of an ellipsoid given its ellipticity and prolateness
        """
        return np.sqrt((1 - 3 * e + p) / (1 + 3 * e + p))

    def a3a2(self, e, p):
        """
        The short:medium axis ratio of an ellipsoid given its ellipticity/prolateness
        """
        return np.sqrt((1 - 2 * p) / (1 + 3 * e + p))

    def gamma(self, r):
        """Bardeen et al. 1986 equation 6.17."""
        sig_0 = self.sigma(r)
        sig_1 = self.sigma(r, order=1)
        sig_2 = self.sigma(r, order=2)
        return sig_1 ** 2 / (sig_0 * sig_2)

    def xi(self, pm, em):
        return ((1 + 4 * pm) ** 2 / (1 - 3 * em + pm) / (1 - 2 * pm)) ** (1.0 / 6.0)

    def a3(self, r):
        g = self.gamma(r)
        xm = self.xm(g, self.nu(r))
        em = self.em(xm)
        pm = self.pm(xm)
        return r / self.xi(pm, em)

    def r_a3(self, rmin, rmax):
        r = np.logspace(np.log(rmin), np.log(rmax), 200, base=np.e)
        a3 = self.a3(r)
        return _spline(a3, r)

    def dlnss_dlnr(self, r):
        a3 = self.a3(r)
        sigma = self.sigma(a3)
        power = _spline(self.k, self.power)(1 / a3)
        return -power / (2 * np.pi ** 2 * sigma ** 2 * a3 ** 3)

    def dlnr_dlnm(self, r):
        a3 = self.a3(r)
        xi = r / a3
        drda = _spline(a3, r).derivative()(a3)
        return xi / 3 / drda
