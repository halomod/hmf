r"""Kernels of the halo mass function fitting functions.

Each function evaluates one functional form of the multiplicity function
:math:`f(\sigma) = \nu f(\nu)`, or one ingredient of it, on plain arrays. They follow
the rules of :mod:`hmf.core._kernels`: pure, vectorised, unit-free, and
batch-size-independent. Every input broadcasts against the others.

The peak height is :math:`\nu = \delta_c / \sigma` (not its square, issue #85).

The models in :mod:`hmf.core.fits` call these kernels with their parameters; the
formulas and their sources are documented on the models.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.special as sp
from scipy.interpolate import CubicSpline

__all__ = [
    "angulo",
    "behroozi_modify_dndm",
    "bhattacharya",
    "bhattacharya_norm",
    "bocquet16_mass_ratio_200c",
    "bocquet16_mass_ratio_500c",
    "jenkins",
    "ln_sigma_inv",
    "log10_delta_spline",
    "peacock",
    "peak_height",
    "press_schechter",
    "reed03_factor",
    "reed07",
    "sheth_tormen",
    "sheth_tormen_norm",
    "tinker08",
    "tinker08_b_exponent",
    "tinker10",
    "tinker10_norm",
    "warren",
    "watson_gamma",
]

FloatArray = npt.NDArray[np.float64]


def _f(x: npt.ArrayLike) -> FloatArray:
    """Convert to a float array (a no-op for float arrays)."""
    return np.asarray(x, dtype=np.float64)


def peak_height(sigma: npt.ArrayLike, delta_c: npt.ArrayLike) -> FloatArray:
    r"""The peak height, :math:`\nu = \delta_c / \sigma`.

    Parameters
    ----------
    sigma
        The mass variance's square root, :math:`\sigma(m, z)`.
    delta_c
        The critical overdensity for collapse.

    Returns
    -------
    numpy.ndarray
        :math:`\nu`.
    """
    return _f(delta_c) / _f(sigma)


def ln_sigma_inv(sigma: npt.ArrayLike) -> FloatArray:
    r"""The natural log of the inverse of :math:`\sigma`, :math:`\ln\sigma^{-1}`."""
    return -np.log(_f(sigma))


def press_schechter(nu: npt.ArrayLike) -> FloatArray:
    r"""Press & Schechter (1974): :math:`\sqrt{2/\pi}\,\nu\exp(-\nu^2/2)`."""
    nu = _f(nu)
    return _f(np.sqrt(2.0 / np.pi) * nu * np.exp(-0.5 * nu**2))


def sheth_tormen_norm(p: npt.ArrayLike) -> FloatArray:
    r"""The amplitude that normalises the Sheth-Tormen form to unit mass.

    :math:`A = [1 + 2^{-p}\Gamma(1/2 - p)/\Gamma(1/2)]^{-1}`, which requires
    :math:`p < 1/2`.
    """
    p = _f(p)
    return _f(1.0 / (1.0 + 2.0**-p * sp.gamma(0.5 - p) / sp.gamma(0.5)))


def sheth_tormen(
    nu: npt.ArrayLike, A: npt.ArrayLike, a: npt.ArrayLike, p: npt.ArrayLike
) -> FloatArray:
    r"""Sheth-Tormen: :math:`A\sqrt{2a/\pi}\,\nu e^{-a\nu^2/2}[1 + (a\nu^2)^{-p}]`."""
    nu, A, a, p = _f(nu), _f(A), _f(a), _f(p)
    anu2 = a * nu**2
    return A * np.sqrt(2.0 * a / np.pi) * nu * np.exp(-anu2 / 2.0) * (1.0 + anu2**-p)


def jenkins(
    sigma: npt.ArrayLike, A: npt.ArrayLike, b: npt.ArrayLike, c: npt.ArrayLike
) -> FloatArray:
    r"""Jenkins et al. (2001): :math:`A\exp(-|\ln\sigma^{-1} + b|^c)`."""
    return _f(A) * np.exp(-(np.abs(ln_sigma_inv(sigma) + _f(b)) ** _f(c)))


def warren(
    sigma: npt.ArrayLike,
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    c: npt.ArrayLike,
    d: npt.ArrayLike,
    e: npt.ArrayLike,
) -> FloatArray:
    r"""The Warren et al. (2006) form: :math:`A[(e/\sigma)^b + c]\exp(-d/\sigma^2)`.

    Many later fits share it, with other parameters (and c = 1 for some).
    """
    sigma = _f(sigma)
    return _f(A) * ((_f(e) / sigma) ** _f(b) + _f(c)) * np.exp(-_f(d) / sigma**2)


def angulo(
    sigma: npt.ArrayLike,
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    c: npt.ArrayLike,
    d: npt.ArrayLike,
) -> FloatArray:
    r"""Angulo et al. (2012): :math:`A[(d/\sigma)^b + 1]\exp(-c/\sigma^2)`."""
    return warren(sigma, A=A, b=b, c=1.0, d=c, e=d)


def reed03_factor(sigma: npt.ArrayLike, c: npt.ArrayLike) -> FloatArray:
    r"""The Reed et al. (2003) factor on Sheth-Tormen: :math:`\exp[-c/(\sigma\cosh^5 2\sigma)]`."""
    sigma = _f(sigma)
    return _f(np.exp(-_f(c) / (sigma * np.cosh(2.0 * sigma) ** 5)))


def reed07(
    sigma: npt.ArrayLike,
    nu: npt.ArrayLike,
    n_eff: npt.ArrayLike,
    A: npt.ArrayLike,
    a: npt.ArrayLike,
    c: npt.ArrayLike,
    p: npt.ArrayLike,
) -> FloatArray:
    r"""Reed et al. (2007), eq. 12.

    .. math::

        A\sqrt{2a/\pi}\,[1 + (a\nu^2)^{-p} + 0.6 G_1 + 0.4 G_2]\,\nu
        \exp\left[-\frac{ca\nu^2}{2} - \frac{0.03\nu^{0.6}}{(n_{\rm eff} + 3)^2}\right],

    with :math:`G_1 = \exp[-(\ln\sigma^{-1} - 0.4)^2 / (2\cdot 0.6^2)]` and
    :math:`G_2 = \exp[-(\ln\sigma^{-1} - 0.75)^2 / (2\cdot 0.2^2)]`.
    """
    nu, n_eff, A, a, c, p = _f(nu), _f(n_eff), _f(A), _f(a), _f(c), _f(p)
    lns = ln_sigma_inv(sigma)
    g1 = np.exp(-((lns - 0.4) ** 2) / (2 * 0.6**2))
    g2 = np.exp(-((lns - 0.75) ** 2) / (2 * 0.2**2))
    return (
        A
        * np.sqrt(2.0 * a / np.pi)
        * (1.0 + (a * nu**2) ** -p + 0.6 * g1 + 0.4 * g2)
        * nu
        * np.exp(-c * a * nu**2 / 2.0 - 0.03 * nu**0.6 / (n_eff + 3.0) ** 2)
    )


def peacock(nu: npt.ArrayLike, a: npt.ArrayLike, b: npt.ArrayLike, c: npt.ArrayLike) -> FloatArray:
    r"""Peacock (2007): :math:`-dF/d\ln\nu` of :math:`F = e^{-c\nu^2}/(1 + a\nu^b)`.

    That is :math:`\nu e^{-c\nu^2}(2cd\nu + ab\nu^{b-1})/d^2` with
    :math:`d = 1 + a\nu^b`.
    """
    nu, a, b, c = _f(nu), _f(a), _f(b), _f(c)
    d = 1.0 + a * nu**b
    return nu * np.exp(-c * nu**2) * (2.0 * c * d * nu + b * a * nu ** (b - 1.0)) / d**2


def watson_gamma(
    sigma: npt.ArrayLike,
    delta_halo: npt.ArrayLike,
    omega_m_z: npt.ArrayLike,
    C_a: npt.ArrayLike,
    d_a: npt.ArrayLike,
    d_b: npt.ArrayLike,
    p: npt.ArrayLike,
    q: npt.ArrayLike,
) -> FloatArray:
    r"""The overdensity correction :math:`\Gamma(\Delta, \sigma, z)` of Watson et al. (2013).

    .. math::

        \Gamma = C(\Delta)\left(\frac{\Delta}{178}\right)^{d(z)}
            \exp\left[\frac{p(1 - \Delta/178)}{\sigma^q}\right],

    with :math:`C = \exp[C_a(\Delta/178 - 1)]` and
    :math:`d = -d_a\Omega_m(z) - d_b`. ``delta_halo`` is relative to the mean density.
    It is exactly 1 at :math:`\Delta = 178`.
    """
    x = _f(delta_halo) / 178.0
    C = np.exp(_f(C_a) * (x - 1.0))
    d = -_f(d_a) * _f(omega_m_z) - _f(d_b)
    return C * x**d * np.exp(_f(p) * (1.0 - x) / _f(sigma) ** _f(q))


def log10_delta_spline(
    delta_tab: npt.ArrayLike, values: npt.ArrayLike, delta_halo: npt.ArrayLike
) -> FloatArray:
    r"""Interpolate parameters tabulated at overdensities ``delta_tab``.

    A natural cubic spline in :math:`\log_{10}\Delta`, as Tinker et al. (2008), App. B,
    recommend. At the tabulated overdensities it returns the tabulated values.

    Parameters
    ----------
    delta_tab
        The tabulated overdensities, increasing, shape ``(n,)``.
    values
        The parameter values at them, shape ``(n,)``.
    delta_halo
        Where to evaluate the spline.

    Returns
    -------
    numpy.ndarray
        The interpolated values, with the shape of ``delta_halo``.
    """
    spline = CubicSpline(np.log10(_f(delta_tab)), _f(values), bc_type="natural")
    return _f(spline(np.log10(_f(delta_halo))))


def tinker08_b_exponent(delta_halo: npt.ArrayLike) -> FloatArray:
    r"""Tinker et al. (2008) eq. 8: :math:`\log_{10}\alpha = -[0.75/\log_{10}(\Delta/75)]^{1.2}`.

    Defined for :math:`\Delta > 75` only.
    """
    return _f(10.0 ** (-((0.75 / np.log10(_f(delta_halo) / 75.0)) ** 1.2)))


def tinker08(
    sigma: npt.ArrayLike,
    A: npt.ArrayLike,
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    c: npt.ArrayLike,
) -> FloatArray:
    r"""Tinker et al. (2008) eq. 3: :math:`A[(\sigma/b)^{-a} + 1]\exp(-c/\sigma^2)`."""
    sigma = _f(sigma)
    return _f(A) * ((sigma / _f(b)) ** -_f(a) + 1.0) * np.exp(-_f(c) / sigma**2)


def tinker10_norm(
    beta: npt.ArrayLike, gamma: npt.ArrayLike, phi: npt.ArrayLike, eta: npt.ArrayLike
) -> FloatArray:
    r"""The :math:`\alpha` that normalises the Tinker et al. (2010) form to unit mass.

    Requires :math:`\beta > 0`, :math:`\gamma > 0`, :math:`\eta > -1/2` and
    :math:`\eta - \phi > -1/2`.
    """
    beta, gamma, phi, eta = _f(beta), _f(gamma), _f(phi), _f(eta)
    return _f(
        1.0
        / (
            2.0 ** (eta - phi - 0.5)
            * beta ** (-2.0 * phi)
            * gamma ** (-0.5 - eta)
            * (
                2.0**phi * beta ** (2.0 * phi) * sp.gamma(eta + 0.5)
                + gamma**phi * sp.gamma(0.5 + eta - phi)
            )
        )
    )


def tinker10(
    nu: npt.ArrayLike,
    alpha: npt.ArrayLike,
    beta: npt.ArrayLike,
    gamma: npt.ArrayLike,
    phi: npt.ArrayLike,
    eta: npt.ArrayLike,
) -> FloatArray:
    r"""Tinker et al. (2010) eq. 8, times :math:`\nu`.

    :math:`\alpha[1 + (\beta\nu)^{-2\phi}]\nu^{2\eta + 1}\exp(-\gamma\nu^2/2)`.
    """
    nu = _f(nu)
    return (
        _f(alpha)
        * (1.0 + (_f(beta) * nu) ** (-2.0 * _f(phi)))
        * nu ** (2.0 * _f(eta) + 1.0)
        * np.exp(-_f(gamma) * nu**2 / 2.0)
    )


def bhattacharya_norm(p: npt.ArrayLike, q: npt.ArrayLike) -> FloatArray:
    r"""The amplitude that normalises the Bhattacharya et al. (2011) form to unit mass.

    The integral of :math:`f/A` over :math:`\ln\sigma^{-1}` is
    :math:`2^{q/2 - p - 1/2}[2^p\Gamma(q/2) + \Gamma(q/2 - p)]/\sqrt{\pi}`, finite for
    :math:`q > 0` and :math:`2p < q`; this returns its reciprocal.
    """
    p, q = _f(p), _f(q)
    return _f(
        np.sqrt(np.pi)
        / (2.0 ** (-0.5 - p + q / 2.0) * (2.0**p * sp.gamma(q / 2.0) + sp.gamma(q / 2.0 - p)))
    )


def bhattacharya(
    nu: npt.ArrayLike,
    A: npt.ArrayLike,
    a: npt.ArrayLike,
    p: npt.ArrayLike,
    q: npt.ArrayLike,
) -> FloatArray:
    r"""Bhattacharya et al. (2011) eq. 12: Sheth-Tormen times :math:`(\sqrt{a}\,\nu)^{q-1}`."""
    return sheth_tormen(nu, A, a, p) * (np.sqrt(_f(a)) * _f(nu)) ** (_f(q) - 1.0)


def behroozi_modify_dndm(
    m_msun: npt.ArrayLike,
    dndm: npt.ArrayLike,
    z: npt.ArrayLike,
    ngtm: npt.ArrayLike,
    h: npt.ArrayLike,
) -> FloatArray:
    r"""The Behroozi et al. (2013) App. G correction to the Tinker (2008) mass function.

    .. math::

        \frac{dn}{dM} = \theta\frac{dn_{\rm T08}}{dM} - n_{\rm T08}(>M)\frac{d\theta}{dM},
        \qquad \theta = 10^{\alpha(M/M_\star)^\gamma},

    with :math:`\alpha = 0.144/(1 + \exp[14.79(a - 0.213)])`,
    :math:`\gamma = 0.5/(1 + \exp(6.5a))`, :math:`a = 1/(1+z)` and
    :math:`M_\star = 10^{11.5}\,M_\odot`. This is the derivative of
    :math:`n(>M) = \theta\, n_{\rm T08}(>M)`.

    Parameters
    ----------
    m_msun
        Halo masses in Msun (**not** Msun/h: the pivot mass is in Msun).
    dndm
        The Tinker (2008) mass function, per unit mass in Msun/h (canonical
        h^4 / (Msun Mpc^3)).
    z
        Redshift.
    ngtm
        The cumulative Tinker (2008) mass function n(>M), in h^3 / Mpc^3.
    h
        The dimensionless Hubble parameter, used to take dtheta/dM per unit Msun/h,
        consistently with ``dndm``.

    Returns
    -------
    numpy.ndarray
        The corrected dn/dM, in the units of ``dndm``. Non-finite inputs give
        non-finite outputs: nothing is replaced by a sentinel.
    """
    m_msun, z, h = _f(m_msun), _f(z), _f(h)
    a = 1.0 / (1.0 + z)
    alpha = 0.144 / (1.0 + np.exp(14.79 * (a - 0.213)))
    gamma = 0.5 / (1.0 + np.exp(6.5 * a))
    mscale = (m_msun / 10**11.5) ** gamma
    theta = 10.0 ** (alpha * mscale)
    dtheta_dm = theta * np.log(10.0) * alpha * gamma * mscale / (m_msun * h)
    return _f(_f(dndm) * theta - _f(ngtm) * dtheta_dm)


def bocquet16_mass_ratio_200c(
    m_msun: npt.ArrayLike, z: npt.ArrayLike, omega_m0: npt.ArrayLike
) -> FloatArray:
    r"""Bocquet et al. (2016) eq. A2: the mass ratio :math:`M_{200c}/M_{200m}`.

    Parameters
    ----------
    m_msun
        Halo mass in Msun (the fit takes :math:`\ln(M/M_\odot)`).
    z
        Redshift.
    omega_m0
        The matter density parameter today.

    Returns
    -------
    numpy.ndarray
        The ratio, an approximation calibrated for NFW haloes with the Duffy et al.
        (2008) concentrations.
    """
    m_msun, z, om = _f(m_msun), _f(z), _f(omega_m0)
    g0 = 3.54e-2 + om**0.09
    g1 = 4.56e-2 + 2.68e-2 / om
    g2 = 0.721 + 3.5e-2 / om
    g3 = 0.628 + 0.164 / om
    d0 = -1.67e-2 + 2.18e-2 * om
    d1 = 6.52e-3 - 6.86e-3 * om
    g = g0 + g1 * np.exp(-(((g2 - z) / g3) ** 2))
    d = d0 + d1 * z
    return _f(g + d * np.log(m_msun))


def bocquet16_mass_ratio_500c(
    m_msun: npt.ArrayLike, z: npt.ArrayLike, omega_m0: npt.ArrayLike
) -> FloatArray:
    r"""Bocquet et al. (2016) eq. 6: the mass ratio :math:`M_{500c}/M_{200m}`.

    Parameters are as for :func:`bocquet16_mass_ratio_200c`.
    """
    m_msun, z, om = _f(m_msun), _f(z), _f(omega_m0)
    alpha_0 = 0.880 + 0.329 * om
    alpha_1 = 1.0 + 4.31e-2 / om
    alpha_2 = -0.365 + 0.254 / om
    alpha = alpha_0 * (alpha_1 * z + alpha_2) / (z + alpha_2)
    beta = -1.7e-2 + om * 3.74e-3
    return _f(alpha + beta * np.log(m_msun))
