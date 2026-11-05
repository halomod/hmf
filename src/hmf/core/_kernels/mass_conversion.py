r"""Kernels of the conversion of halo masses between mass definitions, for a profile.

A halo of density profile :math:`\rho(r) = \rho_s f(r/r_s)` encloses, within
:math:`r = x r_s`, the mass :math:`M(<x r_s) = 4\pi\rho_s r_s^3\,\mu(x)`, with
:math:`\mu(x) = \int_0^x f(y) y^2\,dy`, so its mean enclosed density is
:math:`\bar\rho(<x r_s) = 3\rho_s F(x)`, with :math:`F(x) = \mu(x)/x^3`. For an NFW
profile (Navarro, Frenk & White 1997)

.. math:: \mu(x) = \ln(1 + x) - \frac{x}{1 + x}.

A spherical-overdensity definition A of halo density :math:`\rho_A` sets the
concentration :math:`c_A` by :math:`3\rho_s F(c_A) = \rho_A`. The same halo in a
definition B has the concentration :math:`c_B` with

.. math:: F(c_B) = q\,F(c_A), \qquad q = \rho_B / \rho_A,

and the mass :math:`M_B = M_A\,\mu(c_B)/\mu(c_A)`. Only the ratio of the two halo
densities enters, so it does not matter whether they are comoving or physical, or
relative to the mean or the critical density, as long as both are the same.

A profile enters only through :math:`\ln\mu(x)` and its slope
:math:`\psi(x) = d\ln\mu/d\ln x`, which is below 3 for a density that falls
outwards, so :math:`\ln F` decreases monotonically in :math:`\ln x` and the root is
unique. :func:`concentration_ratio_solve` solves :math:`F(c_B) = q F(c_A)` for every
element at once by a fixed number of Newton steps in :math:`\ln x` from a starting
point, each elementwise, so the result does not depend on the batch size, and
:func:`profile_convert` gives the masses and the analytic Jacobian. For NFW
(:func:`nfw_ln_mu_and_slope`) the starting point is the fit of Hu & Kravtsov (2003,
eq. C11; :func:`hu_kravtsov_inverse`), and :func:`nfw_concentration_ratio_solve` and
:func:`nfw_convert` are the two with it.

The functions follow the rules of :mod:`hmf.core._kernels` and do not check their
inputs: the kernel-level entry points of :mod:`hmf.core.profiles` do.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

__all__ = [
    "NEWTON_STEPS",
    "LnMuAndSlope",
    "concentration_ratio_solve",
    "hu_kravtsov_inverse",
    "nfw_concentration_ratio_solve",
    "nfw_convert",
    "nfw_ln_mean_density",
    "nfw_ln_mu",
    "nfw_ln_mu_and_slope",
    "nfw_ln_mu_slope",
    "nfw_start",
    "power_law_start",
    "profile_convert",
]

FloatArray = npt.NDArray[np.float64]

#: A profile's :math:`x \mapsto (\ln\mu(x), \psi(x))`, on plain arrays.
LnMuAndSlope = Callable[[FloatArray], tuple[FloatArray, FloatArray]]

#: The number of Newton steps that polish the Hu & Kravtsov (2003) starting point.
#: Newton converges quadratically from its error (below 1% for 0 < c < 400): for
#: 1e-3 < c < 1e4 and 1e-4 < q < 1e4 two steps leave 1e-12 and three round-off
#: (~1e-13 in c); the fourth is a margin.
NEWTON_STEPS = 4

#: Below this x, :func:`nfw_ln_mu` uses the series of mu(x), whose closed form loses
#: digits to cancellation (relative error ~ 1e-16 / x^2).
_SERIES_X = 0.01

#: The parameters (a1, a2, a3, a4) of Hu & Kravtsov (2003), eq. C11.
_HK_A = (0.5116, -0.4283, -3.13e-3, -3.52e-5)


def nfw_ln_mu_and_slope(x: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    r"""The NFW :math:`\ln\mu(x)` and :math:`\psi(x)`, from one evaluation of :math:`\mu`.

    See :func:`nfw_ln_mu` and :func:`nfw_ln_mu_slope`.

    Parameters
    ----------
    x
        :math:`r/r_s` (> 0), dimensionless.

    Returns
    -------
    ln_mu, psi : numpy.ndarray
        With the shape of ``x``.
    """
    x = np.asarray(x, dtype=np.float64)
    ln_x = np.log(x)
    ln_1px = np.log1p(x)
    mu = ln_1px - x / (1.0 + x)
    if np.any(x < _SERIES_X):
        xs = np.minimum(x, _SERIES_X)
        # mu(x) / x^2 = 1/2 - 2x/3 + 3x^2/4 - ... (Horner, to x^7).
        series = np.zeros_like(xs)
        for n in range(9, 1, -1):
            series = series * xs + (-1) ** n * (n - 1) / n
        mu = np.where(x < _SERIES_X, series * xs**2, mu)
    ln_mu = np.log(mu)
    psi = np.exp(2.0 * (ln_x - ln_1px) - ln_mu)
    return ln_mu, psi


def nfw_ln_mu(x: npt.ArrayLike) -> FloatArray:
    r"""The log of the NFW mass function :math:`\mu(x) = \ln(1+x) - x/(1+x)`.

    Below x = 0.01 it uses the series
    :math:`\mu(x) = \sum_{n\ge2} (-1)^n (n-1)x^n/n` (to :math:`x^{9}`), which avoids
    the cancellation in the closed form.

    Parameters
    ----------
    x
        :math:`r/r_s` (> 0), dimensionless.

    Returns
    -------
    numpy.ndarray
        :math:`\ln\mu(x)`.
    """
    return nfw_ln_mu_and_slope(np.asarray(x, dtype=np.float64))[0]


def nfw_ln_mu_slope(x: npt.ArrayLike) -> FloatArray:
    r"""The logarithmic slope :math:`\psi(x) = d\ln\mu/d\ln x = x^2/((1+x)^2\mu(x))`.

    It falls from 2 (as :math:`x \to 0`) to 0 (as :math:`x \to \infty`).

    Parameters
    ----------
    x
        :math:`r/r_s` (> 0), dimensionless.

    Returns
    -------
    numpy.ndarray
        :math:`\psi(x)`, dimensionless.
    """
    return nfw_ln_mu_and_slope(np.asarray(x, dtype=np.float64))[1]


def nfw_ln_mean_density(x: npt.ArrayLike) -> FloatArray:
    r""":math:`\ln F(x) = \ln[\mu(x)/x^3]`: the mean enclosed density over :math:`3\rho_s`.

    Its slope in :math:`\ln x` is :math:`\psi(x) - 3`, between -1 and -3, so it
    decreases monotonically.

    Parameters
    ----------
    x
        :math:`r/r_s` (> 0), dimensionless.

    Returns
    -------
    numpy.ndarray
        :math:`\ln F(x)`.
    """
    x = np.asarray(x, dtype=np.float64)
    out: FloatArray = nfw_ln_mu(x) - 3.0 * np.log(x)
    return out


def hu_kravtsov_inverse(ln_f: npt.ArrayLike) -> FloatArray:
    r"""The concentration c with :math:`F(c) = f`, from the fit of Hu & Kravtsov (2003).

    Their eq. C11 inverts :math:`f(x) = x^3[\ln(1 + 1/x) - 1/(1 + x)]`, which is
    :math:`F(1/x)`:

    .. math:: x(f) = \left[a_1 f^{2p} + (3/4)^2\right]^{-1/2} + 2f, \qquad
              p = a_2 + a_3\ln f + a_4(\ln f)^2,

    with :math:`(a_1, \ldots, a_4) = (0.5116, -0.4283, -3.13\times10^{-3},
    -3.52\times10^{-5})`, and :math:`c = 1/x(f)`. They quote masses converted with it
    to better than 1% for 0 < c < 400, and typically 0.3% for c < 20.

    Parameters
    ----------
    ln_f
        :math:`\ln f`, the log of the target mean density over :math:`3\rho_s`.

    Returns
    -------
    numpy.ndarray
        The approximate concentration, dimensionless.
    """
    ln_f = np.asarray(ln_f, dtype=np.float64)
    a1, a2, a3, a4 = _HK_A
    p = a2 + a3 * ln_f + a4 * ln_f**2
    x = (a1 * np.exp(2.0 * p * ln_f) + 0.5625) ** -0.5 + 2.0 * np.exp(ln_f)
    out: FloatArray = 1.0 / x
    return out


def power_law_start(c: FloatArray, ln_q: FloatArray, psi: FloatArray) -> FloatArray:
    r"""A starting point for any profile: :math:`F` as the local power law at c.

    :math:`\ln c' = \ln c + \ln q/(\psi(c) - 3)`, one Newton step from c (exact for a
    power-law profile).

    Parameters
    ----------
    c
        The concentrations (> 0).
    ln_q
        :math:`\ln q`, broadcast with ``c``.
    psi
        :math:`\psi(c)`.

    Returns
    -------
    numpy.ndarray
        The starting concentration.
    """
    out: FloatArray = c * np.exp(ln_q / (psi - 3.0))
    return out


def concentration_ratio_solve(
    ln_mu_and_slope: LnMuAndSlope,
    c: npt.ArrayLike,
    ln_q: npt.ArrayLike,
    c_start: npt.ArrayLike,
    steps: int,
) -> FloatArray:
    r"""The concentration :math:`c'` with :math:`F(c') = q F(c)`, elementwise, for a profile.

    The halo of concentration c in a definition of halo density :math:`\rho` has the
    concentration :math:`c'` in a definition of halo density :math:`q\rho`. The root of
    :math:`g(y) = \ln\mu(e^y) - 3y - \ln q - \ln F(c)` in :math:`y = \ln c'` is found
    by ``steps`` Newton steps from ``c_start`` (:math:`g'(y) = \psi - 3 < 0`). Where
    :math:`q = 1` the result is ``c`` itself, exactly.

    Parameters
    ----------
    ln_mu_and_slope
        The profile's :math:`(\ln\mu, \psi)` (e.g. :func:`nfw_ln_mu_and_slope`).
    c
        The concentrations (> 0), dimensionless.
    ln_q
        :math:`\ln q`, the log of the ratio of the new halo density to the old;
        broadcast with ``c``.
    c_start
        The starting concentrations, broadcast with ``c``.
    steps
        The number of Newton steps.

    Returns
    -------
    numpy.ndarray
        :math:`c'`, with the broadcast shape.
    """
    c, ln_q = np.broadcast_arrays(np.asarray(c, dtype=np.float64), np.asarray(ln_q, np.float64))
    ln_mu_c, _ = ln_mu_and_slope(c)
    target = ln_mu_c - 3.0 * np.log(c) + ln_q
    y = np.array(np.broadcast_to(np.log(c_start), c.shape))
    for _ in range(steps):
        ln_mu, psi = ln_mu_and_slope(np.exp(y))
        y = y - (ln_mu - 3.0 * y - target) / (psi - 3.0)
    out: FloatArray = np.where(ln_q == 0.0, c, np.exp(y))
    return out


def profile_convert(
    ln_mu_and_slope: LnMuAndSlope,
    c: npt.ArrayLike,
    slope: npt.ArrayLike,
    ln_q: npt.ArrayLike,
    c_start: npt.ArrayLike,
    steps: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    r"""Convert haloes of a profile to a definition of ``q`` times the halo density.

    For haloes of mass :math:`M` and concentration :math:`c(M)` in a definition A,
    with :math:`s = d\ln c/d\ln M`, the new concentration :math:`c'` solves
    :math:`F(c') = qF(c)` (:func:`concentration_ratio_solve`) and the new mass is
    :math:`M' = M\mu(c')/\mu(c)`. Differentiating the root implicitly, with
    :math:`\phi = \psi - 3 = d\ln F/d\ln x` (never zero):

    .. math:: \frac{d\ln c'}{d\ln M} = \frac{\phi(c)}{\phi(c')}\,s, \qquad
              \frac{d\ln M'}{d\ln M} = 1 + s\left[\psi(c')\frac{\phi(c)}{\phi(c')}
              - \psi(c)\right].

    Parameters
    ----------
    ln_mu_and_slope
        The profile's :math:`(\ln\mu, \psi)`.
    c
        The concentrations in the old definition (> 0).
    slope
        :math:`s = d\ln c/d\ln M` of the concentration-mass relation, in the old
        definition.
    ln_q
        :math:`\ln q = \ln(\rho_{\rm new}/\rho_{\rm old})`.
    c_start, steps
        As for :func:`concentration_ratio_solve`.

    Returns
    -------
    ln_mass_ratio : numpy.ndarray
        :math:`\ln(M'/M)`.
    c_new : numpy.ndarray
        :math:`c'`.
    dln_m_new_dln_m : numpy.ndarray
        :math:`d\ln M'/d\ln M`.
    dln_c_new_dln_m : numpy.ndarray
        :math:`d\ln c'/d\ln M`.

    All have the broadcast shape of the inputs.
    """
    c_new = concentration_ratio_solve(ln_mu_and_slope, c, ln_q, c_start, steps)
    c_old = np.broadcast_to(np.asarray(c, dtype=np.float64), c_new.shape)
    ln_mu_old, psi_old = ln_mu_and_slope(c_old)
    ln_mu_new, psi_new = ln_mu_and_slope(c_new)
    ratio = (psi_old - 3.0) / (psi_new - 3.0)
    slope = np.asarray(slope, dtype=np.float64)
    ln_mass_ratio = ln_mu_new - ln_mu_old
    dln_m = 1.0 + slope * (psi_new * ratio - psi_old)
    dln_c = ratio * slope
    return ln_mass_ratio, c_new, np.asarray(dln_m, np.float64), np.asarray(dln_c, np.float64)


def nfw_start(c: FloatArray, ln_q: FloatArray) -> FloatArray:
    r"""The NFW starting point: Hu & Kravtsov (2003) at :math:`\ln F(c) + \ln q`."""
    c = np.asarray(c, dtype=np.float64)
    return hu_kravtsov_inverse(nfw_ln_mean_density(c) + ln_q)


def nfw_concentration_ratio_solve(c: npt.ArrayLike, ln_q: npt.ArrayLike) -> FloatArray:
    r"""The NFW concentration :math:`c'` with :math:`F(c') = q F(c)`, elementwise.

    :func:`concentration_ratio_solve` for NFW, from :func:`hu_kravtsov_inverse`, with
    :data:`NEWTON_STEPS` steps, so the error is at round-off.

    Parameters
    ----------
    c
        The concentrations (> 0), dimensionless.
    ln_q
        :math:`\ln q`, broadcast with ``c``.

    Returns
    -------
    numpy.ndarray
        :math:`c'`, with the broadcast shape.
    """
    c = np.asarray(c, dtype=np.float64)
    ln_q = np.asarray(ln_q, dtype=np.float64)
    return concentration_ratio_solve(nfw_ln_mu_and_slope, c, ln_q, nfw_start(c, ln_q), NEWTON_STEPS)


def nfw_convert(
    c: npt.ArrayLike, slope: npt.ArrayLike, ln_q: npt.ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    r""":func:`profile_convert` for NFW haloes (see :func:`nfw_concentration_ratio_solve`).

    Parameters
    ----------
    c
        The concentrations in the old definition (> 0).
    slope
        :math:`s = d\ln c/d\ln M`, in the old definition.
    ln_q
        :math:`\ln q = \ln(\rho_{\rm new}/\rho_{\rm old})`.

    Returns
    -------
    ln_mass_ratio, c_new, dln_m_new_dln_m, dln_c_new_dln_m : numpy.ndarray
        As for :func:`profile_convert`.
    """
    c = np.asarray(c, dtype=np.float64)
    ln_q = np.asarray(ln_q, dtype=np.float64)
    return profile_convert(nfw_ln_mu_and_slope, c, slope, ln_q, nfw_start(c, ln_q), NEWTON_STEPS)
