r"""Kernels of the mass variance and of its interpolation on the mass lattice.

Units: masses in M☉/h, radii in Mpc/h, wavenumbers in h/Mpc, power in (Mpc/h)³ and
densities in M☉ h² / Mpc³ (see :data:`hmf.core.units.CANONICAL_UNITS`). Every other
quantity here is dimensionless.

The mass variance of a filter with window :math:`W(kR)` is

.. math:: s(R) = \sigma^2(R) = \frac{1}{2\pi^2} \int k^3 P(k) W^2(kR)\, d\ln k.

Writing :math:`L = \ln R` and :math:`W' = dW/d\ln x`, :math:`W'' = d^2W/d(\ln x)^2`,

.. math::

    \frac{ds}{dL} = \frac{1}{2\pi^2} \int k^3 P\, 2 W W'\, d\ln k, \qquad
    \frac{d^2s}{dL^2} = \frac{1}{2\pi^2} \int k^3 P\, 2 (W'^2 + W W'')\, d\ln k,

so that :math:`d\ln\sigma/dL = s'/(2s)` and
:math:`d^2\ln\sigma/dL^2 = (s''/s - (s'/s)^2)/2`. All three integrals are computed
from one evaluation of the window per node (:func:`window_log_variance`).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import simpson

__all__ = [
    "GAUSS_LEGENDRE_ORDER",
    "lagrangian_mass",
    "lagrangian_radius",
    "panel_integrals",
    "panel_points",
    "sharpk_log_variance",
    "tail_integral",
    "truncation_errors",
    "window_log_variance",
]

FloatArray = NDArray[np.float64]

#: The number of Gauss-Legendre points per k-lattice interval of the sharp-k integral.
GAUSS_LEGENDRE_ORDER = 4

_GL_X, _GL_W = np.polynomial.legendre.leggauss(GAUSS_LEGENDRE_ORDER)
_TWO_PI2 = 2 * math.pi**2


# ---------------------------------------------------------------------------------
# Mass and radius
# ---------------------------------------------------------------------------------


def lagrangian_radius(m: ArrayLike, rho_mean: float, c: float = 1.0) -> FloatArray:
    r"""The filter radius of a mass: :math:`m = \frac{4\pi}{3} \bar\rho\, (cR)^3`.

    Parameters
    ----------
    m
        Mass, in M☉/h.
    rho_mean
        Mean matter density, in M☉ h² / Mpc³.
    c
        The filter's mass-assignment constant (1 for the top-hat).

    Returns
    -------
    ndarray
        The radius R, in Mpc/h.
    """
    m = np.asarray(m, dtype=float)
    return np.cbrt(3 * m / (4 * math.pi * rho_mean)) / c


def lagrangian_mass(r: ArrayLike, rho_mean: float, c: float = 1.0) -> FloatArray:
    r"""The mass of a filter radius: :math:`m = \frac{4\pi}{3} \bar\rho\, (cR)^3`.

    Parameters
    ----------
    r
        Radius, in Mpc/h.
    rho_mean
        Mean matter density, in M☉ h² / Mpc³.
    c
        The filter's mass-assignment constant (1 for the top-hat).

    Returns
    -------
    ndarray
        The mass, in M☉/h.
    """
    r = np.asarray(r, dtype=float)
    return 4 * math.pi * rho_mean * (c * r) ** 3 / 3


# ---------------------------------------------------------------------------------
# The variance of a smooth window
# ---------------------------------------------------------------------------------


def window_log_variance(
    w: FloatArray,
    dw: FloatArray,
    d2w: FloatArray | None,
    k3p: FloatArray,
    dln_k: float,
) -> tuple[FloatArray, FloatArray, FloatArray | None, FloatArray]:
    r"""The ln(sigma) and its first two derivatives in ln R, from the window on a k grid.

    The integrals use Simpson's rule (:func:`scipy.integrate.simpson`) along the last
    axis, so each row's result does not depend on the other rows.

    The same integrals with Simpson's rule on every other grid point (twice the
    spacing) give a Richardson estimate of the error from the grid's resolution:
    :math:`|I_h - I_{2h}| / 15` for a resolved integrand. Where the grid does not
    resolve the integrand (it aliases the window's oscillations), the two differ by
    as much as the integrals themselves, which flags the failure.

    Parameters
    ----------
    w, dw, d2w
        :math:`W`, :math:`W'` and :math:`W''` at :math:`kR`, of shape ``(..., nk)``.
        ``d2w`` may be None, to skip the second derivative.
    k3p
        :math:`k^3 P(k)` on the k grid, of shape ``(nk,)``, in canonical units
        (dimensionless).
    dln_k
        The spacing of the grid in ln k.

    Returns
    -------
    ln_sigma, dlnsigma_dlnr, d2lnsigma_dlnr2 : ndarray
        Of shape ``(...)``. The last is None if ``d2w`` is.
    resolution_error : ndarray
        The Richardson estimate of the relative error of sigma or of
        :math:`d\ln\sigma/d\ln R`, whichever is larger.
    """
    f0 = k3p * (w * w)
    f1 = k3p * (2 * w * dw)
    s0 = simpson(f0, dx=dln_k, axis=-1) / _TWO_PI2
    s1 = simpson(f1, dx=dln_k, axis=-1) / _TWO_PI2
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = s1 / s0
        d2 = None
        if d2w is not None:
            s2 = simpson(k3p * (2 * (dw * dw + w * d2w)), dx=dln_k, axis=-1) / _TWO_PI2
            d2 = (s2 / s0 - ratio * ratio) / 2
    s0_coarse = simpson(f0[..., ::2], dx=2 * dln_k, axis=-1) / _TWO_PI2
    s1_coarse = simpson(f1[..., ::2], dx=2 * dln_k, axis=-1) / _TWO_PI2
    # A variance that underflows to 0 gives non-finite results, which callers check.
    with np.errstate(divide="ignore", invalid="ignore"):
        err_sigma = np.abs(np.log(s0 / s0_coarse)) / 30
        err_d = np.abs(s1_coarse / s0_coarse / ratio - 1) / 15
        ln_sigma = 0.5 * np.log(s0)
    return ln_sigma, ratio / 2, d2, np.fmax(err_sigma, err_d)


# ---------------------------------------------------------------------------------
# The sharp-k variance
# ---------------------------------------------------------------------------------


def panel_points(ln_a: ArrayLike, ln_b: ArrayLike) -> FloatArray:
    """The Gauss-Legendre points of the panels ``[ln_a, ln_b]`` (in ln k).

    Parameters
    ----------
    ln_a, ln_b
        The ends of each panel, broadcast together.

    Returns
    -------
    ndarray
        Shape ``(..., GAUSS_LEGENDRE_ORDER)``.
    """
    ln_a = np.asarray(ln_a, dtype=float)[..., None]
    ln_b = np.asarray(ln_b, dtype=float)[..., None]
    out: FloatArray = 0.5 * (ln_a + ln_b) + 0.5 * (ln_b - ln_a) * _GL_X
    return out


def panel_integrals(ln_a: ArrayLike, ln_b: ArrayLike, f: FloatArray) -> FloatArray:
    """Gauss-Legendre integrals over the panels ``[ln_a, ln_b]`` in ln k.

    Parameters
    ----------
    ln_a, ln_b
        The ends of each panel.
    f
        The integrand at :func:`panel_points`, shape ``(..., GAUSS_LEGENDRE_ORDER)``.

    Returns
    -------
    ndarray
        One integral per panel.
    """
    half = 0.5 * (np.asarray(ln_b, dtype=float) - np.asarray(ln_a, dtype=float))
    # An explicit sum over the few points, in a fixed order: no BLAS.
    total = f[..., 0] * _GL_W[0]
    for i in range(1, GAUSS_LEGENDRE_ORDER):
        total = total + f[..., i] * _GL_W[i]
    out: FloatArray = half * total
    return out


def sharpk_log_variance(
    s_below: FloatArray, k3p_cut: FloatArray, n_eff_cut: FloatArray, second: bool
) -> tuple[FloatArray, FloatArray, FloatArray | None]:
    r"""The ln(sigma) and its derivatives in ln R, for the sharp-k filter.

    With :math:`g = k^3 P(k) / 2\pi^2` at the cut-off :math:`k = 1/R`,

    .. math::

        \frac{d\ln\sigma}{d\ln R} = -\frac{g}{2s}, \qquad
        \frac{d^2\ln\sigma}{d(\ln R)^2} = \frac{1}{2}\left[\frac{g (3 + n_{\rm eff})}{s}
            - \frac{g^2}{s^2}\right],

    where :math:`n_{\rm eff} = d\ln P / d\ln k` at the cut-off. Both are ratios of the
    power, so they do not depend on its normalisation.

    Parameters
    ----------
    s_below
        :math:`\sigma^2(R) = \frac{1}{2\pi^2}\int^{1/R} k^3 P\, d\ln k`.
    k3p_cut
        :math:`k^3 P(k)` at :math:`k = 1/R`.
    n_eff_cut
        :math:`d\ln P / d\ln k` at :math:`k = 1/R` (only used if ``second``).
    second
        Whether to return the second derivative.

    Returns
    -------
    ln_sigma, dlnsigma_dlnr, d2lnsigma_dlnr2 : ndarray
        The last is None unless ``second``.
    """
    # A variance that underflows to 0 gives non-finite results, which callers check.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        g_over_s = k3p_cut / _TWO_PI2 / s_below
        d2 = (g_over_s * (3 + n_eff_cut) - g_over_s * g_over_s) / 2 if second else None
        ln_sigma = 0.5 * np.log(s_below)
    return ln_sigma, -g_over_s / 2, d2


# ---------------------------------------------------------------------------------
# The truncation estimator
# ---------------------------------------------------------------------------------


def tail_integral(
    k3p_edge: ArrayLike,
    slope: ArrayLike,
    x_edge: ArrayLike,
    envelope: tuple[tuple[float, float], ...],
    high: bool,
    oscillating: tuple[tuple[float, float], ...] = (),
    omega: float = 1.0,
) -> FloatArray:
    r"""Bound a tail of a variance integral beyond the end of the k grid.

    The power beyond the edge is extrapolated as a power law,
    :math:`k^3 P \propto k^{3+n}`, and the window factor is bounded by

    .. math:: \sum_i c_i x^{p_i} + \sum_j a_j x^{q_j} \sin(\omega x + \phi_j),

    a smooth part (``envelope``) and, for the high-k tail only, a part oscillating
    with frequency :math:`\omega` in :math:`x = kR` (``oscillating``). Each smooth
    term integrates exactly: for the high-k tail

    .. math:: \int_{k_e}^\infty k^3 P\, c x^p\, d\ln k = \frac{c\, k_e^3 P_e\, x_e^p}{-(3 + n + p)},

    finite if :math:`3 + n + p < 0` (and the same with the opposite sign, finite if
    :math:`3 + n + p > 0`, for the low-k tail). Each oscillating term is bounded by the
    second mean-value theorem: :math:`|\int_{x_e}^\infty g(x) \sin(\omega x + \phi)
    dx| \le 2 g(x_e)/\omega` for a decreasing :math:`g \ge 0`, here
    :math:`g = k^3 P\, a x^{q - 1}` (decreasing if :math:`3 + n + q - 1 < 0`). Bounding
    the oscillating part this way, rather than by its envelope, matters: for the
    top-hat the envelope of :math:`W W'` falls only as :math:`x^{-3}`, but its tail
    integral as :math:`x^{-4}`. A divergent or unbounded term gives ``inf``.

    Parameters
    ----------
    k3p_edge
        :math:`k^3 P` at the edge of the grid.
    slope
        :math:`n = d\ln P/d\ln k` at the edge.
    x_edge
        :math:`kR` at the edge.
    envelope
        The smooth part, ``((c_0, p_0), (c_1, p_1), ...)``, with ``c_i >= 0``.
    high
        Whether this is the high-k tail (else the low-k one).
    oscillating
        The amplitudes of the oscillating part, ``((a_0, q_0), ...)``, ``a_j >= 0``.
    omega
        The angular frequency of the oscillating part, in x.

    Returns
    -------
    ndarray
        The bound on the tail integral, without the :math:`1/2\pi^2`.
    """
    k3p_edge = np.asarray(k3p_edge, dtype=float)
    slope = np.asarray(slope, dtype=float)
    x_edge = np.asarray(x_edge, dtype=float)
    total = np.zeros(np.broadcast_shapes(k3p_edge.shape, slope.shape, x_edge.shape))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for c, p in envelope:
            if c == 0:
                continue
            rate = -(3 + slope + p) if high else 3 + slope + p
            ok = rate > 0
            total = total + np.where(ok, c * k3p_edge * x_edge**p / np.where(ok, rate, 1), np.inf)
        for a, q in oscillating:
            if a == 0:
                continue
            ok = (3 + slope + q - 1) < 0
            total = total + np.where(ok, 2 * a * k3p_edge * x_edge ** (q - 1) / omega, np.inf)
    return total


def truncation_errors(
    ln_sigma: ArrayLike,
    dlnsigma_dlnr: ArrayLike,
    tail_w2: ArrayLike,
    tail_wdw: ArrayLike,
    tail_ddw: ArrayLike | None = None,
    step: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    r"""Bound the relative errors in sigma and dln(sigma)/dln(R) from missing integrands.

    If the parts missing from (or wrong in) :math:`2\pi^2 s`, :math:`2\pi^2 s'/2` and
    :math:`2\pi^2 s''/2` are bounded by :math:`T_0`, :math:`T_1` and :math:`T_2` (see
    :func:`tail_integral`), then with :math:`D = d\ln\sigma/d\ln R = s'/2s`,

    .. math::

        \left|\frac{\delta\sigma}{\sigma}\right| \le \frac{T_0}{2 \cdot 2\pi^2 s},
        \qquad
        \left|\frac{\delta D}{D}\right| \le \frac{T_1}{2\pi^2 s |D|}
            + \frac{T_0}{2\pi^2 s} + \frac{4}{81} h \frac{T_2}{2\pi^2 s |D|}.

    The last term is the error that a wrong second derivative at a node causes in the
    cubic Hermite interpolant of :math:`\ln|d\ln\sigma/d\ln m|` over a lattice step
    :math:`h` in ln m (at most :math:`\frac{4}{27} h` times the error in its slope).

    The error in :math:`D` is the one that enters :math:`dn/dm`. It is typically
    :math:`kR` times the error in sigma, because :math:`|W'| \sim kR\, |W|` at large
    :math:`kR`: that is why an estimator of the error in sigma alone underestimates the
    error in :math:`dn/dm` by 20-50 times.

    Parameters
    ----------
    ln_sigma
        ln(sigma) at the node.
    dlnsigma_dlnr
        :math:`D` at the node.
    tail_w2, tail_wdw, tail_ddw
        :math:`T_0`, :math:`T_1` and :math:`T_2`. ``tail_ddw`` is None if the second
        derivative is not used.
    step
        The lattice spacing :math:`h` in ln m.

    Returns
    -------
    rel_sigma, rel_dlnsigma : ndarray
        The bounds on the relative errors (``inf`` if a tail diverges).
    """
    s = _TWO_PI2 * np.exp(2 * np.asarray(ln_sigma, dtype=float))
    d = np.abs(np.asarray(dlnsigma_dlnr, dtype=float))
    t0 = np.asarray(tail_w2, dtype=float)
    t1 = np.asarray(tail_wdw, dtype=float)
    if tail_ddw is not None:
        t1 = t1 + (4 / 81) * step * np.asarray(tail_ddw, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_d = np.where(d > 0, t1 / (s * d), np.where(t1 > 0, np.inf, 0.0)) + t0 / s
        rel_sigma = t0 / (2 * s)
    return rel_sigma, rel_d
