r"""Window functions of the smoothing filters, and their logarithmic derivatives.

Each window is a function of the dimensionless :math:`x = kR`. For each filter there
is the window :math:`W(x)`, its first logarithmic derivative :math:`W' = dW/d\ln x` and
its second, :math:`W'' = d^2W/d(\ln x)^2`. All three are dimensionless, so these
kernels have no units at all.

The ``*_window_derivatives`` kernels compute the three together, sharing the
expensive parts (the trigonometric functions of the top-hat).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit

__all__ = [
    "TOPHAT_SERIES_THRESHOLD",
    "sharpk_window",
    "smoothk_window_derivatives",
    "tophat_window_derivatives",
]

#: Below this value of :math:`x = kR`, the top-hat window and its derivatives are
#: evaluated from their Taylor series, because the closed forms lose precision to
#: cancellation (their relative error grows like :math:`\epsilon/x^4`; the closed-form
#: :math:`W'` is entirely wrong by :math:`x = 10^{-4}`, see hmf 3.7.2, PR #403). With
#: the series kept to :math:`x^{10}`, either branch of all three functions has a
#: relative error below about 1e-10 everywhere.
TOPHAT_SERIES_THRESHOLD = 0.1

# Taylor coefficients of W = sum_n a_n x^{2n}: a_n = 3 (-1)^n / ((2n+3) (2n+1)!).
# Then W' = sum 2n a_n x^{2n} and W'' = sum (2n)^2 a_n x^{2n}.
_TOPHAT_A = (1.0, -1 / 10, 1 / 280, -1 / 15120, 1 / 1330560, -1 / 172972800)


def _even_series(x2: NDArray[np.float64], coeffs: tuple[float, ...]) -> NDArray[np.float64]:
    """Evaluate ``sum_n coeffs[n] * x2**n`` by Horner's rule."""
    out = np.full_like(x2, coeffs[-1])
    for c in coeffs[-2::-1]:
        out = out * x2 + c
    return out


def tophat_window_derivatives(x: ArrayLike, order: int = 2) -> tuple[NDArray[np.float64], ...]:
    r"""The top-hat window and its logarithmic derivatives.

    .. math::

        W(x) &= \frac{3 (\sin x - x \cos x)}{x^3}, \\
        \frac{dW}{d\ln x} &= \frac{3 (x^2 \sin x + 3 x \cos x - 3 \sin x)}{x^3}, \\
        \frac{d^2W}{d(\ln x)^2} &= \frac{3 (x^3 \cos x - 4 x^2 \sin x - 9 x \cos x
            + 9 \sin x)}{x^3},

    with Taylor series below :data:`TOPHAT_SERIES_THRESHOLD`.

    Parameters
    ----------
    x
        :math:`kR`, dimensionless, ``>= 0``.
    order
        The highest derivative to return: 0, 1 or 2.

    Returns
    -------
    tuple of ndarray
        ``(W,)``, ``(W, W')`` or ``(W, W', W'')``, each of the shape of ``x``.
    """
    x = np.asarray(x, dtype=float)
    small = x < TOPHAT_SERIES_THRESHOLD
    xs = np.where(small, 1.0, x)
    sin, cos = np.sin(xs), np.cos(xs)
    x2 = x * x
    x3 = xs * xs * xs
    a = _TOPHAT_A
    out = [np.where(small, _even_series(x2, a), 3 * (sin - xs * cos) / x3)]
    if order >= 1:
        series = _even_series(x2, tuple(2 * n * c for n, c in enumerate(a)))
        closed = 3 * (xs * xs * sin + 3 * xs * cos - 3 * sin) / x3
        out.append(np.where(small, series, closed))
    if order >= 2:
        series = _even_series(x2, tuple(4 * n * n * c for n, c in enumerate(a)))
        closed = 3 * (x3 * cos - 4 * xs * xs * sin - 9 * xs * cos + 9 * sin) / x3
        out.append(np.where(small, series, closed))
    return tuple(out)


def smoothk_window_derivatives(
    x: ArrayLike, beta: float, order: int = 2
) -> tuple[NDArray[np.float64], ...]:
    r"""The smooth-k window of Leo et al. (2018) and its logarithmic derivatives.

    .. math::

        W(x) &= \frac{1}{1 + x^\beta}, \\
        \frac{dW}{d\ln x} &= -\beta W (1 - W), \\
        \frac{d^2W}{d(\ln x)^2} &= \beta^2 W (1 - W) (1 - 2W).

    Parameters
    ----------
    x
        :math:`kR`, dimensionless, ``>= 0``.
    beta
        The steepness of the cut-off, ``> 0``.
    order
        The highest derivative to return: 0, 1 or 2.

    Returns
    -------
    tuple of ndarray
        ``(W,)``, ``(W, W')`` or ``(W, W', W'')``, each of the shape of ``x``.
    """
    x = np.asarray(x, dtype=float)
    # W = expit(-beta ln x): no overflow for large beta ln x, and W(0) = 1 exactly.
    with np.errstate(divide="ignore"):
        w = expit(-beta * np.log(x))
    # 1 - W = expit(beta ln x), computed directly so that it does not cancel at small x.
    with np.errstate(divide="ignore"):
        one_minus_w = expit(beta * np.log(x))
    out = [w]
    if order >= 1:
        out.append(-beta * w * one_minus_w)
    if order >= 2:
        out.append(beta * beta * w * one_minus_w * (one_minus_w - w))
    return tuple(out)


def sharpk_window(x: ArrayLike) -> NDArray[np.float64]:
    r"""The sharp-k window, :math:`W(x) = \Theta(1 - x)`, with :math:`W(1) = 1/2`.

    Its derivative is a Dirac delta, so it has no derivative kernel: the mass variance
    of the sharp-k filter is computed by
    :func:`hmf.core._kernels.mass_variance.sharpk_ln_variance`.

    Parameters
    ----------
    x
        :math:`kR`, dimensionless, ``>= 0``.

    Returns
    -------
    ndarray
        The window, of the shape of ``x``.
    """
    x = np.asarray(x, dtype=float)
    return np.where(x < 1, 1.0, np.where(x == 1, 0.5, 0.0))
