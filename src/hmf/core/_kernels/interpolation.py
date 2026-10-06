"""Local interpolation kernels on uniform lattices.

These interpolate in a local coordinate ``u`` in ``[0, 1]`` between two adjacent
nodes ``x0`` and ``x0 + h`` of a uniform lattice, from the values (and derivatives)
at a fixed set of nodes. Each result depends only on those nodes and on ``u``, so
interpolants on a lazily extended lattice do not change as the lattice grows, and
all of them are elementwise (batch-size independent).

* :func:`hermite_cubic`: values and first derivatives at the two ends;
* :func:`hermite_quintic`: values, first and second derivatives at the two ends;
* :func:`lagrange4`: values at four nodes, ``-1, 0, 1, 2`` (in units of ``h``);
* :func:`invert_hermite`: the inverse of a decreasing Hermite interpolant.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["hermite_cubic", "hermite_quintic", "invert_hermite", "lagrange4"]

FloatArray = NDArray[np.float64]


def hermite_cubic(
    u: FloatArray, h: float, y0: FloatArray, y1: FloatArray, d0: FloatArray, d1: FloatArray
) -> FloatArray:
    """The cubic Hermite interpolant on ``[x0, x0 + h]`` at ``x0 + u h``.

    Parameters
    ----------
    u
        The local coordinate, in ``[0, 1]``.
    h
        The width of the interval.
    y0, y1
        The values at its ends.
    d0, d1
        The derivatives at its ends.

    Returns
    -------
    ndarray
        The interpolant.
    """
    u2 = u * u
    u3 = u2 * u
    return (
        (2 * u3 - 3 * u2 + 1) * y0
        + (u3 - 2 * u2 + u) * (h * d0)
        + (3 * u2 - 2 * u3) * y1
        + (u3 - u2) * (h * d1)
    )


def hermite_quintic(
    u: FloatArray,
    h: float,
    y0: FloatArray,
    y1: FloatArray,
    d0: FloatArray,
    d1: FloatArray,
    c0: FloatArray,
    c1: FloatArray,
) -> FloatArray:
    """The quintic Hermite interpolant on ``[x0, x0 + h]`` at ``x0 + u h``.

    Parameters
    ----------
    u
        The local coordinate, in ``[0, 1]``.
    h
        The width of the interval.
    y0, y1
        The values at its ends.
    d0, d1
        The first derivatives at its ends.
    c0, c1
        The second derivatives at its ends.

    Returns
    -------
    ndarray
        The interpolant.
    """
    u2 = u * u
    u3 = u2 * u
    u4 = u3 * u
    u5 = u4 * u
    h2 = h * h
    return (
        (1 - 10 * u3 + 15 * u4 - 6 * u5) * y0
        + (u - 6 * u3 + 8 * u4 - 3 * u5) * (h * d0)
        + (u2 - 3 * u3 + 3 * u4 - u5) * (h2 * c0 / 2)
        + (u3 - 2 * u4 + u5) * (h2 * c1 / 2)
        + (7 * u4 - 4 * u3 - 3 * u5) * (h * d1)
        + (10 * u3 - 15 * u4 + 6 * u5) * y1
    )


def lagrange4(
    u: FloatArray, f_m1: FloatArray, f_0: FloatArray, f_1: FloatArray, f_2: FloatArray
) -> FloatArray:
    """The 4-point Lagrange interpolant through nodes at -1, 0, 1, 2, evaluated at ``u``.

    Parameters
    ----------
    u
        The local coordinate, in ``[0, 1]`` (between the middle two nodes).
    f_m1, f_0, f_1, f_2
        The values at the four nodes.

    Returns
    -------
    ndarray
        The interpolant.
    """
    up1 = u + 1
    um1 = u - 1
    um2 = u - 2
    return (
        -(u * um1 * um2) / 6 * f_m1
        + (up1 * um1 * um2) / 2 * f_0
        - (up1 * u * um2) / 2 * f_1
        + (up1 * u * um1) / 6 * f_2
    )


def invert_hermite(
    target: FloatArray,
    h: float,
    y0: FloatArray,
    y1: FloatArray,
    d0: FloatArray,
    d1: FloatArray,
    c0: FloatArray | None = None,
    c1: FloatArray | None = None,
    n_iter: int = 64,
) -> FloatArray:
    """Solve ``hermite(u) = target`` for ``u`` in ``[0, 1]``, for a decreasing interpolant.

    Uses bisection with a fixed number of steps, so the result is deterministic and
    elementwise. The caller guarantees ``y0 >= target >= y1`` and that the interpolant
    is decreasing on the interval.

    Parameters
    ----------
    target
        The values to find.
    h, y0, y1, d0, d1
        As for :func:`hermite_cubic`.
    c0, c1
        The second derivatives, for :func:`hermite_quintic`; None for the cubic.
    n_iter
        The number of bisection steps (64 reaches machine precision in u).

    Returns
    -------
    ndarray
        ``u``.
    """
    lo = np.zeros(np.shape(target))
    hi = np.ones(np.shape(target))
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if c0 is None or c1 is None:
            g = hermite_cubic(mid, h, y0, y1, d0, d1)
        else:
            g = hermite_quintic(mid, h, y0, y1, d0, d1, c0, c1)
        above = g > target
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    return 0.5 * (lo + hi)
