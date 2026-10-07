"""Interpolation kernels: local interpolants on uniform lattices, and frozen splines.

The local interpolants work in a coordinate ``u`` in ``[0, 1]`` between two adjacent
nodes ``x0`` and ``x0 + h`` of a uniform lattice, from the values (and derivatives)
at a fixed set of nodes. Each result depends only on those nodes and on ``u``, so
interpolants on a lazily extended lattice do not change as the lattice grows, and
all of them are elementwise (batch-size independent).

* :func:`hermite_cubic`: values and first derivatives at the two ends;
* :func:`hermite_quintic`: values, first and second derivatives at the two ends;
* :func:`lagrange4`: values at four nodes, ``-1, 0, 1, 2`` (in units of ``h``);
* :func:`invert_hermite`: the inverse of a decreasing Hermite interpolant.

Tables (of a transfer function, a growth factor, a power spectrum) are interpolated
by cubic splines, fitted once and kept as their coefficients:

* :class:`FrozenSpline`: a :class:`scipy.interpolate.CubicSpline`'s breakpoints and
  piecewise-polynomial coefficients, evaluated by
  :meth:`scipy.interpolate.PPoly.construct_fast`, so it gives exactly the spline's
  values;
* :func:`extrapolate_power_law`: a spline of ln y against ln x, continued beyond its
  ends as straight lines (power laws in y) with its slopes there.
"""

from __future__ import annotations

from typing import Any

import attrs
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline, PPoly

from .arrays import read_only

__all__ = [
    "FrozenSpline",
    "extrapolate_power_law",
    "hermite_cubic",
    "hermite_quintic",
    "invert_hermite",
    "lagrange4",
]

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


@attrs.frozen(eq=False)
class FrozenSpline:
    """A cubic spline, fitted once and kept as its piecewise-polynomial coefficients.

    It is immutable (its arrays are read-only) and holds no scipy object, only
    arrays. Evaluating it builds a :class:`~scipy.interpolate.PPoly` with
    :meth:`~scipy.interpolate.PPoly.construct_fast`, which gives exactly the values
    (and derivatives) of the :class:`~scipy.interpolate.CubicSpline` it was fitted as.
    Build it with :meth:`fit`.
    """

    #: The breakpoints (the table's x), increasing.
    x: FloatArray
    #: The coefficients, of shape ``(4, x.size - 1)``, as ``CubicSpline.c``.
    c: FloatArray
    #: Whether to extrapolate beyond the ends with the end polynomials (else NaN).
    extrapolate: bool = True

    @classmethod
    def fit(
        cls, x: FloatArray, y: FloatArray, *, bc_type: Any = "not-a-knot", extrapolate: bool = True
    ) -> FrozenSpline:
        """Fit a cubic spline through ``(x, y)``.

        Parameters
        ----------
        x
            The nodes, strictly increasing.
        y
            The values at the nodes.
        bc_type
            The boundary conditions, as for :class:`~scipy.interpolate.CubicSpline`.
        extrapolate
            Whether to extrapolate beyond the ends with the end polynomials (else
            the spline is NaN there).

        Returns
        -------
        FrozenSpline
        """
        spline = CubicSpline(x, y, bc_type=bc_type)
        return cls(x=read_only(x), c=read_only(spline.c), extrapolate=extrapolate)

    def __call__(self, x: Any, nu: int = 0) -> FloatArray:
        """The spline (``nu = 0``), or its ``nu``-th derivative, at ``x``.

        Parameters
        ----------
        x
            Where to evaluate it.
        nu
            The order of the derivative.

        Returns
        -------
        ndarray
            With the shape of ``x``.
        """
        out: FloatArray = PPoly.construct_fast(self.c, self.x, extrapolate=self.extrapolate)(x, nu)
        return out


def extrapolate_power_law(spline: FrozenSpline, ln_x: FloatArray) -> FloatArray:
    """A spline of ln y against ln x, continued as power laws beyond its ends.

    Inside the spline's nodes it is the spline. Below the first node and above the
    last it is the straight line (in ln y against ln x, so a power law in y) through
    the spline's value at that node, with the spline's slope there.

    Parameters
    ----------
    spline
        The spline of ln y against ln x.
    ln_x
        Where to evaluate it.

    Returns
    -------
    ndarray
        ln y, with the shape of ``ln_x``.
    """
    lo, hi = spline.x[0], spline.x[-1]
    ln_y = spline(np.clip(ln_x, lo, hi))
    slope_lo, slope_hi = spline(lo, 1), spline(hi, 1)
    ln_y = np.where(ln_x < lo, ln_y + slope_lo * (ln_x - lo), ln_y)
    out: FloatArray = np.where(ln_x > hi, ln_y + slope_hi * (ln_x - hi), ln_y)
    return out
