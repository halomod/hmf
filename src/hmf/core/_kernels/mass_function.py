r"""Kernels of the cumulative mass function: quadrature on the mass lattice.

:math:`n(>m) = \int_m^{M_{\rm top}} g(\ln m')\,d\ln m'`, with :math:`g = dn/d\ln m`
(or :math:`m\,dn/d\ln m` for the mass in haloes above m), is integrated from the values
of g at the nodes of a mass lattice, :math:`\ln m = j\delta`:

* between nodes, :math:`\ln g` is interpolated by the 4-point Lagrange polynomial
  through the nodes :math:`j-1 \ldots j+2` (shifted inwards at the ends of the
  lattice), and the panel :math:`[j, j+1]` is integrated by Gauss-Legendre quadrature
  of its exponential (:func:`log_lagrange_integrals`);
* the panels are summed from the top down, always starting from the top
  (:func:`cumulative_from_top`), so the integral at node :math:`j` is the same, bit
  for bit, however far down the lattice a calculation reaches;
* between nodes, the part of the panel above m is integrated from the same
  interpolant, so the integral is continuous in m and its derivative is the
  interpolated :math:`-g`.

Every step depends only on node indices and the mass asked for, so the result does
not depend on the batch, the order of requests, or how far the lattice was extended.
Sums over quadrature points are written out in a fixed order (:func:`gauss_sum`),
not left to a reduction whose order could depend on the shape of the array.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .interpolation import lagrange4

__all__ = [
    "GAUSS_ORDER",
    "cumulative_from_top",
    "gauss_points",
    "gauss_sum",
    "log_lagrange_integrals",
]

FloatArray = npt.NDArray[np.float64]

#: The number of Gauss-Legendre points per panel (exact for polynomials of degree
#: ``2 * GAUSS_ORDER - 1``).
GAUSS_ORDER = 4

_X, _W = np.polynomial.legendre.leggauss(GAUSS_ORDER)
#: The Gauss-Legendre points mapped to [0, 1], and their weights there.
_U: FloatArray = (_X + 1.0) / 2.0
_WU: FloatArray = _W / 2.0


def gauss_points(a: npt.ArrayLike, b: npt.ArrayLike) -> FloatArray:
    """The Gauss-Legendre points on intervals ``[a, b]``.

    Parameters
    ----------
    a, b
        The ends of the intervals (e.g. in ln m): arrays that broadcast together.

    Returns
    -------
    numpy.ndarray
        The points, with a last axis of length :data:`GAUSS_ORDER` added to the
        broadcast shape of ``a`` and ``b``.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    out: FloatArray = a[..., None] + (b - a)[..., None] * _U
    return out


def gauss_sum(values: npt.ArrayLike, a: npt.ArrayLike, b: npt.ArrayLike) -> FloatArray:
    r"""The Gauss-Legendre estimate of :math:`\int_a^b f`, from ``f`` at :func:`gauss_points`.

    The weighted values are added one point after the other, in a fixed order, so the
    result does not depend on the shape of ``values``.

    Parameters
    ----------
    values
        ``f`` at ``gauss_points(a, b)``: the last axis has length :data:`GAUSS_ORDER`.
    a, b
        The ends of the intervals, as for :func:`gauss_points`.

    Returns
    -------
    numpy.ndarray
        The integrals, with the shape of ``values`` without its last axis.
    """
    values = np.asarray(values, dtype=float)
    total = _WU[0] * values[..., 0]
    for g in range(1, GAUSS_ORDER):
        total = total + _WU[g] * values[..., g]
    out: FloatArray = (np.asarray(b, dtype=float) - np.asarray(a, dtype=float)) * total
    return out


def log_lagrange_integrals(
    y: npt.ArrayLike,
    offset: npt.ArrayLike,
    u_lo: npt.ArrayLike,
    u_hi: npt.ArrayLike,
    step: float,
) -> FloatArray:
    r"""Integrals of y over parts of panels, interpolating ln y by a Lagrange cubic.

    For a panel between nodes :math:`j` and :math:`j+1` of a lattice of spacing
    :math:`\delta` (``step``), with local coordinate :math:`u \in [0, 1]`,
    :math:`L(u)` is the cubic through ln y at the four nodes
    :math:`j + o, \ldots, j + o + 3` (``o`` the ``offset``: -1 inside the lattice, 0 or
    -2 at its lower or upper end), and the result is
    :math:`\delta \int_{u_{\rm lo}}^{u_{\rm hi}} e^{L(u)}\,du`, by Gauss-Legendre
    quadrature. Where a y in the stencil is not finite and > 0 (it underflowed to 0,
    or is negative), y itself is interpolated linearly between the panel's two nodes
    instead.

    Parameters
    ----------
    y
        y at the four nodes of each stencil: the last axis has length 4.
    offset
        The position of each stencil's first node relative to the panel's lower node
        (integers, -2 to 0), broadcast with ``y`` without its last axis.
    u_lo, u_hi
        The ends of the integration in the local coordinate, broadcast likewise.
    step
        The lattice spacing (e.g. in ln m).

    Returns
    -------
    numpy.ndarray
        The integrals, with the broadcast shape (without the stencil axis).
    """
    y = np.asarray(y, dtype=float)
    offset = np.asarray(offset, dtype=np.int64)
    shape = np.broadcast_shapes(y.shape[:-1], offset.shape, np.shape(u_lo), np.shape(u_hi))
    y = np.broadcast_to(y, (*shape, 4))
    offset = np.broadcast_to(offset, shape)
    points = gauss_points(np.broadcast_to(u_lo, shape), np.broadcast_to(u_hi, shape))
    positive = np.all(y > 0, axis=-1) & np.all(np.isfinite(y), axis=-1)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ln_y = np.log(y)
        # lagrange4 has its nodes at -1, 0, 1, 2: the stencil's second node is at 0.
        t = points - (offset + 1)[..., None]
        values = np.exp(lagrange4(t, *(ln_y[..., k, None] for k in range(4))))
    if not np.all(positive):
        lower = np.take_along_axis(y, (-offset)[..., None], axis=-1)
        upper = np.take_along_axis(y, (1 - offset)[..., None], axis=-1)
        linear = lower * (1.0 - points) + upper * points
        values = np.where(positive[..., None], values, linear)
    out: FloatArray = step * gauss_sum(values, u_lo, u_hi)
    return out


def cumulative_from_top(panels: npt.ArrayLike) -> FloatArray:
    """The sums of panels from the top down: the integral from each node to the top.

    Parameters
    ----------
    panels
        The integrals over consecutive panels, the last one ending at the top node,
        along the last axis.

    Returns
    -------
    numpy.ndarray
        One more element along the last axis than ``panels``: element ``i`` is the
        sum of panels ``i`` and above, accumulated from the top panel down (so it is
        the same however many panels lie below it), and the last element, at the top
        node, is 0.
    """
    panels = np.asarray(panels, dtype=float)
    from_top = np.ascontiguousarray(panels[..., ::-1])
    acc = np.cumsum(from_top, axis=-1)[..., ::-1]
    zero = np.zeros((*panels.shape[:-1], 1))
    out: FloatArray = np.concatenate([acc, zero], axis=-1)
    return out
