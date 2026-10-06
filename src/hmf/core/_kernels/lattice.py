"""Lattices: grids whose nodes sit at integer multiples of a fixed step.

Grids built as ``i * step`` for integer ``i`` share their nodes exactly whenever they
share their step, whatever their range. That is what makes the lazily extended grids
of :mod:`hmf.core` bit-for-bit deterministic. Every lattice in :mod:`hmf.core` is
built here, with one rounding rule for its ends (:func:`lattice_index`).
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

__all__ = ["LATTICE_RTOL", "lattice", "lattice_index"]

#: How close, relative to the step, a position must be to a node to count as on it.
#: Positions computed in floating point (e.g. ``log10(m_min) / dlog10_m``) are rarely
#: exact integers even when they are meant to be.
LATTICE_RTOL = 1e-9


def lattice_index(x: float, step: float, *, up: bool) -> int:
    """The index of the lattice node at or beyond ``x``.

    Parameters
    ----------
    x
        A position, in the same unit as ``step``.
    step
        The spacing of the lattice, > 0.
    up
        If True, the first node at or above ``x``; else the last node at or below it.

    Returns
    -------
    int
        The node's index ``i``, i.e. the node is at ``i * step``. A position within
        :data:`LATTICE_RTOL` steps of a node is on it, so rounding errors in ``x`` do
        not add or drop a node.
    """
    t = x / step
    nearest = round(t)
    if abs(t - nearest) < LATTICE_RTOL:
        return int(nearest)
    return math.ceil(t) if up else math.floor(t)


def lattice(i_lo: int, i_hi: int, step: float) -> npt.NDArray[np.float64]:
    """The positions of the lattice nodes ``i_lo`` to ``i_hi`` (both included).

    Parameters
    ----------
    i_lo, i_hi
        The indices of the first and last nodes.
    step
        The spacing of the lattice.

    Returns
    -------
    numpy.ndarray
        ``i * step`` for each integer ``i`` from ``i_lo`` to ``i_hi``, computed as one
        product per node, so the same node has the same value in every lattice.
    """
    out: npt.NDArray[np.float64] = np.arange(i_lo, i_hi + 1, dtype=np.int64) * np.float64(step)
    return out
