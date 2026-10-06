"""Conversions of plain arrays shared by the kernels and the rest of :mod:`hmf.core`.

They are pure numpy: no Quantities (see :mod:`hmf.core._arrays` for the converters
that accept them).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["float_array", "optional_float_array", "read_only"]

FloatArray = npt.NDArray[np.float64]


def float_array(x: npt.ArrayLike) -> FloatArray:
    """Convert to a float array (a no-op for a float array).

    Parameters
    ----------
    x
        An array, or anything numpy can convert to one.

    Returns
    -------
    numpy.ndarray
        ``x`` as float64, not copied if it already is.
    """
    return np.asarray(x, dtype=np.float64)


def optional_float_array(x: npt.ArrayLike | None) -> FloatArray | None:
    """Convert to a float array, leaving ``None`` as it is.

    Parameters
    ----------
    x
        An array, or ``None``.

    Returns
    -------
    numpy.ndarray or None
        ``x`` as float64, or ``None``.
    """
    return None if x is None else float_array(x)


def read_only(x: npt.ArrayLike) -> FloatArray:
    """Return a read-only float copy of an array.

    The copy keeps the array's subclass, if any, so it can't be changed through the
    original or through itself.

    Parameters
    ----------
    x
        An array, or anything numpy can convert to one.

    Returns
    -------
    numpy.ndarray
        A new float64 array, not writeable.
    """
    out: FloatArray = np.array(x, dtype=np.float64, subok=True)
    out.flags.writeable = False
    return out
