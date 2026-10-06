"""Converters of array-like inputs, shared across :mod:`hmf.core`.

The pure-numpy ones live in :mod:`hmf.core._kernels.arrays` (so the kernels can use
them) and are re-exported here; this module adds the ones that accept Quantities.
"""

from __future__ import annotations

from typing import Any

import astropy.units as u
import numpy as np

from ._kernels.arrays import float_array, optional_float_array, read_only

__all__ = ["dimensionless_floats", "float_array", "optional_float_array", "read_only"]


def dimensionless_floats(value: Any) -> tuple[float, ...]:
    """Convert an array of dimensionless numbers to a tuple of floats.

    A dimensionless :class:`~astropy.units.Quantity` is converted to plain numbers (so
    e.g. a percentage is scaled); a Quantity with a dimension raises, rather than
    having its unit silently dropped.

    Parameters
    ----------
    value
        A number, an array of numbers, or a dimensionless Quantity.

    Returns
    -------
    tuple of float
        The values, as a 1D tuple.

    Raises
    ------
    astropy.units.UnitConversionError
        If ``value`` is a Quantity that is not dimensionless.
    """
    if isinstance(value, u.Quantity):
        value = value.to_value(u.dimensionless_unscaled)
    return tuple(float(x) for x in np.atleast_1d(np.asarray(value, dtype=float)))
