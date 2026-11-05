"""Conversions and checks of plain arrays shared by the kernels and the rest of :mod:`hmf.core`.

They are pure numpy: no Quantities (see :mod:`hmf.core._arrays` for the converters
that accept them).

The checkers (``check_*``) validate an array and return it as a float array. Each
raises ``error`` (a :class:`ValueError` by default; e.g.
:class:`~hmf.core.domain.DomainError` for inputs outside a model's domain) with a
message of the form ``"{where}: {name} must be ..."``.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

__all__ = [
    "check_finite_positive",
    "check_in_range",
    "check_increasing",
    "check_table",
    "float_array",
    "optional_float_array",
    "read_only",
]

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


def _raise_if_any(
    bad: npt.NDArray[np.bool_], requirement: str, failure: str, error: type[Exception]
) -> None:
    """Raise ``error`` if any of ``bad`` is True, saying how many fail."""
    if np.any(bad):
        raise error(
            f"{requirement}, but {int(np.count_nonzero(bad))} of {np.size(bad)} value(s) {failure}."
        )


def check_finite_positive(
    name: str, value: npt.ArrayLike, *, where: str, error: type[Exception] = ValueError
) -> FloatArray:
    """Check that every value of an array is finite and strictly positive.

    Parameters
    ----------
    name
        The name of the array, for the message.
    value
        The array (or scalar).
    where
        Who checks it (a class, method or model name), for the message.
    error
        The exception to raise.

    Returns
    -------
    numpy.ndarray
        ``value`` as a float array.

    Raises
    ------
    Exception
        ``error``, if a value is not finite or is <= 0 (NaN included).
    """
    arr = np.asarray(value, dtype=float)
    _raise_if_any(
        ~(np.isfinite(arr) & (arr > 0)),
        f"{where}: {name} must be finite and > 0",
        "are non-positive or not finite",
        error,
    )
    return arr


def check_in_range(
    name: str,
    value: npt.ArrayLike,
    *,
    where: str,
    low: float | None = None,
    high: float | None = None,
    low_open: bool = False,
    high_open: bool = False,
    error: type[Exception] = ValueError,
) -> FloatArray:
    """Check that every value of an array is within bounds (NaN never is).

    An unbounded end is open: with no ``low``, ``-inf`` is out of range, and with no
    ``high``, ``inf`` is.

    Parameters
    ----------
    name
        The name of the array, for the message.
    value
        The array (or scalar).
    where
        Who checks it, for the message.
    low, high
        The bounds, or ``None`` for none.
    low_open, high_open
        Whether each bound is excluded (``>`` and ``<``) rather than included.
    error
        The exception to raise.

    Returns
    -------
    numpy.ndarray
        ``value`` as a float array.

    Raises
    ------
    Exception
        ``error``, if a value is out of range, NaN, or infinite at an unbounded end.
    """
    arr = np.asarray(value, dtype=float)
    ok = ~np.isnan(arr)
    requirement = []
    if low is None or high is None:
        requirement.append("finite")
    if low is None:
        ok &= arr > -np.inf
    else:
        ok &= (arr > low) if low_open else (arr >= low)
        requirement.append(f"{'>' if low_open else '>='} {low:g}")
    if high is None:
        ok &= arr < np.inf
    else:
        ok &= (arr < high) if high_open else (arr <= high)
        requirement.append(f"{'<' if high_open else '<='} {high:g}")
    what = " and ".join(requirement)
    _raise_if_any(~ok, f"{where}: {name} must be {what}", "are out of range or NaN", error)
    return arr


def check_increasing(
    name: str, value: npt.ArrayLike, *, where: str, error: type[Exception] = ValueError
) -> FloatArray:
    """Check that a 1D array is strictly increasing.

    Parameters
    ----------
    name
        The name of the array, for the message.
    value
        The array.
    where
        Who checks it, for the message.
    error
        The exception to raise.

    Returns
    -------
    numpy.ndarray
        ``value`` as a float array.

    Raises
    ------
    Exception
        ``error``, if a value is not greater than the one before it (NaN included).
    """
    arr = np.asarray(value, dtype=float)
    _raise_if_any(
        ~(np.diff(arr) > 0),
        f"{where}: {name} must be strictly increasing",
        "are not greater than the one before",
        error,
    )
    return arr


def check_table(
    columns: Mapping[str, npt.ArrayLike],
    *,
    where: str,
    min_size: int = 1,
    error: type[Exception] = ValueError,
) -> tuple[FloatArray, ...]:
    """Check that the columns of a table are 1D, of the same length, and long enough.

    Parameters
    ----------
    columns
        The columns, by name.
    where
        Who checks it, for the message.
    min_size
        The least number of rows.
    error
        The exception to raise.

    Returns
    -------
    tuple of numpy.ndarray
        The columns as float arrays, in order.

    Raises
    ------
    Exception
        ``error``, if a column is not 1D, the lengths differ, or there are fewer than
        ``min_size`` rows.
    """
    arrays = tuple(np.asarray(c, dtype=float) for c in columns.values())
    shapes = [a.shape for a in arrays]
    if any(len(s) != 1 for s in shapes) or len(set(shapes)) != 1 or shapes[0][0] < min_size:
        raise error(
            f"{where}: {' and '.join(columns)} must be 1D, of the same length, with at "
            f"least {min_size} values (>= {min_size}); got shapes "
            f"{', '.join(str(s) for s in shapes)}."
        )
    return arrays
