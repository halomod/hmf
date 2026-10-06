"""Shared validators: ``attrs`` validators for fields, and checkers for arrays.

The field validators raise a :class:`ValueError` naming the class and the field, e.g.
``"SMT.a must be > 0, got 0.0."``. Use them as ``field(validator=positive, ...)``.

The array checkers (``check_*``) validate the arrays a method or a model is given,
and return them as float arrays. Each raises ``error`` (a :class:`ValueError` by
default; e.g. :class:`~hmf.core.domain.DomainError` for inputs outside a model's
domain) with a message of the form ``"{where}: {name} must be ..."``. They are pure
numpy, so kernels may use them too.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import attrs
import numpy as np
import numpy.typing as npt

__all__ = [
    "check_finite_positive",
    "check_in_range",
    "check_increasing",
    "check_table",
    "finite",
    "less_than",
    "positive",
]

Validator = Callable[[Any, "attrs.Attribute[Any]", Any], None]

FloatArray = npt.NDArray[np.float64]


def positive(instance: Any, attribute: attrs.Attribute[float], value: float) -> None:
    """Validate that a value is finite and strictly positive."""
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{type(instance).__name__}.{attribute.name} must be > 0, got {value}.")


def finite(instance: Any, attribute: attrs.Attribute[float], value: float) -> None:
    """Validate that a value is finite."""
    if not math.isfinite(value):
        raise ValueError(f"{type(instance).__name__}.{attribute.name} must be finite, got {value}.")


def less_than(bound: float) -> Validator:
    """Return a validator that a value is strictly less than ``bound``.

    Parameters
    ----------
    bound
        The exclusive upper bound.

    Returns
    -------
    callable
        The validator.
    """

    def check(instance: Any, attribute: attrs.Attribute[float], value: float) -> None:
        if not value < bound:
            raise ValueError(
                f"{type(instance).__name__}.{attribute.name} must be < {bound:g}, got {value}."
            )

    return check


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

    Infinite values pass if the bounds allow them: with no ``high``, ``inf`` is in
    range.

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
        ``error``, if a value is out of range or NaN.
    """
    arr = np.asarray(value, dtype=float)
    ok = ~np.isnan(arr)
    requirement = []
    if low is not None:
        ok &= (arr > low) if low_open else (arr >= low)
        requirement.append(f"{'>' if low_open else '>='} {low:g}")
    if high is not None:
        ok &= (arr < high) if high_open else (arr <= high)
        requirement.append(f"{'<' if high_open else '<='} {high:g}")
    what = " and ".join(requirement) or "a number"
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
