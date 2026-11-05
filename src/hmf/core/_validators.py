"""Shared validators: ``attrs`` validators for fields, and checkers for arrays.

The field validators raise a :class:`ValueError` naming the class and the field, e.g.
``"SMT.a must be > 0, got 0.0."``. Use them as ``field(validator=positive, ...)``.

The array checkers (``check_*``) validate the arrays a method or a model is given,
and return them as float arrays. Each raises ``error`` (a :class:`ValueError` by
default; e.g. :class:`~hmf.core.domain.DomainError` for inputs outside a model's
domain) with a message of the form ``"{where}: {name} must be ..."``. They are pure
numpy, and are defined in :mod:`hmf.core._kernels.arrays`, so kernels use them from
there; this module re-exports them for the rest of :mod:`hmf.core`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import attrs

from ._kernels.arrays import check_finite_positive, check_in_range, check_increasing, check_table

__all__ = [
    "check_finite_positive",
    "check_in_range",
    "check_increasing",
    "check_table",
    "finite",
    "less_than",
    "one_of",
    "positive",
]

Validator = Callable[[Any, "attrs.Attribute[Any]", Any], None]


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


def one_of(options: tuple[Any, ...]) -> Validator:
    """Return a validator that a value is one of ``options`` (as ``attrs.validators.in_``).

    Parameters
    ----------
    options
        The valid values.

    Returns
    -------
    callable
        The validator.
    """

    def check(instance: Any, attribute: attrs.Attribute[Any], value: Any) -> None:
        if value not in options:
            raise ValueError(
                f"{type(instance).__name__}.{attribute.name} must be one of "
                f"{', '.join(map(repr, options))}; got {value!r}."
            )

    return check
