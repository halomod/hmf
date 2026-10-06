"""Shared ``attrs`` validators for the fields of models, stages and settings.

Each validator raises a :class:`ValueError` naming the class and the field, e.g.
``"SMT.a must be > 0, got 0.0."``. Use them as ``field(validator=positive, ...)``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import attrs

__all__ = ["finite", "less_than", "positive"]

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
