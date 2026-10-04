"""Model domains and the domain policy.

Each model will carry two domains, as class variables:

``valid_domain``
    Where the model's formula is mathematically defined or sensible (e.g. sigma > 0).
    Outside it, evaluation **always raises**.
``calibration_domain``
    Where the model was calibrated against simulations. Outside it, results are
    extrapolations, handled according to the user's :data:`DomainPolicy`.

This module provides the :class:`Domain` type describing either, and
:func:`apply_domain_policy`, which applies a policy to a result.

Bounds are given in the canonical units of :data:`hmf.core.units.CANONICAL_UNITS`.
A bound on a dimensional variable is a :class:`~astropy.units.Quantity`; the values
checked against it must then be Quantities too.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any, Literal, get_args

import astropy.units as u
import attrs
import numpy as np
import numpy.typing as npt

from ..exceptions import HMFExtrapolationWarning
from .units import UnitBoundaryError

__all__ = [
    "DOMAIN_POLICIES",
    "Domain",
    "DomainError",
    "DomainPolicy",
    "HMFExtrapolationWarning",
    "Interval",
    "apply_domain_policy",
]

#: What to do with results outside a model's calibration domain:
#:
#: ``"ignore"``
#:     return them unchanged (the domain can still be queried);
#: ``"warn"``
#:     return them, and emit an :class:`~hmf.exceptions.HMFExtrapolationWarning`;
#: ``"mask"``
#:     replace them with NaN;
#: ``"raise"``
#:     raise a :class:`DomainError`.
DomainPolicy = Literal["ignore", "warn", "mask", "raise"]

#: The valid values of :data:`DomainPolicy`.
DOMAIN_POLICIES: tuple[str, ...] = get_args(DomainPolicy)


class DomainError(ValueError):
    """Inputs are outside a model's domain, and the policy forbids that."""


def _lower(x: float | None) -> float:
    """Convert a lower bound, with ``None`` meaning unbounded."""
    return -np.inf if x is None else float(x)


def _upper(x: float | None) -> float:
    """Convert an upper bound, with ``None`` meaning unbounded."""
    return np.inf if x is None else float(x)


@attrs.frozen
class Interval:
    """A closed interval ``[lower, upper]`` of one variable.

    Parameters
    ----------
    lower
        The lower bound, inclusive. ``-inf`` (or ``None``) for no lower bound.
    upper
        The upper bound, inclusive. ``inf`` (or ``None``) for no upper bound.
    unit
        The unit of the bounds, for a dimensional variable; ``None`` if the variable
        is dimensionless.
    """

    lower: float = attrs.field(converter=_lower)
    upper: float = attrs.field(converter=_upper)
    unit: u.UnitBase | None = None

    @upper.validator
    def _check_order(self, attribute: attrs.Attribute[float], value: float) -> None:
        if np.isnan(self.lower) or np.isnan(value) or not self.lower <= value:
            raise ValueError(
                f"Interval bounds must satisfy lower <= upper, got [{self.lower}, {value}]."
            )

    def contains(self, x: Any, *, name: str = "value") -> bool | npt.NDArray[np.bool_]:
        """Whether ``x`` lies within the interval (bounds included).

        Parameters
        ----------
        x
            A float or array; a Quantity if the interval has a unit.
        name
            The variable's name, for error messages.

        Returns
        -------
        bool or numpy.ndarray of bool
            False for NaN.
        """
        if self.unit is not None:
            if not isinstance(x, u.Quantity):
                raise UnitBoundaryError(
                    f"Domain: {name!r} must be a Quantity in units convertible to "
                    f"'{self.unit}', not {type(x).__name__}."
                )
            value = x.to_value(self.unit)
        elif isinstance(x, u.Quantity):
            value = x.to_value(u.dimensionless_unscaled)
        else:
            value = np.asarray(x)
        inside = (value >= self.lower) & (value <= self.upper)
        return bool(inside) if np.ndim(inside) == 0 else inside


def _to_interval(bound: Interval | tuple[Any, Any] | u.Quantity) -> Interval:
    """Convert one bound specification to an :class:`Interval`."""
    if isinstance(bound, Interval):
        return bound
    if isinstance(bound, u.Quantity):
        if bound.shape != (2,):
            raise ValueError(f"A Quantity bound must have shape (2,), not {bound.shape}.")
        return Interval(bound[0].value, bound[1].value, bound.unit)
    lower, upper = bound
    if isinstance(lower, u.Quantity) or isinstance(upper, u.Quantity):
        unit = lower.unit if isinstance(lower, u.Quantity) else upper.unit
        return Interval(
            None if lower is None else u.Quantity(lower).to_value(unit),
            None if upper is None else u.Quantity(upper).to_value(unit),
            unit,
        )
    return Interval(lower, upper)


def _to_bounds(
    bounds: Mapping[str, Interval | tuple[Any, Any] | u.Quantity],
) -> tuple[tuple[str, Interval], ...]:
    """Convert a mapping of bounds to the sorted tuple :class:`Domain` stores."""
    return tuple(sorted((str(k), _to_interval(v)) for k, v in dict(bounds).items()))


@attrs.frozen
class Domain:
    """A region of the space of a model's input variables: one interval per variable.

    Variables without a bound are unconstrained.

    Parameters
    ----------
    bounds
        A mapping from variable name (``"z"``, ``"sigma"``, ``"m"``, ``"delta_halo"``,
        ...) to its bounds: an :class:`Interval`, a ``(lower, upper)`` pair (``None``
        for unbounded; Quantities for a dimensional variable), or a Quantity of shape
        ``(2,)``.
    source
        Where the domain comes from (e.g. "Tinker et al. 2008, Sec. 4"), if it is a
        calibration domain.

    Examples
    --------
    >>> from hmf.core.units import Msun_h
    >>> d = Domain({"z": (0, 2.5), "m": [1e10, 1e15] * Msun_h})
    >>> d.contains(z=1.0, m=1e12 * Msun_h)
    True
    >>> d.contains(z=np.array([0.5, 3.0]))
    array([ True, False])
    """

    bounds: tuple[tuple[str, Interval], ...] = attrs.field(converter=_to_bounds)
    source: str = ""

    @property
    def variables(self) -> tuple[str, ...]:
        """The names of the bounded variables."""
        return tuple(name for name, _ in self.bounds)

    def __getitem__(self, name: str) -> Interval:
        """Return the interval of variable ``name``."""
        for key, interval in self.bounds:
            if key == name:
                return interval
        raise KeyError(name)

    def contains(self, **values: Any) -> bool | npt.NDArray[np.bool_]:
        """Whether the given values lie within the domain.

        Parameters
        ----------
        **values
            The value(s) of some of the domain's variables, by name. Arrays broadcast
            against each other. Variables not given are not checked.

        Returns
        -------
        bool or numpy.ndarray of bool
            A bool if every value is a scalar, else the broadcast array.

        Raises
        ------
        ValueError
            If a name is not a variable of the domain (it is most likely a typo).
        """
        unknown = sorted(set(values) - set(self.variables))
        if unknown:
            raise ValueError(
                f"The domain has no variable(s) {unknown}; it bounds {list(self.variables)}."
            )
        inside: bool | npt.NDArray[np.bool_] = True
        for name, value in values.items():
            inside = inside & self[name].contains(value, name=name)
        return inside if np.ndim(inside) else bool(inside)


def apply_domain_policy(
    result: Any,
    inside: bool | npt.NDArray[np.bool_],
    policy: DomainPolicy,
    *,
    description: str = "the model",
) -> Any:
    """Apply a domain policy to a result computed (partly) outside a domain.

    Parameters
    ----------
    result
        The computed values: a float, array or Quantity, broadcastable with ``inside``.
    inside
        Whether each value is inside the domain, e.g. from :meth:`Domain.contains`.
    policy
        What to do with values outside it (see :data:`DomainPolicy`).
    description
        Names the model and domain in messages, e.g. "Tinker08's calibration domain".

    Returns
    -------
    float, numpy.ndarray or Quantity
        ``result`` itself, except with the ``"mask"`` policy, which returns a copy
        with NaN outside the domain.

    Raises
    ------
    DomainError
        With the ``"raise"`` policy, if any value is outside the domain.
    ValueError
        If ``policy`` is not a valid policy.

    Warns
    -----
    HMFExtrapolationWarning
        With the ``"warn"`` policy, if any value is outside the domain. Emitting it
        once per stage instance, rather than once per call, is up to the caller.
    """
    if policy not in DOMAIN_POLICIES:
        raise ValueError(f"Unknown domain policy {policy!r}; use one of {DOMAIN_POLICIES}.")
    if policy == "ignore" or np.all(inside):
        return result
    n_out = int(np.size(inside) - np.count_nonzero(inside))
    message = f"{n_out} of {np.size(inside)} value(s) are outside {description}."
    if policy == "raise":
        raise DomainError(message)
    if policy == "warn":
        warnings.warn(message + " They are extrapolations.", HMFExtrapolationWarning, stacklevel=2)
        return result
    # "mask"
    if isinstance(result, u.Quantity):
        return np.where(inside, result.value, np.nan) << result.unit
    masked = np.where(inside, result, np.nan)
    return masked if np.ndim(masked) else float(masked)
