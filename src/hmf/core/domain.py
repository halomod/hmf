"""Model domains, the domain policy, and the errors of :mod:`hmf.core`.

Each model carries two domains, as class variables:

``valid_domain``
    Where the model's formula is mathematically defined or sensible (e.g. sigma > 0).
    Outside it, evaluation **always raises** a :class:`DomainError`.
``calibration_domain``
    Where the model was calibrated against simulations, or ``None`` if it has none.
    Outside it, results are extrapolations, handled according to the user's
    :data:`DomainPolicy`.

This module provides the :class:`Domain` type describing either (with
:meth:`Domain.describe` and :meth:`Domain.check`), :func:`apply_domain_policy`,
which applies a policy to a result, :func:`warn_once`, which emits a warning once
per object, and :func:`check_extent`, the cheap check of a kernel-level entry point's
input against a domain (see :mod:`hmf.core._kernels`).

Bounds are given in the canonical units of :data:`hmf.core.units.CANONICAL_UNITS`.
A bound on a dimensional variable is a :class:`~astropy.units.Quantity`; the values
checked against it must then be Quantities too. Each bound is closed (``>=``,
``<=``) or open (``>``, ``<``).

Errors
------
Every error :mod:`hmf.core` raises about its inputs is one of three types:

:class:`DomainError` (a :class:`ValueError`)
    An input *value* is outside what can be evaluated: a non-positive or NaN k, m
    or sigma; a mass outside the lattice with ``extension="raise"``; a k outside a
    table that is not extrapolated; a z beyond a growth table; anything outside a
    model's ``valid_domain``.
:class:`ValueError`
    A bad option, configuration or combination of them, or a missing input. This
    includes a model that does not apply to the cosmology it is given (e.g.
    ``Eisenstein97Growth`` with a non-flat cosmology).
:class:`TypeError`
    An object of the wrong kind (e.g. a model that is not a model).

Values that are outside a *calibration* domain, or that are extrapolated beyond a
table the user supplied, are not errors: they emit an
:class:`~hmf.exceptions.HMFExtrapolationWarning` (once per object, with
:func:`warn_once`), or follow the :data:`DomainPolicy`.
"""

from __future__ import annotations

import math
import warnings
import weakref
from collections.abc import Mapping
from functools import cached_property
from types import MappingProxyType
from typing import Any, Literal, get_args

import astropy.units as u
import attrs
import numpy as np
import numpy.typing as npt

from ..exceptions import HMFExtrapolationWarning
from .units import _require_quantity, to_canonical

__all__ = [
    "DOMAIN_POLICIES",
    "Domain",
    "DomainError",
    "DomainPolicy",
    "HMFExtrapolationWarning",
    "Interval",
    "apply_domain_policy",
    "check_extent",
    "warn_once",
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
    """An input value is outside what can be evaluated (see "Errors" above).

    E.g. outside a model's valid domain, outside a table that is not extrapolated,
    or outside a calibration domain with the ``"raise"`` policy.
    """


def _lower(x: float | None) -> float:
    """Convert a lower bound, with ``None`` meaning unbounded."""
    return -np.inf if x is None else float(x)


def _upper(x: float | None) -> float:
    """Convert an upper bound, with ``None`` meaning unbounded."""
    return np.inf if x is None else float(x)


def _format_bound(x: float, unit: u.UnitBase | None) -> str:
    """A bound for a message: the number, with its unit if it has one."""
    return f"{x:g}" if unit is None else f"{x:g} {unit.to_string()}"


@attrs.frozen
class Interval:
    """An interval of one variable, each of whose bounds is closed or open.

    Parameters
    ----------
    lower
        The lower bound. ``-inf`` (or ``None``) for no lower bound.
    upper
        The upper bound. ``inf`` (or ``None``) for no upper bound.
    unit
        The unit of the bounds, for a dimensional variable; ``None`` if the variable
        is dimensionless.
    lower_open, upper_open
        Whether each bound is excluded (``>`` and ``<``) rather than included (``>=``
        and ``<=``, the default).

    Examples
    --------
    >>> Interval(0, None, lower_open=True).describe("sigma")
    'sigma > 0'
    >>> Interval(75, 3e4, lower_open=True).describe("delta_halo")
    '75 < delta_halo <= 30000'
    """

    lower: float = attrs.field(converter=_lower)
    upper: float = attrs.field(converter=_upper)
    unit: u.UnitBase | None = None
    lower_open: bool = attrs.field(
        default=False, kw_only=True, validator=attrs.validators.instance_of(bool)
    )
    upper_open: bool = attrs.field(
        default=False, kw_only=True, validator=attrs.validators.instance_of(bool)
    )

    @upper.validator
    def _check_order(self, attribute: attrs.Attribute[float], value: float) -> None:
        if np.isnan(self.lower) or np.isnan(value) or not self.lower <= value:
            raise ValueError(
                f"Interval bounds must satisfy lower <= upper, got [{self.lower}, {value}]."
            )

    def __attrs_post_init__(self) -> None:
        """Check that the interval is not empty."""
        if self.lower == self.upper and (self.lower_open or self.upper_open):
            raise ValueError(
                f"The interval with equal bounds {self.lower} and an open end is empty."
            )

    def describe(self, name: str = "x") -> str:
        """The interval as an inequality on the variable ``name``.

        Parameters
        ----------
        name
            The variable's name.

        Returns
        -------
        str
            E.g. ``"sigma > 0"``, ``"75 < delta_halo <= 30000"``, or ``"any z"`` for no
            bounds. Bounds are followed by the unit, if there is one.
        """
        lo, hi = np.isfinite(self.lower), np.isfinite(self.upper)
        lo_text = _format_bound(self.lower, self.unit)
        hi_text = _format_bound(self.upper, self.unit)
        if lo and hi:
            return (
                f"{lo_text} {'<' if self.lower_open else '<='} {name} "
                f"{'<' if self.upper_open else '<='} {hi_text}"
            )
        if lo:
            return f"{name} {'>' if self.lower_open else '>='} {lo_text}"
        if hi:
            return f"{name} {'<' if self.upper_open else '<='} {hi_text}"
        return f"any {name}"

    def contains(self, x: Any, *, name: str = "value") -> bool | npt.NDArray[np.bool_]:
        """Whether ``x`` lies within the interval (closed bounds included).

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

        Raises
        ------
        hmf.core.units.UnitBoundaryError
            If the interval has a unit and ``x`` is not a Quantity.
        astropy.units.UnitConversionError
            If ``x`` can't be converted to the interval's unit without an H0.
        """
        if self.unit is not None:
            if isinstance(x, u.Quantity) and x.unit is self.unit:
                value = x.value  # already in the canonical unit (the common case)
            else:
                # Bounds are in canonical units, and a domain has no H0: h-units only.
                _require_quantity(x, self.unit, where="Domain", name=name, has_h0=False)
                value = to_canonical(x, self.unit, where="Domain", name=name)
        elif isinstance(x, u.Quantity):
            value = x.to_value(u.dimensionless_unscaled)
        else:
            value = x
        if isinstance(value, (float, int)):  # a scalar (np.float64 too): plain Python
            return bool(
                (value > self.lower if self.lower_open else value >= self.lower)
                and (value < self.upper if self.upper_open else value <= self.upper)
            )
        value = np.asarray(value)
        above = (value > self.lower) if self.lower_open else (value >= self.lower)
        below = (value < self.upper) if self.upper_open else (value <= self.upper)
        inside = above & below
        return bool(inside) if np.ndim(inside) == 0 else inside


#: The interval notations of the 3-tuple shorthand, and whether each end is open.
_BRACKETS: Mapping[str, tuple[bool, bool]] = {
    "[]": (False, False),
    "(]": (True, False),
    "[)": (False, True),
    "()": (True, True),
}


def _to_interval(bound: Interval | tuple[Any, ...] | u.Quantity) -> Interval:
    """Convert one bound specification to an :class:`Interval`."""
    if isinstance(bound, Interval):
        return bound
    if isinstance(bound, u.Quantity):
        if bound.shape != (2,):
            raise ValueError(f"A Quantity bound must have shape (2,), not {bound.shape}.")
        return Interval(bound[0].value, bound[1].value, bound.unit)
    if len(bound) == 3:
        lower, upper, brackets = bound
        if brackets not in _BRACKETS:
            raise ValueError(
                f"The third item of a bound must be one of {list(_BRACKETS)}, not {brackets!r}."
            )
        lower_open, upper_open = _BRACKETS[brackets]
    else:
        lower, upper = bound
        lower_open = upper_open = False
    unit = None
    if isinstance(lower, u.Quantity) or isinstance(upper, u.Quantity):
        unit = lower.unit if isinstance(lower, u.Quantity) else upper.unit
        lower = None if lower is None else u.Quantity(lower).to_value(unit)
        upper = None if upper is None else u.Quantity(upper).to_value(unit)
    return Interval(lower, upper, unit, lower_open=lower_open, upper_open=upper_open)


def _to_bounds(
    bounds: Mapping[str, Interval | tuple[Any, ...] | u.Quantity],
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
        ...) to its bounds: an :class:`Interval`; a ``(lower, upper)`` pair (``None``
        for unbounded; Quantities for a dimensional variable), whose bounds are
        closed; a ``(lower, upper, brackets)`` triple, with ``brackets`` the interval
        notation ``"[]"``, ``"(]"``, ``"[)"`` or ``"()"`` (``"("`` and ``")"`` mark
        an open, excluded, bound: ``(0, None, "(]")`` is ``> 0``); or a Quantity of
        shape ``(2,)``, whose bounds are closed.
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
    >>> Domain({"sigma": (0, None, "(]"), "z": (0, 2.5)}).describe()
    'sigma > 0, 0 <= z <= 2.5'
    """

    bounds: tuple[tuple[str, Interval], ...] = attrs.field(converter=_to_bounds)
    source: str = ""

    @cached_property
    def _intervals(self) -> Mapping[str, Interval]:
        """The intervals by variable name (for fast lookups)."""
        return MappingProxyType(dict(self.bounds))

    @property
    def variables(self) -> tuple[str, ...]:
        """The names of the bounded variables."""
        return tuple(name for name, _ in self.bounds)

    def __getitem__(self, name: str) -> Interval:
        """Return the interval of variable ``name``."""
        return self._intervals[name]

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
        intervals = self._intervals
        unknown = sorted(name for name in values if name not in intervals)
        if unknown:
            raise ValueError(
                f"The domain has no variable(s) {unknown}; it bounds {list(self.variables)}."
            )
        inside: bool | npt.NDArray[np.bool_] = True
        for name, value in values.items():
            inside = inside & intervals[name].contains(value, name=name)
        return inside if np.ndim(inside) else bool(inside)

    def describe(self) -> str:
        """The domain as inequalities on its variables, e.g. ``"sigma > 0, z >= 0"``.

        Returns
        -------
        str
            One inequality per variable (see :meth:`Interval.describe`), or
            ``"unbounded"`` for a domain without variables.
        """
        return ", ".join(interval.describe(name) for name, interval in self.bounds) or "unbounded"

    def check(self, values: Mapping[str, Any], *, where: str) -> None:
        """Raise a :class:`DomainError` if any of the values is outside the domain.

        Parameters
        ----------
        values
            The value(s) of some of the domain's variables, by name, as for
            :meth:`contains`.
        where
            Who checks them (e.g. ``"Tinker08's valid domain"``), for the message.

        Raises
        ------
        DomainError
            If any value is outside the domain (NaN included). The message gives
            the variables out of range and the domain's :meth:`describe`.
        ValueError
            If a name is not a variable of the domain.
        """
        inside = self.contains(**values)
        if inside is True or (inside is not False and np.all(inside)):
            return
        bad = [name for name, value in values.items() if not np.all(self.contains(**{name: value}))]
        n_out = int(np.size(inside) - np.count_nonzero(inside))
        raise DomainError(
            f"{where}: {n_out} of {np.size(inside)} value(s) are outside the domain "
            f"({self.describe()}); out of range: {', '.join(bad)}."
        )


def check_extent(
    name: str,
    x: npt.ArrayLike,
    bounds: Interval | Domain | None = None,
    *,
    where: str,
    ln_values: bool = False,
) -> npt.NDArray[np.float64]:
    """Check that plain values are finite and inside an interval, from their extent.

    The check of a kernel-level entry point (see :mod:`hmf.core._kernels`): it
    compares only the smallest and largest value with the interval's bounds, one
    pass over the array for each, so it is cheap enough for every call. NaN
    propagates to the extent, so it raises too.

    Parameters
    ----------
    name
        The variable's name, for the message, and the variable a :class:`Domain`
        bounds (without the ``ln_`` prefix, with ``ln_values``).
    x
        The values: a float or a plain array, in the canonical unit of the interval.
    bounds
        The bounds, in canonical units (a bound's unit, if it has one, is not
        compared): an :class:`Interval`, or a :class:`Domain`, whose interval of
        ``name`` is used if it bounds it. ``None`` checks only that the values are
        finite.
    where
        Who checks them (e.g. ``"Growth.growth_factor_kernel (FromArray)"``), for the
        message.
    ln_values
        Whether ``x`` holds the natural log of the variable (e.g. ln k for a bound
        on k): the extent is exponentiated before it is compared with the bounds.

    Returns
    -------
    numpy.ndarray
        ``x`` as a float array (not a copy, if it already is one).

    Raises
    ------
    DomainError
        If any value is not finite, or is outside the interval.
    """
    values = np.asarray(x, dtype=np.float64)
    if values.ndim == 0:
        lo = hi = float(values)
    elif values.size == 0:
        return values
    else:
        lo, hi = float(values.min()), float(values.max())
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise DomainError(f"{where}: {name} must be finite, got values in [{lo:g}, {hi:g}].")
    var = name.removeprefix("ln_") if ln_values else name
    interval = bounds._intervals.get(var) if isinstance(bounds, Domain) else bounds
    if interval is None:
        return values
    if ln_values:
        lo, hi = math.exp(lo), math.exp(hi)
    above = lo > interval.lower if interval.lower_open else lo >= interval.lower
    below = hi < interval.upper if interval.upper_open else hi <= interval.upper
    if not (above and below):
        raise DomainError(
            f"{where}: {var} in [{lo:g}, {hi:g}] is outside the domain ({interval.describe(var)})."
        )
    return values


#: The keys of the warnings already emitted, by the id of the object they are about.
#: An entry is removed when its object is garbage collected (so ids can't be reused).
_WARNED: dict[int, set[str]] = {}


def warn_once(
    owner: object,
    key: str | tuple[str, ...],
    message: str,
    category: type[Warning] = HMFExtrapolationWarning,
    *,
    stacklevel: int = 2,
) -> bool:
    """Emit a warning once per object and key.

    The objects of :mod:`hmf.core` are frozen and compare by value, so this keeps the
    keys already warned about by the object's *identity*, outside the object: two
    equal stages each warn once.

    Parameters
    ----------
    owner
        The object the warning is about (e.g. a stage). If it can not be weakly
        referenced, the warning is emitted every time.
    key
        What the warning is about, e.g. ``"k above the table"``. Each key warns once.
        A tuple of keys warns if any of them has not warned yet, and then counts as
        a warning about each (one message about several things).
    message
        The warning's message.
    category
        The warning's class.
    stacklevel
        As for :func:`warnings.warn`, from the caller of this function.

    Returns
    -------
    bool
        Whether the warning was emitted (False if it was already, for this object
        and key).
    """
    try:
        weakref.ref(owner)
    except TypeError:
        warnings.warn(message, category, stacklevel=stacklevel + 1)
        return True
    ident = id(owner)
    done = _WARNED.get(ident)
    if done is None:
        done = _WARNED[ident] = set()
        weakref.finalize(owner, _WARNED.pop, ident, None)
    keys = (key,) if isinstance(key, str) else key
    if done.issuperset(keys):
        return False
    done.update(keys)
    warnings.warn(message, category, stacklevel=stacklevel + 1)
    return True


def apply_domain_policy(
    result: Any,
    inside: bool | npt.NDArray[np.bool_],
    policy: DomainPolicy,
    *,
    description: str = "the model",
    owner: object | None = None,
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
    owner
        The object (e.g. a stage) the ``"warn"`` policy warns once for, per
        ``description`` (see :func:`warn_once`). If ``None``, it warns on every call.

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
        With the ``"warn"`` policy, if any value is outside the domain: once per
        ``owner`` (and ``description``), or on every call without an ``owner``.
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
        message += " They are extrapolations."
        if owner is None:
            warnings.warn(message, HMFExtrapolationWarning, stacklevel=2)
        else:
            warn_once(owner, description, message, stacklevel=2)
        return result
    # "mask"
    if isinstance(result, u.Quantity):
        return np.where(inside, result.value, np.nan) << result.unit
    masked = np.where(inside, result, np.nan)
    return masked if np.ndim(masked) else float(masked)
