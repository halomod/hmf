"""The units boundary of the v4 core.

hmf.core follows one units policy ("option B" of issue #389):

* **Quantities at every public boundary.** Every dimensional field of a stage or
  model, dimensional argument of a public method, and dimensional output is an
  :class:`astropy.units.Quantity`. Dimensionless ones are plain numbers and arrays.
* **Unit-free kernels inside.** The numerics in :mod:`hmf.core._kernels` work on plain
  arrays in the canonical units of :data:`CANONICAL_UNITS`.
* **h-units are explicit**, through :data:`littleh`
  (:data:`astropy.cosmology.units.littleh`). Outputs are in h-units. Inputs may be in
  h-units or in physical units; physical units are converted with the H0 of the
  object whose method is called.

The :func:`unit_boundary` decorator is the bridge between the two. It strips the
units off dimensional arguments, converting them to the canonical unit first if
needed, and puts the canonical unit on the outputs. Its fixed cost is budgeted at
2 µs per call (``benchmarks/test_core_units.py`` measures it), so it uses
:class:`~astropy.units.Quantity` views and per-instance cached conversion factors,
not :func:`astropy.units.quantity_input`.

Rules
-----
* **Use the constants in this module** (:data:`Msun_h`, :data:`Mpc_h`, ...) whenever you
  attach a unit. A freshly built ``u.Msun / cu.littleh`` *equals* :data:`Msun_h`, but
  it is a different object, so it misses the boundary's identity fast path.
* **A bare float or ndarray for a dimensional argument is an error**
  (:class:`UnitBoundaryError`). hmf never guesses units.
* **Dimensionless arguments** (z, sigma, peak height, delta_c, Omega, ...) are plain floats or
  arrays, and pass through the boundary unchanged.
* **One conversion.** Every conversion of a dimensional value to its canonical unit
  goes through :func:`unit_boundary` or, for code that converts a value itself,
  :func:`to_canonical`; both raise the same errors, with the same messages. Errors
  name the class of the object (not the class that defines the method) and the
  argument.
* **H0.** Physical units are converted with the H0 of a :class:`UnitContext`, which
  must be a scalar Quantity in km/s/Mpc. An object with a cosmology provides one (see
  :class:`HasUnitContext`). An object without one (a model, a domain) accepts h-units
  only. A model method that needs h, or physical units converted, takes an ``H0``
  argument and builds its context per call.
* **Unit-carrying fields** of models are declared with :func:`quantity_field`: they
  take a Quantity, are stored as a Quantity in the canonical unit (so ``evolve()``
  round-trips them), and compare and hash by their value in it. Kernels read their
  plain values through a private cached property.
* **The scalar rule.** A scalar input gives a scalar output: a 0-d Quantity for a
  dimensional output, a numpy scalar (:class:`numpy.float64`, which is also a
  :class:`float`) for a dimensionless one. Arrays keep their shape. Every public method
  is decorated with :func:`unit_boundary` (with no arguments if it has no dimensional
  ones), which applies the rule. Dimensionless outputs are plain arrays, never
  dimensionless Quantities.
* **Never enable** :func:`~astropy.cosmology.units.with_H0` **globally** (with
  :func:`astropy.units.set_enabled_equivalencies`): inside it, h-scaled and physical
  values convert into each other silently. hmf only ever passes it explicitly, for
  one conversion at a time.
* **Take logarithms of values in an explicit unit**: ``np.log10(m / Msun_h)``, never
  ``np.log10(m.value)`` (whose unit is whatever the caller used).
"""

from __future__ import annotations

import functools
import inspect
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from types import MappingProxyType
from typing import Any, Concatenate, ParamSpec, Protocol, TypeVar

import astropy.constants as _constants
import astropy.cosmology.units as cu
import astropy.units as u
import attrs
import numpy as np

from ._fields import field
from ._kernels.arrays import read_only
from ._serialise import quantity_key

__all__ = [
    "CANONICAL_UNITS",
    "RHO_CRIT0_H2",
    "H0_unit",
    "HasUnitContext",
    "Mpc_h",
    "Msun_h",
    "UnitBoundaryError",
    "UnitContext",
    "dndm_unit",
    "h_Mpc",
    "kpc_h",
    "littleh",
    "littleh_power",
    "littleh_units_enabled",
    "number_density_unit",
    "parse_unit",
    "power_unit",
    "quantity_field",
    "rho_unit",
    "to_canonical",
    "unit_boundary",
]

# ---------------------------------------------------------------------------------
# Shared unit constants. Always use these objects, so that the boundary's identity
# fast path (``x.unit is canonical``) is taken.
# ---------------------------------------------------------------------------------

#: The dimensionless Hubble parameter, h = H0 / (100 km/s/Mpc), as a unit.
littleh = cu.littleh

#: Mass: M☉/h.
Msun_h = u.Msun / littleh

#: Comoving length: Mpc/h.
Mpc_h = u.Mpc / littleh

#: Comoving length: kpc/h (not canonical: e.g. for softening lengths).
kpc_h = u.kpc / littleh

#: Comoving wavenumber: h/Mpc.
h_Mpc = littleh / u.Mpc  # noqa: N816

#: Comoving number density: h³ Mpc⁻³.
number_density_unit = littleh**3 / u.Mpc**3

#: Mass function dn/dM: h⁴ M☉⁻¹ Mpc⁻³.
dndm_unit = littleh**4 / (u.Msun * u.Mpc**3)

#: Power spectrum: (Mpc/h)³.
power_unit = (u.Mpc / littleh) ** 3

#: Comoving mass density: M☉ h² Mpc⁻³.
rho_unit = u.Msun * littleh**2 / u.Mpc**3

#: The Hubble constant: km/s/Mpc.
H0_unit = u.km / u.s / u.Mpc

#: The canonical unit of each physical kind. Kernels in :mod:`hmf.core._kernels` take
#: and return plain arrays in these units, and :func:`unit_boundary` converts to and
#: from them. The table in the :mod:`hmf.core._kernels` docstring mirrors this
#: mapping, and a test keeps the two in sync.
CANONICAL_UNITS: Mapping[str, u.UnitBase] = MappingProxyType(
    {
        "mass": Msun_h,
        "length": Mpc_h,
        "wavenumber": h_Mpc,
        "number_density": number_density_unit,
        "dndm": dndm_unit,
        "power": power_unit,
        "density": rho_unit,
        "hubble": H0_unit,
    }
)


#: How to write each shared constant in an error message: (h-units, physical units).
_SPELLINGS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "mass": ("hmf.core.units.Msun_h", "u.Msun"),
        "length": ("hmf.core.units.Mpc_h", "u.Mpc"),
        "wavenumber": ("hmf.core.units.h_Mpc", "(1 / u.Mpc)"),
        "number_density": ("hmf.core.units.number_density_unit", "u.Mpc**-3"),
        "dndm": ("hmf.core.units.dndm_unit", "(1 / (u.Msun * u.Mpc**3))"),
        "power": ("hmf.core.units.power_unit", "u.Mpc**3"),
        "density": ("hmf.core.units.rho_unit", "(u.Msun / u.Mpc**3)"),
        "hubble": ("hmf.core.units.H0_unit", "(u.km / u.s / u.Mpc)"),
    }
)


def _missing_unit_message(
    where: str, name: str, x: Any, unit: u.UnitBase, *, has_h0: bool = True
) -> str:
    """The message of the :class:`UnitBoundaryError` for a bare value.

    ``has_h0`` says whether the object can convert physical units (it has an H0);
    if not, only the h-units are suggested.
    """
    kind = next((k for k, v in CANONICAL_UNITS.items() if v == unit), None)
    if kind is None:
        fix = f"`{name} * u.Unit('{unit}')`"
    elif has_h0:
        h_units, physical = _SPELLINGS[kind]
        fix = (
            f"`{name} * {h_units}` (h-units) or `{name} * {physical}` (physical units, "
            "converted with this object's H0)"
        )
    else:
        fix = (
            f"`{name} * {_SPELLINGS[kind][0]}` (h-units: this object has no H0 to convert "
            "physical units with)"
        )
    return (
        f"{where}: argument {name!r} is dimensional, so it must be an astropy Quantity "
        f"in units convertible to '{unit}', not a bare {type(x).__name__}. Multiply it "
        f"by its unit, e.g. {fix}."
    )


def _conversion_message(where: str, name: str, unit: u.UnitBase, error: Exception) -> str:
    """The message of the UnitConversionError for an argument in the wrong unit."""
    return f"{where}: argument {name!r} must be in units convertible to '{unit}': {error}"


class UnitBoundaryError(TypeError):
    """A dimensional argument was given without units (a bare float or array).

    hmf.core never guesses the unit of a dimensional input. Multiply the value by
    its unit, e.g. ``m * hmf.core.units.Msun_h`` or ``m * astropy.units.Msun``.
    """


# ---------------------------------------------------------------------------------
# littleh helpers
# ---------------------------------------------------------------------------------


def littleh_power(unit: u.UnitBase) -> float:
    """Return the power of :data:`littleh` in ``unit`` (0 if it has none).

    Parameters
    ----------
    unit
        Any astropy unit.

    Returns
    -------
    float
        E.g. ``-1`` for :data:`Msun_h`, ``4`` for :data:`dndm_unit`, ``0`` for
        ``u.Msun``.

    Examples
    --------
    >>> littleh_power(Msun_h)
    -1.0
    """
    decomposed = unit.decompose()
    for base, power in zip(decomposed.bases, decomposed.powers, strict=True):
        if base is littleh:
            return float(power)
    return 0.0


@contextmanager
def littleh_units_enabled() -> Iterator[None]:
    """Enable parsing of unit strings that contain ``littleh``.

    ``u.Unit("Msun/littleh")`` raises unless :mod:`astropy.cosmology.units` is
    enabled. This context manager enables it (with
    :func:`astropy.units.add_enabled_units`) for the duration of the block only. It
    does **not** enable the :func:`~astropy.cosmology.units.with_H0` equivalency.

    Examples
    --------
    >>> with littleh_units_enabled():
    ...     unit = u.Unit("Msun / littleh")
    >>> unit == Msun_h
    True
    """
    with u.add_enabled_units(cu):
        yield


def parse_unit(text: str) -> u.UnitBase:
    """Parse a unit string that may contain ``littleh``, e.g. from a config file.

    Parameters
    ----------
    text
        A unit in astropy's generic string format, e.g. ``"Msun / littleh"``.

    Returns
    -------
    astropy.units.UnitBase
        The parsed unit. If it equals one of the shared constants of this module
        (e.g. :data:`Msun_h`), that constant itself is returned, so the result takes
        the boundary's identity fast path.
    """
    with littleh_units_enabled():
        unit = u.Unit(text)
    for canonical in CANONICAL_UNITS.values():
        if unit == canonical:
            return canonical
    return unit


# ---------------------------------------------------------------------------------
# Per-instance conversion context
# ---------------------------------------------------------------------------------


class UnitContext:
    """Per-instance unit-conversion state: the H0, and cached conversion factors.

    Every object with :func:`unit_boundary`-decorated methods may provide one, as its
    ``_unit_context`` attribute (see :class:`HasUnitContext`). The decorator uses it
    for any argument whose unit is not *identical* to the canonical unit.

    Conversion factors are cached, keyed first by the identity of the units (the fast
    lookup) and then by their equality, so each distinct unit is converted once per
    context. The cache is not part of the owning object's value: it is not compared,
    hashed or serialised.

    Parameters
    ----------
    H0
        The Hubble constant used to convert physical units (e.g. M☉) to h-units (e.g.
        M☉/h): a scalar Quantity in units of :data:`H0_unit` (km/s/Mpc), or
        convertible to it. ``None`` if the owner has no cosmology: then only inputs
        that need no H0 (any unit with the canonical power of :data:`littleh`) are
        accepted.
    where
        The name of the owner (e.g. ``"TabulatedPower"``), for error messages.

    Raises
    ------
    UnitBoundaryError
        If ``H0`` is not ``None`` and not a Quantity.
    astropy.units.UnitConversionError
        If ``H0`` is not in units of km/s/Mpc.
    ValueError
        If ``H0`` is not a scalar.

    Notes
    -----
    ``n_computed`` counts the conversion factors this context has computed (its cache
    misses), which tests use to check that the cache is used.
    """

    __slots__ = ("H0", "_by_equality", "_by_identity", "_equivalencies", "n_computed")

    def __init__(self, H0: u.Quantity | None = None, *, where: str = "UnitContext") -> None:
        if H0 is not None:
            if not isinstance(H0, u.Quantity):
                raise UnitBoundaryError(
                    f"{where}: H0 must be a Quantity in km/s/Mpc, e.g. 70 * "
                    f"hmf.core.units.H0_unit, not a bare {type(H0).__name__}."
                )
            if not H0.unit.is_equivalent(H0_unit):
                raise u.UnitConversionError(
                    f"{where}: H0 must be in units convertible to '{H0_unit}', not '{H0.unit}'."
                )
            if H0.ndim != 0:
                raise ValueError(f"{where}: H0 must be a scalar, not of shape {H0.shape}.")
        self.H0 = H0
        self.n_computed = 0
        self._equivalencies: list[Any] | None = None
        # (id(src), id(dst)) -> (src, dst, factor). The units are kept alive, so that
        # their ids can not be reused while they are in the cache.
        self._by_identity: dict[tuple[int, int], tuple[u.UnitBase, u.UnitBase, float]] = {}
        self._by_equality: dict[tuple[u.UnitBase, u.UnitBase], float] = {}

    @property
    def equivalencies(self) -> list[Any]:
        """``cu.with_H0(self.H0)``, built once.

        Always built with this context's H0: ``cu.with_H0()`` without an argument
        silently uses Planck18's.
        """
        if self._equivalencies is None:
            if self.H0 is None:
                raise u.UnitConversionError(
                    "this object has no H0, so it can only accept h-units "
                    "(e.g. hmf.core.units.Msun_h), not physical units."
                )
            self._equivalencies = cu.with_H0(self.H0)
        return self._equivalencies

    def factor(self, src: u.UnitBase, dst: u.UnitBase) -> float:
        """Return the factor converting values in ``src`` to values in ``dst``.

        Parameters
        ----------
        src
            The unit of the input.
        dst
            The canonical unit to convert to.

        Returns
        -------
        float
            ``value_in_dst = value_in_src * factor``.

        Raises
        ------
        astropy.units.UnitConversionError
            If ``src`` has a power of :data:`littleh` that is neither the canonical
            one nor zero (e.g. M☉/h² given as a mass), or the wrong dimension.
        """
        hit = self._by_identity.get((id(src), id(dst)))
        if hit is not None:
            return hit[2]
        factor = self._by_equality.get((src, dst))
        if factor is None:
            factor = self._compute(src, dst)
            self._by_equality[(src, dst)] = factor
        self._by_identity[(id(src), id(dst))] = (src, dst, factor)
        return factor

    def _compute(self, src: u.UnitBase, dst: u.UnitBase) -> float:
        """Compute a conversion factor, checking the power of littleh explicitly."""
        self.n_computed += 1
        src_power = littleh_power(src)
        dst_power = littleh_power(dst)
        if src_power == dst_power:
            # The h's cancel: no H0 needed.
            return float(src.to(dst))
        if src_power == 0:
            # A physical unit: convert with this context's H0.
            return float(src.to(dst, equivalencies=self.equivalencies))
        # cu.with_H0 would happily "convert" e.g. Msun/h**2 to Msun/h, which is a bug in
        # the caller, not a conversion.
        raise u.UnitConversionError(
            f"'{src}' has littleh**{src_power:g}, but '{dst}' needs littleh**"
            f"{dst_power:g} (or no littleh at all, for physical units)."
        )


#: The context used by objects that do not provide their own: no H0.
_NO_H0_CONTEXT = UnitContext(None)


class HasUnitContext(Protocol):
    """An object whose :func:`unit_boundary` methods can convert physical units.

    The decorator reads the instance's ``_unit_context`` attribute when an argument's
    unit is not identical to the canonical unit. Objects without the attribute use a
    shared context with no H0, which accepts any unit with the canonical power of
    :data:`littleh` but rejects physical units that need H0 to convert.

    On a frozen ``attrs`` class, provide it as a :func:`functools.cached_property`
    computed from the object's own H0, so that each instance (including each result
    of ``evolve()``) gets its own cache::

        @functools.cached_property
        def _unit_context(self) -> UnitContext:
            return UnitContext(self.H0)
    """

    @property
    def _unit_context(self) -> UnitContext: ...


# ---------------------------------------------------------------------------------
# The boundary decorator
# ---------------------------------------------------------------------------------

P = ParamSpec("P")
R = TypeVar("R")
S = TypeVar("S")


# Bound once, for the boundary's fast path.
_Quantity: Any = u.Quantity
_view: Callable[..., Any] = np.ndarray.view


def _quantity_from_array(out: Any, unit: u.UnitBase, where: str) -> u.Quantity:
    """Attach ``unit`` to a kernel output without copying it.

    ``Quantity(arr, unit, copy=False)`` raises for scalars under numpy 2, and ``<<``
    costs about 2 µs, so view the (possibly 0-d) array as a Quantity and set its unit.
    A scalar output gives a scalar (0-d) Quantity.
    """
    if type(out) is np.ndarray:
        arr = out
    elif isinstance(out, u.Quantity):
        raise TypeError(
            f"{where} returned a Quantity. Methods decorated with unit_boundary must "
            "return plain arrays in the canonical unit; the decorator attaches it."
        )
    else:
        arr = np.asarray(out)
    q = arr.view(u.Quantity)
    q._unit = unit
    return q


def _check_boundary_units(
    inputs: Mapping[str, Any], returns: u.UnitBase | tuple[u.UnitBase | None, ...] | None
) -> None:
    """Raise a TypeError unless every unit given to :func:`unit_boundary` is a unit.

    The output units are attached without astropy's own checks (see
    :func:`_quantity_from_array`), so they are checked here, once.
    """
    for name, unit in inputs.items():
        if not isinstance(unit, u.UnitBase):
            raise TypeError(f"unit_boundary: the unit of {name!r} must be an astropy unit.")
    for unit in returns if isinstance(returns, tuple) else (returns,):
        if unit is not None and not isinstance(unit, u.UnitBase):
            raise TypeError("unit_boundary: each unit in 'returns' must be an astropy unit.")


def _boundary_specs(
    fn: Callable[..., Any], inputs: Mapping[str, u.UnitBase], where: str
) -> list[tuple[str, int | None, u.UnitBase]]:
    """``(name, position, unit)`` of each argument in ``inputs`` of the method ``fn``.

    The position excludes ``self``, and is None for a keyword-only argument.
    """
    params = list(inspect.signature(fn).parameters.values())[1:]  # drop self
    names = [p.name for p in params]
    specs: list[tuple[str, int | None, u.UnitBase]] = []
    for name, unit in inputs.items():
        if name not in names:
            raise TypeError(f"unit_boundary: {where} has no argument {name!r}.")
        param = params[names.index(name)]
        position = names.index(name) if param.kind is not param.KEYWORD_ONLY else None
        specs.append((name, position, unit))
    return specs


def _method_where(instance: Any, method: str) -> str:
    """Where an error happened, for its message: ``"Class.method()"``.

    Built from the instance's own class (not the method's ``__qualname__``), so that
    an inherited method names the class it was called on. Only called on the error
    paths, so it costs the fast path nothing.
    """
    return f"{type(instance).__name__}.{method}()"


def _convert(
    context: UnitContext, where: str | tuple[Any, str], name: str, x: Any, unit: u.UnitBase
) -> Any:
    """A dimensional value as a plain array in ``unit`` (the one general conversion).

    ``where`` names the place, for error messages: a string, or ``(instance, method)``
    for a method of the boundary, whose name is only built if there is an error.
    """
    if isinstance(x, u.Quantity):
        if x.unit is unit:
            return x.view(np.ndarray)
        try:
            factor = context.factor(x.unit, unit)
        except u.UnitsError as e:
            place = where if isinstance(where, str) else _method_where(*where)
            raise u.UnitConversionError(_conversion_message(place, name, unit, e)) from e
        return x.view(np.ndarray) * factor
    if x is None:
        return None
    place = where if isinstance(where, str) else _method_where(*where)
    raise UnitBoundaryError(
        _missing_unit_message(place, name, x, unit, has_h0=context.H0 is not None)
    )


def _convert_argument(instance: Any, method: str, name: str, x: Any, unit: u.UnitBase) -> Any:
    """An argument of a boundary method as a plain array in ``unit`` (the general path)."""
    context = getattr(instance, "_unit_context", _NO_H0_CONTEXT)
    return _convert(context, (instance, method), name, x, unit)


def to_canonical(
    value: Any,
    unit: u.UnitBase,
    *,
    context: UnitContext | None = None,
    where: str,
    name: str,
) -> Any:
    """Convert a dimensional value to a plain array in a canonical unit.

    This is the conversion :func:`unit_boundary` applies to each dimensional
    argument, for code that converts a value itself (a field converter, a method
    with an ``H0`` argument, ...). Its errors are the boundary's.

    Parameters
    ----------
    value
        A Quantity, or ``None`` (returned as it is, for optional values).
    unit
        The unit to convert to, normally one of :data:`CANONICAL_UNITS`.
    context
        The :class:`UnitContext` whose H0 converts physical units. ``None`` (the
        default) means no H0: only units with the canonical power of :data:`littleh`
        are accepted.
    where
        The object or method the value belongs to (e.g. ``"CAMB"`` or
        ``"Behroozi.modify_dndm()"``), for error messages.
    name
        The name of the value (the argument or field), for error messages.

    Returns
    -------
    numpy.ndarray or None
        The values in ``unit`` (for a scalar, a 0-d array or a numpy scalar). If
        ``value`` is already in exactly ``unit`` (the same object), it is a view of
        ``value``, not a copy.

    Raises
    ------
    UnitBoundaryError
        If ``value`` is a bare number or array.
    astropy.units.UnitConversionError
        If ``value`` can't be converted to ``unit`` (with the context's H0, if any).

    Examples
    --------
    >>> to_canonical([1, 2] * u.kpc / littleh, Mpc_h, where="example", name="r")
    array([0.001, 0.002])
    """
    return _convert(_NO_H0_CONTEXT if context is None else context, where, name, value, unit)


def _require_quantity(
    value: Any, unit: u.UnitBase, *, where: str, name: str, has_h0: bool = True
) -> u.Quantity:
    """Raise the boundary's :class:`UnitBoundaryError` unless ``value`` is a Quantity.

    For a value that is converted later (e.g. once the owner's H0 is known); ``unit``
    is the unit it will be converted to, and ``has_h0`` whether the owner can convert
    physical units.
    """
    if not isinstance(value, u.Quantity):
        raise UnitBoundaryError(_missing_unit_message(where, name, value, unit, has_h0=has_h0))
    return value


def _read_only_quantity(values: Any, unit: u.UnitBase) -> u.Quantity:
    """A read-only float copy of ``values`` (plain, in ``unit``), as a Quantity in ``unit``."""
    q = read_only(values).view(u.Quantity)
    q._unit = unit
    return q


@attrs.frozen
class _QuantityConverter:
    """The converter of a :func:`quantity_field`."""

    unit: u.UnitBase
    optional: bool
    ndim: int | None

    def __call__(self, value: Any, instance: Any, attribute: attrs.Attribute[Any]) -> Any:
        where = type(instance).__name__
        name = attribute.alias or attribute.name
        if value is None and self.optional:
            return None
        if value is None:
            raise UnitBoundaryError(
                _missing_unit_message(where, name, value, self.unit, has_h0=False)
            )
        plain = to_canonical(value, self.unit, where=where, name=name)
        if self.ndim == 1:
            plain = np.atleast_1d(plain)
        if self.ndim is not None and np.ndim(plain) != self.ndim:
            raise ValueError(
                f"{where}: {name!r} must be {'a scalar' if self.ndim == 0 else '1-d'}, not "
                f"of shape {np.shape(plain)}."
            )
        return _read_only_quantity(plain, self.unit)


def quantity_field(
    unit: u.UnitBase,
    *,
    doc: str,
    optional: bool = False,
    ndim: int | None = None,
    **kwargs: Any,
) -> Any:
    """Define an ``attrs`` field (of a model or settings class) that carries a unit.

    The field takes a Quantity in any unit convertible to ``unit`` without an H0
    (models have no cosmology), and stores it as a read-only float Quantity in
    exactly ``unit``, the canonical unit. So reading the field gives a Quantity,
    ``attrs.evolve`` round-trips it, and the field compares and hashes by its value
    in the canonical unit (:func:`hmf.core._serialise.quantity_key`). Kernels read
    the plain values through a private :func:`functools.cached_property` of the
    class.

    Parameters
    ----------
    unit
        The canonical unit, one of the constants of this module (e.g. :data:`h_Mpc`).
    doc
        The field's documentation (see :func:`hmf.core._fields.field`).
    optional
        Whether ``None`` is accepted (and stored as ``None``).
    ndim
        If given, the number of dimensions the value must have: 0 (a scalar) or 1
        (a scalar becomes a 1-element array).
    **kwargs
        Passed on to :func:`attrs.field` (``default``, ``validator``, ...).

    Returns
    -------
    Any
        The field definition.

    Raises
    ------
    UnitBoundaryError
        On construction, for a bare number or array (or ``None`` if not optional).
    astropy.units.UnitConversionError
        On construction, for a value in physical units, or not convertible to ``unit``.
    ValueError
        On construction, for a value with the wrong number of dimensions.
    """
    converter = attrs.Converter(
        _QuantityConverter(unit, optional, ndim), takes_self=True, takes_field=True
    )
    return field(doc=doc, converter=converter, eq=quantity_key, **kwargs)


def _dimensionless(out: Any) -> Any:
    """Apply the scalar rule to a dimensionless output.

    A 0-d array or a Python float becomes a numpy scalar (:class:`numpy.float64` for
    floats, which is also a :class:`float`); arrays (of any other shape) and numpy
    scalars are unchanged, and so is anything else (e.g. a result object, or a tuple:
    a method returning a tuple declares ``returns`` as a tuple).
    """
    if type(out) is np.ndarray:
        return out[()] if out.ndim == 0 else out
    if type(out) is float:
        return np.float64(out)
    return out


def _finish(out: Any, returns: Any, impl: str) -> Any:
    """Attach the output unit(s) ``returns`` to a kernel output.

    ``impl`` is the qualified name of the decorated function, for the error raised if
    it returned a Quantity (a bug in it, not in the caller).
    """
    if returns is None:
        return _dimensionless(out)
    if isinstance(returns, tuple):
        return tuple(
            _dimensionless(o) if r is None else _quantity_from_array(o, r, impl)
            for o, r in zip(out, returns, strict=True)
        )
    return _quantity_from_array(out, returns, impl)


def _wrap_one(
    fn: Callable[..., Any],
    name: str,
    position: int | None,
    unit: u.UnitBase,
    returns: Any,
) -> Callable[..., Any]:
    """The boundary wrapper of a method with one dimensional argument.

    The common case, a plain Quantity in exactly the canonical unit and a plain
    ndarray out, is inlined; anything else goes through :func:`_convert_argument` and
    :func:`_finish`.
    """
    # A keyword-only argument is never in args.
    pos = sys.maxsize if position is None else position
    single_return = isinstance(returns, u.UnitBase)
    method, impl = fn.__name__, fn.__qualname__

    @functools.wraps(fn)
    def wrapper(self: Any, /, *args: Any, **kwargs: Any) -> Any:
        if len(args) > pos:
            x = args[pos]
            if type(x) is _Quantity and x._unit is unit:
                x = _view(x, np.ndarray)
            else:
                x = _convert_argument(self, method, name, x, unit)
            args = (x, *args[1:]) if pos == 0 else (*args[:pos], x, *args[pos + 1 :])
        elif name in kwargs:
            x = kwargs[name]
            if type(x) is _Quantity and x._unit is unit:
                kwargs[name] = _view(x, np.ndarray)
            else:
                kwargs[name] = _convert_argument(self, method, name, x, unit)
        out = fn(self, *args, **kwargs)
        if type(out) is np.ndarray:
            if single_return:
                q = _view(out, _Quantity)
                q._unit = returns
                return q
            if returns is None:
                return out if out.ndim else out[()]
        return _finish(out, returns, impl)

    return wrapper


def _wrap_many(
    fn: Callable[..., Any],
    specs: list[tuple[str, int | None, u.UnitBase]],
    returns: Any,
) -> Callable[..., Any]:
    """The boundary wrapper of a method with any number of dimensional arguments.

    As :func:`_wrap_one`, each argument that is a plain Quantity in exactly its
    canonical unit is viewed as an ndarray, and a plain ndarray output is finished
    inline; anything else goes through :func:`_convert_argument` and :func:`_finish`.
    """
    # The arguments that may be passed by position, as (name, position, unit).
    positional = tuple((n, p, unit) for n, p, unit in specs if p is not None)
    keywords = tuple((n, unit) for n, _, unit in specs)
    single_return = isinstance(returns, u.UnitBase)
    method, impl = fn.__name__, fn.__qualname__

    @functools.wraps(fn)
    def wrapper(self: Any, /, *args: Any, **kwargs: Any) -> Any:
        if args:
            args_list = list(args)
            n_args = len(args_list)
            for name, position, unit in positional:
                if position < n_args:
                    x = args_list[position]
                    if type(x) is _Quantity and x._unit is unit:
                        args_list[position] = _view(x, np.ndarray)
                    else:
                        args_list[position] = _convert_argument(self, method, name, x, unit)
            args = tuple(args_list)
        if kwargs:
            for name, unit in keywords:
                if name in kwargs:
                    x = kwargs[name]
                    if type(x) is _Quantity and x._unit is unit:
                        kwargs[name] = _view(x, np.ndarray)
                    else:
                        kwargs[name] = _convert_argument(self, method, name, x, unit)
        out = fn(self, *args, **kwargs)
        if type(out) is np.ndarray:
            if single_return:
                q = _view(out, _Quantity)
                q._unit = returns
                return q
            if returns is None:
                return out if out.ndim else out[()]
        return _finish(out, returns, impl)

    return wrapper


def unit_boundary(
    *, returns: u.UnitBase | tuple[u.UnitBase | None, ...] | None = None, **inputs: u.UnitBase
) -> Callable[[Callable[Concatenate[S, P], R]], Callable[Concatenate[S, P], R]]:
    """Make a method a units boundary: Quantities outside, plain arrays inside.

    For each dimensional argument named in ``inputs``, the decorated method receives a
    plain :class:`numpy.ndarray` (0-d for a scalar) in the given canonical unit:

    1. if the argument's unit *is* the canonical unit object, it is viewed as an
       ndarray, with no conversion and no copy;
    2. otherwise it is multiplied by a conversion factor cached on the instance's
       :class:`UnitContext` (see :class:`HasUnitContext`): physical units are
       converted with the instance's H0, and a :data:`littleh` power other than the
       canonical one or zero is an error;
    3. a bare float or ndarray raises :class:`UnitBoundaryError`; ``None`` passes
       through, for optional arguments.

    Arguments not named in ``inputs`` (the dimensionless ones) pass through unchanged.

    The method returns plain floats or arrays in the canonical output unit, and the
    decorator applies **the scalar rule**: a scalar in gives a scalar out.

    * A dimensional output (``returns`` is a unit) becomes a Quantity in that unit:
      a 0-d Quantity for a scalar.
    * A dimensionless output (``returns`` is ``None``) stays a plain array, and a
      scalar becomes a numpy scalar (:class:`numpy.float64`, which is also a
      :class:`float`). Anything that is not a float or an array (e.g. a result
      object) is returned as it is.

    Decorate every public method of a model or stage, including those without
    dimensional arguments (``@unit_boundary()``), so that they all follow the scalar
    rule. Errors name the class the method was called on, and the method.

    Library code that is already working in canonical units should call the kernels
    directly, not the decorated methods: the boundary is for users.

    Parameters
    ----------
    returns
        The canonical unit of the output, or ``None`` if it is dimensionless. A
        tuple of units (or ``None`` for an element that is dimensionless) if the
        method returns a tuple.
    **inputs
        The canonical unit of each dimensional argument, by argument name. Use the
        constants of this module (e.g. :data:`Msun_h`), so that inputs built from the
        same constants take the identity fast path.

    Returns
    -------
    callable
        The decorator.

    Raises
    ------
    TypeError
        At decoration time, if ``inputs`` names an argument the method does not have,
        or a unit in ``inputs`` or ``returns`` is not an astropy unit.

    Examples
    --------
    >>> class Toy:
    ...     @unit_boundary(m=Msun_h, returns=Msun_h)
    ...     def double(self, m, z=0.0):
    ...         return 2 * m
    >>> Toy().double(1e12 * Msun_h)
    <Quantity 2.e+12 solMass / littleh>
    """
    _check_boundary_units(inputs, returns)

    def decorator(fn: Callable[Concatenate[S, P], R]) -> Callable[Concatenate[S, P], R]:
        specs = _boundary_specs(fn, inputs, f"{fn.__qualname__}()")
        # The boundary's fixed cost is budgeted at 2 µs per call and gated in
        # benchmarks/test_gates.py. The usual method has one dimensional argument, so
        # that case gets a wrapper without loops.
        if len(specs) == 1:
            name, position, unit = specs[0]
            wrapper = _wrap_one(fn, name, position, unit, returns)
        else:
            wrapper = _wrap_many(fn, specs, returns)
        wrapper.__unit_boundary__ = {"inputs": dict(inputs), "returns": returns}  # type: ignore[attr-defined]
        return wrapper

    return decorator


# ---------------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------------

#: The critical density today over h², :math:`3 (100\,{\rm km\,s^{-1}\,Mpc^{-1}})^2 /
#: (8\pi G)`, from astropy's G and M☉: a Quantity in :data:`rho_unit` (M☉ h² / Mpc³),
#: about 2.775e11.
RHO_CRIT0_H2 = (3 * (100 * u.km / u.s / u.Mpc) ** 2 / (8 * np.pi * _constants.G)).to(
    u.Msun / u.Mpc**3
).value * rho_unit
