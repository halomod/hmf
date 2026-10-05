"""The units boundary of the v4 core.

hmf.core follows one units policy ("option B" of issue #389):

* **Quantities at every public boundary.** Stage fields, public model and stage
  methods, and all their outputs carry :class:`astropy.units.Quantity`.
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
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from types import MappingProxyType
from typing import Any, Concatenate, ParamSpec, Protocol, TypeVar

import astropy.cosmology.units as cu
import astropy.units as u
import numpy as np

__all__ = [
    "CANONICAL_UNITS",
    "H0_unit",
    "HasUnitContext",
    "Mpc_h",
    "Msun_h",
    "UnitBoundaryError",
    "UnitContext",
    "dndm_unit",
    "h_Mpc",
    "littleh",
    "littleh_power",
    "littleh_units_enabled",
    "number_density_unit",
    "parse_unit",
    "power_unit",
    "rho_unit",
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


def _missing_unit_message(where: str, name: str, x: Any, unit: u.UnitBase) -> str:
    """The message of the :class:`UnitBoundaryError` for a bare value."""
    kind = next((k for k, v in CANONICAL_UNITS.items() if v == unit), None)
    if kind is None:
        fix = f"`{name} * u.Unit('{unit}')`"
    else:
        h_units, physical = _SPELLINGS[kind]
        fix = (
            f"`{name} * {h_units}` (h-units) or `{name} * {physical}` (physical units, "
            "converted with this object's H0)"
        )
    return (
        f"{where}: argument {name!r} is dimensional, so it must be an astropy Quantity "
        f"in units convertible to '{unit}', not a bare {type(x).__name__}. Multiply it "
        f"by its unit, e.g. {fix}."
    )


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
        M☉/h). ``None`` if the owner has no cosmology: then only inputs that need no
        H0 (any unit with the canonical power of :data:`littleh`) are accepted.

    Notes
    -----
    ``n_computed`` counts the conversion factors this context has computed (its cache
    misses), which tests use to check that the cache is used.
    """

    __slots__ = ("H0", "_by_equality", "_by_identity", "_equivalencies", "n_computed")

    def __init__(self, H0: u.Quantity | None = None) -> None:
        if H0 is not None and not isinstance(H0, u.Quantity):
            raise UnitBoundaryError(
                f"UnitContext: H0 must be a Quantity, e.g. 70 * hmf.core.units.H0_unit, "
                f"not {type(H0).__name__}."
            )
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
    The method's return value (a float or ndarray in the canonical output unit) is
    wrapped as a Quantity in ``returns``; a scalar in gives a scalar out.

    Library code that is already working in canonical units should call the kernels
    directly, not the decorated methods: the boundary is for users.

    Parameters
    ----------
    returns
        The canonical unit of the output. A tuple of units (or ``None`` for an
        element that is dimensionless) if the method returns a tuple. ``None`` leaves
        the output as it is.
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
        where = f"{fn.__qualname__}()"
        specs = _boundary_specs(fn, inputs, where)
        # The arguments that may be passed by position, as (name, position, unit).
        positional = tuple((n, p, unit) for n, p, unit in specs if p is not None)

        def convert(instance: Any, name: str, x: Any, unit: u.UnitBase) -> Any:
            if isinstance(x, u.Quantity):
                if x.unit is unit:
                    return x.view(np.ndarray)
                context = getattr(instance, "_unit_context", _NO_H0_CONTEXT)
                try:
                    factor = context.factor(x.unit, unit)
                except u.UnitsError as e:
                    raise u.UnitConversionError(
                        f"{where}: argument {name!r} must be in units convertible to '{unit}': {e}"
                    ) from e
                return x.view(np.ndarray) * factor
            if x is None:
                return None
            raise UnitBoundaryError(_missing_unit_message(where, name, x, unit))

        # This wrapper is the boundary's fixed cost, budgeted at 2 µs per call and
        # gated in benchmarks/test_gates.py, so the common case (a plain Quantity
        # in exactly the canonical unit, and a plain ndarray out) is inlined; anything
        # else goes through convert() and _quantity_from_array().
        single_return = isinstance(returns, u.UnitBase)

        @functools.wraps(fn)
        def wrapper(self: S, /, *args: P.args, **kwargs: P.kwargs) -> R:
            if args:
                n_args = len(args)
                for name, position, unit in positional:
                    if position < n_args:
                        x: Any = args[position]
                        if type(x) is _Quantity and x._unit is unit:
                            x = _view(x, np.ndarray)
                        else:
                            x = convert(self, name, x, unit)
                        if position == 0:
                            args = (x, *args[1:])  # type: ignore[assignment]
                        else:
                            args = (*args[:position], x, *args[position + 1 :])  # type: ignore[assignment]
            if kwargs:
                for name, _, unit in specs:
                    if name in kwargs:
                        x = kwargs[name]
                        if type(x) is _Quantity and x._unit is unit:
                            kwargs[name] = _view(x, np.ndarray)
                        else:
                            kwargs[name] = convert(self, name, x, unit)
            out: Any = fn(self, *args, **kwargs)
            if single_return:
                if type(out) is np.ndarray:
                    q = _view(out, _Quantity)
                    q._unit = returns
                    return q  # type: ignore[no-any-return]
                return _quantity_from_array(out, returns, where)  # type: ignore[no-any-return]
            if returns is None:
                return out  # type: ignore[no-any-return]
            return tuple(  # type: ignore[return-value]
                o if r is None else _quantity_from_array(o, r, where)
                for o, r in zip(out, returns, strict=True)
            )

        wrapper.__unit_boundary__ = {"inputs": dict(inputs), "returns": returns}  # type: ignore[attr-defined]
        return wrapper

    return decorator
