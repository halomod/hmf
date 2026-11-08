"""Configurations: a stage tree as plain data, for files, pickles and content hashes.

A *configuration* describes a stage tree completely: the **resolved** value of every
field of every stage (defaults and computed defaults included), under a header with
the **schema version** and the versions of the packages that computed it. It is the
unit of serialisation and reproducibility of :mod:`hmf.core`:

* :func:`to_config` and :func:`from_config` convert a stage tree to and from a nested
  mapping of plain values (numbers, strings, lists, mappings);
* :func:`dumps_toml`, :func:`loads_toml`, :func:`dump_toml` and :func:`load_toml`
  write and read it as TOML;
* a stage pickles as its configuration (see `Pickling`_), never as live objects;
* :func:`content_hash` hashes its physics.

::

    from hmf.core import config
    from hmf.core.mass_function import MassFunction

    mf = MassFunction.build(fit="ST", sigma_8=0.8)
    config.dump_toml(mf, "mf.toml")
    same = config.load_toml("mf.toml")  # equal to mf, with bit-identical results

Layout
------
The configuration mirrors the *fields* of the stage tree: each field of the root stage
is a key, and each stage held as a field is a table of its own fields
(``[linear_power.transfer]``). Derived stages and cached results are not written; they
are rebuilt. The ``[hmf]`` table is the header::

    [hmf]
    schema_version = 1
    stage = "hmf.core.mass_function:MassFunction"

    [hmf.versions]
    hmf = "4.0.0"
    camb = "1.6.6"
    astropy = "7.2.0"
    numpy = "2.4.3"
    scipy = "1.17.1"

``stage`` is the class of the root stage, by its import path. ``versions`` holds the
versions of hmf, camb, classy, astropy, numpy and scipy (``None`` in the mapping, and
left out in TOML, for a package that is not installed).

Encodings
---------
Every value is written so that it reads back to an equal value, bit for bit:

* **Numbers, strings, booleans and literals** (e.g. ``domain_policy``) as themselves;
  tuples of numbers as lists.
* **Models** as a table of the registry's qualified name and every field, recursively
  for fields that are models: ``fit = {model = "hmf.core.fits:Tinker08", A = 0.186,
  ...}``. On reading, the name is looked up through the field's model kind
  (:meth:`Kind.get <hmf.core.model.Model.get>`), so an alias (``model = "ST"``) or a
  plugin's model works too; a bare name (``fit = "ST"``) is the model with its default
  parameters.
* **Quantities** as ``{value = ..., unit = "..."}``, with h-units written explicitly
  (``unit = "solMass / littleh"``); a ``dtype`` key is added if the values are not
  float64. Units are parsed with :func:`~hmf.core.units.parse_unit`, and a unit with
  a power of littleh is converted by :func:`~hmf.core.units.to_canonical`, so the
  littleh power is checked as at the units boundary: ``"solMass / littleh2"`` for a
  mass raises.
* **Arrays** as lists, or ``{value = [...], dtype = "..."}`` if they are not float64.
* **Cosmologies** (astropy) as a table of their class (astropy's name for it), name,
  every parameter (``H0``, ``Om0``, ``Tcmb0``, ``Neff``, ``m_nu`` as a Quantity array,
  ``Ob0``, ...) and their ``meta`` (e.g. ``n`` and ``sigma8``), through astropy's
  mapping I/O. The name and ``meta`` are information only: they are not part of the
  cosmology's value (:func:`~hmf.core._serialise.cosmology_key`) or of the content
  hash. A ``meta`` entry is written with the encodings here (an array always with its
  ``dtype``, so that it reads back as an array); an entry that has none (e.g. a
  :class:`datetime.date`) is left out, with one warning per configuration naming the
  keys left out. The entries the stages read (:data:`READ_META_KEYS`) are never left
  out: if one can't be written, writing the configuration raises, and so does the
  content hash.
* **Accuracy settings** and other ``attrs`` values (a
  :class:`~hmf.core.cache.DiskCache`) as a table of every field.
* **Mass definitions** as a table of their class and fields, e.g.
  ``mass_definition = {class = "hmf.core.mass_definitions:SOCritical",
  overdensity = 500.0}``; a short name (``mass_definition = "500c"``) is read too.
* **A value whose class is not the field's declared class** (a mass definition, the
  power source of a :class:`~hmf.core.mass_variance.MassVariance`, a subclass of a
  stage) has a ``class`` key with its import path. Models always have ``model``.
* **None** is ``None`` in the mapping. TOML has no null: a field whose value is
  ``None`` is left out when its default is ``None``, and is written as
  ``{none = true}`` otherwise (e.g. ``Reed03(A=None)``).
* **Paths** as strings, and **enums** as their values.

Reading
-------
:func:`from_config` reads a configuration with the names of
:meth:`Stage.from_flat <hmf.core.stage.Stage.from_flat>`, through the routing table
(:mod:`hmf.core.routing`): nested tables, dotted keys (``"linear_power.sigma_8"``),
flat names (``sigma_8``) and paths into a model's fields (``"fit.A"``). So a *partial*
configuration written by hand also reads: every field it does not give takes its
default, exactly as :meth:`~hmf.core.mass_function.MassFunction.build` would give it
(e.g. ``sigma_8`` from the cosmology). A model table without ``model`` changes fields
of the default model. Only the header is required::

    [hmf]
    schema_version = 1
    stage = "hmf.core.mass_function:MassFunction"

    sigma_8 = 0.8  # (at the top, before [hmf], in a file)
    fit = "ST"

An unknown field raises a :class:`TypeError` with the closest names; an unknown model
a :class:`~hmf.core.model.ModelNotFoundError` with suggestions; a wrong unit an
:class:`astropy.units.UnitConversionError`. The error notes where in the
configuration the value is.

Trust
-----
A configuration names classes to import (its ``stage``, a model's ``model`` and a value's
``class``), as :meth:`Model.get <hmf.core.model.Model.get>` does for an import path: reading
one imports those modules, which runs their code. Read configurations you trust, as you
would unpickle only pickles you trust.

Schema versions
---------------
:data:`SCHEMA_VERSION` is the version of this layout and of the defaults. A change to
either, including a change to the default of any physics parameter, is a schema
change: it bumps the version, and registers a migration from the previous version
(:func:`register_migration`), a function from a version-n document to a version-n+1
one. Reading an older configuration applies the migrations in turn, and logs (with
:mod:`logging`, at INFO) what each changed; with ``migrate=False`` it raises instead.
A configuration from a newer schema, or one with no migration to this one, raises a
:class:`ValueError`. Package versions that differ from the running ones are logged
(at WARNING); ``strict=True`` raises a :class:`ValueError` instead. Nothing is ever
silently read in another meaning.

Content hash
------------
:func:`content_hash` is the SHA-256 hash of the canonical JSON
(:mod:`hmf.core._serialise`) of the configuration's physics: every field, and the
versions of the packages that compute results (hmf, camb, classy and astropy). It
leaves out the fields marked as changing no result (``physics=False`` in
:func:`hmf.core.field`, e.g. ``disk_cache``), and a cosmology's name and ``meta``. So
it is the same after a round trip through a configuration, and two stage trees with
the same physics have the same hash. :func:`hmf.core._serialise.canonical` (which the
disk cache's keys are computed with) encodes models, Quantities, arrays and
cosmologies through the same converter.

Pickling
--------
A :class:`~hmf.core.stage.Stage` pickles as its configuration, plus the Boltzmann-code
runs its tree has already used, as plain arrays (see
:func:`hmf.core._boltzmann.export_runs`). Unpickling rebuilds the tree from the
configuration, and puts the runs in the in-process memo, so it does not run CAMB or
CLASS again; results are bit-identical. No CAMB or CLASS object is ever pickled.
:func:`copy.copy` and :func:`copy.deepcopy` of a stage return the stage itself (it is
immutable), with everything it has computed.
"""

from __future__ import annotations

import copy
import difflib
import functools
import importlib.metadata
import json
import logging
import os
import re
import tempfile
import tomllib
import types
import typing
import warnings
from collections.abc import Callable, Iterable, Mapping
from contextvars import ContextVar
from enum import Enum
from os import PathLike
from pathlib import Path, PurePath
from typing import Any, Literal

import astropy.units as u
import attrs
import cattrs
import numpy as np
import tomli_w
from astropy.cosmology import Cosmology

from . import routing
from ._boltzmann import BoltzmannRun, export_runs, import_runs
from ._fields import NO_DEFAULT, _resolved_fields, is_physics
from .model import Model, _import_from_path, qualified_name
from .stage import Stage
from .units import CANONICAL_UNITS, littleh, littleh_power, parse_unit, to_canonical

__all__ = [
    "HASHED_PACKAGES",
    "MIGRATIONS",
    "PACKAGES",
    "READ_META_KEYS",
    "SCHEMA_VERSION",
    "Migration",
    "canonical_value",
    "content_hash",
    "converter",
    "dump_toml",
    "dumps_toml",
    "from_config",
    "load_toml",
    "loads_toml",
    "package_versions",
    "register_migration",
    "structure",
    "to_config",
    "unstructure",
]

logger = logging.getLogger(__name__)

#: The version of the configuration schema: its layout, and the defaults of every
#: parameter. A change to either is a schema change (see the module documentation).
SCHEMA_VERSION = 1

#: The packages whose versions a configuration records.
PACKAGES: tuple[str, ...] = ("hmf", "camb", "classy", "astropy", "numpy", "scipy")

#: The packages whose versions are part of the content hash: those that compute
#: results.
HASHED_PACKAGES: tuple[str, ...] = ("hmf", "camb", "classy", "astropy")

#: The key of the header table.
HEADER = "hmf"

#: How TOML, which has no null, writes a None that is not a field's default.
NONE_MARK: Mapping[str, bool] = types.MappingProxyType({"none": True})

#: The keys of an encoded Quantity.
_QUANTITY_KEYS = frozenset({"value", "unit", "dtype"})

#: The type of the settings fields of the Boltzmann models (written as a table).
_PAIRS = tuple[tuple[str, Any], ...]

#: The keys of a cosmology's ``meta`` that the stages read: the defaults of ``n_s``
#: and ``sigma_8`` of :meth:`MassFunction.build
#: <hmf.core.mass_function.MassFunction.build>`. A configuration can't leave them out.
READ_META_KEYS: frozenset[str] = frozenset({"n", "sigma8"})

#: The directory of hmf.core, whose frames a warning skips to reach the caller.
_PACKAGE_DIR = str(Path(__file__).parent)

_Mode = Literal["config", "toml", "hash"]
#: What the converter is writing: a mapping, TOML, or the input of a content hash.
_mode: ContextVar[_Mode] = ContextVar("hmf_config_mode", default="config")
#: The keys of the meta entries left out of the configuration being written.
_dropped_meta: ContextVar[list[str] | None] = ContextVar("hmf_config_dropped_meta", default=None)


# ---------------------------------------------------------------------------------
# Package versions and migrations
# ---------------------------------------------------------------------------------


@functools.cache
def package_versions() -> Mapping[str, str | None]:
    """The installed versions of the packages a configuration records.

    Returns
    -------
    Mapping
        The version of each of :data:`PACKAGES`, or ``None`` if it is not installed.
    """
    out: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return types.MappingProxyType(out)


@attrs.frozen
class Migration:
    """A migration of a configuration from one schema version to the next.

    Parameters
    ----------
    from_version
        The schema version it reads; it writes ``from_version + 1``.
    description
        What it changes, in a sentence: it is logged when it is applied.
    function
        Takes the whole configuration (header included) of version ``from_version``,
        and returns that of the next version. It may modify its argument, which is a
        copy.
    """

    from_version: int
    description: str
    function: Callable[[dict[str, Any]], dict[str, Any]]


#: The registered migrations, by the schema version each reads.
MIGRATIONS: dict[int, Migration] = {}

_F = typing.TypeVar("_F", bound=Callable[[dict[str, Any]], dict[str, Any]])


def register_migration(from_version: int, description: str) -> Callable[[_F], _F]:
    """Register a function as the migration from schema ``from_version`` to the next.

    Parameters
    ----------
    from_version
        The schema version the function reads.
    description
        What the migration changes, in a sentence (it is logged when applied).

    Returns
    -------
    callable
        The decorator, which returns the function unchanged.

    Raises
    ------
    ValueError
        If a migration from that version is already registered.

    Examples
    --------
    A migration renaming a field::

        @register_migration(1, "MassFunction.delta_c is renamed to critical_density.")
        def _rename_delta_c(config):
            if "delta_c" in config:
                config["critical_density"] = config.pop("delta_c")
            return config
    """

    def decorator(function: _F) -> _F:
        if from_version in MIGRATIONS:
            raise ValueError(f"A migration from schema {from_version} is already registered.")
        MIGRATIONS[from_version] = Migration(from_version, description, function)
        return function

    return decorator


# ---------------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------------


def _is_union(tp: Any) -> bool:
    return isinstance(tp, types.UnionType) or typing.get_origin(tp) is typing.Union


def _without_none(tp: Any) -> Any:
    """The type of an optional field without None (the type itself if not optional)."""
    if _is_union(tp):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def _is_class(tp: Any, base: type | tuple[type, ...]) -> bool:
    return isinstance(tp, type) and issubclass(tp, base)


def _is_none(data: Any) -> bool:
    """Whether ``data`` encodes None (None, or TOML's ``{none = true}``)."""
    return data is None or (
        isinstance(data, Mapping) and len(data) == 1 and data.get("none") is True
    )


def _none() -> Any:
    """How None is written in the current mode."""
    return dict(NONE_MARK) if _mode.get() == "toml" else None


def _suggest(name: str, candidates: Iterable[str]) -> str:
    close = difflib.get_close_matches(name, list(candidates), n=3)
    return f" Did you mean {' or '.join(repr(c) for c in close)}?" if close else ""


def _note(error: BaseException, where: str) -> None:
    """Add where in the configuration a value was, to an error about it (once)."""
    note = f"(in the configuration, at {where!r})"
    if where and note not in getattr(error, "__notes__", ()):
        error.add_note(note)


def _import_class(name: Any, base: Any, where: str) -> type:
    """The class at an import path, checked to be a subclass of ``base`` if it is one."""
    if not isinstance(name, str):
        raise TypeError(f"{where}: 'class' must be an import path, not {type(name).__name__}.")
    found = _import_from_path(name)
    if not isinstance(found, type):
        raise ValueError(  # noqa: TRY004 (a configuration error: see hmf.core.domain)
            f"{where}: no class can be imported from {name!r}."
        )
    is_protocol = getattr(base, "_is_protocol", False)
    if isinstance(base, type) and not is_protocol and not issubclass(found, base):
        raise TypeError(f"{where}: {name!r} is not a {base.__name__}.")
    return found


def _fields(cls: type) -> dict[str, attrs.Attribute[Any]]:
    """The fields of an attrs class that its constructor takes, by keyword."""
    return {a.alias or a.name: a for a in _resolved_fields(cls) if a.init}


def _static_default(attribute: attrs.Attribute[Any]) -> Any:
    """A field's default, if it does not depend on the instance (else NO_DEFAULT)."""
    default = attribute.default
    if isinstance(default, attrs.Factory):  # type: ignore[arg-type]
        factory: Any = default
        return NO_DEFAULT if factory.takes_self else factory.factory()
    return default


# ---------------------------------------------------------------------------------
# The converter: unstructuring
# ---------------------------------------------------------------------------------

#: The converter between hmf.core values and plain data.
converter = cattrs.Converter()


def unstructure(value: Any) -> Any:
    """Encode a value as plain data, as in a configuration (see the module documentation).

    Parameters
    ----------
    value
        A model, Quantity, array, cosmology, accuracy setting, stage (its fields), or a
        number, string, list or mapping of these.

    Returns
    -------
    Any
        Nested dicts, lists, numbers, strings, booleans and None.

    Raises
    ------
    TypeError
        If a value has no encoding.
    """
    return converter.unstructure(value)


def _unstructure_unknown(value: Any) -> Any:
    raise TypeError(f"A {type(value).__name__} can't be written in a configuration: {value!r}.")


def _unstructure_plain(value: Any) -> Any:
    return _none() if value is None else value


def _unstructure_sequence(value: Any) -> list[Any]:
    return [unstructure(v) for v in value]


def _unstructure_mapping(
    value: Mapping[Any, Any], encode: Callable[[Any], Any] | None = None
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in value.items():
        if not isinstance(k, str):
            raise TypeError(f"Mapping keys in a configuration must be strings, not {k!r}.")
        out[k] = (encode or unstructure)(v)
    return out


def _plain_values(values: np.ndarray[Any, Any]) -> Any:
    return values.item() if values.ndim == 0 else values.tolist()


def _unstructure_array(value: np.ndarray[Any, Any]) -> Any:
    if value.dtype == np.float64:
        return _plain_values(value)
    return {"value": _plain_values(value), "dtype": value.dtype.name}


def _unstructure_quantity(value: u.Quantity) -> dict[str, Any]:
    values = np.asarray(value.value)
    out: dict[str, Any] = {"value": _plain_values(values), "unit": value.unit.to_string()}
    if values.dtype != np.float64:
        out["dtype"] = values.dtype.name
    return out


def _unstructure_untyped(value: Any) -> Any:
    """Encode a value with no declared type, so that it reads back as the same type.

    Arrays always carry their dtype (a bare list reads back as a list).
    """
    if isinstance(value, np.ndarray) and not isinstance(value, u.Quantity):
        return {"value": _plain_values(value), "dtype": value.dtype.name}
    if isinstance(value, Mapping):
        return _unstructure_mapping(value, _unstructure_untyped)
    if isinstance(value, (list, tuple)):
        return [_unstructure_untyped(v) for v in value]
    return unstructure(value)


def _unstructure_meta(meta: Mapping[Any, Any]) -> tuple[dict[str, Any], list[str]]:
    """Encode a cosmology's ``meta``: the entries that encode, and the keys of the rest.

    Raises
    ------
    TypeError
        If an entry that a stage reads (:data:`READ_META_KEYS`) can't be encoded.
    """
    out: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in meta.items():
        try:
            out.update(_unstructure_mapping({key: value}, _unstructure_untyped))
        except TypeError as error:
            if key in READ_META_KEYS:
                _note(error, f"the cosmology's meta[{key!r}], which the stages read")
                raise
            dropped.append(repr(key))
    return out, dropped


def _unstructure_cosmology(value: Cosmology) -> dict[str, Any]:
    mapping = dict(value.to_format("mapping", cosmology_as_str=True))
    out: dict[str, Any] = {"class": mapping.pop("cosmology")}
    name = mapping.pop("name", None)
    meta, dropped = _unstructure_meta(mapping.pop("meta", None) or {})
    hashing = _mode.get() == "hash"
    if not hashing and name is not None:
        out["name"] = name
    for key, parameter in mapping.items():
        out[key] = unstructure(parameter)
    if hashing:
        return out
    if meta:
        out["meta"] = meta
    if dropped:
        collected = _dropped_meta.get()
        if collected is None:
            _warn_dropped_meta(dropped)
        else:
            collected.extend(dropped)
    return out


def _warn_dropped_meta(keys: Iterable[str]) -> None:
    """Warn, from the caller outside hmf.core, that meta entries are left out."""
    names = ", ".join(dict.fromkeys(keys))
    warnings.warn(
        f"The cosmology's meta entries {names} can't be written in a configuration, and "
        "are left out of it (meta is information only).",
        UserWarning,
        skip_file_prefixes=(_PACKAGE_DIR,),
    )


def _unstructure_field(value: Any, declared: Any) -> Any:
    """Encode the value of a field of the declared type."""
    declared = _without_none(declared)
    if declared == _PAIRS:
        return _unstructure_mapping(dict(value))
    out = unstructure(value)
    if attrs.has(type(value)) and not isinstance(value, Model) and type(value) is not declared:
        out = {"class": qualified_name(type(value)), **out}
    return out


def _unstructure_attrs(value: Any) -> dict[str, Any]:
    """Encode an attrs value (a model, stage, setting, ...) as a table of its fields."""
    cls = type(value)
    mode = _mode.get()
    out: dict[str, Any] = {}
    if isinstance(value, Model):
        out["model"] = qualified_name(cls)
    for key, a in _fields(cls).items():
        if mode == "hash" and not is_physics(a):
            continue
        if key in out:  # pragma: no cover - no model has a field named "model"
            raise TypeError(
                f"{cls.__name__}: the field {key!r} can't be written in a configuration."
            )
        field_value = getattr(value, a.name)
        if field_value is None:
            if mode == "toml" and a.default is None:
                continue  # left out: reading it back gives the default, None
            out[key] = _none()
        else:
            out[key] = _unstructure_field(field_value, a.type)
    return out


converter.register_unstructure_hook_func(lambda t: True, _unstructure_unknown)
converter.register_unstructure_hook_func(
    lambda t: t in (type(None), bool, int, float, str), _unstructure_plain
)
converter.register_unstructure_hook_func(
    lambda t: _is_class(t, np.generic), lambda v: unstructure(v.item())
)
converter.register_unstructure_hook_func(
    lambda t: _is_class(t, (list, tuple)), _unstructure_sequence
)
converter.register_unstructure_hook_func(lambda t: _is_class(t, Mapping), _unstructure_mapping)
converter.register_unstructure_hook_func(lambda t: _is_class(t, PurePath), str)
converter.register_unstructure_hook_func(
    lambda t: _is_class(t, Enum), lambda v: unstructure(v.value)
)
converter.register_unstructure_hook_func(
    lambda t: _is_class(t, u.UnitBase), lambda v: {"unit": v.to_string()}
)
converter.register_unstructure_hook_func(lambda t: _is_class(t, np.ndarray), _unstructure_array)
converter.register_unstructure_hook_func(lambda t: _is_class(t, u.Quantity), _unstructure_quantity)
converter.register_unstructure_hook_func(lambda t: _is_class(t, Cosmology), _unstructure_cosmology)
converter.register_unstructure_hook_func(attrs.has, _unstructure_attrs)


# ---------------------------------------------------------------------------------
# The converter: structuring
# ---------------------------------------------------------------------------------


def structure(data: Any, tp: Any, *, where: str = "") -> Any:
    """Decode plain data as a value of type ``tp`` (the inverse of :func:`unstructure`).

    Parameters
    ----------
    data
        Plain data, as :func:`unstructure` writes it (or a value of the type already).
    tp
        The declared type of the value, e.g. a field's annotation
        (``hmf.core.fits.FittingFunction``, ``float``, ``Quantity | None``, ...).
    where
        Where the value is in the configuration, for error messages.

    Returns
    -------
    Any
        The value. Data that the type's decoding does not apply to (e.g. a bare number
        for a Quantity) is returned as it is, for the field's converter and validator
        to accept or reject.

    Raises
    ------
    TypeError, ValueError
        For data that can't be a value of the type. The error notes ``where``.
    """
    try:
        return converter.structure(data, tp)
    except Exception as e:
        _note(e, where)
        raise


def _strict(kind: type, accept: tuple[type, ...], name: str) -> Callable[[Any, Any], Any]:
    def hook(data: Any, tp: Any) -> Any:
        if isinstance(data, accept) and not (kind is not bool and isinstance(data, bool)):
            return kind(data)
        raise TypeError(f"Expected {name}, not {type(data).__name__} ({data!r}).")

    return hook


def _structure_raw(data: Any, tp: Any) -> Any:
    return data


def _structure_optional(data: Any, tp: Any) -> Any:
    if _is_none(data):
        return None
    inner = _without_none(tp)
    return data if _is_union(inner) else converter.structure(data, inner)


def _structure_literal(data: Any, tp: Any) -> Any:
    options = typing.get_args(tp)
    if data in options and type(data) in {type(o) for o in options}:
        return data
    raise ValueError(f"Expected one of {options}, not {data!r}.")


def _structure_tuple(data: Any, tp: Any) -> Any:
    args = typing.get_args(tp)
    if isinstance(data, Mapping):
        data = list(data.items())
    if isinstance(data, (str, bytes)) or not isinstance(data, (list, tuple, np.ndarray)):
        return data
    if len(args) == 2 and args[1] is Ellipsis:
        return tuple(converter.structure(x, args[0]) for x in data)
    if args and len(args) == len(data):
        return tuple(converter.structure(x, a) for x, a in zip(data, args, strict=True))
    return tuple(data)


def _check_littleh(q: u.Quantity, where: str) -> u.Quantity:
    """A Quantity with littleh in its unit, converted to its canonical unit.

    :func:`~hmf.core.units.to_canonical` checks the power of littleh, as the units
    boundary does. A unit without littleh is returned as it is: it may be converted
    later with a stage's H0.
    """
    power = littleh_power(q.unit)
    if power == 0:
        return q
    physical = q.unit * littleh ** (-power)
    for unit in CANONICAL_UNITS.values():
        if physical.is_equivalent(unit * littleh ** (-littleh_power(unit))):
            values = to_canonical(q, unit, where="configuration", name=where or "value")
            return np.asarray(values) * unit
    return q


def _quantity(data: Mapping[str, Any], where: str) -> u.Quantity:
    unknown = set(data) - _QUANTITY_KEYS
    if unknown or "value" not in data or "unit" not in data:
        raise ValueError(
            f"A Quantity is written as {{value = ..., unit = '...'}} (and an optional dtype), "
            f"not with the keys {sorted(data)}."
        )
    dtype = np.dtype(data.get("dtype", "float64"))
    q = u.Quantity(np.asarray(data["value"], dtype=dtype), parse_unit(data["unit"]), dtype=dtype)
    return _check_littleh(q, where)


def _structure_quantity(data: Any, tp: Any) -> Any:
    if isinstance(data, u.Quantity):
        return _check_littleh(data, "")
    if isinstance(data, Mapping):
        return _quantity(data, "")
    return data  # e.g. a bare number: the field's converter raises UnitBoundaryError


def _structure_array(data: Any, tp: Any) -> Any:
    if isinstance(data, Mapping) and set(data) == {"value", "dtype"}:
        return np.asarray(data["value"], dtype=np.dtype(data["dtype"]))
    if isinstance(data, (list, tuple, int, float)) and not isinstance(data, bool):
        return np.asarray(data, dtype=np.float64)
    return data


def _structure_unit(data: Any, tp: Any) -> Any:
    if isinstance(data, Mapping) and set(data) == {"unit"}:
        return parse_unit(data["unit"])
    if isinstance(data, str):
        return parse_unit(data)
    return data


def _structure_untyped(data: Any, where: str = "") -> Any:
    """Decode data with no declared type: Quantities, arrays, None, and containers of these."""
    if _is_none(data):
        return None
    if isinstance(data, Mapping):
        if "value" in data and "unit" in data and set(data) <= _QUANTITY_KEYS:
            return _quantity(data, where)
        if set(data) == {"value", "dtype"}:
            return np.asarray(data["value"], dtype=np.dtype(data["dtype"]))
        return {k: _structure_untyped(v, f"{where}.{k}") for k, v in data.items()}
    if isinstance(data, list):
        return [_structure_untyped(v, where) for v in data]
    return data


def _structure_cosmology(data: Any, tp: Any) -> Any:
    if not isinstance(data, Mapping):
        return data  # the field's validator rejects it
    mapping = dict(data)
    if "class" not in mapping:
        raise ValueError("A cosmology table needs its 'class' (e.g. 'FlatLambdaCDM').")
    out: dict[str, Any] = {"cosmology": mapping.pop("class"), "name": mapping.pop("name", None)}
    meta = mapping.pop("meta", {})
    out.update({k: _structure_untyped(v, k) for k, v in mapping.items()})
    out["meta"] = _structure_untyped(meta, "meta")
    cosmology = Cosmology.from_format(out, format="mapping")
    if isinstance(tp, type) and not isinstance(cosmology, tp):
        raise TypeError(f"Expected a {tp.__name__}, not a {type(cosmology).__name__}.")
    return cosmology


def _build_attrs(cls: type, data: Mapping[str, Any], *, base: Any = None, where: str) -> Any:
    """An attrs value of ``cls`` from a table of its fields (changing ``base``, if given)."""
    fields = _fields(cls)
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        a = fields.get(key)
        if a is None:
            raise TypeError(
                f"{cls.__name__} (at {where or 'the top'!r}) has no field {key!r}."
                + _suggest(key, fields)
                + f" Fields: {list(fields)}."
            )
        here = f"{where}.{key}" if where else key
        current = getattr(base, a.name) if base is not None else _static_default(a)
        kwargs[key] = _structure_field(value, a, current, here)
    if base is not None:
        return attrs.evolve(base, **kwargs)
    return cls(**kwargs)


def _structure_field(data: Any, a: attrs.Attribute[Any], current: Any, where: str) -> Any:
    """Decode the value of a field; ``current`` is its value (or default) to change."""
    declared = _without_none(a.type)
    if (
        isinstance(data, Mapping)
        and _is_class(declared, Model)
        and "model" not in data
        and not _is_none(data)
    ):
        # A model table without its name changes the fields of the current model.
        if not isinstance(current, Model):
            raise ValueError(f"A {declared.__name__} table needs its 'model' (a registered name).")
        return _build_attrs(type(current), data, base=current, where=where)
    return structure(data, a.type, where=where)


def _structure_model(data: Any, tp: Any) -> Any:
    if isinstance(data, tp):
        return data
    if isinstance(data, (str, type)):
        return tp.get(data)()
    if isinstance(data, Mapping):
        try:
            # A table of a concrete model's fields needs no name.
            cls = tp.get(data.get("model", tp))
        except TypeError:
            if "model" in data:
                raise
            raise ValueError(
                f"A {tp.__name__} table needs its 'model' (a registered name)."
            ) from None
        return _build_attrs(cls, {k: v for k, v in data.items() if k != "model"}, where="")
    return data


def _structure_attrs(data: Any, tp: Any) -> Any:
    if not isinstance(data, Mapping):
        # A value already, or e.g. a mass definition's name: the field's converter reads it.
        return data
    cls = tp
    fields = dict(data)
    if "class" in fields:
        cls = _import_class(fields.pop("class"), tp, "class")
    if not attrs.has(cls):
        raise TypeError(
            f"{qualified_name(cls)} can't be read from a table: it is not an attrs class."
        )
    if issubclass(cls, Stage):
        return _build_stage(cls, fields)
    if issubclass(cls, Model):
        return _structure_model({**fields, "model": qualified_name(cls)}, cls)
    return _build_attrs(cls, fields, where="")


def _structure_class(data: Any, tp: Any) -> Any:
    """A value of a declared class that is not attrs (e.g. a protocol): needs a 'class'."""
    if isinstance(data, Mapping) and "class" in data:
        return _structure_attrs(data, tp)
    return data


def _is_tuple(tp: Any) -> bool:
    origin: Any = typing.get_origin(tp)
    return origin is tuple or tp is tuple


def _is_literal(tp: Any) -> bool:
    origin: Any = typing.get_origin(tp)
    return origin is Literal


def _is_ndarray(tp: Any) -> bool:
    return _is_class(tp, np.ndarray) or typing.get_origin(tp) is np.ndarray


converter.register_structure_hook(bool, _strict(bool, (bool,), "a boolean"))
converter.register_structure_hook(int, _strict(int, (int, np.integer), "an integer"))
converter.register_structure_hook(
    float, _strict(float, (int, float, np.integer, np.floating), "a number")
)
converter.register_structure_hook(str, _strict(str, (str,), "a string"))
converter.register_structure_hook_func(lambda t: True, _structure_raw)
converter.register_structure_hook_func(lambda t: isinstance(t, type), _structure_class)
converter.register_structure_hook_func(attrs.has, _structure_attrs)
converter.register_structure_hook_func(lambda t: _is_class(t, Model), _structure_model)
converter.register_structure_hook_func(lambda t: _is_class(t, Cosmology), _structure_cosmology)
converter.register_structure_hook_func(
    lambda t: _is_class(t, PurePath), lambda d, t: t(d) if isinstance(d, str) else d
)
converter.register_structure_hook_func(
    lambda t: _is_class(t, Enum), lambda d, t: d if isinstance(d, t) else t(d)
)
converter.register_structure_hook_func(_is_ndarray, _structure_array)
converter.register_structure_hook_func(lambda t: _is_class(t, u.UnitBase), _structure_unit)
converter.register_structure_hook_func(lambda t: _is_class(t, u.Quantity), _structure_quantity)
converter.register_structure_hook_func(_is_tuple, _structure_tuple)
converter.register_structure_hook_func(_is_literal, _structure_literal)
converter.register_structure_hook_func(_is_union, _structure_optional)


# ---------------------------------------------------------------------------------
# Stage trees
# ---------------------------------------------------------------------------------


def _set_model_field(model: Any, rest: tuple[str, ...], data: Any, where: str) -> Any:
    """``model`` with the field at the path ``rest`` below it set from ``data``."""
    if not attrs.has(type(model)):
        raise TypeError(f"{where!r}: the value there is a {type(model).__name__}, not a model.")
    fields = _fields(type(model))
    a = fields.get(rest[0])
    if a is None:
        raise TypeError(
            f"{type(model).__name__} (at {where!r}) has no field {rest[0]!r}."
            + _suggest(rest[0], fields)
            + f" Fields: {list(fields)}."
        )
    current = getattr(model, a.name)
    if len(rest) == 1:
        new = _structure_field(data, a, current, where)
    else:
        new = _set_model_field(current, rest[1:], data, where)
    return attrs.evolve(model, **{rest[0]: new})


def _build_stage(cls: type[Stage], body: Mapping[str, Any]) -> Stage:
    """Build a stage tree from the body of a configuration (everything but the header)."""
    table = routing.router(cls)
    values: dict[tuple[str, ...], Any] = {}
    shared: set[tuple[str, ...]] = set()
    origin: dict[tuple[str, ...], str] = {}
    model_changes: dict[tuple[str, ...], list[tuple[tuple[str, ...], Any, str]]] = {}

    def assign(path: tuple[str, ...], value: Any, name: str, soft: bool) -> None:
        if path in values:
            if soft and path not in shared:
                return  # set explicitly: a shared name does not override it
            if not (path in shared and not soft):
                raise TypeError(
                    f"{cls.__name__}: {'.'.join(path)!r} is given twice, as "
                    f"{origin[path]!r} and as {name!r}."
                )
            shared.discard(path)
        values[path] = value
        origin[path] = name
        if soft:
            shared.add(path)

    def decode(path: tuple[str, ...], data: Any, name: str) -> Any:
        if path in table.leaves:
            a = table.leaves[path]
            current = _static_default(a)
            return _structure_field(data, a, None if current is NO_DEFAULT else current, name)
        return structure(data, table.stages[path] | None, where=name)

    def visit(mapping: Mapping[str, Any], prefix: tuple[str, ...]) -> None:
        for key, data in mapping.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"{cls.__name__}: a configuration key must be a string, not {key!r}."
                )
            path = (*prefix, *key.split("."))
            name = ".".join(path)
            if (
                path
                and path in table.stages
                and isinstance(data, Mapping)
                and not _is_none(data)
                and "class" not in data
            ):
                visit(data, path)
                continue
            paths, rest = table.paths_of(name)
            if rest:
                model_changes.setdefault(paths[0], []).append((rest, data, name))
                continue
            soft = len(paths) > 1
            for p in paths:
                assign(p, decode(p, data, name), name, soft)

    visit(body, ())
    for path, changes in model_changes.items():
        if path in values:
            model = values[path]
        else:
            model = _static_default(table.leaves[path])
            if model is NO_DEFAULT:
                raise TypeError(
                    f"{cls.__name__}: {changes[0][2]!r} changes a field of "
                    f"{'.'.join(path)!r}, which has no default: give it."
                )
        for rest, data, name in changes:
            model = _set_model_field(model, rest, data, name)
        values[path] = model
    return routing.from_paths(
        cls, {".".join(p): v for p, v in values.items()}, shared=[".".join(p) for p in shared]
    )


def _stage_class(name: Any) -> type[Stage]:
    if not isinstance(name, str):
        raise ValueError(  # noqa: TRY004 (a configuration error: see hmf.core.domain)
            "The configuration's [hmf] table must give the class of its stage, e.g. "
            "stage = 'hmf.core.mass_function:MassFunction'."
        )
    found = _import_from_path(name)
    if not (isinstance(found, type) and issubclass(found, Stage)):
        raise ValueError(  # noqa: TRY004 (a configuration error: see hmf.core.domain)
            f"The configuration's stage {name!r} is not a stage class."
        )
    return found


def _document(stage: Stage, mode: _Mode, *, versions: bool) -> dict[str, Any]:
    """The configuration of a stage tree, in a mode."""
    if not isinstance(stage, Stage):
        raise TypeError(f"Expected a stage, not a {type(stage).__name__}.")
    token = _mode.set(mode)
    dropped: list[str] = []
    dropped_token = _dropped_meta.set(dropped)
    try:
        header: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "stage": qualified_name(type(stage)),
        }
        if versions:
            names = HASHED_PACKAGES if mode == "hash" else PACKAGES
            installed = package_versions()
            header["versions"] = {
                n: installed[n] for n in names if installed[n] is not None or mode != "toml"
            }
        body = _unstructure_attrs(stage)
    finally:
        _mode.reset(token)
        _dropped_meta.reset(dropped_token)
    if dropped:
        _warn_dropped_meta(dropped)
    if HEADER in body:
        raise TypeError(f"{type(stage).__name__} has a field {HEADER!r}, the header's key.")
    return {HEADER: header, **body}


def to_config(stage: Stage, *, versions: bool = True) -> dict[str, Any]:
    """The configuration of a stage tree: every field's resolved value, and the header.

    Parameters
    ----------
    stage
        The stage at the root of the tree.
    versions
        Whether to record the versions of the packages (:func:`package_versions`).

    Returns
    -------
    dict
        The configuration (see the module documentation):
        ``from_config(to_config(stage))`` equals ``stage``.

    Raises
    ------
    TypeError
        If a field's value can't be written in a configuration.
    """
    return _document(stage, "config", versions=versions)


def _upgrade(document: Mapping[str, Any], *, migrate: bool) -> dict[str, Any]:
    """The configuration, migrated to this schema version (a copy)."""
    header = document.get(HEADER)
    if not isinstance(header, Mapping):
        raise ValueError(  # noqa: TRY004 (a configuration error: see hmf.core.domain)
            "This is not an hmf configuration: it has no [hmf] table with its "
            "schema_version and stage."
        )
    unknown = set(header) - {"schema_version", "stage", "versions"}
    if unknown:
        name = min(unknown)
        raise TypeError(
            f"The configuration's [hmf] table has no key {name!r}."
            + _suggest(name, ["schema_version", "stage", "versions"])
        )
    version = header.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise ValueError(  # noqa: TRY004 (a configuration error: see hmf.core.domain)
            f"The configuration's schema_version must be an integer, not {version!r}."
        )
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"The configuration is of schema version {version}, newer than this version "
            f"of hmf reads ({SCHEMA_VERSION}): upgrade hmf to read it."
        )
    out = copy.deepcopy(dict(document))
    if version == SCHEMA_VERSION:
        return out
    if not migrate:
        raise ValueError(
            f"The configuration is of schema version {version}, not {SCHEMA_VERSION}: "
            "read it with migrate=True to migrate it."
        )
    for n in range(version, SCHEMA_VERSION):
        migration = MIGRATIONS.get(n)
        if migration is None:
            raise ValueError(
                f"The configuration is of schema version {version}, which this version "
                f"of hmf can't read: there is no migration from schema {n}."
            )
        out = migration.function(out)
        out[HEADER] = {**out[HEADER], "schema_version": n + 1}
        logger.info(
            "Migrated the configuration from schema %d to %d: %s",
            n,
            n + 1,
            migration.description,
        )
    return out


def _version_text(version: Any) -> str:
    """A package version for a message: quoted (so a crafted one can't fake a line)."""
    return "not installed" if version is None or _is_none(version) else repr(version)


def _check_versions(written: Any, *, strict: bool) -> None:
    """Log (or raise, if strict) the package versions that differ from the running ones."""
    if written is None:
        return
    if not isinstance(written, Mapping):
        raise ValueError(  # noqa: TRY004 (a configuration error: see hmf.core.domain)
            "The configuration's [hmf.versions] must be a table."
        )
    running = package_versions()
    differ = []
    for name, then in written.items():
        now = running.get(name)
        if name in running and (None if _is_none(then) else then) != now:
            differ.append(f"{name} {_version_text(then)} (now {_version_text(now)})")
    if not differ:
        return
    message = (
        "The configuration was written with "
        + ", ".join(differ)
        + ": its results may differ in this environment."
    )
    if strict:
        raise ValueError(message)
    logger.warning(message)


def from_config(
    mapping: Mapping[str, Any],
    *,
    migrate: bool = True,
    strict: bool = False,
    stage: type[Stage] | None = None,
) -> Stage:
    """Build a stage tree from a configuration.

    Parameters
    ----------
    mapping
        A configuration: as :func:`to_config` writes it, or partial (see the module
        documentation). It is not modified.
    migrate
        Whether to migrate a configuration of an older schema version (logging what
        each migration changes). If False, an older one raises.
    strict
        Whether package versions that differ from the running ones raise, rather than
        being logged.
    stage
        The class of the root stage, if the header does not give it. If both do, they
        must agree.

    Returns
    -------
    Stage

    Raises
    ------
    ValueError
        For a configuration without a header, of a newer or unknown schema version
        (or an older one, with ``migrate=False``), with other package versions (with
        ``strict=True``), or with an invalid value (e.g. an unknown model, with
        suggestions).
    TypeError
        For an unknown, ambiguous or repeated parameter (with suggestions).
    astropy.units.UnitConversionError
        For a Quantity in the wrong unit, or with the wrong power of littleh.
    """
    document = _upgrade(mapping, migrate=migrate)
    header = document.pop(HEADER)
    _check_versions(header.get("versions"), strict=strict)
    cls = _stage_class(header["stage"]) if "stage" in header else stage
    if cls is None:
        _stage_class(None)
    assert cls is not None
    if stage is not None and cls is not stage:
        raise ValueError(f"The configuration is of a {cls.__name__}, not a {stage.__name__}.")
    return _build_stage(cls, document)


# ---------------------------------------------------------------------------------
# TOML
# ---------------------------------------------------------------------------------

_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")


def _toml_key(key: str) -> str:
    return key if _BARE_KEY.fullmatch(key) else json.dumps(key, ensure_ascii=False)


#: The longest line of an inline table.
_MAX_INLINE = 100


def _is_inline(value: Mapping[str, Any]) -> bool:
    """Whether a table is written inline: a short encoded value rather than a section."""
    if not (not value or _is_none(value) or ("value" in value and set(value) <= _QUANTITY_KEYS)):
        return False
    return len(_toml_value(value, inline=True)) <= _MAX_INLINE


def _toml_value(value: Any, *, inline: bool = False) -> str:
    """A value as TOML (on one line if ``inline``, as inside an inline table)."""
    if isinstance(value, Mapping):
        if not value:
            return "{}"
        items = ", ".join(
            f"{_toml_key(k)} = {_toml_value(v, inline=True)}" for k, v in value.items()
        )
        return f"{{ {items} }}"
    if isinstance(value, (list, tuple)) and (inline or any(isinstance(v, Mapping) for v in value)):
        return "[" + ", ".join(_toml_value(v, inline=True) for v in value) + "]"
    return tomli_w.dumps({"x": value}).removeprefix("x = ").rstrip("\n")


def _toml(document: Mapping[str, Any]) -> str:
    lines: list[str] = []

    def section(name: str, table: Mapping[str, Any]) -> None:
        tables = [(k, v) for k, v in table.items() if isinstance(v, Mapping) and not _is_inline(v)]
        values = [
            (k, v) for k, v in table.items() if not (isinstance(v, Mapping) and not _is_inline(v))
        ]
        if name and (values or not tables):
            lines.extend(["", f"[{name}]"])
        lines.extend(f"{_toml_key(k)} = {_toml_value(v)}" for k, v in values)
        for k, v in tables:
            section(f"{name}.{_toml_key(k)}" if name else _toml_key(k), v)

    section("", document)
    return "\n".join(lines).lstrip("\n") + "\n"


def dumps_toml(stage: Stage, *, versions: bool = True) -> str:
    """The configuration of a stage tree as TOML text.

    Parameters
    ----------
    stage
        The stage at the root of the tree.
    versions
        Whether to record the versions of the packages.

    Returns
    -------
    str
        TOML, which :func:`loads_toml` reads back to an equal stage.
    """
    document = _document(stage, "toml", versions=versions)
    head = (
        f"# The configuration of an hmf.core {type(stage).__name__} "
        f"(schema version {SCHEMA_VERSION}): see hmf.core.config.\n"
    )
    return head + _toml(document)


def loads_toml(text: str, **kwargs: Any) -> Stage:
    """Build a stage tree from a configuration in TOML text.

    Parameters
    ----------
    text
        The TOML.
    **kwargs
        Passed on to :func:`from_config` (``migrate``, ``strict``, ``stage``).

    Returns
    -------
    Stage
    """
    return from_config(tomllib.loads(text), **kwargs)


def dump_toml(stage: Stage, path: str | PathLike[str], *, versions: bool = True) -> Path:
    """Write the configuration of a stage tree to a TOML file.

    Parameters
    ----------
    stage
        The stage at the root of the tree.
    path
        The file to write.
    versions
        Whether to record the versions of the packages.

    Returns
    -------
    pathlib.Path
        The file written.
    """
    out = Path(path)
    text = dumps_toml(stage, versions=versions)
    # Write a temporary file next to it and rename it, so an existing file is replaced
    # whole or not at all.
    fd, tmp = tempfile.mkstemp(dir=out.parent, prefix=f".{out.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        Path(tmp).replace(out)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return out


def load_toml(path: str | PathLike[str], **kwargs: Any) -> Stage:
    """Build a stage tree from a TOML configuration file.

    Parameters
    ----------
    path
        The file.
    **kwargs
        Passed on to :func:`from_config` (``migrate``, ``strict``, ``stage``).

    Returns
    -------
    Stage
    """
    with Path(path).open("rb") as f:
        return from_config(tomllib.load(f), **kwargs)


# ---------------------------------------------------------------------------------
# Content hashes and pickling
# ---------------------------------------------------------------------------------


def canonical_value(value: Any) -> Any:
    """The plain data a content hash of ``value`` is computed from.

    A stage gives its configuration without the fields that change no result (and
    without a cosmology's name and ``meta``), with the versions of
    :data:`HASHED_PACKAGES`; any other value its encoding (:func:`unstructure`), with
    the same omissions. :func:`hmf.core._serialise.canonical` calls this for every
    value that is not plain data.

    Parameters
    ----------
    value
        A stage, model, Quantity, array, cosmology, unit or setting.

    Returns
    -------
    Any
        Nested dicts, lists, numbers, strings, booleans and None.
    """
    if isinstance(value, Stage):
        return _document(value, "hash", versions=True)
    token = _mode.set("hash")
    try:
        return unstructure(value)
    finally:
        _mode.reset(token)


def content_hash(stage: Stage) -> str:
    """The content hash of a stage tree's physics (see the module documentation).

    Parameters
    ----------
    stage
        The stage at the root of the tree.

    Returns
    -------
    str
        A SHA-256 hex digest.
    """
    from ._serialise import content_hash as _hash

    if not isinstance(stage, Stage):
        raise TypeError(f"Expected a stage, not a {type(stage).__name__}.")
    return _hash(stage)


def _computed_runs(stage: Stage) -> list[BoltzmannRun]:
    """The Boltzmann runs the solutions already computed in a stage tree came from."""
    runs: list[BoltzmannRun] = []
    seen: set[int] = set()

    def computed(obj: Any, name: str) -> Any:
        descriptor = getattr(type(obj), name, None)
        if descriptor is None:
            return None
        try:
            return descriptor.__get__(obj, type(obj))
        except AttributeError:  # an empty cached_property slot: not computed
            return None

    def walk(obj: Any) -> None:
        if id(obj) in seen or not attrs.has(type(obj)) or isinstance(obj, Model):
            return
        seen.add(id(obj))
        for a in attrs.fields(type(obj)):
            walk(getattr(obj, a.name))
        for name in getattr(type(obj), "_derived_properties", {}):
            walk(computed(obj, name))
        solution = computed(obj, "solution") if isinstance(obj, Stage) else None
        run = getattr(solution, "run", None)
        if isinstance(run, BoltzmannRun) and all(run is not r for r in runs):
            runs.append(run)

    walk(stage)
    return runs


def _pickle_state(stage: Stage) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """What a stage pickles as: its configuration, and the runs its tree has used."""
    return to_config(stage), export_runs(_computed_runs(stage))


def _unpickle(config: Mapping[str, Any], runs: Mapping[str, Mapping[str, Any]]) -> Stage:
    """Rebuild a pickled stage (see :meth:`Stage.__reduce__ <hmf.core.stage.Stage.__reduce__>`)."""
    import_runs(runs)
    return from_config(config)
