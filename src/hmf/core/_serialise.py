"""Canonical serialisation and content hashes of hmf.core inputs.

The disk cache of Boltzmann-code output (:mod:`hmf.core.cache`) is keyed by a content
hash of the code's input. This module defines the canonical serialisation that hash
is computed from: a JSON document with sorted keys and shortest round-trip floats, so
that equal inputs always give the same text, in any process and on any platform.

It also defines :func:`cosmology_key`, the value that a stage's ``cosmology`` field
is compared and hashed by: astropy cosmologies compare by value, but are not
hashable.

This is deliberately small. The config schema (issue #391) will serialise models and
stages through cattrs; when it does, :func:`canonical` should be routed through it.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import PurePath
from typing import Any

import astropy.units as u
import attrs
import numpy as np
from astropy.cosmology import FLRW

from .model import qualified_name

__all__ = ["canonical", "canonical_json", "content_hash", "cosmology_key"]


def _plain(value: Any) -> Any:
    """Convert a parameter value to plain Python numbers (and unit strings)."""
    if isinstance(value, u.Quantity):
        return (_plain(value.value), value.unit.to_string())
    if isinstance(value, np.ndarray):
        return tuple(_plain(v) for v in value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return tuple(_plain(v) for v in value)
    return value


def cosmology_key(cosmo: FLRW) -> tuple[Any, ...]:
    """The physical content of an astropy cosmology, as a hashable tuple.

    It holds the cosmology's class and the values of its parameters (H0, Om0, Ode0,
    Tcmb0, Neff, m_nu, Ob0, w0, ...), but not its ``name`` or ``meta``, which do not
    change any result: ``Planck18`` and an unnamed copy of it have the same key.

    Parameters
    ----------
    cosmo
        An astropy FLRW cosmology.

    Returns
    -------
    tuple
        ``(qualified class name, ((parameter, value), ...))``, with values as floats,
        tuples of floats, and ``(value, unit string)`` pairs for Quantities.
    """
    params = getattr(cosmo, "parameters", None)
    if not isinstance(params, Mapping):  # pragma: no cover - astropy < 6.1
        params = {name: getattr(cosmo, name) for name in type(cosmo).__parameters__}
    return (
        qualified_name(type(cosmo)),
        tuple(sorted((str(name), _plain(value)) for name, value in params.items())),
    )


def canonical(obj: Any) -> Any:
    """Convert ``obj`` to a JSON-compatible structure that identifies its content.

    Parameters
    ----------
    obj
        ``None``, a bool, int, float, str or path; a numpy scalar or array; an
        astropy Quantity or FLRW cosmology; an ``attrs`` instance (a model, an
        accuracy object, ...: its class and the fields that take part in its
        equality); or a mapping, list or tuple of these.

    Returns
    -------
    Any
        Nested dicts, lists, strings, numbers, bools and ``None``.

    Raises
    ------
    TypeError
        If ``obj`` (or something inside it) can not be serialised canonically.
    ValueError
        For a NaN or infinite float.
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"Can't serialise the non-finite float {obj} canonically.")
        return obj
    if isinstance(obj, PurePath):
        return str(obj)
    if isinstance(obj, np.generic):
        return canonical(obj.item())
    if isinstance(obj, u.Quantity):  # before ndarray, which it subclasses
        return {"__quantity__": canonical(obj.value), "unit": obj.unit.to_string()}
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": canonical(obj.tolist()), "dtype": obj.dtype.str}
    if isinstance(obj, u.UnitBase):
        return {"__unit__": obj.to_string()}
    if isinstance(obj, FLRW):
        return {"__cosmology__": canonical(cosmology_key(obj))}
    if attrs.has(type(obj)):
        cls = type(obj)
        out: dict[str, Any] = {"__class__": qualified_name(cls)}
        for a in attrs.fields(cls):
            if a.eq:
                out[a.name] = canonical(getattr(obj, a.name))
        return out
    if isinstance(obj, Mapping):
        keys = [str(k) for k in obj]
        if len(set(keys)) != len(keys):
            raise TypeError("Can't serialise a mapping whose keys collide as strings.")
        return {str(k): canonical(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [canonical(v) for v in obj]
    raise TypeError(f"Can't serialise a {type(obj).__name__} canonically.")


def canonical_json(obj: Any) -> str:
    """The canonical JSON text of ``obj`` (see :func:`canonical`).

    Keys are sorted, there is no whitespace, and floats are written as their
    shortest round-trip representation.
    """
    return json.dumps(canonical(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(obj: Any) -> str:
    """The SHA-256 hex digest of :func:`canonical_json` of ``obj``."""
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()
