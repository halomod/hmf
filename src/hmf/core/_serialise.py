"""Canonical serialisation and content hashes of hmf.core inputs.

The disk cache of Boltzmann-code output (:mod:`hmf.core.cache`) is keyed by a content
hash of the code's input. This module defines the canonical serialisation that hash
is computed from: a JSON document with sorted keys and shortest round-trip floats, so
that equal inputs always give the same text, in any process and on any platform.

It also defines the values that fields holding astropy objects are compared and
hashed by, as ``field(eq=key)``: :func:`cosmology_key` for a cosmology, and
:func:`quantity_key` for a Quantity. Astropy cosmologies compare by value but are
not hashable, and Quantities compare elementwise.

Plain data (numbers, strings, booleans, None, paths, and lists and mappings of these)
is serialised here. Every other value (a stage, a model, a Quantity, an array, a
cosmology, ...) is encoded by the converter of the configurations
(:func:`hmf.core.config.canonical_value`), so a content hash and a configuration
describe a value in the same way. A stage is encoded as its configuration without the
fields that change no result, with the versions of the packages that compute results.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import PurePath
from typing import Any

import astropy.units as u
import numpy as np
from astropy.cosmology import FLRW

from .model import qualified_name

__all__ = ["canonical", "canonical_json", "content_hash", "cosmology_key", "quantity_key"]


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


def quantity_key(value: Any) -> Any:
    """The value of a Quantity, as a hashable tuple: what a Quantity field compares by.

    This is the one comparison of Quantity fields in :mod:`hmf.core`, used as
    ``field(eq=quantity_key)`` (as :func:`hmf.core.units.quantity_field` does). Two
    Quantities are equal if they have the same unit, shape and values. A field of
    :func:`~hmf.core.units.quantity_field` is stored in its canonical unit, so it
    compares by its value in that unit, whatever unit it was given in.

    Parameters
    ----------
    value
        A Quantity, or ``None`` (which is its own key).

    Returns
    -------
    tuple or None
        ``(unit string, shape, values as a tuple of Python numbers)``.
    """
    if value is None:
        return None
    if not isinstance(value, u.Quantity):
        raise TypeError(f"quantity_key: expected a Quantity, not {type(value).__name__}.")
    return (value.unit.to_string(), value.shape, tuple(np.ravel(value.value).tolist()))


def canonical(obj: Any) -> Any:
    """Convert ``obj`` to a JSON-compatible structure that identifies its content.

    Parameters
    ----------
    obj
        ``None``, a bool, int, float, str or path; a numpy scalar; a mapping, list or
        tuple of these; or any value the configurations encode (a stage, model,
        Quantity, array, cosmology, unit, accuracy setting, ...: see
        :func:`hmf.core.config.canonical_value`).

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
    if isinstance(obj, Mapping):
        keys = [str(k) for k in obj]
        if len(set(keys)) != len(keys):
            raise TypeError("Can't serialise a mapping whose keys collide as strings.")
        return {str(k): canonical(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [canonical(v) for v in obj]
    from . import config  # imports every stage: not at import time

    return canonical(config.canonical_value(obj))


def canonical_json(obj: Any) -> str:
    """The canonical JSON text of ``obj`` (see :func:`canonical`).

    Keys are sorted, there is no whitespace, and floats are written as their
    shortest round-trip representation.
    """
    return json.dumps(canonical(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(obj: Any) -> str:
    """The SHA-256 hex digest of :func:`canonical_json` of ``obj``."""
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()
