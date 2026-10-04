"""Utilities for interacting with hmf TOML configs."""

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, time
from inspect import signature
from typing import Any

import numpy as np
from astropy.units import Quantity

from hmf._internals._framework import Framework

from .. import __version__


def _plugin_module(model: type) -> str | None:
    """The module to list under ``plugins`` so that ``model`` can be found by name.

    Returns None for models defined in hmf itself (always registered), and for
    models defined in ``__main__`` (which can not be imported by name).
    """
    mod = model.__module__
    if mod == "__main__" or mod == "hmf" or mod.startswith("hmf."):
        return None
    return mod


def framework_to_dict(obj: Framework, plugins: Sequence[str] = ()) -> dict:
    """Serialize a framework instance to a simple TOML-able dictionary.

    Parameters
    ----------
    obj
        The framework instance to serialize.
    plugins
        Modules to list under the top-level ``plugins`` key of the output, e.g. the
        ``plugins`` of the config ``obj`` was made from. The modules defining any
        models of ``obj`` that are not part of hmf are added to these, so that
        the models (which are written by their class name) can be found again
        when the config is read back.

    Returns
    -------
    dict
        The config. It has a ``plugins`` key only if there are any plugins.
    """
    out = {"created_on": datetime.now(tz=UTC), "hmf_version": __version__, "params": {}}
    plugins = list(plugins)

    for k, v in obj.parameter_values.items():
        if k == "cosmo_model":
            out["params"][k] = v.name
        elif k == "cosmo_params":
            params = {}
            for key in signature(obj.cosmo.__init__).parameters:
                val = getattr(obj.cosmo, key)
                if isinstance(val, Quantity):
                    val = {"value": val.value, "unit": str(val.unit)}
                if key == "meta":
                    continue

                params[key] = val

            out["params"][k] = params

        elif k.endswith("_model"):
            # Model components should just be the name of the class, not a
            # full class __repr__. A model left unset (None) is written as unset
            # rather than as the model it resolved to: e.g. an unset mdef_model means
            # "the fit's own definition", which is not the same as naming that
            # definition explicitly (see #374). TOML has no null, so the key is
            # dropped and reloads as the default.
            val = getattr(obj, k)
            out["params"][k] = None if val is None else val.__name__
            if val is not None and (mod := _plugin_module(val)) and mod not in plugins:
                plugins.append(mod)

        elif k.endswith("_params"):
            if k == "transfer_params" and obj.transfer_model.__name__ == "CAMB":
                # Special case CAMB because its params are weird.
                out["params"][k] = v
            else:
                try:
                    out["params"][k] = getattr(obj, k.split("_params")[0]).params
                except AttributeError:
                    out["params"][k] = None
        else:
            out["params"][k] = v

    if plugins:
        out["plugins"] = plugins

    return out


def to_toml_compatible(obj: Any) -> Any:
    """Convert a config (e.g. from :func:`framework_to_dict`) to TOML-writable types.

    TOML has no null, so ``None`` values in tables are dropped, and reload as the
    default. NumPy scalars and arrays become Python scalars and lists. Any other
    object that TOML can not represent is written as its ``str``.

    Parameters
    ----------
    obj
        The config, or a value in it.

    Returns
    -------
    Any
        A copy of ``obj`` that ``tomli_w`` can write.
    """
    if isinstance(obj, Mapping):
        return {str(k): to_toml_compatible(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, np.ndarray | np.generic):
        return to_toml_compatible(obj.tolist())
    if isinstance(obj, list | tuple):
        return [to_toml_compatible(v) for v in obj]
    if isinstance(obj, str | bool | int | float | date | time):
        return obj
    return str(obj)
