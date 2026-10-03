"""Utilities for interacting with hmf TOML configs."""

from datetime import UTC, datetime
from inspect import signature

from astropy.units import Quantity

from hmf._internals._framework import Framework

from .. import __version__


def framework_to_dict(obj: Framework) -> dict:
    """Serialize a framework instance to a simple TOML-able dictionary."""
    out = {"created_on": datetime.now(tz=UTC), "hmf_version": __version__, "params": {}}

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

    return out
