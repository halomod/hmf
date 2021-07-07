"""Utilities for interacting with hmf TOML configs."""
from astropy.units import Quantity
from datetime import datetime
from inspect import signature

from hmf._internals._framework import Framework

from .. import __version__


def framework_to_dict(obj: Framework) -> dict:
    """Serialize a framework instance to a simple TOML-able dictionary."""

    out = {"created_on": datetime.now(), "hmf_version": __version__, "params": {}}

    for k, v in obj.parameter_values.items():
        if k == "cosmo_model":
            out["params"][k] = v.name
        elif k == "cosmo_params":
            params = {}
            for key in signature(obj.cosmo.__init__).parameters.keys():
                val = getattr(obj.cosmo, key)
                if isinstance(val, Quantity):
                    val = {"value": val.value, "unit": str(val.unit)}
                params[key] = val

            out["params"][k] = params

        elif k.endswith("_model"):
            # Model components should just be the name of the class, not a
            # full class __repr__, and also, we give the actual model, not the input
            # parameter.
            val = getattr(obj, k)
            if val is None:
                obj_val = getattr(obj, k.split("_model")[0])
                out["params"][k] = (
                    None if obj_val is None else obj_val.__class__.__name__
                )
            else:
                out["params"][k] = val.__name__

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
