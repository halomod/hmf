"""Utilities for interacting with hmf TOML configs."""
from hmf._internals._framework import Framework
from datetime import datetime
from .. import __version__
from inspect import signature
from astropy.units import Quantity
import toml


def framework_to_dict(obj: Framework) -> dict:
    """Serialize a framework instance to a simple TOML-able dictionary."""

    out = {"created_on": datetime.now(), "hmf_version": __version__}

    for k, v in obj.parameter_values.items():
        if k == "cosmo_model":
            out[k] = v.name
        elif k == "cosmo_params":
            params = {}
            for key in signature(obj.cosmo.__init__).parameters.keys():
                val = getattr(obj.cosmo, key)
                if isinstance(val, Quantity):
                    val = (val.value, str(val.unit))
                params[key] = val

            out[k] = params

        elif k.endswith("_model"):
            # Model components should just be the name of the class, not a
            # full class __repr__, and also, we give the actual model, not the input
            # parameter.
            out[k] = getattr(obj, k.split("_model")[0]).__class__.__name__
        elif k.endswith("_params"):
            if k == "transfer_params" and obj.transfer_model.__name__ == "CAMB":
                # Special case CAMB because its params are weird.
                out[k] = v
            else:
                out[k] = getattr(obj, k.split("_params")[0]).params
        else:
            out[k] = v

    return out
