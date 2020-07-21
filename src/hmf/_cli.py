"""Module that contains the command line app."""
import warnings

import click
import numpy as np
import toml
from pathlib import Path
import datetime
import importlib

import hmf
from hmf.helpers.functional import get_hmf


def _get_config(config=None):
    if config is None:
        return {}

    with open(config, "r") as fl:
        cfg = toml.load(fl)

    # Import an actual framework.
    fmwk = cfg.get("framework", None)
    if fmwk:
        cfg["framework"] = importlib.import_module(fmwk)

    return cfg


def _ctx_to_dct(args):
    dct = {}
    j = 0
    while j < len(args):
        arg = args[j]
        if "=" in arg:
            a = arg.split("=")
            k = a[0].replace("--", "")
            v = a[-1]
            j += 1
        else:
            k = arg.replace("--", "")
            v = args[j + 1]
            j += 2

        try:
            # For most arguments, this will convert it to the right type.
            v = eval(v)
        except NameError:
            # If it's supposed to be a string, but quotes weren't supplied.
            v = eval('"' + v + '"')

        dct[k] = v

    return dct


#
# def _update(obj, ctx):
#     # Try to use the extra arguments as an override of config.
#     kk = list(ctx.keys())
#     for k in kk:
#         if hasattr(obj, k):
#             try:
#                 given_val = ctx.pop(k)
#                 setattr(obj, k, eval(given_val))
#             except AttributeError:
#                 raise AttributeError(f"Parameter {k} is not a valid parameter to {obj.__class__.__name__}.")
#             except TypeError:
#                 raise TypeError(f"For parameter '{k}', given value {given_val} is not of a valid type.")
#             except Exception:
#                 raise ValueError(f"Something went wrong when specifying parameter {k} with val of {given_val}.")
#
#
# def _override(ctx, obj):
#     # Try to use the extra arguments as an override of config.
#
#     if ctx.args:
#         ctx = _ctx_to_dct(ctx.args)
#         _update(obj, ctx)
#
#         if ctx:
#             warnings.warn(f"The following arguments were not able to be set: {ctx}")


main = click.Group()


@main.command(
    context_settings={  # Doing this allows arbitrary options to override config
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
)
@click.option(
    "-i", "--config", type=click.Path(exists=True, dir_okay=False), default=None,
)
@click.option(
    "-o",
    "--outdir",
    type=click.Path(exists=True, dir_okay=True, file_okay=True),
    default=".",
)
@click.pass_context
def run(ctx, config, outdir):
    """Calculate quantities using hmf and output to a file.

    Parameters
    ----------
    ctx :
        A parameter from the parent CLI function to be able to override config.
    config : str
        Path to the configuration file.
    """
    cfg = _get_config(config)

    # Update the file-based config with options given on the CLI.
    if ctx.args:
        cfg.update(_ctx_to_dct(ctx.args))

    out = get_hmf(
        cfg.get("quantities", ["m", "dndm"]),
        framework=cfg.get("framework", hmf.MassFunction),
        get_label=True,
        **cfg.get("params", {}),
    )

    outdir = Path(outdir)

    for quantities, obj, label in out:
        # Write out quantities
        for qname, q in zip(cfg.get("quantities"), quantities):
            np.savetxt(outdir / f"{label}_{qname}.txt", q)

        # Write out parameters
        with open(outdir / f"{label}_cfg.toml") as fl:
            fl.write(f"# File Created On: {datetime.datetime.now()}\n")
            fl.write("# With version {hmf.__version__} of hmf\n")

            fl.write("\n")

            fl.write(f"quantities = {toml.dumps(cfg.get('quantities'))}\n")

            for k, v in list(obj.parameter_values.items()):
                fl.write(f"{k} = {v} \n")
