"""Module that contains the command line app."""
import warnings

import click
import numpy as np
import toml
from pathlib import Path
import datetime

import hmf
from hmf.helpers.functional import get_hmf


def _get_config(config=None):
    if config is None:
        config = {}

    with open(config, "r") as f:
        cfg = toml.load(f)

    return cfg


def _ctx_to_dct(args):
    dct = {}
    j = 0
    while j < len(args):
        arg = args[j]
        if "=" in arg:
            a = arg.split("=")
            dct[a[0].replace("--", "")] = a[-1]
            j += 1
        else:
            dct[arg.replace("--", "")] = args[j + 1]
            j += 2

    return dct


def _update(obj, ctx):
    # Try to use the extra arguments as an override of config.
    kk = list(ctx.keys())
    for k in kk:
        # noinspection PyProtectedMember
        if hasattr(obj, k):
            try:
                val = getattr(obj, "_" + k)
                setattr(obj, "_" + k, type(val)(ctx[k]))
                ctx.pop(k)
            except (AttributeError, TypeError):
                try:
                    val = getattr(obj, k)
                    setattr(obj, k, type(val)(ctx[k]))
                    ctx.pop(k)
                except AttributeError:
                    pass


def _override(ctx, *param_dicts):
    # Try to use the extra arguments as an override of config.

    if ctx.args:
        ctx = _ctx_to_dct(ctx.args)
        for p in param_dicts:
            _update(p, ctx)

        # Also update globals, always.
        if ctx:
            warnings.warn("The following arguments were not able to be set: %s" % ctx)


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

    out = get_hmf(
        cfg.get("quantities"),
        framework=cfg.get("framework", hmf.MassFunction),
        get_label=True ** cfg.get("params"),
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
