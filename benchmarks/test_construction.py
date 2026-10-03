"""Benchmarks of building a mass function from scratch, and of importing hmf.

Workloads 1, 2, 8, 12 and 13 of ``benchmarks/README.md``.
"""

import subprocess
import sys

import numpy as np
import pytest

from hmf import MassFunction
from hmf.helpers.functional import get_hmf

ZS = np.linspace(0.25, 5, 20)


def test_default_construct_and_dndm(bench):
    """Default ``MassFunction()`` (CAMB) built and ``dndm`` computed."""
    per_round = bench(lambda: MassFunction().dndm, rounds=5, warmup_rounds=1)

    # One CAMB run gives every matter species. #363 removed the second run that
    # normalising sigma_8 to a different species used to trigger.
    assert all(c["camb"] == 1 for c in per_round), per_round


def test_default_construct(bench):
    """Default ``MassFunction()`` construction alone (CAMB runs at construction)."""
    per_round = bench(MassFunction, rounds=5, warmup_rounds=1)
    assert all(c["camb"] == 1 for c in per_round), per_round


def test_default_first_dndm(bench):
    """First ``dndm`` of a freshly built default ``MassFunction()``."""
    holder = {}

    def setup():
        holder["mf"] = MassFunction()

    per_round = bench(lambda: holder["mf"].dndm, setup=setup, rounds=5)
    assert all(c["camb"] == 0 for c in per_round), per_round


def test_default_cached_dndm(bench):
    """Accessing an already computed ``dndm`` again."""
    mf = MassFunction()
    mf.dndm

    per_round = bench(lambda: mf.dndm, rounds=200)
    assert all(c["camb"] == 0 and c["sigma_grid"] == 0 for c in per_round), per_round


def test_eh_construct_and_dndm(bench):
    """``MassFunction(transfer_model="EH").dndm``."""
    per_round = bench(lambda: MassFunction(transfer_model="EH").dndm, rounds=10, warmup_rounds=1)
    assert all(c["camb"] == 0 for c in per_round), per_round


def test_get_hmf_z(bench):
    """``get_hmf("dndm", z=[...20 values])`` with the default (CAMB) transfer."""
    per_round = bench(
        lambda: list(get_hmf("dndm", z=list(ZS))), rounds=3, warmup_rounds=1, n_items=len(ZS)
    )

    # get_hmf's ordering pass uses a fast (BBKS) setup, so the only CAMB run is the
    # one building the real framework: changing z must not re-run it.
    assert all(c["camb"] == 1 for c in per_round), per_round


# The most CAMB runs each introspection classmethod makes today. Each builds a
# default instance, which runs CAMB, so these are upper bounds to stop
# them getting worse; they should drop to 0 once introspection stops instantiating.
INTROSPECTION_MAX_CAMB = {
    "get_all_parameter_names": 1,
    "quantities_available": 1,
    "parameter_info": 1,
    "get_all_parameter_defaults": 2,
}


@pytest.mark.parametrize("method", list(INTROSPECTION_MAX_CAMB))
def test_introspection(bench, method, capsys):
    """The introspection classmethods of ``MassFunction``."""
    per_round = bench(getattr(MassFunction, method), rounds=3)
    capsys.readouterr()  # parameter_info prints.

    assert all(c["camb"] <= INTROSPECTION_MAX_CAMB[method] for c in per_round), per_round


@pytest.mark.parametrize("code", ["pass", "import hmf"], ids=["python", "import_hmf"])
def test_import(bench, code):
    """A fresh interpreter importing hmf, against one doing nothing (subprocess)."""
    bench(
        lambda: subprocess.run([sys.executable, "-c", code], check=True),
        rounds=5,
        warmup_rounds=1,
    )
