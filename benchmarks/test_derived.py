"""Benchmarks of quantities derived from the mass function or the power spectrum.

Workloads 9 and 11 of ``benchmarks/README.md``.
"""

# Mass conversion imports these on first use; import them here so that no timing
# includes the import.
import halomod.concentration
import halomod.profiles  # noqa: F401
import numpy as np
import pytest

from hmf import MassFunction

CONVERSION_ZS = np.linspace(0.5, 2.5, 5)


def _converting_mf():
    """A ``MassFunction`` converting SMT (measured in FoF) to SOMean(200)."""
    return MassFunction(
        hmf_model="SMT",
        mdef_model="SOMean",
        mdef_params={"overdensity": 200},
        disable_mass_conversion=False,
    )


def test_mass_conversion_first(bench):
    """First ``dndm`` with mass conversion, sigma already computed on the grid."""
    holder = {}

    def setup():
        holder["mf"] = mf = _converting_mf()
        mf.sigma

    per_round = bench(lambda: holder["mf"].dndm, setup=setup, rounds=3)
    assert all(c["camb"] == 0 for c in per_round), per_round


def test_mass_conversion_z_loop(bench):
    """``update(z=...)`` then ``dndm`` with mass conversion, over 5 redshifts."""
    mf = _converting_mf()

    def setup():
        mf.update(z=0.0)
        mf.dndm

    def loop():
        for z in CONVERSION_ZS:
            mf.update(z=z)
            mf.dndm

    per_round = bench(loop, setup=setup, rounds=3, n_items=len(CONVERSION_ZS))
    assert all(c["camb"] == 0 for c in per_round), per_round


@pytest.fixture
def mf():
    """A default (CAMB) ``MassFunction`` with the linear power computed."""
    mf = MassFunction()
    mf.power
    return mf


def test_halofit_first(bench):
    """First ``nonlinear_power`` (halofit) of a fresh object with the linear power done."""
    holder = {}

    def setup():
        holder["mf"] = mf = MassFunction()
        mf.power

    per_round = bench(lambda: holder["mf"].nonlinear_power, setup=setup, rounds=5)
    assert all(c["camb"] == 0 for c in per_round), per_round


def test_halofit_after_z_change(bench, mf):
    """``nonlinear_power`` after a change of z."""
    zs = iter(np.tile([1.0, 0.0], 1000))

    def setup():
        mf.update(z=next(zs))
        mf.power

    per_round = bench(lambda: mf.nonlinear_power, setup=setup, rounds=50)
    assert all(c["camb"] == 0 and c["sigma_grid"] == 0 for c in per_round), per_round
