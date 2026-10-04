"""Benchmarks of parameter scans on an existing ``MassFunction``.

Workloads 3-7 and 10 of ``benchmarks/README.md``. Each round starts from an object
whose ``dndm`` is already computed at the start of the scan (set up untimed), so
the timing is that of the updates alone.
"""

import numpy as np
import pytest

from hmf import MassFunction
from hmf.alternatives.wdm import MassFunctionWDM
from hmf.mass_function.fitting_functions import BaseFittingFunction

ZS = np.linspace(0.25, 5, 20)
SIGMA8S = np.linspace(0.7, 0.9, 20)
NS = np.linspace(0.92, 1.0, 5)

# Every built-in fitting function, with a redshift in its domain.
FITS = {
    name: (6.0 if name == "Yung24" else 0.0)
    for name, cls in sorted(BaseFittingFunction.get_models().items())
    if cls.__module__.startswith("hmf.")
}


@pytest.fixture
def mf():
    """A default (CAMB) ``MassFunction`` with ``dndm`` computed."""
    mf = MassFunction()
    mf.dndm
    return mf


def _reset(mf, quantity="dndm", **params):
    """Return a setup function that restores ``params`` and computes ``quantity``."""

    def setup():
        mf.update(**params)
        getattr(mf, quantity)

    return setup


def test_z_loop_dndm(bench, mf):
    """``update(z=...)`` then ``dndm``, over 20 redshifts."""

    def loop():
        for z in ZS:
            mf.update(z=z)
            mf.dndm

    per_round = bench(loop, setup=_reset(mf, z=0.0), rounds=20, n_items=len(ZS))

    # With scale-independent growth only D(z) changes: no CAMB, and no sigma(R).
    assert all(c["camb"] == 0 and c["sigma_grid"] == 0 for c in per_round), per_round


# sigma(R) evaluations per redshift that ngtm makes. sigma for the masses it
# extends the grid with (to integrate to high mass) is z-independent and cached.
NGTM_MAX_SIGMA_GRID_PER_Z = 0


def test_z_loop_ngtm(bench, mf):
    """``update(z=...)`` then ``ngtm``, over 20 redshifts."""

    def loop():
        for z in ZS:
            mf.update(z=z)
            mf.ngtm

    per_round = bench(loop, setup=_reset(mf, "ngtm", z=0.0), rounds=3, n_items=len(ZS))

    assert all(c["camb"] == 0 for c in per_round), per_round
    assert all(c["sigma_grid"] <= NGTM_MAX_SIGMA_GRID_PER_Z * len(ZS) for c in per_round), per_round


def test_sigma8_scan(bench, mf):
    """``update(sigma_8=...)`` then ``dndm``, over 20 values."""
    sigma_8 = mf.sigma_8

    def loop():
        for s8 in SIGMA8S:
            mf.update(sigma_8=s8)
            mf.dndm

    per_round = bench(loop, setup=_reset(mf, sigma_8=sigma_8), rounds=20, n_items=len(SIGMA8S))

    # sigma_8 only rescales the normalisation: sigma(R) must not be recomputed.
    assert all(c["camb"] == 0 and c["sigma_grid"] == 0 for c in per_round), per_round


def test_n_scan(bench, mf):
    """``update(n=...)`` then ``dndm``, over 5 values (each a full sigma recompute)."""
    n = mf.n

    def loop():
        for ni in NS:
            mf.update(n=ni)
            mf.dndm

    per_round = bench(loop, setup=_reset(mf, n=n), rounds=5, n_items=len(NS))

    # The transfer function does not depend on the primordial index: no CAMB run.
    assert all(c["camb"] == 0 for c in per_round), per_round


def test_fit_scan(bench, mf):
    """``update(hmf_model=...)`` then ``dndm``, over every built-in fitting function."""

    def loop():
        for name, z in FITS.items():
            mf.update(hmf_model=name, z=z)
            mf.dndm

    per_round = bench(
        loop, setup=_reset(mf, hmf_model="Tinker08", z=0.0), rounds=20, n_items=len(FITS)
    )

    # Neither the fit nor (scale-independent) z changes sigma(R).
    assert all(c["camb"] == 0 and c["sigma_grid"] == 0 for c in per_round), per_round


def test_wdm_z_loop(bench):
    """``MassFunctionWDM``: ``update(z=...)`` then ``dndm``, over 20 redshifts."""
    mf = MassFunctionWDM()
    mf.dndm

    def loop():
        for z in ZS:
            mf.update(z=z)
            mf.dndm

    per_round = bench(loop, setup=_reset(mf, z=0.0), rounds=10, n_items=len(ZS))

    # The WDM transfer does not depend on z (#372), so
    # a z change must not recompute sigma(R).
    assert all(c["camb"] == 0 and c["sigma_grid"] == 0 for c in per_round), per_round
