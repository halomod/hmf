"""Benchmarks of parameter scans on the v4 ``MassFunction`` stage (hmf.core).

The v4 counterparts of the scans in ``test_scans.py``, on the same masses (v3's
default grid, 10^10 to 10^15 Msun/h every 0.01 dex), CAMB and Planck18. Each round
starts from a stage whose caches are warm (the lattice nodes, the Boltzmann run, the
unnormalised sigma_8), so the timing is that of the scan alone: one ``evolve`` (for
sigma_8 and the fit) and one evaluation per step. The design target is under 0.5 ms
per step for dn/dm (#382). They are tracked with Bencher, not gated (``README.md``).
"""

import numpy as np
import pytest

from hmf.core import _boltzmann
from hmf.core.fits import FittingFunction
from hmf.core.mass_function import MassFunction

M = np.logspace(10, 15, 501)
ZS = np.linspace(0.25, 5, 20)
SIGMA8S = np.linspace(0.7, 0.9, 20)

#: Every built-in fit, with a redshift in its valid domain.
FITS = {
    name: (6.0 if name == "Yung24" else 0.0)
    for name, cls in sorted(FittingFunction.get_aliases().items())
    if cls.startswith("hmf.")
}


@pytest.fixture(scope="module")
def mf():
    """The default (CAMB) v4 mass function, with every cache the scans use warm."""
    mf = MassFunction.build()
    for z in (0.0, *ZS):
        mf.dndm_kernel(m=M, z=z)
        mf.ngtm_kernel(m=M, z=z)
    for name, z in FITS.items():
        mf.evolve(fit=name).dndm_kernel(m=M, z=z)
    return mf


@pytest.fixture
def no_boltzmann_run():
    """Fail if the benchmark runs a Boltzmann code."""
    before = dict(_boltzmann.run_counts())
    yield
    assert dict(_boltzmann.run_counts()) == before


def test_v4_sigma8_scan_dndm(bench, mf, no_boltzmann_run):
    """20 x ``evolve(linear_power=...evolve(sigma_8=...))`` + ``dndm``."""

    def scan():
        for sigma_8 in SIGMA8S:
            mf.evolve(linear_power=mf.linear_power.evolve(sigma_8=sigma_8)).dndm_kernel(m=M, z=0.0)

    bench(scan, rounds=10, warmup_rounds=1, n_items=SIGMA8S.size)


def test_v4_z_loop_dndm(bench, mf, no_boltzmann_run):
    """``dndm`` at each of 20 redshifts."""

    def scan():
        for z in ZS:
            mf.dndm_kernel(m=M, z=z)

    bench(scan, rounds=10, warmup_rounds=1, n_items=ZS.size)


def test_v4_z_loop_ngtm(bench, mf, no_boltzmann_run):
    """``ngtm`` at each of 20 redshifts."""

    def scan():
        for z in ZS:
            mf.ngtm_kernel(m=M, z=z)

    bench(scan, rounds=10, warmup_rounds=1, n_items=ZS.size)


def test_v4_z_vectorised_dndm(bench, mf, no_boltzmann_run):
    """``dndm`` at 20 redshifts in one call, (20, 501) (#247)."""
    bench(lambda: mf.dndm_kernel(m=M[None, :], z=ZS[:, None]), rounds=10, n_items=ZS.size)


def test_v4_fit_scan_dndm(bench, mf, no_boltzmann_run):
    """``evolve(fit=...)`` + ``dndm`` for each of the 28 built-in fits."""

    def scan():
        for name, z in FITS.items():
            mf.evolve(fit=name).dndm_kernel(m=M, z=z)

    bench(scan, rounds=5, warmup_rounds=1, n_items=len(FITS))
