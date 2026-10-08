"""Hard performance gates: checks that fail CI, measured so they are stable on shared runners.

Issue #394 asks for hard gates only where a number can be measured reliably on a
shared CI runner. Timings in general cannot (they are tracked, not gated: see
``README.md``), but two kinds of check can:

* **The units boundary of hmf.core** (``test_unit_boundary_overhead_gate``): the fixed
  cost of :func:`hmf.core.units.unit_boundary` is at most 2 µs per call (#389). It is
  measured with the kernel replaced by the identity, as the difference of the median
  times of a decorated and an undecorated call, over many interleaved repeats, and
  re-measured once before failing.
* **Call counters** (the other tests): how many CAMB runs, or sigma(R) evaluations, a
  workload makes. These count calls and time nothing, so they are exact.

The v4 gates are at the end of this file: Boltzmann runs per input of the v4
Transfer and Growth stages, no recomputation of a MassVariance lattice node, and the
lattice's determinism under lazy extension (#384).
"""

import os
import statistics
import timeit

import astropy.units as u
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM, Flatw0waCDM, FlatwCDM, Planck18
from test_core_units import M, Msun_h, Toy

from hmf import MassFunction
from hmf.core import _boltzmann
from hmf.core.growth import Growth
from hmf.core.mass_variance import MassVariance
from hmf.core.transfer import Transfer
from hmf.core.units import h_Mpc

# ---------------------------------------------------------------------------------
# The units boundary
# ---------------------------------------------------------------------------------

#: The budget of the boundary's fixed cost per call (#389).
UNIT_BOUNDARY_BUDGET = 2e-6
#: Calls per timing, and interleaved timings of each method per measurement.
_CALLS, _REPEATS = 1000, 101


def _unit_boundary_overhead() -> float:
    """The boundary's cost per call: median(decorated) - median(undecorated) [s].

    Each repeat times ``_CALLS`` calls of each method, alternately, so that a slow
    patch of the runner affects both; the median discards the outliers.
    """
    toy = Toy()
    m = M * Msun_h
    decorated, raw = [], []
    for _ in range(_REPEATS):
        decorated.append(timeit.timeit(lambda: toy.decorated(m, z=0.5), number=_CALLS))
        raw.append(timeit.timeit(lambda: toy.undecorated(M, z=0.5), number=_CALLS))
    return (statistics.median(decorated) - statistics.median(raw)) / _CALLS


def test_unit_boundary_overhead_gate():
    """The units boundary costs at most 2 µs per call (one retry before failing)."""
    toy = Toy()
    for _ in range(100):  # warm up
        toy.decorated(M * Msun_h, z=0.5)

    overheads = [_unit_boundary_overhead()]
    if overheads[0] > UNIT_BOUNDARY_BUDGET:
        overheads.append(_unit_boundary_overhead())

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:  # noqa: PTH123
            f.write(
                "\n**Units-boundary gate:** "
                + ", then ".join(f"{o * 1e6:.2f} µs" for o in overheads)
                + f" per call (budget {UNIT_BOUNDARY_BUDGET * 1e6:.0f} µs).\n"
            )
    assert overheads[-1] <= UNIT_BOUNDARY_BUDGET, (
        f"unit_boundary costs {overheads[-1] * 1e6:.2f} µs per call (measured "
        f"{[f'{o * 1e6:.2f}' for o in overheads]} µs), over its "
        f"{UNIT_BOUNDARY_BUDGET * 1e6:.0f} µs budget"
    )


# ---------------------------------------------------------------------------------
# v3 call counters: one CAMB run per input
# ---------------------------------------------------------------------------------

_BASE = {"H0": Planck18.H0, "Om0": Planck18.Om0, "Ob0": Planck18.Ob0, "Tcmb0": Planck18.Tcmb0}

#: Cosmologies, and the CAMB runs one input makes in v3: one for the transfer
#: function, plus, for dark energy with w != -1, those of the default CambGrowth,
#: which does not reuse the transfer's run (two, the second at the first z != 0).
#: The CambGrowth runs are known waste: a fix should lower these.
CAMB_RUNS_PER_INPUT = {
    "planck18": (Planck18, 1),
    "lcdm_massless_nu": (FlatLambdaCDM(**_BASE, m_nu=0 * u.eV, name="massless"), 1),
    "lcdm_mnu0p3": (FlatLambdaCDM(**_BASE, m_nu=[0.1] * 3 * u.eV, name="mnu"), 1),
    "wcdm": (FlatwCDM(**_BASE, m_nu=Planck18.m_nu, w0=-0.9, name="wcdm"), 3),
    "w0wacdm": (Flatw0waCDM(**_BASE, m_nu=Planck18.m_nu, w0=-0.9, wa=0.2, name="w0wa"), 3),
}

#: Updates that leave CAMB's input unchanged, so must not run it again.
NO_CAMB_UPDATES = (
    {"z": 1.0},
    {"z": 2.0},
    {"hmf_model": "Tinker08"},
    {"sigma_8": 0.75},
    {"n": 0.95},
    {"sigma_8_species": "cb"},
    {"filter_model": "SharpK"},
    {"Mmin": 9.0},
    {"delta_c": 1.7},
)


@pytest.mark.parametrize("name", list(CAMB_RUNS_PER_INPUT))
def test_one_camb_run_per_input(calls, name):
    """A CAMB input runs CAMB once, whatever else changes; a new input, once more.

    One CAMB run gives every matter species (#363), so normalising sigma_8 to the
    total matter field while computing the CDM+baryon one must not run it twice.
    """
    cosmo, runs = CAMB_RUNS_PER_INPUT[name]
    mf = MassFunction(cosmo_model=cosmo, transfer_params={"extrapolate_with_eh": True})
    for params in NO_CAMB_UPDATES:
        mf.update(**params)
        mf.dndm
        mf.ngtm
    assert calls["camb"] == runs

    # A new CAMB input (H0) runs it again: once for the transfer, plus, with
    # CambGrowth, once for the growth at the current z (only one, as z != 0 already).
    calls.clear()
    mf.update(cosmo_params={"H0": 70.0})
    mf.dndm
    assert calls["camb"] == runs - (1 if runs > 1 else 0)


def test_no_sigma_recompute_without_power_change(calls):
    """Changing only z, the fit or delta_c leaves sigma(R) at z = 0 cached."""
    mf = MassFunction(transfer_model="EH")
    mf.dndm
    calls.clear()
    for params in ({"z": 1.0}, {"hmf_model": "Tinker08"}, {"delta_c": 1.7}, {"z": 3.0}):
        mf.update(**params)
        mf.dndm
    assert calls["sigma_grid"] == 0


# ---------------------------------------------------------------------------------
# v4 call counters and lattice determinism
# ---------------------------------------------------------------------------------

#: Masses for the MassVariance gates: a narrow range, and a wider one containing it.
_M_NARROW = np.logspace(8, 15, 211) * Msun_h
_M_WIDE = np.logspace(5, 16, 397) * Msun_h
_FILTERS = ("TopHat", "SharpK", "SmoothK")


@pytest.fixture
def boltzmann_runs():
    """CAMB runs made from here on, with the in-process memo of runs cleared."""
    _boltzmann.clear_memo()
    before = dict(_boltzmann.run_counts())
    yield lambda: _boltzmann.run_counts().get("camb", 0) - before.get("camb", 0)
    _boltzmann.clear_memo()


@pytest.fixture(scope="module")
def power_source():
    """Unnormalised EH power (Planck18), from the Transfer stage: no Boltzmann run needed.

    It is a fitting formula, defined at every k, so it is never extrapolated (which
    would warn).
    """
    return Transfer(cosmology=Planck18, model="EH").power_source("cb")


@pytest.mark.parametrize("name", list(CAMB_RUNS_PER_INPUT))
def test_v4_boltzmann_runs_per_input(boltzmann_runs, name):
    """The v4 Transfer and Growth stages run CAMB once per distinct input.

    One run gives every matter species, the transfer function, the power and the
    (scale-independent) CAMB growth at every z up to its z_max; n_s does not enter the
    run. Unlike v3's ratchet above, this holds for w != -1 too.
    """
    cosmo, _ = CAMB_RUNS_PER_INPUT[name]
    k = np.logspace(-4, 2, 50) * h_Mpc
    z = np.array([0.0, 0.5, 2.0, 6.0])
    transfer = Transfer(cosmology=cosmo, model="CAMB")
    for stage in (transfer, transfer.evolve(n_s=0.95)):
        growth = Growth.from_transfer(stage, model="CAMB")
        for species in ("cb", "tot"):
            stage.transfer_function(k, species)
            stage.unnormalised_power(k, species)
            growth.growth_factor(z, species)
            growth.growth_rate(z, species)
    assert boltzmann_runs() == 1

    # A new input (H0) runs it once more.
    transfer.evolve(cosmology=cosmo.clone(H0=70.0)).transfer_function(k)
    assert boltzmann_runs() == 2


@pytest.mark.parametrize("flt", _FILTERS)
def test_v4_sigma_recomputations(monkeypatch, power_source, flt):
    """A MassVariance stage computes each node of its mass lattice at most once.

    Whatever is asked of it (sigma, its slope, the inverse, repeated or overlapping
    mass ranges, a wider range after a narrower one), a node computed for one call is
    reused by every later call, and calls that need no new node compute none.
    """
    computed: list[np.ndarray] = []
    real = MassVariance._compute_nodes

    def counting(self, j):
        computed.append(np.array(j))
        return real(self, j)

    monkeypatch.setattr(MassVariance, "_compute_nodes", counting)
    mv = MassVariance(power=power_source, filter=flt)

    s = mv.sigma(_M_NARROW)
    assert computed, "the first call must compute nodes"
    n_first = len(computed)
    mv.sigma(_M_NARROW)
    mv.dlnsigma_dlnm(_M_NARROW)
    mv.sigma(_M_NARROW[::7])
    mv.dlnsigma_dlnm(_M_NARROW[100])
    assert len(computed) == n_first, "repeated or contained calls computed nodes"

    mv.sigma(_M_WIDE)
    mv.m_from_sigma(s[20:40])
    mv.sigma(_M_WIDE)
    mv.sigma(_M_NARROW)
    nodes = np.concatenate(computed)
    assert np.unique(nodes).size == nodes.size, "a lattice node was computed twice"


@pytest.mark.parametrize("flt", _FILTERS)
def test_v4_lattice_determinism(power_source, flt):
    """Lattice values are bit-identical under lazy extension, in either order (#384).

    sigma and its slope at the narrow masses are the same, to the bit, whether the
    narrow range is asked for first, after the wide one, or after it was itself
    extended; and the same mass gives the same value alone or in a batch. (Step 2b's
    unit tests check this in more cases; this is the gate.)
    """
    narrow_first = MassVariance(power=power_source, filter=flt)
    s1 = narrow_first.sigma(_M_NARROW)
    d1 = narrow_first.dlnsigma_dlnm(_M_NARROW)
    narrow_first.sigma(_M_WIDE)

    wide_first = MassVariance(power=power_source, filter=flt)
    wide_first.sigma(_M_WIDE)

    for stage in (narrow_first, wide_first):
        assert np.array_equal(stage.sigma(_M_NARROW), s1)
        assert np.array_equal(stage.dlnsigma_dlnm(_M_NARROW), d1)
    assert np.array_equal(narrow_first.sigma(_M_WIDE), wide_first.sigma(_M_WIDE))

    one_by_one = MassVariance(power=power_source, filter=flt)
    alone = np.array([one_by_one.sigma(m) for m in _M_NARROW[::-10]])
    assert np.array_equal(alone, s1[::-10])
