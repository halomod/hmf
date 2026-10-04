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

The v4 counters (Boltzmann runs per input of the v4 Transfer stage, sigma
recomputations of the v4 MassVariance stage) and the lattice-determinism gate need v4
stages that do not exist yet; their slots are at the end of this file.
"""

import os
import statistics
import timeit

import astropy.units as u
import pytest
from astropy.cosmology import FlatLambdaCDM, Flatw0waCDM, FlatwCDM, Planck18
from test_core_units import M, Msun_h, Toy

from hmf import MassFunction

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
# Slots for the v4 gates
# ---------------------------------------------------------------------------------


@pytest.mark.skip(reason="slot: needs the v4 Transfer stage (step 2a of #394)")
def test_v4_boltzmann_runs_per_input():
    """One Boltzmann-code run per distinct input of the v4 Transfer stage.

    To fill in (step 2a): count the CAMB runs of the v4 Transfer stage while
    evolving it through parameters that do not change CAMB's input (z, sigma_8, the
    filter, ...) and through every matter species; assert exactly one per input.
    """


@pytest.mark.skip(reason="slot: needs the v4 MassVariance stage (step 2b of #394)")
def test_v4_sigma_recomputations():
    """No sigma(R) recomputation when only z, the fit or delta_c changes in v4.

    To fill in (step 2b): count the sigma-integral kernel calls of the v4
    MassVariance stage across such ``evolve()`` calls; assert none after the first.
    """


@pytest.mark.skip(reason="slot: the lattice-determinism test is added by step 2b of #394")
def test_v4_lattice_determinism():
    """Lattice values are bit-identical under lazy extension, in either order (#384).

    Step 2b adds this as a unit test of the mass lattice (in ``tests/core``), where it
    runs with the test suite; this slot marks it as one of the hard gates of #394.
    Replace it with a call of that test, or delete it once that test exists.
    """
