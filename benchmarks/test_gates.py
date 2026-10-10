"""Hard performance gates: checks that fail CI, measured so they are stable on shared runners.

Issue #394 asks for hard gates only where a number can be measured reliably on a
shared CI runner. Timings in general cannot (they are tracked, not gated: see
``README.md``), but two kinds of check can:

* **The units boundary of hmf.core** (``test_unit_boundary_overhead_gate``): the fixed
  cost of :func:`hmf.core.units.unit_boundary` is at most 2 µs per call (#389), for a
  method with one dimensional argument and for one with two, which go through
  different wrappers. It is measured with the kernel replaced by the identity, as the
  difference of the median times of a decorated and an undecorated call, over many
  interleaved repeats, and re-measured once before failing.
* **Call counters** (the other tests): how many CAMB runs, or sigma(R) evaluations, a
  workload makes. These count calls and time nothing, so they are exact.

The v4 gates are at the end of this file: Boltzmann runs per input of the v4
Transfer and Growth stages, no recomputation of a MassVariance lattice node, the
lattice's determinism under lazy extension (#384), and what a change of a v4
MassFunction recomputes: no lattice node for sigma_8, z, the fit, delta_c or the
policy, no Boltzmann run for any of them, and one run for a whole tree. The last gate
is the cost of routing a flat ``evolve(sigma_8=...)`` through the stage tree (#383),
timed like the units boundary.
"""

import os
import statistics
import timeit

import astropy.units as u
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM, Flatw0waCDM, FlatwCDM, Planck18
from test_core_units import M, Msun_h, Toy

from hmf import MassFunction as MassFunctionV3
from hmf.core import _boltzmann
from hmf.core.growth import Growth
from hmf.core.mass_function import MassFunction
from hmf.core.mass_variance import MassVariance
from hmf.core.transfer import Transfer
from hmf.core.units import Mpc_h, h_Mpc

# ---------------------------------------------------------------------------------
# The units boundary
# ---------------------------------------------------------------------------------

#: The budget of the boundary's fixed cost per call (#389).
UNIT_BOUNDARY_BUDGET = 2e-6
#: Calls per timing, and interleaved timings of each method per measurement.
_CALLS, _REPEATS = 1000, 101


#: Two dimensional arguments, for the wrapper of methods with several of them.
_R = np.linspace(1.0, 10.0, M.size)


def _one_argument(toy):
    """A decorated and an undecorated call with one dimensional argument."""
    m = M * Msun_h
    return lambda: toy.decorated(m, z=0.5), lambda: toy.undecorated(M, z=0.5)


def _two_arguments(toy):
    """A decorated and an undecorated call with two dimensional arguments."""
    m, r = M * Msun_h, _R * Mpc_h
    return lambda: toy.decorated_two(m, r, z=0.5), lambda: toy.undecorated_two(M, _R, z=0.5)


def _unit_boundary_overhead(calls) -> float:
    """The boundary's cost per call: median(decorated) - median(undecorated) [s].

    Each repeat times ``_CALLS`` calls of each method, alternately, so that a slow
    patch of the runner affects both; the median discards the outliers.
    """
    decorated_call, raw_call = calls
    decorated, raw = [], []
    for _ in range(_REPEATS):
        decorated.append(timeit.timeit(decorated_call, number=_CALLS))
        raw.append(timeit.timeit(raw_call, number=_CALLS))
    return (statistics.median(decorated) - statistics.median(raw)) / _CALLS


@pytest.mark.parametrize(
    ("label", "make_calls"),
    [("one argument", _one_argument), ("two arguments", _two_arguments)],
    ids=["one-argument", "two-arguments"],
)
def test_unit_boundary_overhead_gate(label, make_calls):
    """The units boundary costs at most 2 µs per call (one retry before failing).

    Methods with one dimensional argument and with two go through different
    wrappers, so each is gated.
    """
    calls = make_calls(Toy())
    for _ in range(100):  # warm up
        calls[0]()

    overheads = [_unit_boundary_overhead(calls)]
    if overheads[0] > UNIT_BOUNDARY_BUDGET:
        overheads.append(_unit_boundary_overhead(calls))

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:  # noqa: PTH123
            f.write(
                f"\n**Units-boundary gate ({label}):** "
                + ", then ".join(f"{o * 1e6:.2f} µs" for o in overheads)
                + f" per call (budget {UNIT_BOUNDARY_BUDGET * 1e6:.0f} µs).\n"
            )
    assert overheads[-1] <= UNIT_BOUNDARY_BUDGET, (
        f"unit_boundary ({label}) costs {overheads[-1] * 1e6:.2f} µs per call (measured "
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
    mf = MassFunctionV3(cosmo_model=cosmo, transfer_params={"extrapolate_with_eh": True})
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
    mf = MassFunctionV3(transfer_model="EH")
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


# ---------------------------------------------------------------------------------
# v4 MassFunction: what each change recomputes
# ---------------------------------------------------------------------------------

#: The masses of the MassFunction gates (v3's default grid), and redshifts.
_M_MF = np.logspace(10, 15, 501)
_Z_MF = (0.0, 0.5, 1.0, 3.0)


@pytest.fixture
def computed_nodes(monkeypatch):
    """The MassVariance lattice nodes computed from here on (a function returning them)."""
    computed: list[np.ndarray] = []
    real = MassVariance._compute_nodes

    def counting(self, j):
        computed.append(np.array(j))
        return real(self, j)

    monkeypatch.setattr(MassVariance, "_compute_nodes", counting)
    return lambda: np.concatenate(computed) if computed else np.zeros(0, dtype=np.int64)


def _evaluate(mf):
    """Every quantity of a MassFunction on _M_MF at _Z_MF (n(>m) and rho(>m) included)."""
    z = np.array(_Z_MF)[:, None]
    mf.dndm_kernel(m=_M_MF, z=z)
    mf.ngtm_kernel(m=_M_MF, z=z)
    mf.rho_gtm_kernel(m=_M_MF, z=z)
    mf.at(z=1.0, m=_M_MF * Msun_h).dndm


def test_v4_sigma_8_change_recomputes_no_lattice_node(computed_nodes):
    """Changing sigma_8 computes no MassVariance node: the variance is shared (#382).

    The v4 counterpart of ``test_no_sigma_recompute_without_power_change``: sigma_8
    enters only the scalar amplitude of LinearPower, and the variance is of the
    unnormalised power.
    """
    mf = MassFunction.build(transfer_model="EH")
    _evaluate(mf)
    before = computed_nodes().size
    assert before > 0
    for sigma_8 in (0.7, 0.75, 0.85, 0.9):
        # By its flat name (routed, #383), and with the linear power evolved by hand.
        for changed in (
            mf.evolve(sigma_8=sigma_8),
            mf.evolve(linear_power=mf.linear_power.evolve(sigma_8=sigma_8)),
        ):
            assert changed.variance is mf.variance
            _evaluate(changed)
    assert computed_nodes().size == before


def test_v4_z_or_fit_change_recomputes_nothing_expensive(boltzmann_runs, computed_nodes):
    """Changing z, the fit, delta_c or the policy runs no Boltzmann code and recomputes no node.

    With CAMB: one run for the whole tree, then none. No lattice node is ever computed
    twice, and a change that needs no new mass computes no node at all. (Behroozi's
    correction needs n(>m) of Tinker08 at the lowest node of the integrals, one node
    below what the others use: it computes that one node.)
    """
    mf = MassFunction.build(transfer_model="CAMB", growth_model="CAMB")
    _evaluate(mf)
    assert boltzmann_runs() == 1
    before = computed_nodes().size
    m = _M_MF * Msun_h
    for z in (0.25, 2.0, 4.0):
        mf.dndm(m=m, z=z)
        mf.ngtm(m=m, z=z)
    for changes in (
        {"fit": "ST"},
        {"fit": "Watson"},
        {"delta_c": 1.7},
        {"domain_policy": "mask"},
        {"sigma_8": 0.75},
        {"fit.a": 0.75, "fit": "ST"},
    ):
        _evaluate(mf.evolve(**changes))
    assert computed_nodes().size == before
    _evaluate(mf.evolve(fit="Behroozi"))
    nodes = computed_nodes()
    assert nodes.size <= before + 1
    assert np.unique(nodes).size == nodes.size, "a lattice node was computed twice"
    assert boltzmann_runs() == 1


@pytest.mark.parametrize("growth_model", ["ODE", "CAMB"])
def test_v4_build_runs_one_boltzmann_code(boltzmann_runs, growth_model):
    """MassFunction.build() with CAMB runs CAMB once, for every species and quantity.

    The transfer of both species (sigma_8 is normalised with "tot", the power is "cb"),
    the growth (with CambGrowth, from the transfer's run, at the shared KAccuracy) and
    every quantity of the mass function come from one run.
    """
    mf = MassFunction.build(growth_model=growth_model)
    _evaluate(mf)
    mf.linear_power.power(k=np.logspace(-3, 1, 5) * h_Mpc, z=np.array([0.0, 2.0])[:, None])
    assert boltzmann_runs() == 1


# ---------------------------------------------------------------------------------
# v4 parameter routing (#383)
# ---------------------------------------------------------------------------------

#: The budget of a flat ``evolve(sigma_8=...)`` on a warm MassFunction: half the 0.5 ms
#: per-step budget of a scan, so that routing is never the cost of a step.
ROUTED_EVOLVE_BUDGET = 250e-6


def _routed_evolve_time(mf) -> float:
    """The median time of one flat ``evolve(sigma_8=...)`` [s] (see _unit_boundary_overhead)."""
    times = [
        timeit.timeit(lambda: mf.evolve(sigma_8=0.85), number=_CALLS // 10)
        for _ in range(_REPEATS // 4)
    ]
    return statistics.median(times) / (_CALLS // 10)


def test_v4_routed_evolve_gate():
    """A flat ``evolve(sigma_8=...)`` costs at most 0.25 ms (one retry before failing).

    It resolves the name through the routing table, and rebuilds the linear power and
    the mass function (validating both); the variance and the transfer and growth
    stages are shared. Measured on a warm tree, with nothing computed by the new one.
    """
    mf = MassFunction.build(transfer_model="EH")
    mf.linear_power.amplitude
    for _ in range(100):  # warm up
        mf.evolve(sigma_8=0.85)

    times = [_routed_evolve_time(mf)]
    if times[0] > ROUTED_EVOLVE_BUDGET:
        times.append(_routed_evolve_time(mf))

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:  # noqa: PTH123
            f.write(
                "\n**Routed-evolve gate:** "
                + ", then ".join(f"{t * 1e6:.1f} µs" for t in times)
                + f" per flat evolve(sigma_8=...) (budget {ROUTED_EVOLVE_BUDGET * 1e6:.0f} µs).\n"
            )
    assert times[-1] <= ROUTED_EVOLVE_BUDGET, (
        f"a flat evolve(sigma_8=...) costs {times[-1] * 1e6:.1f} µs (measured "
        f"{[f'{t * 1e6:.1f}' for t in times]} µs), over its "
        f"{ROUTED_EVOLVE_BUDGET * 1e6:.0f} µs budget"
    )
