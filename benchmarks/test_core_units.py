"""The fixed overhead of the hmf.core units boundary (workload 14 of README.md).

``unit_boundary`` has a budget of 2 µs per call (issue #389). These benchmarks time
a decorated method whose kernel is the identity, so the time is the decorator's alone,
against the same identity method undecorated. The overhead per call is the
difference of their per-item times. Each round makes ``N_CALLS`` calls.
"""

import timeit
import warnings

import astropy.units as u
import numpy as np
import pytest

from hmf.exceptions import HMFCoreExperimentalWarning

with warnings.catch_warnings():
    warnings.simplefilter("ignore", HMFCoreExperimentalWarning)
    from hmf.core.units import H0_unit, Msun_h, UnitContext, unit_boundary

N_CALLS = 10_000
M = np.logspace(10, 15, 500)


class Toy:
    """An object with an identity method, decorated and not."""

    _unit_context = UnitContext(70 * H0_unit)

    @unit_boundary(m=Msun_h, returns=Msun_h)
    def decorated(self, m, z=0.0):
        """Return ``m``, through the boundary."""
        return m

    def undecorated(self, m, z=0.0):
        """Return ``m``."""
        return m


def _loop(method, m):
    def run():
        for _ in range(N_CALLS):
            method(m, z=0.5)

    return run


@pytest.mark.parametrize(
    ("method", "unit"),
    [
        pytest.param("undecorated", None, id="undecorated"),
        pytest.param("decorated", Msun_h, id="canonical"),
        pytest.param("decorated", u.Msun, id="physical"),
    ],
)
def test_unit_boundary_overhead(bench, method, unit):
    """``N_CALLS`` calls of an identity method on 500 masses."""
    toy = Toy()
    m = M if unit is None else M * unit
    bench(_loop(getattr(toy, method), m), rounds=20, warmup_rounds=1, n_items=N_CALLS)


def test_unit_boundary_overhead_sanity():
    """A generous bound on the overhead, measured here so it runs with timing disabled.

    The budget is 2 µs; this only fails on an order-of-magnitude regression, so a
    noisy runner does not make it flaky.
    """
    toy = Toy()
    m = M * Msun_h
    n = 2000
    decorated = min(timeit.repeat(lambda: toy.decorated(m, z=0.5), number=n, repeat=7)) / n
    raw = min(timeit.repeat(lambda: toy.undecorated(M, z=0.5), number=n, repeat=7)) / n
    assert decorated - raw < 20e-6
