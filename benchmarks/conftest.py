"""Fixtures and helpers for the hmf benchmark suite.

See ``benchmarks/README.md`` for how to run the suite and compare against the
committed baseline.
"""

import collections
import warnings
from collections.abc import Callable

import camb
import numpy as np
import pytest

# Import everything a benchmark touches up front, so that no timed region pays for
# a first import.
import hmf  # noqa: F401
from hmf.alternatives import wdm  # noqa: F401
from hmf.density_field import filters
from hmf.helpers import functional  # noqa: F401


@pytest.fixture(autouse=True)
def _quiet():
    """Silence warnings, so that formatting them is not part of any timing."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture
def calls(monkeypatch) -> collections.Counter:
    """Count the expensive calls made while a benchmark runs.

    The counter has two keys:

    ``camb``
        Calls to :func:`camb.get_transfer_functions`, i.e. CAMB runs.
    ``sigma_grid``
        Evaluations of sigma(R) on an array of radii, i.e. of the sigma integral that
        :meth:`~hmf.density_field.filters.BaseFilter.sigma` and the fused
        :meth:`~hmf.density_field.filters.BaseFilter.sigma_and_dlnss_dlnr` share,
        for more than one radius. A single radius, as in ``sigma_8``, is not counted.
    """
    counter = collections.Counter()

    real_get_transfer_functions = camb.get_transfer_functions

    def get_transfer_functions(*args, **kwargs):
        counter["camb"] += 1
        return real_get_transfer_functions(*args, **kwargs)

    monkeypatch.setattr(camb, "get_transfer_functions", get_transfer_functions)

    real_sigma_integral = filters.BaseFilter._sigma_integral

    def sigma_integral(self, w, *args, **kwargs):
        # w is the window on the (R, k) grid: one row per radius.
        if np.ndim(w) > 1 and np.shape(w)[0] > 1:
            counter["sigma_grid"] += 1
        return real_sigma_integral(self, w, *args, **kwargs)

    monkeypatch.setattr(filters.BaseFilter, "_sigma_integral", sigma_integral)

    return counter


@pytest.fixture
def bench(benchmark, calls):
    """Return :func:`run_benchmark` bound to this test's ``benchmark`` and ``calls``."""

    def run(target: Callable[[], object], **kwargs) -> list[collections.Counter]:
        return run_benchmark(benchmark, calls, target, **kwargs)

    return run


def run_benchmark(
    benchmark,
    calls: collections.Counter,
    target: Callable[[], object],
    *,
    setup: Callable[[], object] | None = None,
    rounds: int,
    warmup_rounds: int = 0,
    n_items: int = 1,
) -> list[collections.Counter]:
    """Time ``target`` and return the expensive calls it made in each round.

    ``setup`` runs before every round, outside the timed region, and the counters
    are reset after it, so they record only the calls made by ``target``. The calls
    are returned even when timing is disabled (``--benchmark-disable``), in which
    case ``target`` runs once, so the call-count assertions always run.

    Parameters
    ----------
    benchmark
        The pytest-benchmark fixture.
    calls
        The counter from the :func:`calls` fixture.
    target
        The workload to time. It takes no arguments.
    setup
        Run before each round, untimed.
    rounds
        The number of timed rounds.
    warmup_rounds
        Untimed rounds to run first (their calls are still returned).
    n_items
        The number of items (redshifts, parameter values, ...) one round loops
        over. It is stored in the results as ``extra_info["n_items"]``, so that
        ``compare.py`` can report the time per item.

    Returns
    -------
    list of Counter
        The calls counted in each round, warm-up rounds included. A key that was
        never called reads as 0.
    """
    per_round: list[collections.Counter] = []

    def _setup():
        if setup is not None:
            setup()
        calls.clear()

    def _teardown(*args, **kwargs):
        per_round.append(collections.Counter(calls))

    benchmark.extra_info["n_items"] = n_items

    if benchmark.disabled:
        _setup()
        target()
        _teardown()
    else:
        benchmark.pedantic(
            target,
            setup=_setup,
            teardown=_teardown,
            rounds=rounds,
            warmup_rounds=warmup_rounds,
        )
    return per_round
