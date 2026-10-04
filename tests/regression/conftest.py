"""Fixtures of the v3.7.2 regression-reference tests (issue #394).

The v4 providers import :mod:`hmf.core`, which warns on import that it is
experimental. Import it here, once, with the warning silenced, so that the ``-W
error`` CI job does not turn that into an error (see ``tests/core/conftest.py``).
"""

import warnings

import pytest
import regression_harness as rh

from hmf.exceptions import HMFCoreExperimentalWarning

with warnings.catch_warnings():
    warnings.simplefilter("ignore", HMFCoreExperimentalWarning)
    import hmf.core  # noqa: F401


@pytest.fixture(scope="session")
def reference() -> rh.Reference:
    """The v3.7.2 regression reference."""
    return rh.load_reference()


@pytest.fixture(scope="session")
def tolerances() -> rh.Tolerances:
    """The per-quantity tolerances."""
    return rh.load_tolerances()
