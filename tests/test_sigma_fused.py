"""Tests that the fused sigma / dlnss_dlnr computation gives unchanged results.

The reference values are computed with the unfused formulas that the filters used
before the fusion (one evaluation of the window per integral).
"""

import numpy as np
import pytest
import scipy.integrate as intg

from hmf import MassFunction
from hmf.density_field import filters

RTOL = 1e-14


def _reference_sigma(f, r):
    """sigma(r) as computed before the fusion (BaseFilter.sigma)."""
    rk = np.outer(r, f.k)
    dlnk = np.log(f.k[1] / f.k[0])
    integ = f.power * f.k**3 * f.k_space(rk) ** 2
    return np.sqrt((0.5 / np.pi**2) * intg.simpson(integ, dx=dlnk, axis=-1))


def _reference_dlnss_dlnr(f, r):
    """dlnss_dlnr(r) as computed before the fusion (BaseFilter.dlnss_dlnr)."""
    dlnk = np.log(f.k[1] / f.k[0])
    s = f.sigma(r)
    rk = np.outer(r, f.k)
    integ = f.k_space(rk) * f.dw_dlnkr(rk) * f.power * f.k**3
    return intg.simpson(integ, dx=dlnk, axis=-1) / (np.pi**2 * s**2)


@pytest.fixture(scope="module")
def power():
    mf = MassFunction(transfer_model="EH", dlog10m=0.05)
    return mf.k, mf._unnormalised_power, mf.radii


ALL_FILTERS = [
    (filters.TopHat, {}),
    (filters.Gaussian, {}),
    (filters.SharpK, {}),
    (filters.SmoothK, {}),
    (filters.SharpKEllipsoid, {"sigma_scale": 1.3}),
]


@pytest.mark.parametrize(("filt", "kwargs"), ALL_FILTERS, ids=lambda x: getattr(x, "__name__", ""))
def test_fused_matches_separate_calls(power, filt, kwargs):
    """sigma_and_dlnss_dlnm returns exactly what sigma and dlnss_dlnm return separately."""
    k, p, r = power
    f = filt(k, p, **kwargs)

    sigma, dlnss_dlnm = f.sigma_and_dlnss_dlnm(r)
    np.testing.assert_allclose(sigma, f.sigma(r), rtol=RTOL, atol=0)
    np.testing.assert_allclose(dlnss_dlnm, f.dlnss_dlnm(r), rtol=RTOL, atol=0)

    sigma, dlnss_dlnr = f.sigma_and_dlnss_dlnr(r)
    np.testing.assert_allclose(sigma, f.sigma(r), rtol=RTOL, atol=0)
    np.testing.assert_allclose(dlnss_dlnr, f.dlnss_dlnr(r), rtol=RTOL, atol=0)


@pytest.mark.parametrize(
    "filt", [filters.TopHat, filters.Gaussian, filters.SmoothK], ids=lambda x: x.__name__
)
def test_fused_matches_unfused_reference(power, filt):
    """For filters using the generic integrals, results equal the pre-fusion formulas."""
    k, p, r = power
    f = filt(k, p)

    sigma, dlnss_dlnr = f.sigma_and_dlnss_dlnr(r)
    np.testing.assert_allclose(sigma, _reference_sigma(f, r), rtol=RTOL, atol=0)
    np.testing.assert_allclose(dlnss_dlnr, _reference_dlnss_dlnr(f, r), rtol=RTOL, atol=0)
    np.testing.assert_allclose(f.dlnss_dlnr(r), _reference_dlnss_dlnr(f, r), rtol=RTOL, atol=0)
    np.testing.assert_allclose(f.sigma(r), _reference_sigma(f, r), rtol=RTOL, atol=0)


@pytest.mark.parametrize("filt", [filters.TopHat, filters.SmoothK], ids=lambda x: x.__name__)
def test_shared_window_matches_separate_window(power, filt):
    """Filters that share work between W and dW/dlnx give the same W and dW."""
    k, p, r = power
    f = filt(k, p)
    rk = np.outer(r, k)
    w, dw = f._window_and_derivative(rk)
    np.testing.assert_allclose(w, f.k_space(rk), rtol=RTOL, atol=0)
    np.testing.assert_allclose(dw, f.dw_dlnkr(rk), rtol=RTOL, atol=0)


def test_subclass_window_override_is_used(power):
    """A subclass overriding k_space/dw_dlnkr of TopHat must not get the TopHat shortcut."""
    k, p, r = power

    class Narrow(filters.TopHat):
        def k_space(self, kr):
            return super().k_space(2 * kr)

        def dw_dlnkr(self, kr):
            return super().dw_dlnkr(2 * kr)

    f = Narrow(k, p)
    sigma, dlnss_dlnr = f.sigma_and_dlnss_dlnr(r)
    np.testing.assert_allclose(sigma, _reference_sigma(f, r), rtol=RTOL, atol=0)
    np.testing.assert_allclose(dlnss_dlnr, _reference_dlnss_dlnr(f, r), rtol=RTOL, atol=0)

    # W(2kR) at R is W(kR) at 2R: a physical check that the override took effect.
    np.testing.assert_allclose(sigma, filters.TopHat(k, p).sigma(2 * r), rtol=RTOL, atol=0)


def test_subclass_sigma_override_is_used(power):
    """A subclass overriding sigma must have it used in dlnss_dlnr and the fused call."""
    k, p, r = power

    class Doubled(filters.TopHat):
        def sigma(self, r, order=0, rk=None):
            return 2 * super().sigma(r, order, rk)

    f = Doubled(k, p)
    base = filters.TopHat(k, p)
    sigma, dlnss_dlnr = f.sigma_and_dlnss_dlnr(r)
    np.testing.assert_allclose(sigma, 2 * base.sigma(r), rtol=RTOL, atol=0)
    # dlnss_dlnr divides by sigma^2, so doubling sigma divides it by 4.
    np.testing.assert_allclose(dlnss_dlnr, base.dlnss_dlnr(r) / 4, rtol=1e-12, atol=0)
    np.testing.assert_allclose(f.dlnss_dlnr(r), dlnss_dlnr, rtol=RTOL, atol=0)


def test_sharpk_fused_computes_sigma_once(power, monkeypatch):
    """SharpK's fused call evaluates its (slow, looped) sigma only once."""
    k, p, r = power
    f = filters.SharpK(k, p)
    expected = (f.sigma(r), f.dlnss_dlnr(r))

    # SharpK.sigma does one simpson integral per radius.
    calls = []
    orig = intg.simpson

    def counted(*args, **kwargs):
        calls.append(1)
        return orig(*args, **kwargs)

    monkeypatch.setattr(intg, "simpson", counted)
    got = f.sigma_and_dlnss_dlnr(r)
    assert len(calls) == len(r)
    np.testing.assert_allclose(got[0], expected[0], rtol=RTOL, atol=0)
    np.testing.assert_allclose(got[1], expected[1], rtol=RTOL, atol=0)


@pytest.mark.parametrize(
    "filter_model", ["TopHat", "Gaussian", "SharpK", "SmoothK", "SharpKEllipsoid"]
)
def test_massfunction_sigma_and_slope_unchanged(filter_model):
    """MassFunction's sigma and dlnsigma/dlnm equal the filter's separate calls."""
    mf = MassFunction(transfer_model="EH", filter_model=filter_model, dlog10m=0.05, z=0.5)
    f = mf.filter
    np.testing.assert_allclose(
        mf.sigma,
        mf._normalisation * f.sigma(mf.radii) * mf.growth_factor,
        rtol=RTOL,
        atol=0,
    )
    np.testing.assert_allclose(mf._dlnsdlnm, 0.5 * f.dlnss_dlnm(mf.radii), rtol=RTOL, atol=0)
