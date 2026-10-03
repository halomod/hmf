"""Tests of HMF."""

import warnings

import numpy as np
import pytest

from hmf import MassFunction


def test_wrong_filter():
    with pytest.raises(ValueError, match=r"2 must be str or Component subclass"):
        MassFunction(filter_model=2)


def test_string_dc():
    with pytest.raises(ValueError, match=r"delta_c must be a number"):
        MassFunction(delta_c="this")


def test_neg_dc():
    with pytest.raises(ValueError, match=r"delta_c must be > 0"):
        MassFunction(delta_c=-1)


def test_big_dc():
    with pytest.raises(ValueError, match=r"delta_c must be < 10.0"):
        MassFunction(delta_c=20.0)


def test_wrong_fit():
    with pytest.raises(ValueError, match=r"must be str or Component subclass"):
        MassFunction(hmf_model=1)


def test_wrong_mf_par():
    with pytest.raises(ValueError, match=r"hmf_params must be a dictionary"):
        MassFunction(hmf_params=2)


def test_str_filter():
    h = MassFunction(filter_model="TopHat", transfer_model="EH")
    h_ = MassFunction(filter_model="TopHat", transfer_model="EH")

    assert np.allclose(h.sigma, h_.sigma)


@pytest.mark.parametrize("z", [0.0, 1.0, 3.0, 8.0])
@pytest.mark.parametrize(("mmin", "mmax"), [(8, 9), (14, 15), (3, 18)])
def test_mass_nonlinear_sigma_equals_delta_c(z, mmin, mmax):
    """sigma(M_nl) = delta_c, whether or not M_nl lies inside the mass grid (regression).

    M_nl used to come from a minimiser whenever nu=1 was off the grid; it returned an
    inaccurate mass (e.g. 1152 instead of 915 Msun/h at z=8), or 0 if it failed.
    """
    h = MassFunction(Mmin=mmin, Mmax=mmax, z=z, transfer_model="EH")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mnl = h.mass_nonlinear
    r = h.filter.mass_to_radius(mnl, h.mean_density0)
    sigma = h.normalised_filter.sigma(r)[0]  # normalised filter uses P(k, z)
    assert sigma == pytest.approx(h.delta_c, rel=1e-8)


@pytest.mark.filterwarnings("ignore:The k-range")
def test_mass_nonlinear_unbracketable_raises():
    # With k < e^-1 h/Mpc, sigma(R >~ 1 Mpc/h) never reaches delta_c at z=20.
    h = MassFunction(z=20, lnk_min=-10, lnk_max=-1, transfer_model="EH")
    with pytest.raises(ValueError, match="nonlinear mass"):
        h.mass_nonlinear


def test_nu_fn_accurate_between_grid_points():
    """nu_fn interpolates in log-log space, so it is accurate even on a coarse grid.

    The old k=5 spline in linear m was off by a factor of ~6e6 between points 0.5 dex
    apart.
    """
    h = MassFunction(Mmin=8, Mmax=15, dlog10m=0.5, transfer_model="EH")
    mid = np.sqrt(h.m[1:] * h.m[:-1])
    sigma = h.normalised_filter.sigma(h.filter.mass_to_radius(mid, h.mean_density0))
    np.testing.assert_allclose(h.nu_fn(mid), (h.delta_c / sigma) ** 2, rtol=1e-4)


def test_nu():
    h = MassFunction(Mmin=8, Mmax=18, transfer_model="EH")
    assert np.allclose(h.nu_fn(h.m), h.nu)


def test_sigma8z():
    h = MassFunction(z=0.0, sigma_8=0.8, Mmin=8, Mmax=18, transfer_model="EH")
    assert np.allclose(h.sigma8_z, 0.8)


def test_sigma8z_matches_input_when_k_range_exceeds_sigma8_fallback_grid():
    """The normalised power on the user's own k-grid must reproduce the input sigma_8.

    For lnk in [-10, 12], sigma_8 normalisation uses a fallback grid (lnk_min > -15).
    That grid used to be fixed at [-8, 8], so the k > e^8 part of the user's grid was
    left out of the normalisation. A very blue spectrum (n=4) makes that part
    non-negligible: sigma8_z was off by ~9e-4. The fallback grid now spans the user's
    range with the same dlnk, so the two integrals are identical and agree to
    floating-point precision. rtol=1e-10 leaves headroom above round-off (~1e-15)
    while still failing on the old grid by six orders of magnitude.
    """
    h = MassFunction(transfer_model="EH", z=0.0, sigma_8=0.8, n=4.0, lnk_min=-10, lnk_max=12)
    assert np.isclose(h.sigma8_z[0], 0.8, rtol=1e-10, atol=0)


def test_neff_at_collapse():
    h = MassFunction(Mmin=8, Mmax=18, transfer_model="EH")
    assert np.allclose(h.n_eff_at_collapse, h.n_eff[np.argmin(np.abs(h.nu - 1.0))], rtol=0.05)


@pytest.mark.parametrize(("mmin", "mmax"), [(10, 13), (12, 13), (13.5, 15), (14.5, 16), (5, 10)])
def test_neff_at_collapse_independent_of_mass_grid(mmin, mmax):
    """n_eff at nu=1 does not depend on the mass grid (regression).

    The reference is n_eff interpolated at nu=1 on a wide, fine grid, which brackets
    nu=1. It used to return the n_eff at the nearest grid edge (or extrapolate wildly)
    when nu=1 was off the grid: e.g. +2.08 for Mmin=13.5 against about -2.01.
    """
    wide = MassFunction(Mmin=3, Mmax=18, dlog10m=0.01, transfer_model="EH")
    ref = np.interp(0.0, np.log(wide.nu), wide.n_eff)

    h = MassFunction(Mmin=mmin, Mmax=mmax, transfer_model="EH")
    assert h.n_eff_at_collapse == pytest.approx(ref, abs=1e-4)
    # Physically sensible: CDM n_eff at the nonlinear scale lies between -3 and -1.
    assert -3 < h.n_eff_at_collapse < -1


def test_default_k_range_does_not_warn():
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*k-range.*")
        h = MassFunction(transfer_model="EH")
    assert h._sigma_k_truncation_error < 1e-3


def test_narrow_lnk_max_with_small_mmin_warns():
    # R(Mmin=6) ~ 0.014 Mpc/h, so k_max = e^3 ~ 20 h/Mpc gives k_max * R_min ~ 0.3.
    with pytest.warns(UserWarning, match="k-range"):
        MassFunction(transfer_model="EH", Mmin=6, lnk_max=3)


def test_high_lnk_min_with_large_mmax_warns():
    # R(Mmax=16) ~ 30 Mpc/h, so k_min = e^-3 ~ 0.05 h/Mpc gives k_min * R_max ~ 1.5.
    with pytest.warns(UserWarning, match="k-range"):
        MassFunction(transfer_model="EH", Mmax=16, lnk_min=-3)


def test_k_truncation_error_matches_wide_grid():
    """The estimated truncation error should match a direct comparison to a wide grid."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        h = MassFunction(transfer_model="EH", Mmin=8, Mmax=16, lnk_min=-4, lnk_max=4)
        wide = MassFunction(transfer_model="EH", Mmin=8, Mmax=16, lnk_min=-20, lnk_max=14)
    expected = np.max(np.abs(h.sigma[[0, -1]] / wide.sigma[[0, -1]] - 1))
    assert expected > 0.01
    # Tails are extrapolated with a BBKS shape, so agreement is approximate.
    assert np.isclose(h._sigma_k_truncation_error, expected, rtol=0.1)


def test_mdef_params_without_measured_mdef():
    """Regression: mdef_params on a fit with no measured mdef (PS) used to crash."""
    from hmf.halos.mass_definitions import SOMean

    mf = MassFunction(hmf_model="PS", mdef_params={"overdensity": 300}, transfer_model="EH")
    assert isinstance(mf.mdef, SOMean)
    assert mf.mdef.params["overdensity"] == 300
    assert np.all(np.isfinite(mf.dndm))
    assert np.all(mf.dndm > 0)

    # PS has no measured mass definition, so no mass conversion is applied and the
    # chosen overdensity cannot change the mass function.
    default = MassFunction(hmf_model="PS", transfer_model="EH")
    np.testing.assert_allclose(mf.dndm, default.dndm, rtol=1e-12, atol=0)


def test_mdef_params_update_measured_mdef():
    """mdef_params override the parameters of a fit's measured mass definition."""
    from hmf.halos.mass_definitions import SOMean

    mf = MassFunction(
        hmf_model="Tinker08", mdef_params={"overdensity": 300}, transfer_model="EH", z=0
    )
    assert isinstance(mf.mdef, SOMean)
    assert mf.mdef.params["overdensity"] == 300
    assert np.all(np.isfinite(mf.dndm))

    # Delta=300m is a tabulated Tinker08 overdensity, so at z=0 the amplitude must be
    # exactly the tabulated A_300 (not the default 200m value).
    np.testing.assert_allclose(mf.hmf.A, mf.hmf.params["A_300"], rtol=1e-12, atol=0)
    assert not np.isclose(mf.hmf.A, mf.hmf.params["A_200"], rtol=1e-6, atol=0)


class _LntCounter:
    """Wrap ``BondEfs.lnt`` so tests can count transfer-function evaluations."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch):
        from hmf.density_field.transfer_models import BondEfs

        self.calls = 0
        original = BondEfs.lnt

        def counting_lnt(model, lnk):
            self.calls += 1
            return original(model, lnk)

        monkeypatch.setattr(BondEfs, "lnt", counting_lnt)


def _bondefs_mf(**kwargs) -> MassFunction:
    return MassFunction(
        transfer_model="BondEfs",
        transfer_params={"a": 37.1, "b": 21.1},
        cosmo_params={"Om0": 0.3, "H0": 70.0},
        **kwargs,
    )


@pytest.mark.parametrize(
    "update",
    [
        {"cosmo_params": {"Om0": 0.3}},
        {"transfer_params": {"a": 37.1}},
        {"cosmo_params": {"Om0": 0.3}, "transfer_params": {"b": 21.1}},
    ],
)
def test_noop_subset_dict_update_does_not_recompute_transfer(monkeypatch, update):
    """Regression test for #109: a no-op partial dict update must not invalidate."""
    counter = _LntCounter(monkeypatch)
    mf = _bondefs_mf()
    dndm = mf.dndm.copy()
    assert counter.calls > 0

    counter.calls = 0
    mf.update(**update)
    np.testing.assert_allclose(mf.dndm, dndm, rtol=0, atol=0)
    assert counter.calls == 0

    # The stored dicts are untouched by the no-op merge.
    assert mf.cosmo_params == {"Om0": 0.3, "H0": 70.0}
    assert mf.transfer_params == {"a": 37.1, "b": 21.1}


@pytest.mark.parametrize(
    ("update", "expected"),
    [
        (
            {"cosmo_params": {"Om0": 0.32}},
            {"cosmo_params": {"Om0": 0.32, "H0": 70.0}},
        ),
        (
            {"transfer_params": {"a": 30.0}},
            {"transfer_params": {"a": 30.0, "b": 21.1}},
        ),
    ],
)
def test_changing_subset_dict_update_invalidates(monkeypatch, update, expected):
    """A partial dict update that changes a value must recompute and merge."""
    counter = _LntCounter(monkeypatch)
    mf = _bondefs_mf()
    dndm_old = mf.dndm.copy()

    counter.calls = 0
    mf.update(**update)
    dndm_new = mf.dndm
    assert counter.calls > 0
    assert not np.allclose(dndm_new, dndm_old, rtol=1e-6, atol=0)

    fresh = _bondefs_mf()
    fresh.update(**expected)
    for key, value in expected.items():
        assert getattr(mf, key) == value
    np.testing.assert_allclose(dndm_new, fresh.dndm, rtol=1e-12, atol=0)


def test_empty_dict_update_clears_params():
    """Passing an empty dict still clears a ``*_params`` dict."""
    mf = _bondefs_mf()
    mf.dndm
    mf.update(cosmo_params={})
    assert mf.cosmo_params == {}

    fresh = MassFunction(transfer_model="BondEfs", transfer_params={"a": 37.1, "b": 21.1})
    np.testing.assert_allclose(mf.dndm, fresh.dndm, rtol=1e-12, atol=0)
