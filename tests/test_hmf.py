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


def test_mass_nonlinear_outside_range():
    h = MassFunction(Mmin=8, Mmax=9, transfer_model="EH")
    with pytest.warns(UserWarning, match="Nonlinear mass outside mass range"):
        assert h.mass_nonlinear > 0


def test_nu():
    h = MassFunction(Mmin=8, Mmax=18, transfer_model="EH")
    assert np.allclose(h.nu_fn(h.m), h.nu)


def test_sigma8z():
    h = MassFunction(z=0.0, sigma_8=0.8, Mmin=8, Mmax=18, transfer_model="EH")
    assert np.allclose(h.sigma8_z, 0.8)


def test_neff_at_collapse():
    h = MassFunction(Mmin=8, Mmax=18, transfer_model="EH")
    assert np.allclose(h.n_eff_at_collapse, h.n_eff[np.argmin(np.abs(h.nu - 1.0))], rtol=0.05)


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
