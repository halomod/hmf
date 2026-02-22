"""Tests of HMF."""

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
