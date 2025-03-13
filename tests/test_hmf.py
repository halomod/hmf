"""Tests of HMF."""

import pytest
from pytest import raises

import numpy as np

from hmf import MassFunction


def test_wrong_filter():
    with raises(ValueError):
        MassFunction(filter_model=2)


def test_string_dc():
    with raises(ValueError):
        MassFunction(delta_c="this")


def test_neg_dc():
    with raises(ValueError):
        MassFunction(delta_c=-1)


def test_big_dc():
    with raises(ValueError):
        MassFunction(delta_c=20.0)


def test_wrong_fit():
    with raises(ValueError):
        MassFunction(hmf_model=1)


def test_wrong_mf_par():
    with raises(ValueError):
        MassFunction(hmf_params=2)


def test_str_filter():
    h = MassFunction(filter_model="TopHat", transfer_model="EH")
    h_ = MassFunction(filter_model="TopHat", transfer_model="EH")

    assert np.allclose(h.sigma, h_.sigma)


def test_mass_nonlinear_outside_range():
    h = MassFunction(Mmin=8, Mmax=9, transfer_model="EH")
    with pytest.warns(UserWarning):
        assert h.mass_nonlinear > 0


def test_nu():
    h = MassFunction(Mmin=8, Mmax=18, transfer_model="EH")
    assert np.allclose(h.nu_fn(h.m), h.nu)


def test_sigma8z():
    h = MassFunction(z=0.0, sigma_8=0.8, Mmin=8, Mmax=18, transfer_model="EH")
    assert np.allclose(h.sigma8_z, 0.8)


def test_neff_at_collapse():
    h = MassFunction(Mmin=8, Mmax=18, transfer_model="EH")
    assert np.allclose(
        h.n_eff_at_collapse, h.n_eff[np.argmin(np.abs(h.nu - 1.0))], rtol=0.05
    )
