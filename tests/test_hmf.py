"""Tests of HMF."""
from pytest import raises

import numpy as np
import warnings

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
    h = MassFunction(Mmin=8, Mmax=9)
    with warnings.catch_warnings(record=True) as w:
        assert h.mass_nonlinear > 0
        assert len(w)
