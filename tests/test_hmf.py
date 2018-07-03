from pytest import raises
from hmf import MassFunction
import numpy as np
import warnings


# @raises(ValueError)
# def test_wrong_fit():
#     hmf = MassFunction(hmf_model=7)
#     assert hmf.hmf_model == 7
#
# @raises(ValueError)
# def test_wrong_dh():
#     hmf = MassFunction(delta_h=-10)
#     assert hmf.delta_h == -10
#
# @raises(ValueError)
# def test_delta_wrt():
#     hmf = MassFunction(delta_wrt="the_moon")
#     assert hmf.delta_wrt == "the_moon"

def test_wrong_filter():
    with raises(ValueError):
        h = MassFunction(filter_model=2)


def test_string_dc():
    with raises(ValueError):
        h = MassFunction(delta_c="this")


def test_neg_dc():
    with raises(ValueError):
        h = MassFunction(delta_c=-1)


def test_big_dc():
    with raises(ValueError):
        h = MassFunction(delta_c=20.)


def test_wrong_fit():
    with raises(ValueError):
        h = MassFunction(hmf_model=1)


def test_wrong_mf_par():
    with raises(ValueError):
        h = MassFunction(hmf_params=2)


def test_str_filter():
    h = MassFunction(filter_model="TopHat")
    h_ = MassFunction(filter_model="TopHat")

    assert np.allclose(h.sigma, h_.sigma)


def test_mass_nonlinear_outside_range():
    h = MassFunction(Mmin=8, Mmax=9)
    with warnings.catch_warnings(record=True) as w:
        assert h.mass_nonlinear > 0
        assert len(w)
