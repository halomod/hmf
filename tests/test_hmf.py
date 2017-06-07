import inspect
import os
LOCATION = "/".join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split("/")[:-1])
from nose.tools import raises
import sys
sys.path.insert(0, LOCATION)
from hmf import MassFunction
from hmf.filters import TopHat
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

def test_delta_halo_mean():
    hmf = MassFunction(delta_h=180, delta_wrt="mean")
    assert hmf.delta_halo == 180


def test_delta_halo_crit():
    hmf = MassFunction(delta_h=180, delta_wrt="crit", cosmo_params={"Om0":0.3})
    assert abs(hmf.delta_halo - 600.0) < 1e-3

@raises(ValueError)
def test_wrong_filter():
    h = MassFunction(filter_model=2)


@raises(ValueError)
def test_string_dc():
    h = MassFunction(delta_c="this")


@raises(ValueError)
def test_neg_dc():
    h = MassFunction(delta_c=-1)


@raises(ValueError)
def test_big_dc():
    h = MassFunction(delta_c=20.)


@raises(ValueError)
def test_wrong_fit():
    h = MassFunction(hmf_model=1)


@raises(ValueError)
def test_wrong_mf_par():
    h = MassFunction(hmf_params=2)

@raises(ValueError)
def test_wrong_dh():
    h = MassFunction(delta_h="string")


@raises(ValueError)
def test_neg_dh():
    h = MassFunction(delta_h=0)


@raises(ValueError)
def test_big_dh():
    h = MassFunction(delta_h=1e5)


@raises(ValueError)
def test_delta_wrt():
    h = MassFunction(delta_wrt="this")


def test_str_filter():
    h = MassFunction(filter_model="TopHat")
    h_ = MassFunction(filter_model="TopHat")

    assert np.allclose(h.sigma,h_.sigma)




def test_mass_nonlinear_outside_range():
    h = MassFunction(Mmin=8,Mmax=9)
    with warnings.catch_warnings(record=True) as w:
        assert h.mass_nonlinear>0
        assert len(w)






