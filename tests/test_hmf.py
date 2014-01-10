import numpy as np
import inspect
import os
LOCATION = "/".join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split("/")[:-1])
from nose.tools import raises
import sys
sys.path.insert(0, LOCATION)
from hmf import MassFunction

@raises(TypeError)
def test_scalar_M():
    hmf = MassFunction(M=10.0)
    assert hmf.M == 10.0

@raises(ValueError)
def test_length1_M():
    hmf = MassFunction(M=[10.0])
    assert hmf.M == [10.0]


@raises(ValueError)
def test_random_M():
    hmf = MassFunction(M=[1, 4, 6, 8, 3.0])
    assert hmf.M == [1, 4, 6, 8, 3.0]

@raises(ValueError)
def test_wrong_fit():
    hmf = MassFunction(mf_fit=7)
    assert hmf.mf_fit == 7

@raises(ValueError)
def test_wrong_dh():
    hmf = MassFunction(delta_h=-10)
    assert hmf.delta_h == 10

@raises(ValueError)
def test_delta_wrt():
    hmf = MassFunction(delta_wrt="the_moon")
    assert hmf.delta_wrt == "the_moon"

def test_delta_halo_mean():
    hmf = MassFunction(delta_h=180, delta_wrt="mean")
    assert hmf.delta_halo == 180


def test_delta_halo_crit():
    hmf = MassFunction(delta_h=180, delta_wrt="mean", omegam=0.5)
    assert abs(hmf.delta_halo - 360.0) < 1e-3





