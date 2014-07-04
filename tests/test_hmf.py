import inspect
import os
LOCATION = "/".join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split("/")[:-1])
from nose.tools import raises
import sys
sys.path.insert(0, LOCATION)
from hmf import _MassFunction, MassFunction

# @raises(ValueError)
# def test_wrong_fit():
#     hmf = _MassFunction(mf_fit=7)
#     assert hmf.mf_fit == 7
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
    hmf = _MassFunction(delta_h=180, delta_wrt="mean")
    assert hmf.delta_halo == 180


def test_delta_halo_crit():
    hmf = _MassFunction(delta_h=180, delta_wrt="crit", omegab=0.05, omegac=0.25)
    assert abs(hmf.delta_halo - 600.0) < 1e-3





