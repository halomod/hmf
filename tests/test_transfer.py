import numpy as np
import inspect
import os
LOCATION = "/".join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split("/")[:-1])
# from nose.tools import raises
import sys
sys.path.insert(0, LOCATION)
from hmf import Transfer

def check_close(t, t2, fit):
    t.update(transfer_fit=fit)
    assert np.mean(np.abs((t.power - t2.power) / t.power)) < 1

def test_fits():
    t = Transfer(transfer_fit="CAMB")
    t2 = Transfer(transfer_fit="CAMB")

    for fit in Transfer.fits:
        yield check_close, t, t2, fit

def check_update(t, t2, k, v):
    t.update(**{k:v})
    assert np.mean(np.abs((t.power - t2.power) / t.power)) < 1 and np.mean(np.abs((t.power - t2.power) / t.power)) > 1e-6

def test_updates():
    t = Transfer()
    t2 = Transfer()
    for k, v in {"z":0.1,
                "wdm_mass":10.0,
                "initial_mode":2,
                "lAccuracyBoost":1.5,
                "AccuracyBoost":1.5,
                "sigma_8":0.82,
                "n":0.95,
                "H0":68.0}.iteritems():
        yield check_update, t, t2, k, v

def test_halofit():
    t = Transfer(lnk=np.linspace(-20, 20, 1000), transfer_fit="EH")
    assert abs(t.power[0] - t.nonlinear_power[0]) < 1e-5
    assert 5 + t.power[-1] < t.nonlinear_power[-1]
