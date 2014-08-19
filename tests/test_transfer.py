import numpy as np
import inspect
import os
LOCATION = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# from nose.tools import raises
import sys
sys.path.insert(0, LOCATION)
from hmf.transfer import Transfer

def rms(a):
    print "RMS: ", np.sqrt(np.mean(np.square(a)))
    return np.sqrt(np.mean(np.square(a)))

def check_close(t, t2, fit):
    t.update(transfer_fit=fit)
    assert np.mean(np.abs((t.power - t2.power) / t.power)) < 1

def check_update(t, t2, k, v):
    t.update(**{k:v})
    assert np.mean(np.abs((t.power - t2.power) / t.power)) < 1 and np.mean(np.abs((t.power - t2.power) / t.power)) > 1e-6

def test_updates():
    t = Transfer()
    t2 = Transfer()
    for k, v in {"z":0.1,
                "wdm_mass":10.0,
                "transfer_options":{"initial_mode":2,
                                    "lAccuracyBoost":1.5,
                                    "AccuracyBoost":1.5},
                "sigma_8":0.82,
                "n":0.95,
                "H0":68.0}.iteritems():
        yield check_update, t, t2, k, v

def test_halofit():
    t = Transfer(lnk_min=-20, lnk_max=20, dlnk=0.05, transfer_fit="EH")
    assert abs(t.power[0] - t.nonlinear_power[0]) < 1e-5
    assert 5 + t.power[-1] < t.nonlinear_power[-1]

def test_data():
    t = Transfer(omegab=0.05, omegac=0.25, omegav=0.7, omegan=0.0, H0=70.0, sigma_8=0.8,
                  n=1, transfer_options={"transfer__k_per_logint":0, "transfer__kmax":100.0},
                  lnk_min=np.log(1e-11), lnk_max=np.log(1e11))
    tdata = np.genfromtxt(LOCATION + "/data/transfer_for_hmf_tests.dat")
    pdata = np.genfromtxt(LOCATION + "/data/power_for_hmf_tests.dat")
    assert rms(np.exp(t._unnormalised_lnT) - tdata[:, 1]) < 0.001
    assert rms(np.exp(t.power) - pdata[:, 1]) < 0.001
