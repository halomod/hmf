import numpy as np


import inspect
import os
LOCATION = "/".join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split("/")[:-1])
# from nose.tools import raises
import sys
sys.path.insert(0, LOCATION)
from hmf import MassFunction
from hmf import fitting_functions as ff
import copy
def check_close(hmf, close_hmf, fit, redshift):
    hmf.update(z=redshift, mf_fit=fit)
    close_hmf.update(z=redshift)
    print hmf.fsigma / close_hmf.fsigma
    diff = hmf.fsigma - close_hmf.fsigma
    inds = np.logical_not(np.isnan(diff))
    assert np.mean(np.abs(diff[inds] / hmf.fsigma[inds])) < 1

def test_all_fits():
    hmf = MassFunction(Mmin=10, Mmax=15, dlog10m=0.1,
                       lnk_min=-16, lnk_max=10, dlnk=0.01,
                       mf_fit='ST', z=0.0, sigma_8=0.8, n=1, Ob0=0.05,
                       cosmo_params={"Om0":0.3, "H0":70.0})

    hmf.fsigma
    close_hmf = copy.deepcopy(hmf)
    ff._allfits.remove("AnguloBound")

    for redshift in [0.0, 2.0]:
        for fit in ff._allfits:
            yield check_close, hmf, close_hmf, fit, redshift
