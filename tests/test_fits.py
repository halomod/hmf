import numpy as np


import inspect
import os
LOCATION = "/".join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split("/")[:-1])
# from nose.tools import raises
import sys
sys.path.insert(0, LOCATION)
from hmf import _MassFunction
from hmf import fitting_functions as ff
def check_close(hmf, close_hmf, fit, redshift):
    hmf.update(z=redshift, mf_fit=fit)
    close_hmf.update(z=redshift)
    assert np.mean(np.abs((hmf.fsigma - close_hmf.fsigma) / hmf.fsigma)) < 1

def test_all_fits():
    hmf = _MassFunction(Mmin=10, Mmax=15, dlog10m=0.05, omegab=0.05, omegac=0.25,
                       omegav=0.7, sigma_8=0.8, n=1, H0=70.0,
                       lnk_min=-16, lnk_max=10, dlnk=0.01, mf_fit='ST', z=0.0)
    close_hmf = _MassFunction(Mmin=10, Mmax=15, dlog10m=0.05, omegab=0.05, omegac=0.25,
                       omegav=0.7, sigma_8=0.8, n=1, H0=70.0,
                       lnk_min=-16, lnk_max=10, dlnk=0.01, mf_fit='ST', z=0.0)
    hmf.fsigma
    close_hmf.fsigma

    for redshift in [0.0, 2.0]:
        for fit in ff._allfits:
            yield check_close, hmf, close_hmf, fit, redshift
