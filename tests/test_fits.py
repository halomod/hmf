import numpy as np


import inspect
import os
LOCATION = "/".join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split("/")[:-1])
# from nose.tools import raises
import sys
sys.path.insert(0, LOCATION)
from hmf import Fits
from hmf import MassFunction

def check_close(hmf, close_hmf, fit, redshift):
    hmf.update(z=redshift, mf_fit=fit)
    close_hmf.update(z=redshift)
    assert np.mean(np.abs((hmf.fsigma - close_hmf.fsigma) / hmf.fsigma)) < 1
def test_all_fits():
    hmf = MassFunction(M=np.linspace(11, 13, 301), omegab=0.05, omegac=0.25,
                       omegav=0.7, sigma_8=0.8, n=1, H0=70.0,
                       lnk=np.linspace(-16, 10, 500), mf_fit='ST', z=0.0)
    close_hmf = MassFunction(M=np.linspace(11, 13, 301), omegab=0.05, omegac=0.25,
                       omegav=0.7, sigma_8=0.8, n=1, H0=70.0,
                       lnk=np.linspace(-16, 10, 500), mf_fit='ST', z=0.0)
    hmf.fsigma
    close_hmf.fsigma

    for redshift in [0.0, 2.0]:
        for fit in Fits.mf_fits:
            if fit != "user_model":
                yield check_close, hmf, close_hmf, fit, redshift
