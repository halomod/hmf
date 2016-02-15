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

allfits = [o for n,o in inspect.getmembers(ff,lambda member: inspect.isclass(member) and issubclass(member,ff.FittingFunction) and member is not ff.FittingFunction)]

class TestFitsCloseness(object):
    """
    This basically tests all implemented fits to check the form for three things:
    1) whether the maximum fsigma is less than in the PS formula (which is known to overestimate)
    2) whether the slope is positive below this maximum
    3) whether the slope is negative above this maximum

    Since it calls each class, any blatant errors should also pop up.
    """
    def __init__(self):

        self.hmf = MassFunction(Mmin=10, Mmax=15, dlog10m=0.1,
                       lnk_min=-16, lnk_max=10, dlnk=0.01,
                       hmf_model='PS', z=0.0, sigma_8=0.8, n=1,
                       cosmo_params={"Om0":0.3, "H0":70.0, "Ob0":0.05})

        self.ps_max = self.hmf.fsigma.max()

    def test_max_lt_ps(self):
        for redshift in [0.0, 2.0]:
            for fit in allfits:
                # if fit is ff.AnguloBound:
                #     continue
                yield self.check_form, fit, redshift

    def check_form(self,fit,redshift):
        self.hmf.update(z=redshift, hmf_model=fit)
        maxarg = np.argmax(self.hmf.fsigma)
        assert self.ps_max >= self.hmf.fsigma[maxarg]
        assert np.all(np.diff(self.hmf.fsigma[:maxarg]) >= 0)
        assert np.all(np.diff(self.hmf.fsigma[maxarg:]) <= 0)
