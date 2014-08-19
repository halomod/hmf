'''
This module provides some tests of mgtm/mean_dens against analytic f_coll.

As such, it is the best test of all calculations after sigma.
'''

import numpy as np
import inspect
import os
LOCATION = "/".join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split("/")[:-1])
# from nose.tools import raises
import sys
sys.path.insert(0, LOCATION)
from hmf import MassFunction
from scipy.special import erfc

class TestFcoll(object):

    def check_fcoll(self, pert, fit):
        if fit == "PS":
            anl = fcoll_PS(np.sqrt(pert.nu))
            num = pert.mgtm / pert.mean_dens

        elif fit == "Peacock":
            anl = fcoll_Peacock(np.sqrt(pert.nu))
            num = pert.mgtm / pert.mean_dens

        err = np.abs((num - anl) / anl)
        print np.max(err)
        print num / anl - 1
        assert np.max(err) < 0.05

    def test_fcolls(self):
        # Note: if Mmax>15, starts going wrong because of numerics at high M
        pert = MassFunction(Mmin=10, Mmax=15, dlog10m=0.01, cut_fit=False)
        fits = ['PS', 'Peacock']

        for fit in fits:
            pert.update(mf_fit=fit)
            yield self.check_fcoll, pert, fit



def fcoll_PS(nu):
    return erfc(nu / np.sqrt(2))

def fcoll_Peacock(nu):
    a = 1.529
    b = 0.704
    c = 0.412

    return (1 + a * nu ** b) ** -1 * np.exp(-c * nu ** 2)


class TestCumulants(object):
    tol = 0.05
    def check(self, hmf, minm, maxm):
        hmf.update(Mmin=minm, Mmax=maxm)
        anl = fcoll_Peacock(np.sqrt(hmf.nu))
        num = hmf.mgtm / hmf.mean_dens
        err = np.abs((num - anl) / anl)[np.logical_and(hmf.M > 10 ** 10, hmf.M < 10 ** 15)]
        err = err[np.logical_not(np.isnan(err))]
        print np.max(err)
        assert np.max(err) < TestCumulants.tol

    def test_ranges_not_cut(self):
        hmf = MassFunction(mf_fit="Peacock", cut_fit=False, dlog10m=0.01)
        TestCumulants.tol = 0.05
        for minm in [9, 10, 11]:  # below, equal and greater than peacock cut
            for maxm in [14, 15, 16, 18, 19]:  # below,equal,greater than peacock cut and integration limit
                yield self.check, hmf, minm, maxm

    def test_ranges_cut(self):
        hmf = MassFunction(mf_fit="Peacock", cut_fit=True, dlog10m=0.01)
        TestCumulants.tol = 0.4
        for minm in [9, 10, 11]:  # below, equal and greater than peacock cut
            for maxm in [14, 15, 16, 18, 19]:  # below,equal,greater than peacock cut and integration limit
                yield self.check, hmf, minm, maxm

    def check_mgtm(self, hmf, maxm):
        hmf.update(Mmin=0, Mmax=maxm, dlog10m=0.01)
        assert np.abs(hmf.mgtm[0] / hmf.mean_dens - 1) < 0.1  # THIS IS PRETTY BIG!


    def test_mgtm(self):
        hmf = MassFunction(mf_fit="PS", cut_fit=False)
        for maxm in [16, 18, 19]:  # below,equal,greater than integration limits
            yield self.check_mgtm, hmf, maxm

    def check_mltm(self, hmf, minm):
        hmf.update(Mmin=minm, Mmax=18, dlog10m=0.01)
        print np.abs(hmf.mltm[-1] / hmf.mean_dens - 1)
        assert np.abs(hmf.mltm[-1] / hmf.mean_dens - 1) < 0.2

    def test_mltm(self):
        hmf = MassFunction(mf_fit="PS", cut_fit=False)
        for minm in [2, 3, 5]:  # below,equal,greater than integration limits
            yield self.check_mltm, hmf, minm
