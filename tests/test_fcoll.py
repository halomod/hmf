'''
This module provides some tests of mgtm/mean_density0 against analytic f_coll.

As such, it is the best test of all calculations after sigma.
'''

import numpy as np
from hmf import MassFunction
from scipy.special import erfc


class TestFcoll(object):

    def check_fcoll(self, pert, fit):
        if fit == "PS":
            anl = fcoll_PS(np.sqrt(pert.nu))
            num = pert.rho_gtm / pert.mean_density0

        elif fit == "Peacock":
            anl = fcoll_Peacock(np.sqrt(pert.nu))
            num = pert.rho_gtm / pert.mean_density0

        err = np.abs((num - anl) / anl)
        print(np.max(err))
        print(num / anl - 1)
        assert np.max(err) < 0.05

    def test_fcolls(self):
        # Note: if Mmax>15, starts going wrong because of numerics at high M
        pert = MassFunction(Mmin=10, Mmax=15, dlog10m=0.01)
        fits = ['PS', 'Peacock']

        for fit in fits:
            pert.update(hmf_model=fit)
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
        num = hmf.rho_gtm / hmf.mean_density0
        err = np.abs((num - anl) / anl)[np.logical_and(hmf.m > 10 ** 10, hmf.m < 10 ** 15)]
        err = err[np.logical_not(np.isnan(err))]
        print((np.max(err)))
        assert np.max(err) < TestCumulants.tol

    def test_ranges_not_cut(self):
        hmf = MassFunction(hmf_model="Peacock", dlog10m=0.01)
        TestCumulants.tol = 0.05
        for minm in [9, 10, 11]:  # below, equal and greater than peacock cut
            for maxm in [14, 15, 16, 18, 19]:  # below,equal,greater than peacock cut and integration limit
                yield self.check, hmf, minm, maxm

    def test_ranges_cut(self):
        hmf = MassFunction(hmf_model="Peacock", dlog10m=0.01)
        TestCumulants.tol = 0.4
        for minm in [9, 10, 11]:  # below, equal and greater than peacock cut
            for maxm in [14, 15, 16, 18, 19]:  # below,equal,greater than peacock cut and integration limit
                yield self.check, hmf, minm, maxm

    def check_mgtm(self, hmf, maxm):
        hmf.update(Mmin=0, Mmax=maxm, dlog10m=0.01)
        print("rhogtm: ", hmf.rho_gtm)
        print("rhomean:", hmf.mean_density0)
        assert np.abs(hmf.rho_gtm[0] / hmf.mean_density0 - 1) < 0.1  # THIS IS PRETTY BIG!

    def test_mgtm(self):
        hmf = MassFunction(hmf_model="Tinker08")
        for maxm in [14, 15, 16, 18, 19]:  # below,equal,greater than integration limits
            yield self.check_mgtm, hmf, maxm

    def check_mltm(self, hmf, maxm):
        hmf.update(Mmin=3, Mmax=maxm, dlog10m=0.01)
        print(np.abs(hmf.rho_ltm[-1] / hmf.mean_density0 - 1))
        assert np.abs(hmf.rho_ltm[-1] / hmf.mean_density0 - 1) < 0.2

    def test_mltm(self):
        hmf = MassFunction(hmf_model="PS")
        for maxm in [14, 15, 16, 18, 19]:  # below,equal,greater than integration limits
            yield self.check_mltm, hmf, maxm
