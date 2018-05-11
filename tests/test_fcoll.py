'''
This module provides some tests of mgtm/mean_density0 against analytic f_coll.

As such, it is the best test of all calculations after sigma.
'''

import numpy as np
from hmf import MassFunction
from scipy.special import erfc
import pytest


@pytest.fixture(params=['PS', "Peacock"])
def getmf(request):
    # Note: if Mmax>15, starts going wrong because of numerics at high M
    return MassFunction(Mmin=10, Mmax=15, dlog10m=0.01, hmf_model=request.param)


def test_fcoll(getmf):

    num = getmf.rho_gtm / getmf.mean_density0

    if getmf.hmf_model.__name__ == "PS":
        anl = fcoll_PS(np.sqrt(getmf.nu))

    elif getmf.hmf_model.__name__ == "Peacock":
        anl = fcoll_Peacock(np.sqrt(getmf.nu))
    else:
        print(getmf.hmf_model.__name__)

    err = np.abs((num - anl) / anl)
    print(np.max(err))
    print(num / anl - 1)
    assert np.max(err) < 0.05

def fcoll_PS(nu):
    return erfc(nu / np.sqrt(2))


def fcoll_Peacock(nu):
    a = 1.529
    b = 0.704
    c = 0.412

    return (1 + a * nu ** b) ** -1 * np.exp(-c * nu ** 2)


class TestCumulants(object):

    @pytest.fixture
    def peacock(self):
        return MassFunction(hmf_model="Peacock", dlog10m = 0.01)

    @pytest.mark.parametrize(['Mmin', "Mmax"],
                             [(9,14), (9,15), (9,16), (9,18), (9,19),
                              (10, 14), (10, 15), (10, 16), (10, 18), (10, 19),
                              (11, 14), (11, 15), (11, 16), (11, 18), (11, 19)])
    def test_ranges_cut(self, peacock, Mmin, Mmax):
        peacock.update(Mmin=Mmin, Mmax=Mmax)

        anl = fcoll_Peacock(np.sqrt(peacock.nu))
        num = peacock.rho_gtm / peacock.mean_density0
        err = np.abs((num - anl) / anl)[np.logical_and(peacock.m > 10 ** 10, peacock.m < 10 ** 15)]
        err = err[np.logical_not(np.isnan(err))]
        print((np.max(err)))
        assert np.max(err) < 0.4

    @pytest.fixture
    def tinker(self):
        return MassFunction(Mmin=0, hmf_model="Tinker08", dlog10m=0.01)

    @pytest.mark.parametrize("Mmax", [14,15,16,18,19])
    def test_mgtm(self, tinker, Mmax):
        tinker.update(Mmax=Mmax)
        print("rhogtm: ", tinker.rho_gtm)
        print("rhomean:", tinker.mean_density0)

        assert np.abs(tinker.rho_gtm[0] / tinker.mean_density0 - 1) < 0.1  # THIS IS PRETTY BIG!

    @pytest.fixture
    def ps(self):
        return MassFunction(hmf_model="PS", Mmin=3, dlog10m=0.01)

    @pytest.mark.parametrize("Mmax", [14, 15, 16, 18, 19])
    def test_mltm(self, ps, Mmax):
        ps.update(Mmax=Mmax)
        print(np.abs(ps.rho_ltm[-1] / ps.mean_density0 - 1))
        assert np.abs(ps.rho_ltm[-1] / ps.mean_density0 - 1) < 0.2
