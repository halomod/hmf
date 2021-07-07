"""
This module contains a number of tests that check hmf's results against those of genmf

We check results for sigma, lnsigma, and the differential and cumulative mass functions against
genmf for two different redshifts (0 and 2). We use precisely the same transfer function here
as we use in genmf (tabulated). Another test tests if the power spectrum is generated
correctly according to this tabulated version.

The data files in the data/ directory are the following:
ST_0 etc :: output from genmf with given fit and redshift, produced with default cosmology here
power_for_hmf_tests.dat :: the power spectrum used in genmf
transfer_for_hmf_tests.dat :: the transfer function used in hmf (corresponds directly to the power)

The power was generated with hmf.transfer itself, so can be used as a direct test
for later versions.

To be more explicit, the power spectrum in all cases is produced with the following parameters:

    "w_lam"    :-1,
    "omegab"   : 0.05,
    "omegac"   : 0.25,
    "omegav"   : 0.7,
    "omegan"   : 0.0,
    "H0"       : 70,
    'cs2_lam'  : 1,
    'TCMB'     : 2.725,
    'yhe'      : 0.24,
    'Num_NuMassless' : 3.04,
    'reion__redshift': 10.3,
    'reion__optical_depth': 0.085
    "sigma_8":0.8,
    "n":1,
    "delta_c":1.686,
    "crit_dens":27.755 * 10 ** 10
    'Num_NuMassive'  : 0,
    'reion__fraction' :-1,
    'reion__delta_redshift' : 1.5,
    'lAccuracyBoost' : 1,
    'lSampleBoost'   : 1,
    'AccuracyBoost'  : 1,
    'transfer__k_per_logint': 0,
    'transfer__kmax':100.0
"""
import pytest

import numpy as np
from astropy.cosmology import LambdaCDM
from itertools import product

from hmf import MassFunction


def rms_diff(vec1, vec2, tol):
    mask = np.logical_and(
        np.logical_not(np.isnan(vec1)), np.logical_not(np.isnan(vec2))
    )
    vec1 = vec1[mask]
    vec2 = vec2[mask]
    err = np.sqrt(np.mean(((vec1 - vec2) / vec2) ** 2))
    print("RMS Error: ", err, "(> ", tol, ")")
    return err < tol


def max_diff_rel(vec1, vec2, tol):
    mask = np.logical_and(
        np.logical_not(np.isnan(vec1)), np.logical_not(np.isnan(vec2))
    )
    vec1 = vec1[mask]
    vec2 = vec2[mask]
    err = np.max(np.abs((vec1 - vec2) / vec2))
    print("Max Diff: ", err, "(> ", tol, ")")
    return err < tol


def max_diff(vec1, vec2, tol):
    mask = np.logical_and(
        np.logical_not(np.isnan(vec1)), np.logical_not(np.isnan(vec2))
    )
    vec1 = vec1[mask]
    vec2 = vec2[mask]
    err = np.max(np.abs(vec1 - vec2))
    print("Max Diff: ", err, "(> ", tol, ")")
    return err < tol


# ===============================================================================
# The Test Classes
# ===============================================================================
class TestGenMF:
    @pytest.fixture(scope="class")
    def hmf(self):
        return MassFunction(
            Mmin=7,
            Mmax=15.001,
            dlog10m=0.01,
            sigma_8=0.8,
            n=1,
            cosmo_model=LambdaCDM(Ob0=0.05, Om0=0.3, Ode0=0.7, H0=70.0, Tcmb0=0),
            lnk_min=-11,
            lnk_max=11,
            dlnk=0.01,
            transfer_params={"fname": "tests/data/transfer_for_hmf_tests.dat"},
            hmf_model="ST",
            z=0.0,
            transfer_model="FromFile",
            growth_model="GenMFGrowth",
        )

    @staticmethod
    def check_col(pert, fit, redshift, col):
        """Able to check all columns"""
        data = np.genfromtxt("tests/data/" + fit + "_" + str(int(redshift)))[::-1][
            400:1201
        ]

        # We have to do funky stuff to the data if its been cut by genmf
        if col == "sigma":
            assert max_diff_rel(pert.sigma, data[:, 5], 0.004)
        elif col == "lnsigma":
            # We just do diff on this one because it passes through 0
            assert max_diff(pert.lnsigma, data[:, 3], 0.001)
        elif col == "n_eff":
            assert max_diff_rel(pert.n_eff, data[:, 6], 0.001)
        elif col == "dndlog10m":
            assert rms_diff(pert.dndlog10m, 10 ** data[:, 1], 0.004)
        elif col == "fsigma":
            assert rms_diff(pert.fsigma, data[:, 4], 0.004)
        elif col == "ngtm":
            # The reason this is only good to 5% is GENMF's problem -- it uses
            # poor integration.
            assert rms_diff(pert.ngtm, 10 ** data[:, 2], 0.047)

    @pytest.mark.parametrize(
        ["redshift", "col"],
        [
            (0.0, "sigma"),
            (0.0, "lnsigma"),
            (0.0, "n_eff"),
            (2.0, "sigma"),
            (2.0, "lnsigma"),
            (2.0, "n_eff"),
        ],
    )
    def test_sigmas(self, hmf, redshift, col):
        # # Test z=0,2. Higher redshifts are poor in genmf.
        hmf.update(z=redshift, hmf_model="ST")
        self.check_col(hmf, "ST", redshift, col)

    @pytest.mark.parametrize(
        ["redshift", "fit", "col"],
        product(
            [0.0, 2.0],
            ["ST", "PS", "Reed03", "Warren", "Jenkins", "Reed07"],
            ["dndlog10m", "ngtm", "fsigma"],
        ),
    )
    def test_fits(self, hmf, redshift, fit, col):
        hmf.update(z=redshift, hmf_model=fit)
        self.check_col(hmf, fit, redshift, col)
