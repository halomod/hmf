import pytest
from pytest import raises

import inspect
import itertools
import numpy as np

from hmf import MassFunction
from hmf.mass_function import fitting_functions as ff

allfits = [
    o
    for n, o in inspect.getmembers(
        ff,
        lambda member: inspect.isclass(member)
        and issubclass(member, ff.FittingFunction)
        and member is not ff.FittingFunction
        and member is not ff.PS,
    )
]


@pytest.fixture(scope="module")
def hmf():
    return MassFunction(
        Mmin=10,
        Mmax=15,
        dlog10m=0.1,
        lnk_min=-16,
        lnk_max=10,
        dlnk=0.01,
        hmf_model="PS",
        z=0.0,
        sigma_8=0.8,
        n=1,
        cosmo_params={"Om0": 0.3, "H0": 70.0, "Ob0": 0.05},
        transfer_model="EH",
    )


@pytest.fixture(scope="module")
def ps_max(hmf):
    hmf.update(hmf_model="PS")
    return hmf.fsigma.max()


@pytest.mark.parametrize("redshift, fit", itertools.product([0.0, 2.0], allfits))
def test_allfits(hmf, ps_max, redshift, fit):
    """
    This basically tests all implemented fits to check the form for three things:
    1) whether the maximum fsigma is less than in the PS formula (which is known to overestimate)
    2) whether the slope is positive below this maximum
    3) whether the slope is negative above this maximum

    Since it calls each class, any blatant errors should also pop up.
    """
    hmf.update(z=redshift, hmf_model=fit)
    maxarg = np.argmax(hmf.fsigma)
    assert ps_max >= hmf.fsigma[maxarg]
    assert np.all(np.diff(hmf.fsigma[:maxarg]) >= 0)
    assert np.all(np.diff(hmf.fsigma[maxarg:]) <= 0)


def test_tinker08_dh():
    h = MassFunction(
        hmf_model="Tinker08",
        mdef_model="SOMean",
        mdef_params={"overdensity": 200},
        transfer_model="EH",
    )
    h1 = MassFunction(
        hmf_model="Tinker08",
        mdef_model="SOMean",
        mdef_params={"overdensity": 200.1},
        transfer_model="EH",
    )

    assert np.allclose(h.fsigma, h1.fsigma, rtol=1e-2)


def test_tinker10_dh():
    h = MassFunction(hmf_model="Tinker10", transfer_model="EH")
    h1 = MassFunction(
        hmf_model="Tinker10",
        mdef_model="SOMean",
        mdef_params={"overdensity": 200.1},
        transfer_model="EH",
    )

    assert np.allclose(h.fsigma, h1.fsigma, rtol=1e-2)


def test_tinker10_neg_gam():
    with raises(ValueError):
        h = MassFunction(
            hmf_model="Tinker10", hmf_params={"gamma_200": -1}, transfer_model="EH"
        )
        h.fsigma


def test_tinker10_neg_eta():
    with raises(ValueError):
        h = MassFunction(
            hmf_model="Tinker10", hmf_params={"eta_200": -1}, transfer_model="EH"
        )
        h.fsigma


def test_tinker10_neg_etaphi():
    with raises(ValueError):
        h = MassFunction(
            hmf_model="Tinker10",
            hmf_params={"eta_200": -1, "phi_200": 0},
            transfer_model="EH",
        )
        h.fsigma


def test_tinker10_neg_beta():
    with raises(ValueError):
        h = MassFunction(
            hmf_model="Tinker10", hmf_params={"beta_200": -1}, transfer_model="EH"
        )
        h.fsigma
