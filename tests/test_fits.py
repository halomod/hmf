import inspect
import itertools

import numpy as np
import pytest

from hmf import MassFunction
from hmf.mass_function import fitting_functions as ff

allfits = [
    o
    for n, o in inspect.getmembers(
        ff,
        lambda member: (
            inspect.isclass(member)
            and issubclass(member, ff.BaseFittingFunction)
            and member is not ff.BaseFittingFunction
            and member is not ff.PS
        ),
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


@pytest.mark.parametrize(("redshift", "fit"), itertools.product([0.0, 2.0], allfits))
def test_allfits(hmf, ps_max, redshift, fit):
    """
    Test all implemented fits for correct form and behavior.

    Tests that:
    1) the maximum fsigma is less than in the PS formula (which is known to overestimate)
    2) the slope is positive below this maximum
    3) the slope is negative above this maximum

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
    with pytest.raises(ValueError):
        h = MassFunction(hmf_model="Tinker10", hmf_params={"gamma_200": -1}, transfer_model="EH")
        h.fsigma


def test_tinker10_neg_eta():
    with pytest.raises(ValueError):
        h = MassFunction(hmf_model="Tinker10", hmf_params={"eta_200": -1}, transfer_model="EH")
        h.fsigma


def test_tinker10_neg_etaphi():
    with pytest.raises(ValueError):
        h = MassFunction(
            hmf_model="Tinker10",
            hmf_params={"eta_200": -1, "phi_200": 0},
            transfer_model="EH",
        )
        h.fsigma


def test_tinker10_neg_beta():
    with pytest.raises(ValueError):
        h = MassFunction(hmf_model="Tinker10", hmf_params={"beta_200": -1}, transfer_model="EH")
        h.fsigma


@pytest.mark.parametrize("z", [0.0, 1.0, 2.0])
def test_behroozi_mass_definition_consistency(z):
    """Behroozi dndm must be consistent between equivalent SO mass definitions.

    SOCritical(200) and SOMean(200/Omega_m(z)) define the same physical density
    threshold, so Behroozi (and any other HMF) must give the same dndm for both.
    This was broken before because the mass-definition conversion in dndm
    always used z=0 and the default Planck15 cosmology instead of the actual
    redshift and cosmology of the computation.
    """
    from astropy.cosmology import FlatLambdaCDM

    # Use non-Planck15 cosmology to verify the cosmo parameter is correctly
    # propagated in the mass-definition conversion (bug was: Planck15 used everywhere).
    cosmo_params = {"Om0": 0.3, "H0": 70.0, "Ob0": 0.05}
    cosmo = FlatLambdaCDM(**cosmo_params)

    # At redshift z, SOCritical(200) has the same density threshold as SOMean(equiv)
    equiv_overdensity = 200.0 / cosmo.Om(z)

    common = {
        "transfer_model": "EH",
        "Mmin": 10,
        "Mmax": 15,
        "dlog10m": 0.1,
        "z": z,
        "disable_mass_conversion": False,
        "cosmo_params": cosmo_params,
    }

    h_crit = MassFunction(
        hmf_model="Behroozi",
        mdef_model="SOCritical",
        mdef_params={"overdensity": 200},
        **common,
    )
    h_mean = MassFunction(
        hmf_model="Behroozi",
        mdef_model="SOMean",
        mdef_params={"overdensity": equiv_overdensity},
        **common,
    )

    # Allow ~2% tolerance: a small residual comes from the floating-point
    # imprecision of equiv_overdensity = 200/Om(z) and the NFW c-M approximation.
    # The original bug produced 20-50% errors, so this tolerance is well above that.
    np.testing.assert_allclose(h_crit.dndm, h_mean.dndm, rtol=2e-2)
