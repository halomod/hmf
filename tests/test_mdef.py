import pytest

import numpy as np
from colossus.cosmology.cosmology import setCosmology

# require colossus for this test
from colossus.halo.mass_defs import changeMassDefinition

import hmf.halos.mass_definitions as md
from hmf import MassFunction

# Set COLOSSUS cosmology


@pytest.fixture(scope="function")
def colossus_cosmo():
    setCosmology("planck15")


def test_mean_to_mean_nfw(colossus_cosmo):
    mdef = md.SOMean(overdensity=200)
    mdef2 = md.SOMean(overdensity=300)
    cduffy = mdef._duffy_concentration(1e12)

    mnew, rnew, cnew = mdef.change_definition(1e12, mdef2)

    mnew_, rnew_, cnew_ = changeMassDefinition(1e12, cduffy, 0, "200m", "300m", "nfw")

    assert np.isclose(mnew, mnew_, rtol=1e-2)
    assert np.isclose(rnew * 1e3, rnew_, rtol=1e-2)
    assert np.isclose(cnew, cnew_, rtol=1e-2)


def test_mean_to_crit_nfw(colossus_cosmo):
    mdef = md.SOMean(overdensity=200)
    mdef2 = md.SOCritical(overdensity=300)

    cduffy = mdef._duffy_concentration(1e12)

    mnew, rnew, cnew = mdef.change_definition(1e12, mdef2)
    mnew_, rnew_, cnew_ = changeMassDefinition(1e12, cduffy, 0, "200m", "300c", "nfw")

    assert np.isclose(mnew, mnew_, rtol=1e-2)
    assert np.isclose(rnew * 1e3, rnew_, rtol=1e-2)
    assert np.isclose(cnew, cnew_, rtol=1e-2)


def test_mean_to_crit_z1_nfw(colossus_cosmo):
    mdef = md.SOMean(overdensity=200)
    mdef2 = md.SOCritical(overdensity=300)

    cduffy = mdef._duffy_concentration(1e12, z=1)

    print("c=", cduffy)
    mnew, rnew, cnew = mdef.change_definition(1e12, mdef2, z=1)
    mnew_, rnew_, cnew_ = changeMassDefinition(1e12, cduffy, 1, "200m", "300c", "nfw")

    assert np.isclose(mnew, mnew_, rtol=1e-2)
    assert np.isclose(rnew * 1e3, rnew_, rtol=1e-2)
    assert np.isclose(cnew, cnew_, rtol=1e-2)


def test_mean_to_vir_nfw(colossus_cosmo):
    mdef = md.SOMean()
    mdef2 = md.SOVirial()

    cduffy = mdef._duffy_concentration(1e12)

    mnew, rnew, cnew = mdef.change_definition(1e12, mdef2)
    mnew_, rnew_, cnew_ = changeMassDefinition(1e12, cduffy, 0, "200m", "vir", "nfw")

    print(mnew, mnew_)

    assert np.isclose(mnew, mnew_, rtol=1e-2)
    assert np.isclose(rnew * 1e3, rnew_, rtol=1e-2)
    assert np.isclose(cnew, cnew_, rtol=1e-2)


def test_colossus_name(colossus_cosmo):
    assert md.SOMean().colossus_name == "200m"
    assert md.SOCritical().colossus_name == "200c"
    assert md.SOVirial().colossus_name == "vir"
    assert md.FOF().colossus_name == "fof"


def test_from_colossus_name(colossus_cosmo):
    assert md.from_colossus_name("200c") == md.SOCritical()
    assert md.from_colossus_name("200m") == md.SOMean()
    assert md.from_colossus_name("fof") == md.FOF()
    assert md.from_colossus_name("800c") == md.SOCritical(overdensity=800)
    assert md.from_colossus_name("vir") == md.SOVirial()

    with pytest.raises(ValueError):
        md.from_colossus_name("derp")


def test_change_dndm(colossus_cosmo):
    with pytest.warns(UserWarning):
        h = MassFunction(
            mdef_model="SOVirial", hmf_model="Warren", disable_mass_conversion=False
        )

    dndm = h.dndm

    h.update(mdef_model="FOF")

    assert not np.allclose(h.dndm, dndm, atol=0, rtol=0.15)


def test_change_dndm_bocquet():
    h200m = MassFunction(
        mdef_model="SOMean",
        mdef_params={"overdensity": 200},
        hmf_model="Bocquet200mDMOnly",
    )
    h200c = MassFunction(
        mdef_model="SOCritical",
        mdef_params={"overdensity": 200},
        hmf_model="Bocquet200cDMOnly",
    )

    np.testing.assert_allclose(h200m.fsigma / h200c.fsigma, h200m.dndm / h200c.dndm)
