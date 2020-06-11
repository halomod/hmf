# require colossus for this test
from colossus.halo.mass_defs import changeMassDefinition
from colossus.cosmology.cosmology import setCosmology

import hmf.halos.mass_definitions as md

from astropy.cosmology import Planck15
import numpy as np

# Set COLOSSUS cosmology
setCosmology("planck15")


def test_mean_to_mean_nfw():
    mdef = md.SOMean(overdensity=200)
    mdef2 = md.SOMean(overdensity=300)
    cduffy = mdef._duffy_concentration(1e12)

    mnew, rnew, cnew = mdef.change_definition(1e12, mdef2)

    mnew_, rnew_, cnew_ = changeMassDefinition(1e12, cduffy, 0, "200m", "300m", "nfw")

    assert np.isclose(mnew, mnew_, rtol=1e-2)
    assert np.isclose(rnew * 1e3, rnew_, rtol=1e-2)
    assert np.isclose(cnew, cnew_, rtol=1e-2)


def test_mean_to_crit_nfw():
    mdef = md.SOMean(overdensity=200)
    mdef2 = md.SOCritical(overdensity=300)

    cduffy = mdef._duffy_concentration(1e12)

    mnew, rnew, cnew = mdef.change_definition(1e12, mdef2)
    mnew_, rnew_, cnew_ = changeMassDefinition(1e12, cduffy, 0, "200m", "300c", "nfw")

    assert np.isclose(mnew, mnew_, rtol=1e-2)
    assert np.isclose(rnew * 1e3, rnew_, rtol=1e-2)
    assert np.isclose(cnew, cnew_, rtol=1e-2)


def test_mean_to_crit_z1_nfw():
    mdef = md.SOMean(overdensity=200)
    mdef2 = md.SOCritical(overdensity=300)

    cduffy = mdef._duffy_concentration(1e12, z=1)

    print("c=", cduffy)
    mnew, rnew, cnew = mdef.change_definition(1e12, mdef2, z=1)
    mnew_, rnew_, cnew_ = changeMassDefinition(1e12, cduffy, 1, "200m", "300c", "nfw")

    assert np.isclose(mnew, mnew_, rtol=1e-2)
    assert np.isclose(rnew * 1e3, rnew_, rtol=1e-2)
    assert np.isclose(cnew, cnew_, rtol=1e-2)


def test_mean_to_vir_nfw():
    mdef = md.SOMean()
    mdef2 = md.SOVirial()

    cduffy = mdef._duffy_concentration(1e12)

    mnew, rnew, cnew = mdef.change_definition(1e12, mdef2)
    mnew_, rnew_, cnew_ = changeMassDefinition(1e12, cduffy, 0, "200m", "vir", "nfw")

    assert np.isclose(mnew, mnew_, rtol=1e-2)
    assert np.isclose(rnew * 1e3, rnew_, rtol=1e-2)
    assert np.isclose(cnew, cnew_, rtol=1e-2)
