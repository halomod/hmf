import builtins

import numpy as np
import pytest
from colossus.cosmology.cosmology import setCosmology

# require colossus for this test
from colossus.halo.mass_defs import changeMassDefinition

import hmf.halos.mass_definitions as md
from hmf import MassFunction

# Set COLOSSUS cosmology


@pytest.fixture
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

    with pytest.raises(ValueError, match=r"name 'derp' is an unknown mass definition to colossus"):
        md.from_colossus_name("derp")


def test_change_dndm(colossus_cosmo):
    with pytest.warns(
        UserWarning,
        match=r"Your input mass definition 'SOVirial' does not match the mass definition",
    ):
        h = MassFunction(
            mdef_model="SOVirial",
            hmf_model="Warren",
            disable_mass_conversion=False,
            transfer_params={"extrapolate_with_eh": True},
        )

    dndm = h.dndm

    h.update(mdef_model="FOF")

    assert not np.allclose(h.dndm, dndm, atol=0, rtol=0.15)


def test_change_dndm_bocquet():
    h200m = MassFunction(
        mdef_model="SOMean",
        mdef_params={"overdensity": 200},
        hmf_model="Bocquet200mDMOnly",
        transfer_model="EH",
    )
    h200c = MassFunction(
        mdef_model="SOCritical",
        mdef_params={"overdensity": 200},
        hmf_model="Bocquet200cDMOnly",
        transfer_model="EH",
    )

    np.testing.assert_allclose(h200m.fsigma / h200c.fsigma, h200m.dndm / h200c.dndm)


def test_mass_definition_base_errors():
    base = md.MassDefinition()

    with pytest.raises(AttributeError, match="halo_density does not exist"):
        base.halo_density()

    assert base.colossus_name is None

    with pytest.raises(AttributeError, match="cannot convert mass to radius"):
        base.m_to_r(1.0)

    with pytest.raises(AttributeError, match="cannot convert radius to mass"):
        base.r_to_m(1.0)


def test_mass_definition_overdensity_helpers():
    class Dummy(md.MassDefinition):
        def halo_density(self, z=0, cosmo=md.Planck15):
            return 10.0

    dummy = Dummy()

    assert np.isclose(
        dummy.halo_overdensity_mean(),
        dummy.halo_density() / dummy.mean_density(),
    )
    assert np.isclose(
        dummy.halo_overdensity_crit(),
        dummy.halo_density() / dummy.critical_density(),
    )


def test_so_str_and_sogeneric():
    assert str(md.SOMean()) == "SOMean(200)"
    assert str(md.SOCritical(overdensity=500)) == "SOCritical(500)"
    assert str(md.SOVirial()) == "SOVirial"
    assert str(md.FOF()) == "FoF(l=0.2)"

    generic = md.SOGeneric()
    assert str(generic) == "SOGeneric"
    assert generic == md.SOMean()


def test_change_definition_requires_halomod(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        if name.startswith("halomod"):
            raise ImportError("No module named 'halomod'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    base = md.SOMean()
    with pytest.raises(ImportError, match="without halomod installed"):
        base.change_definition(1e12, md.SOCritical())


def test_change_definition_m_c_length_mismatch(monkeypatch):
    class DummyProfile:
        z = 0
        _h = None

        def cm_relation(self, m):
            return 5.0

        def _rho_s(self, c):
            return 1.0

    monkeypatch.setattr(md, "_find_new_concentration", lambda *args, **kwargs: 1.0)

    base = md.SOMean()
    other = md.SOCritical()
    profile = DummyProfile()

    with pytest.raises(ValueError, match="same length"):
        base.change_definition(np.array([1.0, 2.0]), other, profile=profile, c=np.array([1.0]))


def test_change_definition_broadcasts_and_warns(monkeypatch):
    class DummyProfile:
        def __init__(self, z):
            self.z = z
            self._h = None

        def cm_relation(self, m):
            return 5.0

        def _rho_s(self, c):
            c_arr = np.atleast_1d(c)
            return np.ones_like(c_arr, dtype=float)

    monkeypatch.setattr(md, "_find_new_concentration", lambda *args, **kwargs: 2.0)

    base = md.SOMean()
    other = md.SOCritical()
    profile = DummyProfile(z=2)

    with pytest.warns(UserWarning, match="Redshift of given profile"):
        base.change_definition(np.array([1.0, 2.0]), other, profile=profile, c=3.0, z=0)

    with pytest.warns(UserWarning, match="Redshift of given profile"):
        base.change_definition(1.0, other, profile=DummyProfile(z=2), c=np.array([3.0, 4.0]), z=0)

    with pytest.warns(UserWarning, match="Redshift of given profile"):
        base.change_definition(1.0, other, profile=DummyProfile(z=2), c=None, z=0)


def test_find_new_concentration_default_h(monkeypatch):
    def fake_brentq(fnc, xmin, xmax):
        fnc(1.0)
        return 1.0

    monkeypatch.setattr(md.sp.optimize, "brentq", fake_brentq)

    out = md._find_new_concentration(rho_s=1.0, halo_density=0.1, h=None, x_guess=5.0)
    assert out == 1.0


def test_find_new_concentration_failure_warns():
    with (
        pytest.warns(UserWarning, match="raised following error"),
        pytest.raises(md.OptimizationError, match="Could not determine x"),
    ):
        md._find_new_concentration(
            rho_s=1.0,
            halo_density=100.0,
            h=lambda x: 0.0,
            x_guess=5.0,
        )
