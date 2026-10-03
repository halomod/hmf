import builtins

import numpy as np
import pytest
from colossus.cosmology.cosmology import setCosmology

# require colossus for this test
from colossus.halo.mass_defs import changeMassDefinition
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

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


@pytest.mark.filterwarnings("ignore:matter_species was not set")
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
    base = md.BaseMassDefinition()

    with pytest.raises(AttributeError, match="halo_density does not exist"):
        base.halo_density()

    assert base.colossus_name is None

    with pytest.raises(AttributeError, match="cannot convert mass to radius"):
        base.m_to_r(1.0)

    with pytest.raises(AttributeError, match="cannot convert radius to mass"):
        base.r_to_m(1.0)


def test_mass_definition_overdensity_helpers():
    class Dummy(md.BaseMassDefinition):
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


def _smt(**kwargs):
    """An SMT mass function (measured in SOVirial) at z=0 with the EH transfer function."""
    return MassFunction(hmf_model="SMT", transfer_model="EH", z=0.0, dlog10m=0.05, **kwargs)


@pytest.mark.filterwarnings("ignore:Your input mass definition")
@pytest.mark.parametrize("overdensity", [200, 1600])
def test_mass_conversion_conserves_cumulative_counts(overdensity):
    """A one-to-one relabelling of halo masses must conserve their number.

    For M' = M'(M) the measured-definition mass of a halo with mass M in the new
    definition, n_new(>M) = n_meas(>M'(M)).
    """
    meas = _smt(Mmin=9, Mmax=17)
    assert meas.mdef == md.SOVirial()
    ln_ngtm_meas = Spline(np.log(meas.m), np.log(meas.ngtm))

    new = _smt(
        Mmin=10,
        Mmax=15.5,
        mdef_model="SOMean",
        mdef_params={"overdensity": overdensity},
        disable_mass_conversion=False,
    )
    m_meas = new.mdef.change_definition(new.m, meas.mdef, z=0.0, cosmo=new.cosmo)[0]
    # The conversion is non-trivial over the whole range (M'/M ~ 0.81-0.89 for 200m
    # and ~ 1.35-1.66 for 1600m), so this isn't satisfied by accident.
    assert np.all(np.abs(m_meas / new.m - 1) > 0.05)

    sel = new.m <= 3e15
    ratio = new.ngtm[sel] / np.exp(ln_ngtm_meas(np.log(m_meas[sel])))

    # Before the fix the ratio drifted from ~1.02 to ~0.46 (200m) and from ~0.95 to
    # ~12 (1600m). After it, the measured maximum deviation is 3.5e-4 (200m) and 1.9e-3
    # (1600m) at dlog10m=0.05, falling ~20x at dlog10m=0.01: it is the error from
    # integrating dn/dm on the grid and from the finite-difference Jacobian, both
    # O(dlog10m^2). 5e-3 leaves room for that, and is ~100x below the bug's size.
    np.testing.assert_allclose(ratio, 1, rtol=5e-3)


@pytest.mark.filterwarnings("ignore:Your input mass definition")
def test_mass_conversion_to_same_definition_is_identity():
    """Converting to a definition with the same halo density as the measured one is a no-op."""
    virial = _smt()
    cosmo = virial.cosmo

    # SOMean with the same density threshold as SOVirial at z=0.
    overdensity = md.SOVirial().halo_density(0.0, cosmo) / md.SOMean().mean_density(0.0, cosmo)
    same = _smt(
        mdef_model="SOMean",
        mdef_params={"overdensity": overdensity},
        disable_mass_conversion=False,
    )
    assert same.mdef != virial.mdef  # so that the conversion is actually applied
    assert same._mass_conversion_active

    np.testing.assert_allclose(same.dndm, virial.dndm, rtol=1e-6, atol=0)


@pytest.mark.filterwarnings("ignore:Your input mass definition")
def test_disabled_mass_conversion_is_unchanged(monkeypatch):
    """With conversion disabled (the default), dndm is the fit evaluated at m, untouched."""
    monkeypatch.setattr(MassFunction, "ERROR_ON_BAD_MDEF", False)
    native = _smt()
    for kwargs in [{}, {"mdef_model": "SOMean", "mdef_params": {"overdensity": 1600}}]:
        h = _smt(**kwargs)
        assert not h._mass_conversion_active
        expected = h.fsigma * h.mean_density0 * np.abs(h._dlnsdlnm) / h.m**2
        np.testing.assert_array_equal(h.dndm, expected)
        np.testing.assert_array_equal(h.dndm, native.dndm)
