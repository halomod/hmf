"""Tests of the CLASS growth factor, which needs the optional classy package."""

import copy
import pickle

import numpy as np
import pytest
from astropy import cosmology
from astropy import units as u
from astropy.cosmology import Planck13
from test_growth import _independent_growth

from hmf import MassFunction, Transfer
from hmf.cosmology import growth_factor
from hmf.cosmology.growth_factor import ClassGrowth

classy = pytest.importorskip("classy")

LCDM = cosmology.FlatLambdaCDM(H0=67.7, Om0=0.31, Ob0=0.049, Tcmb0=2.7255)
WCDM = cosmology.FlatwCDM(H0=67.7, Om0=0.31, Ob0=0.049, w0=-0.8, Tcmb0=2.7255)
Z = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])


@pytest.mark.parametrize("cosmo", [LCDM, WCDM], ids=["lcdm", "wcdm_w-0.8"])
def test_agrees_with_camb(cosmo):
    """CLASS and CAMB, two independent Boltzmann codes, give the same growth factor.

    Both evaluate the synchronous-gauge density contrast at k = 0.01/Mpc. Tolerance:
    CLASS and CAMB agree on the linear P(k) to ~0.1% (Lesgourgues 2011, arXiv:1104.2934),
    so to ~1e-3 on ratios of it; measured < 2e-5 here.
    """
    d_class = ClassGrowth(cosmo).growth_factor(Z)
    d_camb = growth_factor.CambGrowth(cosmo).growth_factor(Z)
    np.testing.assert_allclose(d_class, d_camb, rtol=1e-3)


@pytest.mark.parametrize("cosmo", [LCDM, WCDM], ids=["lcdm", "wcdm_w-0.8"])
def test_matches_independent_ode(cosmo):
    """D and f agree with an independent solve of the sub-horizon growth equation.

    The ODE neglects radiation and dark-energy perturbations and the residual scale
    dependence at k = 0.01/Mpc, which CLASS includes. Tolerance: 1e-3, the accuracy
    expected between a Boltzmann code and the ODE at this scale; measured < 2.5e-4 in
    D and < 3e-4 in f (f is a spline derivative through CLASS's redshift table).
    """
    d_ref, f_ref = _independent_growth(cosmo, Z)
    gf = ClassGrowth(cosmo)
    np.testing.assert_allclose(gf.growth_factor(Z), d_ref, rtol=1e-3)
    np.testing.assert_allclose(gf.growth_rate(Z), f_ref, rtol=1e-3)


def test_agrees_with_growth_factor_in_lcdm():
    """In LCDM, CLASS's growth agrees with hmf's default (analytic/ODE) growth factor."""
    np.testing.assert_allclose(
        ClassGrowth(LCDM).growth_factor(Z),
        growth_factor.GrowthFactor(LCDM).growth_factor(Z),
        rtol=1e-3,
    )


@pytest.mark.parametrize("matter_species", ["cb", "tot"])
def test_growth_rate_matches_linder(matter_species):
    """The growth rate is f ~ Omega_m(z)^0.55 (Linder 2005), to ~1% in LCDM."""
    gf = ClassGrowth(Planck13, matter_species=matter_species)
    z = np.array([0.0, 1.0, 3.0])
    np.testing.assert_allclose(gf.growth_rate(z), Planck13.Om(z) ** 0.55, rtol=1e-2)


def test_growth_rate_matches_class():
    """The growth rate is that given by CLASS's own scale-dependent growth rate.

    CLASS computes it with its own spline (in z, of ln P) through the same table, so
    the two differ only by interpolation error, which is largest at the ends of it.
    """
    gf = ClassGrowth(LCDM, matter_species="tot")
    cl = classy.Class()
    cl.set(gf._class_input())
    cl.compute()
    z = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    f_class = np.array([cl.scale_dependent_growth_factor_f(gf.k_ref, zi) for zi in z])
    d_class = np.array([cl.scale_dependent_growth_factor_D(gf.k_ref, zi) for zi in z])
    cl.struct_cleanup()
    cl.empty()
    # Tolerance: measured < 5e-4 for f, 4e-7 for D.
    np.testing.assert_allclose(gf.growth_rate(z), f_class, rtol=1e-3)
    np.testing.assert_allclose(gf.growth_factor(z), d_class, rtol=1e-5)


def test_growth_rate_is_log_derivative():
    """F is d ln D / d ln a, checked by finite differences of the growth factor."""
    gf = ClassGrowth(WCDM)
    zr = np.array([0.5, 1.0, 2.0, 5.0])
    lna = -np.log1p(zr)
    eps = 0.02
    fd = (
        np.log(gf.growth_factor(np.exp(-(lna + eps)) - 1))
        - np.log(gf.growth_factor(np.exp(-(lna - eps)) - 1))
    ) / (2 * eps)
    # Tolerance: O(eps^2) truncation error.
    np.testing.assert_allclose(gf.growth_rate(zr), fd, rtol=1e-3)


@pytest.mark.parametrize("m_nu", [0.0, 0.3])
def test_matter_species(m_nu):
    """With massive neutrinos, D_cb / D_tot >= 1 and rises with z (as for CambGrowth)."""
    cosmo = cosmology.FlatLambdaCDM(
        H0=70.0, Om0=0.3, Ob0=0.05, Tcmb0=2.7255, m_nu=[0.0, 0.0, m_nu] * u.eV
    )
    z = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
    d_tot = ClassGrowth(cosmo, matter_species="tot").growth_factor(z)
    d_cb = ClassGrowth(cosmo, matter_species="cb").growth_factor(z)

    if m_nu == 0:
        np.testing.assert_allclose(d_cb, d_tot, rtol=1e-6)
        return

    # delta_tot = (1 - f_nu) delta_cb + f_nu delta_nu with 0 <= delta_nu <= delta_cb,
    # and delta_nu catches up with delta_cb as the neutrinos cool, so the z=0-normalised
    # cb growth factor is larger at z > 0, by at most 1 / (1 - f_nu).
    f_nu = cosmo.Onu0 / (cosmo.Onu0 + cosmo.Om0)
    ratio = d_cb[1:] / d_tot[1:]
    assert d_cb[0] == pytest.approx(1.0)
    assert np.all(ratio > 1 + 1e-4)
    assert np.all(ratio <= 1 / (1 - f_nu))
    assert np.all(np.diff(ratio) > 0)


@pytest.mark.parametrize("matter_species", ["cb", "tot"])
def test_massive_neutrinos_agree_with_camb(matter_species):
    """With a 0.3 eV neutrino, CLASS and CAMB agree for both matter species.

    Tolerance: as in test_agrees_with_camb.
    """
    cosmo = cosmology.FlatLambdaCDM(
        H0=70.0, Om0=0.3, Ob0=0.05, Tcmb0=2.7255, m_nu=[0.0, 0.0, 0.3] * u.eV
    )
    z = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    np.testing.assert_allclose(
        ClassGrowth(cosmo, matter_species=matter_species).growth_factor(z),
        growth_factor.CambGrowth(cosmo, matter_species=matter_species).growth_factor(z),
        rtol=1e-3,
    )


def test_normalised_and_decreasing():
    gf = ClassGrowth(LCDM)
    d = gf.growth_factor(Z)
    assert d[0] == pytest.approx(1.0, rel=1e-12)
    assert np.all(np.diff(d) < 0)
    assert isinstance(gf.growth_factor(1.0), float)
    assert isinstance(gf.growth_rate(1.0), float)


def test_default_matter_species_warns_for_massive_neutrinos():
    with pytest.warns(UserWarning, match="matter_species was not set for ClassGrowth"):
        gf = ClassGrowth(Planck13)
    assert gf.params["matter_species"] == "cb"


def test_bad_matter_species():
    with pytest.raises(ValueError, match="matter_species must be one of"):
        ClassGrowth(LCDM, matter_species="nu")


def test_redshift_range():
    gf = ClassGrowth(LCDM, z_max=5.0)
    assert gf.growth_factor(5.0) > 0
    with pytest.raises(ValueError, match="z_max"):
        gf.growth_factor(6.0)
    with pytest.raises(ValueError, match="z_max"):
        gf.growth_rate(-0.1)
    with pytest.raises(ValueError, match="z_max must be positive"):
        ClassGrowth(LCDM, z_max=0)


def test_rejects_unsupported_cosmology():
    with pytest.raises(ValueError, match="ClassGrowth will only work with LCDM or wCDM"):
        ClassGrowth(cosmology.w0wzCDM(H0=70, Om0=0.3, Ode0=0.7, Ob0=0.05, Tcmb0=2.7255))
    with pytest.raises(ValueError, match="baryon density"):
        ClassGrowth(cosmology.FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.7255))


def test_pickle_and_copy():
    gf = ClassGrowth(LCDM)
    d = gf.growth_factor(Z)
    for restored in (pickle.loads(pickle.dumps(gf)), copy.deepcopy(gf)):
        np.testing.assert_array_equal(restored.growth_factor(Z), d)


def test_mass_function_by_name():
    """``growth_model="ClassGrowth"`` works, and matches CambGrowth's mass function."""
    kw = {
        "z": 2.0,
        "transfer_model": "EH",
        "growth_params": {"matter_species": "cb"},
    }
    mf_class = MassFunction(growth_model="ClassGrowth", **kw)
    mf_camb = MassFunction(growth_model="CambGrowth", **kw)
    assert isinstance(mf_class.growth, ClassGrowth)
    # Tolerance: D agrees to < 1e-3, and dn/dm ~ exp(-nu^2/2) amplifies that at high mass.
    np.testing.assert_allclose(mf_class.dndm, mf_camb.dndm, rtol=1e-2)


def test_missing_classy(monkeypatch):
    monkeypatch.setattr(growth_factor, "HAVE_CLASS", False)
    with pytest.raises(ImportError, match="pip install hmf\\[class\\]"):
        ClassGrowth(LCDM)
    with pytest.raises(ValueError, match="classy isn't installed"):
        Transfer(growth_model="ClassGrowth")
