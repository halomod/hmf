"""Physical tests of halo mass definitions and conversions between them.

References: the spherical-collapse virial overdensity 18 pi^2 (and the Bryan & Norman
1998 fit), identities between mean- and critical-density definitions, and an NFW
enclosed-mass solve written here from the profile (not from hmf's code). The
Hu & Kravtsov (2003) fitting formula gives a second independent reference.
"""

import warnings

import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM, LambdaCDM, Planck15
from scipy.optimize import brentq

from hmf.halos import mass_definitions as md

EDS = LambdaCDM(H0=70.0, Om0=1.0, Ode0=0.0, Tcmb0=0.0)


# ---------------------------------------------------------------------------------------
# Overdensities
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("z", [0.0, 1.0, 5.0])
def test_virial_overdensity_is_18pi2_in_eds(z):
    """Spherical top-hat collapse in EdS virialises at Delta = 18 pi^2 ~ 177.65.

    In EdS rho_crit = rho_mean, so the overdensity is the same w.r.t. either.
    """
    vir = md.SOVirial()
    # Tolerance: exact closed form; floating-point only.
    assert vir.halo_overdensity_crit(z, EDS) == pytest.approx(18 * np.pi**2, rel=1e-12)
    assert vir.halo_overdensity_mean(z, EDS) == pytest.approx(18 * np.pi**2, rel=1e-12)


def test_virial_overdensity_tends_to_eds_at_high_z():
    """In LCDM, Omega_m(z) -> 1 at high z, so Delta_vir -> 18 pi^2."""
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3, Tcmb0=0.0)
    vir = md.SOVirial()
    # Tolerance: at z = 20, 1 - Omega_m ~ 2e-4, changing Delta_c by 82 x 2e-4 ~ 0.02.
    assert vir.halo_overdensity_crit(20.0, cosmo) == pytest.approx(18 * np.pi**2, abs=0.05)


def test_virial_overdensity_lcdm_today():
    r"""Bryan & Norman (1998): for Omega_m = 0.3 (flat), Delta_c ~ 100, Delta_m ~ 330.

    The values for this cosmology are widely quoted (e.g. Bryan & Norman 1998 Fig. 1;
    Diemer & Kravtsov's COLOSSUS docs give 102 and 337 for Planck-like Omega_m).
    """
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3, Tcmb0=0.0)
    vir = md.SOVirial()
    # Bounds: literature values for Omega_m in [0.27, 0.32] range over 95-105
    # (critical) and 320-360 (mean).
    assert 95 < vir.halo_overdensity_crit(0.0, cosmo) < 105
    assert 320 < vir.halo_overdensity_mean(0.0, cosmo) < 360


def test_virial_overdensity_mean_decreases_with_redshift():
    """The virial overdensity approaches 18 pi^2 at high z from both sides.

    Relative to the mean density it falls towards 18 pi^2, while relative to the
    critical density it rises towards it.
    """
    z = np.linspace(0, 5, 20)
    vir = md.SOVirial()
    d_mean = np.array([vir.halo_overdensity_mean(zz, Planck15) for zz in z])
    d_crit = np.array([vir.halo_overdensity_crit(zz, Planck15) for zz in z])
    assert np.all(np.diff(d_mean) < 0)
    assert np.all(np.diff(d_crit) > 0)
    assert np.all(d_mean > 18 * np.pi**2)
    assert np.all(d_crit < 18 * np.pi**2)


@pytest.mark.parametrize("z", [0.0, 0.5, 2.0])
def test_mean_and_critical_definitions_are_equivalent(z):
    """SOCritical(D) and SOMean(D / Omega_m(z)) define the same density threshold."""
    crit = md.SOCritical(overdensity=200)
    mean = md.SOMean(overdensity=200 / Planck15.Om(z))
    # Tolerance: floating-point only.
    assert crit.halo_density(z, Planck15) == pytest.approx(
        mean.halo_density(z, Planck15), rel=1e-12
    )


def test_radius_mass_relation_encloses_the_halo_density():
    """r_to_m and m_to_r define a sphere whose mean density is the halo density."""
    mdef = md.SOCritical(overdensity=500)
    m = np.logspace(10, 15, 6)
    r = mdef.m_to_r(m, 0.5, Planck15)
    mean_enclosed_density = m / (4 * np.pi * r**3 / 3)
    # Tolerance: floating-point only.
    np.testing.assert_allclose(mean_enclosed_density, mdef.halo_density(0.5, Planck15), rtol=1e-12)


# ---------------------------------------------------------------------------------------
# NFW mass conversions
# ---------------------------------------------------------------------------------------
def _nfw_convert(m, c, mdef_in, mdef_out, z, cosmo):
    r"""Convert an NFW halo mass between SO definitions, from the profile directly.

    :math:`M(<r) = 4\pi\rho_s r_s^3\mu(r/r_s)`, :math:`\mu(x)=\ln(1+x)-x/(1+x)`; the
    new radius is where the mean enclosed density equals the new halo density.
    """
    rho_in = mdef_in.halo_density(z, cosmo)
    rho_out = mdef_out.halo_density(z, cosmo)

    def mu(x):
        return np.log(1 + x) - x / (1 + x)

    r_in = (3 * m / (4 * np.pi * rho_in)) ** (1 / 3)
    rs = r_in / c
    rho_s = m / (4 * np.pi * rs**3 * mu(c))
    x = brentq(lambda x: 3 * rho_s * mu(x) / x**3 - rho_out, 1e-4, 1e4)
    return 4 * np.pi * rho_s * rs**3 * mu(x), x


def _hu_kravtsov_x(f):
    r"""Hu & Kravtsov (2003) Eq. C11 inverse of :math:`f(x) = x^3[\ln(1+1/x) - 1/(1+x)]`."""
    a1, a2, a3, a4 = 0.5116, -0.4283, -3.13e-3, -3.52e-5
    p = a2 + a3 * np.log(f) + a4 * np.log(f) ** 2
    return 1 / np.sqrt(a1 * f ** (2 * p) + 0.5625) + 2 * f


CONVERSIONS = [
    (md.SOMean(overdensity=200), md.SOCritical(overdensity=200)),
    (md.SOMean(overdensity=200), md.SOCritical(overdensity=500)),
    (md.SOCritical(overdensity=200), md.SOMean(overdensity=200)),
    (md.SOMean(overdensity=200), md.SOVirial()),
    (md.SOCritical(overdensity=500), md.SOMean(overdensity=200)),
]


@pytest.mark.parametrize(("mdef_in", "mdef_out"), CONVERSIONS, ids=str)
@pytest.mark.parametrize("z", [0.0, 1.0])
@pytest.mark.parametrize("c", [4.0, 10.0])
def test_change_definition_matches_nfw_solve(mdef_in, mdef_out, z, c):
    """change_definition (with a given concentration) solves the NFW mass equation."""
    m = 1e13
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_new, _, c_new = mdef_in.change_definition(m, mdef_out, c=c, z=z, cosmo=Planck15)
    m_ref, x_ref = _nfw_convert(m, c, mdef_in, mdef_out, z, Planck15)
    # Tolerance: both are root-finds of the same smooth equation (brentq, xtol ~ 1e-12
    # relative); 1e-6 allows for halomod's profile-normalisation conventions.
    assert m_new == pytest.approx(m_ref, rel=1e-6)
    assert c_new == pytest.approx(x_ref, rel=1e-6)


@pytest.mark.parametrize("c", [6.0, 10.0, 15.0])
def test_change_definition_matches_hu_kravtsov(c):
    r"""Compare to the Hu & Kravtsov (2003) Appendix C fitting formula.

    For an NFW halo with concentration c in definition v, the mass in definition h is
    :math:`M_h = M_v \frac{\Delta_h}{\Delta_v}(c\,x_h)^{-3}`, where
    :math:`x_h = x(f_h)` (their Eq. C11) with
    :math:`f_h = \frac{\Delta_h}{\Delta_v} f(1/c)` and
    :math:`f(x) = x^3[\ln(1+1/x) - 1/(1+x)]` (Eq. C10).
    """
    mdef_in = md.SOMean(overdensity=200)
    mdef_out = md.SOCritical(overdensity=500)
    z = 0.0
    m = 1e14
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_new = mdef_in.change_definition(m, mdef_out, c=c, z=z, cosmo=Planck15)[0]

    ratio = mdef_out.halo_density(z, Planck15) / mdef_in.halo_density(z, Planck15)
    xv = 1 / c
    f_v = xv**3 * (np.log(1 + 1 / xv) - 1 / (1 + xv))
    x_h = _hu_kravtsov_x(ratio * f_v)
    m_hk = m * ratio * (c * x_h) ** -3
    # Tolerance: HK03 quote Eq. C11 to <0.3% in x for typical concentrations, i.e.
    # ~1% in M (M ~ x^-3); measured max 0.4% for 6 <= c <= 15 (the fit degrades to
    # ~1% for c <~ 4, outside the range tested here).
    assert m_new == pytest.approx(m_hk, rel=1e-2)


@pytest.mark.parametrize("c", [3.0, 6.0, 12.0])
def test_higher_overdensity_gives_smaller_mass(c):
    """A higher density contour encloses less of the same halo: M_500c < M_200c < M_200m."""
    m200m = 1e14
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m200c = md.SOMean(overdensity=200).change_definition(
            m200m, md.SOCritical(overdensity=200), c=c, z=0.0, cosmo=Planck15
        )[0]
        m500c = md.SOMean(overdensity=200).change_definition(
            m200m, md.SOCritical(overdensity=500), c=c, z=0.0, cosmo=Planck15
        )[0]
    assert m500c < m200c < m200m


def test_change_definition_round_trip():
    """Converting to another definition and back recovers the original halo."""
    mdef_a = md.SOMean(overdensity=200)
    mdef_b = md.SOCritical(overdensity=500)
    m = np.array([1e12, 1e13, 1e14])
    c = np.array([8.0, 6.0, 4.5])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_b, _, c_b = mdef_a.change_definition(m, mdef_b, c=c, z=0.3, cosmo=Planck15)
        m_back, _, c_back = mdef_b.change_definition(m_b, mdef_a, c=c_b, z=0.3, cosmo=Planck15)
    # Tolerance: two root-finds; 1e-6.
    np.testing.assert_allclose(m_back, m, rtol=1e-6)
    np.testing.assert_allclose(c_back, c, rtol=1e-6)
