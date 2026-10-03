"""Physical tests of the warm dark matter models.

References: the defining property of the half-mode scale (Schneider et al. 2012:
the WDM/CDM transfer function equals 1/2 there), the large-scale CDM limit, the
CDM limit of an infinitely heavy particle, the free-streaming and half-mode
masses tabulated by Schneider et al. (2012), the smoothing scales and masses quoted
by Bode, Ostriker & Turok (2001; arXiv:astro-ph/0010389v3), and the
particle-temperature form of the break scale in Viel et al. (2005;
arXiv:astro-ph/0501562v2, eq. 7).
"""

import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

from hmf import MassFunction
from hmf.alternatives import wdm

MODELS = [wdm.Viel05, wdm.Bode01]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("mx", [0.5, 1.0, 3.0, 10.0])
def test_transfer_is_one_half_at_half_mode_scale(model, mx):
    """The half-mode wavenumber k_hm = 2 pi / lambda_hm is where T_WDM/T_CDM = 1/2."""
    w = model(mx=mx)
    # Tolerance: closed form inversion; floating-point only.
    assert w.transfer(2 * np.pi / w.lam_hm) == pytest.approx(0.5, rel=1e-10)


@pytest.mark.parametrize("model", MODELS)
def test_transfer_shape(model):
    """Free-streaming only erases small-scale power.

    T -> 1 on large scales, falls monotonically, and tends to zero far below the
    free-streaming length.
    """
    w = model(mx=1.0)
    k = np.logspace(-4, 3, 200)
    t = w.transfer(k)
    # Tolerance: at k = 1e-4 h/Mpc, (alpha k)^{2 nu} ~ 1e-13 so T = 1 - O(1e-12).
    assert t[0] == pytest.approx(1.0, abs=1e-10)
    assert np.all(np.diff(t) < 0)
    assert t[-1] < 1e-6


# Schneider et al. (2012) Table 1: (m_WDM [keV], M_fs, M_hm [Msun/h]).
_SCHNEIDER12_TABLE1 = [
    (1.25, 2.3e6, 6.3e9),
    (1.0, 4.9e6, 1.3e10),
    (0.75, 1.3e7, 3.4e10),
    (0.5, 4.9e7, 1.3e11),
    (0.25, 5.0e8, 1.3e12),
]
# Their WMAP7 cosmology (Sect. 3).
_SCHNEIDER12_COSMO = FlatLambdaCDM(H0=70.4, Om0=0.2726, Ob0=0.046, Tcmb0=2.725)


@pytest.mark.parametrize(("mx", "m_fs", "m_hm"), _SCHNEIDER12_TABLE1)
def test_free_streaming_and_half_mode_masses_match_schneider12(mx, m_fs, m_hm):
    """M_fs and M_hm reproduce Table 1 of Schneider et al. (2012) for their cosmology."""
    w = wdm.Viel05(mx=mx, cosmo=_SCHNEIDER12_COSMO)
    # Tolerance: the table quotes 2 significant figures (up to 4% rounding) and does
    # not state the exact mean density used; measured deviations are 3-8%.
    assert w.m_fs == pytest.approx(m_fs, rel=0.1)
    assert w.m_hm == pytest.approx(m_hm, rel=0.1)


def test_half_mode_to_free_streaming_ratio():
    """Schneider et al. (2012) Eq. 8: lambda_hm ~ 13.93 lambda_fs for nu = 1.12."""
    w = wdm.Viel05(mx=1.0)
    # Tolerance: the paper quotes 4 significant figures.
    assert w.lam_hm / w.lam_eff_fs == pytest.approx(13.93, abs=5e-3)


def test_half_mode_scale_shrinks_with_particle_mass():
    """Heavier (colder) particles free-stream less: lambda_hm and M_hm fall with m_x."""
    mx = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    lam = np.array([wdm.Viel05(mx=m).lam_hm for m in mx])
    mhm = np.array([wdm.Viel05(mx=m).m_hm for m in mx])
    assert np.all(np.diff(lam) < 0)
    assert np.all(np.diff(mhm) < 0)


@pytest.fixture(scope="module")
def mf_pair():
    common = {"transfer_model": "EH", "Mmin": 7, "Mmax": 15, "dlog10m": 0.05}
    cdm = MassFunction(**common)
    warm = wdm.MassFunctionWDM(wdm_mass=1.0, **common)
    return cdm, warm


def test_wdm_sigma_suppressed_and_converges_to_cdm(mf_pair):
    """WDM removes small-scale power, so sigma_WDM <= sigma_CDM, converging at high M."""
    cdm, warm = mf_pair
    # Before the sigma_8 normalisation, T_WDM <= 1 at every k, so the variance can only
    # be lower, strictly so at every mass.
    assert np.all(warm._unn_sigma0 < cdm._unn_sigma0)
    # Both are then normalised to the same sigma_8. WDM slightly suppresses
    # sigma(8 Mpc/h) itself, so its amplitude is scaled up by a factor just above 1
    # (measured 1 + 1.9e-4 for m_x = 1 keV).
    renorm = warm._normalisation / cdm._normalisation
    assert 1 < renorm < 1 + 1e-3
    # The suppression weakens monotonically with scale, so sigma_WDM/sigma_CDM rises
    # with mass towards that renormalisation factor.
    ratio = warm.sigma / cdm.sigma
    assert np.all(np.diff(ratio) > 0)
    assert np.all(ratio <= renorm)
    # Tolerance: at M = 1e15 (R ~ 13 Mpc/h >> lambda_hm ~ 0.7 Mpc/h) only the sigma_8
    # renormalisation remains; measured 1.1e-4.
    assert warm.sigma[-1] == pytest.approx(cdm.sigma[-1], rel=1e-3)
    # Well below M_hm (~1e10) sigma is strongly suppressed.
    assert warm.sigma[0] < 0.7 * cdm.sigma[0]


def test_wdm_mass_function_suppressed_below_half_mode_mass(mf_pair):
    """Haloes below M_hm are suppressed relative to CDM; massive haloes are not."""
    cdm, warm = mf_pair
    low = warm.m < warm.wdm.m_hm / 10
    high = warm.m > warm.wdm.m_hm * 1000
    assert np.any(low)
    assert np.any(high)
    assert np.all(warm.dndm[low] < cdm.dndm[low])
    # Tolerance: above 1000 M_hm the transfer function deviates from 1 by < 1e-4.
    np.testing.assert_allclose(warm.dndm[high], cdm.dndm[high], rtol=1e-2)


def test_wdm_tends_to_cdm_for_heavy_particle():
    """As m_x -> infinity the particle is cold and WDM must reduce to CDM."""
    common = {"transfer_model": "EH", "Mmin": 7, "Mmax": 15, "dlog10m": 0.1}
    cdm = MassFunction(**common)
    warm = wdm.MassFunctionWDM(wdm_mass=1e4, **common)
    # Tolerance: for m_x = 10^4 keV, lambda_hm ~ 1e-5 Mpc/h, far below the smallest
    # radius (R ~ 0.04 Mpc/h at 1e7 Msun/h); measured max difference 3e-7.
    np.testing.assert_allclose(warm.dndm, cdm.dndm, rtol=1e-4)


@pytest.mark.parametrize("model", ["Schneider12_vCDM", "Schneider12", "Lovell14"])
def test_recalibrations_only_suppress_and_vanish_at_high_mass(model, mf_pair):
    """Empirical WDM recalibrations only suppress, and vanish at high mass.

    They multiply dn/dm by a factor in (0, 1] that rises monotonically with mass and
    tends to 1 for M >> M_hm.
    """
    _, warm = mf_pair
    cls = getattr(wdm, model)
    m = np.logspace(6, 16, 100)
    dndm0 = np.ones_like(m)
    factor = cls(m=m, dndm0=dndm0, wdm=warm.wdm).dndm_alter()
    assert np.all(factor > 0)
    assert np.all(factor <= 1)
    assert np.all(np.diff(factor) > 0)
    # Tolerance: at M = 1e6 M_hm the factor is 1 - O(beta gamma 1e-6) ~ 3e-6.
    assert factor[-1] == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("model", MODELS)
def test_transfer_small_scale_asymptote_is_independent_of_nu(model):
    """Far below the break, T -> (alpha k)^-10 whatever nu is.

    [1 + (alpha k)^{2 nu}]^{-5/nu} -> (alpha k)^{2 nu * (-5/nu)} = (alpha k)^{-10}
    (Bode et al. 2001, eq. A8).
    """
    w = model(mx=1.0)
    x = np.array([1e3, 1e4])
    # Tolerance: the relative correction is (5/nu) (alpha k)^{-2 nu} < 1e-6 here.
    np.testing.assert_allclose(w.transfer(x / w.lam_eff_fs) * x**10, 1.0, rtol=1e-5)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("k", [0.1, 10.0, 100.0])
def test_transfer_tends_to_one_for_heavy_particle(model, k):
    """At fixed k, T -> 1 as m_x -> infinity (the particle becomes cold)."""
    t = np.array([model(mx=m).transfer(k) for m in [1.0, 10.0, 100.0, 1e4, 1e6]])
    assert np.all(np.diff(t) >= 0)
    # Tolerance: for m_x = 1e6 keV, alpha ~ 1e-8 Mpc/h so 1 - T < 1e-10 even at k = 100.
    assert t[-1] == pytest.approx(1.0, abs=1e-8)


# Bode, Ostriker & Turok (2001), astro-ph/0010389v3. Their reference cosmology
# (Sect. 2: "Omega_DM = 0.3 and h = 0.65 unless otherwise stated"). Ob0 = 0 so that
# hmf's Omega_X = Om0 - Ob0 is their Omega_X = 0.3.
_BODE01_COSMO = FlatLambdaCDM(H0=65.0, Om0=0.3, Ob0=0.0, Tcmb0=2.725)

# Smoothing scale R_s, "the comoving half-wavelength of the mode for which the linear
# perturbation amplitude is suppressed by two" (Bode et al. 2001, text above eq. 1),
# quoted in Sect. 1 (p. 5): R_s = 2.3 and 1.0 Mpc/h for m_X = 175 and 350 eV, and
# 0.19 Mpc/h for m_X = 1.5 keV.
_BODE01_RS = [(0.175, 2.3), (0.35, 1.0), (1.5, 0.19)]


@pytest.mark.parametrize(("mx", "r_s"), _BODE01_RS)
def test_bode01_smoothing_scale_matches_paper(mx, r_s):
    """Half the half-mode wavelength reproduces Bode et al.'s quoted R_s."""
    w = wdm.Bode01(mx=mx, cosmo=_BODE01_COSMO)
    # Tolerance: the quoted R_s come from eq. 1 (R_s = 0.31 (keV/m_X)^1.15 Mpc/h),
    # whose prefactor is ~5% above what the eq. A8/A9 fit gives (0.294); measured
    # deviations are 1.5-5%.
    assert w.lam_hm / 2 == pytest.approx(r_s, rel=0.06)


# Bode et al. (2001), Sect. 7: haloes are suppressed below (4/3) pi rho_b R_s^3,
# "4 x 10^11 and 4 x 10^12 Msun/h for m_X = 350 and 175 eV respectively".
_BODE01_MASSES = [(0.35, 4e11), (0.175, 4e12)]


@pytest.mark.parametrize(("mx", "mass"), _BODE01_MASSES)
def test_bode01_smoothing_mass_matches_paper(mx, mass):
    """The half-mode mass is Bode et al.'s (4/3) pi rho_b R_s^3."""
    w = wdm.Bode01(mx=mx, cosmo=_BODE01_COSMO)
    # Tolerance: the masses are quoted to one significant figure (up to 12.5%) and,
    # being R_s^3, carry three times the ~5% eq. 1 vs eq. A9 offset above; measured
    # deviations are -17% and -9%.
    assert w.m_hm == pytest.approx(mass, rel=0.2)


@pytest.mark.parametrize(
    ("model", "exponent"),
    [
        # Bode et al. (2001), eqs 1 and A9: R_s, alpha ~ m_X^-1.15.
        (wdm.Bode01, -1.15),
        # Viel et al. (2005), eq. 7: alpha ~ m_x^-1.11.
        (wdm.Viel05, -1.11),
    ],
)
def test_half_mode_scale_power_law_in_particle_mass(model, exponent):
    """lambda_hm and M_hm scale with m_x with each paper's exponent (and 3x it)."""
    mx = np.array([0.3, 1.0, 3.0, 10.0])
    lam = np.array([model(mx=m).lam_hm for m in mx])
    mhm = np.array([model(mx=m).m_hm for m in mx])
    # Tolerance: an exact power law; floating-point only.
    np.testing.assert_allclose(np.diff(np.log(lam)) / np.diff(np.log(mx)), exponent, rtol=1e-10)
    np.testing.assert_allclose(np.diff(np.log(mhm)) / np.diff(np.log(mx)), 3 * exponent, rtol=1e-10)


def _viel05_alpha_from_temperature(mx, omega_x_h2, h):
    """Viel et al. (2005), eq. 7, first line, for a thermal relic, in Mpc/h.

    alpha = 0.24 [(m_x/T_x)/(1 keV/T_nu)]^-0.83 [omega_x/(0.25 * 0.7^2)]^-0.16 Mpc, with
    the relic temperature fixed by the density through their eq. 2,
    omega_x = (T_x/T_nu)^3 (m_x / 94 eV).
    """
    # Check of eq. 2: m_x = 1 keV, omega_x = 0.25 * 0.7^2 gives T_x/T_nu = 0.2258,
    # as quoted (0.226) in the caption of their Fig. 1.
    tx_over_tnu = (omega_x_h2 * 0.094 / mx) ** (1 / 3)
    alpha_mpc = 0.24 * (mx / tx_over_tnu) ** -0.83 * (omega_x_h2 / (0.25 * 0.7**2)) ** -0.16
    return alpha_mpc * h


@pytest.mark.parametrize("mx", [0.5, 1.0, 2.0, 5.0])
@pytest.mark.parametrize(("om", "h"), [(0.25, 0.7), (0.3, 0.65), (0.22, 0.72)])
def test_viel05_break_scale_matches_temperature_form(mx, om, h):
    """Viel05's alpha equals eq. 7's thermal-relic form written in terms of m_x/T_x.

    The second line of eq. 7 (used by the code) is the first line with T_x eliminated
    via eq. 2; the paper rounds the resulting exponents (-0.83 * 4/3 = -1.107 -> -1.11,
    etc.).
    """
    cosmo = FlatLambdaCDM(H0=100 * h, Om0=om + 0.05, Ob0=0.05, Tcmb0=2.725)
    w = wdm.Viel05(mx=mx, cosmo=cosmo)
    # Tolerance: the rounded exponents and the 2-figure prefactors (0.24, 0.049) give a
    # measured spread of up to 1.5% over this range.
    assert w.lam_eff_fs == pytest.approx(_viel05_alpha_from_temperature(mx, om * h**2, h), rel=0.02)


def test_bode01_and_viel05_break_scales_agree_at_bode_reference():
    """At Bode's reference cosmology and 1 keV the two papers' break scales agree.

    Viel et al. (2005) say their eq. 7 "is close to that of [Bode et al. 2001]":
    0.048 (0.3/0.4)^0.15 = 0.0460 vs 0.049 (0.3/0.25)^0.11 (0.65/0.7)^1.22 = 0.0457.
    """
    b = wdm.Bode01(mx=1.0, cosmo=_BODE01_COSMO)
    v = wdm.Viel05(mx=1.0, cosmo=_BODE01_COSMO)
    # Tolerance: measured 0.6% difference.
    assert b.lam_eff_fs == pytest.approx(v.lam_eff_fs, rel=0.01)


def test_bode01_and_viel05_differ():
    """Bode01 (nu = 1.2) and Viel05 (nu = 1.12) give different half-mode scales.

    With the break scales equal (previous test), lambda_hm differs only through the
    factor (2^{nu/5} - 1)^{-1/(2 nu)} from T = 1/2: 2.217 for nu = 1.12 against 2.038
    for nu = 1.2, so lambda_hm(Viel05)/lambda_hm(Bode01) = 1.088 and the half-mode
    masses differ by 1.088^3 = 1.29.
    """
    b = wdm.Bode01(mx=1.0, cosmo=_BODE01_COSMO)
    v = wdm.Viel05(mx=1.0, cosmo=_BODE01_COSMO)
    assert b.params["nu"] == 1.2
    assert v.params["nu"] == 1.12
    # Tolerance: the ~0.6% break-scale difference above, and 3x that for the masses.
    assert v.lam_hm / b.lam_hm == pytest.approx(1.088, rel=0.01)
    assert v.m_hm / b.m_hm == pytest.approx(1.29, rel=0.03)


@pytest.mark.parametrize("model", MODELS)
def test_mu_is_deprecated_alias_of_nu(model):
    """The old parameter name ``mu`` still works, with a warning, and means ``nu``."""
    with pytest.warns(DeprecationWarning, match="'mu' has been renamed to 'nu'"):
        old = model(mx=1.0, mu=1.3)
    new = model(mx=1.0, nu=1.3)
    assert old.params["nu"] == 1.3
    assert "mu" not in old.params
    k = np.logspace(-1, 2, 20)
    np.testing.assert_array_equal(old.transfer(k), new.transfer(k))


@pytest.mark.parametrize("model", MODELS)
def test_mu_and_nu_together_is_an_error(model):
    with pytest.raises(ValueError, match="both 'mu' and 'nu'"):
        model(mx=1.0, mu=1.3, nu=1.2)


def test_mu_via_wdm_params_in_framework():
    """``mu`` passed through ``wdm_params`` reaches the model as ``nu``."""
    common = {"transfer_model": "EH", "wdm_mass": 1.0, "wdm_model": "Bode01"}
    new = wdm.TransferWDM(wdm_params={"nu": 1.3}, **common)
    old = wdm.TransferWDM(wdm_params={"mu": 1.3}, **common)
    with pytest.warns(DeprecationWarning, match="'mu' has been renamed to 'nu'"):
        old_wdm = old.wdm
    assert old_wdm.params["nu"] == 1.3
    np.testing.assert_allclose(old.power, new.power, rtol=1e-12)
