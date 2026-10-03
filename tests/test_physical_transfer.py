"""Physical tests of transfer functions, power spectra and background densities.

Covers the transfer models, the linear and non-linear (halofit) power spectrum and
the mean/critical densities. References are limits that every transfer function must
satisfy, accuracies quoted in the defining papers (Eisenstein & Hu 1998), analytic
results for power-law spectra, fundamental constants, and independent codes (CAMB).
"""

import numpy as np
import pytest
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM, Planck18
from scipy.special import gamma as gamma_fn

from hmf.cosmology.cosmo import Cosmology
from hmf.density_field import transfer_models as tm
from hmf.density_field.halofit import _get_spec
from hmf.density_field.transfer import Transfer
from hmf.halos.mass_definitions import BaseMassDefinition

# ---------------------------------------------------------------------------------------
# Transfer functions
# ---------------------------------------------------------------------------------------
ANALYTIC_MODELS = [tm.EH_BAO, tm.EH_NoBAO, tm.BBKS, tm.BondEfs]


@pytest.mark.parametrize("model", ANALYTIC_MODELS, ids=lambda m: m.__name__)
def test_transfer_tends_to_unity_on_large_scales(model):
    """Modes far outside the horizon at equality are unprocessed: T(k -> 0) = 1."""
    t = model(Planck18)
    lnt = t.lnt(np.log(np.array([1e-7, 1e-6])))
    # Tolerance: the leading correction is O(k/k_eq) with k_eq ~ 0.01 h/Mpc, i.e.
    # ~1e-4 at k = 1e-6 h/Mpc; measured < 1.1e-4 for every model.
    np.testing.assert_allclose(np.exp(lnt), 1.0, atol=2e-4)


@pytest.mark.parametrize("model", [tm.EH_NoBAO, tm.BBKS, tm.BondEfs], ids=lambda m: m.__name__)
def test_transfer_decreases_monotonically_without_baryon_oscillations(model):
    """Without acoustic oscillations, small-scale modes are always more suppressed."""
    t = model(Planck18)
    lnt = t.lnt(np.log(np.logspace(-5, 3, 500)))
    assert np.all(np.diff(lnt) < 0)


def test_eh_bao_and_nobao_agree_on_large_scales():
    """The BAO and no-BAO EH98 forms are the same function above the sound horizon."""
    k = np.logspace(-6, -3.5, 20)
    lnt_b = tm.EH_BAO(Planck18).lnt(np.log(k))
    lnt_n = tm.EH_NoBAO(Planck18).lnt(np.log(k))
    # Tolerance: both tend to T=1; at k < 3e-4 h/Mpc the difference is < 2e-5.
    np.testing.assert_allclose(np.exp(lnt_b - lnt_n), 1.0, atol=1e-4)


def test_eh_bao_and_nobao_agree_on_small_scales():
    """Below the Silk scale the oscillations are damped, so the forms agree.

    EH98 build the no-wiggle form to track the full form's envelope, with residuals
    of order 1-2% (their Sect. 4.2).
    """
    k = np.logspace(0, 2, 20)
    ratio = np.exp(tm.EH_BAO(Planck18).lnt(np.log(k)) - tm.EH_NoBAO(Planck18).lnt(np.log(k)))
    # Tolerance: EH98's quoted few-percent agreement; measured max 1.2%.
    np.testing.assert_allclose(ratio, 1.0, atol=2e-2)


def test_eh_sound_horizon_fit_matches_exact_integral():
    """EH98 Eq. 26 approximates the sound horizon of Eq. 6 "to 2%" (EH98 text)."""
    t = tm.EH_BAO(Planck18)
    s_exact = t.sound_horizon  # Mpc
    s_fit = t.sound_horizon_fit / Planck18.h  # Mpc/h -> Mpc
    assert s_fit == pytest.approx(s_exact, rel=0.02)


def test_eh_sound_horizon_close_to_planck_measurement():
    """The EH98 sound horizon at the drag epoch is close to Planck 2018's r_drag.

    Planck 2018 (VI, Table 2, TT,TE,EE+lowE+lensing) gives r_drag = 147.09 Mpc.
    """
    s = tm.EH_BAO(Planck18).sound_horizon
    # Tolerance: EH98's drag-redshift fit (their Eq. 4) is accurate to a few percent,
    # which propagates to a few percent in r_drag; measured +2.7%. 5% is a bound.
    assert s == pytest.approx(147.09, rel=0.05)


def test_bbks_matches_eh_zero_baryon_shape():
    """With no baryons, BBKS and the EH98 no-wiggle form describe the same CDM shape."""
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3, Ob0=1e-6, Tcmb0=2.7255)
    k = np.logspace(-3, 1, 40)
    lnt_bbks = tm.BBKS(cosmo, use_liddle_baryons=False).lnt(np.log(k))
    lnt_eh = tm.EH_NoBAO(cosmo).lnt(np.log(k))
    # Tolerance: BBKS was fit to ~10% accuracy and differs from the more accurate EH98
    # zero-baryon form by up to ~6% (measured 5.9% at k ~ 0.1 h/Mpc).
    np.testing.assert_allclose(np.exp(lnt_bbks - lnt_eh), 1.0, atol=0.1)


def test_bondefs_depends_only_on_k_over_shape_parameter():
    """BondEfs is a function of q = k/Gamma alone, with Gamma = Om0 h (EBW92 eq. 7).

    Two cosmologies with the same Om0 h give the same T(k), and changing Gamma shifts
    T exactly along k.
    """
    k = np.logspace(-3, 1, 40)
    # Gamma = 0.21 in both: the GIF LCDM (0.3, 0.7) and an Om0 h-matched (0.42, 0.5).
    t1 = tm.BondEfs(FlatLambdaCDM(H0=70.0, Om0=0.3, Ob0=0.04)).lnt(np.log(k))
    t2 = tm.BondEfs(FlatLambdaCDM(H0=50.0, Om0=0.42, Ob0=0.04)).lnt(np.log(k))
    np.testing.assert_allclose(t1, t2, rtol=1e-12, atol=0)

    # Gamma = 0.15: T(k; Gamma) = T(k * 0.21 / 0.15; 0.21).
    t3 = tm.BondEfs(FlatLambdaCDM(H0=50.0, Om0=0.3, Ob0=0.04)).lnt(np.log(k * 0.15 / 0.21))
    np.testing.assert_allclose(t3, t1, rtol=1e-12, atol=0)


@pytest.mark.parametrize("om0", [0.2, 0.3, 0.4])
def test_bondefs_matches_eh_zero_baryon_shape(om0):
    """In the CDM-only limit, BondEfs matches the EH98 zero-baryon transfer function.

    Gamma = Om0 h is the exact shape parameter only when the baryon fraction vanishes
    (EH98 Sect. 4.2), so the comparison is made with Ob0 -> 0. EH98's zero-baryon form
    (their eq. 29) matches CMBFAST to 1%.
    """
    cosmo = FlatLambdaCDM(H0=70.0, Om0=om0, Ob0=1e-6, Tcmb0=2.7255)
    k = np.logspace(-3, 0, 40)
    lnt_be = tm.BondEfs(cosmo).lnt(np.log(k))
    lnt_eh = tm.EH_NoBAO(cosmo).lnt(np.log(k))
    # Tolerance: EBW92 took their coefficients from a BE84 model with a 3% baryon
    # fraction, which EH98 (Fig. 6) show sits a few percent below the zero-baryon
    # curve. Jenkins et al. (1998, Sect. 3.1) quote a maximum difference in amplitude
    # of 6% between this fit and CMBFAST. Measured max 5.5% for every om0 here, at
    # k ~ 0.3-1 h/Mpc. The BE84 Omega=0.3 coefficients used before the fix are off by 16%.
    np.testing.assert_allclose(np.exp(lnt_be - lnt_eh), 1.0, atol=0.07)


def test_bondefs_reproduces_bond_efstathiou_standard_cdm_fit():
    """At Om0=1, h=0.75, BondEfs reproduces BE84's own fit for that model.

    BE84 Table 1 (row Omega=1, Omega_B=0.03, h=0.75) gives a=11.3, b=5.29, c=3.10 Mpc
    and nu=1.13 for their eq. 6, with k in 1/Mpc. EBW92 state that their eq. 7 fits
    standard CDM if Gamma = h.
    """
    h = 0.75
    k = np.logspace(-3, 0, 40)  # h/Mpc
    kmpc = k * h  # 1/Mpc, the units of BE84's coefficients
    t_be84 = (1 + (11.3 * kmpc + (5.29 * kmpc) ** 1.5 + (3.10 * kmpc) ** 2) ** 1.13) ** (-1 / 1.13)

    cosmo = FlatLambdaCDM(H0=100 * h, Om0=1.0, Ob0=0.03)
    t = np.exp(tm.BondEfs(cosmo).lnt(np.log(k)))
    # Tolerance: EBW92's coefficients are this row times Gamma = Omega h^2 = 0.5625
    # (6.36, 2.98, 1.74), rounded to two significant figures. Measured max 0.47% at
    # k <= 1 h/Mpc; the coefficients of the Omega=0.3 row, used before the fix, are 11% off.
    np.testing.assert_allclose(t, t_be84, rtol=0.01)


def test_power_spectrum_follows_primordial_slope_on_large_scales():
    """With T -> 1 on large scales, P(k) -> A k^{n_s}."""
    t = Transfer(
        transfer_model="EH", n=0.9667, lnk_min=np.log(1e-7), lnk_max=np.log(1e-4), dlnk=0.1
    )
    slope = np.gradient(np.log(t.power), np.log(t.k))
    # Tolerance: dlnT^2/dlnk ~ 2 k/k_eq, below 1e-3 at k < 1e-5 h/Mpc.
    np.testing.assert_allclose(slope[t.k < 1e-5], 0.9667, atol=1e-3)


def test_eh_matches_camb_power_spectrum():
    """CAMB and EH98 agree to the few-percent accuracy that EH98 quote.

    Both normalised to the same sigma_8; massless neutrinos so that the matter
    species doesn't matter.
    """
    cosmo = FlatLambdaCDM(H0=67.7, Om0=0.31, Ob0=0.049, Tcmb0=2.7255, m_nu=0)
    kw = {"cosmo_model": cosmo, "lnk_min": np.log(1e-4), "lnk_max": np.log(10), "dlnk": 0.05}
    p_camb = Transfer(
        transfer_model="CAMB", transfer_params={"extrapolate_with_eh": True}, **kw
    ).power
    p_eh = Transfer(transfer_model="EH", **kw).power
    # Tolerance: EH98 quote residuals of a few percent in T, so ~2x that in P; the
    # measured max is 3.8% (at the first BAO trough, k ~ 0.1 h/Mpc).
    np.testing.assert_allclose(p_eh / p_camb, 1.0, atol=0.06)


def test_power_scales_with_growth_squared():
    """Linear modes grow in proportion to D(z), so P(k, z) = D(z)^2 P(k, 0)."""
    t0 = Transfer(transfer_model="EH", z=0.0)
    t2 = Transfer(transfer_model="EH", z=2.0)
    # Tolerance: linear theory exact; floating-point only.
    np.testing.assert_allclose(t2.power, t0.power * t2.growth_factor**2, rtol=1e-12)
    assert t2.growth_factor < 1


# ---------------------------------------------------------------------------------------
# Halofit
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [-2.0, -1.5, -1.0])
def test_halofit_spectral_parameters_for_power_law(n):
    r"""For :math:`\Delta^2 = A k^{n+3}` the Gaussian-smoothed variance is analytic.

    :math:`\sigma^2(R) = \frac{A}{2}\Gamma\left(\frac{n+3}{2}\right) R^{-(n+3)}`, so the
    effective index is exactly n, the curvature is exactly 0, and
    :math:`k_{\rm nl} = [A\Gamma((n+3)/2)/2]^{-1/(n+3)}`.
    """
    amp = 0.3
    k = np.exp(np.arange(-12, 12, 0.02))
    knl, neff, curv = _get_spec(k, amp * k ** (n + 3))
    knl_exact = (amp * gamma_fn((n + 3) / 2) / 2) ** (-1 / (n + 3))
    # Tolerance: Simpson on a fine grid over a range extending well beyond the window;
    # measured errors < 2e-6.
    assert neff == pytest.approx(n, abs=1e-4)
    assert curv == pytest.approx(0.0, abs=1e-4)
    assert knl == pytest.approx(knl_exact, rel=1e-4)


def test_halofit_linear_at_high_redshift():
    """At high redshift quasi-linear scales are still linear: P_nl -> P_lin.

    At z = 10 the non-linear scale is k_nl ~ 200 h/Mpc and Delta^2_lin(k) < 0.01 for
    k < 0.1 h/Mpc, so the halofit correction there must be small.
    """
    t = Transfer(transfer_model="EH", z=10.0, lnk_min=np.log(1e-4), lnk_max=np.log(1e3), dlnk=0.05)
    mask = t.k < 0.1
    # Tolerance: the one-halo term is suppressed by (k/k_nl)^3 and the quasi-linear
    # correction is O(Delta^2_lin) < 1e-2; measured max deviation 0.6%.
    np.testing.assert_allclose(t.nonlinear_power[mask] / t.power[mask], 1.0, atol=1e-2)


# ---------------------------------------------------------------------------------------
# Background densities
# ---------------------------------------------------------------------------------------
def _rho_crit_h2_from_constants():
    """3 H^2 / (8 pi G) for H0 = 100 km/s/Mpc, in Msun / Mpc^3 (i.e. h^2 Msun/Mpc^3)."""
    h100 = 100 * u.km / u.s / u.Mpc
    return (3 * h100**2 / (8 * np.pi * const.G)).to(u.Msun / u.Mpc**3).value


def test_critical_density_matches_fundamental_constants():
    """rho_crit,0 = 3H0^2/(8 pi G) = 2.775e11 h^2 Msun/Mpc^3 (e.g. PDG)."""
    rho_c = BaseMassDefinition.critical_density(0, Planck18)
    # Tolerance: the PDG value 2.77537e11 is quoted to 6 significant figures.
    assert rho_c == pytest.approx(2.77537e11, rel=1e-5)
    assert rho_c == pytest.approx(_rho_crit_h2_from_constants(), rel=1e-10)


@pytest.mark.parametrize("cosmo", [Planck18, FlatLambdaCDM(H0=70, Om0=0.25, Tcmb0=0)])
def test_mean_density_is_omega_m_times_critical(cosmo):
    """The comoving mean matter density is Omega_m0 rho_crit,0."""
    c = Cosmology(cosmo_model=cosmo)
    # Tolerance: floating-point / unit-conversion rounding.
    assert c.mean_density0 == pytest.approx(cosmo.Om0 * _rho_crit_h2_from_constants(), rel=1e-10)


@pytest.mark.parametrize("z", [0.5, 1.0, 3.0, 10.0])
def test_mean_density_dilutes_as_volume(z):
    """Matter is conserved, so the physical mean density scales as (1+z)^3.

    BaseMassDefinition computes it as Omega_m(z) rho_crit(z); compare to the z=0
    comoving density diluted by the expansion.
    """
    rho_z = BaseMassDefinition.mean_density(z, Planck18)
    rho_0 = Cosmology(cosmo_model=Planck18).mean_density0
    # Tolerance: astropy's Om(z) and critical_density(z) are exact to rounding.
    assert rho_z == pytest.approx(rho_0 * (1 + z) ** 3, rel=1e-10)
