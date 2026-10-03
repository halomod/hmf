"""Physical tests of the halo mass function fitting functions.

Every test here checks a fit against something known independently of the code: an
exact normalisation (mass conservation), a parameter choice that reduces one fit to
another, a number quoted in the defining paper, or a physically required bound or
trend. None of the reference values are produced by re-implementing the fit itself.
"""

import inspect

import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import simpson
from scipy.optimize import brentq

from hmf import MassFunction
from hmf.halos import mass_definitions as md
from hmf.mass_function import fitting_functions as ff

DELTA_C = 1.68647

ALL_FITS = [
    cls
    for _, cls in inspect.getmembers(
        ff,
        lambda m: (
            inspect.isclass(m)
            and issubclass(m, ff.BaseFittingFunction)
            and m is not ff.BaseFittingFunction
        ),
    )
]

# Fits whose amplitude multiplies f(sigma) by a mass-dependent mass-conversion factor
# (Bocquet16 critical-overdensity variants), so that f is not a function of sigma alone.
_MASS_DEPENDENT_FITS = {
    ff.Bocquet200cDMOnly,
    ff.Bocquet200cHydro,
    ff.Bocquet500cDMOnly,
    ff.Bocquet500cHydro,
}

# Fits whose sim_definition is SOGeneric need an explicit SO mass definition when they
# are instantiated directly.
_NEEDS_SO_MDEF = {ff.Tinker08, ff.Tinker10, ff.Watson, ff.Behroozi}


def _make_fit(cls, nu2, m=None, z=None, **kwargs):
    """Instantiate a fit at its natural redshift with whatever extra inputs it needs."""
    if z is None:
        z = 7.0 if cls is ff.Yung24 else 0.0  # Yung24 is only defined for 6 <= z <= 19
    if m is None:
        m = np.full_like(nu2, 1e12)
    if cls in _NEEDS_SO_MDEF and "mass_definition" not in kwargs:
        kwargs["mass_definition"] = md.SOMean(overdensity=200)
    return cls(nu2=nu2, m=m, z=z, n_eff=np.full_like(nu2, -2.0), **kwargs)


def _mass_fraction(fit, lnnu):
    r"""Total mass fraction in haloes, :math:`\int_0^\infty f(\sigma)\, d\ln\nu`.

    Note :math:`d\ln\sigma^{-1} = d\ln\nu`. The integral is done with Simpson's rule
    over ``lnnu``; below the first point ``f`` is a power law in ``nu`` for every
    normalised fit, so the tail is added analytically as ``f(nu_0) / s`` with ``s``
    the local log-slope at ``nu_0``.
    """
    f = fit.fsigma
    slope = (np.log(f[1]) - np.log(f[0])) / (lnnu[1] - lnnu[0])
    return simpson(f, x=lnnu) + f[0] / slope


# A grid in ln(nu) that resolves f(sigma) everywhere. Above nu = 20 every fit is below
# exp(-0.4 * 20^2 / 2) ~ 1e-35 of its peak.
LNNU = np.linspace(np.log(1e-10), np.log(20.0), 40001)
NU2 = np.exp(2 * LNNU)


# ---------------------------------------------------------------------------------------
# Normalisation: fits that claim to place all mass in haloes
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("cls", [ff.PS, ff.SMT, ff.ST, ff.Manera, ff.Peacock])
def test_normalised_fits_conserve_mass(cls):
    """Fits normalised by construction put exactly all the mass into haloes.

    For PS this is the Press-Schechter "fudge factor of 2"; SMT/ST and Manera use the
    analytic normalisation of the SMT form (A=None); Peacock (2007) is built as
    -dF/dnu of a cumulative F with F(0) = 1.
    """
    assert cls.normalized
    # Tolerance: Simpson's rule with 4e4 points on a smooth integrand gives ~1e-12;
    # 1e-6 leaves room for the analytic low-nu tail approximation.
    assert _mass_fraction(_make_fit(cls, NU2), LNNU) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("z", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("overdensity", [200, 250, 400, 800, 1600, 3200])
def test_tinker10_normalisation_away_from_tabulated_alpha(z, overdensity):
    """Tinker+10 normalise f(nu) analytically so that all mass is in haloes.

    Away from z=0 (and for non-tabulated overdensities) hmf computes the normalisation
    alpha from the other parameters (Tinker+10 Eq. 8 with their constraint that the
    integral is unity), so the mass fraction must be exactly one.
    """
    fit = ff.Tinker10(nu2=NU2, z=z, mass_definition=md.SOMean(overdensity=overdensity))
    # Tolerance: numerical integration only (see test above).
    assert _mass_fraction(fit, LNNU) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("overdensity", [200, 300, 400, 600, 800, 1200, 1600, 2400, 3200])
def test_tinker10_published_alpha_normalises(overdensity):
    """The published z=0 alpha values (Tinker+10 Table 4) normalise f(nu) to unity.

    At z=0 and tabulated overdensities hmf uses the paper's alpha directly, so this
    checks the published numbers (and their transcription) against mass conservation.
    """
    fit = ff.Tinker10(nu2=NU2, z=0.0, mass_definition=md.SOMean(overdensity=overdensity))
    # Tolerance: every parameter in Table 4 is quoted to 3 significant figures, so
    # each is uncertain by up to ~0.2%; the five together allow ~0.5% in the integral.
    # The largest measured deviation is 0.41% (Delta=800).
    assert _mass_fraction(fit, LNNU) == pytest.approx(1.0, abs=5e-3)


def test_smt_normalisation_matches_published_amplitude():
    """The analytic SMT normalisation reproduces A = 0.3222 (Sheth, Mo & Tormen 2001)."""
    fit = ff.SMT(nu2=np.array([1.0]))
    # Tolerance: the published value is quoted to 4 significant figures.
    assert fit._norm() == pytest.approx(0.3222, abs=5e-5)


def test_smt_reduces_to_press_schechter():
    """SMT with a=1, p=0 (spherical collapse, no low-mass excess) is exactly PS."""
    smt = ff.SMT(nu2=NU2, a=1.0, p=0.0)
    ps = ff.PS(nu2=NU2)
    # Tolerance: identical closed forms, so agreement to floating-point rounding.
    np.testing.assert_allclose(smt.fsigma, ps.fsigma, rtol=1e-12)


def test_tinker10_reduces_to_smt():
    r"""Tinker+10 with eta=0, beta=sqrt(a), gamma=a, phi=p is the SMT form.

    Then :math:`f(\nu)=\alpha[1+(a\nu^2)^{-p}]e^{-a\nu^2/2}` is exactly the SMT shape,
    and both fits are normalised to unity, so they must coincide -- this checks the
    Tinker10 normalisation formula against the independently verified SMT one.
    """
    a, p = 0.707, 0.3
    t10 = ff.Tinker10(
        nu2=NU2,
        z=0.5,  # z != 0 so that alpha is computed rather than tabulated
        mass_definition=md.SOMean(overdensity=200),
        beta_200=np.sqrt(a),
        gamma_200=a,
        phi_200=p,
        eta_200=0.0,
        beta_exp=0.0,
        gamma_exp=0.0,
        phi_exp=0.0,
        eta_exp=0.0,
    )
    smt = ff.SMT(nu2=NU2, a=a, p=p)
    # Tolerance: two exact closed forms; floating-point rounding only.
    np.testing.assert_allclose(t10.fsigma, smt.fsigma, rtol=1e-10)


# ---------------------------------------------------------------------------------------
# Bounds and shape for every fit
# ---------------------------------------------------------------------------------------
SIGMA = np.logspace(np.log10(0.15), np.log10(8.0), 400)
SIGMA_NU2 = (DELTA_C / SIGMA) ** 2


@pytest.mark.parametrize("cls", ALL_FITS, ids=lambda c: c.__name__)
def test_fsigma_positive_and_finite(cls):
    """A multiplicity function is a number density and must be finite and positive."""
    f = _make_fit(cls, SIGMA_NU2).fsigma
    assert np.all(np.isfinite(f))
    assert np.all(f > 0)


@pytest.mark.parametrize("cls", ALL_FITS, ids=lambda c: c.__name__)
def test_fsigma_has_exponential_high_mass_cutoff(cls):
    r"""Rare, high peaks are exponentially (Gaussian-ly) suppressed.

    In every model the high-:math:`\nu` tail is :math:`\propto e^{-c\nu^2/2}` (with
    c ~ 0.7-1.3), so at sigma = 0.2 (nu ~ 8.4) the log-slope
    :math:`d\ln f/d\ln\nu \approx -c\nu^2` is of order -50; a power-law tail would
    give a slope of order unity. f(sigma=0.25) is also many orders of magnitude below
    the peak.
    """
    f = _make_fit(cls, SIGMA_NU2).fsigma
    lnnu = np.log(np.sqrt(SIGMA_NU2))
    slope = np.gradient(np.log(f), lnnu)
    i02 = np.argmin(np.abs(SIGMA - 0.2))
    i025 = np.argmin(np.abs(SIGMA - 0.25))
    # Bounds: the softest cutoff of any fit has slope ~ -35 at sigma=0.2 (Jenkins);
    # -20 is far from any power law. Measured f(0.25)/f_max <= 1e-6 for all fits.
    assert slope[i02] < -20
    assert f[i025] / f.max() < 1e-5


@pytest.mark.parametrize(
    "cls", [c for c in ALL_FITS if c not in _MASS_DEPENDENT_FITS], ids=lambda c: c.__name__
)
def test_fsigma_is_unimodal(cls):
    """f(sigma) rises from low to intermediate nu then falls: it has a single peak.

    This is the shape of every published fit over 0.15 < sigma < 8: a low-mass power
    law and a high-mass exponential cutoff.
    """
    f = _make_fit(cls, SIGMA_NU2).fsigma[::-1]  # increasing nu
    i = np.argmax(f)
    # Tolerance: exact monotonicity, to floating-point rounding.
    assert np.all(np.diff(f[: i + 1]) >= -1e-15)
    assert np.all(np.diff(f[i:]) <= 1e-15)


@pytest.mark.parametrize("cls", [c for c in ALL_FITS if c.normalized], ids=lambda c: c.__name__)
def test_normalised_fits_mass_fraction_below_unity_over_finite_range(cls):
    """Haloes in any finite range of sigma can hold at most all the mass."""
    lnnu = np.linspace(np.log(DELTA_C / 8), np.log(DELTA_C / 0.15), 2001)
    f = _make_fit(cls, np.exp(2 * lnnu)).fsigma
    assert simpson(f, x=lnnu) < 1.0


@pytest.mark.parametrize(
    "cls",
    [c for c in ALL_FITS if not c.normalized and c not in _MASS_DEPENDENT_FITS],
    ids=lambda c: c.__name__,
)
def test_unnormalised_fits_mass_fraction_below_unity_in_calibrated_range(cls):
    """Within the sigma range of the calibrating simulations, f(sigma) can't exceed 1.

    Fits that are not normalised may not converge as sigma -> infinity, but the mass
    fraction in haloes with -0.55 < ln(1/sigma) < 1.05 (the range that is common to the
    calibration ranges of these fits) is a fraction of the total, so it must be < 1.
    """
    lnnu = np.linspace(np.log(DELTA_C) - 0.55, np.log(DELTA_C) + 1.05, 2001)
    f = _make_fit(cls, np.exp(2 * lnnu)).fsigma
    assert simpson(f, x=lnnu) < 1.0


def test_watson_overdensity_correction_is_unity_at_reference():
    r"""Watson+13's :math:`\Gamma(\Delta,\sigma,z)` (their Eq. 18) is 1 at Delta=178.

    The correction rescales the Delta=178 (mean) fit to other overdensities, so it
    must leave the reference overdensity unchanged at every sigma and redshift.
    """
    nu2 = SIGMA_NU2
    for z in (0.0, 1.0, 3.0):
        fit = ff.Watson(nu2=nu2, z=z, mass_definition=md.SOMean(overdensity=178))
        # Tolerance: floating-point rounding (C = exp(0), (1)^d and exp(0) exactly).
        np.testing.assert_allclose(fit.gamma(), 1.0, rtol=1e-12)


@pytest.mark.parametrize("cls", [ff.Tinker08, ff.Tinker10, ff.Watson])
def test_cumulative_counts_drop_with_overdensity(cls):
    """Raising the SO overdensity shrinks every halo's mass, so n(>M) must drop.

    A halo's mass within a higher-density contour is smaller, so the number of haloes
    above any fixed mass is lower for a higher overdensity. This is a rigorous
    inequality on the cumulative mass function.
    """
    mf = MassFunction(
        hmf_model=cls,
        transfer_model="EH",
        Mmin=11,
        Mmax=15,
        dlog10m=0.05,
        z=0.5,
        mdef_model="SOMean",
    )
    ngtm = []
    for overdensity in (200, 400, 800, 1600):
        mf.update(mdef_params={"overdensity": overdensity})
        ngtm.append(mf.ngtm.copy())
    assert np.all(np.diff(np.array(ngtm), axis=0) < 0)


# ---------------------------------------------------------------------------------------
# Redshift trends
# ---------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def coarse_mf():
    return MassFunction(
        transfer_model="EH",
        Mmin=11,
        Mmax=15,
        dlog10m=0.05,
        lnk_min=-12,
        lnk_max=6,
        dlnk=0.05,
    )


@pytest.mark.parametrize(
    "cls",
    [c for c in ALL_FITS if c is not ff.Yung24],
    ids=lambda c: c.__name__,
)
def test_massive_haloes_rarer_at_higher_redshift(coarse_mf, cls):
    """Structure grows hierarchically: there are fewer massive haloes at higher z.

    Checked at M = 1e13 and ~1e15 Msun/h, above the nonlinear mass at every z used.
    Watson+13's SO fit uses a separate z=0 parameter set that is discontinuous with
    its z>0 parameterisation (by construction of the paper), so for Watson the trend
    is checked only among z > 0.
    """
    redshifts = [0.5, 1.0, 2.0] if cls is ff.Watson else [0.0, 0.5, 1.0, 2.0]
    dndm = []
    for z in redshifts:
        coarse_mf.update(hmf_model=cls, z=z)
        dndm.append(coarse_mf.dndm[[40, -1]])
    assert np.all(np.diff(np.array(dndm), axis=0) < 0)


def test_yung24_massive_haloes_rarer_at_higher_redshift():
    """As above, for Yung+24 within its calibrated redshift range (6 <= z <= 19)."""
    mf = MassFunction(
        hmf_model="Yung24",
        mdef_model="SOVirial",
        transfer_model="EH",
        Mmin=8,
        Mmax=12,
        dlog10m=0.1,
        z=6.0,
    )
    dndm = []
    for z in (6.0, 8.0, 10.0, 14.0):
        mf.update(z=z)
        dndm.append(mf.dndm[[10, -1]])
    assert np.all(np.diff(np.array(dndm), axis=0) < 0)


# ---------------------------------------------------------------------------------------
# Bocquet+16 mass-definition conversion
# ---------------------------------------------------------------------------------------
_BOCQUET_COSMO = FlatLambdaCDM(H0=70.4, Om0=0.272, Ob0=0.0456, Tcmb0=2.7255)


def _nfw_mass_ratio(m_new, z, overdensity_crit, cosmo):
    r"""M_{Delta c} / M_{200m} for an NFW halo with the Duffy+08 c(M_{200m}) relation.

    Solved directly from the NFW enclosed mass,
    :math:`M(<r) = 4\pi\rho_s r_s^3 [\ln(1+x) - x/(1+x)]`, independently of hmf.
    Masses in Msun/h; densities in h^2 Msun/Mpc^3.
    """
    rho_c = (cosmo.critical_density(z) / cosmo.h**2).to("Msun/Mpc3").value
    rho_200m = 200 * cosmo.Om(z) * rho_c

    def mu(x):
        return np.log(1 + x) - x / (1 + x)

    def m_delta_from_m200m(m200m):
        # Duffy+08, full sample, M200m: A=10.14, B=-0.081, C=-1.01, Mpivot=2e12 Msun/h.
        c = 10.14 * (m200m / 2e12) ** -0.081 * (1 + z) ** -1.01
        rs = (3 * m200m / (4 * np.pi * rho_200m)) ** (1 / 3) / c
        rho_s = m200m / (4 * np.pi * rs**3 * mu(c))
        x = brentq(lambda x: 3 * rho_s * mu(x) / x**3 - overdensity_crit * rho_c, 1e-3, 1e3)
        return 4 * np.pi * rho_s * rs**3 * mu(x)

    m200m = brentq(lambda m: m_delta_from_m200m(m) - m_new, m_new, 10 * m_new)
    return m_new / m200m


_BOCQUET_CRIT_FITS = [
    (ff.Bocquet200cDMOnly, 200),
    (ff.Bocquet200cHydro, 200),
    (ff.Bocquet500cDMOnly, 500),
    (ff.Bocquet500cHydro, 500),
]


@pytest.mark.parametrize(
    ("cls", "overdensity"), _BOCQUET_CRIT_FITS, ids=lambda c: getattr(c, "__name__", str(c))
)
@pytest.mark.parametrize("z", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("m", [1e13, 1e14, 1e15])
def test_bocquet_mass_conversion_matches_nfw(cls, overdensity, z, m):
    """Bocquet+16's mass-ratio fits agree with a direct NFW solve.

    Their Eqs. 6 and A2 fit M_{500c}/M_{200m} and M_{200c}/M_{200m} for NFW haloes
    with the Duffy+08 concentrations.

    The paper's Eqs. 6 and A2 depend on mass, Omega_m and z but not on the baryonic
    physics, so the DM-only and Hydro fits share them; all four are checked.
    """
    cosmo = _BOCQUET_COSMO
    fit = cls(nu2=np.array([1.0]), m=np.array([m]), z=z, cosmo=cosmo)
    expected = _nfw_mass_ratio(m, z, overdensity, cosmo)
    # Tolerance: the paper quotes "few percent" accuracy for its fits; the largest
    # measured deviation is 5.6% (500c, z=1, 1e14). 7% also absorbs the M vs M/h
    # ambiguity in the ln(M) term of the fit (~1%).
    assert fit.convert_mass()[0] == pytest.approx(expected, rel=0.07)


@pytest.mark.parametrize("variant", ["DMOnly", "Hydro"])
@pytest.mark.parametrize("z", [0.0, 0.5, 1.0, 2.0])
def test_bocquet_mass_conversion_ordering(z, variant):
    """Mass within a higher-density contour is smaller: M500c < M200c < M200m.

    Over the calibrated range 1e13 < M < 1e16, so the fitted ratios must satisfy
    0 < M500c/M200m < M200c/M200m < 1.
    """
    m = np.logspace(13, 16, 10)
    nu2 = np.ones_like(m)
    cls200 = getattr(ff, f"Bocquet200c{variant}")
    cls500 = getattr(ff, f"Bocquet500c{variant}")
    r200c = cls200(nu2=nu2, m=m, z=z, cosmo=_BOCQUET_COSMO).convert_mass()
    r500c = cls500(nu2=nu2, m=m, z=z, cosmo=_BOCQUET_COSMO).convert_mass()
    assert np.all(r500c > 0)
    assert np.all(r500c < r200c)
    assert np.all(r200c < 1)
