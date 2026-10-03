"""Tests that fitting functions use the parameters of their *published* papers.

Reference values here are typed in from the papers (the version is cited on each), not
derived from the code under test.
"""

import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

from hmf.halos import mass_definitions as md
from hmf.mass_function import fitting_functions as ff

DELTA_C = 1.68647


def _nu2(sigma):
    return (DELTA_C / sigma) ** 2


# ---------------------------------------------------------------------------------------
# Watson et al. 2013 (MNRAS 433, 1230; arXiv:1212.0095v4)
# ---------------------------------------------------------------------------------------

# Watson+13's simulations all use Omega_m = 0.27 (WMAP).
WATSON_COSMO = FlatLambdaCDM(H0=70.0, Om0=0.27, Ob0=0.044)


def _watson(z, sigma=None):
    sigma = np.array([1.0]) if sigma is None else sigma
    return ff.Watson(
        nu2=_nu2(sigma),
        z=z,
        cosmo=WATSON_COSMO,
        # Delta = 178 (mean) makes the overdensity correction Gamma (eq. 18) exactly 1,
        # so fsigma is the bare eq. 12.
        mass_definition=md.SOMean(overdensity=178),
    )


def test_watson_redshift_fit_reproduces_table2_at_z0():
    """Eqs 14-16 of Watson+13 v4, taken to z -> 0+, give its Table 2 CPMSO+AHF column.

    Table 2 (v4, p.12), column "CPMSO+AHF, z=0": A = 0.316, alpha = 2.234,
    beta = 1.478. The table quotes three decimals, but it lists the z=0 values of the
    fitted curves (eqs 14-16) rather than evaluating them exactly: eqs 14/15 at z=0
    give 2.228/1.481. So the tolerance is 0.01 in absolute terms (<0.5%). With alpha
    and beta swapped (the arXiv v1 labelling) the error is ~0.75, way outside it.
    """
    A, alpha, beta, gamma = _watson(z=1e-8).get_params()

    np.testing.assert_allclose(A, 0.316, atol=0.01)
    np.testing.assert_allclose(alpha, 2.234, atol=0.01)
    np.testing.assert_allclose(beta, 1.478, atol=0.01)
    # gamma is fixed at its universal value (v4 text below eq. 12 and Table 2).
    assert gamma == 1.318


def test_watson_redshift_fit_continuous_with_z0_fit():
    """At small z, the z-dependent SO fit should agree with the z=0 AHF fit.

    Watson+13 v4 states the redshift-dependent AHF fit (eqs 14-16, gamma = 1.318) is
    "accurate to ~10% for redshifts less than z = 15" (p.10), and the z=0 AHF fit
    (Table 2) is the more precise one at z=0. Both describe the same AHF data over
    -0.55 <= ln(1/sigma) < 1.05, so at z = 0.01 (Omega_m = 0.27, the simulation value)
    they must agree to within that ~10%. With the published coefficients the ratio is
    0.99-1.07 across the range (1.00-1.09 as z -> 0+); with alpha and beta swapped it
    runs from 1.49 down to 0.69.
    """
    lnsigma_inv = np.linspace(-0.55, 1.05, 50)
    sigma = np.exp(-lnsigma_inv)

    ratio = _watson(z=0.01, sigma=sigma).fsigma / _watson(z=0.0, sigma=sigma).fsigma

    assert np.all(ratio > 0.9)
    assert np.all(ratio < 1.1)


# ---------------------------------------------------------------------------------------
# Bocquet et al. 2016 (MNRAS 456, 2361; arXiv:1502.07357v3)
# ---------------------------------------------------------------------------------------

# Bocquet+16 v3 Table 2, in the paper's notation for eq. 3,
#   f(sigma) = A [(sigma/b)^-a + 1] exp(-c/sigma^2),
# columns (A, a, b, c, A_z, a_z, b_z, c_z).
BOCQUET16_TABLE2 = {
    ff.Bocquet200mDMOnly: (0.175, 1.53, 2.55, 1.19, -0.012, -0.040, -0.194, -0.021),
    ff.Bocquet200mHydro: (0.228, 2.15, 1.69, 1.30, 0.285, -0.058, -0.366, -0.045),
    ff.Bocquet200cDMOnly: (0.222, 1.71, 2.24, 1.46, 0.269, 0.321, -0.621, -0.153),
    ff.Bocquet200cHydro: (0.202, 2.21, 2.00, 1.57, 1.147, 0.375, -1.074, -0.196),
    ff.Bocquet500cDMOnly: (0.241, 2.18, 2.35, 2.02, 0.370, 0.251, -0.698, -0.310),
    ff.Bocquet500cHydro: (0.180, 2.29, 2.44, 1.97, 1.088, 0.150, -1.008, -0.322),
}

# hmf writes eq. 3 as A[(e/sigma)^b + 1] exp(-d/sigma^2), so the paper's (a, b, c)
# are hmf's (b, e, d).
PAPER_TO_HMF_KEYS = ("A", "b", "e", "d", "A_z", "b_z", "e_z", "d_z")

# Bocquet+16's simulations use WMAP7 (Section 2.1): Omega_m = 0.272, H0 = 70.4.
BOCQUET_COSMO = FlatLambdaCDM(H0=70.4, Om0=0.272, Ob0=0.0456)


@pytest.mark.parametrize("cls", list(BOCQUET16_TABLE2))
def test_bocquet16_defaults_match_published_table(cls):
    """Defaults are Bocquet+16 v3 (published) Table 2, not the arXiv v1 preprint table."""
    expected = dict(zip(PAPER_TO_HMF_KEYS, BOCQUET16_TABLE2[cls], strict=True))
    actual = {key: cls._defaults[key] for key in PAPER_TO_HMF_KEYS}
    assert actual == expected


def _bocquet16_m500c_over_m200m(om, z, m_sun):
    """Bocquet+16 v3 eqs 6-7, with coefficients typed in from the paper."""
    beta = -1.70e-2 + om * 3.74e-3
    alpha_0 = 0.880 + 0.329 * om
    alpha_1 = 1.00 + 4.31e-2 / om
    alpha_2 = -0.365 + 0.254 / om
    alpha = alpha_0 * (alpha_1 * z + alpha_2) / (z + alpha_2)
    return alpha + beta * np.log(m_sun)


def _bocquet16_m200c_over_m200m(om, z, m_sun):
    """Bocquet+16 v3 eqs A2-A4, with coefficients typed in from the paper."""
    gamma_0 = 3.54e-2 + om**0.09
    gamma_1 = 4.56e-2 + 2.68e-2 / om
    gamma_2 = 0.721 + 3.50e-2 / om
    gamma_3 = 0.628 + 0.164 / om
    delta_0 = -1.67e-2 + 2.18e-2 * om
    delta_1 = 6.52e-3 - 6.86e-3 * om
    gamma = gamma_0 + gamma_1 * np.exp(-(((gamma_2 - z) / gamma_3) ** 2))
    delta = delta_0 + delta_1 * z
    return gamma + delta * np.log(m_sun)


@pytest.mark.parametrize(
    ("cls", "paper_ratio"),
    [
        (ff.Bocquet500cDMOnly, _bocquet16_m500c_over_m200m),
        (ff.Bocquet500cHydro, _bocquet16_m500c_over_m200m),
        (ff.Bocquet200cDMOnly, _bocquet16_m200c_over_m200m),
        (ff.Bocquet200cHydro, _bocquet16_m200c_over_m200m),
    ],
)
@pytest.mark.parametrize("z", [0.0, 0.5, 1.5])
def test_bocquet16_mass_conversion_uses_msun(cls, paper_ratio, z):
    """The SO mass ratio of eqs 6/A2 takes ln(M/Msun), but hmf's m is in Msun/h.

    The paper's fits are valid for 1e13 < M/Msun < 1e16 and its masses carry no h
    (H0 = 70.4 km/s/Mpc), so at fixed m [Msun/h] the ratio must be evaluated at
    M = m/h. Using ln(m) instead shifts it by beta*ln(h), i.e. 0.5-1%.
    """
    h = BOCQUET_COSMO.h
    m = np.logspace(13, 16, 7) * h  # Msun/h, i.e. 1e13 - 1e16 Msun.
    fit = cls(nu2=_nu2(np.ones_like(m)), m=m, z=z, cosmo=BOCQUET_COSMO)

    expected = paper_ratio(BOCQUET_COSMO.Om0, z, m / h)

    # A more compact definition encloses less mass; at 1e16 Msun and z=0 (low
    # concentration) M500c/M200m drops to ~0.38.
    assert np.all(expected < 1)
    assert np.all(expected > 0.3)
    np.testing.assert_allclose(fit.convert_mass(), expected, rtol=1e-12)


@pytest.mark.parametrize("cls", list(BOCQUET16_TABLE2))
@pytest.mark.parametrize("z", [0.0, 1.0, 2.0])
def test_bocquet16_fsigma_positive_with_exponential_tail(cls, z):
    """f(sigma) > 0, and at large 1/sigma it falls off as exp(-c(z)/sigma^2) (eq. 3)."""
    lnsigma_inv = np.linspace(-1.0, 2.0, 200)
    sigma = np.exp(-lnsigma_inv)
    m = np.full_like(sigma, 1e14)  # Only used by the (mild) SO mass conversion.
    fit = cls(nu2=_nu2(sigma), m=m, z=z, cosmo=BOCQUET_COSMO)

    f = fit.fsigma
    assert np.all(f > 0)

    # d ln f / d(sigma^-2) -> -c(z) as sigma -> 0. The power-law term adds
    # a / (2 sigma^-2) to this slope, which is < 2.5% of c(z) at ln(1/sigma) = 2.
    c_z = cls._defaults["d"] * (1 + z) ** cls._defaults["d_z"]
    slope = np.diff(np.log(f[-2:])) / np.diff(sigma[-2:] ** -2.0)
    np.testing.assert_allclose(slope, -c_z, rtol=0.05)


def test_bocquet16_reference_spelling():
    assert "Bocquet, S." in ff.Bocquet200mDMOnly._ref
    assert "Bocuet" not in ff.Bocquet200mDMOnly.__doc__
