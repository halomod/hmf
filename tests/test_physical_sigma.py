r"""Physical tests of sigma(M), n_eff, normalisation, dn/dm and cumulative integrals.

Covers sigma(M), nu, n_eff, the sigma_8 normalisation, dn/dm, ngtm and rho_gtm. The
main reference is the self-similar case: for a power-law spectrum
:math:`P(k)\propto k^n` and a real-space top-hat filter,
:math:`\sigma(R) = \sigma_8 (R/8)^{-(n+3)/2}` exactly, so :math:`n_{\rm eff}=n`, and
the Press-Schechter mass function and its integrals have closed forms.
"""

import numpy as np
import pytest
from scipy.special import erfc, expn

from hmf import MassFunction
from hmf.density_field import filters
from hmf.mass_function.integrate_hmf import hmf_integral_gtm

DELTA_C = 1.68647
SIGMA_8 = 0.8
COSMO_PARAMS = {"Om0": 0.3, "H0": 70.0, "Ob0": 0.05}


def _power_law_mf(n, **kwargs):
    """A MassFunction whose linear power spectrum is exactly A k^n (T(k) = 1)."""
    k = np.exp(np.arange(-25, 25, 0.1))
    params = {
        "transfer_model": "FromArray",
        "transfer_params": {"k": k, "T": np.ones_like(k)},
        "n": n,
        "hmf_model": "PS",
        "Mmin": 8,
        "Mmax": 16,
        "dlog10m": 0.02,
        "lnk_min": -20,
        "lnk_max": 20,
        "dlnk": 0.02,
        "sigma_8": SIGMA_8,
        "cosmo_params": COSMO_PARAMS,
    }
    params.update(kwargs)
    return MassFunction(**params)


def _sigma_self_similar(mf, n):
    """sigma(M) for P ~ k^n with a top-hat filter, normalised at R = 8 Mpc/h."""
    radius = (3 * mf.m / (4 * np.pi * mf.mean_density0)) ** (1 / 3)
    return SIGMA_8 * (radius / 8.0) ** (-(n + 3) / 2)


@pytest.fixture(scope="module")
def mf_n2():
    return _power_law_mf(-2.0)


@pytest.fixture(scope="module")
def mf_n1():
    return _power_law_mf(-1.0)


# ---------------------------------------------------------------------------------------
# sigma, n_eff, normalisation
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [-2.5, -2.0, -1.0])
def test_sigma_self_similar_power_law(n):
    """sigma(M) = sigma_8 (R/8)^{-(n+3)/2} for a power-law spectrum and top-hat filter."""
    mf = _power_law_mf(n)
    # Tolerance: Simpson integration on dlnk = 0.02; measured max error 5e-5.
    np.testing.assert_allclose(mf.sigma, _sigma_self_similar(mf, n), rtol=1e-3)


def test_neff_equals_spectral_index_for_power_law(mf_n2):
    """The effective spectral index of a pure power law is the index itself."""
    # Tolerance: n_eff comes from d ln sigma/d ln m, a top-hat-weighted integral that
    # oscillates at high k; measured max deviation 8e-5 for n = -2.
    np.testing.assert_allclose(mf_n2.n_eff, -2.0, atol=1e-3)


def test_neff_equals_spectral_index_for_power_law_n_minus1(mf_n1):
    """As above for n = -1, where the top-hat derivative integrand converges slowly.

    The oscillatory tail of W dW/dlnkR P k^3 ~ (kR)^{-1} cos(kR) is sampled at
    dlnk = 0.02, so n_eff carries small wiggles (documented in MassFunction.n_eff).
    """
    # Tolerance: measured wiggles up to 0.0084.
    np.testing.assert_allclose(mf_n1.n_eff, -1.0, atol=2e-2)


def test_sigma_8_normalisation_for_power_law(mf_n2):
    """The normalised power gives sigma(R = 8 Mpc/h) = sigma_8."""
    # Tolerance: integration only; measured 4e-7.
    assert mf_n2.normalised_filter.sigma(8.0)[0] == pytest.approx(SIGMA_8, rel=1e-4)


def test_sigma_decreases_with_mass(mf_n2):
    """Variance on larger smoothing scales is smaller (for n > -3)."""
    assert np.all(np.diff(mf_n2.sigma) < 0)


def test_nonlinear_mass_self_similar(mf_n2):
    r"""M_* defined by sigma(M_*) = delta_c has a closed form for a power law.

    :math:`M_* = M_8 (\sigma_8/\delta_c)^{6/(n+3)}` with
    :math:`M_8 = \frac{4\pi}{3} 8^3 \bar\rho`.
    """
    m8 = 4 * np.pi / 3 * 8.0**3 * mf_n2.mean_density0
    m_star = m8 * (SIGMA_8 / DELTA_C) ** (6 / (-2.0 + 3))
    # Tolerance: spline inversion of nu(m); measured 1e-6.
    assert mf_n2.mass_nonlinear == pytest.approx(m_star, rel=1e-3)


def test_nonlinear_mass_grows_with_time():
    """Hierarchical growth: the collapse mass scale M_* increases as z decreases."""
    mf = MassFunction(transfer_model="EH", Mmin=8, Mmax=15, dlog10m=0.05)
    m_star = []
    for z in (2.0, 1.0, 0.5, 0.0):
        mf.update(z=z)
        m_star.append(mf.mass_nonlinear)
    assert np.all(np.diff(m_star) > 0)


# ---------------------------------------------------------------------------------------
# Filters with power-law spectra
# ---------------------------------------------------------------------------------------
def test_tophat_white_noise_variance_is_poisson():
    r"""For white noise P(k) = P_0, the top-hat variance is :math:`P_0/V`.

    :math:`\sigma^2 = \frac{P_0}{2\pi^2}\int_0^\infty k^2 W^2(kR)\,dk
    = \frac{P_0}{2\pi^2 R^3}\cdot\frac{3\pi}{2} = \frac{P_0}{4\pi R^3/3}`, using
    :math:`\int_0^\infty x^2 W^2(x)dx = 9\int_0^\infty j_1^2(x)dx = 3\pi/2`.
    """
    k = np.exp(np.arange(np.log(1e-6), np.log(1e6), 0.002))
    p0 = 7.0
    radius = np.array([0.5, 1.0, 4.0])
    sigma2 = filters.TopHat(k, np.full_like(k, p0)).sigma(radius) ** 2
    # Tolerance: the integrand's tail ~ cos^2(kR)/(kR)^2 is truncated at k = 1e6 h/Mpc
    # (fractional loss ~ 1/(3 pi k_max R) < 1e-6) and sampled on a grid that resolves
    # the oscillations up to kR ~ 1e3; measured error < 2e-4.
    np.testing.assert_allclose(sigma2, p0 / (4 * np.pi * radius**3 / 3), rtol=1e-3)


@pytest.mark.parametrize("n", [-2.5, -2.0, -1.0])
def test_sharpk_power_law_variance(n):
    r"""For P = k^n and a sharp-k filter, :math:`\sigma^2 = R^{-(n+3)}/[2\pi^2(n+3)]`."""
    k = np.exp(np.arange(np.log(1e-12), np.log(1e3), 0.01))
    radius = np.array([0.5, 2.0, 8.0])
    sigma2 = filters.SharpK(k, k**n).sigma(radius) ** 2
    # Tolerance: the integral starts at k = 1e-12 instead of 0 (missing fraction
    # (1e-12 R)^{n+3} < 1e-6) and uses Simpson's rule; measured < 1e-5.
    np.testing.assert_allclose(sigma2, radius ** (-(n + 3)) / (2 * np.pi**2 * (n + 3)), rtol=1e-4)


# ---------------------------------------------------------------------------------------
# dn/dm, ngtm, rho_gtm
# ---------------------------------------------------------------------------------------
def _ps_dndm_self_similar(mf, n):
    r"""Closed-form PS mass function for P ~ k^n (top-hat filter).

    :math:`\frac{dn}{dM} = \sqrt{\frac{2}{\pi}}\frac{\bar\rho}{M^2}\nu e^{-\nu^2/2}
    \frac{n+3}{6}`, with :math:`\nu = \delta_c/\sigma(M)`.
    """
    nu = DELTA_C / _sigma_self_similar(mf, n)
    return np.sqrt(2 / np.pi) * nu * np.exp(-(nu**2) / 2) * (n + 3) / 6 * mf.mean_density0 / mf.m**2


def test_ps_dndm_self_similar(mf_n2):
    """MassFunction's PS dn/dm matches the closed form for a power-law spectrum."""
    expected = _ps_dndm_self_similar(mf_n2, -2.0)
    # Tolerance: errors in sigma (5e-5) are amplified by nu^2 ~ 30 at the high-mass
    # end, and |dln sigma/dln m| carries 1e-4 errors; measured max 8e-5.
    np.testing.assert_allclose(mf_n2.dndm, expected, rtol=1e-3)


@pytest.mark.parametrize("fixture", ["mf_n2", "mf_n1"])
def test_ps_collapsed_fraction_is_erfc(fixture, request):
    r"""PS mass fraction in haloes above M is :math:`{\rm erfc}(\nu/\sqrt2)`.

    This is the defining result of Press & Schechter (1974) (with their factor 2),
    valid for any spectrum; here sigma is the analytic self-similar one, so it tests
    dn/dm assembly and the integration in rho_gtm together.
    """
    mf = request.getfixturevalue(fixture)
    n = -2.0 if fixture == "mf_n2" else -1.0
    nu = DELTA_C / _sigma_self_similar(mf, n)
    mask = nu < 3.0
    # Tolerance: trapezoidal cumulative integration with dlog10m = 0.02 gives relative
    # errors growing into the exponential tail; for nu < 3 the measured max is 3e-3.
    np.testing.assert_allclose(
        mf.rho_gtm[mask] / mf.mean_density0, erfc(nu[mask] / np.sqrt(2)), rtol=5e-3
    )


def test_ps_number_density_above_mass_self_similar(mf_n1):
    r"""For n = -1 the PS cumulative number density has a closed form.

    With :math:`\nu=(M/M_*)^{1/3}`,
    :math:`n(>M) = \frac{\bar\rho}{M_*}\sqrt{\frac2\pi}\int_\nu^\infty x^{-3}e^{-x^2/2}dx
    = \frac{\bar\rho}{M_*}\sqrt{\frac2\pi}\frac{E_2(y)}{4y}`, :math:`y=\nu^2/2`.
    """
    mf = mf_n1
    nu = DELTA_C / _sigma_self_similar(mf, -1.0)
    m_star = mf.m * nu ** (-3.0)
    y = nu**2 / 2
    expected = mf.mean_density0 / m_star * np.sqrt(2 / np.pi) * expn(2, y) / (4 * y)
    mask = nu < 3.0
    # Tolerance: as for rho_gtm; measured max deviation 1.6e-3 for nu < 3.
    np.testing.assert_allclose(mf.ngtm[mask], expected[mask], rtol=5e-3)


@pytest.mark.parametrize("model", ["PS", "SMT", "Tinker08", "Warren"])
def test_cumulative_densities_decrease_with_mass(model):
    """n(>M) and rho(>M) are integrals of a positive density, so they fall with M."""
    mf = MassFunction(hmf_model=model, transfer_model="EH", Mmin=8, Mmax=15, dlog10m=0.05)
    assert np.all(np.diff(mf.ngtm) < 0)
    assert np.all(np.diff(mf.rho_gtm) < 0)
    # Mass in haloes above M can't exceed the total matter density for a fit that
    # places all mass in haloes, and can't be negative.
    assert np.all(mf.rho_gtm > 0)
    if mf.hmf.normalized:
        assert np.all(mf.rho_gtm < mf.mean_density0)


def test_ps_all_mass_in_haloes_at_low_mass():
    r"""For a normalised fit, rho(>M) -> rho_mean as M -> 0.

    With n = -2.5 sigma grows as M^{-1/12}, so at M = 1e-2 Msun/h nu ~ 0.09 and
    erfc(nu/sqrt2) = 0.93: almost all mass is in haloes.
    """
    mf = _power_law_mf(-2.5, Mmin=-2, Mmax=16, dlog10m=0.02)
    nu0 = DELTA_C / mf.sigma[0]
    # Tolerance: the collapsed fraction at the lowest mass is erfc(nu0/sqrt2) analytically
    # (asserted to 0.5% in test_ps_collapsed_fraction_is_erfc); here check it is
    # close to unity, the physical limit.
    assert mf.rho_gtm[0] / mf.mean_density0 == pytest.approx(erfc(nu0 / np.sqrt(2)), rel=5e-3)
    assert mf.rho_gtm[0] / mf.mean_density0 > 0.9


def test_integral_of_power_law_dndm_is_analytic():
    r"""hmf_integral_gtm integrates dn/dm = A M^{-a} to :math:`A M^{1-a}/(a-1)`.

    The integral extends to 1e18 (with extrapolation), so compare to the definite
    integral up to 1e18.
    """
    m = np.logspace(8, 16, 801)
    a = 2.5
    dndm = m**-a
    expected = (m ** (1 - a) - 1e18 ** (1 - a)) / (a - 1)
    rho_expected = (m ** (2 - a) - 1e18 ** (2 - a)) / (a - 2)
    # Tolerance: trapezoidal rule with dlnm = 0.023 on an exponential integrand in lnm
    # has relative error ~ (a-1)^2 dlnm^2 / 12 ~ 1e-4.
    np.testing.assert_allclose(hmf_integral_gtm(m, dndm), expected, rtol=1e-3)
    np.testing.assert_allclose(
        hmf_integral_gtm(m, dndm, mass_density=True), rho_expected, rtol=1e-3
    )
