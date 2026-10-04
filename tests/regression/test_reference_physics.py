"""Physical checks of the v3.7.2 regression reference itself (issue #394).

The reference is only as good as v3.7.2, so it is checked against physics before
v4 is compared with it: closed-form solutions (power-law Press-Schechter,
Einstein-de Sitter growth, the radiation-free LCDM growth integral), exact
normalisations (fits built to put all mass in haloes), and physical bounds (sigma
falls with mass and redshift, T(k) <= 1, ...). Each test computes its expected values
independently of hmf, from the formulae in its docstring.
"""

import mpmath
import numpy as np
import pytest
from scipy.integrate import quad, simpson


def _analytic(reference, name):
    return np.asarray(reference.arrays[f"analytic/{name}"])


# ---------------------------------------------------------------------------------
# Power-law Press-Schechter
# ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def powerlaw(reference):
    """The power-law PS case: P(k) = A k^n, TopHat filter, z = 0."""
    meta = reference.metadata["analytic"]["powerlaw_ps"]
    n = meta["n"]
    m = _analytic(reference, "powerlaw_ps/m")
    rho = float(_analytic(reference, "powerlaw_ps/mean_density0"))
    sigma_8 = reference.settings["sigma_8"]
    delta_c = reference.settings["delta_c"]
    # For P = A k^n and a TopHat, sigma^2 is proportional to R^-(n+3), and sigma_8
    # fixes the amplitude: sigma = sigma_8 (R / 8 Mpc/h)^(-(n+3)/2), R = (3M/4 pi rho)^1/3.
    r = (3 * m / (4 * np.pi * rho)) ** (1 / 3)
    alpha = (n + 3) / 6  # sigma ∝ M^-alpha
    sigma = sigma_8 * (r / 8) ** (-(n + 3) / 2)
    return {
        "n": n,
        "m": m,
        "rho": rho,
        "alpha": alpha,
        "sigma": sigma,
        "nu": delta_c / sigma,
        "delta_c": delta_c,
    }


def test_powerlaw_sigma(reference, powerlaw):
    """sigma(M) of a power-law spectrum is the power law fixed by sigma_8.

    Tolerance 1e-5, the sigma tolerance: for n = -2 the k range truncates the
    integral by under 1e-6 at both ends (the integrand goes as (kR)^(n+3) at low k and
    falls as (kR)^(n-1) at high k, with kR > 2000 at the smallest mass), so anything
    larger is a v3 error.
    """
    np.testing.assert_allclose(
        _analytic(reference, "powerlaw_ps/sigma"), powerlaw["sigma"], rtol=1e-5
    )


def test_powerlaw_dlnsdlnm(reference, powerlaw):
    """Dln sigma/dln M = -(n+3)/6 exactly, for every mass."""
    np.testing.assert_allclose(
        _analytic(reference, "powerlaw_ps/dlnsdlnm"), -powerlaw["alpha"], rtol=1e-5
    )


def test_powerlaw_ps_dndm(reference, powerlaw):
    r"""Press-Schechter dn/dM against the closed form.

    .. math:: \frac{dn}{dM} = \sqrt{\frac{2}{\pi}} \frac{\bar\rho}{M^2} \alpha\,\nu
              e^{-\nu^2/2}, \quad \nu = \delta_c/\sigma, \ \alpha = (n+3)/6.

    The tolerance is the dn/dm tolerance with the sigma error propagated, 1e-4 +
    nu^2 1e-5 (|dln f/dln sigma| = nu^2 - 1 for PS), at the points not underflowed.
    """
    nu, alpha, m, rho = powerlaw["nu"], powerlaw["alpha"], powerlaw["m"], powerlaw["rho"]
    expected = np.sqrt(2 / np.pi) * rho / m**2 * alpha * nu * np.exp(-(nu**2) / 2)
    actual = _analytic(reference, "powerlaw_ps/dndm")
    ok = expected > 1e-290
    assert ok.sum() > 100
    rel = np.abs(actual[ok] / expected[ok] - 1)
    assert np.all(rel <= 1e-4 + nu[ok] ** 2 * 1e-5), rel.max()


def test_powerlaw_ps_ngtm(reference, powerlaw):
    r"""Press-Schechter n(>M) against the closed form.

    With :math:`M(\nu) \propto \nu^{p}`, :math:`p = 1/\alpha`, integrating dn/dM above
    M gives

    .. math:: n(>M) = \sqrt{\frac{2}{\pi}} \frac{\bar\rho\,\nu^{p}}{M}
              \int_\nu^\infty t^{-p} e^{-t^2/2}\,dt
              = \sqrt{\frac{2}{\pi}} \frac{\bar\rho\,\nu^{p}}{M}
              2^{-(p+1)/2}\, \Gamma\!\left(\frac{1-p}{2}, \frac{\nu^2}{2}\right),

    with Γ the upper incomplete gamma function. Same tolerance as dn/dM.
    """
    nu, m, rho, p = powerlaw["nu"], powerlaw["m"], powerlaw["rho"], 1 / powerlaw["alpha"]
    gamma = np.array([float(mpmath.gammainc((1 - p) / 2, a=x)) for x in nu**2 / 2])
    expected = np.sqrt(2 / np.pi) * rho * nu**p / m * 2 ** (-(p + 1) / 2) * gamma
    actual = _analytic(reference, "powerlaw_ps/ngtm")
    ok = expected > 1e-290
    assert ok.sum() > 100
    rel = np.abs(actual[ok] / expected[ok] - 1)
    assert np.all(rel <= 1e-4 + nu[ok] ** 2 * 1e-5), rel.max()


# ---------------------------------------------------------------------------------
# Growth
# ---------------------------------------------------------------------------------


def test_eds_growth_is_scale_factor(reference):
    """In Einstein-de Sitter, D(z) = a = 1/(1+z) exactly (to the growth tolerance)."""
    z = _analytic(reference, "z_growth")
    np.testing.assert_allclose(_analytic(reference, "eds/growth"), 1 / (1 + z), rtol=1e-6)


def test_radiation_free_lcdm_growth_matches_integral(reference):
    r"""Flat LCDM without radiation: the ODE growth against Heath's integral.

    .. math:: D(a) \propto E(a) \int_0^a \frac{da'}{(a' E(a'))^3}, \quad
              E(a) = \sqrt{\Omega_m a^{-3} + 1 - \Omega_m},

    computed here by quadrature and normalised to D(a=1) = 1. The reference's
    growth model must reproduce it to the growth tolerance, 2e-6.
    """
    om = float(_analytic(reference, "lcdm_no_radiation/Om0"))
    z = _analytic(reference, "z_growth")

    def e(a):
        return np.sqrt(om / a**3 + 1 - om)

    def d(a):
        return e(a) * quad(lambda x: 1 / (x * e(x)) ** 3, 0, a, epsabs=0, epsrel=1e-12)[0]

    expected = np.array([d(1 / (1 + zz)) for zz in z]) / d(1.0)
    np.testing.assert_allclose(
        _analytic(reference, "lcdm_no_radiation/growth"), expected, rtol=2e-6
    )


def test_growth_is_normalised_and_decreasing(reference):
    """D(0) = 1, and structure only grows: D > 0 falls strictly with z."""
    for case in reference.cases("growth"):
        d = reference.values(case)
        assert d[0] == pytest.approx(1.0, abs=1e-12), case.key
        assert np.all(np.diff(d) < 0), case.key
        assert np.all(d > 0), case.key


# ---------------------------------------------------------------------------------
# Fitting functions
# ---------------------------------------------------------------------------------

#: Fits normalised to unity only approximately, with the reason.
_NORMALISATION_TOLERANCE = {
    # The published z = 0 alpha of Tinker+10 (their Table 4) normalises f to 0.5%
    # (tests/test_physical_fits.py); at z > 0 hmf computes alpha exactly.
    ("Tinker10", "z=0"): 5e-3,
}


def test_normalised_fits_integrate_to_one(reference):
    r"""Fits that are normalised by construction put all the mass in haloes.

    :math:`\int f(\sigma)\, d\ln\sigma^{-1} = 1`, integrated over the stored grid,
    ln(1/sigma) from -40 to 4 (the f of every such fit falls at least as
    :math:`\nu^{0.4}` at low nu, so the part below the grid is under 1e-7, and
    exponentially at high nu). Tolerance 1e-6 (integration only), except where
    noted in ``_NORMALISATION_TOLERANCE``.
    """
    x = _analytic(reference, "fsigma_wide/ln_inv_sigma")
    keys = [k for k in reference.arrays if k.startswith("analytic/fsigma_wide/") and "=" in k]
    assert {k.split("/")[2] for k in keys} >= {"PS", "SMT", "Tinker10", "Peacock"}
    for k in keys:
        fit, z = k.split("/")[2:]
        f = np.asarray(reference.arrays[k])
        assert np.all(np.isfinite(f)), k
        total = simpson(f, x=x)
        assert total == pytest.approx(1.0, abs=_NORMALISATION_TOLERANCE.get((fit, z), 1e-6)), k


# ---------------------------------------------------------------------------------
# Bounds and trends
# ---------------------------------------------------------------------------------


def test_sigma_falls_with_mass_and_redshift(reference):
    """sigma(M, z) is strictly decreasing in M (dln sigma/dln M < 0) and in z."""
    for case in reference.cases("sigma"):
        s = reference.values(case)
        assert np.all(np.diff(s, axis=1) < 0), case.key
        assert np.all(np.diff(s, axis=0) < 0), case.key
    for case in reference.cases("dlnsdlnm"):
        assert np.all(reference.values(case) < 0), case.key


def test_transfer_is_bounded(reference):
    """0 < T(k) <= 1, with T -> 1 at low k.

    Perturbations are suppressed only once inside the horizon, so
    T(k) = 1 - O((k/k_eq)^2) for k << k_eq ~ 0.01 h/Mpc.

    The CAMB cases start at CAMB's lowest k (~1e-4 h/Mpc), where T is 1 to ~1e-3;
    EH starts at 1.5e-8 h/Mpc. 1e-6 above 1 allows for interpolating CAMB's table.
    """
    for case in reference.cases("transfer"):
        t = reference.values(case)
        t = t[np.isfinite(t)]
        assert np.all(t > 0), case.key
        assert np.all(t <= 1 + 1e-6), case.key
        assert t[0] == pytest.approx(1.0, abs=1e-3 if case.transfer == "CAMB" else 1e-6), case.key


def test_mass_function_bounds(reference):
    """dn/dM >= 0 and n(>M) is non-increasing in M wherever it is computed."""
    for case in reference.cases("dndm"):
        v = reference.values(case)
        assert np.all(v[np.isfinite(v)] >= 0), case.key
    for case in reference.cases("ngtm"):
        v = reference.values(case)
        ok = np.isfinite(v[:, :-1]) & np.isfinite(v[:, 1:])
        assert np.all(np.diff(v, axis=1)[ok] <= 0), case.key
