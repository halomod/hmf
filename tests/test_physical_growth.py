"""Physical tests of the linear growth factor and growth rate.

References are exact limits (Einstein-de Sitter), the definition of the growth rate,
and published approximations with known accuracy (Carroll, Press & Turner 1992;
Linder 2005). No reference re-implements the method under test.
"""

import numpy as np
import pytest
from astropy import cosmology

from hmf.cosmology import growth_factor as gf

Z = np.array([0.0, 0.5, 1.0, 3.0, 10.0])

# Einstein-de Sitter: flat, matter only, no radiation.
EDS = cosmology.LambdaCDM(H0=70.0, Om0=1.0, Ode0=0.0, Tcmb0=0.0)

# Flat LCDM without radiation (so every model's assumptions hold).
LCDM = cosmology.FlatLambdaCDM(H0=67.7, Om0=0.31, Tcmb0=0.0)


@pytest.mark.parametrize(
    "model",
    [
        "ODEGrowthFactor",
        "IntegralGrowthFactor",
        "Heath77GrowthFactor",
        "GenMFGrowth",
        "Carroll1992",
    ],
)
def test_einstein_de_sitter_growth_is_scale_factor(model):
    """In EdS the growing mode is exactly D(a) = a, i.e. D(z) = 1/(1+z), with f = 1."""
    g = getattr(gf, model)(EDS)
    # Tolerance: ODE solved at rtol=1e-6 and integrated splines; measured <1e-9.
    np.testing.assert_allclose(g.growth_factor(Z) * (1 + Z), 1.0, rtol=1e-6)
    np.testing.assert_allclose(g.growth_rate(Z), 1.0, rtol=1e-6)


@pytest.mark.parametrize("model", ["GrowthFactor", "Eisenstein97GrowthFactor"])
def test_near_einstein_de_sitter_limit(model):
    """Flat models approach D = a continuously as Omega_Lambda -> 0.

    With Omega_Lambda = 1e-6 the departure from D = a is of order
    Omega_Lambda (1+z)^-3 (Eisenstein 1997 Eq. 10 series), i.e. < 1e-6.
    """
    cosmo = cosmology.FlatLambdaCDM(H0=70.0, Om0=1 - 1e-6, Tcmb0=0.0)
    g = getattr(gf, model)(cosmo)
    # Tolerance: 1e-5 covers the O(Omega_Lambda) physical departure plus rounding.
    np.testing.assert_allclose(g.growth_factor(Z) * (1 + Z), 1.0, rtol=1e-5)
    np.testing.assert_allclose(g.growth_rate(Z), 1.0, rtol=1e-5)


@pytest.mark.parametrize(
    "model",
    [
        "GrowthFactor",
        "ODEGrowthFactor",
        "IntegralGrowthFactor",
        "Eisenstein97GrowthFactor",
        "GenMFGrowth",
    ],
)
def test_lcdm_growth_rate_matches_linder(model):
    r"""In flat LCDM the growth rate is :math:`f \approx \Omega_m(z)^{0.55}`.

    Linder (2005) gives gamma = 0.55 for a cosmological constant, accurate to better
    than ~0.5% in f for Omega_m ~ 0.3 at all z.
    """
    z = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
    g = getattr(gf, model)(LCDM)
    # Tolerance: 1% (Linder's quoted accuracy plus margin); measured max 0.52% (z=0).
    np.testing.assert_allclose(g.growth_rate(z), LCDM.Om(z) ** 0.55, rtol=1e-2)


def test_lcdm_growth_rate_matches_linder_with_radiation():
    """The automatic selector with a realistic CMB temperature still obeys Linder."""
    cosmo = cosmology.FlatLambdaCDM(H0=67.7, Om0=0.31, Tcmb0=2.7255)
    z = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
    g = gf.GrowthFactor(cosmo)
    # Tolerance: as above; radiation changes f by <1e-4 at z <= 3.
    np.testing.assert_allclose(g.growth_rate(z), cosmo.Om(z) ** 0.55, rtol=1e-2)


@pytest.mark.parametrize("om0", [0.2, 0.3, 0.5])
def test_growth_suppression_matches_carroll_press_turner(om0):
    r"""The LCDM growth suppression g0 = D(a=1)/D_EdS(a=1) agrees with CPT92.

    Normalising to D -> a deep in matter domination (z = 1000, no radiation),
    :math:`g_0 = 1/[D(z)(1+z)]`. Carroll, Press & Turner (1992) Eq. 29 approximate it
    as :math:`\frac52\Omega_m/[\Omega_m^{4/7}-\Omega_\Lambda+(1+\Omega_m/2)(1+\Omega_\Lambda/70)]`.
    """
    cosmo = cosmology.FlatLambdaCDM(H0=70.0, Om0=om0, Tcmb0=0.0)
    z_early = 1000.0
    d_early = gf.GrowthFactor(cosmo).growth_factor(np.array([z_early]))[0]
    g0 = 1 / (d_early * (1 + z_early))

    ol = 1 - om0
    cpt = 2.5 * om0 / (om0 ** (4 / 7) - ol + (1 + om0 / 2) * (1 + ol / 70))
    # Tolerance: CPT92 is accurate to ~1% for 0.1 < Omega_m < 1 (Lahav et al. 1991);
    # measured max deviation 0.54% (Omega_m = 0.2). Residual (1+z)^-3 terms at
    # z = 1000 are ~1e-9.
    assert cpt == pytest.approx(g0, rel=1e-2)


@pytest.mark.parametrize(
    ("model", "cosmo"),
    [
        ("GrowthFactor", LCDM),
        ("ODEGrowthFactor", LCDM),
        ("IntegralGrowthFactor", LCDM),
        ("Eisenstein97GrowthFactor", LCDM),
        ("GenMFGrowth", LCDM),
        ("IntegralGrowthFactor", cosmology.LambdaCDM(H0=70, Om0=0.3, Ode0=0.5, Tcmb0=0.0)),
    ],
)
def test_growth_rate_is_log_derivative_of_growth_factor(model, cosmo):
    r"""The growth rate is by definition :math:`f = d\ln D/d\ln a`.

    Several models compute f with a separate analytic formula; it must equal a
    central finite difference of the model's own D(a).
    """
    g = getattr(gf, model)(cosmo)
    z = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    lna = np.log(1 / (1 + z))
    eps = 1e-4

    def lnd(x):
        return np.log(g.growth_factor(np.exp(-x) - 1))

    fd = (lnd(lna + eps) - lnd(lna - eps)) / (2 * eps)
    # Tolerance: O(eps^2) truncation ~1e-8 plus spline interpolation error; 1e-5.
    np.testing.assert_allclose(g.growth_rate(z), fd, rtol=1e-5)


def test_growth_rate_at_z0_carroll1992_matches_lahav():
    """Carroll1992's growth rate agrees with the exact LCDM growth rate.

    It is the Lahav et al. (1991) approximation, accurate to ~1%.
    """
    z = np.array([0.0, 0.5, 1.0])
    exact = gf.ODEGrowthFactor(LCDM).growth_rate(z)
    approx = gf.Carroll1992(LCDM).growth_rate(z)
    # Tolerance: Lahav+91's fit is good to ~1% for 0.1 < Omega_m < 1; measured 0.5%.
    np.testing.assert_allclose(approx, exact, rtol=1e-2)


@pytest.mark.parametrize("model", ["ODEGrowthFactor", "IntegralGrowthFactor"])
@pytest.mark.parametrize("om0", [0.1, 0.3, 0.5])
def test_open_universe_growth_rate_matches_peebles(model, om0):
    r"""In an open, Lambda=0 universe, :math:`f_0 \approx \Omega_m^{0.6}` (Peebles 1980).

    Growth freezes as curvature takes over, so 0 < f < 1 today, and Peebles'
    approximation holds to a few percent for 0.1 < Omega_m < 1.
    """
    cosmo = cosmology.LambdaCDM(H0=70.0, Om0=om0, Ode0=0.0, Tcmb0=0.0)
    f0 = getattr(gf, model)(cosmo).growth_rate(np.array([0.0]))[0]
    assert 0 < f0 < 1
    # Tolerance: Peebles' approximation is good to a few percent; measured max
    # deviation 2.0% (Omega_m = 0.1).
    assert f0 == pytest.approx(om0**0.6, rel=0.05)
