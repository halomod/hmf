"""Physical tests of the v4 growth models and the Growth stage.

References: the exact Einstein-de Sitter solution, the agreement of independent exact
methods (ODE, integral form, closed forms) where their assumptions hold, an
independent integration of the growth equation written in another form (with
scipy's DOP853 at tight tolerances), published approximations with known accuracy
(Linder 2005, Peebles 1980, Carroll, Press & Turner 1992), and CAMB.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy import cosmology
from scipy.integrate import solve_ivp

from hmf.core import growth_models as gm
from hmf.core.domain import DomainError
from hmf.core.growth import Growth

Z = np.array([0.0, 0.5, 1.0, 3.0, 10.0, 100.0])

EDS = cosmology.LambdaCDM(H0=70.0, Om0=1.0, Ode0=0.0, Tcmb0=0.0)
LCDM = cosmology.FlatLambdaCDM(H0=67.7, Om0=0.31, Tcmb0=0.0, name="LCDM")
OPEN = cosmology.LambdaCDM(H0=70.0, Om0=0.3, Ode0=0.0, Tcmb0=0.0)
OPEN_LAMBDA = cosmology.LambdaCDM(H0=70.0, Om0=0.3, Ode0=0.5, Tcmb0=0.0)
WCDM = cosmology.FlatwCDM(H0=67.7, Om0=0.31, w0=-0.8, Tcmb0=2.7255)
W0WA = cosmology.w0waCDM(H0=67.7, Om0=0.31, Ode0=0.72, w0=-0.9, wa=-0.3, Tcmb0=2.7255)
NU = cosmology.FlatLambdaCDM(
    H0=67.7, Om0=0.31, Ob0=0.049, Tcmb0=2.7255, m_nu=[0, 0, 0.3] * u.eV, name="nu"
)

LCDM_RADIATION = cosmology.FlatLambdaCDM(H0=67.7, Om0=0.31, Tcmb0=2.7255)

GRIDDED = ["ODE", "Integral", "Eisenstein97", "Heath77", "GenMF", "Carroll92"]


def growth(model, cosmo):
    return Growth(cosmology=cosmo, model=model)


@pytest.mark.parametrize("model", GRIDDED)
def test_einstein_de_sitter_growth_is_the_scale_factor(model):
    """In EdS the growing mode is exactly D = a, with f = 1."""
    g = growth(model, EDS)
    # Tolerance: RK4 (ODE) and Simpson (Integral, GenMF) errors on the grid; measured
    # max 4e-10 (ODE) and 3e-8 (Integral); the closed forms are exact.
    np.testing.assert_allclose(g.growth_factor(Z) * (1 + Z), 1.0, rtol=1e-7)
    np.testing.assert_allclose(g.growth_rate(Z), 1.0, rtol=1e-7)


@pytest.mark.parametrize("model", ["Integral", "Eisenstein97", "GenMF"])
def test_lcdm_ode_agrees_with_exact_integral_forms(model):
    """Without radiation, w = -1: the ODE and the integral form are both exact."""
    ode, other = growth("ODE", LCDM), growth(model, LCDM)
    z = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 20.0])
    # Tolerance: RK4 (h = 0.01) and Simpson errors; measured max 1.1e-8 in D and
    # 3.1e-8 in f.
    np.testing.assert_allclose(other.growth_factor(z), ode.growth_factor(z), rtol=1e-7)
    np.testing.assert_allclose(other.growth_rate(z), ode.growth_rate(z), rtol=1e-6)


@pytest.mark.parametrize("model", ["Integral", "Heath77", "GenMF"])
def test_open_universe_ode_agrees_with_exact_forms(model):
    ode, other = growth("ODE", OPEN), growth(model, OPEN)
    z = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 20.0])
    # Tolerance: as above; measured max 6.6e-9 in D and 1.3e-8 in f.
    np.testing.assert_allclose(other.growth_factor(z), ode.growth_factor(z), rtol=1e-7)
    np.testing.assert_allclose(other.growth_rate(z), ode.growth_rate(z), rtol=1e-6)


def independent_growth(cosmo, z):
    r"""Integrate the growth equation in momentum form, with DOP853.

    :math:`d(a^3 E\, dD/da)/da = \frac32 \Omega_{m,0} D/(a^2 E)` needs only astropy's
    E(z), no derivative of it. It starts on the Meszaros growing mode
    :math:`D = 1 + 3y/2`, :math:`y = a/a_{\rm eq}`, deep in radiation domination.
    """
    a_ini = 1e-7
    rad = cosmo.Ogamma0 * (1 + cosmo.nu_relative_density(1 / a_ini - 1)) if cosmo.Tcmb0.value else 0
    if rad:
        a_eq = rad / cosmo.Om0
        y = a_ini / a_eq
        d0, dd0 = 1 + 1.5 * y, 1.5 / a_eq
    else:
        d0, dd0 = a_ini, 1.0

    def e(a):
        return cosmo.efunc(1 / a - 1)

    def rhs(a, s):
        d, p = s  # p = a^3 E dD/da
        return [p / (a**3 * e(a)), 1.5 * cosmo.Om0 * d / (a**2 * e(a))]

    # z is increasing from 0, so a is decreasing from 1: integrate on reversed z.
    a_out = 1 / (1 + z[::-1])
    sol = solve_ivp(
        rhs, (a_ini, 1.0), [d0, a_ini**3 * e(a_ini) * dd0], method="DOP853",
        t_eval=a_out, rtol=1e-11, atol=1e-30,
    )  # fmt: skip
    assert sol.success
    d, p = sol.y
    f = p / (a_out**3 * e(a_out)) * a_out / d
    return (d / d[-1])[::-1], f[::-1]


@pytest.mark.parametrize(
    "cosmo", [WCDM, W0WA, OPEN, OPEN_LAMBDA, NU, LCDM_RADIATION],
    ids=["wCDM", "w0waCDM", "open", "open+Lambda", "massive-nu", "LCDM+radiation"],
)  # fmt: skip
def test_ode_matches_an_independent_integration(cosmo):
    z = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0])
    d_ref, f_ref = independent_growth(cosmo, z)
    g = growth("ODE", cosmo)
    # Tolerance: RK4 at h = 0.01 against DOP853 at rtol = 1e-11; measured max 2.1e-10
    # in D and 5.0e-10 in f.
    np.testing.assert_allclose(g.growth_factor(z), d_ref, rtol=1e-8)
    np.testing.assert_allclose(g.growth_rate(z), f_ref, rtol=1e-8)


@pytest.mark.parametrize("model", [*GRIDDED, "CAMB"])
def test_growth_is_normalised_and_monotonic(model):
    cosmo = OPEN if model == "Heath77" else (LCDM if model != "CAMB" else NU)
    g = growth(model, cosmo)
    assert g.growth_factor(0.0) == 1.0
    z = np.linspace(0, 20, 200)
    assert np.all(np.diff(g.growth_factor(z)) < 0)
    assert np.all((g.growth_rate(z) > 0) & (g.growth_rate(z) <= 1 + 1e-9))


@pytest.mark.parametrize("model", ["ODE", "Integral", "Eisenstein97", "GenMF", "Carroll92"])
def test_lcdm_growth_rate_matches_linder(model):
    """In flat LCDM, f ~ Omega_m(z)^0.55 to ~0.5% (Linder 2005)."""
    z = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
    # Tolerance: measured max 0.52% (z = 0; Carroll92: 0.31%).
    np.testing.assert_allclose(growth(model, LCDM).growth_rate(z), LCDM.Om(z) ** 0.55, rtol=1e-2)


@pytest.mark.parametrize("om0", [0.1, 0.3, 0.5])
def test_open_universe_growth_rate_matches_peebles(om0):
    """In an open universe without Lambda, f0 ~ Omega_m^0.6 (Peebles 1980)."""
    cosmo = cosmology.LambdaCDM(H0=70.0, Om0=om0, Ode0=0.0, Tcmb0=0.0)
    # Tolerance: Peebles' fit is good to a few percent; measured max 2.0% (0.1).
    assert growth("ODE", cosmo).growth_rate(0.0) == pytest.approx(om0**0.6, rel=0.05)


@pytest.mark.parametrize("om0", [0.2, 0.3, 0.5])
def test_lcdm_growth_suppression_matches_carroll_press_turner(om0):
    """g0 = D(a=1) / a in units where D -> a early, against CPT92 Eq. 29 (~1%)."""
    cosmo = cosmology.FlatLambdaCDM(H0=70.0, Om0=om0, Tcmb0=0.0)
    g0 = 1 / (growth("ODE", cosmo).growth_factor(1000.0) * 1001)
    ol = 1 - om0
    cpt = 2.5 * om0 / (om0 ** (4 / 7) - ol + (1 + om0 / 2) * (1 + ol / 70))
    assert cpt == pytest.approx(g0, rel=1e-2)


@pytest.mark.parametrize("model", ["ODE", "Integral", "Eisenstein97", "CAMB"])
def test_growth_rate_is_the_log_derivative_of_the_growth_factor(model):
    g = growth(model, LCDM if model != "CAMB" else NU)
    z = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    ln_a, eps = -np.log1p(z), 1e-4

    def ln_d(x):
        return np.log(g.growth_factor(np.expm1(-x)))

    fd = (ln_d(ln_a + eps) - ln_d(ln_a - eps)) / (2 * eps)
    # Tolerance: O(eps^2) plus spline interpolation (the Boltzmann codes' growth is
    # tabulated at 201 redshifts); measured max 1.4e-6 (Integral) and 1.1e-6 (CAMB).
    np.testing.assert_allclose(g.growth_rate(z), fd, rtol=1e-5)


@pytest.mark.parametrize("m_nu", [0.06, 0.3])
def test_massive_neutrino_ode_matches_camb_cb_on_small_scales(m_nu):
    """Below the free-streaming scale CAMB's delta_nonu grows as the ODE's D.

    Uses CAMB directly at k/h = 5, independently of hmf's CAMB growth (which is at
    k = 0.01/Mpc, where the neutrinos partly cluster).
    """
    camb = pytest.importorskip("camb")
    from hmf.core._boltzmann import _camb_params
    from hmf.core.accuracy import KAccuracy
    from hmf.core.transfer_models import CAMB

    cosmo = NU.clone(m_nu=[0, 0, m_nu] * u.eV)
    results = camb.get_transfer_functions(_camb_params(CAMB().run_input(cosmo, KAccuracy())))
    z = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 10.0])
    delta = results.get_redshift_evolution(5.0 * cosmo.h, z, ["delta_nonu"]).ravel()
    # Tolerance: CAMB's baryon and residual scale dependence at k/h = 5; measured
    # max 4.4e-5 (v3, same physics).
    np.testing.assert_allclose(growth("ODE", cosmo).growth_factor(z), delta / delta[0], rtol=2e-4)


def test_camb_growth_is_scale_dependent_with_massive_neutrinos():
    """At k = 0.01/Mpc the neutrinos partly cluster: cb grows faster than the ODE."""
    camb, ode = growth("CAMB", NU), growth("ODE", NU)
    # Normalised to 1 today, faster growth means smaller D at high z: about -2% at
    # z = 10 for a 0.3 eV neutrino (measured -2.2%, as in v3).
    ratio = camb.growth_factor(10.0, "cb") / ode.growth_factor(10.0)
    assert 0.96 < ratio < 0.99
    # Without massive neutrinos they agree, up to baryons and radiation at this scale.
    massless = NU.clone(m_nu=[0, 0, 0] * u.eV)
    z = np.array([1.0, 3.0, 10.0])
    # Tolerance: measured 3.6e-5.
    np.testing.assert_allclose(
        growth("CAMB", massless).growth_factor(z),
        growth("ODE", massless).growth_factor(z),
        rtol=2e-4,
    )


# ---------------------------------------------------------------------------------
# Models and stage
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "cosmo", "error"),
    [
        ("Integral", WCDM, TypeError),
        ("GenMF", WCDM, TypeError),
        ("Eisenstein97", OPEN_LAMBDA, ValueError),
        ("Heath77", LCDM, ValueError),
        ("GenMF", cosmology.LambdaCDM(H0=70, Om0=0.5, Ode0=0.7, Tcmb0=0), ValueError),
        ("CAMB", LCDM, ValueError),  # no baryons
    ],
)
def test_models_reject_cosmologies_they_do_not_apply_to(model, cosmo, error):
    with pytest.raises(error):
        Growth(cosmology=cosmo, model=model)


def test_redshift_domain():
    g = Growth(model="ODE")
    with pytest.raises(DomainError):
        g.growth_factor(-0.1)
    with pytest.raises(DomainError):
        g.growth_factor(np.array([0.0, np.nan]))
    with pytest.raises(DomainError):
        Growth(cosmology=NU, model="CAMB").growth_factor(25.0)  # z_max = 20


def test_redshifts_broadcast_like_numpy():
    g = Growth()
    assert isinstance(g.growth_factor(1.0), float)
    assert isinstance(g.growth_rate(np.float64(1.0)), float)
    assert g.growth_factor(np.ones((2, 3))).shape == (2, 3)
    assert g.growth_factor([0.0, 1.0]).shape == (2,)


def test_fromarray_and_fromfile(tmp_path):
    z = np.linspace(0, 10, 50)
    d = 3.0 / (1 + z)
    g = Growth(cosmology=EDS, model=gm.FromArray(z=z, d=d))
    np.testing.assert_allclose(g.growth_factor([0.5, 2.0]), [1 / 1.5, 1 / 3], rtol=1e-9)
    np.testing.assert_allclose(g.growth_rate([0.5, 2.0]), 1.0, rtol=1e-6)
    np.savetxt(tmp_path / "d.txt", np.column_stack([z, d]))
    g2 = Growth(cosmology=EDS, model=gm.FromFile(fname=tmp_path / "d.txt"))
    assert g2.growth_factor(2.0) == pytest.approx(g.growth_factor(2.0), rel=1e-12)
    with pytest.raises(ValueError, match="include 0"):
        gm.FromArray(z=z + 1, d=d)


def test_models_are_registered():
    for name in [
        "ODE",
        "Integral",
        "Eisenstein97",
        "Heath77",
        "GenMF",
        "Carroll92",
        "CAMB",
        "CLASS",
    ]:
        assert issubclass(gm.GrowthModel.get(name), gm.GrowthModel)


def test_species():
    g = Growth()
    assert g.growth_factor(1.0, "tot") == g.growth_factor(1.0, "cb")
    with pytest.raises(ValueError, match="species"):
        g.growth_factor(1.0, "nu")
