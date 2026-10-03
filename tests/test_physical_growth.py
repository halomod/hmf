"""Physical tests of the linear growth factor and growth rate.

References are exact limits (Einstein-de Sitter), the definition of the growth rate,
CAMB on scales below the neutrino free-streaming length, an independent formulation
of the growth equation, and published approximations with known accuracy
(Carroll, Press & Turner 1992; Linder 2005). No reference re-implements the method under test.
"""

import astropy.units as u
import camb
import numpy as np
import pytest
from astropy import cosmology
from scipy.integrate import solve_ivp

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


def _massive_nu_lcdm(m_nu: float) -> cosmology.FlatLambdaCDM:
    """Flat LCDM with one massive neutrino of mass ``m_nu`` eV."""
    return cosmology.FlatLambdaCDM(
        H0=67.7, Om0=0.31, Ob0=0.049, Tcmb0=2.7255, m_nu=[0.0, 0.0, m_nu] * u.eV
    )


@pytest.mark.parametrize(
    "cosmo", [cosmology.Planck18, _massive_nu_lcdm(0.06), _massive_nu_lcdm(0.3)]
)
def test_dlne_dlna_is_log_derivative_of_efunc_with_massive_neutrinos(cosmo):
    r"""``dlne_dlna`` is by definition :math:`d\ln E/d\ln a` of astropy's ``efunc``.

    With massive neutrinos the radiation term :math:`\Omega_\gamma (1+N(z)) a^{-4}`
    has a time-dependent :math:`N(z)` as the neutrinos turn non-relativistic, whose
    derivative must be included. Before that was fixed the error was ~2e-3 near
    z ~ 8 for Planck18.
    """
    g = gf.ODEGrowthFactor(cosmo)
    z = np.array([0.0, 1.0, 8.0, 100.0, 1e3, 1e5])
    lna = -np.log1p(z)
    eps = 1e-5

    def lne(x):
        return np.log(cosmo.efunc(np.expm1(-x)))

    fd = (lne(lna + eps) - lne(lna - eps)) / (2 * eps)
    # Tolerance: O(eps^2) truncation plus the code's own O(1e-8) finite difference
    # of N(z); measured max 6e-11.
    np.testing.assert_allclose(g.dlne_dlna(z), fd, rtol=1e-8)


@pytest.mark.parametrize("m_nu", [0.06, 0.3])
def test_massive_neutrino_ode_growth_matches_camb_cb_on_small_scales(m_nu):
    r"""The ODE growth matches CAMB's CDM+baryon growth well below the free-streaming scale.

    The growth ODE treats massive neutrinos as a smooth background that adds to
    :math:`H(z)` but not to the source term. This is exact on scales much smaller
    than the neutrino free-streaming length, where CAMB's ``delta_nonu`` grows
    scale-independently. At k/h = 5 (versus k_fs/h <~ 0.3 for m_nu <= 0.3 eV at
    z <= 10) that is the right external reference. Larger scales, where neutrinos
    partly cluster, are not, because the cb growth there is enhanced.
    """
    cosmo = _massive_nu_lcdm(m_nu)
    z = np.array([0.5, 1.0, 2.0, 4.0, 6.0, 10.0])

    p = gf.CambGrowth._get_camb_params(cosmo)
    p.set_matter_power(redshifts=[0.0], kmax=20.0)
    transfers = camb.get_transfer_functions(p)
    k = 5.0 * cosmo.h  # CAMB takes k in 1/Mpc

    def delta_cb(zz):
        return transfers.get_redshift_evolution(k, zz, ["delta_nonu"]).flatten()

    camb_growth = delta_cb(z) / delta_cb(0.0)[0]

    # Tolerance: measured max 4.4e-5, residual from CAMB's baryon and scale
    # dependence at k/h = 5. Before the dlne_dlna fix the error was 1.9e-3
    # (0.06 eV) and 9.2e-3 (0.3 eV) at z = 10.
    np.testing.assert_allclose(gf.ODEGrowthFactor(cosmo).growth_factor(z), camb_growth, rtol=2e-4)


@pytest.mark.parametrize("model", ["GrowthFactor", "ODEGrowthFactor"])
@pytest.mark.parametrize("m_nu", [0.06, 0.3])
def test_massive_neutrino_growth_matches_momentum_form_ode(model, m_nu):
    r"""Growth with massive neutrinos matches an independent momentum-form ODE.

    The linear growth equation can be written without any derivative of
    :math:`H`, as
    :math:`d(a^2 E\, dD/d\ln a)/d\ln a = \tfrac32 \Omega_{m,0} D/(aE)`, which needs
    only astropy's :math:`E(z)`. It is started on the exact Meszaros growing mode
    :math:`D \propto 1 + 3a/(2a_{\rm eq})` deep in radiation domination. It checks both
    D and :math:`f = d\ln D/d\ln a`, which CAMB does not provide directly.
    """
    cosmo = _massive_nu_lcdm(m_nu)
    z = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 10.0])

    # Neutrinos are fully relativistic at a = 1e-6 for m_nu <= 0.3 eV.
    a_ini = 1e-6
    a_eq = cosmo.Ogamma0 * (1 + cosmo.nu_relative_density(1 / a_ini - 1)) / cosmo.Om0
    y = a_ini / a_eq

    def efunc(lna):
        return cosmo.efunc(np.expm1(-lna))

    def rhs(lna, state):
        a = np.exp(lna)
        d, p = state
        e = efunc(lna)
        return [p / (a**2 * e), 1.5 * cosmo.Om0 * d / (a * e)]

    lna_out = np.sort(-np.log1p(z))
    sol = solve_ivp(
        rhs,
        (np.log(a_ini), 0.0),
        [1 + 1.5 * y, a_ini**2 * efunc(np.log(a_ini)) * 1.5 * y],
        t_eval=lna_out,
        rtol=1e-10,
        atol=1e-14,
        method="DOP853",
    )
    a = np.exp(sol.t)
    d, p = sol.y
    ref_d = (d / d[-1])[::-1]
    ref_f = (p / (a**2 * efunc(sol.t)) / d)[::-1]

    g = getattr(gf, model)(cosmo)
    # Tolerance: measured max 3.6e-7 in D and 5.0e-6 in f (spline derivative).
    # Before the dlne_dlna fix the errors were 1.9e-3 / 8.9e-4 (0.06 eV) and
    # 9.2e-3 / 4.5e-3 (0.3 eV).
    np.testing.assert_allclose(g.growth_factor(z), ref_d, rtol=2e-5)
    np.testing.assert_allclose(g.growth_rate(z), ref_f, rtol=2e-5)
