import copy

import numpy as np
import pytest
from astropy import cosmology
from astropy import units as u
from astropy.cosmology import Planck13, w0waCDM

from hmf import MassFunction
from hmf.cosmology import growth_factor


@pytest.fixture(scope="module")
def gf():
    return growth_factor.GrowthFactor(Planck13)


@pytest.fixture(scope="module")
def genf():
    return growth_factor.GenMFGrowth(Planck13)


def test_growth_rate_ode_high_z_with_rad():
    """Test that for radiation-dominated universe, D+ ~ a^2 at high z."""
    cosmo = Planck13.clone(Om0=0.3, Tcmb0=2.7)
    a = cosmo.Ogamma0 / cosmo.Om0 / 100

    gf = growth_factor.ODEGrowthFactor(cosmo, amin=a / 10)

    z = 1 / a - 1
    assert np.isclose(gf._growth_rate(z), 0, atol=0.01)
    np.testing.assert_allclose(gf._d_plus_unnormalized(z), 2 * gf.Or0 / 3 / cosmo.Om0, rtol=1e-2)


@pytest.mark.parametrize(
    "model",
    [
        "GrowthFactor",
        "ODEGrowthFactor",
        "IntegralGrowthFactor",
        "Eisenstein97GrowthFactor",
        "Carroll1992",
        "GenMFGrowth",
    ],
)
def test_growth_rate_at_high_z_no_rad(model):
    """Test that for matter-dominated universe (no radiation), D+ ~ a at high z."""
    cosmo = Planck13.clone(Tcmb0=0.0, Om0=0.3)
    gf = getattr(growth_factor, model)(cosmo)

    z = 199
    assert np.isclose(gf.growth_rate(z), 1, rtol=1e-2)


@pytest.mark.parametrize("omegal", [0.0, 0.7, 1.3])
@pytest.mark.parametrize("omegam", [0.3, 0.1, 1.0])
def test_ode_vs_integral_method(omegam, omegal):
    """Test that ODE and integral methods give same answer for growth factor."""
    cosmo = Planck13.clone(Tcmb0=0.0, Om0=omegam, Ode0=omegal, to_nonflat=True)
    gf_ode = growth_factor.ODEGrowthFactor(cosmo)
    gf_integral = growth_factor.IntegralGrowthFactor(cosmo)

    z = np.linspace(0, 100, 1000)
    d_ode = gf_ode.growth_factor(z)
    d_integral = gf_integral.growth_factor(z)

    np.testing.assert_allclose(d_ode, d_integral, rtol=1e-3)


@pytest.mark.parametrize("omegam", [0.3, 0.1, 1.5])
def test_heath_vs_ode_no_omegal(omegam):
    """Test that Heath's formula matches ODE solution for growth factor when Omegal=0."""
    cosmo = Planck13.clone(Tcmb0=0.0, Om0=omegam, Ode0=0.0, to_nonflat=True)
    gf_ode = growth_factor.ODEGrowthFactor(cosmo)
    gf_heath = growth_factor.Heath77GrowthFactor(cosmo)

    z = np.linspace(0, 100, 1000)
    d_ode = gf_ode.growth_factor(z)
    d_heath = gf_heath.growth_factor(z)

    np.testing.assert_allclose(d_ode, d_heath, rtol=1e-3)


@pytest.mark.filterwarnings(
    "ignore:The IntegralGrowthFactor is not accurate when the radiation density"
)
@pytest.mark.filterwarnings(
    "ignore:The Eisenstein97GrowthFactor is not accurate when the radiation density"
)
@pytest.mark.parametrize(
    "model",
    [
        "GrowthFactor",
        "ODEGrowthFactor",
        "IntegralGrowthFactor",
        "Eisenstein97GrowthFactor",
        "Carroll1992",
        "CambGrowth",
        "GenMFGrowth",
    ],
)
@pytest.mark.filterwarnings("ignore:matter_species was not set")
def test_growth_factor_monotonic(model):
    cosmo = Planck13
    gf = getattr(growth_factor, model)(cosmo)
    z = np.linspace(0, 100, 1000)

    d = gf.growth_factor(z[::-1])
    assert np.all(np.diff(d) > 0)


def test_integral_matches_ode_for_no_radiation():
    cosmo = Planck13.clone(Tcmb0=0.0)
    gf_ode = growth_factor.ODEGrowthFactor(cosmo)
    gf_integral = growth_factor.IntegralGrowthFactor(cosmo)

    z = np.linspace(0, 100, 1000)
    d_ode = gf_ode.growth_factor(z)
    d_integral = gf_integral.growth_factor(z)

    np.testing.assert_allclose(d_ode, d_integral, rtol=1e-2)


@pytest.mark.parametrize("omegam", [0.3, 0.1, 0.9])
def test_eisenstein_matches_ode_for_flat_no_radiation(omegam):
    cosmo = Planck13.clone(Tcmb0=0.0, Om0=omegam)
    gf_ode = growth_factor.ODEGrowthFactor(cosmo)
    gf_eisenstein = growth_factor.Eisenstein97GrowthFactor(cosmo)

    z = np.linspace(0, 100, 1000)
    d_ode = gf_ode.growth_factor(z)
    d_eisenstein = gf_eisenstein.growth_factor(z)

    np.testing.assert_allclose(d_ode, d_eisenstein, rtol=1e-3)


def test_carroll_good_approximation():
    cosmo = Planck13.clone(Tcmb0=0.0)
    gf_ode = growth_factor.ODEGrowthFactor(cosmo)
    gf_carroll = growth_factor.Carroll1992(cosmo)

    z = np.linspace(0, 10, 100)
    d_ode = gf_ode.growth_factor(z)
    d_carroll = gf_carroll.growth_factor(z)

    np.testing.assert_allclose(d_ode, d_carroll, rtol=0.05)


def test_genmf_good_approximation():
    cosmo = Planck13.clone(Tcmb0=0.0)
    gf_ode = growth_factor.ODEGrowthFactor(cosmo)
    gf_genmf = growth_factor.GenMFGrowth(cosmo)

    z = np.linspace(0, 10, 100)
    d_ode = gf_ode.growth_factor(z)
    d_genmf = gf_genmf.growth_factor(z)

    np.testing.assert_allclose(d_ode, d_genmf, rtol=0.05)


def test_unsupported_cosmo():
    cosmo = w0waCDM(H0=70.0, Om0=0.3, Ode0=0.7, w0=-0.9, Ob0=0.05, Tcmb0=2.7)
    with pytest.raises(ValueError, match="only accurate with a cosmological constant"):
        growth_factor.GenMFGrowth(cosmo=cosmo).growth_factor(0)

    # But shouldn't raise error for CAMBGrowth
    growth_factor.CambGrowth(cosmo=cosmo).growth_factor(0)


@pytest.mark.filterwarnings("ignore:matter_species was not set")
def test_pickleability_of_cambgrowth():
    gf = growth_factor.CambGrowth(Planck13)
    gf_at_1 = gf.growth_factor(1.0)

    gf2 = copy.deepcopy(gf)

    assert gf2.growth_factor(1.0) == gf_at_1


def test_from_file(datadir):
    cosmo = w0waCDM(H0=70.0, Om0=0.3, Ode0=0.7, w0=-0.9, Ob0=0.05, Tcmb0=2.7)
    gf = growth_factor.FromFile(cosmo=cosmo, fname=f"{datadir}/growth_for_hmf_tests.dat")
    data_in = np.genfromtxt(f"{datadir}/growth_for_hmf_tests.dat")[:, [0, 1]]
    z = data_in[:, 0]
    d = data_in[:, 1]

    np.testing.assert_allclose(gf.growth_factor(z), d, rtol=0.05)


def test_from_array(datadir):
    cosmo = w0waCDM(H0=70.0, Om0=0.3, Ode0=0.7, w0=-0.9, Ob0=0.05, Tcmb0=2.7)
    data_in = np.genfromtxt(f"{datadir}/growth_for_hmf_tests.dat")[:, [0, 1]]
    z = data_in[:, 0]
    d = data_in[:, 1]

    gf = growth_factor.FromArray(cosmo=cosmo, z=z, d=d)
    np.testing.assert_allclose(gf.growth_factor(z), d, rtol=0.05)


def test_growth_factor_w0wa_but_actually_lambdacdm():
    """Test that if we give a w0waCDM with w0=-1 and wa=0, we get same as LambdaCDM."""
    cosmo = w0waCDM(H0=70.0, Om0=0.3, Ode0=0.7, w0=-1.0, wa=0.0, Ob0=0.05, Tcmb0=2.7)
    cosmo_lambda = cosmology.LambdaCDM(H0=70.0, Om0=0.3, Ode0=0.7, Ob0=0.05, Tcmb0=2.7)
    gf_w0wa = growth_factor.GrowthFactor(cosmo)
    gf_lambda = growth_factor.GrowthFactor(cosmo_lambda)

    z = np.linspace(0, 100, 1000)
    d_w0wa = gf_w0wa.growth_factor(z)
    d_lambda = gf_lambda.growth_factor(z)

    np.testing.assert_allclose(d_w0wa, d_lambda, rtol=1e-3)


def test_using_ode_when_it_is_already_computed():
    """Test that when the ODE solution is already computed, it doesn't recompute."""
    cosmo = Planck13.clone(Tcmb0=2.725)
    gf = growth_factor.GrowthFactor(cosmo)

    gf.growth_factor(100)  # This will trigger using the ODE solver
    gf._growth_rate(100)

    # This normally wouldn't need the ODE solver, but since it's already
    # instantiated, it might as well use it.
    gf.growth_factor(1)
    gf._growth_rate(1)


def test_growth_rate_uses_integral():
    """Test it uses the integral method."""
    cosmo = Planck13.clone(Tcmb0=2.725, Ode0=1.1, to_nonflat=True)
    gf = growth_factor.GrowthFactor(cosmo)

    # At a low redshift, this will trigger using the integral method, which should work
    # even when Ode0 is not 0.
    gf._growth_rate(0.1)


def test_growth_rate_uses_ode_at_highz():
    """Test that if we call growth_rate at high z, it uses the ODE method."""
    cosmo = Planck13.clone(Tcmb0=2.725)
    gf = growth_factor.GrowthFactor(cosmo)

    # At a high redshift, this will trigger using the ODE method.
    gf._growth_rate(1000)


def test_growth_selector_switches_at_calibrated_radiation_threshold():
    """The default selector should swap to ODE once the calibrated threshold is exceeded."""
    cosmo = cosmology.FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486, Tcmb0=2.7255)
    gf = growth_factor.GrowthFactor(cosmo)

    assert gf.radiation_density(1.0) < growth_factor.LOW_RADIATION_THRESHOLD
    assert gf._choose_solution(1.0) is gf._eisenstein_gf

    assert gf.radiation_density(2.0) > growth_factor.LOW_RADIATION_THRESHOLD
    assert gf._choose_solution(2.0) is gf._ode_gf


def test_growth_selector_keeps_tinker08_within_one_percent_below_threshold():
    """Below the selector threshold, default Tinker08 should stay within 1% of ODE."""
    common = {
        "hmf_model": "Tinker08",
        "transfer_model": "EH",
        "mdef_model": "SOMean",
        "mdef_params": {"overdensity": 200},
        "Mmin": 10,
        "Mmax": 14.2,
        "dlog10m": 0.02,
        "z": 1.0,
        "cosmo_params": {"H0": 67.74, "Om0": 0.3089, "Ob0": 0.0486},
        "sigma_8": 0.8159,
        "n": 0.9667,
    }
    default = MassFunction(**common)
    ode = MassFunction(growth_model="ODEGrowthFactor", **common)

    mask = default.m < 1e14
    relative_error = np.abs(default.dndlnm[mask] - ode.dndlnm[mask]) / ode.dndlnm[mask]
    assert np.max(relative_error) < 1e-2


def test_expected_warnings():
    """Test that we get expected warnings for unsupported cosmologies."""
    cosmo = Planck13
    with pytest.warns(UserWarning, match="not accurate when the radiation density is significant"):
        growth_factor.IntegralGrowthFactor(cosmo=cosmo).growth_factor(10000)

    with pytest.warns(UserWarning, match="only accurate for cosmologies with a constant"):
        growth_factor.IntegralGrowthFactor(
            cosmo=cosmology.w0waCDM(H0=70.0, Om0=0.3, Ode0=0.7, w0=-0.9, Ob0=0.05, Tcmb0=2.7)
        ).growth_factor(0)

    with pytest.raises(ValueError, match=r"Redshifts <0 not supported"):
        growth_factor.IntegralGrowthFactor(cosmo=cosmo).growth_factor(-0.5)

    with (
        pytest.raises(ValueError, match="Cannot compute integral"),
        pytest.warns(UserWarning, match="not accurate when the radiation density"),
    ):
        growth_factor.IntegralGrowthFactor(cosmo, amin=1e-3).growth_factor(1e5)

    with pytest.raises(ValueError, match="Eisenstein97GrowthFactor only supports flat"):
        growth_factor.Eisenstein97GrowthFactor(
            cosmo=Planck13.clone(Tcmb0=0.0, Om0=0.3, Ode0=0.8, to_nonflat=True)
        ).growth_factor(0)

    with (
        pytest.warns(
            UserWarning,
            match=("only accurate for cosmologies with a constant dark energy"),
        ),
        pytest.warns(
            UserWarning,
            match=("The Heath77GrowthFactor is only accurate for cosmologies with Lambda=0"),
        ),
    ):
        growth_factor.Heath77GrowthFactor(
            cosmo=cosmology.w0waCDM(H0=70.0, Tcmb0=0.0, Om0=0.3, Ode0=0.7, w0=-0.9, wa=0.3)
        ).growth_factor(0)

    with pytest.raises(ValueError, match=r"You must supply an array for both z and d"):
        growth_factor.FromArray(
            z=np.array([0, 1, 2]),
            cosmo=cosmo,
        ).growth_factor(0)

    with pytest.raises(ValueError, match=r"z and d must have same length"):
        growth_factor.FromArray(
            z=np.array([0, 1, 2]),
            d=np.array([1, 0.5]),
            cosmo=cosmo,
        ).growth_factor(0)

    with pytest.raises(ValueError, match="GenMFGrowth only supports flat or open"):
        growth_factor.GenMFGrowth(
            Planck13.clone(Om0=1.2, Ode0=-0.1, to_nonflat=True)
        ).growth_factor(0)


def heath_growth_factor_einstein_de_sitter():
    """Test that Heath77GrowthFactor is correct for an Einstein-de Sitter universe."""
    cosmo = Planck13.clone(Tcmb0=0.0, Om0=1.0, Ode0=0.0)
    gf_heath = growth_factor.Heath77GrowthFactor(cosmo)

    z = np.linspace(0, 100, 1000)
    d_heath = gf_heath.growth_factor(z)

    np.testing.assert_allclose(d_heath, 1 / (1 + z), rtol=1e-3)


def test_genmf_vs_integral_negative_omegal():
    """Test that GenMFGrowth matches the integral method even for negative Omegal."""
    cosmo = Planck13.clone(Tcmb0=0.0, Om0=0.3, Ode0=-0.1, to_nonflat=True)
    gf_genmf = growth_factor.GenMFGrowth(cosmo)
    gf_integral = growth_factor.IntegralGrowthFactor(cosmo)

    z = np.linspace(0, 10, 100)
    d_genmf = gf_genmf.growth_factor(z)
    d_integral = gf_integral.growth_factor(z)

    np.testing.assert_allclose(d_genmf, d_integral, rtol=0.05)


@pytest.mark.parametrize("m_nu", [0.0, 0.3])
def test_cambgrowth_matter_species(m_nu):
    cosmo = cosmology.FlatLambdaCDM(
        H0=70.0, Om0=0.3, Ob0=0.05, Tcmb0=2.7255, m_nu=[0.0, 0.0, m_nu] * u.eV
    )
    z = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
    gf_tot = growth_factor.CambGrowth(cosmo, matter_species="tot")
    d_tot = gf_tot.growth_factor(z)
    d_cb = growth_factor.CambGrowth(cosmo, matter_species="cb").growth_factor(z)

    if m_nu == 0:
        np.testing.assert_allclose(d_cb, d_tot, rtol=1e-6)
    else:
        # delta_tot = (1 - f_nu) delta_cb + f_nu delta_nu with 0 <= delta_nu <= delta_cb.
        # As neutrinos cool, delta_nu catches up with delta_cb, so delta_tot grows faster
        # than delta_cb and the z=0-normalised cb growth factor is larger at z > 0.
        # Since delta_tot / delta_cb lies in [1 - f_nu, 1], D_cb / D_tot <= 1 / (1 - f_nu).
        p = gf_tot.p
        f_nu = p.omnuh2 / (p.omnuh2 + p.omch2 + p.ombh2)
        ratio = d_cb[1:] / d_tot[1:]
        assert d_cb[0] == pytest.approx(1.0)
        assert np.all(ratio > 1 + 1e-4)
        assert np.all(ratio <= 1 / (1 - f_nu))
        # The difference grows with redshift as the neutrinos were hotter.
        assert np.all(np.diff(ratio) > 0)


def test_cambgrowth_bad_matter_species():
    with pytest.raises(ValueError, match="matter_species must be one of"):
        growth_factor.CambGrowth(Planck13, matter_species="nu")


def test_cambgrowth_default_matter_species_is_cb_with_warning():
    cosmo = cosmology.FlatLambdaCDM(
        H0=70.0, Om0=0.3, Ob0=0.05, Tcmb0=2.7255, m_nu=[0.0, 0.0, 0.06] * u.eV
    )
    with pytest.warns(UserWarning, match="matter_species was not set for CambGrowth"):
        gf = growth_factor.CambGrowth(cosmo)
    assert gf.params["matter_species"] == "cb"


@pytest.mark.parametrize("matter_species", ["cb", "tot"])
def test_cambgrowth_growth_rate(matter_species):
    """CambGrowth's growth rate is f ~ Omega_m(z)^0.55 (Linder 2005), to ~1% in LCDM.

    Regression test: the extra matter_species parameter used to be passed on to the
    ODE solver that computes the growth rate, which rejected it.
    """
    gf = growth_factor.CambGrowth(Planck13, matter_species=matter_species)
    for z in (0.0, 1.0, 3.0):
        assert gf.growth_rate(z) == pytest.approx(Planck13.Om(z) ** 0.55, rel=1e-2)


def test_growthfactor_open_universe_uses_heath77():
    """In an open, Lambda=0, radiation-free universe, GrowthFactor uses Heath (1977).

    Heath's closed form is exact here, so it must agree with the numerical ODE solution.
    """
    cosmo = cosmology.LambdaCDM(H0=70.0, Om0=0.3, Ode0=0.0, Tcmb0=0.0)
    gf = growth_factor.GrowthFactor(cosmo)
    z = np.array([0.5, 1.0, 3.0, 10.0])

    assert isinstance(gf._choose_solution(z), growth_factor.Heath77GrowthFactor)
    np.testing.assert_allclose(
        gf.growth_factor(z), growth_factor.ODEGrowthFactor(cosmo).growth_factor(z), rtol=1e-5
    )
