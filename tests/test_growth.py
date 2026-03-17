import copy

import numpy as np
import pytest
from astropy import cosmology
from astropy.cosmology import Planck13, w0waCDM

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
    assert np.isclose(gf.growth_rate(z), 0, atol=0.01)
    np.testing.assert_allclose(
        gf._d_plus_unnormalized(z), 2 * cosmo.Ogamma0 / 3 / cosmo.Om0, rtol=1e-2
    )


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


@pytest.mark.filterwarnings("ignore:not accurate at high redshifts")
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
    with pytest.raises(ValueError, match="only accurate for LambdaCDM cosmologies"):
        growth_factor.GenMFGrowth(cosmo=cosmo).growth_factor(0)

    # But shouldn't raise error for CAMBGrowth
    growth_factor.CambGrowth(cosmo=cosmo)


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
    gf.growth_rate(100)

    # This normally wouldn't need the ODE solver, but since it's already
    # instantiated, it might as well use it.
    gf.growth_factor(1)
    gf.growth_rate(1)


def test_growth_rate_uses_integral():
    """Test it uses the integral method."""
    cosmo = Planck13.clone(Tcmb0=2.725, Ode0=1.1, to_nonflat=True)
    gf = growth_factor.GrowthFactor(cosmo)

    # At a low redshift, this will trigger using the integral method, which should work
    # even when Ode0 is not 0.
    gf.growth_rate(0.1)


def test_growth_rate_uses_ode_at_highz():
    """Test that if we call growth_rate at high z, it uses the ODE method."""
    cosmo = Planck13.clone(Tcmb0=2.725)
    gf = growth_factor.GrowthFactor(cosmo)

    # At a high redshift, this will trigger using the ODE method.
    gf.growth_rate(1000)


def test_expected_warnings():
    """Test that we get expected warnings for unsupported cosmologies."""
    cosmo = Planck13
    with pytest.warns(UserWarning, match="not accurate at high redshifts"):
        growth_factor.IntegralGrowthFactor(cosmo=cosmo).growth_factor(10000)

    with pytest.warns(UserWarning, match="only accurate for cosmologies with a constant"):
        growth_factor.IntegralGrowthFactor(
            cosmo=cosmology.w0waCDM(H0=70.0, Om0=0.3, Ode0=0.7, w0=-0.9, Ob0=0.05, Tcmb0=2.7)
        ).growth_factor(0)

    with pytest.raises(ValueError, match=r"Redshifts <0 not supported"):
        growth_factor.IntegralGrowthFactor(cosmo=cosmo).growth_factor(-0.5)

    with (
        pytest.raises(ValueError, match="Cannot compute integral"),
        pytest.warns(UserWarning, match="not accurate at high redshifts"),
    ):
        growth_factor.IntegralGrowthFactor(cosmo, amin=1e-3).growth_factor(1e5)

    with pytest.raises(ValueError, match="Eisenstein97GrowthFactor only supports flat"):
        growth_factor.Eisenstein97GrowthFactor(
            cosmo=Planck13.clone(Tcmb0=0.0, Om0=0.3, Ode0=0.7, to_nonflat=True)
        ).growth_factor(0)

    with pytest.warns(
        UserWarning,
        match=(
            "The Heath77GrowthFactor is only accurate for cosmologies with a constant dark energy"
        ),
    ):
        growth_factor.Heath77GrowthFactor(
            cosmo=cosmology.w0waCDM(H0=70.0, Tcmb0=0.0, Om0=0.3, Ode0=0.7, w0=-0.9, wa=0.3)
        ).growth_factor(0)

    with pytest.raises(ValueError, match=r"Heath77GrowthFactor cannot compute OmegaM = 1 case"):
        growth_factor.Heath77GrowthFactor(
            Planck13.clone(Tcmb0=0.0, Om0=1.0, Ode0=0.3, to_nonflat=True)
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
            Planck13.clone(Tcmb0=0.0, Om0=1.2, Ode0=-0.1, to_nonflat=True)
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
