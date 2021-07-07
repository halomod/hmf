import pytest

import numpy as np
from astropy.cosmology import Planck13, w0waCDM

from hmf.cosmology import growth_factor


@pytest.fixture(scope="module")
def gf():
    return growth_factor.GrowthFactor(Planck13)


@pytest.fixture(scope="module")
def genf():
    return growth_factor.GenMFGrowth(Planck13, zmax=10.0)


@pytest.mark.parametrize("z", np.arange(0, 8, 0.5))
def test_gf(z, gf, genf):
    print(gf.growth_factor(z), genf.growth_factor(z))
    assert np.isclose(
        gf.growth_factor(z),
        genf.growth_factor(z),
        rtol=1e-2 + z / 500.0,
    )


@pytest.mark.parametrize("z", np.arange(0, 8, 0.5))
def test_gr(z, gf, genf):
    gf.growth_rate(z), genf.growth_rate(z)
    assert np.isclose(gf.growth_rate(z), genf.growth_rate(z), rtol=1e-2 + z / 100.0)


def test_gfunc(gf, genf):
    gf_func = gf.growth_factor_fn(0.0)
    genf_func = genf.growth_factor_fn(0.0)

    print(gf_func(np.linspace(0, 5, 10)), genf_func(np.linspace(0, 5, 10)))
    assert np.allclose(
        gf_func(np.linspace(0, 5, 10)), genf_func(np.linspace(0, 5, 10)), rtol=1e-2
    )


def test_gr_func(gf, genf):
    gr_func = gf.growth_rate_fn(0.0)
    genf_func = genf.growth_rate_fn(0.0)

    print(gr_func(np.linspace(0, 5, 10)), genf_func(np.linspace(0, 5, 10)))
    assert np.allclose(
        gr_func(np.linspace(0, 5, 10)), genf_func(np.linspace(0, 5, 10)), rtol=1e-2
    )


def test_inverse(gf, genf):
    gf_func = gf.growth_factor_fn(0.0, inverse=True)
    genf_func = genf.growth_factor_fn(0.0, inverse=True)

    gf = np.linspace(0.15, 0.99, 10)
    print(gf_func(gf), genf_func(gf))
    assert np.allclose(gf_func(gf), genf_func(gf), rtol=1e-1)


def test_unsupported_cosmo():
    cosmo = w0waCDM(H0=70.0, Om0=0.3, Ode0=0.7, w0=-0.9, Ob0=0.05, Tcmb0=2.7)
    with pytest.raises(ValueError):
        growth_factor.GenMFGrowth(cosmo=cosmo)

    # But shouldn't raise error for CAMBGrowth
    growth_factor.CambGrowth(cosmo=cosmo)


def test_carroll(gf):
    cgf = growth_factor.Carroll1992(Planck13)

    z = np.arange(6)
    np.testing.assert_allclose(gf.growth_rate(z), cgf.growth_rate(z), rtol=0.05)
