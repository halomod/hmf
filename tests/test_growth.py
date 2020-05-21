import numpy as np
from hmf.cosmology import growth_factor
from astropy.cosmology import Planck13
import pytest


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
        gf.growth_factor(z), genf.growth_factor(z), rtol=1e-2 + z / 500.0,
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
