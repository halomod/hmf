import pytest

import numpy as np
from astropy.cosmology import FlatLambdaCDM, FlatwCDM

from hmf.cosmology.growth_factor import CambGrowth
from hmf.density_field.transfer import Transfer


@pytest.fixture(scope="module")
def wcdm():
    return FlatwCDM(H0=70.0, w0=-0.9, Om0=0.3, Ob0=0.05, Tcmb0=2.7)


@pytest.fixture(scope="module")
def lcdm():
    return FlatLambdaCDM(H0=70.0, Om0=0.3, Ob0=0.05, Tcmb0=2.7)


@pytest.fixture(scope="module")
def t_wcdm(wcdm):
    return Transfer(cosmo_model=wcdm)


@pytest.fixture(scope="module")
def t_lcdm(lcdm):
    return Transfer(cosmo_model=lcdm)


def test_defaults(t_wcdm):
    assert isinstance(t_wcdm.growth, CambGrowth)


def test_trivial(t_wcdm, t_lcdm):
    t_wcdm.update(z=1.0, cosmo_params={"w0": -1.0})
    t_lcdm.update(z=1.0)

    assert np.isclose(t_lcdm.growth_factor, t_wcdm.growth_factor, 1e-2)

    t_lcdm.update(z=5.0)
    t_wcdm.update(z=5.0)

    assert np.isclose(t_lcdm.growth_factor, t_wcdm.growth_factor, 1e-2)


def test_divergence(t_lcdm, t_wcdm):
    t_wcdm.update(cosmo_params={"w0": -0.5}, z=4)
    t_lcdm.update(z=4)
    assert not np.isclose(t_lcdm.growth_factor, t_wcdm.growth_factor, 1e-1)
