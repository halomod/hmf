from hmf.transfer import Transfer
from astropy.cosmology import FlatwCDM, FlatLambdaCDM
from hmf.growth_factor import CambGrowth
import numpy as np


def test_defaults():
    cosmo = FlatwCDM(H0 = 70.0, w0 = -0.9, Om0 = 0.3, Ob0 = 0.05, Tcmb0=2.7)

    t = Transfer(cosmo_model=cosmo)
    assert isinstance(t.growth, CambGrowth)


def test_trivial():
    cosmo1 = FlatwCDM(H0 = 70.0, w0 = -1.0, Om0 = 0.3, Ob0 = 0.05, Tcmb0=2.7)
    cosmo2 = FlatLambdaCDM(H0=70.0, Om0 = 0.3, Ob0 = 0.05, Tcmb0=2.7)

    t1 = Transfer(cosmo_model=cosmo1, z=1.0)
    t2 = Transfer(cosmo_model=cosmo2, z=1.0)

    assert np.isclose(t1.growth_factor, t2.growth_factor, 1e-2)

    t1.z = 5.0
    t2.z = 5.0

    assert np.isclose(t1.growth_factor, t2.growth_factor, 1e-2)


def test_divergence():
    cosmo1 = FlatwCDM(H0 = 70.0, w0=-0.5, Om0 = 0.3, Ob0 = 0.05, Tcmb0=2.7)
    cosmo2 = FlatLambdaCDM(H0= 70.0, Om0 = 0.3, Ob0 = 0.05, Tcmb0=2.7)

    t1 = Transfer(cosmo_model=cosmo1, z=4.0)
    t2 = Transfer(cosmo_model=cosmo2, z=4.0)

    assert not np.isclose(t1.growth_factor, t2.growth_factor, 1e-1)
