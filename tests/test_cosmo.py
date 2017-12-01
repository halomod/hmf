import inspect
import os

LOCATION = "/".join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split("/")[:-1])
from nose.tools import raises
import sys
sys.path.insert(0, LOCATION)
from hmf.cosmo import Cosmology, WMAP7
import numpy as np


def eq(actual, expected):
    return abs(actual - expected) < 0.000001

#
# def test_pycamb_dict():
#     bits = ["w_lam", "TCMB", "yhe", "reion__redshift", "Num_Nu_massless", "omegab",
#             "omegac", "H0", "omegav", "omegak", "omegan", "cs2_lam",
#             "scalar_index", "Num_Nu_massive"]
#     c = Cosmology(w=-1, t_cmb=2.74, y_he=0.24, z_reion=10, N_nu=3.04, omegab=0.2,
#                   omegac=0.25, h=0.7, force_flat=True, omegan=0.0, cs2_lam=-1,
#                   n=1, N_nu_massive=0)
#
#     for bit in bits:
#         assert bit in c.pycamb_dict


def test_string_cosmo():
    c = Cosmology(cosmo_model="WMAP7")
    assert c.cosmo.Ob0 > 0


class TestUpdate():
    def __init__(self):
        self.c = Cosmology(cosmo_model="Planck13")

    def test_cosmo_model(self):
        self.c.update(cosmo_model=WMAP7)

        assert self.c.cosmo.Om0 == 0.272
        print(self.c.mean_density0)
        assert np.isclose(self.c.mean_density0, 75489962610.27452, atol=1e-3) # this number *can* change when updated constants are used.

    def test_cosmo_params(self):
        self.c.update(cosmo_params={"H0":0.6})
        print((self.c.cosmo.H0.value))
        assert self.c.cosmo.H0.value == 0.6
        self.c.update(cosmo_params={"Om0":0.2})
        assert self.c.cosmo.Om0 == 0.2
        assert self.c.cosmo.H0.value == 0.6
        assert self.c.cosmo_params == {"Om0":0.2, "H0":0.6}
