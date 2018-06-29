from hmf.cosmology.cosmo import Cosmology, WMAP7
import numpy as np


def eq(actual, expected):
    return abs(actual - expected) < 0.000001

def test_string_cosmo():
    c = Cosmology(cosmo_model="WMAP7")
    assert c.cosmo.Ob0 > 0


class TestUpdate():
    def setup_method(self, test_method):
        self.c = Cosmology(cosmo_model="Planck13")

    def test_cosmo_model(self):
        self.c.update(cosmo_model=WMAP7)

        assert self.c.cosmo.Om0 == 0.272
        assert np.isclose(self.c.mean_density0, 75489962610.27452, atol=1e-3) # this number *can* change when updated constants are used.

    def test_cosmo_params(self):
        self.c.update(cosmo_params={"H0":0.6})
        assert self.c.cosmo.H0.value == 0.6
        self.c.update(cosmo_params={"Om0":0.2})
        assert self.c.cosmo.Om0 == 0.2
        assert self.c.cosmo.H0.value == 0.6
        assert self.c.cosmo_params == {"Om0":0.2, "H0":0.6}
