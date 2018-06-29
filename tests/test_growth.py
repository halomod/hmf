import numpy as np
from hmf.cosmology import growth_factor as gf
from hmf.cosmology.cosmo import Planck13


class TestSimilarity(object):
    """
    Simply test similarity between standard GrowthFunction and others (at this point, just GenMFGrowth)
    that should be similar.
    """

    def setup_method(self, test_method):
        self.gf = gf.GrowthFactor(Planck13)
        self.genf = gf.GenMFGrowth(Planck13,zmax=10.0)

    def test_gf(self):
        for z in np.arange(0,8,0.5):
            print(self.gf.growth_factor(z),self.genf.growth_factor(z))
            assert np.isclose(self.gf.growth_factor(z),self.genf.growth_factor(z),rtol=1e-2 + z/500.0)

    def test_gr(self):
        for z in np.arange(0,8,0.5):
            self.gf.growth_rate(z),self.genf.growth_rate(z)
            assert np.isclose(self.gf.growth_rate(z),self.genf.growth_rate(z),rtol=1e-2+ z/100.0)

    def test_gfunc(self):
        gf_func = self.gf.growth_factor_fn(0.0)
        genf_func = self.genf.growth_factor_fn(0.0)

        print(gf_func(np.linspace(0,5,10)),genf_func(np.linspace(0,5,10)))
        assert np.allclose(gf_func(np.linspace(0,5,10)),genf_func(np.linspace(0,5,10)),rtol=1e-2)

    def test_gr_func(self):
        gr_func = self.gf.growth_rate_fn(0.0)
        genf_func = self.genf.growth_rate_fn(0.0)

        print(gr_func(np.linspace(0,5,10)),genf_func(np.linspace(0,5,10)))
        assert np.allclose(gr_func(np.linspace(0,5,10)),genf_func(np.linspace(0,5,10)),rtol=1e-2)

    def test_inverse(self):
        gf_func = self.gf.growth_factor_fn(0.0,inverse=True)
        genf_func = self.genf.growth_factor_fn(0.0,inverse=True)

        gf = np.linspace(0.15,0.99,10)
        print(gf_func(gf),genf_func(gf))
        assert np.allclose(gf_func(gf),genf_func(gf),rtol=1e-1)