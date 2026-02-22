"""
Tests of the filters module.

Analytic functions for this test are defined in "analytic_filter.ipynb"
in the development/ directory.
"""

import warnings

import numpy as np
import pytest
from numpy import cos, pi, sin

from hmf.density_field import filters

# Need to do the following to catch repeated warnings.
warnings.simplefilter("always", UserWarning)


class TestTopHat:
    @pytest.fixture(scope="class")
    def cls(self):
        k = np.logspace(-6, 0, 10000)
        pk = k**2
        return filters.TopHat(k, pk)

    def test_sigma(self, cls):
        R = 1.0
        true = (
            9 * R**2 * sin(R) ** 2 / 2
            + 9 * R**2 * cos(R) ** 2 / 2
            + 9 * R * sin(R) * cos(R) / 2
            - 9 * sin(R) ** 2
        ) / (2 * pi**2 * R**6)

        print(true, cls.sigma(R) ** 2)
        assert np.isclose(cls.sigma(R)[0] ** 2, true)

    def test_sigma1(self, cls):
        R = 1.0
        true = (
            3 * R**2 * sin(R) ** 2 / 2
            + 3 * R**2 * cos(R) ** 2 / 2
            + 9 * R * sin(R) * cos(R) / 2
            - 9 * sin(R) ** 2 / 4
            + 45 * cos(R) ** 2 / 4
            - 45 * sin(R) * cos(R) / (4 * R)
        ) / (2 * pi**2 * R**6)

        print(true, cls.sigma(R, 1) ** 2)
        assert np.isclose(cls.sigma(R, 1)[0] ** 2, true)

    def test_dwdlnkr(self, cls):
        x = 1.0
        true = x * (3 * sin(x) / x**2 - 3 * (-3 * x * cos(x) + 3 * sin(x)) / x**4)
        assert np.isclose(cls.dw_dlnkr(x), true)

    def test_dlnssdlnr(self, cls):
        R = 1.0
        true = (
            2
            * R**4
            * (
                -45 * sin(R) ** 2 / (4 * R**2)
                - 27 * cos(R) ** 2 / (4 * R**2)
                - 81 * sin(R) * cos(R) / (4 * R**3)
                + 27 * sin(R) ** 2 / R**4
            )
            / (
                9 * R**2 * sin(R) ** 2 / 2
                + 9 * R**2 * cos(R) ** 2 / 2
                + 9 * R * sin(R) * cos(R) / 2
                - 9 * sin(R) ** 2
            )
        )

        print(true, cls.dlnss_dlnr(R))
        assert np.isclose(cls.dlnss_dlnr(R), true)

    def test_real_space_edges(self, cls):
        R = 1.0
        r = np.array([0.5, 1.0, 1.5])

        assert np.array_equal(cls.real_space(R, r), np.array([1.0, 0.5, 0.0]))

    def test_k_space_small(self, cls):
        kr = np.array([1e-8, 1e-7, 1e-2])
        w = cls.k_space(kr)

        assert np.allclose(w[:2], 1.0)
        assert np.isfinite(w[2])

    def test_nu(self, cls):
        R = 1.0
        nu = cls.nu(R)

        assert np.isfinite(nu).all()

    def test_mass_radius_roundtrip(self, cls):
        rho = 2.0
        m = 1.0e12
        r = cls.mass_to_radius(m, rho)

        assert np.isclose(cls.radius_to_mass(r, rho), m)

    def test_dlnr_dlnm(self, cls):
        r = np.array([0.5, 1.0])

        assert np.allclose(cls.dlnr_dlnm(r), 1.0 / 3.0)
        assert np.allclose(cls.dlnss_dlnm(r), cls.dlnss_dlnr(r) / 3.0)


class TestSharpK:
    @pytest.fixture(scope="class")
    def cls(self):
        k = np.logspace(-6, 0, 10000)
        pk = k**2
        return filters.SharpK(k, pk)

    def test_sigma(self, cls):
        R = 1.0
        t = 2 + 2 + 1
        true = 1.0 / (2 * pi**2 * t * R**t)

        print(true, cls.sigma(R) ** 2)
        assert np.isclose(cls.sigma(R)[0] ** 2, true)

    def test_sigma1(self, cls):
        R = 1.0
        t = 4 + 2 + 1
        true = 1.0 / (2 * pi**2 * t * R**t)

        print(true, cls.sigma(R, 1) ** 2)
        assert np.isclose(cls.sigma(R, 1)[0] ** 2, true)

    def test_dlnssdlnr(self, cls):
        R = 1.0
        t = 2 + 2 + 1
        sigma2 = 1.0 / (2 * pi**2 * t * R**t)
        true = -1.0 / (2 * pi**2 * sigma2 * R ** (3 + 2))

        print(true, cls.dlnss_dlnr(R))
        assert np.isclose(cls.dlnss_dlnr(R), true)

    def test_sigma_R3(self, cls):
        R = 3.0
        t = 2 + 2 + 1
        true = 1.0 / (2 * pi**2 * t * R**t)

        print(true, cls.sigma(R) ** 2)
        assert np.isclose(cls.sigma(R)[0] ** 2, true)

    def test_sigma1_R3(self, cls):
        R = 3.0
        t = 4 + 2 + 1
        true = 1.0 / (2 * pi**2 * t * R**t)

        print(true, cls.sigma(R, 1) ** 2)
        assert np.isclose(cls.sigma(R, 1)[0] ** 2, true)

    def test_dlnssdlnr_R3(self, cls):
        R = 3.0
        t = 2 + 2 + 1
        sigma2 = 1.0 / (2 * pi**2 * t * R**t)
        true = -1.0 / (2 * pi**2 * sigma2 * R ** (3 + 2))

        print(true, cls.dlnss_dlnr(R))
        assert np.isclose(cls.dlnss_dlnr(R), true)

    def test_sigma_Rhalf(self, cls):
        thisr = 1.0 / cls.k.max()
        t = 2 + 2 + 1
        true = 1.0 / (2 * pi**2 * t * thisr**t)

        # should also raise a warning
        R = 0.5
        with pytest.warns(UserWarning, match=""):
            s2 = cls.sigma(R)[0] ** 2
        assert np.isclose(s2, true)

    def test_sigma1_Rhalf(self, cls):
        thisr = 1.0 / cls.k.max()

        t = 4 + 2 + 1
        true = 1.0 / (2 * pi**2 * t * thisr**t)

        # should also raise a warning
        R = 0.5
        with pytest.warns(UserWarning, match=""):
            s2 = cls.sigma(R, 1)[0] ** 2
        assert np.isclose(s2, true)

    def test_dlnssdlnr_Rhalf(self, cls):
        R = 3.0
        t = 2 + 2 + 1
        sigma2 = 1.0 / (2 * pi**2 * t * R**t)
        true = -1.0 / (2 * pi**2 * sigma2 * R ** (3 + 2))

        print(true, cls.dlnss_dlnr(R))
        assert np.isclose(cls.dlnss_dlnr(R), true)

    def test_k_space_edges(self, cls):
        kr = np.array([0.5, 1.0, 1.5])

        assert np.array_equal(cls.k_space(kr), np.array([1.0, 0.5, 0.0]))

    def test_real_space_and_dw(self, cls):
        R = 2.0
        r = np.array([1.0, 2.0])

        assert np.isfinite(cls.real_space(R, r)).all()
        assert np.array_equal(cls.dw_dlnkr(np.array([1.0, 2.0])), np.array([1.0, 0.0]))

    def test_mass_radius_roundtrip(self, cls):
        rho = 2.5
        m = 1.0e12
        r = cls.mass_to_radius(m, rho)

        assert np.isclose(cls.radius_to_mass(r, rho), m)


class TestGaussian:
    @pytest.fixture(scope="class")
    def cls(self):
        k = np.logspace(-6, 1, 151)
        pk = k**2
        return filters.Gaussian(k, pk)

    def test_sigma(self, cls):
        R = 10.0
        true = 3.0 / (16 * pi ** (3.0 / 2.0) * R**5)

        print(true, cls.sigma(R) ** 2)
        assert np.isclose(cls.sigma(R)[0] ** 2, true)

    def test_sigma1(self, cls):
        R = 10.0
        true = 15 / (32 * pi ** (3.0 / 2.0) * R**7)

        print(true, cls.sigma(R, 1) ** 2)
        assert np.isclose(cls.sigma(R, 1)[0] ** 2, true)

    def test_dlnssdlnr(self, cls):
        R = 10.0
        true = -5

        print(true, cls.dlnss_dlnr(R))
        assert np.isclose(cls.dlnss_dlnr(R), true)

    def test_real_space_and_k_space(self, cls):
        R = 2.0
        r = np.array([1.0, 2.0])
        kr = np.array([0.0, 1.0])

        assert np.isfinite(cls.real_space(R, r)).all()
        assert np.allclose(cls.k_space(kr), np.exp(-(kr**2) / 2.0))

    def test_mass_radius_roundtrip(self, cls):
        rho = 1.5
        m = 3.0e11
        r = cls.mass_to_radius(m, rho)

        assert np.isclose(cls.radius_to_mass(r, rho), m)


class TestSharpKEllipsoid:
    @pytest.fixture(scope="class")
    def cls(self):
        k = np.logspace(-3, 1, 200)
        pk = np.ones_like(k)
        return filters.SharpKEllipsoid(k, pk)

    def test_shape_helpers(self, cls):
        g = 0.5
        v = 1.2
        xm = cls.xm(g, v)
        em = cls.em(xm)
        pm = cls.pm(xm)

        assert np.isfinite(xm)
        assert np.isfinite(em)
        assert np.isfinite(pm)
        assert np.isfinite(cls.a3a1(em, pm))
        assert np.isfinite(cls.a3a2(em, pm))

    def test_gamma_xi_a3(self, cls):
        r = np.array([0.5])
        g = cls.gamma(r)
        xm = cls.xm(g, cls.nu(r))
        em = cls.em(xm)
        pm = cls.pm(xm)

        assert np.isfinite(cls.xi(pm, em)).all()
        assert np.isfinite(cls.a3(r)).all()

    def test_r_a3_and_derivatives(self, cls):
        spline = cls.r_a3(0.2, 1.0)
        r = np.array([0.4, 0.5, 0.6, 0.7])

        a3 = cls.a3(r)
        assert np.isfinite(spline(a3)).all()
        assert np.isfinite(cls.dlnss_dlnr(r)).all()
        assert np.isfinite(cls.dlnr_dlnm(r)).all()
