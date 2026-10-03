"""
Tests of the filters module.

Analytic functions for this test are defined in "analytic_filter.ipynb"
in the development/ directory.
"""

import warnings

import numpy as np
import pytest
from numpy import cos, pi, sin
from scipy.special import beta as beta_fn

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


def _smoothk_sigma2_powerlaw(beta, n, R, order=0):
    r"""Closed-form :math:`\sigma_j^2(R)` for the smooth-k filter with :math:`P(k)=k^n`.

    With :math:`x=kR` and :math:`a = n + 3 + 2j`,

    .. math:: \sigma_j^2 = \frac{1}{2\pi^2 R^a}\int_0^\infty
              \frac{x^{a-1}}{(1+x^\beta)^2}dx
              = \frac{1}{2\pi^2 R^a}\frac{1}{\beta}B(a/\beta, 2-a/\beta),

    obtained by substituting :math:`t=x^\beta`. Valid for :math:`0 < a < 2\beta`.
    """
    a = n + 3 + 2 * order
    assert 0 < a < 2 * beta
    return beta_fn(a / beta, 2 - a / beta) / beta / (2 * pi**2 * R**a)


class TestSmoothK:
    @pytest.fixture(scope="class")
    def cls(self):
        # k_max must be large: the sigma_1 integrand only falls as (kR)^(-2.6) per ln k.
        k = np.logspace(-6, 4, 10000)
        pk = k**2
        return filters.SmoothK(k, pk)

    def test_defaults(self, cls):
        assert cls.params["beta"] == 4.8
        assert cls.params["c"] == 3.3

    def test_k_space(self, cls):
        kr = np.array([0.0, 1e-3, 0.5, 1.0, 2.0, 10.0])
        assert np.allclose(cls.k_space(kr), 1 / (1 + kr**4.8), rtol=1e-12, atol=0)
        assert cls.k_space(np.array([1.0]))[0] == 0.5

    def test_k_space_large_beta_no_overflow(self):
        f = filters.SmoothK(np.array([1.0]), np.array([1.0]), beta=1000)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            w = f.k_space(np.array([0.0, 0.5, 1e3]))
            dw = f.dw_dlnkr(np.array([0.0, 0.5, 1e3]))
        assert np.allclose(w, [1.0, 1.0, 0.0])
        assert np.all(np.isfinite(dw))

    def test_dwdlnkr(self, cls):
        x = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        h = 1e-4
        fd = (cls.k_space(x * np.exp(h)) - cls.k_space(x * np.exp(-h))) / (2 * h)
        # Central difference: O(h^2) truncation plus roundoff where W ~ 1.
        assert np.allclose(cls.dw_dlnkr(x), fd, rtol=1e-6, atol=0)
        b = 4.8
        assert np.allclose(cls.dw_dlnkr(x), -b * x**b / (1 + x**b) ** 2, rtol=1e-12)

    @pytest.mark.parametrize("R", [1.0, 3.0])
    def test_sigma(self, cls, R):
        true = _smoothk_sigma2_powerlaw(4.8, 2, R)
        assert np.isclose(cls.sigma(R)[0] ** 2, true, rtol=1e-6, atol=0)

    @pytest.mark.parametrize("R", [1.0, 3.0])
    def test_sigma1(self, cls, R):
        true = _smoothk_sigma2_powerlaw(4.8, 2, R, order=1)
        assert np.isclose(cls.sigma(R, 1)[0] ** 2, true, rtol=1e-6, atol=0)

    def test_sigma_r_scaling(self, cls):
        # For P ~ k^n, sigma^2 ~ R^-(n+3) for any filter shape.
        s1, s3 = cls.sigma(np.array([1.0, 3.0])) ** 2
        assert np.isclose(s1 / s3, 3.0**5, rtol=1e-6)

    @pytest.mark.parametrize("R", [1.0, 3.0])
    def test_dlnssdlnr(self, cls, R):
        # For P ~ k^n, d ln sigma^2 / d ln R = -(n+3) exactly.
        assert np.isclose(cls.dlnss_dlnr(R)[0], -5.0, rtol=1e-6, atol=0)

    def test_mass_radius_roundtrip(self, cls):
        rho = 2.5
        m = np.array([1.0e10, 1.0e12, 1.0e14])
        r = cls.mass_to_radius(m, rho)
        assert np.allclose(cls.radius_to_mass(r, rho), m, rtol=1e-12)
        assert np.allclose(r, (3 * m / (4 * pi * rho)) ** (1 / 3) / 3.3, rtol=1e-12)

    def test_dlnr_dlnm(self, cls):
        r = np.array([0.5, 1.0])
        assert np.allclose(cls.dlnss_dlnm(r), cls.dlnss_dlnr(r) / 3.0)

    def test_real_space_bad_beta(self):
        f = filters.SmoothK(np.array([1.0]), np.array([1.0]), beta=1.0)
        with pytest.raises(ValueError, match="beta > 1"):
            f.real_space(1.0, np.array([1.0]))


class TestSmoothKSharpKLimit:
    """As beta -> infinity, SmoothK tends to SharpK (Leo et al. 2018, fig. 2)."""

    @pytest.fixture(scope="class")
    def filts(self):
        k = np.logspace(-6, 2, 10000)
        pk = k**2
        return (
            filters.SmoothK(k, pk, beta=100, c=2.5),
            filters.SharpK(k, pk, c=2.5),
        )

    def test_sigma_m(self, filts):
        smooth, sharp = filts
        rho = 1.0
        m = sharp.radius_to_mass(np.logspace(0, 1.5, 7), rho)
        s_smooth = smooth.sigma(smooth.mass_to_radius(m, rho))
        s_sharp = sharp.sigma(sharp.mass_to_radius(m, rho))
        # With P ~ k^n the ratio is scale-free. To leading order in 1/beta,
        # sigma_smooth/sigma_sharp - 1 = -(n+3)/(2 beta), i.e. -2.5% for n=2,
        # beta=100. The measured offset is -2.33% at every mass, so 3% is the
        # tolerance. (n=2 is a harsh case: realistic spectra have n+3 < 5.)
        assert np.allclose(s_smooth, s_sharp, rtol=0.03, atol=0)
        assert np.all(s_smooth < s_sharp)

    def test_real_space(self, filts):
        smooth, sharp = filts
        r = np.array([0.5, 1.0, 2.0, 5.0])
        # Measured agreement is ~0.15%.
        assert np.allclose(smooth.real_space(1.0, r), sharp.real_space(1.0, r), rtol=5e-3)


def test_smoothk_in_mass_function():
    from hmf import MassFunction

    mf = MassFunction(filter_model="SmoothK", transfer_params={"extrapolate_with_eh": True})
    assert isinstance(mf.filter, filters.SmoothK)
    assert np.all(np.isfinite(mf.dndm))
    assert np.all(mf.dndm > 0)
