"""Tests of hmf.core.profiles: density profiles for the mass conversion."""

import math
import pickle

import attrs
import numpy as np
import pytest

from hmf.core import _references as refs
from hmf.core.domain import DomainError
from hmf.core.mass_conversion import ProfileConversion
from hmf.core.mass_definitions import SOCritical, SOMean, SOVirial
from hmf.core.profiles import NFW, DensityProfile


@attrs.frozen(kw_only=True)
class _Hernquist(DensityProfile, abstract=True):
    """Hernquist (1990), f(x) = 1 / [x (1 + x)^3]: mu = x^2 / [2 (1 + x)^2], psi = 2 / (1 + x).

    A second profile through the default (power-law) start and the generic solve.
    """

    def _ln_mu_and_slope(self, x):
        return 2 * np.log(x) - 2 * np.log1p(x) - math.log(2), 2 / (1 + x)


@attrs.frozen(kw_only=True)
class _Isothermal(DensityProfile, abstract=True):
    """The singular isothermal sphere, f(x) = 1 / x^2: mu = x, psi = 1."""

    def _ln_mu_and_slope(self, x):
        return np.log(x), np.ones_like(x)


C = np.geomspace(1e-2, 1e3, 60)[:, None]
LN_Q = np.linspace(math.log(1e-3), math.log(1e3), 41)[None, :]


def test_registered_and_cited():
    assert DensityProfile.get("NFW") is NFW
    assert refs.NFW97 in NFW.references
    assert pickle.loads(pickle.dumps(NFW())) == NFW()
    assert ProfileConversion(profile=_Hernquist()).profile == _Hernquist()


def test_nfw_enclosed_mass():
    """mu(1) = ln 2 - 1/2, and mu -> x^2 / 2 at small x (the density ~ 1/x)."""
    assert NFW().mu(x=1.0) == pytest.approx(math.log(2) - 0.5, rel=1e-15)
    assert isinstance(NFW().mu(x=1.0), float)
    assert NFW().mu(x=1e-5) == pytest.approx(0.5e-10, rel=1e-4)
    # The closed form, mu = ln(1 + x) - x / (1 + x), and its slope
    # psi = x mu'(x) / mu = x^2 / [(1 + x)^2 mu].
    x = np.array([0.5, 2.0])
    mu = np.log1p(x) - x / (1 + x)
    ln_mu, psi = NFW().ln_mu_and_slope_kernel(x=x)
    np.testing.assert_allclose(ln_mu, np.log(mu), rtol=1e-14)
    np.testing.assert_allclose(psi, x**2 / ((1 + x) ** 2 * mu), rtol=1e-14)
    with pytest.raises(DomainError, match="x in"):
        NFW().mu(x=-1.0)


def test_nfw_solve_reaches_round_off_from_hu_kravtsov():
    """From Hu & Kravtsov (2003, eq. C11), NFW needs fewer Newton steps than the default."""
    assert NFW.newton_steps < DensityProfile.newton_steps
    _, c_new, _, _ = NFW().convert_concentration_kernel(c=C, slope=0.0, ln_q=LN_Q)
    ln_mu_new, _ = NFW().ln_mu_and_slope_kernel(x=c_new)
    ln_mu_old, _ = NFW().ln_mu_and_slope_kernel(x=np.broadcast_to(C, c_new.shape))
    residual = (ln_mu_new - 3 * np.log(c_new)) - (ln_mu_old - 3 * np.log(C)) - LN_Q
    assert np.abs(residual).max() < 2e-13


def test_kernels_take_keyword_arguments_only():
    with pytest.raises(TypeError, match="positional"):
        NFW().mu(1.0)
    with pytest.raises(TypeError, match="positional"):
        NFW().ln_mu_and_slope_kernel(np.array([1.0]))
    with pytest.raises(TypeError, match="positional"):
        NFW().convert_concentration_kernel(5.0, slope=0.0, ln_q=0.5)


def test_isothermal_sphere_is_analytic():
    """F = x^-2, so c' = c q^(-1/2) and M'/M = q^(-1/2), for any c (dM'/dM = that)."""
    ln_ratio, c_new, dln_m, dln_c = _Isothermal().convert_concentration_kernel(
        c=C, slope=-0.1, ln_q=LN_Q
    )
    np.testing.assert_allclose(c_new, C * np.exp(-LN_Q / 2), rtol=1e-14)
    np.testing.assert_allclose(ln_ratio, np.broadcast_to(-LN_Q / 2, ln_ratio.shape), atol=1e-13)
    # psi = 1 everywhere: d ln M' / d ln M = 1 + s (1 - 1) = 1, and c' follows c.
    np.testing.assert_allclose(dln_m, 1.0, rtol=1e-15)
    np.testing.assert_allclose(dln_c, -0.1, rtol=1e-15)


def test_hernquist_solve_reaches_round_off_from_the_default_start():
    ln_mu_and_slope = _Hernquist()._ln_mu_and_slope
    _, c_new, _, _ = _Hernquist().convert_concentration_kernel(c=C, slope=0.0, ln_q=LN_Q)
    ln_mu_new, _ = ln_mu_and_slope(c_new)
    ln_mu_old, _ = ln_mu_and_slope(np.broadcast_to(C, c_new.shape))
    residual = (ln_mu_new - 3 * np.log(c_new)) - (ln_mu_old - 3 * np.log(C)) - LN_Q
    assert np.abs(residual).max() < 1e-12


def test_hernquist_conversion_round_trips_and_its_jacobian_is_the_derivative():
    conv = ProfileConversion(profile=_Hernquist())
    m = np.geomspace(1e10, 1e15, 11)
    om = 0.3
    there = conv.convert_kernel(m=m, z=0.5, omega_m_z=om, source=SOMean(), target=SOVirial())
    back = conv.convert_kernel(m=there.m, z=0.5, omega_m_z=om, source=SOVirial(), target=SOMean())
    np.testing.assert_allclose(back.m, m, rtol=1e-12)
    h = 1e-5
    hi = conv.convert_kernel(
        m=m * math.exp(h), z=0.5, omega_m_z=om, source=SOMean(), target=SOVirial()
    )
    lo = conv.convert_kernel(
        m=m * math.exp(-h), z=0.5, omega_m_z=om, source=SOMean(), target=SOVirial()
    )
    numeric = (np.log(hi.m) - np.log(lo.m)) / (2 * h) * there.m / m
    np.testing.assert_allclose(there.dm_dm, numeric, rtol=1e-8)


def test_a_more_concentrated_profile_shifts_masses_less():
    """Hernquist falls as x^-4 outside, NFW as x^-3: less mass between the two radii."""
    m = np.geomspace(1e11, 1e15, 5)
    args = {"z": 0.0, "omega_m_z": 0.3, "source": SOCritical(), "target": SOMean()}
    nfw = ProfileConversion().convert_kernel(m=m, **args).m / m
    hernquist = ProfileConversion(profile=_Hernquist()).convert_kernel(m=m, **args).m / m
    assert np.all((hernquist > 1) & (hernquist < nfw))


def test_inputs_are_checked():
    with pytest.raises(DomainError, match="c in"):
        NFW().convert_concentration_kernel(c=[5.0, 0.0], slope=0.0, ln_q=0.5)
    with pytest.raises(DomainError, match="slope must be finite"):
        NFW().convert_concentration_kernel(c=5.0, slope=np.nan, ln_q=0.5)
