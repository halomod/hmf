"""Tests of hmf.core._kernels.interpolation: exactness on polynomials, inversion, splines."""

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from hmf.core._kernels.interpolation import (
    FrozenSpline,
    extrapolate_power_law,
    hermite_cubic,
    hermite_quintic,
    invert_hermite,
    lagrange4,
)

U = np.linspace(0, 1, 101)
X0, H = 0.7, 0.3


def _poly(coeffs):
    """A polynomial in x, and its first two derivatives."""
    p = np.polynomial.Polynomial(coeffs)
    return p, p.deriv(1), p.deriv(2)


def test_hermite_cubic_is_exact_for_cubics():
    p, dp, _ = _poly([0.3, -1.2, 2.5, -0.7])
    x1 = X0 + H
    got = hermite_cubic(U, H, p(X0), p(x1), dp(X0), dp(x1))
    np.testing.assert_allclose(got, p(X0 + U * H), rtol=0, atol=1e-14)


def test_hermite_quintic_is_exact_for_quintics():
    p, dp, d2p = _poly([0.3, -1.2, 2.5, -0.7, 1.1, -0.4])
    x1 = X0 + H
    got = hermite_quintic(U, H, p(X0), p(x1), dp(X0), dp(x1), d2p(X0), d2p(x1))
    np.testing.assert_allclose(got, p(X0 + U * H), rtol=0, atol=1e-14)


def test_hermite_quintic_is_not_exact_beyond_its_degree():
    """A sanity check that the exactness tests can fail: x^6 is not reproduced."""
    p, dp, d2p = _poly([0, 0, 0, 0, 0, 0, 1.0])
    x1 = X0 + H
    got = hermite_quintic(U, H, p(X0), p(x1), dp(X0), dp(x1), d2p(X0), d2p(x1))
    assert np.max(np.abs(got - p(X0 + U * H))) > 1e-8


def test_lagrange4_is_exact_for_cubics():
    p, _, _ = _poly([0.3, -1.2, 2.5, -0.7])
    nodes = X0 + H * np.array([-1, 0, 1, 2])
    got = lagrange4(U, *p(nodes))
    np.testing.assert_allclose(got, p(X0 + U * H), rtol=0, atol=1e-14)


def test_interpolants_hit_the_nodes():
    y0, y1, d0, d1, c0, c1 = 1.5, -0.5, -2.0, -1.0, 0.3, -0.2
    ends = np.array([0.0, 1.0])
    np.testing.assert_array_equal(hermite_cubic(ends, H, y0, y1, d0, d1), [y0, y1])
    np.testing.assert_array_equal(hermite_quintic(ends, H, y0, y1, d0, d1, c0, c1), [y0, y1])
    np.testing.assert_array_equal(lagrange4(ends, 9.0, y0, y1, 7.0), [y0, y1])


@pytest.mark.parametrize("quintic", [False, True])
def test_invert_hermite_round_trips(quintic):
    """A decreasing interpolant (here ln of a power law) inverts to machine precision."""
    y0, y1, d0, d1 = 0.0, -1.0, -1 / H, -1 / H
    cs = (0.0, 0.0) if quintic else (None, None)
    u = np.linspace(0, 1, 37)
    if quintic:
        target = hermite_quintic(u, H, y0, y1, d0, d1, *cs)
    else:
        target = hermite_cubic(u, H, y0, y1, d0, d1)
    got = invert_hermite(target, H, np.full_like(u, y0), np.full_like(u, y1), d0, d1, *cs)
    np.testing.assert_allclose(got, u, rtol=0, atol=1e-14)


def test_batch_size_independence():
    rng = np.random.default_rng(3)
    y0, y1, d0, d1, c0, c1 = rng.normal(size=(6, 50))
    u = rng.random(50)
    whole = hermite_quintic(u, H, y0, y1, d0, d1, c0, c1)
    single = [hermite_quintic(u[i], H, y0[i], y1[i], d0[i], d1[i], c0[i], c1[i]) for i in range(50)]
    assert np.array_equal(whole, np.array(single))


# ---------------------------------------------------------------------------------
# FrozenSpline and power-law extrapolation
# ---------------------------------------------------------------------------------

#: An uneven table, and points inside and beyond it.
X_TAB = np.cumsum(np.linspace(0.1, 0.4, 30)) - 3.0
Y_TAB = np.sin(X_TAB) + 0.1 * X_TAB**2
X_EVAL = np.linspace(X_TAB[0] - 1, X_TAB[-1] + 1, 1001)


@pytest.mark.parametrize(
    "bc_type",
    ["not-a-knot", "natural", ((1, 0.0), (1, 0.3))],
    ids=["not-a-knot", "natural", "clamped"],
)
@pytest.mark.parametrize("nu", [0, 1, 2])
def test_frozen_spline_is_bit_identical_to_cubic_spline(bc_type, nu):
    """FrozenSpline gives exactly a CubicSpline's values and derivatives."""
    direct = CubicSpline(X_TAB, Y_TAB, bc_type=bc_type)
    frozen = FrozenSpline.fit(X_TAB, Y_TAB, bc_type=bc_type)
    np.testing.assert_array_equal(frozen.c, direct.c)
    np.testing.assert_array_equal(frozen(X_EVAL, nu), direct(X_EVAL, nu))
    np.testing.assert_array_equal(frozen(X_TAB[3], nu), direct(X_TAB[3], nu))


def test_frozen_spline_without_extrapolation_is_nan_outside():
    frozen = FrozenSpline.fit(X_TAB, Y_TAB, extrapolate=False)
    direct = CubicSpline(X_TAB, Y_TAB, extrapolate=False)
    np.testing.assert_array_equal(frozen(X_EVAL), direct(X_EVAL))
    assert np.isnan(frozen(X_TAB[0] - 0.1))
    assert np.isfinite(frozen(X_TAB[0]))


def test_frozen_spline_is_exact_for_cubics():
    """A not-a-knot cubic spline reproduces a cubic polynomial (an analytic limit)."""
    p = np.polynomial.Polynomial([0.3, -1.2, 2.5, -0.7])
    frozen = FrozenSpline.fit(X_TAB, p(X_TAB))
    np.testing.assert_allclose(frozen(X_EVAL), p(X_EVAL), rtol=0, atol=1e-11)
    np.testing.assert_allclose(frozen(X_EVAL, 1), p.deriv()(X_EVAL), rtol=0, atol=1e-10)


def test_frozen_spline_is_immutable_and_does_not_alias_its_input():
    x, y = X_TAB.copy(), Y_TAB.copy()
    frozen = FrozenSpline.fit(x, y)
    before = frozen(X_EVAL)
    x[:] = 0.0
    y[:] = 0.0
    np.testing.assert_array_equal(frozen(X_EVAL), before)
    assert not frozen.x.flags.writeable
    assert not frozen.c.flags.writeable


def test_extrapolate_power_law_is_bit_identical_to_the_direct_computation():
    """The kernel matches the spline-plus-straight-lines computation, bit for bit."""
    direct = CubicSpline(X_TAB, Y_TAB)
    lo, hi = X_TAB[0], X_TAB[-1]
    expected = direct(np.clip(X_EVAL, lo, hi))
    expected = np.where(lo > X_EVAL, expected + direct(lo, 1) * (X_EVAL - lo), expected)
    expected = np.where(hi < X_EVAL, expected + direct(hi, 1) * (X_EVAL - hi), expected)
    got = extrapolate_power_law(FrozenSpline.fit(X_TAB, Y_TAB), X_EVAL)
    np.testing.assert_array_equal(got, expected)


def test_extrapolate_power_law_continues_a_power_law_exactly():
    """Ln y linear in ln x (a power law) is continued exactly beyond the table."""
    ln_x = np.linspace(-2.0, 3.0, 40)
    spline = FrozenSpline.fit(ln_x, -1.5 * ln_x + 0.7)
    ln_x_eval = np.linspace(-12.0, 15.0, 301)
    got = extrapolate_power_law(spline, ln_x_eval)
    np.testing.assert_allclose(got, -1.5 * ln_x_eval + 0.7, rtol=0, atol=1e-12)


def test_extrapolate_power_law_is_batch_size_independent():
    spline = FrozenSpline.fit(X_TAB, Y_TAB)
    whole = extrapolate_power_law(spline, X_EVAL)
    parts = np.concatenate(
        [extrapolate_power_law(spline, X_EVAL[i : i + 7]) for i in range(0, X_EVAL.size, 7)]
    )
    np.testing.assert_array_equal(whole, parts)
