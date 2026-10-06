"""Tests of hmf.core._kernels.interpolation: exactness on polynomials, and inversion."""

import numpy as np
import pytest

from hmf.core._kernels.interpolation import (
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
