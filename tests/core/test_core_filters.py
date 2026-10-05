"""Tests of hmf.core.filters: the window functions, against analytic results."""

import math

import mpmath
import numpy as np
import pytest

from hmf.core.filters import Filter, SharpK, SmoothK, TopHat

mpmath.mp.dps = 50


def _tophat_mp(x):
    """W(x) of the top-hat at 50 digits."""
    x = mpmath.mpf(x)
    return 3 * (mpmath.sin(x) - x * mpmath.cos(x)) / x**3


def _log_derivative(f, x, n):
    """The n-th derivative of f(e^t) in t, at t = ln x, at 50 digits."""
    return mpmath.diff(lambda t: f(mpmath.exp(t)), mpmath.log(mpmath.mpf(x)), n)


X_TEST = np.concatenate([np.logspace(-6, 1, 57), [0.0999999, 0.1, 0.1000001, 3.0, 30.0]])


def test_lookup_by_alias():
    assert Filter.get("TopHat") is TopHat
    assert Filter.get("SharpK") is SharpK
    assert Filter.get("SmoothK") is SmoothK


@pytest.mark.parametrize("flt", [TopHat(), SharpK(), SmoothK(), SmoothK(beta=2.0)])
def test_window_at_zero_is_one(flt):
    assert flt.window(np.array([0.0, 1e-12]))[0] == 1.0
    assert flt.window(np.array([1e-12]))[0] == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize("order", [0, 1, 2])
def test_tophat_against_50_digits(order):
    """Both branches (series below x = 0.1, closed form above) to 1e-9, for W, W' and W''."""
    got = TopHat().window_derivatives(X_TEST, order=2)[order]
    want = np.array([float(_log_derivative(_tophat_mp, x, order)) for x in X_TEST])
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=0)


def test_tophat_small_kr_limit():
    """dW/dln(kR) -> -(kR)^2 / 5 and d2W/dln(kR)^2 -> -2 (kR)^2 / 5 as kR -> 0."""
    x = np.logspace(-6, -3, 10)
    _, dw, d2w = TopHat().window_derivatives(x)
    np.testing.assert_allclose(dw / (-(x**2) / 5), 1, rtol=1e-6)
    np.testing.assert_allclose(d2w / (-2 * x**2 / 5), 1, rtol=1e-6)


@pytest.mark.parametrize("beta", [2.0, 4.8, 12.0])
@pytest.mark.parametrize("order", [1, 2])
def test_smoothk_derivatives_against_50_digits(beta, order):
    def window(x):
        return 1 / (1 + x**beta)

    x = np.logspace(-3, 2, 23)
    got = SmoothK(beta=beta).window_derivatives(x)[order]
    want = np.array([float(_log_derivative(window, xx, order)) for xx in x])
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-300)


def test_smoothk_tends_to_sharpk():
    """As beta -> infinity the smooth-k window becomes the sharp-k window."""
    x = np.array([0.1, 0.5, 0.9, 0.99, 1.0, 1.01, 1.1, 2.0, 10.0])
    np.testing.assert_allclose(SmoothK(beta=5000.0).window(x), SharpK().window(x), atol=1e-20)


def test_sharpk_derivative_is_a_delta():
    with pytest.raises(ValueError, match="Dirac delta"):
        SharpK().dwindow_dlnx(np.array([0.5]))
    assert SharpK().window(np.array([0.5, 1.0, 1.5])).tolist() == [1.0, 0.5, 0.0]


def test_mass_assignment():
    assert TopHat().mass_assignment == 1.0
    assert SharpK().mass_assignment == 2.5
    assert SmoothK().mass_assignment == 3.3
    assert SharpK(c=2.7).mass_assignment == 2.7


@pytest.mark.parametrize("bad", [{"c": 0.0}, {"c": -1.0}, {"c": math.inf}])
def test_parameters_validated(bad):
    with pytest.raises(ValueError, match="must be > 0"):
        SharpK(**bad)
    with pytest.raises(ValueError, match="must be > 0"):
        SmoothK(**bad)


@pytest.mark.parametrize("flt", [TopHat(), SmoothK(), SmoothK(beta=2.0)])
def test_tail_bounds_bound_the_window(flt):
    """The truncation estimator's bounds hold: |W^2| and |W W'| below smooth + oscillating."""

    def total(envelope, x):
        return sum(c * x**p for c, p in envelope) if envelope else 0 * x

    tb = flt.tail_bounds()
    x = np.logspace(np.log10(2.0), 4, 20001)
    w, dw = flt.window_derivatives(x, order=1)
    slack = 1 + 1e-12  # the SmoothK bounds are tight to rounding at large x
    assert np.all(w**2 <= slack * (total(tb.high_w2, x) + total(tb.high_w2_oscillating, x)))
    bound = total(tb.high_wdw, x) + total(tb.high_wdw_oscillating, x)
    assert np.all(np.abs(w * dw) <= slack * bound)
    x = np.logspace(-8, -1, 100)
    w, dw = flt.window_derivatives(x, order=1)
    assert np.all(w**2 <= slack * total(tb.low_w2, x))
    assert np.all(np.abs(w * dw) <= slack * total(tb.low_wdw, x))


def test_tophat_tail_decomposition_is_exact():
    """The top-hat bounds come from an exact split of W^2 and W W' into mean + oscillation."""
    x = np.logspace(0.5, 3, 1000)
    w, dw = TopHat().window_derivatives(x, order=1)
    c, s = np.cos(2 * x), np.sin(2 * x)
    w2 = 4.5 / x**6 * ((1 + x**2) + (x**2 - 1) * c - 2 * x * s)
    wdw = -9 * (x**2 + 1.5) / x**6 + 4.5 / x**6 * ((3 - 4 * x**2) * c + (6 * x - x**3) * s)
    # Relative to the envelope of each (the functions themselves cross zero).
    assert np.all(np.abs(w**2 - w2) <= 1e-12 * 9 / x**4)
    assert np.all(np.abs(w * dw - wdw) <= 1e-12 * 9 / x**3)
