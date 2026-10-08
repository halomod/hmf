"""Tests of hmf.core.mass_variance: the MassVariance stage and its mass lattice.

The physical tests compare against closed forms (power laws), known limits (small-R
expansion of a truncated spectrum) and physical bounds (monotonicity, signs). The
comparison against hmf v3 is a regression cross-check only.
"""

import math
import warnings

import astropy.units as u
import attrs
import mpmath
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM
from power_models import (
    RHO_CRIT0,
    AnalyticPower,
    EisensteinHuNoWiggle,
    PowerLaw,
    WDMTruncated,
    WithBAO,
)
from scipy import integrate
from scipy.special import gamma

from hmf.core.accuracy import KAccuracy, MassAccuracy
from hmf.core.domain import DomainError
from hmf.core.filters import SharpK, SmoothK, TopHat
from hmf.core.mass_variance import (
    INTERPOLATION_RTOL,
    RESOLUTION_RTOL,
    MassVariance,
    n_eff_kernel,
)
from hmf.core.power_source import TabulatedPower
from hmf.core.transfer import Transfer
from hmf.core.units import Mpc_h, Msun_h, UnitBoundaryError, h_Mpc, power_unit, rho_unit
from hmf.exceptions import HMFExtrapolationWarning

FILTERS = ["TopHat", "SharpK", "SmoothK"]
EH = AnalyticPower(EisensteinHuNoWiggle())
BAO = AnalyticPower(WithBAO())
WDM = AnalyticPower(WDMTruncated())
LN10 = math.log(10)

#: Where n(>M) matters for the accuracy targets.
LOG10_M_RANGE = (6.0, 15.5)


def _mv(power=EH, flt="TopHat", **kwargs):
    return MassVariance(power=power, filter=flt, **kwargs)


# ---------------------------------------------------------------------------------
# Analytic: power laws
# ---------------------------------------------------------------------------------


def _tophat_integral(n):
    """int_0^inf x^(2+n) W_TH(x)^2 dx: closed forms, or 30 digits by mpmath.quadosc."""
    closed = {-2.0: 3 * math.pi / 5, -1.0: 9 / 4, 0.0: 3 * math.pi / 2}
    if n in closed:
        return closed[n]
    with mpmath.workdps(30):
        f = lambda x: x ** (2 + n) * (3 * (mpmath.sin(x) - x * mpmath.cos(x)) / x**3) ** 2  # noqa: E731
        return float(mpmath.quad(f, [0, 1]) + mpmath.quadosc(f, [1, mpmath.inf], period=mpmath.pi))


def _power_law_reference(mv, m, n, integral):
    """Return sigma and dln(sigma)/dln(m) of P = k^n, for a window integral I(n).

    The grid starts at k_min, below which W = 1 to O((k_min R)^2), so the integral
    misses (k_min R)^(3+n) / (3+n) of I(n).
    """
    r = mv.radius_from_m(m).value
    k_min = mv._k_grid.k[0]
    s2 = (r ** -(3 + n) * integral - k_min ** (3 + n) / (3 + n)) / (2 * math.pi**2)
    d = -(3 + n) / 6 * r ** -(3 + n) * integral / (2 * math.pi**2 * s2)
    return np.sqrt(s2), d


def _smoothk_integral(n, beta):
    """int_0^inf x^(2+n) / (1 + x^beta)^2 dx = Gamma(a) Gamma(2 - a) / beta, a = (3+n)/beta."""
    a = (3 + n) / beta
    return gamma(a) * gamma(2 - a) / beta


@pytest.mark.parametrize("n", [-2.5, -2.0])
@pytest.mark.parametrize(
    ("k_accuracy", "rtol_d"),
    [(KAccuracy(), 2e-4), (KAccuracy.high(), 1e-5)],
    ids=["default", "high"],
)
def test_power_law_tophat_closed_form(n, k_accuracy, rtol_d):
    """sigma^2 = R^-(3+n) I(n) / 2 pi^2 and dln(sigma)/dln(m) = -(n+3)/6.

    The tolerance of the slope is that of the k grid: the top-hat's oscillations
    converge slowly for a power law (unlike a physical P ~ k^-3), and the error
    falls with dlnk.
    """
    mv = _mv(AnalyticPower(PowerLaw(n)), "TopHat", k_accuracy=k_accuracy)
    m = 10 ** (np.linspace(6, 15, 61) + 0.013) * Msun_h
    sigma, slope = _power_law_reference(mv, m, n, _tophat_integral(n))
    np.testing.assert_allclose(mv.sigma(m), sigma, rtol=1e-6)
    np.testing.assert_allclose(mv.dlnsigma_dlnm(m), slope, rtol=rtol_d)
    np.testing.assert_allclose(slope, -(n + 3) / 6, rtol=1e-3)


@pytest.mark.parametrize("n", [-2.9, -2.0, -1.0, 0.5, 2.0])
def test_n_eff_of_a_power_law_slope_is_its_index(n):
    """For P ~ k^n, sigma ~ m^-(n+3)/6 with any filter, which n_eff maps back to n."""
    slope = np.full(3, -(n + 3) / 6)
    np.testing.assert_allclose(n_eff_kernel(slope), n, rtol=0, atol=1e-15)
    assert n_eff_kernel(-0.5) == 0.0  # sigma ~ m^-1/2: white noise, n = 0


@pytest.mark.parametrize("n", [-1.0, 1.0])
@pytest.mark.parametrize("flt", ["SharpK", SmoothK(beta=4.8)])
def test_n_eff_of_a_power_law_spectrum_is_its_index(n, flt):
    """n_eff from MassVariance's slope recovers the index of a power-law spectrum.

    With n >= -1 the part of the integral below the grid's k_min, which bends sigma
    at the largest masses, is below round-off, so n_eff = n to round-off.
    """
    mv = _mv(AnalyticPower(PowerLaw(n)), flt)
    _, slope = mv.ln_sigma_and_slope_kernel(np.logspace(6, 14, 33))
    np.testing.assert_allclose(n_eff_kernel(slope), n, rtol=0, atol=1e-12)


def test_shallow_power_law_with_tophat_raises():
    """For P ~ k^0 the top-hat's integrals raise, rather than give a wrong number.

    dln(sigma)/dln(m) only converges conditionally, and the grid aliases it.
    """
    mv = _mv(AnalyticPower(PowerLaw(0.0)), "TopHat")
    with pytest.raises(DomainError, match=r"does not resolve|not finite"):
        mv.sigma(np.logspace(8, 15, 20) * Msun_h)
    mv = _mv(
        AnalyticPower(PowerLaw(0.0)), "TopHat", mass_accuracy=MassAccuracy(second_derivative=False)
    )
    with pytest.raises(DomainError, match="does not resolve"):
        mv.sigma(np.logspace(8, 15, 20) * Msun_h)
    assert RESOLUTION_RTOL == 1e-2


@pytest.mark.parametrize("n", [-2.5, -1.0, 1.0])
@pytest.mark.parametrize("beta", [4.8, 8.0])
def test_power_law_smoothk_closed_form(n, beta):
    """SmoothK converges for n < 2 beta - 3; the integral is a beta function."""
    mv = _mv(AnalyticPower(PowerLaw(n)), SmoothK(beta=beta))
    m = np.logspace(6, 15, 37) * Msun_h
    sigma, slope = _power_law_reference(mv, m, n, _smoothk_integral(n, beta))
    np.testing.assert_allclose(mv.sigma(m), sigma, rtol=1e-7)
    np.testing.assert_allclose(mv.dlnsigma_dlnm(m), slope, rtol=1e-6)
    np.testing.assert_allclose(slope, -(n + 3) / 6, rtol=1e-3)


@pytest.mark.parametrize("n", [-2.5, -1.0, 1.0, 3.0])
def test_power_law_sharpk_closed_form(n):
    """sigma^2 = (R^-(3+n) - k_min^(3+n)) / (2 pi^2 (3+n)), for the grid's k_min."""
    mv = _mv(AnalyticPower(PowerLaw(n)), "SharpK")
    m = np.logspace(2, 16, 57) * Msun_h
    r = mv.radius_from_m(m).value
    k_min = mv._k_grid.k[0]
    s2 = (r ** -(3 + n) - k_min ** (3 + n)) / (2 * math.pi**2 * (3 + n))
    np.testing.assert_allclose(mv.sigma(m), np.sqrt(s2), rtol=1e-12)
    dlnsigma = -(3 + n) / 6 * r ** -(3 + n) / (r ** -(3 + n) - k_min ** (3 + n))
    np.testing.assert_allclose(mv.dlnsigma_dlnm(m), dlnsigma, rtol=1e-10)
    np.testing.assert_allclose(mv.dlnsigma_dlnm(m), -(n + 3) / 6, rtol=1e-3)


def test_sharpk_mass_assignment():
    """R = (3 m / 4 pi rho)^(1/3) / c, so sigma(m; c) = sigma(m / c^3; c = 1)."""
    p = AnalyticPower(PowerLaw(-1.0))
    m = np.logspace(9, 14, 11) * Msun_h
    a = _mv(p, SharpK(c=2.5)).sigma(m)
    b = _mv(p, SharpK(c=1.0)).sigma(m / 2.5**3)
    np.testing.assert_allclose(a, b, rtol=1e-12)


# ---------------------------------------------------------------------------------
# Analytic: a truncated spectrum at small R (the limit behind hmf 3.7.2's fix)
# ---------------------------------------------------------------------------------


def test_truncated_spectrum_small_r_limit():
    """Check the small-R limit of a truncated spectrum.

    For P truncated at k_c and k_c R << 1, W' = -(kR)^2/5 gives
    dln(sigma)/dln(R) = -R^2 <k^2> / 5, with <k^2> = int k^5 P dlnk / int k^3 P dlnk.
    """
    fn = WDMTruncated()

    def moment(p):
        f = lambda ln_k: math.exp((3 + p) * ln_k) * float(fn(np.array(math.exp(ln_k))))  # noqa: E731
        return integrate.quad(f, math.log(1e-8), math.log(1e5), limit=500, epsabs=0)[0]

    k2 = moment(2) / moment(0)
    mv = _mv(WDM, "TopHat")
    m = np.logspace(0, 2, 5) * Msun_h
    r = mv.radius_from_m(m).value
    assert np.all(r * math.sqrt(k2) < 0.02)
    want = -(r**2) * k2 / 5 / 3  # dln(R)/dln(m) = 1/3
    np.testing.assert_allclose(mv.dlnsigma_dlnm(m), want, rtol=2e-4)


# ---------------------------------------------------------------------------------
# Bit-identity under extension, and batch-size independence
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize("power", [EH, BAO], ids=["smooth", "bao"])
@pytest.mark.parametrize("flt", FILTERS)
@pytest.mark.parametrize("acc", [MassAccuracy(), MassAccuracy.fast()], ids=["default", "fast"])
def test_extension_is_bit_identical(power, flt, acc):
    """[1e8, 1e16] then [1e0, 1e18], or the opposite order: identical shared values."""
    acc = attrs.evolve(acc, log10_m_max=18.0)
    shared = np.logspace(8, 16, 397) * Msun_h
    wide = np.logspace(0, 18, 1001) * Msun_h

    narrow_first = _mv(power, flt, mass_accuracy=acc)
    s1, d1 = narrow_first.sigma(shared), narrow_first.dlnsigma_dlnm(shared)
    narrow_first.sigma(wide)
    s1_after, d1_after = narrow_first.sigma(shared), narrow_first.dlnsigma_dlnm(shared)

    wide_first = _mv(power, flt, mass_accuracy=acc)
    wide_first.sigma(wide)
    s2, d2 = wide_first.sigma(shared), wide_first.dlnsigma_dlnm(shared)

    for a in (s1_after, s2):
        assert np.array_equal(a, s1)
    for a in (d1_after, d2):
        assert np.array_equal(a, d1)
    # The wide grid's values at its own masses agree too.
    assert np.array_equal(narrow_first.sigma(wide), wide_first.sigma(wide))


@pytest.mark.parametrize("flt", FILTERS)
def test_batch_size_independence(flt):
    """One array, element by element, and in a fresh stage: bit-identical."""
    m = np.logspace(1, 17, 23) * Msun_h
    mv = _mv(BAO, flt)
    s, d = mv.sigma(m), mv.dlnsigma_dlnm(m)
    fresh = _mv(BAO, flt)
    assert np.array_equal(np.array([fresh.sigma(mm) for mm in m]), s)
    fresh = _mv(BAO, flt)
    assert np.array_equal(np.array([fresh.dlnsigma_dlnm(mm) for mm in m[::-1]])[::-1], d)
    # The node computation itself, in batches of different sizes.
    j = np.arange(-10, 900, 7)
    one_by_one = [mv._compute_nodes(j[i : i + 1]) for i in range(j.size)]
    assert np.array_equal(
        mv._compute_nodes(j)[:3], np.concatenate(one_by_one, axis=1)[:3], equal_nan=True
    )


def test_shape_is_preserved():
    mv = _mv()
    m = np.logspace(10, 14, 6).reshape(2, 3) * Msun_h
    assert mv.sigma(m).shape == (2, 3)
    assert mv.dlnsigma_dlnm(m).shape == (2, 3)
    assert np.ndim(mv.sigma(1e12 * Msun_h)) == 0
    assert mv.m_from_sigma(mv.sigma(m)).shape == (2, 3)


# ---------------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------------


def _interpolation_errors(mv, log10_m):
    """Max relative error of the lattice interpolant against direct evaluation."""
    ln_sigma, d = mv._interpolate(log10_m)
    direct = mv._direct(log10_m * LN10)
    return np.max(np.abs(np.expm1(ln_sigma - direct[0]))), np.max(np.abs(d / direct[1] - 1))


@pytest.mark.parametrize("power", [EH, BAO], ids=["smooth", "bao"])
@pytest.mark.parametrize("flt", FILTERS)
def test_interpolation_accuracy(power, flt):
    """The interpolant vs direct evaluation on the same k grid, at the default settings.

    The targets are 1e-5 in sigma and 1e-4 in dln(sigma)/dln(m) (SharpK + BAO is the
    hardest case for the interpolation).
    """
    log10_m = np.linspace(*LOG10_M_RANGE, 1901) + 0.0037
    err_sigma, err_d = _interpolation_errors(_mv(power, flt), log10_m)
    assert err_sigma <= 1e-5
    assert err_d <= 1e-4


def test_second_derivative_is_worthwhile():
    """d2 (quintic sigma, Hermite slope) beats the cubic / 4-point Lagrange schemes."""
    log10_m = np.linspace(*LOG10_M_RANGE, 1901) + 0.0037
    with_d2 = _interpolation_errors(_mv(BAO, "SharpK"), log10_m)
    without = _interpolation_errors(
        _mv(BAO, "SharpK", mass_accuracy=MassAccuracy(second_derivative=False)), log10_m
    )
    assert with_d2[0] < without[0] / 30
    assert with_d2[1] < without[1] / 3


@pytest.mark.parametrize("flt", FILTERS)
def test_default_against_high(flt):
    """The default settings against high() (mass and k), all errors included."""
    m = np.logspace(*LOG10_M_RANGE, 301) * Msun_h
    default = _mv(BAO, flt)
    high = _mv(BAO, flt, mass_accuracy=MassAccuracy.high(), k_accuracy=KAccuracy.high())
    np.testing.assert_allclose(default.sigma(m), high.sigma(m), rtol=1e-5)
    np.testing.assert_allclose(default.dlnsigma_dlnm(m), high.dlnsigma_dlnm(m), rtol=1e-4)


# ---------------------------------------------------------------------------------
# Physical bounds
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize("power", [EH, BAO], ids=["smooth", "bao"])
@pytest.mark.parametrize("flt", FILTERS)
def test_sigma_decreases_and_slope_is_negative(power, flt):
    m = np.logspace(0, 17.5, 1751) * Msun_h
    mv = _mv(power, flt)
    s = mv.sigma(m)
    assert np.all(np.diff(s) < 0)
    d = mv.dlnsigma_dlnm(m)
    assert np.all(d < 0)
    # CDM: the slope steepens from about -0.03 at 1 Msun/h to below -0.3 at 1e17.5.
    assert -0.06 < d[0] < -0.02
    assert d[-1] < -0.3


@pytest.mark.parametrize("flt", FILTERS)
def test_wdm_flattening_does_not_blow_up(flt):
    """A truncated P(k) flattens sigma(m) at low mass: the slope -> 0^- but stays finite."""
    m = np.logspace(0, 17.5, 876) * Msun_h
    wdm, cdm = _mv(WDM, flt), _mv(EH, flt)
    s, d = wdm.sigma(m), wdm.dlnsigma_dlnm(m)
    assert np.all(np.isfinite(s))
    assert np.all(np.isfinite(d))
    assert np.all(d < 0)
    assert np.all(np.diff(s) <= 0)
    low = m < 1e6 * Msun_h
    assert np.all(np.abs(d[low]) < 0.1 * np.abs(cdm.dlnsigma_dlnm(m[low])))
    # Large masses are unaffected by the truncation.
    np.testing.assert_allclose(d[-10:], cdm.dlnsigma_dlnm(m[-10:]), rtol=1e-3)


# ---------------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize("flt", [TopHat(), SharpK(), SmoothK()])
def test_radius_mass_round_trip(flt):
    mv = _mv(EH, flt)
    m = np.logspace(-2, 18, 50) * Msun_h
    r = mv.radius_from_m(m)
    assert r.unit is Mpc_h
    # m = 4 pi / 3 rho (c R)^3
    rho = EH.rho_mean0
    np.testing.assert_allclose(
        4 * math.pi / 3 * rho * (flt.mass_assignment * r.value) ** 3, m.value, rtol=1e-14
    )
    back = mv.m_from_radius(r)
    assert back.unit is Msun_h
    np.testing.assert_allclose(back.value, m.value, rtol=1e-14)


@pytest.mark.parametrize("power", [EH, BAO], ids=["smooth", "bao"])
@pytest.mark.parametrize("flt", FILTERS)
@pytest.mark.parametrize("second", [True, False])
def test_m_from_sigma_round_trip(power, flt, second):
    mv = _mv(power, flt, mass_accuracy=MassAccuracy(second_derivative=second))
    m = np.logspace(0.01, 17.49, 400) * Msun_h
    back = mv.m_from_sigma(mv.sigma(m))
    assert back.unit is Msun_h
    np.testing.assert_allclose(back.value, m.value, rtol=1e-11)


def test_m_from_sigma_out_of_range():
    mv = _mv()
    s_lo, s_hi = mv.sigma(10**17.5 * Msun_h), mv.sigma(1 * Msun_h)
    with pytest.raises(DomainError, match="outside"):
        mv.m_from_sigma(1.01 * s_hi)
    with pytest.raises(DomainError, match="outside"):
        mv.m_from_sigma(0.99 * s_lo)
    with pytest.raises(
        DomainError, match=r"outside the domain \(.*sigma > 0\); out of range: sigma"
    ):
        mv.m_from_sigma(-1.0)


def test_m_from_sigma_raises_where_sigma_is_not_monotonic():
    """A spike in P(k) makes the top-hat sigma(R) oscillate at R >~ 4.5 / k_spike."""

    def spiky(k):
        return 1e-3 * EisensteinHuNoWiggle()(k) + np.exp(-0.5 * (np.log(k / 0.05) / 0.1) ** 2)

    mv = _mv(AnalyticPower(spiky), "TopHat")
    m = np.logspace(10, 17.5, 751) * Msun_h
    s = mv.sigma(m)
    rising = np.flatnonzero(np.diff(s) > 0)
    assert rising.size, "the spectrum should make sigma non-monotonic"
    with pytest.raises(DomainError, match="not monotonically decreasing"):
        mv.m_from_sigma(s[rising[0] + 1])
    # Where sigma is monotonic, inversion still works.
    np.testing.assert_allclose(mv.m_from_sigma(s[0]).value, m[0].value, rtol=1e-10)


# ---------------------------------------------------------------------------------
# The k grid and the truncation estimator
# ---------------------------------------------------------------------------------


def test_k_grid_is_fixed_by_the_settings():
    """K nodes at ln k = i dlnk, k_max >= k_max_r_min / R(m_min), independent of requests."""
    mv = _mv()
    grid = mv._k_grid
    i = np.round(grid.ln_k / grid.dln_k)
    np.testing.assert_array_equal(grid.ln_k, i * grid.dln_k)
    assert np.all(np.diff(i) == 1)
    assert grid.k.size % 2 == 1
    r_min = mv.radius_from_m(1 * Msun_h).value
    assert grid.k[-1] * r_min >= 20.0
    assert grid.k[-1] * r_min < 20.0 * math.exp(2 * grid.dln_k)
    assert grid.k[0] <= 1e-8 < grid.k[0] * math.exp(grid.dln_k)
    mv.sigma(np.logspace(0, 18, 50) * Msun_h)
    assert mv._k_grid is grid


@pytest.mark.parametrize("flt", ["TopHat", "SmoothK"])
def test_truncation_bound_covers_the_true_error(flt):
    """The bound exceeds the true truncation error of sigma and of dln(sigma)/dln(m).

    The true error is measured against a grid extended to much higher k, at a fine
    dlnk that resolves the window's oscillations, so that only the truncation differs.
    """
    log10_m = np.linspace(-2.5, 3, 45)
    ka = KAccuracy(dln_k=0.004)
    mv = _mv(EH, flt, k_accuracy=ka)
    ref = _mv(EH, flt, k_accuracy=ka, mass_accuracy=MassAccuracy(log10_m_min=-7))
    a, b = mv._direct(log10_m * LN10), ref._direct(log10_m * LN10)
    true_sigma = np.abs(np.expm1(a[0] - b[0]))
    true_d = np.abs(a[1] / b[1] - 1)
    roundoff = 1e-14
    assert np.all(true_sigma <= a[3] + roundoff)
    assert np.all(true_d <= a[4] + roundoff)
    if flt == "TopHat":
        # dndm is far more sensitive than sigma: a sigma-only bound would miss the error.
        assert np.max(true_d / a[3]) > 20


def test_masses_below_the_resolving_limit_raise():
    mv = _mv()
    mv.sigma(0.3 * Msun_h)  # below log10_m_min, but resolved
    with pytest.raises(ValueError, match="low-mass end"):
        mv.sigma(1e-3 * Msun_h)
    with pytest.raises(ValueError, match="low-mass end"):
        mv.dlnsigma_dlnm(1e-3 * Msun_h)
    # Lowering log10_m_min moves k_max up, and the limit down.
    _mv(mass_accuracy=MassAccuracy(log10_m_min=-4)).sigma(1e-3 * Msun_h)
    # SharpK: only k = 1/R needs to be on the grid.
    sk = _mv(flt="SharpK")
    sk.sigma(1e-3 * Msun_h)
    with pytest.raises(ValueError, match="low-mass end"):
        sk.sigma(1e-8 * Msun_h)


def test_masses_above_the_resolving_limit_raise():
    mv = _mv(k_accuracy=KAccuracy(ln_k_min=math.log(1e-2)))
    mv.sigma(1e12 * Msun_h)
    with pytest.raises(ValueError, match="high-mass end"):
        mv.sigma(1e18 * Msun_h)


def test_underflow_raises():
    """A power so small that sigma^2 underflows (to 0 at R ~ 1e4 Mpc/h) gives an error."""
    mv = _mv(AnalyticPower(PowerLaw(0.0, amplitude=1e-310)), "SharpK")
    mv.sigma(1e15 * Msun_h)
    with pytest.raises(DomainError, match="not finite"):
        mv.sigma(1e26 * Msun_h)


def test_extension_raise():
    mv = _mv(mass_accuracy=MassAccuracy(extension="raise", log10_m_min=6, log10_m_max=15))
    mv.sigma(np.array([1e6, 1e15]) * Msun_h)
    with pytest.raises(DomainError, match="extension='raise'"):
        mv.sigma(1e5 * Msun_h)
    with pytest.raises(DomainError, match="extension='raise'"):
        mv.dlnsigma_dlnm(1e16 * Msun_h)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_invalid_masses_raise(bad):
    with pytest.raises(
        DomainError, match=r"MassVariance.sigma: 1 of 1 value\(s\) are outside the domain \("
    ):
        _mv().sigma(bad * Msun_h)


# ---------------------------------------------------------------------------------
# Units boundary
# ---------------------------------------------------------------------------------


def test_units_boundary():
    mv = _mv()
    for method in (mv.sigma, mv.dlnsigma_dlnm, mv.radius_from_m):
        with pytest.raises(UnitBoundaryError):
            method(1e12)
    with pytest.raises(UnitBoundaryError):
        mv.m_from_radius(1.0)
    with pytest.raises(u.UnitConversionError):
        mv.sigma(1.0 * Mpc_h)
    # Physical units are converted with the power source's H0 (70 km/s/Mpc).
    assert mv.sigma(1e12 * u.Msun) == mv.sigma(0.7e12 * Msun_h)
    assert mv.dlnsigma_dlnm(1e12 * u.Msun) == mv.dlnsigma_dlnm(0.7e12 * Msun_h)
    np.testing.assert_allclose(
        mv.m_from_radius(1 * u.Mpc).value, mv.m_from_radius(0.7 * Mpc_h).value, rtol=1e-14
    )
    assert mv.radius_from_m(1e12 * u.Msun) == mv.radius_from_m(0.7e12 * Msun_h)
    # sigma is dimensionless: m_from_sigma takes a plain number.
    assert mv.m_from_sigma(float(mv.sigma(1e12 * Msun_h))).unit is Msun_h


# ---------------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------------


def test_node_cache_is_not_part_of_the_value():
    a, b = _mv(), _mv()
    a.sigma(1e12 * Msun_h)
    assert a == b
    assert hash(a) == hash(b)
    assert a.evolve(filter="SharpK") != a


def test_evolve_starts_a_new_lattice():
    a = _mv()
    s = a.sigma(1e12 * Msun_h)
    b = a.evolve(filter="SmoothK")
    assert b._nodes is not a._nodes
    assert b.sigma(1e12 * Msun_h) != s
    assert isinstance(b.filter, SmoothK)


def test_invalid_fields():
    with pytest.raises(TypeError, match="PowerSource"):
        MassVariance(power=object())
    with pytest.raises(TypeError):
        MassVariance(power=EH, mass_accuracy=KAccuracy())


def test_with_a_tabulated_source():
    """A TabulatedPower of a power law gives the power-law closed form."""
    k = np.logspace(-4, 3, 30)
    src = TabulatedPower(
        k=k * h_Mpc, pk=k**-1.0 * power_unit, mean_density=0.3 * RHO_CRIT0 * rho_unit
    )
    mv = MassVariance(power=src, filter="SharpK")
    m = np.logspace(8, 14, 5) * Msun_h
    # The k grid reaches beyond the table, which is extrapolated: exactly, for a power law.
    with pytest.warns(HMFExtrapolationWarning, match="extrapolated as a power law"):
        np.testing.assert_allclose(mv.dlnsigma_dlnm(m), -1 / 3, rtol=1e-8)


# ---------------------------------------------------------------------------------
# Regression against hmf v3 (a cross-check, not a physical test)
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize("flt", FILTERS)
def test_against_v3(flt):
    from hmf.density_field import filters as v3

    fn = WithBAO()
    k = np.exp(np.arange(math.log(1e-8), math.log(1e6), 0.002))
    params = {"SharpK": {"c": 2.5}, "SmoothK": {"beta": 4.8, "c": 3.3}}.get(flt, {})
    old = getattr(v3, flt)(k, fn(k), **params)
    mv = _mv(BAO, flt)
    m = np.logspace(6, 15.5, 20) * Msun_h
    r = mv.radius_from_m(m).value
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s3, dlnss = old.sigma_and_dlnss_dlnm(r)
    np.testing.assert_allclose(mv.sigma(m), s3, rtol=1e-5)
    np.testing.assert_allclose(mv.dlnsigma_dlnm(m), dlnss / 2, rtol=2e-4)


def test_tail_integral_kernel():
    """A power-law tail integrates in closed form; divergent or zero terms are handled."""
    from hmf.core._kernels.mass_variance import tail_integral

    # int_{k_e}^inf k^3 P (kR)^-4 dlnk with k^3 P = k^(3+n), n = -3: k_e^0 x_e^-4 / 4.
    got = tail_integral(1.0, -3.0, 10.0, ((1.0, -4.0),), high=True)
    assert got == pytest.approx(1e-4 / 4, rel=1e-14)
    # Low-k: int_0^{k_e} k^(3+n) dlnk with n = 1 is k3p_e / 4.
    assert tail_integral(2.0, 1.0, 1e-3, ((1.0, 0.0),), high=False) == pytest.approx(0.5)
    # Divergent terms give inf, unless their coefficient is 0.
    assert tail_integral(1.0, 2.0, 10.0, ((1.0, -4.0),), high=True) == np.inf
    assert tail_integral(1.0, 2.0, 10.0, ((0.0, -4.0),), high=True, oscillating=((0.0, -1.0),)) == 0
    assert tail_integral(1.0, 2.0, 10.0, (), high=True, oscillating=((1.0, -1.0),)) == np.inf
    # The oscillating bound is 2 a k3p x^(q-1) / omega.
    got = tail_integral(1.0, -3.0, 10.0, (), high=True, oscillating=((1.0, -3.0),), omega=2.0)
    assert got == pytest.approx(1e-4, rel=1e-14)


def test_sharpk_tail_bounds():
    tb = SharpK().tail_bounds()
    assert tb.high_w2 == () and tb.high_wdw == () and tb.low_wdw == ()  # noqa: PT018
    assert tb.low_w2 == ((1.0, 0.0),)


def test_m_from_sigma_with_off_lattice_range():
    """The default range need not sit on lattice nodes: the inside nodes are used."""
    mv = _mv(mass_accuracy=MassAccuracy(log10_m_min=0.011, log10_m_max=17.49))
    m = np.logspace(0.03, 17.47, 50) * Msun_h
    np.testing.assert_allclose(mv.m_from_sigma(mv.sigma(m)).value, m.value, rtol=1e-11)
    with pytest.raises(ValueError, match="outside"):
        mv.m_from_sigma(mv.sigma(10**0.015 * Msun_h))


def test_non_positive_power_raises():
    with pytest.raises(ValueError, match="finite and > 0"):
        _mv(AnalyticPower(lambda k: np.where(k > 1e3, 0.0, k**-1.0))).sigma(1e12 * Msun_h)


@pytest.mark.parametrize("flt", FILTERS)
def test_kernel_entry_point_matches_the_public_methods(flt):
    """ln_sigma_and_slope_kernel takes plain Msun/h and gives sigma's ln and slope."""
    mv = _mv(EH, flt)
    m = np.logspace(8, 15, 12).reshape(3, 4)
    ln_sigma, slope = mv.ln_sigma_and_slope_kernel(m)
    assert ln_sigma.shape == slope.shape == (3, 4)
    np.testing.assert_array_equal(np.exp(ln_sigma), mv.sigma(m * Msun_h))
    np.testing.assert_array_equal(slope, mv.dlnsigma_dlnm(m * Msun_h))
    with pytest.raises(DomainError):
        mv.ln_sigma_and_slope_kernel(np.array([-1.0]))


# ---------------------------------------------------------------------------------
# The mass of a filter radius: CDM + baryons
# ---------------------------------------------------------------------------------

#: A cosmology with 0.3 eV of massive neutrinos: the total-matter density is about
#: 2% above the CDM + baryon one.
MASSIVE_NU = FlatLambdaCDM(
    H0=67.66, Om0=0.30966, Ob0=0.04897, Tcmb0=2.7255, m_nu=[0.1, 0.1, 0.1] * u.eV
)


def _rho_cb(cosmology):
    """The CDM + baryon density in Msun h^2 / Mpc^3, from astropy's critical density."""
    h = cosmology.H0.value / 100
    rho_crit = cosmology.critical_density0.to_value(u.Msun / u.Mpc**3) / h**2
    # astropy counts massive neutrinos in Onu0, not Om0.
    return cosmology.Om0 * rho_crit


@pytest.mark.parametrize("flt", [TopHat(), SharpK(), SmoothK()])
def test_mass_radius_uses_cdm_plus_baryons_for_every_species(flt):
    """With massive neutrinos, the tot and cb power give the same R(M), that of rho_cb.

    R = (3M / 4 pi rho_cb)^(1/3) / c, with c the filter's mass assignment: haloes are
    made of CDM and baryons, whichever species' power sets sigma.
    """
    transfer = Transfer(cosmology=MASSIVE_NU, model="EH")
    tot = MassVariance(power=transfer.power_kernel("tot"), filter=flt)
    cb = MassVariance(power=transfer.power_kernel("cb"), filter=flt)
    m = np.logspace(0, 18, 37)
    r_tot, r_cb = tot.radius_from_m_kernel(m), cb.radius_from_m_kernel(m)
    np.testing.assert_array_equal(r_tot, r_cb)
    want = (3 * m / (4 * math.pi * _rho_cb(MASSIVE_NU))) ** (1 / 3) / flt.mass_assignment
    np.testing.assert_allclose(r_tot, want, rtol=1e-12)
    # The total-matter density would differ by the neutrinos' share, about 2%.
    rho_tot = _rho_cb(MASSIVE_NU) + MASSIVE_NU.Onu0 * _rho_cb(MASSIVE_NU) / MASSIVE_NU.Om0
    assert rho_tot / _rho_cb(MASSIVE_NU) - 1 > 0.01
    np.testing.assert_allclose(tot.m_from_radius_kernel(want), m, rtol=1e-12)


# ---------------------------------------------------------------------------------
# Kernels: sigma(R), the converters and the inverse
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [-2.5, -2.0])
def test_sigma_at_radius_of_a_power_law_scales_as_r_to_the_minus_n_plus_3_over_2(n):
    """For P = k^n with the top-hat, sigma(R) = sigma(1) R^(-(n+3)/2).

    The integral misses the part below the grid's k_min, (k_min R)^(3+n) / (3+n) of
    the integral (see _power_law_reference), which is added back: then
    sigma^2 R^(3+n) is constant, to the quadrature's accuracy (1e-6, from the
    top-hat's oscillating tail beyond k_max), and equals the closed form.
    """
    mv = _mv(AnalyticPower(PowerLaw(n)), "TopHat")
    r = np.geomspace(0.05, 20.0, 37)
    s2 = np.exp(2 * mv.ln_sigma_at_radius_kernel(r))
    k_min = mv._k_grid.k[0]
    s2_full = s2 + k_min ** (3 + n) / (3 + n) / (2 * math.pi**2)
    scaled = s2_full * r ** (3 + n)
    np.testing.assert_allclose(scaled, scaled[0], rtol=1e-6)
    np.testing.assert_allclose(scaled, _tophat_integral(n) / (2 * math.pi**2), rtol=1e-6)
    # The log-slope of sigma in R.
    slope = np.polyfit(np.log(r), 0.5 * np.log(s2_full), 1)[0]
    assert slope == pytest.approx(-(n + 3) / 2, rel=1e-6)


@pytest.mark.parametrize("power", [EH, BAO], ids=["smooth", "bao"])
@pytest.mark.parametrize("flt", FILTERS)
def test_sigma_at_radius_agrees_with_the_lattice(power, flt):
    """Sigma at a radius, evaluated directly, and the lattice's at its mass agree.

    To INTERPOLATION_RTOL, the documented accuracy of the lattice's interpolant at
    the default settings; at a lattice node they are the same integral.
    """
    mv = _mv(power, flt)
    m = 10 ** (np.linspace(*LOG10_M_RANGE, 211) + 0.0037)
    ln_lattice, _ = mv.ln_sigma_and_slope_kernel(m)
    ln_direct = mv.ln_sigma_at_radius_kernel(mv.radius_from_m_kernel(m))
    np.testing.assert_allclose(np.exp(ln_direct), np.exp(ln_lattice), rtol=INTERPOLATION_RTOL)
    nodes = 10.0 ** np.arange(8, 15)
    np.testing.assert_allclose(
        mv.ln_sigma_at_radius_kernel(mv.radius_from_m_kernel(nodes)),
        mv.ln_sigma_and_slope_kernel(nodes)[0],
        rtol=1e-13,
    )


def test_sigma_at_radius_is_independent_of_the_batch_and_shape():
    mv = _mv(EH, "TopHat")
    r = np.geomspace(0.1, 30.0, 150)
    batch = mv.ln_sigma_at_radius_kernel(r)
    alone = np.array([mv.ln_sigma_at_radius_kernel(x) for x in r[::17]])
    np.testing.assert_array_equal(alone, batch[::17])
    assert mv.ln_sigma_at_radius_kernel(r.reshape(10, 15)).shape == (10, 15)
    assert np.ndim(mv.ln_sigma_at_radius_kernel(8.0)) == 0


def test_sigma_at_radius_ignores_the_lattice_range():
    """A direct evaluation is not limited by mass_accuracy.extension='raise'."""
    mv = _mv(EH, "TopHat", mass_accuracy=MassAccuracy(log10_m_max=14, extension="raise"))
    r_large = float(mv.radius_from_m_kernel(1e16))
    assert np.isfinite(mv.ln_sigma_at_radius_kernel(r_large))
    with pytest.raises(DomainError, match="extension='raise'"):
        mv.ln_sigma_and_slope_kernel(1e16)


def test_sigma_at_radius_raises_where_the_grid_cannot_resolve():
    """Below the k grid's reach, the direct evaluation raises, as the lattice does."""
    mv = _mv(EH, "TopHat")
    with pytest.raises(DomainError, match="can't evaluate"):
        mv.ln_sigma_at_radius_kernel(1e-6)
    for bad in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(DomainError, match=r"ln_sigma_at_radius_kernel: r"):
            mv.ln_sigma_at_radius_kernel(np.array([8.0, bad]))


@pytest.mark.parametrize("flt", [TopHat(), SharpK(), SmoothK()])
def test_mass_radius_kernels_round_trip(flt):
    """M -> R -> m and R -> m -> R to 1e-12, and the kernels are the public methods."""
    mv = _mv(EH, flt)
    m = np.logspace(-3, 19, 89).reshape(89, 1)
    r = mv.radius_from_m_kernel(m)
    assert r.shape == m.shape
    np.testing.assert_allclose(mv.m_from_radius_kernel(r), m, rtol=1e-12)
    radii = np.geomspace(1e-3, 1e2, 50)
    m_of_r = mv.m_from_radius_kernel(radii)
    np.testing.assert_allclose(mv.radius_from_m_kernel(m_of_r), radii, rtol=1e-12)
    np.testing.assert_array_equal(mv.radius_from_m(m * Msun_h).value, r)
    np.testing.assert_array_equal(mv.m_from_radius(radii * Mpc_h).value, m_of_r)


def test_m_from_sigma_kernel_matches_the_public_method():
    mv = _mv(EH, "TopHat")
    m = np.logspace(6, 15, 24).reshape(4, 6)
    sigma = np.exp(mv.ln_sigma_and_slope_kernel(m)[0])
    back = mv.m_from_sigma_kernel(sigma)
    assert back.shape == (4, 6)
    np.testing.assert_allclose(back, m, rtol=1e-11)
    np.testing.assert_array_equal(mv.m_from_sigma(sigma).value, back)


# ---------------------------------------------------------------------------------
# The stage's domain
# ---------------------------------------------------------------------------------


def test_valid_domain():
    mv = _mv()
    domain = mv.valid_domain
    assert domain.variables == ("m", "r", "sigma")
    assert domain["m"].unit is Msun_h
    assert domain["r"].unit is Mpc_h
    assert domain.contains(m=1e-30 * Msun_h, r=1e5 * Mpc_h, sigma=1e3)
    for bad in (0.0, -1.0, np.inf, np.nan):
        assert not domain.contains(m=bad * Msun_h)
        assert not domain.contains(r=bad * Mpc_h)
        assert not domain.contains(sigma=bad)


def test_valid_domain_with_extension_raise_is_the_lattice():
    mv = _mv(mass_accuracy=MassAccuracy(log10_m_min=6, log10_m_max=15, extension="raise"))
    interval = mv.valid_domain["m"]
    assert interval.lower == pytest.approx(1e6, rel=1e-11)
    assert interval.upper == pytest.approx(1e15, rel=1e-11)
    # The ends themselves are in it.
    mv.sigma(np.array([1e6, 1e15]) * Msun_h)
    with pytest.raises(DomainError, match=r"MassVariance.sigma \(the mass lattice"):
        mv.sigma(1e16 * Msun_h)
    # Converting a mass to a radius is not limited to the lattice.
    assert mv.radius_from_m(1e16 * Msun_h).value > 0


@pytest.mark.parametrize(
    ("call", "where"),
    [
        (lambda mv: mv.sigma(np.array([1e12, -1.0]) * Msun_h), "MassVariance.sigma"),
        (lambda mv: mv.dlnsigma_dlnm(np.nan * Msun_h), "MassVariance.dlnsigma_dlnm"),
        (lambda mv: mv.m_from_sigma(np.inf), "MassVariance.m_from_sigma"),
        (lambda mv: mv.m_from_radius(0.0 * Mpc_h), "MassVariance.m_from_radius"),
        (lambda mv: mv.radius_from_m(-1.0 * Msun_h), "MassVariance.radius_from_m"),
    ],
)
def test_public_methods_check_the_domain(call, where):
    """The public methods raise Domain.check's DomainError: count, domain and variable."""
    with pytest.raises(DomainError, match=rf"^{where}: 1 of \d value\(s\) are outside the domain"):
        call(_mv())
