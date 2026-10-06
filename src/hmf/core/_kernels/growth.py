r"""Kernels of the linear growth factor.

Everything here is a function of the scale factor ``a`` (dimensionless), through
``ln_a``. Growth factors are returned unnormalised; the growth stage normalises them
to 1 today. The growth rate is :math:`f = d\ln D / d\ln a`.

The cosmology enters as plain floats (density parameters today) or as arrays of
background quantities tabulated on the ``ln_a`` grid by the caller.
"""

from __future__ import annotations

import attrs
import numpy as np
import numpy.typing as npt
from scipy.integrate import cumulative_simpson
from scipy.interpolate import CubicSpline, PPoly
from scipy.special import ellipeinc, ellipkinc

from .arrays import read_only
from .lattice import lattice, lattice_index

__all__ = [
    "GrowthTable",
    "carroll92_growth",
    "eisenstein97_growth",
    "genmf_growth",
    "heath77_growth",
    "integral_growth",
    "ln_a_grid",
    "solve_growth_ode",
    "tabulate_growth",
]

Array = npt.NDArray[np.float64]


def ln_a_grid(a_min: float, dln_a: float) -> Array:
    """A grid in ln a from ``ln(a_min)`` to 0 (included), with spacing ``dln_a``.

    The nodes are on the lattice ``ln a = i * dln_a`` for integers ``i <= 0`` (see
    :mod:`~hmf.core._kernels.lattice`), so that grids with the same spacing share their
    nodes and a = 1 is always a node.

    Parameters
    ----------
    a_min
        The smallest scale factor; the grid starts at the first node at or below it.
    dln_a
        The spacing of the grid.

    Returns
    -------
    numpy.ndarray
        Increasing ln a, ending at exactly 0.
    """
    return lattice(lattice_index(float(np.log(a_min)), dln_a, up=False), 0, dln_a)


def solve_growth_ode(
    ln_a: Array,
    dln_e_dln_a: Array,
    omega_m_a: Array,
    d_init: float,
    f_init: float,
) -> tuple[Array, Array]:
    r"""Solve the linear growth equation with fixed-step fourth-order Runge-Kutta.

    The equation, in :math:`x = \ln a`, is

    .. math:: D'' + \left(2 + \frac{d\ln E}{d\ln a}\right) D'
              - \frac{3}{2}\Omega_m(a) D = 0,

    with :math:`\Omega_m(a) = \Omega_{m,0} a^{-3} / E^2(a)` the density parameter of
    the matter that clusters. It is solved for :math:`(D, D')`.

    Parameters
    ----------
    ln_a
        An evenly spaced grid in ln a, with an **odd** number of points. The solution
        is returned at every other point (``ln_a[::2]``); the points in between are the
        Runge-Kutta midpoints.
    dln_e_dln_a
        :math:`d\ln E/d\ln a` on ``ln_a``.
    omega_m_a
        :math:`\Omega_m(a)` on ``ln_a``.
    d_init, f_init
        The growth factor and growth rate :math:`D'/D` at ``ln_a[0]``.

    Returns
    -------
    d, f : numpy.ndarray
        The growth factor (normalised to ``d_init`` at the start) and the growth rate,
        on ``ln_a[::2]``.
    """
    ln_a = np.asarray(ln_a, dtype=float)
    if ln_a.size < 3 or ln_a.size % 2 == 0:
        raise ValueError("solve_growth_ode needs an odd number (>= 3) of grid points.")
    h = ln_a[2] - ln_a[0]
    # Coefficients of y' = A(x) y, with y = (D, D'): A = [[0, 1], [c0, -c1]].
    # Python floats: the loop below is scalar, and indexing lists is faster.
    c1 = (2.0 + np.asarray(dln_e_dln_a, dtype=float)).tolist()
    c0 = (1.5 * np.asarray(omega_m_a, dtype=float)).tolist()
    h = float(h)

    n = (ln_a.size - 1) // 2
    d = [float(d_init)]
    dp = [float(d_init * f_init)]
    y0, y1 = d[0], dp[0]
    for i in range(n):
        j = 2 * i
        k1a, k1b = y1, c0[j] * y0 - c1[j] * y1
        y0b, y1b = y0 + 0.5 * h * k1a, y1 + 0.5 * h * k1b
        k2a, k2b = y1b, c0[j + 1] * y0b - c1[j + 1] * y1b
        y0c, y1c = y0 + 0.5 * h * k2a, y1 + 0.5 * h * k2b
        k3a, k3b = y1c, c0[j + 1] * y0c - c1[j + 1] * y1c
        y0d, y1d = y0 + h * k3a, y1 + h * k3b
        k4a, k4b = y1d, c0[j + 2] * y0d - c1[j + 2] * y1d
        y0 = y0 + h / 6 * (k1a + 2 * k2a + 2 * k3a + k4a)
        y1 = y1 + h / 6 * (k1b + 2 * k2b + 2 * k3b + k4b)
        d.append(y0)
        dp.append(y1)
    d_out = np.array(d)
    f: Array = np.array(dp) / d_out
    return d_out, f


def integral_growth(
    ln_a: Array, efunc: Array, dln_e_dln_a: Array, omega_m: float
) -> tuple[Array, Array]:
    r"""The integral form of the growth factor (Heath 1977), and its growth rate.

    .. math:: D^+(a) = \frac52 \Omega_{m,0} E(a) \int_0^a \frac{da'}{(a' E(a'))^3}

    is exact for a cosmological constant and negligible radiation (in any curvature).
    The integral is done with Simpson's rule on the grid, plus the contribution
    below ``ln_a[0]`` in matter domination, :math:`a_0^{5/2}/(\frac52\Omega_{m,0}^{3/2})`.
    The growth rate is the exact derivative of that expression,

    .. math:: f = \frac{d\ln E}{d\ln a} + \frac{5\Omega_{m,0}}{2 a^2 E^2 D^+},

    so it is consistent with :math:`D^+` under the same assumptions.

    Parameters
    ----------
    ln_a
        Increasing grid in ln a.
    efunc
        :math:`E(a) = H(a)/H_0` on the grid.
    dln_e_dln_a
        :math:`d\ln E/d\ln a` on the grid.
    omega_m
        :math:`\Omega_{m,0}`.

    Returns
    -------
    d, f : numpy.ndarray
        :math:`D^+` (with :math:`D^+ \to a` as :math:`a \to 0`) and the growth rate.
    """
    a = np.exp(ln_a)
    integrand = a / (a * efunc) ** 3  # d/dln a of the integral
    start = a[0] ** 2.5 / (2.5 * omega_m**1.5)
    integral = start + cumulative_simpson(integrand, x=ln_a, initial=0.0)
    d: Array = 2.5 * omega_m * efunc * integral
    f: Array = dln_e_dln_a + 2.5 * omega_m / (a**2 * efunc**2 * d)
    return d, f


def eisenstein97_growth(ln_a: Array, omega_m: float) -> tuple[Array, Array]:
    r"""The growth factor of a flat universe with Lambda and no radiation (Eisenstein 1997).

    Eisenstein (1997), Eqs. 8-10: the closed form of the integral growth factor in
    terms of incomplete elliptic integrals. At :math:`v > 8` the series of their
    Eq. 10 is used, which avoids a loss of precision. The growth rate is that of
    :func:`integral_growth`, with :math:`E^2 = \Omega_{m,0} a^{-3} + 1 - \Omega_{m,0}`.

    Parameters
    ----------
    ln_a
        Grid in ln a.
    omega_m
        :math:`\Omega_{m,0}`, in (0, 1]; Lambda is :math:`1 - \Omega_{m,0}`.

    Returns
    -------
    d, f : numpy.ndarray
        :math:`D^+` (with :math:`D^+ \to a` as :math:`a \to 0`) and the growth rate.
    """
    a = np.exp(ln_a)
    omega_l = 1.0 - omega_m
    e2 = omega_m * a**-3 + omega_l
    if omega_l == 0:
        return a.copy(), np.ones_like(a)
    v = (omega_m / omega_l) ** (1 / 3) / a
    sqrt3 = np.sqrt(3.0)
    m = np.sin(np.deg2rad(75.0)) ** 2
    beta = np.arccos((v + 1 - sqrt3) / (v + 1 + sqrt3))
    term1 = 3**0.25 * np.sqrt(1 + v**3) * (ellipeinc(beta, m) - ellipkinc(beta, m) / (3 + sqrt3))
    term2 = (1 - (sqrt3 + 1) * v**2) / (v + 1 + sqrt3)
    with np.errstate(over="ignore", invalid="ignore"):
        exact = 5 / 3 * v * (term1 + term2)
    series = 1 - (2 / 11) * v**-3 + (16 / 187) * v**-6
    d: Array = a * np.where(v > 8, series, exact)
    dln_e = -1.5 * omega_m * a**-3 / e2
    f: Array = dln_e + 2.5 * omega_m / (a**2 * e2 * d)
    return d, f


def heath77_growth(ln_a: Array, omega_m: float) -> tuple[Array, Array]:
    r"""The growth factor of an open or closed universe without Lambda or radiation.

    Heath (1977), Eq. 13, with :math:`\sigma_0 = \Omega_{m,0}/2`, divided by its
    small-a limit :math:`|1-\Omega_{m,0}| a/(\frac52 \Omega_{m,0})` so that
    :math:`D^+ \to a`. The growth rate is that of :func:`integral_growth`, with
    :math:`E^2 = \Omega_{m,0} a^{-3} + (1 - \Omega_{m,0}) a^{-2}`.

    Parameters
    ----------
    ln_a
        Grid in ln a.
    omega_m
        :math:`\Omega_{m,0} > 0`; the curvature is :math:`1 - \Omega_{m,0}`.

    Returns
    -------
    d, f : numpy.ndarray
    """
    a = np.exp(ln_a)
    e2 = omega_m * a**-3 + (1 - omega_m) * a**-2
    if omega_m == 1:
        return a.copy(), np.ones_like(a)
    z = 1 / a - 1
    s0 = omega_m / 2
    p = (2 * s0 * z + 1) * (1 + z) ** 2
    x = (s0 * z - s0 + 1) / (s0 * (1 + z))
    with np.errstate(invalid="ignore"):
        theta = np.arccos(np.clip(x, -1, 1)) if s0 > 0.5 else np.arccosh(np.maximum(x, 1))
    k = np.abs(2 * s0 - 1)
    closed = ((6 * s0 * z + 4 * s0 + 1) / k - 3 * theta * s0 * np.sqrt(p) / k**1.5) * 5 * s0 / k
    # Eq. 13 cancels catastrophically at small a. There, use the series of the same
    # growing mode in y = (1/Omega_m - 1) a (see genmf_growth), D = a S(y) / (0.4 y).
    y = (1 / omega_m - 1) * a
    small = np.abs(y) < 1e-2
    d: Array = np.where(small, a * _open_series(np.where(small, y, 1.0)) / 0.4, closed)
    dln_e = (-1.5 * omega_m * a**-3 - (1 - omega_m) * a**-2) / e2
    f: Array = dln_e + 2.5 * omega_m / (a**2 * e2 * d)
    return d, f


def genmf_growth(ln_a: Array, omega_m: float, omega_l: float) -> Array:
    r"""The growth factor of the ``genmf`` code (Reed et al. 2007).

    Exact for a flat universe with Lambda, or an open one without it, and no
    radiation. With Lambda it is the integral of the flat case in terms of
    :math:`x = a (2 w)^{1/3}`, :math:`w = 1/\Omega_{m,0} - 1`; without it, the
    closed form for the open universe in terms of :math:`x = w a` (with a Taylor
    series at small x, where the closed form cancels catastrophically).

    Parameters
    ----------
    ln_a
        Increasing grid in ln a.
    omega_m
        :math:`\Omega_{m,0}`.
    omega_l
        :math:`\Omega_{\Lambda,0}`: either 0 (open) or :math:`1 - \Omega_{m,0}` (flat).

    Returns
    -------
    numpy.ndarray
        The growth factor, normalised to :math:`D \to a` as :math:`a \to 0`.
    """
    a = np.exp(ln_a)
    if omega_m == 1:
        return a.copy()
    w = 1 / omega_m - 1
    if omega_l > 0:
        xn = (2 * w) ** (1 / 3)
        x = a * xn
        # g(x) = int_0^x (y / (y^3 + 2))^1.5 dy, done in ln y on the grid plus the
        # small-y part, where the integrand is (y/2)^1.5.
        integrand = (x / (x**3 + 2)) ** 1.5 * x
        g = x[0] ** 2.5 / (2.5 * 2**1.5) + cumulative_simpson(integrand, x=ln_a, initial=0.0)
        d_flat: Array = np.sqrt(x**3 + 2) * g / x**1.5
        # Normalise to D -> a: d_flat -> sqrt(2) x / (2.5 * 2^1.5) as x -> 0.
        out: Array = d_flat / (np.sqrt(2) * xn / (2.5 * 2**1.5))
        return out
    x = w * a
    small = x < 1e-2
    xs = np.where(small, x, 1.0)
    xl = np.where(small, 1.0, x)
    series = xs * _open_series(xs)
    closed = 1 + 3 / xl + 3 * np.sqrt(1 + xl) / xl**1.5 * np.log(np.sqrt(1 + xl) - np.sqrt(xl))
    open_case: Array = np.where(small, series, closed) / (0.4 * w)
    return open_case


def _open_series(y: Array) -> Array:
    r"""The Lambda-free growing mode divided by y, as a series in y = (1/Omega_m - 1) a.

    ``y * _open_series(y)`` is :math:`1 + 3/y - 3\sqrt{1+y}\,{\rm arcsinh}(\sqrt y)/y^{3/2}`
    (and its continuation to y < 0, for a closed universe), to O(y^6).
    """
    out: Array = 2 / 5 + y * (
        -8 / 35 + y * (16 / 105 + y * (-128 / 1155 + y * (256 / 3003 - y * 1024 / 15015)))
    )
    return out


def carroll92_growth(a: Array, omega_m_a: Array, omega_l_a: Array) -> tuple[Array, Array]:
    r"""The approximate growth factor and rate of Carroll, Press & Turner (1992).

    :math:`D^+ = \frac52 a\,\Omega_m / [\Omega_m^{4/7} - \Omega_\Lambda +
    (1 + \Omega_m/2)(1 + \Omega_\Lambda/70)]` (CPT92 Eq. 29, after Lahav et al.
    1991), with the density parameters at ``a``, and
    :math:`f = \Omega_m^{4/7} + \Omega_\Lambda (1 + \Omega_m/2)/70`.

    Parameters
    ----------
    a
        Scale factors.
    omega_m_a, omega_l_a
        :math:`\Omega_m(a)` and :math:`\Omega_\Lambda(a)`.

    Returns
    -------
    d, f : numpy.ndarray
    """
    d: Array = (
        2.5
        * a
        * omega_m_a
        / (omega_m_a ** (4 / 7) - omega_l_a + (1 + 0.5 * omega_m_a) * (1 + omega_l_a / 70))
    )
    f: Array = omega_m_a ** (4 / 7) + omega_l_a / 70 * (1 + omega_m_a / 2)
    return d, f


# ---------------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------------


@attrs.frozen(eq=False)
class GrowthTable:
    """A growth factor and rate tabulated in ln a, normalised to D = 1 at a = 1.

    Both are interpolated with cubic splines (ln D and f in ln a). Build it with
    :func:`tabulate_growth`.
    """

    #: The nodes, in ln a (increasing, ending at 0).
    ln_a: Array
    #: ln D at the nodes (0 at a = 1).
    ln_d: Array
    #: The growth rate at the nodes.
    f: Array
    _ln_d_coefficients: Array
    _f_coefficients: Array

    @property
    def z_max(self) -> float:
        """The largest redshift of the table."""
        return float(np.expm1(-self.ln_a[0]))

    def growth_factor(self, z: Array) -> Array:
        """The normalised growth factor at redshifts ``z`` (within the table)."""
        ln_a = -np.log1p(np.asarray(z, dtype=float))
        out: Array = np.exp(PPoly.construct_fast(self._ln_d_coefficients, self.ln_a)(ln_a))
        return out

    def growth_rate(self, z: Array) -> Array:
        """The growth rate at redshifts ``z`` (within the table)."""
        ln_a = -np.log1p(np.asarray(z, dtype=float))
        out: Array = PPoly.construct_fast(self._f_coefficients, self.ln_a)(ln_a)
        return out


def tabulate_growth(ln_a: Array, d: Array, f: Array | None = None) -> GrowthTable:
    """Build a :class:`GrowthTable`, normalising D to 1 at the last node.

    Parameters
    ----------
    ln_a
        Increasing nodes in ln a; the last one is taken to be a = 1 (z = 0).
    d
        The (unnormalised, positive) growth factor at the nodes.
    f
        The growth rate at the nodes. If not given, it is the derivative of the
        cubic spline of ln D.

    Returns
    -------
    GrowthTable
    """
    ln_a = np.array(ln_a, dtype=float)
    ln_d = np.log(np.asarray(d, dtype=float))
    ln_d = ln_d - ln_d[-1]
    ln_d_spline = CubicSpline(ln_a, ln_d)
    f = ln_d_spline(ln_a, 1) if f is None else np.array(f, dtype=float)
    f_spline = CubicSpline(ln_a, f)
    return GrowthTable(
        ln_a=read_only(ln_a),
        ln_d=read_only(ln_d),
        f=read_only(f),
        ln_d_coefficients=read_only(ln_d_spline.c),
        f_coefficients=read_only(f_spline.c),
    )
