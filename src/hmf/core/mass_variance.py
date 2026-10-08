r"""The :class:`MassVariance` stage: sigma(m) and dln(sigma)/dln(m) on a deterministic mass lattice.

The mass variance of the *unnormalised* linear power at z = 0,

.. math:: \sigma^2(M) = \frac{1}{2\pi^2} \int k^3 P(k)\, W^2(kR)\, d\ln k,

and its logarithmic slope :math:`d\ln\sigma/d\ln M`, as functions of mass. sigma8
normalisation, redshift and peak height come later, in the stages built on this one.

The mass lattice
----------------
Both are computed at the nodes of a lattice in mass, and interpolated between them:

* **Nodes** sit at :math:`\log_{10}(m / (M_\odot/h)) = j\,\Delta` for integer
  :math:`j`, with :math:`\Delta` = ``MassAccuracy.dlog10_m``. Positions are computed as
  ``j * Δ``, so two lattices with the same spacing share their nodes exactly.
* **One fused integral per node** gives :math:`\ln\sigma`,
  :math:`d = d\ln\sigma/d\ln M` and (if ``MassAccuracy.second_derivative``)
  :math:`d_2 = d^2\ln\sigma/d(\ln M)^2`, from one evaluation of the window. The
  quadrature is Simpson's rule along the k axis, so a node's values do not depend on
  which other nodes are computed with it.
* **sigma** is interpolated in :math:`\ln\sigma` against :math:`\ln M`, by a quintic
  Hermite interpolant using :math:`d` and :math:`d_2` (a cubic Hermite without
  :math:`d_2`).
* **dln(sigma)/dln(m) is interpolated separately**, in :math:`\ln|d|`: a cubic Hermite
  interpolant with slope :math:`d_2/d`, or, without :math:`d_2`, the local 4-point
  Lagrange interpolant through the nodes :math:`j-1 \ldots j+2`. (Where :math:`d`
  changes sign across the nodes used, the same schemes are applied to :math:`d`
  itself.) The sigma interpolant is never differentiated: that amplifies quadrature
  noise by :math:`1/\Delta`, and fails where :math:`d \to 0`.
* **Lazy extension.** Nodes are computed when a mass first needs them and memoised.
  The memo is not part of the stage's value or hash, and computing a node is
  idempotent, so results are bit-for-bit independent of the order of requests: a
  lattice built for [10⁸, 10¹⁶] and extended to [10⁰, 10¹⁸] gives the same values as
  one built in the opposite order.

The k grid
----------
The k grid is a lattice too: :math:`\ln k = i\,\delta` for integer :math:`i`, with
:math:`\delta` = ``KAccuracy.dln_k``, from ``KAccuracy.ln_k_min`` up to
:math:`k_{\max} \ge` ``k_max_r_min`` :math:`/R(M_{\min})`. :math:`M_{\min}` is
``MassAccuracy.log10_m_min``, *not* the smallest mass requested: the k grid is fixed
by the settings, so it does not change when the mass lattice extends, which keeps
extension bit-for-bit deterministic.

If the power source extrapolates a table the user supplied (its ``table_range``,
see :mod:`hmf.core.power_source`), and that table does not cover the grid's range
from ``ln_k_min`` to :math:`k_{\max}`, the stage warns once per power source and
end of the table. The ends of the grid are rounded outwards onto lattice nodes, by
less than one step: a table that reaches the unrounded ends (to
:data:`~hmf.core._kernels.lattice.LATTICE_RTOL` steps) covers the grid.

Masses the grid can't resolve
-----------------------------
Every node carries a bound on the error that the truncation of the k integrals at
both ends of the grid causes, in sigma *and* in dln(sigma)/dln(m) (which enters dn/dm, and
is typically kR times more sensitive). A request that needs a node whose bound
exceeds ``truncation_rtol``, or where sigma or its derivatives are not finite, raises a
:class:`~hmf.core.domain.DomainError`.

Every node also carries a Richardson estimate of the error from the resolution of the
k grid (Simpson's rule on the grid against every other grid point). A request that
needs a node where it exceeds :data:`RESOLUTION_RTOL` raises too. This catches gross
failures only: with the top-hat, the grid aliases the window's oscillations at
:math:`kR \gtrsim \pi / (2\,\delta)`, which is harmless for physical spectra
(:math:`P \propto k^{-3}` at high k) but not for ones that fall more slowly than
about :math:`k^{-2}` (e.g. a power law with :math:`n \ge -1`, for which
dln(sigma)/dln(m) only converges conditionally).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import cached_property
from typing import Any, NamedTuple

import attrs
import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

from ._fields import field
from ._kernels import interpolation as interp
from ._kernels import mass_variance as kern
from ._kernels.lattice import LATTICE_RTOL, lattice, lattice_index
from ._validators import check_finite_positive, positive
from .accuracy import KAccuracy, MassAccuracy
from .domain import DomainError, check_extent
from .filters import Filter, TopHat
from .power_source import PowerSource
from .stage import Stage
from .units import Mpc_h, Msun_h, UnitContext, unit_boundary

__all__ = ["MassVariance", "n_eff_kernel"]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_LN10 = math.log(10)

#: Rows of the node table: ln(sigma), d = dln(sigma)/dln(m), d2 = dd/dln(m), bounds on the
#: relative truncation errors of sigma and of dln(sigma)/dln(m), and an estimate of the
#: relative error from the k grid's resolution (the larger of the two).
_LN_SIGMA, _D1, _D2, _ERR_SIGMA, _ERR_D1, _ERR_GRID = range(6)
_N_ROWS = 6

#: The largest estimated relative error from the k grid's resolution accepted for a
#: mass. Exceeding it is a gross failure (the grid aliases the oscillations of the
#: window, for a power spectrum that falls more slowly than about k^-2 at high k), so
#: this is a sanity limit, not an accuracy target.
RESOLUTION_RTOL = 1e-2

#: Nodes are computed in chunks of this many, to bound the memory of the
#: (nodes, k) arrays. Nodes are independent, so the chunking does not change results.
_CHUNK = 64

#: The step in ln k of the central difference for d ln P / d ln k (sharp-k only).
_DLN_K_SLOPE = 1e-3


class _KGrid(NamedTuple):
    """The k lattice and the power on it."""

    i_min: int
    dln_k: float
    ln_k: FloatArray
    k: FloatArray
    k3p: FloatArray
    ln_k3p: FloatArray
    slope: FloatArray
    slope_lo: float
    slope_hi: float


class _NodeCache:
    """The memo of computed lattice nodes: a table indexed by the node number ``j``.

    It is mutable, but only ever *adds* nodes, each computed by a deterministic
    function of ``j`` alone. It is not part of the owning stage's value or hash.
    """

    __slots__ = ("done", "j0", "table")

    def __init__(self) -> None:
        self.j0 = 0
        self.table: FloatArray = np.empty((_N_ROWS, 0))
        self.done: NDArray[np.bool_] = np.zeros(0, dtype=bool)

    def _grow(self, j_lo: int, j_hi: int) -> None:
        """Make the table cover nodes ``j_lo..j_hi``."""
        n = self.done.size
        if n and j_lo >= self.j0 and j_hi < self.j0 + n:
            return
        new_lo = min(j_lo, self.j0) if n else j_lo
        new_hi = max(j_hi, self.j0 + n - 1) if n else j_hi
        table = np.full((_N_ROWS, new_hi - new_lo + 1), np.nan)
        done = np.zeros(new_hi - new_lo + 1, dtype=bool)
        if n:
            table[:, self.j0 - new_lo : self.j0 - new_lo + n] = self.table
            done[self.j0 - new_lo : self.j0 - new_lo + n] = self.done
        self.j0, self.table, self.done = new_lo, table, done

    def get(self, j: IntArray, compute: Callable[[IntArray], FloatArray]) -> FloatArray:
        """The table's columns for nodes ``j``, computing the missing ones first."""
        self._grow(int(j.min()), int(j.max()))
        idx = j - self.j0
        missing = np.unique(j[~self.done[idx]])
        if missing.size:
            self.table[:, missing - self.j0] = compute(missing)
            self.done[missing - self.j0] = True
        return self.table[:, idx]


def _is_power_source(instance: Any, attribute: attrs.Attribute[Any], value: Any) -> None:
    if not isinstance(value, PowerSource):
        raise TypeError(
            f"MassVariance.power must implement hmf.core.power_source.PowerSource "
            f"(e.g. Transfer.power_source() or a TabulatedPower), not {type(value).__name__}."
        )


@attrs.frozen(kw_only=True)
class MassVariance(Stage):
    """The mass variance sigma(m) of the unnormalised linear power at z = 0, and its slope.

    See the module documentation for the mass lattice behind it. The results do not
    depend on sigma8 (the power is unnormalised) or on redshift: those enter later.

    Examples
    --------
    >>> import numpy as np
    >>> from hmf.core.power_source import TabulatedPower
    >>> from hmf.core.units import Msun_h, h_Mpc, power_unit, rho_unit
    >>> k = np.logspace(-5, 3, 400)
    >>> source = TabulatedPower(
    ...     k=k * h_Mpc, pk=1e4 * k / (1 + (k / 0.02) ** 2.5) ** 1.6 * power_unit,
    ...     mean_density=8.5e10 * rho_unit,
    ... )
    >>> mv = MassVariance(power=source, filter="TopHat")
    >>> s = mv.sigma([1e10, 1e12, 1e14] * Msun_h)
    >>> bool(np.all(np.diff(s) < 0))
    True
    """

    power: PowerSource = field(
        validator=_is_power_source,
        doc="The source of the unnormalised linear power at z = 0 (a PowerSource).",
    )
    filter: Filter = field(
        factory=TopHat,
        converter=Filter.coerce,
        doc="The smoothing filter: a Filter instance, or the alias of one (e.g. 'SharpK').",
    )
    mass_accuracy: MassAccuracy = field(
        factory=MassAccuracy,
        validator=attrs.validators.instance_of(MassAccuracy),
        doc="The settings of the mass lattice.",
    )
    k_accuracy: KAccuracy = field(
        factory=KAccuracy,
        validator=attrs.validators.instance_of(KAccuracy),
        doc="The settings of the k grid. Its upper end is set from mass_accuracy.log10_m_min.",
    )
    truncation_rtol: float = field(
        default=1e-3,
        converter=float,
        validator=positive,
        doc=(
            "The largest bound on the relative error in sigma or dln(sigma)/dln(m), from "
            "truncating the k integrals at the ends of the grid, accepted for a mass. "
            "Masses beyond it raise a DomainError."
        ),
    )

    # ---------------------------------------------------------------------------------
    # Setup
    # ---------------------------------------------------------------------------------

    @cached_property
    def _unit_context(self) -> UnitContext:
        """The units context of the power source (with its H0)."""
        return self.power._unit_context

    @cached_property
    def _rho_mean(self) -> float:
        return float(self.power.rho_mean0)

    @cached_property
    def _step(self) -> float:
        """The lattice spacing in ln M."""
        return self.mass_accuracy.dlog10_m * _LN10

    @cached_property
    def _k_grid(self) -> _KGrid:
        """The k lattice, from ln_k_min to k_max = k_max_r_min / R(m_min) or beyond."""
        ka = self.k_accuracy
        dln_k = ka.dln_k
        r_min = float(
            kern.lagrangian_radius(
                10.0**self.mass_accuracy.log10_m_min, self._rho_mean, self.filter.mass_assignment
            )
        )
        ln_k_max = math.log(ka.k_max_r_min / r_min)
        i_min = lattice_index(ka.ln_k_min, dln_k, up=False)
        i_max = lattice_index(ln_k_max, dln_k, up=True)
        if (i_max - i_min) % 2:
            i_max += 1  # an odd number of points, for Simpson's rule
        ln_k = lattice(i_min, i_max, dln_k)
        table = self.power.table_range
        if table is not None:
            # The unrounded ends: rounding them onto the lattice is not extrapolation.
            table.warn_outside(
                self.power,
                math.exp(ka.ln_k_min),
                math.exp(ln_k_max),
                rtol=LATTICE_RTOL * dln_k,
                stacklevel=2,
            )
        k = np.exp(ln_k)
        p = self._power_at(k)
        ln_p = np.log(p)
        return _KGrid(
            i_min=i_min,
            dln_k=dln_k,
            ln_k=ln_k,
            k=k,
            k3p=k**3 * p,
            ln_k3p=3 * ln_k + ln_p,
            slope=np.gradient(ln_p, dln_k),
            slope_lo=float((ln_p[1] - ln_p[0]) / dln_k),
            slope_hi=float((ln_p[-1] - ln_p[-2]) / dln_k),
        )

    def _power_at(self, k: FloatArray) -> FloatArray:
        """The source's power at k (h/Mpc), checked to be finite and positive."""
        p = np.exp(self.power.ln_power_kernel(np.log(k)))
        return check_finite_positive("the power spectrum", p, where="MassVariance")

    @cached_property
    def _sharpk_cumulative(self) -> FloatArray:
        """The integral of k³P d ln k from the first k node to each k node (sharp-k)."""
        ln_k = self._k_grid.ln_k
        points = kern.panel_points(ln_k[:-1], ln_k[1:])
        k = np.exp(points)
        panels = kern.panel_integrals(ln_k[:-1], ln_k[1:], k**3 * self._power_at(k))
        return np.concatenate([[0.0], np.cumsum(panels)])

    @cached_property
    def _nodes(self) -> _NodeCache:
        """The memo of lattice nodes (not part of the stage's value)."""
        return _NodeCache()

    # ---------------------------------------------------------------------------------
    # Nodes
    # ---------------------------------------------------------------------------------

    def _direct(self, ln_m: FloatArray) -> FloatArray:
        """Evaluate the node quantities directly at masses ``exp(ln_m)`` (no interpolation).

        Returns
        -------
        ndarray
            Shape ``(6, n)``: ln(sigma), dln(sigma)/dln(m), its derivative in ln(m) (NaN
            if not computed), the bounds on the relative truncation errors of sigma
            and of dln(sigma)/dln(m), and the estimate of the relative error from the k
            grid's resolution.
        """
        ln_m = np.asarray(ln_m, dtype=float).ravel()
        ln_r = np.log(
            kern.lagrangian_radius(np.exp(ln_m), self._rho_mean, self.filter.mass_assignment)
        )
        out = np.empty((_N_ROWS, ln_m.size))
        for start in range(0, ln_m.size, _CHUNK):
            sl = slice(start, start + _CHUNK)
            if self.filter.sharp_cutoff:
                out[:, sl] = self._sharpk_nodes(ln_r[sl])
            else:
                out[:, sl] = self._window_nodes(ln_r[sl])
        return out

    def _compute_nodes(self, j: IntArray) -> FloatArray:
        """The node quantities at lattice nodes ``j``, i.e. at log10 m = j * Δ."""
        log10_m = j * np.float64(self.mass_accuracy.dlog10_m)
        return self._direct(log10_m * _LN10)

    def _window_nodes(self, ln_r: FloatArray) -> FloatArray:
        """Node quantities for a filter with a smooth window."""
        grid = self._k_grid
        second = self.mass_accuracy.second_derivative
        r = np.exp(ln_r)
        derivs = self.filter.window_derivatives_kernel(
            r[:, None] * grid.k, order=2 if second else 1
        )
        ln_sigma, d_r, d2_r, err_grid = kern.window_ln_variance(
            derivs[0], derivs[1], derivs[2] if second else None, grid.k3p, grid.dln_k
        )
        # Truncation: the tails beyond both ends of the grid.
        tb = self.filter.tail_bounds()
        hi = (grid.k3p[-1], grid.slope_hi, r * grid.k[-1])
        lo = (grid.k3p[0], grid.slope_lo, r * grid.k[0])
        t0 = kern.tail_integral(*hi, tb.high_w2, True, tb.high_w2_oscillating, tb.omega)
        t0 = t0 + kern.tail_integral(*lo, tb.low_w2, False)
        t1 = kern.tail_integral(*hi, tb.high_wdw, True, tb.high_wdw_oscillating, tb.omega)
        t1 = t1 + kern.tail_integral(*lo, tb.low_wdw, False)
        t2 = None
        if second:
            t2 = kern.tail_integral(*hi, tb.high_ddw, True, tb.high_ddw_oscillating, tb.omega)
            t2 = t2 + kern.tail_integral(*lo, tb.low_ddw, False)
        err_sigma, err_d = kern.truncation_errors(ln_sigma, d_r, t0, t1, t2, self._step)
        # R ∝ M^(1/3) for every filter here.
        d2 = d2_r / 9 if d2_r is not None else np.full_like(ln_sigma, np.nan)
        return np.stack([ln_sigma, d_r / 3, d2, err_sigma, err_d, err_grid])

    def _sharpk_nodes(self, ln_r: FloatArray) -> FloatArray:
        """Node quantities for the sharp-k filter: the integral ends exactly at k = 1/R."""
        grid = self._k_grid
        ln_k = grid.ln_k
        ln_cut = -ln_r
        valid = (ln_cut > ln_k[0]) & (ln_cut <= ln_k[-1])
        ln_cut_ok = np.where(valid, ln_cut, ln_k[1])
        # The k node at or below the cut-off, and the partial panel from it to the cut-off.
        idx = np.floor(ln_cut_ok / grid.dln_k).astype(np.int64) - grid.i_min
        idx = np.clip(idx, 0, ln_k.size - 2)
        ln_a = ln_k[idx]
        k_pts = np.exp(kern.panel_points(ln_a, ln_cut_ok))
        partial = kern.panel_integrals(ln_a, ln_cut_ok, k_pts**3 * self._power_at(k_pts))
        s = (self._sharpk_cumulative[idx] + partial) / (2 * math.pi**2)
        k_cut = np.exp(ln_cut_ok)
        p_cut = self._power_at(k_cut)
        second = self.mass_accuracy.second_derivative
        if second:
            h = _DLN_K_SLOPE
            p_up = self._power_at(k_cut * math.exp(h))
            p_dn = self._power_at(k_cut * math.exp(-h))
            dln_p_dln_k_cut = (np.log(p_up) - np.log(p_dn)) / (2 * h)
        else:
            dln_p_dln_k_cut = np.zeros_like(p_cut)
        ln_sigma, d_r, d2_r = kern.sharpk_ln_variance(s, k_cut**3 * p_cut, dln_p_dln_k_cut, second)
        # Beyond the cut-off the window is 0; below the grid, W = 1 and W' = 0.
        x_lo = np.exp(ln_r) * grid.k[0]
        t0 = kern.tail_integral(grid.k3p[0], grid.slope_lo, x_lo, ((1.0, 0.0),), high=False)
        err_sigma, err_d = kern.truncation_errors(ln_sigma, d_r, t0, np.zeros_like(t0))
        d2 = d2_r / 9 if d2_r is not None else np.full_like(ln_sigma, np.nan)
        out = np.stack([ln_sigma, d_r / 3, d2, err_sigma, err_d, np.zeros_like(err_d)])
        out[:, ~valid] = np.nan
        out[_ERR_SIGMA:_ERR_GRID, ~valid] = np.inf
        out[_ERR_GRID, ~valid] = 0.0
        return out

    # ---------------------------------------------------------------------------------
    # Interpolation
    # ---------------------------------------------------------------------------------

    def _check_nodes(self, nodes: FloatArray, log10_m: FloatArray) -> None:
        """Raise if a node used for a mass is not finite or is not resolved by the k grid."""
        rows = [_LN_SIGMA, _D1, _D2] if self.mass_accuracy.second_derivative else [_LN_SIGMA, _D1]
        bad_value = ~np.all(np.isfinite(nodes[rows]), axis=0)
        err = np.fmax(nodes[_ERR_SIGMA], nodes[_ERR_D1])
        err_grid = nodes[_ERR_GRID]
        bad = bad_value | ~(err <= self.truncation_rtol) | ~(err_grid <= RESOLUTION_RTOL)
        if not np.any(bad):
            return
        i = int(np.argmax(bad))
        m = 10.0 ** log10_m[i]
        r = float(kern.lagrangian_radius(m, self._rho_mean, self.filter.mass_assignment))
        grid = self._k_grid
        k_lo, k_hi = grid.k[0], grid.k[-1]
        where = (
            f"The k grid spans [{k_lo:.3g}, {k_hi:.3g}] h/Mpc, with dln_k = {grid.dln_k:g}, "
            f"and R = {r:.3g} Mpc/h."
        )
        if not bad_value[i] and not err_grid[i] <= RESOLUTION_RTOL:
            raise DomainError(
                f"MassVariance: can't evaluate m = {m:.4g} Msun/h: the k grid does not resolve "
                f"the integrals there (estimated error {err_grid[i]:.2g} > {RESOLUTION_RTOL:g}). "
                f"Decrease k_accuracy.dln_k. (With the {type(self.filter).__name__} filter, a "
                f"power spectrum that falls more slowly than about k^-2 at high k, here "
                f"dlnP/dlnk = {grid.slope_hi:.3g} at k_max, makes the grid alias the window's "
                f"oscillations: physical spectra fall like k^-3.) {where}"
            )
        side = "high" if r * k_lo > 1 / (r * k_hi) else "low"
        hint = (
            "lower k_accuracy.ln_k_min"
            if side == "high"
            else (
                "lower mass_accuracy.log10_m_min (which sets k_max), or raise "
                "k_accuracy.k_max_r_min"
            )
        )
        what = (
            "sigma or its derivatives are not finite there (they underflow, or the k grid "
            "does not reach it)"
            if bad_value[i]
            else f"the bound on the error from truncating the k grid is {err[i]:.2g}, more "
            f"than truncation_rtol = {self.truncation_rtol:g}"
        )
        raise DomainError(
            f"MassVariance: can't evaluate m = {m:.4g} Msun/h, at the {side}-mass end: "
            f"{what}. {where} To reach this mass, {hint}."
        )

    def _interpolate(self, log10_m: FloatArray) -> tuple[FloatArray, FloatArray]:
        """The ln(sigma) and dln(sigma)/dln(m) at masses 10**log10_m (1D), from the lattice."""
        acc = self.mass_accuracy
        if acc.extension == "raise":
            outside = (log10_m < acc.log10_m_min - 1e-12) | (log10_m > acc.log10_m_max + 1e-12)
            if np.any(outside):
                raise DomainError(
                    f"MassVariance: masses outside the lattice [1e{acc.log10_m_min:g}, "
                    f"1e{acc.log10_m_max:g}] Msun/h, with mass_accuracy.extension='raise'."
                )
        t = log10_m / acc.dlog10_m
        # The interval [j, j + 1] that contains each mass, so 0 <= u < 1. A plain floor,
        # not lattice_index: snapping a mass near a node onto it would move it to the
        # next interval (u slightly < 0) and, with the 4-point stencil, change the nodes
        # used, which changes the result.
        j = np.floor(t).astype(np.int64)
        u = t - j
        second = acc.second_derivative
        offsets = np.array([0, 1]) if second else np.array([-1, 0, 1, 2])
        js = j[None, :] + offsets[:, None]
        nodes = self._nodes.get(js.ravel(), self._compute_nodes).reshape(_N_ROWS, *js.shape)
        self._check_nodes(nodes.reshape(_N_ROWS, -1), np.tile(log10_m, offsets.size))
        h = self._step
        a, b = (0, 1) if second else (1, 2)  # the rows of nodes j and j + 1
        y0, y1 = nodes[_LN_SIGMA, a], nodes[_LN_SIGMA, b]
        d0, d1 = nodes[_D1, a], nodes[_D1, b]
        if second:
            c0, c1 = nodes[_D2, a], nodes[_D2, b]
            ln_sigma = interp.hermite_quintic(u, h, y0, y1, d0, d1, c0, c1)
            same_sign = (d0 * d1) > 0
            with np.errstate(divide="ignore", invalid="ignore"):
                ln_abs = interp.hermite_cubic(
                    u, h, np.log(np.abs(d0)), np.log(np.abs(d1)), c0 / d0, c1 / d1
                )
            linear = interp.hermite_cubic(u, h, d0, d1, c0, c1)
            slope = np.where(same_sign, np.sign(d0) * np.exp(ln_abs), linear)
        else:
            ln_sigma = interp.hermite_cubic(u, h, y0, y1, d0, d1)
            d = nodes[_D1]
            same_sign = np.all(d * d[1] > 0, axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                ln_abs = interp.lagrange4(u, *np.log(np.abs(d)))
            linear = interp.lagrange4(u, *d)
            slope = np.where(same_sign, np.sign(d0) * np.exp(ln_abs), linear)
        return ln_sigma, slope

    def _log10_mass(self, m: FloatArray) -> FloatArray:
        """log10 of masses in Msun/h, checked to be finite and positive."""
        return np.log10(check_finite_positive("masses", m, where="MassVariance", error=DomainError))

    def ln_sigma_and_slope_kernel(self, m: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
        """ln(sigma) and dln(sigma)/dln(m) at masses in Msun/h, at kernel level.

        The values of :meth:`sigma` (as its ln) and :meth:`dlnsigma_dlnm`, from one
        lookup of the lattice, on plain arrays in canonical units, for library code
        (see :mod:`hmf.core._kernels`). Results do not depend on the batch size, or on
        the order of requests.

        Parameters
        ----------
        m
            Masses in Msun/h: a plain array, of any shape.

        Returns
        -------
        ln_sigma, dlnsigma_dlnm : numpy.ndarray
            Dimensionless, with the shape of ``m``.

        Raises
        ------
        DomainError
            As for :meth:`sigma`.
        """
        m = np.asarray(m, dtype=float)
        ln_sigma, slope = self._interpolate(self._log10_mass(m).ravel())
        return ln_sigma.reshape(m.shape), slope.reshape(m.shape)

    # ---------------------------------------------------------------------------------
    # Public methods
    # ---------------------------------------------------------------------------------

    @unit_boundary(m=Msun_h)
    def sigma(self, m: Any) -> FloatArray:
        """The mass variance sigma(m) of the unnormalised linear power at z = 0.

        Parameters
        ----------
        m : Quantity
            Masses, in Msun/h (or Msun, if the power source has an H0).

        Returns
        -------
        numpy.float64 or ndarray
            sigma, dimensionless, of the shape of ``m`` (a scalar for a scalar ``m``).

        Raises
        ------
        DomainError
            If a mass is not finite and > 0, is outside the lattice (with
            ``extension='raise'``), can't be resolved by the k grid, or sigma is not
            finite there.
        """
        return np.exp(self.ln_sigma_and_slope_kernel(m)[0])

    @unit_boundary(m=Msun_h)
    def dlnsigma_dlnm(self, m: Any) -> FloatArray:
        """The logarithmic slope dln(sigma)/dln(m) (independent of the normalisation of the power).

        Parameters
        ----------
        m : Quantity
            Masses, in Msun/h (or Msun, if the power source has an H0).

        Returns
        -------
        numpy.float64 or ndarray
            dln(sigma)/dln(m), dimensionless, of the shape of ``m`` (a scalar for a
            scalar ``m``).

        Raises
        ------
        DomainError
            As for :meth:`sigma`.
        """
        return self.ln_sigma_and_slope_kernel(m)[1]

    @unit_boundary(returns=Msun_h)
    def m_from_sigma(self, sigma: Any) -> FloatArray:
        """The mass at which sigma(m) takes the given values: the inverse of :meth:`sigma`.

        sigma(m) is inverted on the lattice nodes of the default range
        ``[mass_accuracy.log10_m_min, mass_accuracy.log10_m_max]``, then within the bracketing
        interval by bisection on the sigma interpolant, so ``m_from_sigma(sigma(m))``
        returns ``m`` to about 1e-12.

        Parameters
        ----------
        sigma : float or array_like
            Values of sigma (dimensionless, unnormalised as in :meth:`sigma`).

        Returns
        -------
        Quantity
            Masses, in Msun/h.

        Raises
        ------
        DomainError
            If a value is not finite and > 0, is outside the range of sigma on the
            default lattice range, or sigma(m) is not monotonically decreasing where
            the value is crossed.
        """
        target = check_finite_positive(
            "sigma", sigma, where="MassVariance.m_from_sigma", error=DomainError
        )
        ln_t = np.log(target).ravel()
        acc = self.mass_accuracy
        j_lo = lattice_index(acc.log10_m_min, acc.dlog10_m, up=True)
        j_hi = lattice_index(acc.log10_m_max, acc.dlog10_m, up=False)
        j = np.arange(j_lo, j_hi + 1, dtype=np.int64)
        nodes = self._nodes.get(j, self._compute_nodes)
        ln_s, d = nodes[_LN_SIGMA], nodes[_D1]
        if np.any((ln_t > ln_s.max()) | (ln_t < ln_s.min())):
            raise DomainError(
                f"MassVariance.m_from_sigma: sigma outside [{math.exp(ln_s.min()):.4g}, "
                f"{math.exp(ln_s.max()):.4g}], its range on the lattice "
                f"[1e{acc.log10_m_min:g}, 1e{acc.log10_m_max:g}] Msun/h."
            )
        # Each interval [j, j + 1] that sigma crosses downwards (ln_s[j] >= target >
        # ln_s[j + 1]) or upwards: there must be exactly one, downwards.
        ge = ln_s[None, :] >= ln_t[:, None]
        down = ge[:, :-1] & ~ge[:, 1:]
        down[ln_t == ln_s[-1], -1] = True  # the target is the last node itself
        up = ~ge[:, :-1] & ge[:, 1:]
        interval = np.argmax(down, axis=1)
        unique = (np.sum(down, axis=1) == 1) & ~np.any(up, axis=1)
        monotonic = (d[interval] < 0) & (d[interval + 1] < 0)
        bad = ~(unique & monotonic)
        if np.any(bad):
            raise DomainError(
                "MassVariance.m_from_sigma: sigma(m) is not monotonically decreasing "
                f"where it crosses sigma = {math.exp(ln_t[np.argmax(bad)]):.6g}, so the "
                "inverse is not unique."
            )
        self._check_nodes(
            nodes[:, np.concatenate([interval, interval + 1])],
            np.tile(j[interval] * acc.dlog10_m, 2),
        )
        a, b = interval, interval + 1
        cs = (nodes[_D2, a], nodes[_D2, b]) if acc.second_derivative else (None, None)
        u = interp.invert_hermite(ln_t, self._step, ln_s[a], ln_s[b], d[a], d[b], *cs)
        m: FloatArray = 10.0 ** ((j[a] + u) * acc.dlog10_m)
        return m.reshape(target.shape)

    @unit_boundary(r=Mpc_h, returns=Msun_h)
    def m_from_radius(self, r: Any) -> FloatArray:
        """The mass of a filter radius: m = (4π/3) rho_mean (cR)³, with the filter's c.

        Parameters
        ----------
        r : Quantity
            Radii, in Mpc/h (or Mpc, if the power source has an H0).

        Returns
        -------
        Quantity
            Masses, in Msun/h.
        """
        return kern.lagrangian_mass(r, self._rho_mean, self.filter.mass_assignment)

    @unit_boundary(m=Msun_h, returns=Mpc_h)
    def radius_from_m(self, m: Any) -> FloatArray:
        """The filter radius of a mass: the inverse of :meth:`m_from_radius`.

        Parameters
        ----------
        m : Quantity
            Masses, in Msun/h (or Msun, if the power source has an H0).

        Returns
        -------
        Quantity
            Radii, in Mpc/h.
        """
        return kern.lagrangian_radius(m, self._rho_mean, self.filter.mass_assignment)


def n_eff_kernel(dlnsigma_dlnm: npt.ArrayLike) -> FloatArray:
    r"""The effective spectral index at a mass, from the slope of sigma(m).

    .. math:: n_{\rm eff} = -3\left(2\frac{d\ln\sigma}{d\ln m} + 1\right),

    the index n of the power law :math:`P \propto k^n` that has the same slope of
    sigma at this mass: for a power law, :math:`\sigma \propto m^{-(n+3)/6}`, so
    :math:`n_{\rm eff} = n` exactly. It is the ``n_eff`` input of the fits in
    :mod:`hmf.core.fits`. A pure, elementwise kernel on plain arrays (see
    :mod:`hmf.core._kernels`).

    Parameters
    ----------
    dlnsigma_dlnm
        :math:`d\ln\sigma/d\ln m`, dimensionless (e.g. the second result of
        :meth:`MassVariance.ln_sigma_and_slope_kernel`).

    Returns
    -------
    numpy.ndarray
        :math:`n_{\rm eff}`, dimensionless, with the shape of ``dlnsigma_dlnm``.

    Raises
    ------
    DomainError
        If a value is not finite.
    """
    slope = check_extent("dlnsigma_dlnm", dlnsigma_dlnm, where="n_eff_kernel")
    out: FloatArray = -3.0 * (2.0 * slope + 1.0)
    return out
