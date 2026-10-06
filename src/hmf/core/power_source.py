r"""Sources of the linear matter power spectrum at z = 0.

:class:`~hmf.core.mass_variance.MassVariance` needs only three things from the
power spectrum, which make up the :class:`PowerSource` protocol:

* ``_power(k)``: the *unnormalised* linear power spectrum at z = 0, as a kernel-level
  function of plain arrays in canonical units (k in h/Mpc, P in (Mpc/h)³). The
  normalisation (sigma8) is applied later, as a scalar;
* ``_rho_mean0``: the mean matter density today, in M☉ h² / Mpc³;
* ``_unit_context``: the :class:`~hmf.core.units.UnitContext` (with its H0) used by
  the units boundary of the stages built on it.

The members are private because they are kernel-level (plain arrays, not Quantities):
library code uses them, users do not. The ``LinearPower`` stage of the v4 core will
implement this protocol. :class:`TabulatedPower` implements it for a power spectrum
given as a table.
"""

from __future__ import annotations

import math
from functools import cached_property
from typing import Any, Protocol, runtime_checkable

import astropy.units as u
import attrs
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from ._fields import field
from ._validators import check_finite_positive, check_increasing, check_table
from .stage import Stage
from .units import (
    H0_unit,
    UnitBoundaryError,
    UnitContext,
    h_Mpc,
    power_unit,
    rho_unit,
    unit_boundary,
)

__all__ = ["PowerSource", "TabulatedPower"]


@runtime_checkable
class PowerSource(Protocol):
    """A source of the unnormalised linear matter power spectrum at z = 0.

    See the module documentation. Implementations must be immutable and hashable
    (they are fields of frozen stages), and ``_power`` must be a pure, elementwise
    function, so that results do not depend on the batch size.
    """

    def _power(self, k: NDArray[np.float64]) -> NDArray[np.float64]:
        """The unnormalised linear power at z = 0, in (Mpc/h)³, at k in h/Mpc."""
        ...

    @property
    def _rho_mean0(self) -> float:
        """The mean matter density today, in M☉ h² / Mpc³."""
        ...

    @property
    def _unit_context(self) -> UnitContext:
        """The units context (with the H0) of stages built on this source."""
        ...


def _quantity_eq(a: Any, b: Any) -> bool:
    """Equality of two Quantities (or None): same unit and values."""
    if a is None or b is None:
        return a is b
    return bool(a.unit == b.unit and a.shape == b.shape and np.array_equal(a.value, b.value))


_qeq = attrs.cmp_using(eq=_quantity_eq)


def _is_quantity(name: str) -> Any:
    """A validator that the value is a Quantity."""

    def validate(instance: Any, attribute: attrs.Attribute[Any], value: Any) -> None:
        if not isinstance(value, u.Quantity):
            raise UnitBoundaryError(
                f"TabulatedPower: {name!r} must be an astropy Quantity, not a bare "
                f"{type(value).__name__}."
            )

    return validate


def _table_column(name: str) -> Any:
    """A converter to a read-only float copy of a Quantity (bare values raise)."""

    def convert(value: Any) -> u.Quantity:
        if not isinstance(value, u.Quantity):
            raise UnitBoundaryError(
                f"TabulatedPower: {name!r} must be an astropy Quantity, not a bare "
                f"{type(value).__name__}."
            )
        out = u.Quantity(value, dtype=float, copy=True)
        out.flags.writeable = False
        return out

    return convert


@attrs.frozen(kw_only=True)
class TabulatedPower(Stage):
    r"""A linear power spectrum at z = 0, given as a table.

    The power is interpolated with a cubic spline in :math:`\ln P` against
    :math:`\ln k`. Beyond the ends of the table it is extrapolated as a power law,
    with the slope :math:`d\ln P/d\ln k` of the spline at each end, if
    ``extrapolate`` is True, and is an error otherwise. The internal k grid of
    :class:`~hmf.core.mass_variance.MassVariance` spans
    :math:`10^{-8}` to :math:`\sim 10^5` h/Mpc by default, so a typical table *will*
    be extrapolated: give a wide table (or a model whose asymptotic slopes are
    right) when the extremes matter.

    The power need not be normalised: sigma8 normalisation is applied later.

    Examples
    --------
    >>> import numpy as np
    >>> from hmf.core.units import h_Mpc, power_unit, rho_unit
    >>> k = np.logspace(-4, 2, 200)
    >>> source = TabulatedPower(
    ...     k=k * h_Mpc, pk=1e4 * k / (1 + (k / 0.02) ** 2.5) ** 1.6 * power_unit,
    ...     mean_density=8.5e10 * rho_unit,
    ... )
    """

    k: u.Quantity = field(
        eq=_qeq,
        hash=False,
        converter=_table_column("k"),
        doc="The wavenumbers of the table: 1D, strictly increasing, > 0 (h/Mpc or 1/Mpc).",
    )
    pk: u.Quantity = field(
        eq=_qeq,
        hash=False,
        converter=_table_column("pk"),
        doc="The power at k: finite and > 0 ((Mpc/h)^3 or Mpc^3).",
    )
    mean_density: u.Quantity = field(
        eq=_qeq,
        hash=False,
        validator=_is_quantity("mean_density"),
        doc="The mean matter density today (Msun h^2 / Mpc^3, or Msun / Mpc^3).",
    )
    H0: u.Quantity | None = field(
        default=None,
        eq=_qeq,
        hash=False,
        doc=(
            "The Hubble constant, used to convert inputs given in physical units. "
            "If None, every dimensional input must be in h-units."
        ),
    )
    extrapolate: bool = field(
        default=True,
        validator=attrs.validators.instance_of(bool),
        doc="Whether to extrapolate the power beyond the table as a power law.",
    )

    def __attrs_post_init__(self) -> None:
        """Validate the table, and its units (with the H0)."""
        if self.H0 is not None and not isinstance(self.H0, u.Quantity):
            raise UnitBoundaryError("TabulatedPower: H0 must be a Quantity, e.g. 70 * H0_unit.")
        if self.H0 is not None and not self.H0.unit.is_equivalent(H0_unit):
            raise u.UnitConversionError(f"TabulatedPower: H0 must be in {H0_unit}.")
        where = "TabulatedPower"
        k, pk = check_table({"k": self.k.value, "pk": self.pk.value}, where=where, min_size=4)
        # Converting to canonical units multiplies by a factor > 0, which keeps these.
        check_finite_positive("k", k, where=where)
        check_increasing("k", k, where=where)
        check_finite_positive("pk", pk, where=where)
        self._table
        check_finite_positive("mean_density", self._rho_mean0, where=where)

    @cached_property
    def _unit_context(self) -> UnitContext:
        """The units context, with this source's H0."""
        return UnitContext(self.H0)

    def _canonical(self, q: u.Quantity, unit: u.UnitBase, name: str) -> NDArray[np.float64]:
        """The values of a field in the canonical unit."""
        try:
            factor = self._unit_context.factor(q.unit, unit)
        except u.UnitsError as e:
            raise u.UnitConversionError(
                f"TabulatedPower: {name!r} must be in units convertible to '{unit}': {e}"
            ) from e
        return np.asarray(q.value * factor, dtype=float)

    @cached_property
    def _table(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The ln k and ln P of the table, in canonical units."""
        k = self._canonical(self.k, h_Mpc, "k")
        p = self._canonical(self.pk, power_unit, "pk")
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(k), np.log(p)

    @cached_property
    def _spline(self) -> CubicSpline:
        """The spline of ln P against ln k."""
        return CubicSpline(*self._table)

    @cached_property
    def _rho_mean0(self) -> float:
        """The mean matter density today, in M☉ h² / Mpc³."""
        return float(self._canonical(self.mean_density, rho_unit, "mean_density"))

    def _power(self, k: NDArray[np.float64]) -> NDArray[np.float64]:
        """The power at k (h/Mpc), in (Mpc/h)³: the kernel-level :class:`PowerSource` method.

        Raises
        ------
        ValueError
            If ``k`` is outside the table and ``extrapolate`` is False.
        """
        ln_k = np.log(np.asarray(k, dtype=float))
        table_k = self._table[0]
        lo, hi = table_k[0], table_k[-1]
        below, above = ln_k < lo, ln_k > hi
        if not self.extrapolate and (np.any(below) or np.any(above)):
            raise ValueError(
                f"TabulatedPower: k outside the table [{math.exp(lo):.4g}, {math.exp(hi):.4g}] "
                "h/Mpc, and extrapolate=False."
            )
        spline = self._spline
        inside = np.clip(ln_k, lo, hi)
        ln_p = spline(inside)
        slope_lo, slope_hi = spline(lo, 1), spline(hi, 1)
        ln_p = np.where(below, ln_p + slope_lo * (ln_k - lo), ln_p)
        ln_p = np.where(above, ln_p + slope_hi * (ln_k - hi), ln_p)
        return np.asarray(np.exp(ln_p), dtype=float)

    @unit_boundary(k=h_Mpc, returns=power_unit)
    def power(self, k: Any) -> NDArray[np.float64]:
        """The (unnormalised) linear power at z = 0.

        Parameters
        ----------
        k : Quantity
            Wavenumbers, in h/Mpc (or 1/Mpc, with an H0).

        Returns
        -------
        Quantity
            The power, in (Mpc/h)³.
        """
        return self._power(k)
