r"""Sources of the linear matter power spectrum at z = 0.

:class:`~hmf.core.mass_variance.MassVariance` needs only three things from the
power spectrum, which make up the :class:`PowerSource` protocol. The first two are
kernel-level (see :mod:`hmf.core._kernels`): plain floats and arrays in canonical
units, for library code.

* ``ln_power_kernel(ln_k)``: the natural log of the linear power spectrum of one
  matter species at z = 0, with k in h/Mpc. Its amplitude is arbitrary: the
  normalisation to sigma_8 is applied later, as a scalar, by the ``LinearPower``
  stage, so only the shape matters and it has no fixed unit. A
  :class:`TabulatedPower` gives its table in (Mpc/h)³, while
  :meth:`Transfer.power_kernel <hmf.core.transfer.Transfer.power_kernel>`'s
  :math:`k^{n_s} T(k)^2` is dimensionless;
* ``rho_mean0``: the mean comoving density today of the *same* matter species, in
  M☉ h² / Mpc³ (:data:`~hmf.core.units.rho_unit`);
* ``_unit_context``: the :class:`~hmf.core.units.UnitContext` (with its H0) used by
  the units boundary of the stages built on it (see
  :class:`~hmf.core.units.HasUnitContext`).

:meth:`Transfer.power_kernel <hmf.core.transfer.Transfer.power_kernel>` implements it
for the power of a :class:`~hmf.core.transfer.Transfer` stage, and
:class:`TabulatedPower` for a power spectrum given as a table.
"""

from __future__ import annotations

import math
from functools import cached_property
from typing import Any, Protocol, get_args, runtime_checkable

import astropy.units as u
import attrs
import numpy as np
from numpy.typing import NDArray

from ._fields import field
from ._kernels.arrays import read_only
from ._kernels.interpolation import FrozenSpline, extrapolate_power_law
from ._serialise import quantity_key
from ._validators import check_finite_positive, check_in_range, check_increasing, check_table
from .accuracy import Extension
from .domain import DomainError, warn_once
from .stage import Stage
from .units import (
    HasUnitContext,
    UnitContext,
    _require_quantity,
    h_Mpc,
    power_unit,
    rho_unit,
    to_canonical,
    unit_boundary,
)

__all__ = ["PowerSource", "TabulatedPower"]


@runtime_checkable
class PowerSource(HasUnitContext, Protocol):
    """A source of the linear matter power spectrum of one species at z = 0.

    See the module documentation. Implementations must be immutable and hashable
    (they are fields of frozen stages), and ``ln_power_kernel`` must be a pure,
    elementwise function, so that results do not depend on the batch size. The
    ``_unit_context`` (with the H0) of stages built on the source comes from
    :class:`~hmf.core.units.HasUnitContext`.
    """

    def ln_power_kernel(self, ln_k: NDArray[np.float64]) -> NDArray[np.float64]:
        """The ln of the power at z = 0 (arbitrary amplitude), at ln(k / (h/Mpc)).

        Parameters
        ----------
        ln_k
            ln k, with k in h/Mpc: a plain array.

        Returns
        -------
        numpy.ndarray
            ln P, with the shape of ``ln_k``.
        """
        ...

    @property
    def rho_mean0(self) -> float:
        """The mean comoving density today of the species, in Msun h^2 / Mpc^3."""
        ...


def _quantity(name: str, unit: u.UnitBase) -> Any:
    """A validator that the value is a Quantity (it is converted to ``unit`` later)."""

    def validate(instance: Any, attribute: attrs.Attribute[Any], value: Any) -> None:
        _require_quantity(value, unit, where="TabulatedPower", name=name)

    return validate


def _table_column(name: str, unit: u.UnitBase) -> Any:
    """A converter to a read-only float copy of a Quantity (bare values raise).

    The column keeps its unit: it is converted to ``unit`` once the H0 is known.
    """

    def convert(value: Any) -> u.Quantity:
        out: u.Quantity = read_only(
            _require_quantity(value, unit, where="TabulatedPower", name=name)
        )
        return out

    return convert


@attrs.frozen(kw_only=True)
class TabulatedPower(Stage):
    r"""A linear power spectrum at z = 0, given as a table.

    The power is interpolated with a cubic spline in :math:`\ln P` against
    :math:`\ln k`. Beyond the ends of the table it is extrapolated as a power law,
    with the slope :math:`d\ln P/d\ln k` of the spline at each end, if
    ``extension`` is ``"auto"`` (the default), with an
    :class:`~hmf.exceptions.HMFExtrapolationWarning` (once per instance and end of
    the table); with ``extension="raise"``, k outside the table raises a
    :class:`~hmf.core.domain.DomainError`. The internal k grid of
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
        eq=quantity_key,
        converter=_table_column("k", h_Mpc),
        doc="The wavenumbers of the table: 1D, strictly increasing, > 0 (h/Mpc or 1/Mpc).",
    )
    pk: u.Quantity = field(
        eq=quantity_key,
        converter=_table_column("pk", power_unit),
        doc="The power at k: finite and > 0 ((Mpc/h)^3 or Mpc^3).",
    )
    mean_density: u.Quantity = field(
        eq=quantity_key,
        validator=_quantity("mean_density", rho_unit),
        doc="The mean matter density today (Msun h^2 / Mpc^3, or Msun / Mpc^3).",
    )
    H0: u.Quantity | None = field(
        default=None,
        eq=quantity_key,
        doc=(
            "The Hubble constant, used to convert inputs given in physical units. "
            "If None, every dimensional input must be in h-units."
        ),
    )
    extension: Extension = field(
        default="auto",
        validator=attrs.validators.in_(get_args(Extension)),
        doc=(
            "What to do with k outside the table: 'auto' extrapolates the power as a "
            "power law (with an HMFExtrapolationWarning, once per instance), 'raise' "
            "raises a DomainError."
        ),
    )

    def __attrs_post_init__(self) -> None:
        """Validate the H0, the table, and its units (with the H0)."""
        _ = self._unit_context  # UnitContext validates the H0.
        where = "TabulatedPower"
        k, pk = check_table({"k": self.k.value, "pk": self.pk.value}, where=where, min_size=4)
        # Converting to canonical units multiplies by a factor > 0, which keeps these.
        check_finite_positive("k", k, where=where)
        check_increasing("k", k, where=where)
        check_finite_positive("pk", pk, where=where)
        _ = self._table  # Converts the table now, so unit errors raise here.
        check_finite_positive("mean_density", self.rho_mean0, where=where)

    @cached_property
    def _unit_context(self) -> UnitContext:
        """The units context, with this source's H0 (which it validates)."""
        return UnitContext(self.H0, where="TabulatedPower")

    def _canonical(self, q: u.Quantity, unit: u.UnitBase, name: str) -> NDArray[np.float64]:
        """The values of a field in the canonical unit."""
        values = to_canonical(
            q, unit, context=self._unit_context, where="TabulatedPower", name=name
        )
        return np.asarray(values, dtype=float)

    @cached_property
    def _table(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The ln k and ln P of the table, in canonical units."""
        k = self._canonical(self.k, h_Mpc, "k")
        p = self._canonical(self.pk, power_unit, "pk")
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(k), np.log(p)

    @cached_property
    def _spline(self) -> FrozenSpline:
        """The spline of ln P against ln k."""
        return FrozenSpline.fit(*self._table)

    @cached_property
    def rho_mean0(self) -> float:
        """``mean_density`` in Msun h^2 / Mpc^3, as a plain float (kernel level)."""
        return float(self._canonical(self.mean_density, rho_unit, "mean_density"))

    def ln_power_kernel(self, ln_k: NDArray[np.float64]) -> NDArray[np.float64]:
        """The ln of the power, in (Mpc/h)³, at ln(k / (h/Mpc)) (:class:`PowerSource`).

        Parameters
        ----------
        ln_k
            ln k, with k in h/Mpc: a plain array.

        Returns
        -------
        numpy.ndarray
            ln(P / (Mpc/h)³), with the shape of ``ln_k``.

        Raises
        ------
        DomainError
            If ``ln_k`` is outside the table and ``extension`` is ``"raise"``.

        Warns
        -----
        HMFExtrapolationWarning
            If ``ln_k`` is outside the table and ``extension`` is ``"auto"``: once per
            instance for each end of the table.
        """
        ln_k = np.asarray(ln_k, dtype=float)
        table_k = self._table[0]
        lo, hi = table_k[0], table_k[-1]
        below, above = ln_k < lo, ln_k > hi
        for side, outside in (("below", below), ("above", above)):
            if not np.any(outside):
                continue
            message = (
                f"TabulatedPower: {np.count_nonzero(outside)} value(s) of k {side} the table "
                f"[{math.exp(lo):.4g}, {math.exp(hi):.4g}] h/Mpc"
            )
            if self.extension == "raise":
                raise DomainError(message + ", with extension='raise'.")
            warn_once(
                self,
                f"k {side} the table",
                message + " are extrapolated as a power law. Give a wider table to avoid "
                "this, or extension='raise' to make it an error.",
            )
        return np.asarray(extrapolate_power_law(self._spline, ln_k), dtype=float)

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

        Raises
        ------
        DomainError
            If a k is not > 0, or is outside the table with ``extension="raise"``.
        """
        check_in_range("k", k, where="TabulatedPower", low=0.0, low_open=True, error=DomainError)
        return np.exp(self.ln_power_kernel(np.log(np.asarray(k, dtype=float))))
