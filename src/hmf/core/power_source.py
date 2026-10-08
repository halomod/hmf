r"""Sources of the linear matter power spectrum at z = 0.

:class:`~hmf.core.mass_variance.MassVariance` needs only four things from the
power spectrum, which make up the :class:`PowerSource` protocol. The first three are
kernel-level (see :mod:`hmf.core._kernels`): plain floats and arrays in canonical
units, for library code.

* ``ln_power_kernel(ln_k)``: the natural log of the linear power spectrum of one
  matter species at z = 0, with k in h/Mpc. Its amplitude is arbitrary: the
  normalisation to sigma_8 is a scalar applied later, by a stage built on
  MassVariance (a ``LinearPower`` stage, which does not exist yet), so only the
  shape matters and it has no fixed unit. A
  :class:`TabulatedPower` gives its table in (Mpc/h)³, while
  :meth:`Transfer.power_kernel <hmf.core.transfer.Transfer.power_kernel>`'s
  :math:`k^{n_s} T(k)^2` is dimensionless;
* ``rho_mean0``: the mean comoving density today of the *same* matter species, in
  M☉ h² / Mpc³ (:data:`~hmf.core.units.rho_unit`);
* ``table_range``: the :class:`TableRange` of a table the user supplied, beyond
  which ``ln_power_kernel`` extrapolates, or ``None`` if it extrapolates no user
  table (a fitting formula, a Boltzmann code's table, or a table that is not
  extrapolated). ``ln_power_kernel`` does not warn: the stage that evaluates it
  warns, with :meth:`TableRange.warn_outside`;
* ``_unit_context``: the :class:`~hmf.core.units.UnitContext` (with its H0) used by
  the units boundary of the stages built on it (see
  :class:`~hmf.core.units.HasUnitContext`).

:meth:`Transfer.power_kernel <hmf.core.transfer.Transfer.power_kernel>` implements it
for the power of a :class:`~hmf.core.transfer.Transfer` stage, and
:class:`TabulatedPower` for a power spectrum given as a table.

Extrapolation warnings
----------------------
Every path that evaluates a user-supplied table outside its range warns through
one mechanism, :meth:`TableRange.warn_outside`: an
:class:`~hmf.exceptions.HMFExtrapolationWarning` once per owner (with
:func:`~hmf.core.domain.warn_once`) and end of the table. The owner is the object
the table belongs to: the :class:`~hmf.core.transfer.Transfer` stage for its public
methods, and the power source for :class:`~hmf.core.mass_variance.MassVariance`
(which checks the extent of its k grid once) and for
:meth:`TabulatedPower.power`. Extrapolation by design (a Boltzmann code's tail, the
lazy extension of the mass lattice) has no :class:`TableRange`, so it never warns.
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
from .domain import DomainError, check_extent, warn_once
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

__all__ = ["PowerSource", "TableRange", "TabulatedPower"]


@attrs.frozen
class TableRange:
    """The k range of a table the user supplied, beyond which it is extrapolated.

    The one mechanism for extrapolation warnings of user tables (see the module
    documentation). A :class:`PowerSource` exposes it as ``table_range``.
    """

    #: The smallest wavenumber of the table, in h/Mpc.
    k_min: float = attrs.field(converter=float)
    #: The largest wavenumber of the table, in h/Mpc.
    k_max: float = attrs.field(converter=float)
    #: Whose table it is, for the message (e.g. ``"Transfer (FromArray)"``).
    where: str
    #: How it is extrapolated, for the message (e.g. ``"as a power law"``).
    how: str
    #: Advice appended to the message, if any.
    advice: str = ""

    def warn_outside(
        self, owner: object, k_lo: float, k_hi: float, *, rtol: float = 0.0, stacklevel: int = 2
    ) -> None:
        """Warn, once per owner and end of the table, if [k_lo, k_hi] extends beyond it.

        Both ends in one call give one warning, which names both.

        Parameters
        ----------
        owner
            The object the warning is about (see :func:`~hmf.core.domain.warn_once`).
        k_lo, k_hi
            The smallest and largest wavenumber evaluated, in h/Mpc.
        rtol
            How far beyond an end, relative to it, k may reach without a warning: e.g.
            the rounding of a lattice's ends outwards onto its nodes.
        stacklevel
            As for :func:`warnings.warn`, from the caller of this method.

        Warns
        -----
        HMFExtrapolationWarning
            If ``[k_lo, k_hi]`` extends beyond an end of the table that has not
            warned yet for ``owner``.
        """
        below = k_lo < self.k_min * (1 - rtol)
        above = k_hi > self.k_max * (1 + rtol)
        parts = []
        if below:
            parts.append(f"k below the table's smallest wavenumber, {self.k_min:.4g} h/Mpc,")
        if above:
            parts.append(f"k above the table's largest wavenumber, {self.k_max:.4g} h/Mpc,")
        if not parts:
            return
        keys = (("k below the table",) if below else ()) + (("k above the table",) if above else ())
        warn_once(
            owner,
            keys,
            f"{self.where}: {' and '.join(parts)} is extrapolated {self.how}.{self.advice}",
            stacklevel=stacklevel + 1,
        )


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

        A kernel-level entry point: it raises a
        :class:`~hmf.core.domain.DomainError` for ln k outside the source's domain
        (e.g. not finite), and does not warn (see ``table_range``).

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

    @property
    def table_range(self) -> TableRange | None:
        """The range of the user's table that ``ln_power_kernel`` extrapolates, if any."""
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

    @cached_property
    def table_range(self) -> TableRange | None:
        """The table's range, beyond which it is extrapolated (:class:`PowerSource`).

        ``None`` with ``extension="raise"``, which never extrapolates it.
        """
        if self.extension == "raise":
            return None
        k = self._canonical(self.k, h_Mpc, "k")
        return TableRange(
            k_min=k[0],
            k_max=k[-1],
            where="TabulatedPower",
            how="as a power law",
            advice=" Give a wider table to avoid this, or extension='raise' to make it an error.",
        )

    def ln_power_kernel(self, ln_k: NDArray[np.float64]) -> NDArray[np.float64]:
        """The ln of the power, in (Mpc/h)³, at ln(k / (h/Mpc)) (:class:`PowerSource`).

        Beyond the table it is extrapolated as a power law, without a warning: the
        caller warns, with :attr:`table_range` (see the module documentation).

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
            If a ln k is not finite, or is outside the table and ``extension`` is
            ``"raise"``.
        """
        ln_k = check_extent("ln_k", ln_k, where="TabulatedPower.ln_power_kernel")
        if self.extension == "raise" and ln_k.size:
            table_k = self._table[0]
            lo, hi = table_k[0], table_k[-1]
            for side, outside in (("below", ln_k.min() < lo), ("above", ln_k.max() > hi)):
                if outside:
                    n_out = np.count_nonzero(ln_k < lo if side == "below" else ln_k > hi)
                    raise DomainError(
                        f"TabulatedPower: {n_out} value(s) of k {side} the table "
                        f"[{math.exp(lo):.4g}, {math.exp(hi):.4g}] h/Mpc, with "
                        "extension='raise'."
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

        Warns
        -----
        HMFExtrapolationWarning
            If a k is outside the table and ``extension`` is ``"auto"``: once per
            instance for each end of the table.
        """
        check_in_range("k", k, where="TabulatedPower", low=0.0, low_open=True, error=DomainError)
        k = np.asarray(k, dtype=float)
        out = np.exp(self.ln_power_kernel(np.log(k)))
        table = self.table_range
        if table is not None and k.size:
            table.warn_outside(self, float(k.min()), float(k.max()), stacklevel=3)
        return out
