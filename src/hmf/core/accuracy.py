"""Accuracy settings of the internal grids.

The v4 core evaluates expensive quantities on internal grids, and interpolates
between their nodes. These classes hold the settings of those grids. They are typed,
validated, immutable containers, given to the stages that build the grids (e.g.
:class:`~hmf.core.mass_variance.MassVariance`).

Each class has three presets: the defaults (the class's constructor), :meth:`fast`
(roughly v3's accuracy, at about a quarter of the per-node cost) and :meth:`high`
(for convergence checks). Each preset accepts overrides of individual settings.

Grid spacings are in logarithms with an explicit base: ``log10`` for masses, ``ln``
for wavenumbers.

One KAccuracy for the k-space calculation
-----------------------------------------
Three stages have a ``k_accuracy`` field, and each reads only some of its settings:

=================================================  ====================================
Stage                                              Settings of :class:`KAccuracy` read
=================================================  ====================================
:class:`~hmf.core.transfer.Transfer`               ``dln_k``: the precision and k
                                                   sampling of the CAMB or CLASS run
                                                   (finer than the code's own below
                                                   ``dln_k = 0.02``, with CAMB's
                                                   ``high_precision``). Other models
                                                   read none.
:class:`~hmf.core.growth.Growth`                   ``dln_k``, as Transfer, for a run
                                                   its CAMB or CLASS growth model makes
                                                   itself (not one shared with its
                                                   transfer stage).
:class:`~hmf.core.mass_variance.MassVariance`      ``dln_k``, ``ln_k_min`` and
                                                   ``k_max_r_min``: the k grid of the
                                                   sigma integrals.
=================================================  ====================================

So ``KAccuracy.high()`` refines the Boltzmann runs as well as the grid of the sigma
integrals, ``KAccuracy.fast()`` coarsens the grid but leaves the runs at the codes'
own precision, and the defaults leave the runs at the codes' own precision too.

The rule is that a stage composed of these passes **one** ``KAccuracy`` to all of
them, so that a single setting controls the accuracy of the whole k-space
calculation. :func:`check_consistent` checks it. A :class:`~hmf.core.growth.Growth`
stage built with :meth:`Growth.from_transfer <hmf.core.growth.Growth.from_transfer>`
takes its transfer stage's ``k_accuracy``, and shares the transfer's Boltzmann run
only if their accuracies are equal.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, ClassVar, Literal, Protocol, Self

import attrs

from ._fields import Documented, field
from ._validators import finite, positive

__all__ = ["Accuracy", "Extension", "HasKAccuracy", "KAccuracy", "MassAccuracy", "check_consistent"]

#: What to do with requests outside an internal grid or a table: the out-of-range
#: switch of :mod:`hmf.core`. What ``"auto"`` does depends on the object:
#:
#: ``"auto"``
#:     handle them: :attr:`MassAccuracy.extension` extends the mass lattice lazily,
#:     with nodes on the same lattice, so that results do not depend on the order of
#:     requests; :attr:`TabulatedPower.extension
#:     <hmf.core.power_source.TabulatedPower.extension>` extrapolates the table as a
#:     power law (with an :class:`~hmf.exceptions.HMFExtrapolationWarning`);
#: ``"raise"``
#:     raise a :class:`~hmf.core.domain.DomainError`.
#:
#: It is not a :data:`~hmf.core.domain.DomainPolicy`, which is what to do outside a
#: model's *calibration* domain.
Extension = Literal["auto", "raise"]


_extension = attrs.validators.in_(("auto", "raise"))


@attrs.frozen(kw_only=True)
class Accuracy(Documented):
    """Base class of the accuracy settings.

    Subclasses define their settings as fields, with the defaults as field defaults,
    and the :meth:`fast` and :meth:`high` presets as the class variables
    ``_fast`` and ``_high`` (mappings of the settings that differ from the
    defaults).
    """

    _fast: ClassVar[Mapping[str, Any]] = MappingProxyType({})
    _high: ClassVar[Mapping[str, Any]] = MappingProxyType({})

    @classmethod
    def fast(cls, **overrides: Any) -> Self:
        """Return the fast preset: about v3's accuracy, at lower cost.

        Parameters
        ----------
        **overrides
            Settings to change from the preset.

        Returns
        -------
        Accuracy
            A new instance.
        """
        return cls(**{**cls._fast, **overrides})

    @classmethod
    def high(cls, **overrides: Any) -> Self:
        """Return the high-accuracy preset, for convergence checks.

        Parameters
        ----------
        **overrides
            Settings to change from the preset.

        Returns
        -------
        Accuracy
            A new instance.
        """
        return cls(**{**cls._high, **overrides})


@attrs.frozen(kw_only=True)
class MassAccuracy(Accuracy):
    """Settings of the internal mass grid (the lattice in log10 m).

    Nodes sit at ``log10(m / (Msun/h)) = i * dlog10_m`` for integer ``i``, so grids with
    the same spacing share their nodes, whatever their range.
    """

    _fast: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {"dlog10_m": 0.05, "second_derivative": False}
    )
    _high: ClassVar[Mapping[str, Any]] = MappingProxyType({"dlog10_m": 0.01})

    dlog10_m: float = field(
        default=0.02,
        converter=float,
        validator=positive,
        doc="Spacing of the mass lattice, in log10(m / (Msun/h)).",
    )
    log10_m_min: float = field(
        default=0.0,
        converter=float,
        validator=finite,
        doc="log10 of the smallest mass of the default grid, in Msun/h.",
    )
    log10_m_max: float = field(
        default=17.5,
        converter=float,
        validator=finite,
        doc="log10 of the largest mass of the default grid, in Msun/h.",
    )
    extension: Extension = field(
        default="auto",
        validator=_extension,
        doc="What to do with masses outside the grid: 'auto' or 'raise'.",
    )
    second_derivative: bool = field(
        default=True,
        validator=attrs.validators.instance_of(bool),
        doc=(
            "Whether to compute the second derivative d2 = d^2 ln(sigma) / d(ln m)^2 at "
            "each node, for higher-order interpolation of sigma and dln(sigma)/dln(m) "
            "(about 1.6x the cost per node)."
        ),
    )

    @log10_m_max.validator
    def _check_range(self, attribute: attrs.Attribute[float], value: float) -> None:
        if not self.log10_m_min < value:
            raise ValueError(
                f"MassAccuracy: log10_m_min ({self.log10_m_min}) must be less than "
                f"log10_m_max ({value})."
            )


@attrs.frozen(kw_only=True)
class KAccuracy(Accuracy):
    """Settings of the internal wavenumber grid (the lattice in ln k).

    The grid's upper end is not a setting: it is derived from the mass lattice's
    settings, as ``k_max >= k_max_r_min / R(m_min)`` with ``m_min`` the smallest mass of
    the default mass grid (``MassAccuracy.log10_m_min``), not the smallest mass
    requested. So the k grid does not change when the mass lattice extends.
    """

    _fast: ClassVar[Mapping[str, Any]] = MappingProxyType({"dln_k": 0.05})
    _high: ClassVar[Mapping[str, Any]] = MappingProxyType({"dln_k": 0.005})

    dln_k: float = field(
        default=0.02,
        converter=float,
        validator=positive,
        doc="Spacing of the wavenumber grid, in ln(k / (h/Mpc)).",
    )
    ln_k_min: float = field(
        default=math.log(1e-8),
        converter=float,
        validator=finite,
        doc="ln of the smallest wavenumber of the grid, in h/Mpc.",
    )
    k_max_r_min: float = field(
        default=20.0,
        converter=float,
        validator=positive,
        doc=(
            "The grid extends to at least k_max = k_max_r_min / R, with R the "
            "Lagrangian radius of the smallest mass of the default mass grid "
            "(MassAccuracy.log10_m_min)."
        ),
    )


class HasKAccuracy(Protocol):
    """An object with a ``k_accuracy`` (a stage that computes something in k-space)."""

    @property
    def k_accuracy(self) -> KAccuracy:
        """The wavenumber accuracy."""
        ...


def check_consistent(first: HasKAccuracy, *others: HasKAccuracy) -> KAccuracy:
    """Check that stages share one :class:`KAccuracy`, and return it.

    The rule of a stage composed of others (see the module documentation): one
    ``KAccuracy`` drives the whole k-space calculation. Equal settings count as one
    (a ``KAccuracy`` compares by value).

    Parameters
    ----------
    first, *others
        The stages, each with a ``k_accuracy`` (e.g. a
        :class:`~hmf.core.transfer.Transfer`, a :class:`~hmf.core.growth.Growth` and a
        :class:`~hmf.core.mass_variance.MassVariance`).

    Returns
    -------
    KAccuracy
        The ``k_accuracy`` they share.

    Raises
    ------
    ValueError
        If their ``k_accuracy`` differ. The message lists each stage's.

    Examples
    --------
    >>> from hmf.core.growth import Growth
    >>> from hmf.core.transfer import Transfer
    >>> transfer = Transfer(model="EH", k_accuracy=KAccuracy.high())
    >>> check_consistent(transfer, Growth.from_transfer(transfer)) == KAccuracy.high()
    True
    """
    shared = first.k_accuracy
    stages = (first, *others)
    if all(stage.k_accuracy == shared for stage in others):
        return shared
    listing = "; ".join(f"{type(stage).__name__}: {stage.k_accuracy!r}" for stage in stages)
    raise ValueError(
        "The stages must share one KAccuracy, so that one setting drives the whole "
        f"k-space calculation; they have {listing}. Pass the same k_accuracy to each."
    )
