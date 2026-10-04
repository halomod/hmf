"""Accuracy settings of the internal grids (issue #384).

The v4 core evaluates expensive quantities on internal grids, and interpolates
between their nodes. These classes hold the settings of those grids. They are typed,
validated, immutable containers; the stages that use them come later.

Each class has three presets: the defaults (the class's constructor), :meth:`fast`
(roughly v3's accuracy, at about a quarter of the per-node cost) and :meth:`high`
(for convergence checks). Each preset accepts overrides of individual settings.

Grid spacings are in logarithms with an explicit base: ``log10`` for masses, ``ln``
for wavenumbers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, ClassVar, Literal, Self

import attrs

from ._fields import FieldInfo, add_parameters_section, field, fields_info

__all__ = ["Accuracy", "Extension", "KAccuracy", "MassAccuracy"]

#: What to do with requests outside an internal grid:
#:
#: ``"auto"``
#:     extend the grid lazily, with nodes on the same lattice, so that results do not
#:     depend on the order of requests;
#: ``"raise"``
#:     raise an error.
Extension = Literal["auto", "raise"]


def _positive(instance: Any, attribute: attrs.Attribute[float], value: float) -> None:
    """Validate that a setting is finite and strictly positive."""
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{type(instance).__name__}.{attribute.name} must be > 0, got {value}.")


def _finite(instance: Any, attribute: attrs.Attribute[float], value: float) -> None:
    """Validate that a setting is finite."""
    if not math.isfinite(value):
        raise ValueError(f"{type(instance).__name__}.{attribute.name} must be finite, got {value}.")


_extension = attrs.validators.in_(("auto", "raise"))


@attrs.frozen(kw_only=True)
class Accuracy:
    """Base class of the accuracy settings.

    Subclasses define their settings as fields, with the defaults as field defaults,
    and the :meth:`fast` and :meth:`high` presets as the class variables
    ``_fast`` and ``_high`` (mappings of the settings that differ from the
    defaults).
    """

    _fast: ClassVar[Mapping[str, Any]] = MappingProxyType({})
    _high: ClassVar[Mapping[str, Any]] = MappingProxyType({})

    def __init_subclass__(cls, /, **kwargs: Any) -> None:
        """Generate the subclass's docstring "Parameters" section from its fields."""
        super().__init_subclass__(**kwargs)
        if "__attrs_attrs__" in cls.__dict__:
            add_parameters_section(cls)

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

    @classmethod
    def fields_info(cls) -> tuple[FieldInfo, ...]:
        """Describe the settings, without creating an instance.

        Returns
        -------
        tuple of FieldInfo
            The name, type, default and documentation of each setting.
        """
        return fields_info(cls)


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
        validator=_positive,
        doc="Spacing of the mass lattice, in log10(m / (Msun/h)).",
    )
    log10_m_min: float = field(
        default=0.0,
        converter=float,
        validator=_finite,
        doc="log10 of the smallest mass of the default grid, in Msun/h.",
    )
    log10_m_max: float = field(
        default=17.5,
        converter=float,
        validator=_finite,
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

    The grid's upper end is not a setting: it is derived from the smallest mass
    requested, as ``k_max >= k_max_r_min / R(m_min)``.
    """

    _fast: ClassVar[Mapping[str, Any]] = MappingProxyType({"dln_k": 0.05})
    _high: ClassVar[Mapping[str, Any]] = MappingProxyType({"dln_k": 0.005})

    dln_k: float = field(
        default=0.02,
        converter=float,
        validator=_positive,
        doc="Spacing of the wavenumber grid, in ln(k / (h/Mpc)).",
    )
    ln_k_min: float = field(
        default=math.log(1e-8),
        converter=float,
        validator=_finite,
        doc="ln of the smallest wavenumber of the grid, in h/Mpc.",
    )
    k_max_r_min: float = field(
        default=20.0,
        converter=float,
        validator=_positive,
        doc=(
            "The grid extends to at least k_max = k_max_r_min / R, with R the "
            "Lagrangian radius of the smallest mass requested."
        ),
    )
    extension: Extension = field(
        default="raise",
        validator=_extension,
        doc="What to do with wavenumbers outside the grid: 'auto' or 'raise'.",
    )
