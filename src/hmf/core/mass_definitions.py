r"""Halo mass definitions, as hashable values.

A mass definition says which mass a halo has: the mass inside the radius where its
mean enclosed density reaches a threshold (a *spherical overdensity*, SO), or the
mass of a friends-of-friends (FoF) group, or the bound mass of a subhalo finder.
Each is a frozen ``attrs`` value: two definitions are equal if, and only if, they are
of the same class with the same parameters, equality is symmetric, and they hash
by value, so they can be dictionary keys and cache keys.

* :class:`SOMean`: :math:`\Delta` (``overdensity``) times the mean density;
* :class:`SOCritical`: :math:`\Delta` times the critical density;
* :class:`SOVirial`: the virial overdensity of Bryan & Norman (1998);
* :class:`SOGeneric`: any SO definition (that of a fit parameterised by it), with a
  ``preferred`` one;
* :class:`FOF`: friends-of-friends groups of linking length b (``linking_length``);
* :class:`SelfBound`: the bound mass of SUBFIND (sub)haloes.

Densities
---------
Every overdensity here is relative to the **mean density of CDM + baryons**,
:math:`\bar\rho_{\rm cb}` (as everywhere in :mod:`hmf.core`; see
:func:`hmf.core.species.omega_m` and :func:`hmf.core.species.rho_mean0`), and
:math:`\Omega_m(z)` is that of CDM + baryons. The halo overdensity relative to the
mean, :math:`\Delta_m`, is :meth:`~MassDefinition.delta_halo_mean_kernel`; the
critical density is :math:`\bar\rho_{\rm cb}(z)/\Omega_m(z)`.

Radii are **comoving**, in Mpc/h, like the filter radii of
:class:`~hmf.core.mass_variance.MassVariance`: a halo of mass m has the radius
:math:`r = [3m/(4\pi\Delta_m\bar\rho_{\rm cb,0})]^{1/3}`, with
:math:`\bar\rho_{\rm cb,0}` the comoving (z = 0) mean density
(:meth:`~MassDefinition.radius_from_m_kernel`); the physical radius is
:math:`r/(1+z)`. The comoving halo density is
:math:`\Delta_m\bar\rho_{\rm cb,0}` (:meth:`~MassDefinition.halo_density_kernel`).

Compatibility
-------------
:meth:`MassDefinition.accepts` says whether a fit measured in one definition can be
evaluated directly in another, without converting masses: every definition accepts
itself, and an :class:`SOGeneric` accepts every SO definition (the fit takes the
overdensity as an input). It is not symmetric: ``SOGeneric(...).accepts(SOMean())``
is True, ``SOMean().accepts(SOGeneric(...))`` is False. Equality is not
compatibility: ``SOGeneric(...) == SOMean()`` is False.

Converting masses between definitions is the job of a
:class:`~hmf.core.mass_conversion.MassConversion`.

Examples
--------
>>> from hmf.core.mass_definitions import MassDefinition, SOCritical, SOGeneric, SOMean
>>> SOMean(overdensity=200) == SOMean(overdensity=200.0)
True
>>> len({SOMean(), SOMean(overdensity=200), SOCritical()})
2
>>> SOGeneric(preferred=SOMean()).accepts(SOCritical(overdensity=500))
True
>>> SOMean().accepts(SOCritical())
False
>>> MassDefinition.from_name("500c")
SOCritical(overdensity=500.0)
"""

from __future__ import annotations

import abc
import math
import re
from typing import Any, ClassVar

import attrs
import numpy as np
import numpy.typing as npt

from . import _references as refs
from ._validators import positive
from .domain import Interval, check_extent
from .units import Mpc_h, Msun_h, rho_unit, unit_boundary

__all__ = [
    "FOF",
    "MassDefinition",
    "SOCritical",
    "SOGeneric",
    "SOMean",
    "SOVirial",
    "SelfBound",
    "SphericalOverdensity",
    "coerce",
]

FloatArray = npt.NDArray[np.float64]

#: Finite and > 0: masses, radii, densities and Omega_m(z).
_POSITIVE = Interval(0, None, lower_open=True)

#: 18 pi^2: the virial overdensity in Einstein-de Sitter.
_EDS_VIRIAL = 18 * np.pi**2


@attrs.frozen(kw_only=True)
class MassDefinition(abc.ABC):
    """Base class of the halo mass definitions (see the module documentation).

    Subclasses are frozen ``attrs`` values; they compare and hash by class and value.
    """

    references: ClassVar[tuple[str, ...]] = ()

    # ------------------------------------------------------------------------------
    # Identity and compatibility
    # ------------------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """A short name, e.g. ``"200m"``, ``"500c"``, ``"vir"``, ``"fof(b=0.2)"``.

        :meth:`from_name` reads it back to an equal definition: a number that ``:g``
        would round (e.g. 18 pi^2) is written in full.
        """

    def __str__(self) -> str:
        """The :attr:`name`."""
        return self.name

    @property
    def resolved(self) -> MassDefinition:
        """The definition masses are actually in: itself (an SOGeneric's ``preferred``)."""
        return self

    def accepts(self, other: MassDefinition) -> bool:
        """Whether a fit measured in this definition can be evaluated in ``other`` directly.

        Every definition accepts itself (an equal value); :class:`SOGeneric` also
        accepts every :class:`SphericalOverdensity`.

        Parameters
        ----------
        other
            The definition asked for.

        Returns
        -------
        bool
        """
        return self == other

    @classmethod
    def from_name(cls, name: str) -> MassDefinition:
        """The definition of a short name: ``"<Delta>m"``, ``"<Delta>c"``, ``"vir"``, ...

        Parameters
        ----------
        name
            ``"200m"`` (:class:`SOMean`), ``"500c"`` (:class:`SOCritical`), ``"vir"``
            (:class:`SOVirial`), ``"so(preferred 200m)"`` (:class:`SOGeneric`, with
            the name of its ``preferred`` definition), ``"fof"`` or ``"fof(b=0.3)"``
            (:class:`FOF`) or ``"self_bound"`` (:class:`SelfBound`); case and spaces
            are ignored.

        Returns
        -------
        MassDefinition

        Raises
        ------
        ValueError
            If the name is not one of these.
        """
        key = name.strip().lower().replace(" ", "")
        if key == "vir":
            return SOVirial()
        if key in ("self_bound", "self-bound", "selfbound"):
            return SelfBound()
        if key == "fof":
            return FOF()
        generic = re.fullmatch(r"so\(preferred(.+)\)", key)
        if generic:
            preferred = MassDefinition.from_name(generic.group(1))
            if not isinstance(preferred, SOMean | SOCritical | SOVirial):
                raise ValueError(
                    f"Unknown mass definition name {name!r}: the preferred definition of "
                    "an SOGeneric must be an SOMean, SOCritical or SOVirial."
                )
            return SOGeneric(preferred=preferred)
        fof = re.fullmatch(r"fof\(b=([0-9.eE+-]+)\)", key)
        if fof:
            return FOF(linking_length=float(fof.group(1)))
        so = re.fullmatch(r"([0-9.eE+]+)([mc])", key)
        if so:
            delta = float(so.group(1))
            return (
                SOMean(overdensity=delta) if so.group(2) == "m" else SOCritical(overdensity=delta)
            )
        raise ValueError(
            f"Unknown mass definition name {name!r}: use e.g. '200m', '500c', 'vir', "
            "'so(preferred 200m)', 'fof', 'fof(b=0.3)' or 'self_bound'."
        )

    # ------------------------------------------------------------------------------
    # Kernel-level entry points
    # ------------------------------------------------------------------------------

    def _delta_halo_mean(self, omega_m_z: FloatArray) -> FloatArray:
        """The overdensity relative to the mean, on checked Omega_m(z)."""
        raise ValueError(f"The mass definition {self} has no overdensity.")

    def delta_halo_mean_kernel(self, *, omega_m_z: npt.ArrayLike) -> FloatArray:
        r"""The halo overdensity relative to the mean density, :math:`\Delta_m`.

        A pure, elementwise kernel of :math:`\Omega_m(z)` (of CDM + baryons), on plain
        arrays. Each subclass documents its formula.

        Parameters
        ----------
        omega_m_z
            :math:`\Omega_m(z)`, dimensionless.

        Returns
        -------
        numpy.ndarray
            :math:`\Delta_m`, dimensionless, with the shape of ``omega_m_z``.

        Raises
        ------
        DomainError
            If an :math:`\Omega_m(z)` is not finite and > 0.
        ValueError
            For a definition without an overdensity (:class:`SelfBound`).
        """
        om = check_extent(
            "omega_m_z",
            omega_m_z,
            _POSITIVE,
            where=f"{type(self).__name__}.delta_halo_mean_kernel ({self})",
        )
        return self._delta_halo_mean(om)

    def halo_density_kernel(
        self, *, omega_m_z: npt.ArrayLike, rho_mean0: npt.ArrayLike
    ) -> FloatArray:
        r"""The comoving halo density, :math:`\Delta_m\bar\rho_{\rm cb,0}`.

        Parameters
        ----------
        omega_m_z
            :math:`\Omega_m(z)` of CDM + baryons, dimensionless.
        rho_mean0
            The comoving mean density of CDM + baryons, in Msun h^2 / Mpc^3
            (:func:`hmf.core.species.rho_mean0`); broadcast with ``omega_m_z``.

        Returns
        -------
        numpy.ndarray
            In Msun h^2 / Mpc^3 (comoving); times :math:`(1+z)^3` for the physical one.

        Raises
        ------
        DomainError
            If a value is not finite and > 0.
        ValueError
            For a definition without an overdensity.
        """
        rho = check_extent(
            "rho_mean0", rho_mean0, _POSITIVE, where=f"{type(self).__name__}.halo_density_kernel"
        )
        out: FloatArray = self.delta_halo_mean_kernel(omega_m_z=omega_m_z) * rho
        return out

    def radius_from_m_kernel(
        self, *, m: npt.ArrayLike, omega_m_z: npt.ArrayLike, rho_mean0: npt.ArrayLike
    ) -> FloatArray:
        r"""The comoving radius of haloes of mass m: :math:`[3m/(4\pi\rho_{\rm halo})]^{1/3}`.

        Parameters
        ----------
        m
            Masses in Msun/h, in this definition.
        omega_m_z
            :math:`\Omega_m(z)` of CDM + baryons, broadcast with ``m``.
        rho_mean0
            The comoving mean density of CDM + baryons, in Msun h^2 / Mpc^3.

        Returns
        -------
        numpy.ndarray
            Comoving radii in Mpc/h, with the broadcast shape.

        Raises
        ------
        DomainError
            If a value is not finite and > 0.
        ValueError
            For a definition without an overdensity.
        """
        m = check_extent("m", m, _POSITIVE, where=f"{type(self).__name__}.radius_from_m_kernel")
        rho = self.halo_density_kernel(omega_m_z=omega_m_z, rho_mean0=rho_mean0)
        out: FloatArray = np.cbrt(3.0 * m / (4.0 * np.pi * rho))
        return out

    def m_from_radius_kernel(
        self, *, r: npt.ArrayLike, omega_m_z: npt.ArrayLike, rho_mean0: npt.ArrayLike
    ) -> FloatArray:
        r"""The mass of haloes of comoving radius r: :math:`4\pi r^3\rho_{\rm halo}/3`.

        Parameters
        ----------
        r
            Comoving radii in Mpc/h.
        omega_m_z
            :math:`\Omega_m(z)` of CDM + baryons, broadcast with ``r``.
        rho_mean0
            The comoving mean density of CDM + baryons, in Msun h^2 / Mpc^3.

        Returns
        -------
        numpy.ndarray
            Masses in Msun/h, in this definition, with the broadcast shape.

        Raises
        ------
        DomainError
            If a value is not finite and > 0.
        ValueError
            For a definition without an overdensity.
        """
        r = check_extent("r", r, _POSITIVE, where=f"{type(self).__name__}.m_from_radius_kernel")
        rho = self.halo_density_kernel(omega_m_z=omega_m_z, rho_mean0=rho_mean0)
        out: FloatArray = 4.0 * np.pi * r**3 * rho / 3.0
        return out

    # ------------------------------------------------------------------------------
    # Public methods (Quantities)
    # ------------------------------------------------------------------------------

    @unit_boundary()
    def delta_halo_mean(self, *, omega_m_z: npt.ArrayLike) -> FloatArray:
        r"""The halo overdensity relative to the mean density of CDM + baryons.

        See :meth:`delta_halo_mean_kernel`.

        Parameters
        ----------
        omega_m_z : float or array_like
            :math:`\Omega_m(z)` of CDM + baryons.

        Returns
        -------
        numpy.float64 or numpy.ndarray
            :math:`\Delta_m`.
        """
        return self.delta_halo_mean_kernel(omega_m_z=omega_m_z)

    @unit_boundary(rho_mean0=rho_unit, returns=rho_unit)
    def halo_density(self, *, omega_m_z: npt.ArrayLike, rho_mean0: Any) -> FloatArray:
        r"""The comoving halo density (see :meth:`halo_density_kernel`).

        Parameters
        ----------
        omega_m_z : float or array_like
            :math:`\Omega_m(z)` of CDM + baryons.
        rho_mean0 : Quantity
            The comoving mean density of CDM + baryons, in Msun h^2 / Mpc^3.

        Returns
        -------
        Quantity
            In Msun h^2 / Mpc^3.
        """
        return self.halo_density_kernel(omega_m_z=omega_m_z, rho_mean0=rho_mean0)

    @unit_boundary(m=Msun_h, rho_mean0=rho_unit, returns=Mpc_h)
    def radius_from_m(self, *, m: Any, omega_m_z: npt.ArrayLike, rho_mean0: Any) -> FloatArray:
        r"""The comoving radius of haloes of mass m (see :meth:`radius_from_m_kernel`).

        Parameters
        ----------
        m : Quantity
            Masses in Msun/h, in this definition.
        omega_m_z : float or array_like
            :math:`\Omega_m(z)` of CDM + baryons.
        rho_mean0 : Quantity
            The comoving mean density of CDM + baryons, in Msun h^2 / Mpc^3.

        Returns
        -------
        Quantity
            Comoving radii in Mpc/h.
        """
        return self.radius_from_m_kernel(m=m, omega_m_z=omega_m_z, rho_mean0=rho_mean0)

    @unit_boundary(r=Mpc_h, rho_mean0=rho_unit, returns=Msun_h)
    def m_from_radius(self, *, r: Any, omega_m_z: npt.ArrayLike, rho_mean0: Any) -> FloatArray:
        r"""The mass of haloes of comoving radius r (see :meth:`m_from_radius_kernel`).

        Parameters
        ----------
        r : Quantity
            Comoving radii in Mpc/h.
        omega_m_z : float or array_like
            :math:`\Omega_m(z)` of CDM + baryons.
        rho_mean0 : Quantity
            The comoving mean density of CDM + baryons, in Msun h^2 / Mpc^3.

        Returns
        -------
        Quantity
            Masses in Msun/h, in this definition.
        """
        return self.m_from_radius_kernel(r=r, omega_m_z=omega_m_z, rho_mean0=rho_mean0)


@attrs.frozen(kw_only=True)
class SphericalOverdensity(MassDefinition):
    """Base class of the spherical-overdensity definitions."""


def _number(value: float) -> str:
    """``value`` in a name: as ``:g`` (``200``) where that is exact, else in full."""
    short = f"{value:g}"
    return short if float(short) == value else repr(value)


def _overdensity(default: float) -> Any:
    return attrs.field(default=default, converter=float, validator=positive)


@attrs.frozen(kw_only=True)
class SOMean(SphericalOverdensity):
    r"""Spherical overdensity of ``overdensity`` times the mean density.

    :math:`\Delta_m = \Delta`, at every z.

    Parameters
    ----------
    overdensity
        :math:`\Delta`, relative to the mean density of CDM + baryons (default 200).
    """

    overdensity: float = _overdensity(200.0)

    @property
    def name(self) -> str:
        """``"<overdensity>m"``, e.g. ``"200m"``."""
        return f"{_number(self.overdensity)}m"

    def _delta_halo_mean(self, omega_m_z: FloatArray) -> FloatArray:
        return np.full_like(omega_m_z, self.overdensity)


@attrs.frozen(kw_only=True)
class SOCritical(SphericalOverdensity):
    r"""Spherical overdensity of ``overdensity`` times the critical density.

    :math:`\Delta_m = \Delta_c/\Omega_m(z)`.

    Parameters
    ----------
    overdensity
        :math:`\Delta_c`, relative to the critical density (default 200).
    """

    overdensity: float = _overdensity(200.0)

    @property
    def name(self) -> str:
        """``"<overdensity>c"``, e.g. ``"500c"``."""
        return f"{_number(self.overdensity)}c"

    def _delta_halo_mean(self, omega_m_z: FloatArray) -> FloatArray:
        out: FloatArray = self.overdensity / omega_m_z
        return out


@attrs.frozen(kw_only=True)
class SOVirial(SphericalOverdensity):
    r"""The virial spherical overdensity of Bryan & Norman (1998).

    Relative to the critical density, :math:`\Delta_c = 18\pi^2 + 82x - 39x^2` with
    :math:`x = \Omega_m(z) - 1`, so relative to the mean
    :math:`\Delta_m = (18\pi^2 + 82x - 39x^2)/\Omega_m(z)`: :math:`18\pi^2` in
    Einstein-de Sitter (where it is the same definition as ``SOMean(18 pi^2)``), and
    about 333 at z = 0 for Planck18. Their fit is for flat cosmologies with a
    cosmological constant.
    """

    references: ClassVar[tuple[str, ...]] = (refs.BRYAN98,)

    @property
    def name(self) -> str:
        """``"vir"``."""
        return "vir"

    def _delta_halo_mean(self, omega_m_z: FloatArray) -> FloatArray:
        x = omega_m_z - 1
        out: FloatArray = (_EDS_VIRIAL + 82 * x - 39 * x**2) / omega_m_z
        return out


def _concrete_so(instance: Any, attribute: attrs.Attribute[Any], value: Any) -> None:
    if not isinstance(value, SphericalOverdensity) or isinstance(value, SOGeneric):
        raise TypeError(
            f"SOGeneric.preferred must be an SOMean, SOCritical or SOVirial, got {value!r}."
        )


@attrs.frozen(kw_only=True)
class SOGeneric(SphericalOverdensity):
    """Any spherical overdensity: the definition of a fit that takes it as an input.

    Such a fit (e.g. Tinker08) can be evaluated in every SO definition directly, with
    that definition's overdensity, so :meth:`accepts` every
    :class:`SphericalOverdensity`. It is not itself a definition masses can be in:
    its overdensity, and :attr:`resolved`, are those of ``preferred``.

    Parameters
    ----------
    preferred
        The SO definition used when no other is asked for (e.g. ``SOMean(200)``).
    """

    preferred: SphericalOverdensity = attrs.field(validator=_concrete_so)

    @property
    def name(self) -> str:
        """``"so(preferred <name>)"``."""
        return f"so(preferred {self.preferred.name})"

    @property
    def resolved(self) -> MassDefinition:
        """``preferred``."""
        return self.preferred

    def accepts(self, other: MassDefinition) -> bool:
        """True for every :class:`SphericalOverdensity` (see :meth:`MassDefinition.accepts`).

        Parameters
        ----------
        other
            The definition asked for.

        Returns
        -------
        bool
        """
        return isinstance(other, SphericalOverdensity)

    def _delta_halo_mean(self, omega_m_z: FloatArray) -> FloatArray:
        return self.preferred._delta_halo_mean(omega_m_z)


@attrs.frozen(kw_only=True)
class FOF(MassDefinition):
    r"""Friends-of-friends groups of linking length b.

    Its overdensity is that of the isodensity surface b selects for a singular
    isothermal sphere, :math:`\Delta_m = 9/(2\pi b^3)` (White, Hernquist & Springel
    2001; about 179 for b = 0.2), independent of :math:`\Omega_m(z)`, as in hmf 3.x.
    It is an approximation: FoF groups are not spheres, and their masses map onto SO
    masses only roughly.

    Parameters
    ----------
    linking_length
        b, in units of the mean interparticle separation (default 0.2).
    """

    references: ClassVar[tuple[str, ...]] = (refs.DAVIS85, refs.WHITE01)

    linking_length: float = attrs.field(default=0.2, converter=float, validator=positive)

    @property
    def name(self) -> str:
        """``"fof(b=<linking_length>)"``."""
        return f"fof(b={_number(self.linking_length)})"

    def _delta_halo_mean(self, omega_m_z: FloatArray) -> FloatArray:
        return np.full_like(omega_m_z, 9 / (2 * math.pi * self.linking_length**3))


@attrs.frozen(kw_only=True)
class SelfBound(MassDefinition):
    """The bound mass of SUBFIND (sub)haloes: no overdensity, and no radius."""

    @property
    def name(self) -> str:
        """``"self_bound"``."""
        return "self_bound"


def coerce(value: MassDefinition | str) -> MassDefinition:
    """A mass definition from an instance or a name (:meth:`MassDefinition.from_name`).

    Parameters
    ----------
    value
        A :class:`MassDefinition`, or its short name (e.g. ``"200c"``).

    Returns
    -------
    MassDefinition

    Raises
    ------
    TypeError
        If ``value`` is neither.
    ValueError
        If it is a name that is not one.
    """
    if isinstance(value, MassDefinition):
        return value
    if isinstance(value, str):
        return MassDefinition.from_name(value)
    raise TypeError(
        f"Expected a MassDefinition or its name (e.g. '200m'), got {type(value).__name__}."
    )
