r"""The :class:`LinearPower` stage: the linear power spectrum P(k, z), normalised to sigma_8.

The stage combines a :class:`~hmf.core.transfer.Transfer` stage (the shape of the
power at z = 0) and its growth, a :class:`~hmf.core.growth.Growth` stage built from
the transfer stage and ``growth_model`` (:attr:`LinearPower.growth`), with the
normalisation :math:`\sigma_8`:

.. math:: P(k, z) = A\, D^2(z)\, P_{\rm raw}(k),

with :math:`P_{\rm raw} = k^{n_s} T^2(k)` the unnormalised power of ``species``
(:meth:`Transfer.power_source <hmf.core.transfer.Transfer.power_source>`), and
:math:`D(z)` the growth factor of the same species, with :math:`D(0) = 1`.

The sigma_8 normalisation
-------------------------
The amplitude :math:`A = (\sigma_8 / \sigma_{8,\rm raw})^2` is a plain scalar
(:attr:`LinearPower.amplitude`). :math:`\sigma_{8,\rm raw}` is the rms of the
unnormalised power of ``sigma_8_species``, smoothed with a top-hat of radius
8 Mpc/h: :meth:`MassVariance.ln_sigma_at_radius_kernel
<hmf.core.mass_variance.MassVariance.ln_sigma_at_radius_kernel>` of a
:class:`~hmf.core.filters.TopHat` :class:`~hmf.core.mass_variance.MassVariance` of
that species' power, on the k grid of the transfer stage's ``k_accuracy``. By default ``species`` is
``"cb"`` and ``sigma_8_species`` is ``"tot"``, as in hmf 3.x: the CDM + baryon power,
normalised so that the *total* matter field has the rms sigma_8 that CMB experiments
quote. With massive neutrinos the CDM + baryon field then has a slightly larger rms
than sigma_8 at 8 Mpc/h.

:attr:`LinearPower.power_source` is the *unnormalised* power of ``species``, the power
of the mass variance (see :class:`~hmf.core.mass_function.MassFunction`). sigma_8
enters only through :math:`A`, so changing it changes only that scalar, and never the
lattice of a :class:`~hmf.core.mass_variance.MassVariance`.

Scale-independent growth
------------------------
P(k, z) factorises into a function of k and one of z. That holds for the linear
power of CDM + baryons on the scales of haloes, and is how hmf 3.x computes it; with
massive neutrinos the growth of the total matter field depends on scale. Every
z-dependence of the linear power goes through :meth:`LinearPower.sigma_scale_kernel`
(:math:`\sqrt{A}\,D(z)`, the ratio of sigma(m, z) to the unnormalised sigma(m)) and
:meth:`LinearPower.power_kernel`, which is its square times :math:`P_{\rm raw}(k)`. A
scale-dependent P(k, z), e.g. from a Boltzmann code at each z, replaces those two,
and then needs the mass variance of P(k, z) itself rather than a rescaling of
:math:`\sigma_{\rm raw}`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from functools import cached_property
from types import MappingProxyType
from typing import Any, ClassVar

import attrs
import numpy as np
import numpy.typing as npt
from astropy.cosmology import FLRW

# For the fully qualified annotations below.
import hmf.core.growth
import hmf.core.transfer

from ._fields import field
from ._validators import positive
from .accuracy import KAccuracy
from .domain import check_extent
from .filters import TopHat
from .growth import Growth
from .growth_models import GrowthModel, ODEGrowth
from .mass_variance import MassVariance
from .species import check_species
from .stage import Stage, derived_property
from .transfer import Transfer, UnnormalisedPower
from .units import UnitContext, h_Mpc, power_unit, unit_boundary

__all__ = ["SIGMA_8_RADIUS", "LinearPower"]

FloatArray = npt.NDArray[np.float64]

#: The radius of the top-hat that defines sigma_8, in Mpc/h.
SIGMA_8_RADIUS = 8.0


def _check_growth_model(instance: LinearPower, attribute: attrs.Attribute[Any], value: Any) -> None:
    """Validate the growth model: it must apply to the transfer stage's cosmology."""
    value.check_cosmology(instance.transfer.cosmology)


@attrs.frozen(kw_only=True)
class LinearPower(Stage):
    """The linear power spectrum P(k, z), normalised to sigma_8.

    See the module documentation.

    Examples
    --------
    >>> from hmf.core.transfer import Transfer
    >>> from hmf.core.units import Mpc_h, h_Mpc
    >>> lp = LinearPower(transfer=Transfer(model="EH"), sigma_8=0.8)
    >>> p = lp.power(k=[0.01, 0.1, 1.0] * h_Mpc, z=0.0)
    >>> p.unit == (Mpc_h**3)
    True
    """

    # Fully qualified, so that the docs can tell it from v3's Transfer.
    transfer: hmf.core.transfer.Transfer = field(
        validator=attrs.validators.instance_of(Transfer),
        doc=(
            "The transfer stage: the shape of the power at z = 0. Its cosmology, "
            "k_accuracy and disk cache are the growth's too."
        ),
    )
    growth_model: GrowthModel = field(
        factory=ODEGrowth,
        converter=GrowthModel.coerce,
        validator=_check_growth_model,
        doc="The growth model: an instance, a class or a registered name (e.g. 'ODE').",
    )
    sigma_8: float = field(
        converter=float,
        validator=positive,
        doc=(
            "The rms of the linear density field of sigma_8_species at z = 0, smoothed "
            "with a top-hat of radius 8 Mpc/h."
        ),
    )
    species: str = field(
        default="cb",
        converter=check_species,
        doc="The matter species of the power and its growth: 'cb' (CDM + baryons) or 'tot'.",
    )
    sigma_8_species: str = field(
        default="tot",
        converter=check_species,
        doc=(
            "The matter species whose rms sigma_8 is: 'tot' (total matter, as quoted by "
            "CMB experiments, and hmf 3.x's default) or 'cb'."
        ),
    )

    #: The transfer model, whose field name (``model``) reads badly flat.
    parameter_aliases: ClassVar[Mapping[str, str]] = MappingProxyType(
        {"transfer_model": "transfer.model"}
    )

    @property
    def cosmology(self) -> FLRW:
        """The cosmology (the transfer stage's)."""
        return self.transfer.cosmology

    @property
    def k_accuracy(self) -> KAccuracy:
        """The wavenumber accuracy of the whole k-space calculation (the transfer stage's)."""
        return self.transfer.k_accuracy

    @cached_property
    def _unit_context(self) -> UnitContext:
        """Units context with the cosmology's H0, to convert physical units to h-units."""
        return UnitContext(self.cosmology.H0)

    # Fully qualified, so that the docs can tell it from v3's growth.
    @derived_property("transfer", "growth_model")
    def growth(self) -> hmf.core.growth.Growth:
        """The growth stage: the growth factor D(z) of ``growth_model``.

        Built from the transfer stage (:meth:`Growth.from_transfer
        <hmf.core.growth.Growth.from_transfer>`), so a CAMB or CLASS growth model
        shares its Boltzmann run.
        """
        return Growth.from_transfer(self.transfer, model=self.growth_model)

    @cached_property
    def power_source(self) -> UnnormalisedPower:
        r"""The unnormalised power of ``species`` at z = 0, a PowerSource.

        :math:`P_{\rm raw}(k) = k^{n_s} T^2(k)`, from the transfer stage: the power of
        the :class:`~hmf.core.mass_variance.MassVariance` of a mass function. sigma_8
        does not change it.
        """
        return self.transfer.power_source(self.species)

    @derived_property("transfer", "sigma_8_species")
    def unnormalised_sigma_8(self) -> float:
        """sigma_8 of the unnormalised power of ``sigma_8_species``, a plain float.

        A direct evaluation (no mass lattice), with a top-hat on the k grid of the
        transfer stage's ``k_accuracy`` (see the module documentation).
        """
        variance = MassVariance(
            power=self.transfer.power_source(self.sigma_8_species),
            filter=TopHat(),
            k_accuracy=self.k_accuracy,
        )
        return math.exp(float(variance.ln_sigma_at_radius_kernel(SIGMA_8_RADIUS)))

    @cached_property
    def amplitude(self) -> float:
        r"""The normalisation of the power, :math:`A = (\sigma_8/\sigma_{8,\rm raw})^2`.

        A plain float: the factor that turns the unnormalised power into the power at
        z = 0, in (Mpc/h)³ (the unnormalised power has k in h/Mpc).
        """
        return (self.sigma_8 / self.unnormalised_sigma_8) ** 2

    def sigma_scale_kernel(self, z: npt.ArrayLike) -> FloatArray:
        r"""The ratio of sigma(m, z) to the unnormalised sigma(m), at kernel level.

        :math:`\sqrt{A}\,D(z)`, with the growth factor of ``species``. For library code
        working in plain arrays (see :mod:`hmf.core._kernels`); it checks z as
        :meth:`Growth.growth_factor_kernel
        <hmf.core.growth.Growth.growth_factor_kernel>` does.

        Parameters
        ----------
        z
            Redshift(s), dimensionless: a float or a plain array.

        Returns
        -------
        numpy.ndarray
            Dimensionless, with the shape of ``z``.

        Raises
        ------
        DomainError
            If a z is not finite, is < 0, or is outside the growth model's domain or
            table.
        """
        out: FloatArray = math.sqrt(self.amplitude) * self.growth.growth_factor_kernel(
            z, self.species
        )
        return out

    def power_kernel(self, *, k: npt.ArrayLike, z: npt.ArrayLike) -> FloatArray:
        r"""The linear power :math:`A D^2(z) P_{\rm raw}(k)`, at kernel level.

        For library code working in plain arrays (see :mod:`hmf.core._kernels`). It
        checks k against the transfer model's valid domain, and z as
        :meth:`sigma_scale_kernel` does, and does not warn when it extrapolates a
        user's table (see :meth:`power`).

        Parameters
        ----------
        k
            Wavenumbers in h/Mpc: a plain array.
        z
            Redshift(s), dimensionless; broadcast with ``k`` like numpy.

        Returns
        -------
        numpy.ndarray
            The power in (Mpc/h)³, with the broadcast shape of ``k`` and ``z``.

        Raises
        ------
        DomainError
            If a k is not finite or outside the transfer model's valid domain, or a z
            is outside the growth's domain.
        """
        model = type(self.transfer.model)
        k = check_extent(
            "k", k, model.valid_domain, where=f"LinearPower.power_kernel ({model.__name__})"
        )
        scale = self.sigma_scale_kernel(z)
        out: FloatArray = scale**2 * np.exp(self.power_source.ln_power_kernel(np.log(k)))
        return out

    @unit_boundary(k=h_Mpc, returns=power_unit)
    def power(self, *, k: Any, z: Any) -> FloatArray:
        """The linear power spectrum P(k, z), normalised to sigma_8.

        Parameters
        ----------
        k : Quantity
            Wavenumbers, in h/Mpc (or 1/Mpc, converted with the cosmology's H0).
        z : float or array_like
            Redshift(s), dimensionless; broadcast with ``k`` like numpy, so
            ``power(k=k[None, :], z=z[:, None])`` has the shape ``(nz, nk)``.

        Returns
        -------
        Quantity
            The power in (Mpc/h)³, with the broadcast shape of ``k`` and ``z`` (a
            scalar for scalar inputs).

        Raises
        ------
        DomainError
            If a k is outside the transfer model's valid domain (k > 0), or a z is
            outside the growth's (z >= 0, and within its table).

        Warns
        -----
        HMFExtrapolationWarning
            If a k is outside a table the user supplied (once per stage and end of
            the table).
        """
        model = type(self.transfer.model)
        model.valid_domain.check({"k": k << h_Mpc}, where=f"LinearPower.power ({model.__name__})")
        table = self.power_source.table_range
        if table is not None and np.size(k):
            table.warn_outside(self, float(np.min(k)), float(np.max(k)), stacklevel=3)
        return self.power_kernel(k=k, z=z)
