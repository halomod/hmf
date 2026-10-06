r"""The :class:`Growth` stage: the linear growth factor and growth rate as functions of z.

The stage holds a cosmology and a growth model, and optionally the
:class:`~hmf.core.transfer.Transfer` stage it goes with. It gives, for each matter
species, the growth factor D(z) (normalised to D(0) = 1) and the growth rate
:math:`f = d\ln D/d\ln a`. Redshifts are dimensionless and broadcast like numpy:
a scalar gives a float, an array an array of the same shape.

If the growth model is backed by the same Boltzmann code as the transfer stage's
model (:class:`~hmf.core.growth_models.CambGrowth` with
:class:`~hmf.core.transfer_models.CAMB`, or
:class:`~hmf.core.growth_models.ClassGrowth` with
:class:`~hmf.core.transfer_models.CLASS`), the growth comes from the transfer's run:
one run for both.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any

import attrs
from astropy.cosmology import FLRW, Planck18

# For the fully qualified annotations below.
import hmf.core.transfer

from ._fields import field
from ._serialise import cosmology_key
from ._validators import check_in_range
from .cache import DiskCache, to_disk_cache
from .domain import DomainError
from .growth_models import GrowthModel, GrowthSolution, ODEGrowth
from .stage import Stage
from .transfer import Transfer

__all__ = ["Growth"]


def _to_growth_model(value: Any) -> GrowthModel:
    """Convert the ``model`` field: a model, a model class, or a name to look up."""
    if isinstance(value, GrowthModel):
        return value
    if isinstance(value, (str, type)):
        return GrowthModel.get(value)()
    raise TypeError(f"model must be a GrowthModel, a class or a name, not {type(value).__name__}.")


@attrs.frozen(kw_only=True)
class Growth(Stage):
    """The linear growth factor and growth rate.

    See the module documentation.

    Examples
    --------
    >>> from hmf.core.growth import Growth
    >>> g = Growth()
    >>> g.growth_factor(0.0)
    1.0
    """

    cosmology: FLRW = field(
        default=Planck18,
        validator=attrs.validators.instance_of(FLRW),
        eq=cosmology_key,
        doc=(
            "The cosmology, an astropy FLRW. It is compared by its class and parameter "
            "values (not its name or metadata). It must equal the transfer stage's, if "
            "one is given (see from_transfer)."
        ),
    )
    model: GrowthModel = field(
        factory=ODEGrowth,
        converter=_to_growth_model,
        doc="The growth model: an instance, a class or a registered name (e.g. 'ODE').",
    )
    # Fully qualified, so that the docs can tell it from v3's Transfer.
    transfer: hmf.core.transfer.Transfer | None = field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.instance_of(Transfer)),
        doc=(
            "The transfer stage this growth goes with, if any. If its model uses the same "
            "Boltzmann code as the growth model, the growth comes from its run."
        ),
    )
    disk_cache: DiskCache | None = field(
        default=None,
        converter=to_disk_cache,
        eq=False,
        doc=(
            "Where to cache Boltzmann-code output on disk (see Transfer.disk_cache), for "
            "a run the growth model makes itself."
        ),
    )

    @model.validator
    def _check_model(self, attribute: attrs.Attribute[GrowthModel], value: GrowthModel) -> None:
        """Check that the model applies to the cosmology."""
        value.check_cosmology(self.cosmology)

    @transfer.validator
    def _check_transfer(
        self, attribute: attrs.Attribute[Transfer | None], value: Transfer | None
    ) -> None:
        """Check that the transfer stage has the same cosmology."""
        if value is not None and cosmology_key(value.cosmology) != cosmology_key(self.cosmology):
            raise ValueError(
                "Growth: the cosmology differs from the transfer stage's. Use "
                "Growth.from_transfer(transfer, ...) to take the transfer's cosmology."
            )

    @classmethod
    def from_transfer(cls, transfer: hmf.core.transfer.Transfer, **kwargs: Any) -> Growth:
        """The growth stage of a transfer stage, with its cosmology and disk cache.

        Parameters
        ----------
        transfer
            The transfer stage.
        **kwargs
            Other fields (``model``, ...).

        Returns
        -------
        Growth
        """
        kwargs.setdefault("disk_cache", transfer.disk_cache)
        return cls(cosmology=transfer.cosmology, transfer=transfer, **kwargs)

    @cached_property
    def solution(self) -> GrowthSolution:
        """The kernel-level growth of every species (one model solve)."""
        run = None
        if (
            self.transfer is not None
            and self.model.backend is not None
            and self.transfer.model.backend == self.model.backend
        ):
            run = self.transfer.boltzmann_run
        return self.model.solve(self.cosmology, run=run, disk_cache=self.disk_cache)

    def _redshifts(self, z: Any) -> Any:
        return check_in_range(
            "z",
            z,
            where=type(self.model).__name__,
            low=0.0,
            high=self.solution.z_max * (1 + 1e-12),
            error=DomainError,
        )

    def growth_factor(self, z: Any, species: str = "cb") -> Any:
        """The growth factor D(z), normalised to D(0) = 1.

        Parameters
        ----------
        z
            Redshift(s), dimensionless.
        species
            ``"cb"`` (CDM + baryons) or ``"tot"`` (total matter).

        Returns
        -------
        float or numpy.ndarray
            With the shape of ``z``.
        """
        z_arr = self._redshifts(z)
        out = self.solution.growth_factor(z_arr, species)
        return float(out) if out.ndim == 0 else out

    def growth_rate(self, z: Any, species: str = "cb") -> Any:
        r"""The growth rate :math:`f = d\ln D/d\ln a`.

        Parameters
        ----------
        z
            Redshift(s), dimensionless.
        species
            ``"cb"`` or ``"tot"``.

        Returns
        -------
        float or numpy.ndarray
            With the shape of ``z``.
        """
        z_arr = self._redshifts(z)
        out = self.solution.growth_rate(z_arr, species)
        return float(out) if out.ndim == 0 else out
