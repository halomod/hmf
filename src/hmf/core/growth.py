r"""The :class:`Growth` stage: the linear growth factor and growth rate as functions of z.

The stage holds a cosmology and a growth model, and optionally the
:class:`~hmf.core.transfer.Transfer` stage it goes with. It gives, for each matter
species, the growth factor D(z) (normalised to D(0) = 1) and the growth rate
:math:`f = d\ln D/d\ln a`. Redshifts are dimensionless and broadcast like numpy:
a scalar gives a scalar (a :class:`numpy.float64`), an array an array of the same
shape.

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
import numpy as np
from astropy.cosmology import FLRW

# For the fully qualified annotations below.
import hmf.core.transfer

from ._fields import field
from .cache import DiskCache
from .domain import Domain
from .growth_models import GrowthModel, GrowthSolution, ODEGrowth
from .stage import CosmologyStage, _check_model_cosmology, _cosmology_field, _disk_cache_field
from .transfer import Transfer
from .units import unit_boundary

__all__ = ["Growth"]


@attrs.frozen(kw_only=True)
class Growth(CosmologyStage):
    """The linear growth factor and growth rate.

    See the module documentation.

    Examples
    --------
    >>> from hmf.core.growth import Growth
    >>> g = Growth()
    >>> float(g.growth_factor(0.0))
    1.0
    """

    cosmology: FLRW = _cosmology_field(
        also="It must equal the transfer stage's, if one is given (see from_transfer)."
    )
    model: GrowthModel = field(
        factory=ODEGrowth,
        converter=GrowthModel.coerce,
        validator=_check_model_cosmology,
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
    disk_cache: DiskCache | None = _disk_cache_field(
        doc=(
            "Where to cache Boltzmann-code output on disk (see Transfer.disk_cache), for "
            "a run the growth model makes itself."
        ),
    )

    @transfer.validator
    def _check_transfer(
        self, attribute: attrs.Attribute[Transfer | None], value: Transfer | None
    ) -> None:
        """Check that the transfer stage has the same cosmology."""
        if value is not None and not self.same_cosmology(value):
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

    @cached_property
    def _table_domain(self) -> Domain:
        """The redshifts the solution covers: 0 <= z <= z_max (to round-off)."""
        return Domain(
            {"z": (0.0, self.solution.z_max * (1 + 1e-12))},
            source=f"The redshifts {type(self.model).__name__} is solved at.",
        )

    def _redshifts(self, z: Any) -> Any:
        """Check z against the model's valid domain and the solution's table.

        The domain is that of the model's class, so a model can narrow it. Neither is
        extrapolated: z outside either raises.
        """
        name = type(self.model).__name__
        type(self.model).valid_domain.check({"z": z}, where=f"Growth ({name})")
        self._table_domain.check({"z": z}, where=f"Growth ({name}), the solution's table")
        return np.asarray(z, dtype=float)

    @unit_boundary()
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
        numpy.float64 or numpy.ndarray
            With the shape of ``z`` (a scalar for a scalar ``z``).

        Raises
        ------
        DomainError
            If a z is outside the model's valid domain (z >= 0), or above the largest
            redshift its solution covers (e.g. a table's).
        """
        z_arr = self._redshifts(z)
        return self.solution.growth_factor(z_arr, species)

    @unit_boundary()
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
        numpy.float64 or numpy.ndarray
            With the shape of ``z`` (a scalar for a scalar ``z``).

        Raises
        ------
        DomainError
            As for :meth:`growth_factor`.
        """
        z_arr = self._redshifts(z)
        return self.solution.growth_rate(z_arr, species)
