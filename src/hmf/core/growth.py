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
:class:`~hmf.core.transfer_models.CLASS`), and both stages have the same
``k_accuracy``, the growth comes from the transfer's run: one run for both.
Otherwise such a model makes its own run, at the growth stage's ``k_accuracy``
(see :mod:`hmf.core.accuracy`).

Library code working in plain arrays (see :mod:`hmf.core._kernels`) calls the
kernel-level entry points :meth:`Growth.growth_factor_kernel` and
:meth:`Growth.growth_rate_kernel`, which check z as the public methods do.
:attr:`Growth.solution` is the data behind them: it does not check z, and is NaN
outside the redshifts it covers.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any

import attrs
import numpy as np
import numpy.typing as npt
from astropy.cosmology import FLRW

# For the fully qualified annotations below.
import hmf.core.transfer

from ._fields import field
from .accuracy import KAccuracy
from .cache import DiskCache
from .domain import Domain, Interval, check_extent
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
    k_accuracy: KAccuracy = field(
        factory=KAccuracy,
        validator=attrs.validators.instance_of(KAccuracy),
        doc=(
            "The wavenumber accuracy; it sets the precision of a Boltzmann run the growth "
            "model makes itself (CambGrowth, ClassGrowth). Give the transfer stage's "
            "(from_transfer does), so that the run is shared."
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
        """The growth stage of a transfer stage, with its cosmology, accuracy and disk cache.

        Parameters
        ----------
        transfer
            The transfer stage.
        **kwargs
            Other fields (``model``, ...). ``k_accuracy`` and ``disk_cache`` default to
            the transfer stage's.

        Returns
        -------
        Growth
        """
        kwargs.setdefault("disk_cache", transfer.disk_cache)
        kwargs.setdefault("k_accuracy", transfer.k_accuracy)
        return cls(cosmology=transfer.cosmology, transfer=transfer, **kwargs)

    @cached_property
    def solution(self) -> GrowthSolution:
        """The growth of every species (one model solve): data, not an entry point.

        Its methods do not check z, and are NaN outside the redshifts it covers:
        library code calls :meth:`growth_factor_kernel` and :meth:`growth_rate_kernel`.
        """
        run = None
        if (
            self.transfer is not None
            and self.model.backend is not None
            and self.transfer.model.backend == self.model.backend
            and self.transfer.k_accuracy == self.k_accuracy
        ):
            run = self.transfer.boltzmann_run
        return self.model.solve(
            self.cosmology, run=run, disk_cache=self.disk_cache, k_accuracy=self.k_accuracy
        )

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

    @cached_property
    def _z_interval(self) -> Interval:
        """The redshifts that are both in the model's valid domain and in the table."""
        table = self._table_domain["z"]
        valid = type(self.model).valid_domain
        if "z" not in valid.variables:
            return table
        bound = valid["z"]
        lower, lower_open = max((bound.lower, bound.lower_open), (table.lower, table.lower_open))
        # The smaller upper bound; at equal bounds, open if either is.
        upper, upper_closed = min(
            (bound.upper, not bound.upper_open), (table.upper, not table.upper_open)
        )
        return Interval(lower, upper, lower_open=lower_open, upper_open=not upper_closed)

    def _redshifts_kernel(self, z: npt.ArrayLike, method: str) -> npt.NDArray[np.float64]:
        """Check plain redshifts as :meth:`_redshifts` does, from their extent only."""
        return check_extent(
            "z",
            z,
            self._z_interval,
            where=f"Growth.{method} ({type(self.model).__name__}: the model's valid "
            "domain and the solution's table)",
        )

    def growth_factor_kernel(
        self, z: npt.ArrayLike, species: str = "cb"
    ) -> npt.NDArray[np.float64]:
        """:meth:`growth_factor` at kernel level, on plain arrays.

        For library code (see :mod:`hmf.core._kernels`). It checks z as
        :meth:`growth_factor` does, from the smallest and largest z only.

        Parameters
        ----------
        z
            Redshift(s), dimensionless: a float or a plain array.
        species
            ``"cb"`` (CDM + baryons) or ``"tot"`` (total matter).

        Returns
        -------
        numpy.ndarray
            D(z)/D(0), dimensionless, with the shape of ``z``.

        Raises
        ------
        DomainError
            As for :meth:`growth_factor`, or if a z is not finite.
        """
        return self.solution.growth_factor(
            self._redshifts_kernel(z, "growth_factor_kernel"), species
        )

    def growth_rate_kernel(self, z: npt.ArrayLike, species: str = "cb") -> npt.NDArray[np.float64]:
        r""":meth:`growth_rate` at kernel level, on plain arrays.

        For library code (see :mod:`hmf.core._kernels`). It checks z as
        :meth:`growth_rate` does, from the smallest and largest z only.

        Parameters
        ----------
        z
            Redshift(s), dimensionless: a float or a plain array.
        species
            ``"cb"`` or ``"tot"``.

        Returns
        -------
        numpy.ndarray
            :math:`f = d\ln D/d\ln a`, dimensionless, with the shape of ``z``.

        Raises
        ------
        DomainError
            As for :meth:`growth_rate`, or if a z is not finite.
        """
        return self.solution.growth_rate(self._redshifts_kernel(z, "growth_rate_kernel"), species)

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
