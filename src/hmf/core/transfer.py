"""The :class:`Transfer` stage: the transfer function and unnormalised P(k) at z = 0.

The stage holds a cosmology, a transfer model and the wavenumber accuracy. It solves
the model once (one Boltzmann run, for a CAMB or CLASS model) and gives, for each
matter species (``"cb"``, CDM + baryons; ``"tot"``, total matter):

* :meth:`Transfer.transfer_function`: T(k), normalised to 1 as k -> 0;
* :meth:`Transfer.unnormalised_power`: :math:`k^{n_s} T(k)^2`, the shape of the
  linear power spectrum at z = 0 (normalising it, e.g. to sigma_8, is the job of a
  later stage);
* :meth:`Transfer.power_kernel`: the same at kernel level, as a pure function of k
  in h/Mpc on plain arrays, for later stages.

Every species comes from the same solution, so asking for the power spectrum of one
species and normalising with another never runs the Boltzmann code twice. Runs are
also memoised by the content of their input, so stages that differ only in ``n_s``
(or in fields the run does not depend on) share one run, and they can be cached on
disk across processes (``disk_cache``).

The cosmology field
-------------------
The cosmology is an astropy :class:`~astropy.cosmology.FLRW`. Astropy cosmologies are
immutable and compare by value, but are not hashable, so the field is compared and
hashed by :func:`~hmf.core._serialise.cosmology_key`: its class and parameter values,
without its name and metadata. Two stages with equal parameters are therefore equal
(and hash equal) even if their cosmologies have different names.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any

import attrs
import numpy as np
import numpy.typing as npt
from astropy.cosmology import FLRW, Planck18

from ._boltzmann import BoltzmannRun
from ._fields import field
from ._serialise import cosmology_key
from ._species import check_species
from ._validators import check_in_range
from .accuracy import KAccuracy
from .cache import DiskCache, to_disk_cache
from .domain import DomainError
from .stage import Stage
from .transfer_models import CAMB, TransferModel, TransferSolution
from .units import UnitContext, h_Mpc, unit_boundary

__all__ = ["Transfer", "UnnormalisedPower"]

Array = npt.NDArray[np.float64]


def _to_transfer_model(value: Any) -> TransferModel:
    """Convert the ``model`` field: a model, a model class, or a name to look up."""
    if isinstance(value, TransferModel):
        return value
    if isinstance(value, (str, type)):
        return TransferModel.get(value)()
    raise TypeError(
        f"model must be a TransferModel, a class or a name, not {type(value).__name__}."
    )


def _scalar_like(out: Array, like: Any) -> Any:
    """A float if ``like`` is 0-d, else ``out``."""
    return float(out) if np.ndim(like) == 0 else out


def _check_k(k: Array) -> None:
    # Note: this accepts k = inf, unlike the other checks of array inputs in hmf.core,
    # which require them to be finite too. Whether it should is still to be decided.
    check_in_range("k", k, where="Transfer", low=0.0, low_open=True, error=DomainError)


@attrs.frozen(eq=False)
class UnnormalisedPower:
    r"""The shape of the linear power spectrum of one species at z = 0, at kernel level.

    :math:`P(k) \propto k^{n_s} T(k)^2`, with k in h/Mpc, on plain arrays. It is a
    pure function: it has no state besides its (immutable) solution.
    """

    #: The transfer solution.
    solution: TransferSolution
    #: The species.
    species: str
    #: The spectral index.
    n_s: float

    def ln_power(self, ln_k: Array) -> Array:
        """Ln of :math:`k^{n_s} T^2` at ``ln k`` (k in h/Mpc)."""
        ln_k = np.asarray(ln_k, dtype=float)
        out: Array = self.n_s * ln_k + 2 * self.solution.ln_transfer(np.exp(ln_k), self.species)
        return out

    def power(self, k: Array) -> Array:
        """:math:`k^{n_s} T(k)^2` at ``k`` (in h/Mpc)."""
        k = np.asarray(k, dtype=float)
        out: Array = k**self.n_s * self.solution.transfer(k, self.species) ** 2
        return out

    __call__ = power


@attrs.frozen(kw_only=True)
class Transfer(Stage):
    """The transfer function and the shape of the linear power spectrum at z = 0.

    See the module documentation.

    Examples
    --------
    >>> from hmf.core.transfer import Transfer
    >>> from hmf.core.units import h_Mpc
    >>> t = Transfer(model="EH_NoBAO")
    >>> round(t.transfer_function(1e-7 * h_Mpc), 6)
    1.0
    """

    cosmology: FLRW = field(
        default=Planck18,
        validator=attrs.validators.instance_of(FLRW),
        eq=cosmology_key,
        doc=(
            "The cosmology, an astropy FLRW. It is compared by its class and parameter "
            "values (not its name or metadata)."
        ),
    )
    model: TransferModel = field(
        factory=CAMB,
        converter=_to_transfer_model,
        doc="The transfer model: an instance, a class or a registered name (e.g. 'EH').",
    )
    n_s: float = field(
        default=0.9665,
        converter=float,
        doc="The spectral index of the primordial power spectrum (default: Planck18).",
    )
    accuracy: KAccuracy = field(
        factory=KAccuracy,
        validator=attrs.validators.instance_of(KAccuracy),
        doc="The wavenumber accuracy; it sets the sampling of the Boltzmann codes.",
    )
    disk_cache: DiskCache | None = field(
        default=None,
        converter=to_disk_cache,
        eq=False,
        doc=(
            "Where to cache the Boltzmann code's output on disk: a DiskCache, True (the "
            "default directory), a directory, or None (no disk cache). It does not change "
            "any result, so it is not part of the stage's value."
        ),
    )

    @model.validator
    def _check_model(self, attribute: attrs.Attribute[TransferModel], value: TransferModel) -> None:
        """Check that the model applies to the cosmology."""
        value.check_cosmology(self.cosmology)

    @cached_property
    def _unit_context(self) -> UnitContext:
        """Units context with this stage's H0, to convert physical 1/Mpc to h/Mpc."""
        return UnitContext(self.cosmology.H0)

    @cached_property
    def solution(self) -> TransferSolution:
        """The kernel-level transfer function of every species (one model solve)."""
        return self.model.solve(self.cosmology, self.accuracy, disk_cache=self.disk_cache)

    @property
    def boltzmann_run(self) -> BoltzmannRun | None:
        """The Boltzmann run behind the solution, if the model has one."""
        return self.solution.run

    @property
    def k_max_table(self) -> float | None:
        """The largest tabulated wavenumber (h/Mpc), above which T is extrapolated."""
        return self.solution.k_max_table

    @unit_boundary(k=h_Mpc)
    def transfer_function(self, k: Any, species: str = "cb") -> Any:
        """The transfer function T(k) at z = 0, normalised to 1 as k -> 0.

        Parameters
        ----------
        k
            Wavenumbers: a Quantity in h/Mpc, or in 1/Mpc (converted with the
            cosmology's H0).
        species
            ``"cb"`` (CDM + baryons) or ``"tot"`` (total matter).

        Returns
        -------
        float or numpy.ndarray
            Dimensionless, with the shape of ``k``.
        """
        _check_k(k)
        return _scalar_like(self.solution.transfer(k, species), k)

    @unit_boundary(k=h_Mpc)
    def unnormalised_power(self, k: Any, species: str = "cb") -> Any:
        """The shape of the linear power spectrum at z = 0, :math:`k^{n_s} T(k)^2`.

        Parameters
        ----------
        k
            Wavenumbers: a Quantity in h/Mpc, or in 1/Mpc.
        species
            ``"cb"`` or ``"tot"``.

        Returns
        -------
        float or numpy.ndarray
            In arbitrary units (k in h/Mpc), with the shape of ``k``. It is normalised
            by a later stage.
        """
        _check_k(k)
        return _scalar_like(self.power_kernel(species).power(k), k)

    def power_kernel(self, species: str = "cb") -> UnnormalisedPower:
        """The kernel-level shape of the power spectrum of ``species``.

        For library code working in canonical units: its ``power(k)`` takes k in h/Mpc
        as a plain array.
        """
        return UnnormalisedPower(self.solution, check_species(species), self.n_s)
