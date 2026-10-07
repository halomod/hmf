"""The :class:`Transfer` stage: the transfer function and unnormalised P(k) at z = 0.

The stage holds a cosmology, a transfer model and the wavenumber accuracy
(``k_accuracy``). It solves the model once (one Boltzmann run, for a CAMB or CLASS
model) and gives, for each matter species (``"cb"``, CDM + baryons; ``"tot"``, total matter):

* :meth:`Transfer.transfer_function`: T(k), normalised to 1 as k -> 0;
* :meth:`Transfer.unnormalised_power`: :math:`k^{n_s} T(k)^2`, the shape of the
  linear power spectrum at z = 0. It is a plain (dimensionless) array, with k in
  h/Mpc and an arbitrary amplitude: normalising it, e.g. to sigma_8, and giving it
  the units of a power spectrum, is the job of a later stage;
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

import astropy.units as u
import attrs
import numpy as np
import numpy.typing as npt

from ._boltzmann import BoltzmannRun
from ._fields import field
from ._species import check_species
from .accuracy import KAccuracy
from .cache import DiskCache
from .domain import warn_once
from .stage import CosmologyStage, _check_model_cosmology, _disk_cache_field
from .transfer_models import CAMB, TransferModel, TransferSolution
from .units import h_Mpc, unit_boundary

__all__ = ["Transfer", "UnnormalisedPower"]

Array = npt.NDArray[np.float64]


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
class Transfer(CosmologyStage):
    """The transfer function and the shape of the linear power spectrum at z = 0.

    See the module documentation.

    Examples
    --------
    >>> from hmf.core.transfer import Transfer
    >>> from hmf.core.units import h_Mpc
    >>> t = Transfer(model="EH_NoBAO")
    >>> float(round(t.transfer_function(1e-7 * h_Mpc), 6))
    1.0
    """

    model: TransferModel = field(
        factory=CAMB,
        converter=TransferModel.coerce,
        validator=_check_model_cosmology,
        doc="The transfer model: an instance, a class or a registered name (e.g. 'EH').",
    )
    n_s: float = field(
        default=0.9665,
        converter=float,
        doc="The spectral index of the primordial power spectrum (default: Planck18).",
    )
    k_accuracy: KAccuracy = field(
        factory=KAccuracy,
        validator=attrs.validators.instance_of(KAccuracy),
        doc="The wavenumber accuracy; it sets the sampling of the Boltzmann codes.",
    )
    disk_cache: DiskCache | None = _disk_cache_field(
        doc=(
            "Where to cache the Boltzmann code's output on disk: a DiskCache, True (the "
            "default directory), a directory, or None (no disk cache). It does not change "
            "any result, so it is not part of the stage's value."
        ),
    )

    @cached_property
    def solution(self) -> TransferSolution:
        """The kernel-level transfer function of every species (one model solve)."""
        return self.model.solve(self.cosmology, self.k_accuracy, disk_cache=self.disk_cache)

    @property
    def boltzmann_run(self) -> BoltzmannRun | None:
        """The Boltzmann run behind the solution, if the model has one."""
        return self.solution.run

    @property
    def k_max_table(self) -> u.Quantity | None:
        """The largest tabulated wavenumber, above which T is extrapolated.

        A Quantity in h/Mpc; ``None`` for a model that is not a table (a fitting
        formula).
        """
        k_max = self.solution.k_max_table
        return None if k_max is None else u.Quantity(k_max, h_Mpc)

    def _check_k(self, k: Array) -> None:
        """Check k (in h/Mpc) against the model's valid domain, and warn if extrapolated.

        The domain is that of the model's class, so a model can narrow it. A table the
        user supplied (``FromArray``, ``FromFile``) warns, once per stage, if k is
        outside it; the Boltzmann codes' tables are extrapolated by design, silently.
        """
        model = type(self.model)
        model.valid_domain.check({"k": k << h_Mpc}, where=f"Transfer ({model.__name__})")
        if not model._user_table:
            return
        k_min, k_max = self.solution.k_min_table, self.solution.k_max_table
        # A table has both.
        assert k_min is not None
        assert k_max is not None
        for side, outside, bound in (("below", k < k_min, k_min), ("above", k > k_max, k_max)):
            if np.any(outside):
                warn_once(
                    self,
                    f"k {side} the table",
                    f"Transfer ({model.__name__}): k {side} the table's "
                    f"{'smallest' if side == 'below' else 'largest'} wavenumber, "
                    f"{bound:.4g} h/Mpc, is extrapolated (with the shape of EH98).",
                    stacklevel=4,  # the caller of the public method
                )

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
        numpy.float64 or numpy.ndarray
            Dimensionless, with the shape of ``k`` (a scalar for a scalar ``k``).

        Raises
        ------
        DomainError
            If a k is outside the model's valid domain (k > 0).

        Warns
        -----
        HMFExtrapolationWarning
            If a k is outside a table the user supplied (once per stage).
        """
        self._check_k(k)
        return self.solution.transfer(k, species)

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
        numpy.float64 or numpy.ndarray
            A plain, dimensionless array (not a Quantity), with the shape of ``k`` (a
            scalar for a scalar ``k``): :math:`k^{n_s} T(k)^2` with k in h/Mpc, so its
            amplitude is arbitrary. A later stage normalises it (e.g. to sigma_8) and
            gives it the units of a power spectrum, (Mpc/h)³.

        Raises
        ------
        DomainError
            As for :meth:`transfer_function`.

        Warns
        -----
        HMFExtrapolationWarning
            As for :meth:`transfer_function`.
        """
        self._check_k(k)
        return self.power_kernel(species).power(k)

    def power_kernel(self, species: str = "cb") -> UnnormalisedPower:
        """The kernel-level shape of the power spectrum of ``species``.

        For library code working in canonical units: its ``power(k)`` takes k in h/Mpc
        as a plain array.
        """
        return UnnormalisedPower(self.solution, check_species(species), self.n_s)
