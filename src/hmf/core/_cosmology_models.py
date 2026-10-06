"""Mixins of the model kinds whose models are computed from a cosmology.

:class:`_CosmologyModel` holds the class-level scaffolding that such a kind
(:class:`~hmf.core.transfer_models.TransferModel`,
:class:`~hmf.core.growth_models.GrowthModel`) declares: the Boltzmann ``backend``,
the ``valid_domain`` and ``calibration_domain``, and the ``check_cosmology`` hook.
:class:`_BoltzmannBacked` is the cosmology check of the models computed by a
Boltzmann code (CAMB or CLASS), for transfer and growth alike.

Both are plain mixins (with empty ``__slots__``), not models: a kind lists one before
:class:`~hmf.core.model.Model` (or before its own kind) in its bases.
"""

from __future__ import annotations

from typing import ClassVar

from astropy.cosmology import FLRW

from ._boltzmann import check_boltzmann_cosmology
from .domain import Domain


class _CosmologyModel:
    """The class-level scaffolding of a kind of models computed from a cosmology.

    The kind sets :attr:`valid_domain`; the other class variables default to
    "none".
    """

    __slots__ = ()

    #: The Boltzmann code that computes the model (``"camb"`` or ``"class"``), if any.
    backend: ClassVar[str | None] = None

    #: Where the model can be evaluated. The stage checks its inputs against the
    #: model's class's domain, so a model can narrow it.
    valid_domain: ClassVar[Domain]

    #: Where the model was calibrated, if that is stated by its source.
    calibration_domain: ClassVar[Domain | None] = None

    def check_cosmology(self, cosmology: FLRW) -> None:
        """Raise if the model does not apply to ``cosmology`` (by default it does).

        Raises
        ------
        ValueError
            If it does not apply: a model that does not apply to a cosmology is a
            configuration error.
        """


class _BoltzmannBacked:
    """The cosmology check of a model computed by a Boltzmann code (CAMB or CLASS)."""

    __slots__ = ()

    def check_cosmology(self, cosmology: FLRW) -> None:
        """Raise if the Boltzmann code can't compute ``cosmology``.

        Raises
        ------
        ValueError
            If ``cosmology`` is not a LambdaCDM, wCDM or w0waCDM (or a flat one), or
            does not set the baryon density or the CMB temperature.
        """
        check_boltzmann_cosmology(cosmology, type(self).__name__)
