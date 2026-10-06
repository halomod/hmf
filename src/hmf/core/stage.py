"""The :class:`Stage` base class.

A *stage* is one step of a calculation (transfer function, growth, linear power,
mass variance, mass function, ...). Stages are immutable, keyword-only ``attrs``
classes, composed rather than inherited: each stage holds the stage it depends on
as a field. The conventions every stage follows:

* **Immutable.** Stages are ``@attrs.frozen(kw_only=True)``. A changed parameter
  gives a new stage, through :meth:`Stage.evolve`, which is atomic: the new stage is
  validated by its constructor, and if that fails the original is untouched (it is
  never modified in the first place).
* **Validation in the constructor**, with ``attrs`` validators (or
  ``__attrs_post_init__``), so no invalid stage can exist.
* **Expensive results are** :func:`functools.cached_property`. ``attrs`` (≥ 24.1)
  turns these into slots on slotted classes, so a stage keeps no ``__dict__``.
  Because stages are immutable, a cached result can never go stale; and because
  ``evolve()`` shares the unchanged sub-stages, their caches carry over.
* **Introspection without instantiation**: :meth:`Stage.fields_info` lists the fields
  from the class alone.
* **Units**: public methods are :func:`~hmf.core.units.unit_boundary` methods. The
  base stage's ``_unit_context`` has no H0, so it accepts only h-units; a stage that
  can convert physical units (because it has a cosmology, or an H0) should override
  ``_unit_context`` to provide its H0.
* **Cosmology**: a stage computed from a cosmology subclasses
  :class:`CosmologyStage`, which holds the ``cosmology`` field and takes its units
  context from the cosmology's H0.

Routing flat parameter names through a tree of stages (issue #383) is not part of
this base class yet.
"""

from __future__ import annotations

import difflib
from functools import cached_property
from typing import Any, Self

import attrs
from astropy.cosmology import FLRW, Planck18

from ._fields import Documented, field
from ._serialise import cosmology_key
from .cache import to_disk_cache
from .units import UnitContext

__all__ = ["CosmologyStage", "Stage"]

#: The documentation of the ``cosmology`` field of a :class:`CosmologyStage`.
_COSMOLOGY_DOC = (
    "The cosmology, an astropy FLRW. It is compared by its class and parameter values "
    "(not its name or metadata)."
)


@attrs.frozen(kw_only=True)
class Stage(Documented):
    """Base class of every v4 calculation stage.

    Subclasses must be decorated with ``@attrs.frozen(kw_only=True)``, and get the
    "Parameters" section of their docstring generated from their fields (defined
    with :func:`hmf.core.field`).

    Examples
    --------
    >>> import attrs
    >>> from functools import cached_property
    >>> from hmf.core import field
    >>> @attrs.frozen(kw_only=True)
    ... class Squares(Stage):
    ...     n: int = field(default=3, doc="How many squares.")
    ...
    ...     @cached_property
    ...     def values(self) -> list[int]:
    ...         return [i**2 for i in range(self.n)]
    >>> s = Squares()
    >>> s.values
    [0, 1, 4]
    >>> s.evolve(n=4).values
    [0, 1, 4, 9]
    """

    def evolve(self, **changes: Any) -> Self:
        """Return a copy of this stage with some fields changed.

        The copy is built (and so validated) by the constructor. If that fails, the
        error propagates and this stage, which is immutable, is unchanged. Cached
        results are not copied: the new stage computes its own.

        Parameters
        ----------
        **changes
            New values of fields, by name.

        Returns
        -------
        Stage
            A new stage of the same class.

        Raises
        ------
        TypeError
            If a name is not a field of this stage (with suggestions).
        """
        names = [info.name for info in self.fields_info()]
        unknown = [name for name in changes if name not in names]
        if unknown:
            msg = f"{type(self).__name__} has no field(s) {unknown}."
            close = difflib.get_close_matches(unknown[0], names, n=3)
            if close:
                msg += f" Did you mean {' or '.join(repr(c) for c in close)}?"
            msg += f" Fields: {names}."
            raise TypeError(msg)
        return attrs.evolve(self, **changes)

    @cached_property
    def _unit_context(self) -> UnitContext:
        """The units context of this stage's boundary methods (see :mod:`hmf.core.units`).

        The base stage has no H0, so it only accepts inputs in h-units. A stage that
        can convert physical units should override this to return ``UnitContext(H0)``.
        """
        return UnitContext(None)


def _cosmology_field(*, also: str = "") -> Any:
    """The ``cosmology`` field of a :class:`CosmologyStage`.

    It defaults to Planck18, must be an astropy FLRW, and is compared and hashed by
    :func:`~hmf.core._serialise.cosmology_key`. A subclass whose cosmology has a
    further constraint redefines the field with this, stating the constraint in
    ``also``.

    Parameters
    ----------
    also
        Further documentation, appended to the field's.

    Returns
    -------
    Any
        The field definition.
    """
    return field(
        default=Planck18,
        validator=attrs.validators.instance_of(FLRW),
        eq=cosmology_key,
        doc=f"{_COSMOLOGY_DOC} {also}" if also else _COSMOLOGY_DOC,
    )


def _disk_cache_field(*, doc: str) -> Any:
    """A ``disk_cache`` field: where to cache Boltzmann-code output on disk.

    It is converted by :func:`~hmf.core.cache.to_disk_cache` (a DiskCache, True for
    the default directory, a directory, or None), defaults to None, and is not part of
    the stage's value (``eq=False``), since it changes no result.

    Parameters
    ----------
    doc
        The field's documentation.

    Returns
    -------
    Any
        The field definition.
    """
    return field(default=None, converter=to_disk_cache, eq=False, doc=doc)


def _check_model_cosmology(
    instance: CosmologyStage, attribute: attrs.Attribute[Any], value: Any
) -> None:
    """Validate a model field: the model must apply to the stage's cosmology.

    Use it as the ``validator`` of a field holding a model with a
    ``check_cosmology(cosmology)`` method, which raises if it does not apply.
    """
    value.check_cosmology(instance.cosmology)


@attrs.frozen(kw_only=True)
class CosmologyStage(Stage):
    """Base class of the stages computed from a cosmology.

    It holds the ``cosmology`` field (an astropy FLRW, compared and hashed by its
    class and parameter values, see :func:`~hmf.core._serialise.cosmology_key`), and
    converts physical units with the cosmology's H0. Subclasses must be decorated with
    ``@attrs.frozen(kw_only=True)``.
    """

    cosmology: FLRW = _cosmology_field()

    @cached_property
    def _unit_context(self) -> UnitContext:
        """Units context with this stage's H0, to convert physical units to h-units."""
        return UnitContext(self.cosmology.H0)

    def same_cosmology(self, other: CosmologyStage) -> bool:
        """Whether ``other`` has the same cosmology as this stage.

        Cosmologies are compared by class and parameter values (see
        :func:`~hmf.core._serialise.cosmology_key`), not by name or metadata.

        Parameters
        ----------
        other
            Another stage with a cosmology.

        Returns
        -------
        bool
        """
        return cosmology_key(self.cosmology) == cosmology_key(other.cosmology)
