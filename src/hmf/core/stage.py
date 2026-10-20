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
* **Parameters by name**: :meth:`Stage.evolve` and :meth:`Stage.from_flat` take the
  parameters of the whole stage tree by flat name, alias or dotted path, through a
  routing table built from the classes (see :mod:`hmf.core.routing`). A stage that
  holds stages declares its flat aliases in :attr:`Stage.parameter_aliases`, its
  derived fields in :attr:`Stage.derivations`, and its shared parameters with
  ``field(..., shared=True)``.
* **Introspection without instantiation**: :meth:`Stage.fields_info`,
  :meth:`Stage.parameter_info`, :meth:`Stage.quantities_available` and
  :meth:`Stage.invalidated_by` describe a stage tree from the classes alone.
* **Units**: public methods are :func:`~hmf.core.units.unit_boundary` methods. The
  base stage's ``_unit_context`` has no H0, so it accepts only h-units; a stage that
  can convert physical units (because it has a cosmology, or an H0) should override
  ``_unit_context`` to provide its H0.
* **Cosmology**: a stage computed from a cosmology subclasses
  :class:`CosmologyStage`, which holds the ``cosmology`` field and takes its units
  context from the cosmology's H0.
"""

from __future__ import annotations

import difflib
from collections.abc import Mapping
from functools import cached_property
from types import MappingProxyType
from typing import Any, ClassVar, Self

import attrs
from astropy.cosmology import FLRW, Planck18

from . import routing
from ._fields import Documented, field
from ._serialise import cosmology_key
from .cache import to_disk_cache
from .units import UnitContext

__all__ = ["CosmologyStage", "Stage"]

_NO_ALIASES: Mapping[str, str] = MappingProxyType({})

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

    #: Flat aliases of parameters of the stage tree: alias -> dotted path below this
    #: stage, for fields whose names are repeated in the tree or read badly flat
    #: (e.g. ``{"transfer_model": "transfer.model"}``). A stage holding this one
    #: routes the alias too.
    parameter_aliases: ClassVar[Mapping[str, str]] = _NO_ALIASES
    #: The fields of the stage tree computed from other fields, as
    #: :class:`~hmf.core.routing.Derivation` objects, with paths relative to this stage.
    #: :meth:`evolve` computes them again when what they are derived from changes.
    derivations: ClassVar[tuple[routing.Derivation, ...]] = ()

    def evolve(self, **changes: Any) -> Self:
        """Return a copy of this stage tree with some parameters changed.

        Parameters are named by their flat name, a flat alias, a shared name or their
        dotted path (see :mod:`hmf.core.routing` and :meth:`parameter_names`); a
        dotted path into a model field (``"fit.A"``) changes that field of the model,
        and a stage's dotted path replaces that stage. Only the stages on the path
        from each changed field to this one are rebuilt, each by its
        :meth:`evolve_own`; every other stage is shared, with its cached results.
        Derived fields (:attr:`derivations`) are computed again when what they are
        derived from changes.

        Every name is checked before anything is built. The new stages are built,
        and so validated, by their constructors; if that fails, the error propagates
        and this stage, which is immutable, is unchanged.

        Parameters
        ----------
        **changes
            New values of parameters, by name or dotted path.

        Returns
        -------
        Stage
            A new stage of the same class (this one, if nothing changes).

        Raises
        ------
        TypeError
            If a name is not a parameter (with suggestions), is ambiguous (with the
            dotted paths to choose from), is derived, or is given twice.
        """
        return routing.evolve(self, changes)  # type: ignore[return-value]

    def evolve_own(self, **changes: Any) -> Self:
        """Return a copy of this stage with some of its own fields changed.

        No routing: each name is a field of this stage, and nothing else changes. It
        is the step :meth:`evolve` takes at each stage it rebuilds, which a stage may
        override (e.g. to share a memo the change does not affect). Cached results
        are not copied: the new stage computes its own.

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
        fields = routing.field_names(type(self))
        unknown = [name for name in changes if name not in fields]
        if unknown:
            names = list(fields)
            msg = f"{type(self).__name__} has no field(s) {unknown}."
            close = difflib.get_close_matches(unknown[0], names, n=3)
            if close:
                msg += f" Did you mean {' or '.join(repr(c) for c in close)}?"
            msg += f" Fields: {names}."
            raise TypeError(msg)
        return attrs.evolve(self, **changes)

    @classmethod
    def from_flat(cls, mapping: Mapping[str, Any]) -> Self:
        """Build the stage tree from parameters by name.

        Names are those of :meth:`evolve`; a mapping value of a stage's dotted path
        (``{"linear_power": {"sigma_8": 0.8}}``) gives the fields below that stage.
        Every stage of the tree that is not given is built from its parameters, and
        every derived field is computed. Parameters not given take their field's
        default, or the one :meth:`computed_defaults` gives.

        Parameters
        ----------
        mapping
            Values of parameters, by name or dotted path, or nested mappings.

        Returns
        -------
        Stage

        Raises
        ------
        TypeError
            For an unknown, ambiguous, derived or repeated name, or a required
            parameter that is not given.
        """
        return routing.from_flat(cls, mapping)  # type: ignore[return-value]

    @classmethod
    def computed_defaults(cls, given: routing.GivenParameters) -> Mapping[str, Any]:
        """Defaults of :meth:`from_flat` that are computed from the given parameters.

        A hook for stages whose defaults depend on other parameters (e.g. sigma_8 from
        the cosmology); the base stage has none. Defaults for parameters (or below
        stages) that are given are ignored.

        Parameters
        ----------
        given
            The parameters given, by dotted path.

        Returns
        -------
        Mapping
            Values by name or dotted path.
        """
        return {}

    def to_flat(self) -> dict[str, Any]:
        """The value of every parameter of the stage tree, by the name it is routed by.

        ``type(stage).from_flat(stage.to_flat())`` equals the stage, and the keys are
        :meth:`parameter_names`.

        Returns
        -------
        dict
        """
        info = routing.router(type(self)).parameter_info
        return {name: _walk_path(self, p.path) for name, p in info.items()}

    @classmethod
    def parameter_info(cls) -> dict[str, routing.ParameterInfo]:
        """Describe every parameter of the stage tree, without creating a stage.

        Returns
        -------
        dict
            A :class:`~hmf.core.routing.ParameterInfo` (dotted paths, type, default,
            doc, whether it is shared) by the name the parameter is routed by.
        """
        return dict(routing.router(cls).parameter_info)

    @classmethod
    def parameter_names(cls) -> tuple[str, ...]:
        """The names of the parameters of the stage tree (see :meth:`parameter_info`).

        Returns
        -------
        tuple of str
        """
        return tuple(routing.router(cls).parameter_info)

    @classmethod
    def parameter_defaults(cls) -> dict[str, Any]:
        """The default of each parameter that has one, without creating a stage.

        A default computed by a factory that does not depend on the stage is called
        (it gives a model or a setting, e.g. ``KAccuracy()``), and the
        :meth:`computed_defaults` of the stages of the tree, from no given
        parameters, are included.
        ``from_flat(parameter_defaults())`` equals ``from_flat({})``.

        Returns
        -------
        dict
        """
        return routing.parameter_defaults(cls)

    @classmethod
    def quantities_available(cls) -> dict[str, tuple[str, ...]]:
        """The public outputs of each stage of the tree.

        From the classes alone: each stage's :func:`~hmf.core.units.unit_boundary`
        methods (e.g. ``dndm``) and its public properties and cached properties.
        Other methods (e.g. ``MassFunction.at``, which gives a view, and the
        plain-array ``..._kernel`` entry points for library code), fields, private
        names and constants are left out.

        Returns
        -------
        dict
            The names, by the dotted path of the stage (``""`` for this one).
        """
        return routing.router(cls).quantities_available()

    @classmethod
    def invalidated_by(cls, name: str) -> tuple[str, ...]:
        """The stages of the tree that a change of a parameter rebuilds.

        The stage holding it and every stage above it, and the stages holding a
        derived field that is computed from it (and the stages above those), if the
        derived links hold. Every other stage is shared with the original tree. It is
        an upper bound: :meth:`evolve` keeps a stage whose derived field does not
        change (e.g. after a ``disk_cache`` change, which is not part of any stage's
        value).

        Parameters
        ----------
        name
            A parameter, by any name :meth:`evolve` accepts.

        Returns
        -------
        tuple of str
            The dotted paths of the stages (``""`` for this one), from the top.

        Raises
        ------
        TypeError
            If the name is not a parameter, or is ambiguous.
        """
        return routing.router(cls).invalidated_by(name)

    @cached_property
    def _unit_context(self) -> UnitContext:
        """The units context of this stage's boundary methods (see :mod:`hmf.core.units`).

        The base stage has no H0, so it only accepts inputs in h-units. A stage that
        can convert physical units should override this to return ``UnitContext(H0)``.
        """
        return UnitContext(None)


def _walk_path(obj: Any, dotted: str) -> Any:
    """The value at a dotted path below ``obj``."""
    for name in dotted.split("."):
        obj = getattr(obj, name)
    return obj


def _cosmology_field(*, also: str = "") -> Any:
    """The ``cosmology`` field of a :class:`CosmologyStage`.

    It defaults to Planck18, must be an astropy FLRW, and is compared and hashed by
    :func:`~hmf.core._serialise.cosmology_key`. It is a shared parameter. A subclass
    whose cosmology has a further constraint redefines the field with this, stating
    the constraint in ``also``.

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
        shared=True,
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
    return field(default=None, converter=to_disk_cache, eq=False, shared=True, doc=doc)


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
