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
* **Derived stages** (a stage computed from others, e.g. a mass function's mass
  variance) are :func:`derived_property` cached properties, not fields. Each
  declares the fields it is computed from, and :meth:`Stage.evolve_own` carries it,
  with its caches, to the new stage when none of them changes.
* **Parameters by name**: :meth:`Stage.evolve` and :meth:`Stage.from_flat` take the
  parameters of the whole stage tree by flat name or (a unique suffix of their)
  dotted path, through a routing table built from the classes (see
  :mod:`hmf.core.routing`). A stage that holds stages declares the names of its
  subtree that are one parameter in :attr:`Stage.shared_parameters`.
* **Introspection without instantiation**: :meth:`Stage.fields_info`,
  :meth:`Stage.parameter_info`, :meth:`Stage.quantities_available` and
  :meth:`Stage.invalidated_by` describe a stage tree from the classes alone.
* **Units**: public methods are :func:`~hmf.core.units.unit_boundary` methods. The
  base stage's ``_unit_context`` has no H0, so it accepts only h-units; a stage that
  can convert physical units (because it has a cosmology, or an H0) should override
  ``_unit_context`` to provide its H0.
* **Pickling and copying**: a stage pickles as its configuration
  (:mod:`hmf.core.config`), with the Boltzmann-code runs it has used as plain arrays,
  never as live objects. Because it is immutable, :func:`copy.copy` and
  :func:`copy.deepcopy` return the stage itself, with its caches.
* **Cosmology**: a stage computed from a cosmology subclasses
  :class:`CosmologyStage`, which holds the ``cosmology`` field and takes its units
  context from the cosmology's H0.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Mapping
from functools import cached_property
from types import MappingProxyType
from typing import Any, ClassVar, Self, TypeVar

import attrs
from astropy.cosmology import FLRW, Planck18

from . import routing
from ._fields import Documented, field
from ._serialise import cosmology_key
from .cache import to_disk_cache
from .units import UnitContext

__all__ = [
    "CosmologyStage",
    "DerivedProperty",
    "Stage",
    "derived_property",
    "undeclared_inputs",
]

_T = TypeVar("_T")

#: The class attribute holding a stage class's derived properties (see
#: :func:`derived_property`), by name.
DERIVED_ATTRIBUTE = "_derived_properties"


class DerivedProperty(cached_property[_T]):
    """A cached property computed from some fields of its stage (see :func:`derived_property`).

    Parameters
    ----------
    func
        Computes the value from the stage.
    inputs
        The dotted paths, below the stage, of the fields it is computed from.
    """

    def __init__(self, func: Callable[[Any], _T], inputs: tuple[str, ...]) -> None:
        super().__init__(func)
        self.inputs = inputs

    def __set_name__(self, owner: type, name: str) -> None:
        """Record the property and its inputs on the class (``attrs`` removes the descriptor)."""
        super().__set_name__(owner, name)
        own = owner.__dict__.get(DERIVED_ATTRIBUTE)
        if own is None:
            own = {}
            for base in reversed(owner.__mro__[1:]):
                own.update(base.__dict__.get(DERIVED_ATTRIBUTE, {}))
            setattr(owner, DERIVED_ATTRIBUTE, own)
        own[name] = self


def derived_property(*inputs: str) -> Callable[[Callable[[Any], _T]], DerivedProperty[_T]]:
    """Declare a cached property of a stage computed from some of its fields.

    The property is computed once per stage, like a :func:`functools.cached_property`.
    :meth:`Stage.evolve_own` carries its value, once computed, to the new stage when
    none of its inputs changes (each is the same object, or equal), so that the
    caches of a derived stage (e.g. the mass variance of a mass function) survive a
    change that does not affect it.

    Parameters
    ----------
    *inputs
        The dotted paths, below the stage, of the fields the value is computed from
        (e.g. ``"linear_power.transfer"``, ``"filter"``). Everything the value depends
        on must be in them, or below one of them.

    Returns
    -------
    callable
        The decorator.

    Examples
    --------
    >>> import attrs
    >>> from hmf.core import field
    >>> @attrs.frozen(kw_only=True)
    ... class Sum(Stage):
    ...     a: int = field(default=1, doc="One term.")
    ...     b: int = field(default=2, doc="Another.")
    ...     scale: int = field(default=1, doc="Not an input of total.")
    ...
    ...     @derived_property("a", "b")
    ...     def total(self) -> list[int]:
    ...         return [self.a + self.b]
    >>> s = Sum()
    >>> total = s.total
    >>> s.evolve(scale=3).total is total
    True
    >>> s.evolve(a=5).total
    [7]
    """

    def decorator(func: Callable[[Any], _T]) -> DerivedProperty[_T]:
        return DerivedProperty(func, tuple(inputs))

    return decorator


def _same(a: Any, b: Any) -> bool:
    """Whether two values are the same object or compare equal (False if eq fails)."""
    if a is b:
        return True
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001 - an eq that can't answer means "not the same"
        return False


def _computed_value(stage: Any, name: str) -> Any:
    """The value of a cached property of ``stage`` if computed, else raise AttributeError."""
    # On a slotted class it is a slot: an empty one raises AttributeError, without
    # computing the value.
    return getattr(type(stage), name).__get__(stage, type(stage))


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

    #: Names that are one parameter in this stage's tree: every field of that name
    #: below this stage (e.g. a ``cosmology`` held by two stages it composes) is set
    #: together by the flat name, so that they stay equal. The stage that composes the
    #: others declares it, whatever the stages it holds declare.
    shared_parameters: ClassVar[frozenset[str]] = frozenset()
    #: Fields whose names are too generic to stand alone (e.g. ``model``): in a tree
    #: holding this stage, :meth:`parameter_names` lists them with the name of the
    #: stage field holding it (``"transfer.model"``). Any unique name still routes.
    qualified_parameters: ClassVar[frozenset[str]] = frozenset()
    #: The derived properties of the class (see :func:`derived_property`), by name,
    #: recorded when the class is created.
    _derived_properties: ClassVar[Mapping[str, DerivedProperty[Any]]] = MappingProxyType({})

    _computed: routing.ComputedRecord = field(
        default=routing.NO_COMPUTED,
        init=False,
        eq=False,
        repr=False,
        physics=False,
        doc=(
            "The parameters below the stage that hold a default computed by its "
            ":meth:`computed_defaults` (see :meth:`computed_parameters`). Not part of "
            "the stage's value."
        ),
    )

    def evolve(self, **changes: Any) -> Self:
        """Return a copy of this stage tree with some parameters changed.

        Parameters are named by their flat name, a shared name, their dotted path or
        a dotted suffix of it unique in the tree (see :mod:`hmf.core.routing` and
        :meth:`parameter_names`); a
        dotted path into a model field (``"fit.A"``) changes that field of the model,
        and a stage's dotted path replaces that stage. Only the stages on the path
        from each changed field to this one are rebuilt, each by its
        :meth:`evolve_own`; every other stage is shared, with its cached results, and
        so is every :func:`derived_property` whose inputs do not change.

        A parameter that holds a default computed from other parameters (see
        :meth:`computed_defaults`, e.g. a mass function's sigma_8 and n_s, from the
        cosmology's ``meta``) is computed again when a parameter it is computed from
        changes, so that ``build(**a).evolve(**b)`` equals ``build(**a, **b)``. A
        parameter set explicitly, here or when the tree was built, keeps its value
        (see :meth:`computed_parameters`).

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
            dotted paths to choose from), is a derived property, or is given twice.
        ValueError
            If a default that is computed again cannot be computed from the new
            parameters (e.g. the new cosmology does not give sigma_8), or for an
            invalid value.
        """
        return routing.evolve(self, changes)  # type: ignore[return-value]

    def evolve_own(self, **changes: Any) -> Self:
        """Return a copy of this stage with some of its own fields changed.

        No routing: each name is a field of this stage, and nothing else changes. It
        is the step :meth:`evolve` takes at each stage it rebuilds. The value of each
        :func:`derived_property` already computed is carried to the new stage if none
        of its inputs changes; every other cached result is computed again. No
        default is computed again: a field given here, and every parameter below it,
        is explicit in the new stage (see :meth:`computed_parameters`).

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
        new = attrs.evolve(self, **changes)
        record = self._computed
        if record.paths:
            # A field given here is explicit, with every parameter below it.
            if any(p[0] in changes for p in record.paths):
                record = record.explicit((n,) for n in changes)
            object.__setattr__(new, "_computed", record)
        for name, prop in self._derived_properties.items():
            try:
                value = _computed_value(self, name)
            except AttributeError:
                continue
            if all(_same(_walk_path(self, p), _walk_path(new, p)) for p in prop.inputs):
                # A derived value is not part of the stage's value: sharing it changes
                # no result.
                object.__setattr__(new, name, value)
        return new

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

        The stage records which parameters took a computed default, and which paths of
        ``given`` the hook looked up (see :meth:`computed_parameters`). When
        :meth:`evolve` changes one of those paths, it calls the hook again, with every
        parameter of the new tree except the ones still computed, and the computed
        parameters take the new defaults. A parameter set explicitly, in
        :meth:`from_flat` or in :meth:`evolve`, is never computed again. So the hook
        must compute its defaults from ``given`` alone, and look up (with ``in``,
        ``[]`` or ``get``) every parameter it depends on.

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

    def computed_parameters(self) -> tuple[str, ...]:
        """The parameters of the stage tree that hold a default computed from others.

        Those that took a default from a stage's :meth:`computed_defaults` when the
        tree was built (by :meth:`from_flat`, or :meth:`build` where a stage has one)
        and have not been set explicitly since. :meth:`evolve` computes them again when
        a parameter they are computed from changes (see :meth:`computed_defaults`).
        Which parameters they are is not part of the stage's value: it does not change
        ``==``, the hash or the content hash, and a configuration does not record it
        (a stage read from a configuration has every value it gives explicit), but a
        pickle or a copy keeps it.

        Returns
        -------
        tuple of str
            Their dotted paths, sorted.

        Examples
        --------
        >>> from hmf.core.mass_function import MassFunction
        >>> mf = MassFunction.build(**{"transfer.model": "EH"})
        >>> mf.computed_parameters()
        ('linear_power.sigma_8', 'linear_power.transfer.n_s')
        >>> mf.evolve(sigma_8=0.8).computed_parameters()
        ('linear_power.transfer.n_s',)
        """
        return tuple(
            sorted(
                ".".join((*prefix, *path))
                for prefix, record in routing.computed_records(self).items()
                for path in record.paths
            )
        )

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

    def __copy__(self) -> Self:
        """The stage itself: a stage is immutable, so a copy of it is the stage."""
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        """The stage itself, with everything it has computed.

        A stage is immutable and its cached results never go stale, so a deep copy
        would only repeat the work (rebuilding the tree and recomputing its caches).

        Parameters
        ----------
        memo
            The memo of :func:`copy.deepcopy`.
        """
        return self

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle the stage as its configuration, never as live objects.

        The state is the stage tree's configuration (:func:`hmf.core.config.to_config`)
        and the Boltzmann-code runs it has already used, as plain arrays; unpickling
        rebuilds the tree from the configuration, without running the code again (see
        :mod:`hmf.core.config`).
        """
        from . import config

        return (config._unpickle, config._pickle_state(self))

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


def _fresh(stage: Stage) -> Stage:
    """A copy of a stage tree with nothing computed: every stage of it is a new one."""
    changes: dict[str, Any] = {}
    for a in attrs.fields(type(stage)):
        value = getattr(stage, a.name)
        if a.init and isinstance(value, Stage):
            changes[a.alias or a.name] = _fresh(value)
    return attrs.evolve(stage, **changes)


def _stage_paths(stage: Stage, path: tuple[str, ...] = ()) -> dict[int, tuple[str, ...]]:
    """The path of every stage of a tree that is a field, by its id."""
    out = {id(stage): path}
    for a in attrs.fields(type(stage)):
        value = getattr(stage, a.name)
        if isinstance(value, Stage):
            out.update(_stage_paths(value, (*path, a.name)))
    return out


def _captured(
    value: Any, paths: dict[int, tuple[str, ...]], seen: set[int]
) -> set[tuple[str, ...]]:
    """The paths of the stages of a tree that ``value`` holds (through attrs fields)."""
    if id(value) in seen:
        return set()
    seen.add(id(value))
    if id(value) in paths:
        return {paths[id(value)]}
    if isinstance(value, (list, tuple)):
        items: list[Any] = list(value)
    elif isinstance(value, dict):
        items = list(value.values())
    elif attrs.has(type(value)):
        items = [getattr(value, a.name) for a in attrs.fields(type(value))]
    else:
        return set()
    out: set[tuple[str, ...]] = set()
    for item in items:
        out |= _captured(item, paths, seen)
    return out


def undeclared_inputs(stage: Stage) -> dict[str, tuple[str, ...]]:
    """The fields each derived property of a stage reads, but does not declare as inputs.

    A check for tests, of a stage's :func:`derived_property` declarations. Each derived
    property is evaluated on a fresh copy of the stage's tree (nothing computed), while
    every read of a field of a stage of the tree is recorded, and so is every stage of
    the tree that the value holds (e.g. a derived stage built from it, which may read
    its fields later). A read is covered by an input that is the field or a stage above
    it; reading a stage field to reach the stage it holds is not a read of its own. An
    input that is never read (e.g. one used on another branch) is allowed: declaring
    too much is safe, declaring too little is not, because :meth:`Stage.evolve_own`
    would carry a stale value.

    Parameters
    ----------
    stage
        A stage whose derived properties to check (those of the stages it holds are
        not checked: call it on them too).

    Returns
    -------
    dict
        For each derived property that reads an undeclared field, the dotted paths
        (below the stage) of those fields. Empty if every read is declared.

    Examples
    --------
    >>> import attrs
    >>> from hmf.core import field
    >>> @attrs.frozen(kw_only=True)
    ... class Sum(Stage):
    ...     a: int = field(default=1, doc="One term.")
    ...     b: int = field(default=2, doc="Another.")
    ...
    ...     @derived_property("a")
    ...     def total(self) -> int:
    ...         return self.a + self.b
    >>> undeclared_inputs(Sum())
    {'total': ('b',)}
    """
    out = {}
    for name, prop in type(stage)._derived_properties.items():
        fresh = _fresh(stage)
        paths = _stage_paths(fresh)
        reads: set[tuple[str, ...]] = set()

        def tracing(
            obj: Any,
            attr: str,
            paths: dict[int, tuple[str, ...]] = paths,
            reads: set[tuple[str, ...]] = reads,
        ) -> Any:
            value = object.__getattribute__(obj, attr)
            path = paths.get(id(obj))
            # A field whose value is not a stage of the tree (reaching one is not a read).
            if (
                path is not None
                and attr in routing.field_names(type(obj))
                and id(value) not in paths
            ):
                reads.add((*path, attr))
            return value

        type.__setattr__(Stage, "__getattribute__", tracing)
        try:
            value = getattr(fresh, name)
        finally:
            type.__delattr__(Stage, "__getattribute__")
        reads |= _captured(value, paths, set())
        inputs = [tuple(p.split(".")) for p in prop.inputs]
        missing = sorted(".".join(r) for r in reads if not any(r[: len(i)] == i for i in inputs))
        if missing:
            out[name] = tuple(missing)
    return out


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
    the stage's value (``eq=False``) or of its content hash (``physics=False``), since it
    changes no result.

    Parameters
    ----------
    doc
        The field's documentation.

    Returns
    -------
    Any
        The field definition.
    """
    return field(default=None, converter=to_disk_cache, eq=False, physics=False, doc=doc)


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
