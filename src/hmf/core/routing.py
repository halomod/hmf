"""Parameter routing: flat names, aliases and dotted paths through a tree of stages.

A stage holds the stages it is computed from as fields (see :mod:`hmf.core.stage`),
so together they form a *stage tree*. This module builds, once per stage class and
from the ``attrs`` fields of the classes alone, a :class:`Router`: the table that
maps every parameter of the tree to the field that holds it. No stage is created and
nothing is computed to build it. :meth:`Stage.evolve <hmf.core.stage.Stage.evolve>`,
:meth:`Stage.from_flat <hmf.core.stage.Stage.from_flat>` and the introspection class
methods (:meth:`~hmf.core.stage.Stage.parameter_info`,
:meth:`~hmf.core.stage.Stage.invalidated_by`, ...) all use it.

Names
-----
A parameter (a field of a stage of the tree that is not itself a stage) is reached
by:

* its **dotted path** from the root, always: ``"linear_power.sigma_8"``,
  ``"variance.filter"``, ``"linear_power.transfer.model"``;
* its **flat name**, if no other field of the tree has the same name: ``sigma_8``,
  ``filter``, ``delta_c``;
* a **flat alias** declared by a stage in :attr:`Stage.parameter_aliases
  <hmf.core.stage.Stage.parameter_aliases>`, for a field whose name is repeated or
  reads badly flat: ``transfer_model`` for ``linear_power.transfer.model``;
* a **shared** name: a field declared with ``field(..., shared=True)`` in every stage
  that holds it (``cosmology``, ``k_accuracy``, ``disk_cache``) is one parameter,
  and a change to it changes every holder, so that they stay equal.

A field of the root is also reached by its own name. Any other repeated name is
*ambiguous*: it raises a :class:`TypeError` that lists the dotted paths. A name that
is not a parameter raises a :class:`TypeError` with the closest names. A stage field
is reached by its dotted path only, and replaces that whole stage.

A dotted path into a *model* field (``"fit.A"``, ``"transfer_model.z_max"``) changes
that field of the model. A model's class is known only from the stage's value, so
these are resolved when they are used, without computing anything.

Derived fields
--------------
Some fields are computed from others, and are not parameters. A stage declares them
in :attr:`Stage.derivations <hmf.core.stage.Stage.derivations>`, as
:class:`Derivation` objects: the field (a dotted path below the stage), the paths it is
computed from, and how. :meth:`~hmf.core.stage.Stage.evolve` computes a derived field
again when one of its sources changes, if it held its derived value before (a link
the user did not make is left alone), and if its new value differs from the current
one. :meth:`~hmf.core.stage.Stage.from_flat` always computes it.

Structural sharing
------------------
``evolve`` rebuilds only the stages on the path from each changed field to the root
(and the stages holding a derived field that changes), each with its own
:meth:`~hmf.core.stage.Stage.evolve_own`. Every other stage is shared, by identity,
with the original tree, and so are its cached results.
"""

from __future__ import annotations

import difflib
import types
import typing
from collections.abc import Callable, Iterator, Mapping
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

import attrs

from ._fields import DOC_KEY, NO_DEFAULT, SHARED_KEY

if TYPE_CHECKING:
    from .stage import Stage

__all__ = [
    "Derivation",
    "GivenParameters",
    "ParameterInfo",
    "Router",
    "field_names",
    "parameter_defaults",
    "router",
]

#: A path from a stage to a field below it, as a tuple of field names.
Path = tuple[str, ...]


def _dotted(path: Path) -> str:
    """A path as a dotted string."""
    return ".".join(path)


def _split(name: str) -> Path:
    """A dotted string as a path."""
    return tuple(name.split("."))


def _overlaps(a: Path, b: Path) -> bool:
    """Whether one path is the other or below it."""
    n = min(len(a), len(b))
    return a[:n] == b[:n]


def _same(a: Any, b: Any) -> bool:
    """Whether two values are the same object or compare equal (False if eq fails)."""
    if a is b:
        return True
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001 - an eq that can't answer means "not the same"
        return False


def _walk(obj: Any, path: Path) -> Any:
    """The value at ``path`` below ``obj``."""
    for name in path:
        obj = getattr(obj, name)
    return obj


def _suggest(name: str, candidates: list[str]) -> str:
    """A " Did you mean ...?" sentence for an unknown name, or an empty string."""
    close = difflib.get_close_matches(name, candidates, n=3)
    return f" Did you mean {' or '.join(repr(c) for c in close)}?" if close else ""


@attrs.frozen(kw_only=True)
class Derivation:
    """A field of a stage tree that is computed from other fields, not set.

    A stage lists them in :attr:`Stage.derivations
    <hmf.core.stage.Stage.derivations>`, with paths relative to itself: the derived
    field is below one of the stage's stage fields (or is a field of the stage), and
    the sources are anywhere below the stage. A derivation whose source is another's
    field comes after it.

    Parameters
    ----------
    field
        The dotted path of the derived field, e.g. ``"variance.power"``.
    sources
        The dotted paths it is computed from. A change to one of them, or to a field
        below one, or to a stage above one, computes the field again.
    derive
        Computes the field's value. It is called with one argument, ``get``: a
        function returning the value at a dotted path (one of ``sources``) in the
        tree being built.
    doc
        What the derivation is, for the documentation.

    Examples
    --------
    The variance of a :class:`~hmf.core.mass_function.MassFunction` is of the
    unnormalised power of its linear power's species::

        Derivation(
            field="variance.power",
            sources=("linear_power.transfer", "linear_power.species"),
            derive=lambda get: get("linear_power.transfer").power_source(
                get("linear_power.species")
            ),
        )
    """

    field: str
    sources: tuple[str, ...]
    derive: Callable[[Callable[[str], Any]], Any] = attrs.field(eq=False)
    doc: str = ""

    @cached_property
    def target(self) -> Path:
        """The derived field's path, as a tuple."""
        return _split(self.field)

    @cached_property
    def source_paths(self) -> tuple[Path, ...]:
        """The sources' paths, as tuples."""
        return tuple(_split(s) for s in self.sources)


@attrs.frozen(kw_only=True)
class ParameterInfo:
    """A description of one parameter of a stage tree.

    Parameters
    ----------
    name
        The name it is routed by: its flat name, a flat alias, its shared name, or
        (if none of them exists) its dotted path.
    paths
        The dotted paths of the fields it sets: one, or every holder of a shared
        parameter.
    type
        The field's annotated type.
    default
        Its default: an :class:`attrs.Factory` for a computed one, and
        :data:`hmf.core._fields.NO_DEFAULT` if it is required (or computed by
        :meth:`Stage.computed_defaults <hmf.core.stage.Stage.computed_defaults>`).
    doc
        Its documentation.
    shared
        Whether it is held by several stages, which a change to it changes together.
    """

    name: str
    paths: tuple[str, ...]
    type: Any
    default: Any
    doc: str
    shared: bool

    @property
    def path(self) -> str:
        """The dotted path of the (first) field it sets."""
        return self.paths[0]

    @property
    def required(self) -> bool:
        """Whether it has no static default."""
        return self.default is NO_DEFAULT


class GivenParameters(Mapping[str, Any]):
    """The parameters given to :meth:`Stage.from_flat <hmf.core.stage.Stage.from_flat>`.

    A read-only mapping from dotted paths to values, passed to
    :meth:`Stage.computed_defaults <hmf.core.stage.Stage.computed_defaults>`. Looking
    up a path below a stage given whole returns the value in that stage.

    Parameters
    ----------
    values
        The given values, by path.
    """

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[Path, Any]) -> None:
        self._values = dict(values)

    def __getitem__(self, key: str) -> Any:
        """The value at a dotted path (a ``KeyError`` if it is not given)."""
        path = _split(key)
        for i in range(len(path), 0, -1):
            if path[:i] in self._values:
                try:
                    return _walk(self._values[path[:i]], path[i:])
                except AttributeError:
                    raise KeyError(key) from None
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        """The given dotted paths."""
        return (_dotted(p) for p in self._values)

    def __len__(self) -> int:
        """The number of given paths."""
        return len(self._values)


@attrs.frozen
class _Leaf:
    """A parameter field of the tree: where it is, and its attribute."""

    path: Path
    attribute: attrs.Attribute[Any]
    shared: bool


@attrs.frozen
class _Node:
    """What evolve and from_flat need to know of one stage class."""

    cls: type
    #: The constructor's keyword of each field.
    fields: Mapping[str, attrs.Attribute[Any]]
    #: The stage fields, not derived: name -> (stage class, whether it may be None).
    children: Mapping[str, tuple[type, bool]]
    derivations: tuple[Derivation, ...]
    required: tuple[str, ...]


def _stage_type(tp: Any) -> tuple[type | None, bool]:
    """The stage class of a field's annotation, and whether it may be None."""
    from .stage import Stage

    if isinstance(tp, type):
        return (tp, False) if issubclass(tp, Stage) else (None, False)
    if isinstance(tp, types.UnionType) or typing.get_origin(tp) is typing.Union:
        args = typing.get_args(tp)
        stages = [a for a in args if isinstance(a, type) and issubclass(a, Stage)]
        if len(stages) == 1:
            return stages[0], type(None) in args
    return None, False


_NODES: dict[type, _Node] = {}
_QUANTITIES: dict[type, tuple[str, ...]] = {}
_ROUTERS: dict[type, Router] = {}


def _node(cls: type) -> _Node:
    """The static description of a stage class (cached per class)."""
    node = _NODES.get(cls)
    if node is None:
        node = _NODES[cls] = _describe(cls)
    return node


def _describe(cls: type) -> _Node:
    """The static description of a stage class."""
    with_types = attrs.resolve_types(cls) if attrs.has(cls) else cls
    fields = {a.alias or a.name: a for a in attrs.fields(with_types) if a.init}
    derivations: tuple[Derivation, ...] = tuple(cls.derivations)  # type: ignore[attr-defined]
    derived_here = {d.target[0] for d in derivations if len(d.target) == 1}
    children = {}
    for name, a in fields.items():
        stage_cls, optional = _stage_type(a.type)
        if stage_cls is not None and name not in derived_here:
            children[name] = (stage_cls, optional)
    required = tuple(n for n, a in fields.items() if a.default is NO_DEFAULT)
    return _Node(
        cls=cls,
        fields=fields,
        children=children,
        derivations=derivations,
        required=required,
    )


def _is_field(cls: type, path: Path) -> bool:
    """Whether ``path`` is a field of the stage tree of ``cls`` (stage fields included)."""
    for i, name in enumerate(path):
        a = _node(cls).fields.get(name)
        if a is None:
            return False
        if i < len(path) - 1:
            stage_cls, _ = _stage_type(a.type)
            if stage_cls is None:
                return False
            cls = stage_cls
    return True


def field_names(cls: type) -> Mapping[str, attrs.Attribute[Any]]:
    """The fields of a stage class, by constructor keyword (cached per class).

    Parameters
    ----------
    cls
        A stage class.

    Returns
    -------
    Mapping
        The ``attrs`` attribute of each field, in definition order.
    """
    return _node(cls).fields


@attrs.frozen
class _Resolved:
    """Changes resolved to paths: values, the paths set by a shared name, model changes."""

    values: dict[Path, Any]
    soft: frozenset[Path]
    models: dict[Path, dict[Path, Any]]


class Router:
    """The routing table of a stage class: every parameter of its tree, and its name.

    Build it with :func:`router` (cached per class). It is computed from the classes'
    ``attrs`` fields, :attr:`~hmf.core.stage.Stage.parameter_aliases` and
    :attr:`~hmf.core.stage.Stage.derivations`; it creates no stage.

    Parameters
    ----------
    cls
        The stage class at the root of the tree.

    Raises
    ------
    TypeError
        If a stage of the tree declares an alias or a derivation that is not a field
        of the tree, or an alias that is also a field's name.
    """

    def __init__(self, cls: type[Stage]) -> None:
        self.cls = cls
        #: The parameter fields, in depth-first order.
        self.leaves: dict[Path, _Leaf] = {}
        #: The stages of the tree (the root is ``()``), and their classes.
        self.stages: dict[Path, type] = {}
        #: The derived fields: path -> (path of the declaring stage, derivation).
        self.derived: dict[Path, tuple[Path, Derivation]] = {}
        aliases: dict[str, set[Path]] = {}
        self._visit(cls, (), aliases)

        by_name: dict[str, list[Path]] = {}
        for path in self.leaves:
            by_name.setdefault(path[-1], []).append(path)

        #: Flat names (and aliases and shared names) -> the paths they set.
        self.names: dict[str, tuple[Path, ...]] = {}
        #: Repeated names -> the paths they could mean.
        self.ambiguous: dict[str, tuple[Path, ...]] = {}
        for name, found in by_name.items():
            paths = tuple(found)
            if len(paths) == 1:
                self.names[name] = (paths[0],)
            elif all(self.leaves[p].shared for p in paths):
                self.names[name] = paths
            elif (name,) in self.leaves:
                self.names[name] = ((name,),)
            else:
                self.ambiguous[name] = paths
        for alias, targets in aliases.items():
            if alias in by_name:
                raise TypeError(
                    f"{cls.__name__}: the alias {alias!r} is also the name of a field "
                    f"({', '.join(_dotted(p) for p in by_name[alias])})."
                )
            if len(targets) == 1:
                self.names[alias] = (next(iter(targets)),)
            else:
                self.ambiguous[alias] = tuple(sorted(targets))

        #: The name each parameter field is routed by.
        self.preferred: dict[Path, str] = {}
        for name, paths in self.names.items():
            for path in paths:
                current = self.preferred.get(path)
                # A field's own (flat or shared) name beats an alias.
                if current is None or (name == path[-1] and current != path[-1]):
                    self.preferred[path] = name
        for path in self.leaves:
            self.preferred.setdefault(path, _dotted(path))

        self._candidates = sorted(
            set(self.names) | set(self.ambiguous) | {_dotted(p) for p in self.leaves}
        )

    # -- building --------------------------------------------------------------------

    def _visit(self, cls: type, prefix: Path, aliases: dict[str, set[Path]]) -> None:
        """Add the fields of the stage class ``cls`` at ``prefix`` (depth first)."""
        node = _node(cls)
        self.stages[prefix] = cls
        for d in node.derivations:
            self.derived[prefix + d.target] = (prefix, d)
        for name, a in node.fields.items():
            path = (*prefix, name)
            if path in self.derived:
                continue
            if name in node.children:
                self._visit(node.children[name][0], path, aliases)
            else:
                self.leaves[path] = _Leaf(path, a, bool(a.metadata.get(SHARED_KEY, False)))
        for alias, target in getattr(cls, "parameter_aliases", {}).items():
            aliases.setdefault(alias, set()).add(prefix + _split(target))
        # Checked once the whole subtree is known.
        for alias, target in getattr(cls, "parameter_aliases", {}).items():
            if prefix + _split(target) not in self.leaves:
                raise TypeError(
                    f"{cls.__name__}.parameter_aliases: {alias!r} -> {target!r} is not a "
                    "parameter field."
                )
        for d in node.derivations:
            for p in (d.target, *d.source_paths):
                if not _is_field(cls, p):
                    raise TypeError(
                        f"{cls.__name__}.derivations: {_dotted(p)!r} is not a field of the "
                        "stage tree."
                    )

    # -- resolving names -------------------------------------------------------------

    def _ambiguous_message(self, name: str) -> str:
        paths = self.ambiguous[name]
        options = [repr(_dotted(p)) for p in paths]
        named = sorted(
            {self.preferred[p] for p in paths if p in self.preferred} - {_dotted(p) for p in paths}
        )
        msg = (
            f"{self.cls.__name__}: the parameter name {name!r} is ambiguous; give one of "
            f"the dotted paths {', '.join(options)}"
        )
        if named:
            msg += f", or the names {', '.join(repr(n) for n in named)}"
        return msg + "."

    def _unknown_message(self, name: str) -> str:
        return (
            f"{self.cls.__name__} has no parameter {name!r}."
            + _suggest(name, self._candidates)
            + " See parameter_names() for the parameters."
        )

    def paths_of(self, name: str) -> tuple[tuple[Path, ...], Path]:
        """The paths a name sets, and the rest of it, a path into a model field.

        Parameters
        ----------
        name
            A flat name, alias, shared name or dotted path, possibly followed by
            ``.field`` of a model.

        Returns
        -------
        tuple
            The paths of the fields of the tree (several for a shared name), and the
            path into the model held by the field (empty if none).

        Raises
        ------
        TypeError
            If the name is unknown, ambiguous, or a derived field.
        """
        found = self.names.get(name)
        if found is not None:
            return found, ()
        if name in self.ambiguous:
            raise TypeError(self._ambiguous_message(name))
        path = _split(name)
        if path in self.leaves or (path in self.stages and path):
            return (path,), ()
        if path in self.derived:
            declared, d = self.derived[path]
            sources = ", ".join(repr(_dotted(declared + s)) for s in d.source_paths)
            raise TypeError(
                f"{self.cls.__name__}: {name!r} is derived from {sources}, and cannot be "
                "set. Change what it is derived from."
            )
        # A path into a model field: the longest prefix that is a parameter.
        for i in range(len(path) - 1, 0, -1):
            head = _dotted(path[:i])
            if head in self.ambiguous:
                raise TypeError(self._ambiguous_message(head))
            try:
                paths, rest = self.paths_of(head)
            except TypeError:
                continue
            if len(paths) == 1 and paths[0] in self.leaves:
                return paths, rest + path[i:]
            break
        raise TypeError(self._unknown_message(name))

    def resolve(self, changes: Mapping[str, Any]) -> _Resolved:
        """Resolve changes by name to changes by path, checking for conflicts.

        Parameters
        ----------
        changes
            Values by flat name, alias, shared name or dotted path. A mapping value
            of a stage's path (e.g. ``{"linear_power": {"sigma_8": 0.8}}``) is read
            as values of the fields below that stage.

        Returns
        -------
        _Resolved

        Raises
        ------
        TypeError
            For an unknown, ambiguous or derived name, or if one field is given twice
            (by two names, or with a stage above it).
        """
        values: dict[Path, Any] = {}
        soft: set[Path] = set()
        models: dict[Path, dict[Path, Any]] = {}
        # The name each field (or field of a model) was given by.
        origin: dict[Path, str] = {}

        def claim(key: str, path: Path) -> None:
            if path in origin:
                raise TypeError(
                    f"{self.cls.__name__}: {_dotted(path)!r} is given twice, as "
                    f"{origin[path]!r} and as {key!r}."
                )
            origin[path] = key

        def add(key: str, value: Any) -> None:
            if isinstance(value, Mapping) and key and _split(key) in self.stages:
                for sub, v in value.items():
                    add(f"{key}.{sub}", v)
                return
            paths, rest = self.paths_of(key)
            if rest:
                claim(key, paths[0] + rest)
                models.setdefault(paths[0], {})[rest] = value
                return
            for path in paths:
                claim(key, path)
                values[path] = value
                if len(paths) > 1:
                    soft.add(path)

        for key, value in changes.items():
            add(key, value)

        # A field and a stage above it can't both be given. A shared name sets only
        # the holders that are not in a stage given whole (which has its own).
        for path, key in list(origin.items()):
            for i in range(1, len(path)):
                if path[:i] in values and path[:i] in self.stages:
                    if path in soft:
                        del values[path]
                        soft.discard(path)
                        break
                    raise TypeError(
                        f"{self.cls.__name__}: {key!r} sets {_dotted(path)!r}, in the stage "
                        f"{_dotted(path[:i])!r}, which is also given (as "
                        f"{origin[path[:i]]!r}). Give one or the other."
                    )
        return _Resolved(values=values, soft=frozenset(soft), models=models)

    # -- introspection ---------------------------------------------------------------

    @cached_property
    def parameter_info(self) -> dict[str, ParameterInfo]:
        """A :class:`ParameterInfo` per parameter, by the name it is routed by."""
        out: dict[str, ParameterInfo] = {}
        for path, leaf in self.leaves.items():
            name = self.preferred[path]
            if name in out:
                continue
            paths = self.names.get(name, (path,))
            holders = [self.leaves[p].attribute for p in paths]
            static = [a.default for a in holders if not _takes_self(a.default)]
            default = static[0] if static else holders[0].default
            out[name] = ParameterInfo(
                name=name,
                paths=tuple(_dotted(p) for p in paths),
                type=leaf.attribute.type,
                default=default,
                doc=leaf.attribute.metadata.get(DOC_KEY, ""),
                shared=len(paths) > 1,
            )
        return out

    def invalidated_by(self, name: str) -> tuple[str, ...]:
        """The stages that a change of ``name`` rebuilds (see :meth:`Stage.invalidated_by`)."""
        paths, _ = self.paths_of(name)
        changed = set(paths)
        derived = sorted(self.derived.items())
        grew = True
        while grew:
            grew = False
            for target, (prefix, d) in derived:
                if target in changed:
                    continue
                if any(_overlaps(c, prefix + s) for c in changed for s in d.source_paths):
                    changed.add(target)
                    grew = True
        rebuilt: set[Path] = set()
        for path in changed:
            if path in self.stages:
                rebuilt.add(path)
            rebuilt.update(path[:i] for i in range(len(path)))
        return tuple(_dotted(p) for p in sorted(rebuilt, key=lambda p: (len(p), p)))

    def quantities_available(self) -> dict[str, tuple[str, ...]]:
        """The public output methods and properties of each stage of the tree."""
        return {_dotted(path): _quantities(cls) for path, cls in self.stages.items()}


def _takes_self(default: Any) -> bool:
    """Whether a default is computed from the instance."""
    return isinstance(default, attrs.Factory) and default.takes_self  # type: ignore[arg-type]


def _quantities(cls: type) -> tuple[str, ...]:
    """The public output methods and properties of a stage class (cached per class)."""
    if cls in _QUANTITIES:
        return _QUANTITIES[cls]
    from .stage import CosmologyStage, Stage

    base = set(dir(CosmologyStage)) | set(dir(Stage))
    fields = {a.name for a in attrs.fields(cls)} if attrs.has(cls) else set()
    out = set()
    for klass in cls.__mro__:
        if klass in (Stage, CosmologyStage) or not issubclass(klass, Stage):
            continue
        for name, value in vars(klass).items():
            if name.startswith("_") or name in base or name in fields or name.endswith("_kernel"):
                continue
            # A cached_property of a slotted attrs class is a slot that is not a field.
            if isinstance(
                value,
                (property, cached_property, types.FunctionType, types.MemberDescriptorType),
            ):
                out.add(name)
    _QUANTITIES[cls] = tuple(sorted(out))
    return _QUANTITIES[cls]


def router(cls: type[Stage]) -> Router:
    """The routing table of a stage class, built once per class.

    Parameters
    ----------
    cls
        A stage class.

    Returns
    -------
    Router
    """
    table = _ROUTERS.get(cls)
    if table is None:
        table = _ROUTERS[cls] = Router(cls)
    return table


# -- evolve and from_flat ------------------------------------------------------------


def _convert(attribute: attrs.Attribute[Any], value: Any) -> Any:
    """A value passed through a field's converter (if it has a plain one)."""
    conv: Any = attribute.converter
    if conv is None:
        return value
    if isinstance(conv, attrs.Converter):
        wrapped: Any = conv  # attrs' stubs don't declare the wrapped callable
        return wrapped.converter(value)
    return conv(value)


def _default(attribute: attrs.Attribute[Any], where: str) -> Any:
    """A field's default value, if it does not depend on the instance."""
    default = attribute.default
    if default is NO_DEFAULT or _takes_self(default):
        raise TypeError(f"{where} has no default to change: give it first.")
    if isinstance(default, attrs.Factory):  # type: ignore[arg-type]
        return cast("Any", default).factory()
    return default


def _evolve_model(model: Any, changes: Mapping[Path, Any], where: str) -> Any:
    """``model`` with the fields at the given paths below it changed."""
    if not attrs.has(type(model)):
        name = where + "." + _dotted(next(iter(changes)))
        raise TypeError(
            f"{name!r}: {where!r} is a {type(model).__name__}, not a model with fields."
        )
    fields = {a.alias or a.name: a for a in attrs.fields(type(model)) if a.init}
    own: dict[str, Any] = {}
    deeper: dict[str, dict[Path, Any]] = {}
    for path, value in changes.items():
        if path[0] not in fields:
            raise TypeError(
                f"{type(model).__name__} (at {where!r}) has no field {path[0]!r}."
                + _suggest(path[0], list(fields))
                + f" Fields: {list(fields)}."
            )
        if len(path) == 1:
            own[path[0]] = value
        else:
            deeper.setdefault(path[0], {})[path[1:]] = value
    for name, sub in deeper.items():
        base = _convert(fields[name], own[name]) if name in own else getattr(model, name)
        own[name] = _evolve_model(base, sub, f"{where}.{name}")
    return attrs.evolve(model, **own)


def _apply_models(
    table: Router, resolved: _Resolved, base: Callable[[Path], Any]
) -> dict[Path, Any]:
    """The resolved values, with the model changes applied to their models."""
    values = dict(resolved.values)
    for path, changes in resolved.models.items():
        leaf = table.leaves[path]
        model = _convert(leaf.attribute, values[path]) if path in values else base(path)
        values[path] = _evolve_model(model, changes, table.preferred[path])
    return values


def evolve(stage: Stage, changes: Mapping[str, Any]) -> Stage:
    """A copy of ``stage`` with parameters changed, by name (see :meth:`Stage.evolve`).

    Parameters
    ----------
    stage
        The stage at the root of the tree.
    changes
        New values, by flat name, alias, shared name or dotted path.

    Returns
    -------
    Stage
    """
    table = router(type(stage))
    resolved = table.resolve(changes)
    values = resolved.values
    if resolved.models:
        values = _apply_models(table, resolved, lambda p: _walk(stage, p))
    return _Rebuild(type(stage), stage, values, resolved.soft, ()).run()  # type: ignore[no-any-return]


def from_flat(cls: type[Stage], mapping: Mapping[str, Any]) -> Stage:
    """Build a stage tree from parameters by name (see :meth:`Stage.from_flat`).

    Parameters
    ----------
    cls
        The stage class at the root of the tree.
    mapping
        Values by flat name, alias, shared name or dotted path, or nested mappings.

    Returns
    -------
    Stage
    """
    table = router(cls)
    resolved = _with_computed_defaults(table, table.resolve(mapping))

    def base(path: Path) -> Any:
        return _default(table.leaves[path].attribute, repr(table.preferred[path]))

    values = resolved.values
    if resolved.models:
        values = _apply_models(table, resolved, base)
    return _Rebuild(cls, None, values, resolved.soft, ()).run()  # type: ignore[no-any-return]


def _with_computed_defaults(table: Router, resolved: _Resolved) -> _Resolved:
    """Add the computed defaults of each stage of the tree that is not given whole.

    Each stage's :meth:`~hmf.core.stage.Stage.computed_defaults` sees the parameters
    given below it; stages are visited from the top, and a default is used only for a
    field that is not given (or set by a default from a stage above), and not below a
    given stage.
    """
    values = dict(resolved.values)
    soft = set(resolved.soft)
    given = [*resolved.values, *resolved.models]
    for prefix, stage_cls in table.stages.items():
        if any(prefix[:i] in values for i in range(len(prefix) + 1)):
            continue
        n = len(prefix)
        below = {p[n:]: v for p, v in values.items() if p[:n] == prefix}
        computed = stage_cls.computed_defaults(GivenParameters(below))  # type: ignore[attr-defined]
        if not computed:
            continue
        extra = router(stage_cls).resolve(computed)
        for path, value in extra.values.items():
            full = prefix + path
            if not any(_overlaps(full, g) for g in given):
                values[full] = value
                given.append(full)
                if path in extra.soft:
                    soft.add(full)
    return _Resolved(values=values, soft=frozenset(soft), models=resolved.models)


def parameter_defaults(cls: type[Stage]) -> dict[str, Any]:
    """The default of each parameter that has one (see :meth:`Stage.parameter_defaults`).

    Parameters
    ----------
    cls
        The stage class at the root of the tree.

    Returns
    -------
    dict
        Values by the name each parameter is routed by, in the order of
        :meth:`Stage.parameter_names`.
    """
    table = router(cls)
    computed = _with_computed_defaults(table, _Resolved(values={}, soft=frozenset(), models={}))
    out: dict[str, Any] = {}
    for name, info in table.parameter_info.items():
        path = _split(info.path)
        if path in computed.values:
            out[name] = computed.values[path]
        elif not info.required and not _takes_self(info.default):
            default = info.default
            is_factory = isinstance(default, attrs.Factory)  # type: ignore[arg-type]
            out[name] = cast("Any", default).factory() if is_factory else default
    return out


#: A derived value that is left as it is.
_UNSET: Any = object()


class _Rebuild:
    """Rebuild one stage of a tree with changes (or, with ``old`` None, build it).

    ``changes`` are by path relative to the stage; ``soft`` are the paths set by a
    shared name, which a stage field that is None (e.g. ``Growth.transfer``) ignores;
    ``where`` is the stage's path from the root, for messages.
    """

    def __init__(
        self, cls: type, old: Any, changes: Mapping[Path, Any], soft: frozenset[Path], where: Path
    ) -> None:
        self.cls, self.old, self.soft, self.where = cls, old, soft, where
        self.node = node = _node(cls)
        self.own: dict[str, Any] = {}
        self.sub: dict[str, dict[Path, Any]] = {}
        for path, value in changes.items():
            if len(path) == 1:
                self.own[path[0]] = value
            else:
                self.sub.setdefault(path[0], {})[path[1:]] = value
        # The derivations to compute: those whose sources change (in declaration
        # order, so that a derived source counts as changed), or all, to build.
        self.fired: list[Derivation] = []
        if node.derivations:
            changed = list(changes)
            for d in node.derivations:
                if d.target[0] in self.own:
                    continue
                if old is None or any(_overlaps(c, s) for c in changed for s in d.source_paths):
                    self.fired.append(d)
                    changed.append(d.target)
        #: The fired derivations of this stage's own fields.
        self.derived_own = {d.target[0]: d for d in self.fired if len(d.target) == 1}
        self.built: dict[str, Any] = {}

    def get(self, dotted: str) -> Any:
        """The value at a dotted path in the new tree."""
        path = _split(dotted)
        return _walk(self.child(path[0]), path[1:])

    def get_old(self, dotted: str) -> Any:
        """The value at a dotted path in the old tree."""
        return _walk(self.old, _split(dotted))

    def derived_value(self, d: Derivation) -> Any:
        """The new value of a derived field, or ``_UNSET`` to leave it.

        A field is left if its value does not change, or if it did not hold its
        derived value before (a link the user did not make).
        """
        value = d.derive(self.get)
        if self.old is None:
            return value
        current = _walk(self.old, d.target)
        if _same(current, value) or not _same(current, d.derive(self.get_old)):
            return _UNSET
        return value

    def child(self, name: str) -> Any:
        """The new value of the field ``name`` (built once)."""
        if name not in self.built:
            if name in self.own:
                value = self.own[name]
            elif name in self.node.children:
                value = self.stage_field(name)
            elif name in self.derived_own:
                value = self.derived_value(self.derived_own[name])
                if value is _UNSET:
                    value = getattr(self.old, name)
            elif self.old is not None:
                value = getattr(self.old, name)
            else:
                value = _default(self.node.fields[name], repr(_dotted((*self.where, name))))
            self.built[name] = value
        return self.built[name]

    def stage_field(self, name: str) -> Any:
        """The new value of the stage field ``name``, not given whole."""
        child_cls, optional = self.node.children[name]
        changes = dict(self.sub.get(name, {}))
        hard = any((name, *p) not in self.soft for p in changes)
        for d in self.fired:
            if d.target[0] == name and len(d.target) > 1:
                new = self.derived_value(d)
                if new is not _UNSET:
                    changes[d.target[1:]] = new
                    hard = True
        old_child = None if self.old is None else getattr(self.old, name)
        if self.old is not None and not changes:
            return old_child
        if optional and old_child is None and not hard:
            # Only shared parameters: an absent optional stage stays absent.
            return old_child if self.old is not None else _UNSET
        if self.old is not None and old_child is None:
            given = ", ".join(repr(_dotted((*self.where, name, *p))) for p in changes)
            raise ValueError(
                f"{_dotted((*self.where, name))} is None, so {given} cannot be set. Give "
                "the stage first."
            )
        soft = frozenset(p[1:] for p in self.soft if p[0] == name and len(p) > 1)
        return _Rebuild(child_cls, old_child, changes, soft, (*self.where, name)).run()

    def run(self) -> Any:
        """The new stage (the old one if nothing changes)."""
        old = self.old
        kwargs = dict(self.own)
        for name in self.node.children:
            if name in self.own:
                continue
            if old is None or name in self.sub or any(d.target[0] == name for d in self.fired):
                value = self.child(name)
                if value is not _UNSET and (old is None or value is not getattr(old, name)):
                    kwargs[name] = value
        for name in self.derived_own:
            value = self.child(name)
            if old is None or value is not getattr(old, name):
                kwargs[name] = value
        if old is None:
            missing = [_dotted((*self.where, n)) for n in self.node.required if n not in kwargs]
            if missing:
                raise TypeError(f"{self.cls.__name__}: missing required parameter(s) {missing}.")
            return self.cls(**kwargs)
        if not kwargs:
            return old
        return old.evolve_own(**kwargs)
