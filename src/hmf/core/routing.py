"""Parameter routing: flat names and dotted paths through a tree of stages.

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
  ``"linear_power.transfer.model"``;
* any **dotted suffix** of its path that ends no other field's path: its flat name
  (``sigma_8``, ``filter``, ``delta_c``) if no other field has the same name, and
  otherwise a longer suffix, such as ``"transfer.model"`` where several stages have
  a ``model``;
* a **shared** name: a name that a stage lists in :attr:`Stage.shared_parameters
  <hmf.core.stage.Stage.shared_parameters>` is one parameter in that stage's tree,
  and a change to it changes every field of that name below the stage, so that they
  stay equal. The stage that composes the others declares it.

The full path of a field (e.g. the name of a field of the root) always means that
field. Any other suffix that ends several paths is *ambiguous*: it raises a
:class:`TypeError` that lists the names to choose from. A name that is not a
parameter raises a :class:`TypeError` with the closest names. A stage field is
reached by its dotted path only, and replaces that whole stage.

A dotted path into a *model* field (``"fit.A"``, ``"transfer.model.z_max"``) changes
that field of the model. A model's class is known only from the stage's value, so
these are resolved when they are used, without computing anything.

Derived properties
------------------
A stage computed from other stages' fields, such as a mass function's mass variance,
is not a field but a :func:`~hmf.core.stage.derived_property`: a cached property
that declares its inputs. It is not a parameter; its own settings are fields of the
stage that holds it. The table knows the derived properties of each class, so that
:meth:`~hmf.core.stage.Stage.invalidated_by` names those a change recomputes.

Structural sharing
------------------
``evolve`` rebuilds only the stages on the path from each changed field to the
root, each with its own :meth:`~hmf.core.stage.Stage.evolve_own`, which carries
every derived property whose inputs do not change. Every other stage is shared, by
identity, with the original tree, and so are its cached results.
"""

from __future__ import annotations

import difflib
import types
import typing
from collections.abc import Callable, Iterator, Mapping
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

import attrs

from ._fields import DOC_KEY, NO_DEFAULT, _resolved_fields

if TYPE_CHECKING:
    from .stage import Stage

__all__ = [
    "GivenParameters",
    "ParameterInfo",
    "Router",
    "field_names",
    "parameter_defaults",
    "router",
]

#: The path from a stage to a field below it, as a tuple of field names (a dotted
#: name split at its dots).
FieldPath = tuple[str, ...]


def _dotted(path: FieldPath) -> str:
    """A path as a dotted string."""
    return ".".join(path)


def _split(name: str) -> FieldPath:
    """A dotted string as a path."""
    return tuple(name.split("."))


def _overlaps(a: FieldPath, b: FieldPath) -> bool:
    """Whether one path is the other or below it."""
    n = min(len(a), len(b))
    return a[:n] == b[:n]


def _walk(obj: Any, path: FieldPath) -> Any:
    """The value at ``path`` below ``obj``."""
    for name in path:
        obj = getattr(obj, name)
    return obj


def _suggest(name: str, candidates: list[str]) -> str:
    """A " Did you mean ...?" sentence for an unknown name, or an empty string."""
    close = difflib.get_close_matches(name, candidates, n=3)
    return f" Did you mean {' or '.join(repr(c) for c in close)}?" if close else ""


@attrs.frozen(kw_only=True)
class ParameterInfo:
    """A description of one parameter of a stage tree.

    Parameters
    ----------
    name
        The name it is routed by: its shortest name (its flat name, a shared name, or
        the shortest dotted suffix of its path that is unique in the tree).
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

    def __init__(self, values: Mapping[FieldPath, Any]) -> None:
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
class _Node:
    """What evolve and from_flat need to know of one stage class."""

    cls: type
    #: The constructor's keyword of each field.
    fields: Mapping[str, attrs.Attribute[Any]]
    #: The stage fields: name -> (stage class, whether it may be None).
    children: Mapping[str, tuple[type, bool]]
    required: tuple[str, ...]
    #: The derived properties: name -> (their inputs, the stage class they give, if one).
    derived: Mapping[str, tuple[tuple[FieldPath, ...], type | None]]


def _stage_type(tp: Any) -> tuple[type | None, bool]:
    """The stage class of an annotation, and whether it may be None."""
    from .stage import Stage

    if isinstance(tp, type):
        return (tp, False) if issubclass(tp, Stage) else (None, False)
    if isinstance(tp, types.UnionType) or typing.get_origin(tp) is typing.Union:
        args = typing.get_args(tp)
        stages = [a for a in args if isinstance(a, type) and issubclass(a, Stage)]
        if len(stages) == 1:
            return stages[0], type(None) in args
    return None, False


def _derived_stage_type(prop: Any) -> type | None:
    """The stage class a derived property gives, from its return annotation, if one."""
    try:
        hints = typing.get_type_hints(prop.func)
    except NameError:
        return None
    return _stage_type(hints.get("return"))[0]


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
    fields = {a.alias or a.name: a for a in _resolved_fields(cls) if a.init}
    children = {}
    for name, a in fields.items():
        stage_cls, optional = _stage_type(a.type)
        if stage_cls is not None:
            children[name] = (stage_cls, optional)
    required = tuple(n for n, a in fields.items() if a.default is NO_DEFAULT)
    derived = {
        name: (tuple(_split(p) for p in prop.inputs), _derived_stage_type(prop))
        for name, prop in getattr(cls, "_derived_properties", {}).items()
    }
    return _Node(cls=cls, fields=fields, children=children, required=required, derived=derived)


def _is_field(cls: type, path: FieldPath) -> bool:
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

    values: dict[FieldPath, Any]
    soft: frozenset[FieldPath]
    models: dict[FieldPath, dict[FieldPath, Any]]


class Router:
    """The routing table of a stage class: every parameter of its tree, and its name.

    Build it with :func:`router` (cached per class). It is computed from the classes'
    ``attrs`` fields, :attr:`~hmf.core.stage.Stage.shared_parameters` and their
    derived properties; it creates no stage.

    Parameters
    ----------
    cls
        The stage class at the root of the tree.

    Raises
    ------
    TypeError
        If a stage of the tree has a derived property whose input is not a field.
    """

    def __init__(self, cls: type[Stage]) -> None:
        self.cls = cls
        #: The parameter fields (in depth-first order), and their attributes.
        self.leaves: dict[FieldPath, attrs.Attribute[Any]] = {}
        #: The stages that are fields of the tree (the root is ``()``), and their classes.
        self.stages: dict[FieldPath, type] = {}
        #: The derived properties of the tree's stages: path -> their inputs' paths.
        self.derived: dict[FieldPath, tuple[FieldPath, ...]] = {}
        #: The derived properties that are stages, and their classes.
        self.derived_stages: dict[FieldPath, type] = {}
        #: The stages declaring each shared name.
        sharers: dict[str, list[FieldPath]] = {}
        #: The fields listed by their name with the stage field above it.
        self._qualified: set[FieldPath] = set()
        self._visit(cls, (), sharers)

        # Every dotted suffix of every field's path: "model", "transfer.model", ...
        by_suffix: dict[FieldPath, list[FieldPath]] = {}
        for path in self.leaves:
            for i in range(len(path)):
                by_suffix.setdefault(path[i:], []).append(path)

        #: Names (dotted suffixes, shared names and paths) -> the paths they set.
        self.names: dict[str, tuple[FieldPath, ...]] = {}
        #: Names that end more than one path -> the paths they could mean.
        self.ambiguous: dict[str, tuple[FieldPath, ...]] = {}
        for suffix, found in by_suffix.items():
            name, paths = _dotted(suffix), tuple(found)
            if len(paths) == 1:
                self.names[name] = paths
            elif any(all(p[: len(s)] == s for p in paths) for s in sharers.get(suffix[-1], ())):
                # A stage above all of them declares the name shared.
                self.names[name] = paths
            elif suffix in self.leaves:
                # The full path of a field (e.g. a field of the root) means that field.
                self.names[name] = (suffix,)
            else:
                self.ambiguous[name] = paths

        #: The name each parameter field is routed by: its shortest name (with the
        #: name of the stage field above it, for a field its stage lists in
        #: ``qualified_parameters``).
        self.preferred: dict[FieldPath, str] = {}
        for path in self.leaves:
            longest = len(path) - (2 if path in self._qualified and len(path) > 1 else 1)
            for i in range(longest, -1, -1):
                name = _dotted(path[i:])
                if path in self.names.get(name, ()):
                    self.preferred[path] = name
                    break

        self._candidates = sorted(set(self.names) | set(self.ambiguous))

    # -- building --------------------------------------------------------------------

    def _visit(self, cls: type, prefix: FieldPath, sharers: dict[str, list[FieldPath]]) -> None:
        """Add the fields of the stage class ``cls`` at ``prefix`` (depth first)."""
        node = _node(cls)
        self.stages[prefix] = cls
        for name in getattr(cls, "shared_parameters", ()):
            sharers.setdefault(name, []).append(prefix)
        for name, a in node.fields.items():
            path = (*prefix, name)
            if name in node.children:
                self._visit(node.children[name][0], path, sharers)
            else:
                self.leaves[path] = a
                if name in getattr(cls, "qualified_parameters", ()):
                    self._qualified.add(path)
        for name, (inputs, stage_cls) in node.derived.items():
            for p in inputs:
                if not _is_field(cls, p):
                    raise TypeError(
                        f"{cls.__name__}.{name}: the input {_dotted(p)!r} is not a field of "
                        "the stage tree."
                    )
            self.derived[(*prefix, name)] = tuple(prefix + p for p in inputs)
            if stage_cls is not None:
                self.derived_stages[(*prefix, name)] = stage_cls

    # -- resolving names -------------------------------------------------------------

    def _ambiguous_message(self, name: str) -> str:
        options = ", ".join(repr(self.preferred[p]) for p in self.ambiguous[name])
        return (
            f"{self.cls.__name__}: the parameter name {name!r} is ambiguous; give one of {options}."
        )

    def _unknown_message(self, name: str) -> str:
        return (
            f"{self.cls.__name__} has no parameter {name!r}."
            + _suggest(name, self._candidates)
            + " See parameter_names() for the parameters."
        )

    def paths_of(self, name: str) -> tuple[tuple[FieldPath, ...], FieldPath]:
        """The paths a name sets, and the rest of it, a path into a model field.

        Parameters
        ----------
        name
            A flat name, dotted suffix, shared name or dotted path, possibly followed by
            ``.field`` of a model.

        Returns
        -------
        tuple
            The paths of the fields of the tree (several for a shared name), and the
            path into the model held by the field (empty if none).

        Raises
        ------
        TypeError
            If the name is unknown, ambiguous, or a derived property.
        """
        found = self.names.get(name)
        if found is not None:
            return found, ()
        if name in self.ambiguous:
            raise TypeError(self._ambiguous_message(name))
        path = _split(name)
        if path in self.leaves or (path in self.stages and path):
            return (path,), ()
        for i in range(len(path), 0, -1):
            if path[:i] in self.derived:
                inputs = ", ".join(repr(_dotted(p)) for p in self.derived[path[:i]])
                raise TypeError(
                    f"{self.cls.__name__}: {name!r} is computed (in {_dotted(path[:i])!r}, "
                    f"from {inputs}), and cannot be set. Change what it is computed from."
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
            Values by flat name, dotted suffix, shared name or dotted path. A mapping value
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
        values: dict[FieldPath, Any] = {}
        soft: set[FieldPath] = set()
        models: dict[FieldPath, dict[FieldPath, Any]] = {}
        # The name each field (or field of a model) was given by.
        origin: dict[FieldPath, str] = {}

        def claim(key: str, path: FieldPath) -> None:
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
        for path, attribute in self.leaves.items():
            name = self.preferred[path]
            if name in out:
                continue
            paths = self.names.get(name, (path,))
            holders = [self.leaves[p] for p in paths]
            static = [a.default for a in holders if not _takes_self(a.default)]
            default = static[0] if static else holders[0].default
            out[name] = ParameterInfo(
                name=name,
                paths=tuple(_dotted(p) for p in paths),
                type=attribute.type,
                default=default,
                doc=attribute.metadata.get(DOC_KEY, ""),
                shared=len(paths) > 1,
            )
        return out

    def invalidated_by(self, name: str) -> tuple[str, ...]:
        """What a change of ``name`` rebuilds (see :meth:`Stage.invalidated_by`).

        Parameters
        ----------
        name
            A parameter name: flat, a dotted suffix or a dotted path.

        Returns
        -------
        tuple of str
            The dotted paths of the stages that are rebuilt (``""`` for the root).

        Raises
        ------
        TypeError
            If ``name`` is not a parameter of the tree, or is ambiguous.
        """
        paths, _ = self.paths_of(name)
        rebuilt: set[FieldPath] = set()
        for path in paths:
            if path in self.stages:
                rebuilt.add(path)
            rebuilt.update(path[:i] for i in range(len(path)))
        for derived, inputs in self.derived.items():
            if any(_overlaps(p, i) for p in paths for i in inputs):
                rebuilt.add(derived)
        return tuple(_dotted(p) for p in sorted(rebuilt, key=lambda p: (len(p), p)))

    def quantities_available(self) -> dict[str, tuple[str, ...]]:
        """The public outputs of each stage of the tree (see :meth:`Stage.quantities_available`).

        Returns
        -------
        dict
            The names of the outputs (a tuple of str) of each stage, by its dotted path.
        """
        out = {}
        for path, cls in (*self.stages.items(), *self.derived_stages.items()):
            stages_here = {p[-1] for p in self.derived_stages if p[:-1] == path}
            out[_dotted(path)] = tuple(n for n in _quantities(cls) if n not in stages_here)
        return out


def _takes_self(default: Any) -> bool:
    """Whether a default is computed from the instance."""
    return isinstance(default, attrs.Factory) and default.takes_self  # type: ignore[arg-type]


def _quantities(cls: type) -> tuple[str, ...]:
    """The public outputs of a stage class (cached per class).

    Its :func:`~hmf.core.units.unit_boundary` methods and its public properties and
    cached properties, without fields or the names of the base stage classes.
    """
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
            is_property = isinstance(value, (property, cached_property, types.MemberDescriptorType))
            # Of the methods, the outputs are those at the units boundary.
            is_output = isinstance(value, types.FunctionType) and hasattr(
                value, "__unit_boundary__"
            )
            if is_property or is_output:
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


def _evolve_model(model: Any, changes: Mapping[FieldPath, Any], where: str) -> Any:
    """``model`` with the fields at the given paths below it changed."""
    if not attrs.has(type(model)):
        name = where + "." + _dotted(next(iter(changes)))
        raise TypeError(
            f"{name!r}: {where!r} is a {type(model).__name__}, not a model with fields."
        )
    fields = {a.alias or a.name: a for a in attrs.fields(type(model)) if a.init}
    own: dict[str, Any] = {}
    deeper: dict[str, dict[FieldPath, Any]] = {}
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
    table: Router, resolved: _Resolved, base: Callable[[FieldPath], Any]
) -> dict[FieldPath, Any]:
    """The resolved values, with the model changes applied to their models."""
    values = dict(resolved.values)
    for path, changes in resolved.models.items():
        model = _convert(table.leaves[path], values[path]) if path in values else base(path)
        values[path] = _evolve_model(model, changes, table.preferred[path])
    return values


def evolve(stage: Stage, changes: Mapping[str, Any]) -> Stage:
    """A copy of ``stage`` with parameters changed, by name (see :meth:`Stage.evolve`).

    Parameters
    ----------
    stage
        The stage at the root of the tree.
    changes
        New values, by flat name, dotted suffix, shared name or dotted path.

    Returns
    -------
    Stage
    """
    table = router(type(stage))
    resolved = table.resolve(changes)
    values = resolved.values
    if resolved.models:
        values = _apply_models(table, resolved, lambda p: _walk(stage, p))
    return _rebuild(type(stage), stage, values, resolved.soft, ())  # type: ignore[no-any-return]


def from_flat(cls: type[Stage], mapping: Mapping[str, Any]) -> Stage:
    """Build a stage tree from parameters by name (see :meth:`Stage.from_flat`).

    Parameters
    ----------
    cls
        The stage class at the root of the tree.
    mapping
        Values by flat name, dotted suffix, shared name or dotted path, or nested mappings.

    Returns
    -------
    Stage
    """
    table = router(cls)
    resolved = _with_computed_defaults(table, table.resolve(mapping))

    def base(path: FieldPath) -> Any:
        return _default(table.leaves[path], repr(table.preferred[path]))

    values = resolved.values
    if resolved.models:
        values = _apply_models(table, resolved, base)
    return _rebuild(cls, None, values, resolved.soft, ())  # type: ignore[no-any-return]


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


def _rebuild(
    cls: type,
    old: Any,
    changes: Mapping[FieldPath, Any],
    soft: frozenset[FieldPath],
    where: FieldPath,
) -> Any:
    """Rebuild one stage of a tree with changes (or, with ``old`` None, build it).

    ``changes`` are by path relative to the stage; ``soft`` are the paths set by a
    shared name, which a stage field that is None (e.g. ``Growth.transfer``) ignores;
    ``where`` is the stage's path from the root, for messages.
    """
    node = _node(cls)
    kwargs: dict[str, Any] = {}
    sub: dict[str, dict[FieldPath, Any]] = {}
    for path, value in changes.items():
        if len(path) == 1:
            kwargs[path[0]] = value
        else:
            sub.setdefault(path[0], {})[path[1:]] = value

    for name, (child_cls, optional) in node.children.items():
        if name in kwargs or (old is not None and name not in sub):
            continue
        child_changes = sub.get(name, {})
        old_child = None if old is None else getattr(old, name)
        if optional and old_child is None and all((name, *p) in soft for p in child_changes):
            # Only shared parameters: an absent optional stage stays absent.
            continue
        if old is not None and old_child is None:
            given = ", ".join(repr(_dotted((*where, name, *p))) for p in child_changes)
            raise ValueError(
                f"{_dotted((*where, name))} is None, so {given} cannot be set. Give the "
                "stage first."
            )
        child_soft = frozenset(p[1:] for p in soft if p[0] == name and len(p) > 1)
        new = _rebuild(child_cls, old_child, child_changes, child_soft, (*where, name))
        if old is None or new is not old_child:
            kwargs[name] = new

    if old is None:
        missing = [_dotted((*where, n)) for n in node.required if n not in kwargs]
        if missing:
            raise TypeError(f"{cls.__name__}: missing required parameter(s) {missing}.")
        return cls(**kwargs)
    if not kwargs:
        return old
    return old.evolve_own(**kwargs)
