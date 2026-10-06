"""The :class:`Model` base class and the per-kind model registry.

A *model* is one interchangeable implementation of a piece of physics: a fitting
function, a transfer function, a filter, a mass definition, ... Models are frozen,
keyword-only ``attrs`` classes whose fields are the model's parameters. They have no
cosmology of their own: their methods take every physical input as an explicit
argument (issue #389).

Kinds and registration
----------------------
A *kind* is an abstract base class declared with ``kind=True``. It owns a registry of
its models::

    @attrs.frozen(kw_only=True)
    class FittingFunction(Model, kind=True):
        @abc.abstractmethod
        def fsigma(self, sigma): ...

    @attrs.frozen(kw_only=True)
    class Tinker08(FittingFunction, alias="Tinker08"):
        A: float = field(default=0.186, doc="Amplitude.")

        def fsigma(self, sigma): ...

Every concrete subclass of a kind is registered in that kind (and in any enclosing
kind) under its **qualified name**, ``package.module:Class``, and, if it gives one,
under a short **alias**. An alias is unique within a kind: registering a second class
under the same alias is an error, unless the new class passes ``override=True``. A
class defined in user code therefore never silently replaces a built-in model.
Intermediate base classes that should not be registered pass ``abstract=True``.

Re-defining a class with the same qualified name (re-running a notebook cell, or
:func:`importlib.reload`) replaces the old registration.

Lookup
------
Lookup is per kind only; there is no global lookup by name.
``FittingFunction.get(name)`` accepts an alias, a qualified name, an import path
(``"pkg.mod:Cls"`` or ``"pkg.mod.Cls"``) of a class not yet imported, or a class. If
that fails, it loads the plugins advertised in the ``hmf.models`` entry-point group
(once each) and tries again, and finally raises :class:`ModelNotFoundError` with
suggestions. A plugin package advertises its models in its ``pyproject.toml``::

    [project.entry-points."hmf.models"]
    my_models = "my_package.models"

The entry point may name a module (whose import registers its models) or a class.

Subclasses must be decorated with ``@attrs.frozen(kw_only=True)``
-------------------------------------------------------------------
Registration happens in ``__attrs_init_subclass__``, which ``attrs`` calls once the
decorator has built the final class. So a class is registered only when it is
decorated, and the registry always holds the final (slotted) class. The
"Parameters" section of the class docstring is generated from the fields at the
same point.

Slotted ``attrs`` classes are created twice: once by the ``class`` statement, and
again by the decorator, which calls ``__init_subclass__`` a second time *without*
the class keywords. :class:`Model` therefore only validates and stores its keywords in
``__init_subclass__``. Mixins with their own ``__init_subclass__`` should likewise
tolerate the second call.
"""

from __future__ import annotations

import abc
import difflib
import importlib
import importlib.metadata
import inspect
import warnings
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, ClassVar, NamedTuple, Self

import attrs

from ._fields import Documented

__all__ = [
    "ENTRY_POINT_GROUP",
    "DuplicateAliasError",
    "Model",
    "ModelNotFoundError",
]

#: The entry-point group in which plugin packages advertise their models.
ENTRY_POINT_GROUP = "hmf.models"

#: Entry points already loaded, by their value (``"pkg.mod"`` or ``"pkg.mod:Cls"``).
_loaded_entry_points: set[str] = set()


class ModelNotFoundError(LookupError):
    """No model of the requested kind is registered or importable under a name."""


class DuplicateAliasError(ValueError):
    """A model was registered under an alias already taken within its kind."""


class _Options(NamedTuple):
    """The class keywords a :class:`Model` subclass was defined with."""

    alias: str | None
    abstract: bool
    override: bool
    kind: bool


def qualified_name(cls: type) -> str:
    """Return the qualified name of a class: ``package.module:Class``."""
    return f"{cls.__module__}:{cls.__qualname__}"


@attrs.frozen(kw_only=True)
class Model(Documented, abc.ABC):
    """Base class of every v4 model.

    Do not subclass it directly to write a model: subclass a model *kind* (a direct
    subclass declared with ``kind=True``). See the module documentation for how
    models are registered and looked up.

    Subclasses accept these class keywords (``class M(Kind, alias="M"): ...``):

    ``alias`` (str, optional)
        A short name to register the model under, in addition to its qualified name.
        It must not contain ``.`` or ``:``.
    ``abstract`` (bool)
        If True, the class is a base for other models and is not registered.
    ``override`` (bool)
        If True, the alias may replace another model's alias in the same kind.
    ``kind`` (bool)
        If True, the class is a new model kind with its own registry. A kind is
        never registered as a model itself.
    """

    #: References to cite when the model is used, each a formatted citation.
    references: ClassVar[tuple[str, ...]] = ()

    #: The exact source (paper, version, table or equation) of the parameter
    #: defaults, e.g. "Tinker et al. 2008, ApJ 688, 709 (published), Table 2".
    parameter_source: ClassVar[str] = ""

    #: The short name the model is registered under, if any. Set by the ``alias``
    #: class keyword; never inherited.
    alias: ClassVar[str | None] = None

    # Set on each kind: its registry (qualified name -> class) and aliases
    # (alias -> qualified name).
    _registry: ClassVar[dict[str, type[Model]]]
    _aliases: ClassVar[dict[str, str]]
    _model_options: ClassVar[_Options]

    def __init_subclass__(
        cls,
        /,
        *,
        alias: str | None = None,
        abstract: bool = False,
        override: bool = False,
        kind: bool = False,
        **kwargs: Any,
    ) -> None:
        """Validate and store the class keywords of a new model class."""
        super().__init_subclass__(**kwargs)
        if "_model_options" in cls.__dict__:
            # attrs is building the slotted class, and called us again without the
            # class keywords: the ones stored the first time are already copied over.
            return

        if alias is not None and (not alias or "." in alias or ":" in alias):
            raise ValueError(
                f"{cls.__qualname__}: alias {alias!r} must be a non-empty name without '.' "
                "or ':' (those mark import paths)."
            )
        if kind and alias is not None:
            raise TypeError(f"{cls.__qualname__}: a model kind can not have an alias.")
        if not (kind or abstract or cls._kinds()):
            raise TypeError(
                f"{cls.__qualname__} subclasses Model directly. A model must subclass a "
                "model kind (a class defined with `kind=True`), or pass `abstract=True`."
            )
        cls._model_options = _Options(alias, abstract, override, kind)
        cls.alias = alias

    @classmethod
    def __attrs_init_subclass__(cls) -> None:
        """Register the class attrs has just built, and document its fields."""
        options = cls._model_options
        if options.kind:
            cls._registry = {}
            cls._aliases = {}
        elif not options.abstract:
            kinds = cls._kinds()
            # Check every kind before changing any, so a failure registers nothing.
            for k in kinds:
                k._check_alias(cls, options)
            for k in kinds:
                k._register(cls, options)
        super().__attrs_init_subclass__()

    @classmethod
    def _kinds(cls) -> list[type[Model]]:
        """The kinds this class belongs to, nearest first (excluding itself)."""
        return [k for k in cls.__mro__[1:] if "_registry" in k.__dict__]

    @classmethod
    def _check_alias(cls, model: type[Model], options: _Options) -> None:
        """Raise if ``model``'s alias is taken in this kind. ``cls`` must be a kind."""
        if options.alias is None or options.override:
            return
        name = qualified_name(model)
        existing = cls._aliases.get(options.alias)
        if existing is not None and existing != name:
            raise DuplicateAliasError(
                f"Can't register {name} as the {cls.__name__} model {options.alias!r}: "
                f"that alias already belongs to {existing}. Choose another alias, or pass "
                "`override=True` in the class definition to replace it."
            )

    @classmethod
    def _register(cls, model: type[Model], options: _Options) -> None:
        """Register ``model`` in this kind's registry. ``cls`` must be a kind."""
        name = qualified_name(model)
        if options.alias is not None:
            cls._aliases[options.alias] = name
        # The same qualified name again is a re-definition of the same class (a module
        # reload, a re-run notebook cell): replace it.
        cls._registry[name] = model

    @classmethod
    def qualified_name(cls) -> str:
        """Return this class's qualified name, ``package.module:Class``."""
        return qualified_name(cls)

    @classmethod
    def _kind(cls) -> type[Model]:
        """The kind that owns lookups made through ``cls``."""
        if "_registry" in cls.__dict__:
            return cls
        kinds = cls._kinds()
        if not kinds:
            raise TypeError(
                f"{cls.__qualname__} is not a model kind, nor a subclass of one, so it has "
                "no registry. Look models up through their kind, e.g. FittingFunction.get()."
            )
        return kinds[0]

    @classmethod
    def get_models(cls) -> Mapping[str, type[Self]]:
        """Return the registered models of this kind, by qualified name.

        Plugins advertised in the ``hmf.models`` entry-point group are loaded first.

        Returns
        -------
        Mapping
            A read-only mapping from qualified name to class. Called on a kind, it is
            a live view of the registry. Called on a non-kind subclass, it holds only
            the models that subclass it.
        """
        kind = cls._kind()
        _load_entry_points()
        if kind is cls:
            return MappingProxyType(cls._registry)  # type: ignore[arg-type]
        return MappingProxyType(
            {name: m for name, m in kind._registry.items() if issubclass(m, cls)}
        )

    @classmethod
    def get_aliases(cls) -> Mapping[str, str]:
        """Return a read-only mapping from alias to qualified name, for this kind."""
        kind = cls._kind()
        return MappingProxyType(
            {a: n for a, n in kind._aliases.items() if issubclass(kind._registry[n], cls)}
        )

    @classmethod
    def get(cls, name: str | type[Model]) -> type[Self]:
        """Look up a model of this kind.

        Parameters
        ----------
        name
            An alias (``"Tinker08"``), a qualified name
            (``"hmf.core.fits:Tinker08"``), an import path to a class not yet imported
            (``"pkg.mod:Cls"`` or ``"pkg.mod.Cls"``), or a class.

        Returns
        -------
        type
            The model class (not an instance). It is always a subclass of ``cls``.

        Raises
        ------
        ModelNotFoundError
            If no model of this kind is known by ``name``, even after loading the
            ``hmf.models`` entry points and trying ``name`` as an import path.
        TypeError
            If ``name`` resolves to something that is not a concrete subclass of
            ``cls``.
        """
        kind = cls._kind()
        found: object
        if isinstance(name, str):
            found = kind._lookup(name)
            if found is None and _is_import_path(name):
                found = _import_from_path(name)
            if found is None:
                _load_entry_points()
                found = kind._lookup(name)
            if found is None:
                raise ModelNotFoundError(cls._not_found_message(name))
        else:
            found = name
        return cls._check_model(found, name)

    @classmethod
    def _lookup(cls, name: str) -> type[Model] | None:
        """Find ``name`` among this kind's aliases and qualified names."""
        qualname = cls._aliases.get(name, name)
        return cls._registry.get(qualname)

    @classmethod
    def _check_model(cls, found: object, name: object) -> type[Self]:
        """Check that a looked-up object is a concrete model of this class."""
        if not (isinstance(found, type) and issubclass(found, cls)):
            raise TypeError(f"{name!r} resolved to {found!r}, which is not a {cls.__name__}.")
        options = found.__dict__.get("_model_options")
        if inspect.isabstract(found) or (
            options is not None and (options.abstract or options.kind)
        ):
            raise TypeError(
                f"{name!r} resolved to {qualified_name(found)}, which is abstract: it can "
                "not be used as a model."
            )
        return found

    @classmethod
    def _not_found_message(cls, name: str) -> str:
        """The message of the error raised when ``name`` can not be found."""
        kind = cls._kind()
        candidates = [*kind._aliases, *kind._registry]
        msg = f"No {cls.__name__} model is known as {name!r}."
        close = difflib.get_close_matches(name, candidates, n=3)
        if close:
            msg += f" Did you mean {' or '.join(repr(c) for c in close)}?"
        msg += f" Available aliases: {tuple(sorted(kind._aliases))}."
        msg += (
            " A model from another package can be given by its import path, e.g. "
            f"'package.module:Class', or advertised in the {ENTRY_POINT_GROUP!r} "
            "entry-point group."
        )
        return msg


def _is_import_path(name: str) -> bool:
    """Whether a model name looks like an import path rather than an alias."""
    return "." in name or ":" in name


def _import_from_path(path: str) -> object | None:
    """Import ``package.module:Name`` or ``package.module.Name``; None if impossible."""
    if ":" in path:
        modname, _, attr = path.partition(":")
    else:
        modname, _, attr = path.rpartition(".")
    if not modname or not attr:
        return None
    try:
        obj: object = importlib.import_module(modname)
    except ImportError:
        return None
    for part in attr.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def _load_entry_points() -> None:
    """Load the ``hmf.models`` entry points not loaded yet (registering their models).

    A plugin that fails to load is reported with a warning and skipped, so that one
    broken plugin does not break every lookup.
    """
    for ep in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
        if ep.value in _loaded_entry_points:
            continue
        _loaded_entry_points.add(ep.value)
        try:
            ep.load()
        except Exception as e:  # noqa: BLE001 - any failure of a third-party plugin
            warnings.warn(
                f"Could not load the {ENTRY_POINT_GROUP!r} entry point {ep.name!r} "
                f"({ep.value}): {type(e).__name__}: {e}",
                RuntimeWarning,
                stacklevel=3,
            )
