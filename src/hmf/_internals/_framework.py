"""Classes defining the overall structure of the hmf framework."""

import copy
import difflib
import importlib
import logging
import re
import sys
import types
import warnings
from collections.abc import Mapping
from typing import Any, ClassVar, Literal, overload

import deprecation

from ._cache import hidden_loc

logger = logging.getLogger(__name__)


class Component:
    """
    Base class representing a component model.

    All components should be subclassed from this. Components are generally parts
    of the calculation which can take different models, example the HMF fitting
    functions, bias models, growth functions, etc.

    The feature of this class is that it contains a class variable called
    ``_defaults`` containing the defaults for the parameters of any specific model.
    These are checked and updated with passed parameters by
    the __init__ method.
    """

    _defaults: ClassVar[dict[str, Any]] = {}

    #: References to cite when this model is used, each a formatted citation string.
    #: Subclasses inherit their parent's references unless they set their own.
    references: ClassVar[tuple[str, ...]] = ()

    def __init__(self, **model_params):
        # Check that all parameters passed are valid
        for k in model_params:
            if k not in self._defaults:
                raise ValueError(f"{k} is not a valid argument for {self.__class__.__name__}.")

        # Gather model parameters
        self.params = copy.deepcopy(self._defaults)
        self.params.update(model_params)

    @classmethod
    def get_models(cls) -> Mapping[str, type]:
        """Get a read-only mapping of all implemented models for this component.

        The mapping is a live view of the registry, so it shows models registered
        later, but it can not be used to add or remove models.
        """
        return types.MappingProxyType(cls._plugins)


#: The reference for hmf itself, always included by :meth:`Framework.get_acknowledgments`.
HMF_REFERENCE = (
    "Murray, S. G., Power, C., Robotham, A. S. G., 2013. Astronomy and Computing 3, 23. "
    "arXiv:1306.6721"
)


def _class_path(cls: type) -> str:
    """A readable, fully-qualified name for a class, e.g. ``hmf.cosmology.cosmo.Cosmology``."""
    return f"{cls.__module__}.{cls.__qualname__}"


def get_base_components() -> list[type[Component]]:
    """Get a list of classes defining base components."""
    return Component.__subclasses__()


def get_base_component(name: [str, type[Component]]) -> type[Component]:
    """Return an actual class representing a component.

    Parameters
    ----------
    name
        The name of the component for which to return a class. If ``name`` is a class,
        then just return it (after checking that it is a Component).

    Returns
    -------
    cmp
        The Component subclass defining the desired component.
    """
    if isinstance(name, str):
        avail = [cmp for cmp in get_base_components() if cmp.__name__ == name]
        if not avail:
            raise ValueError(
                f"There are no components called '{name}'. Available: "
                f"{tuple(sorted(cmp.__name__ for cmp in get_base_components()))}"
            )
        if len(avail) > 1:
            warnings.warn(
                f"More than one component called '{name}'. Returning {_class_path(avail[-1])}.",
                stacklevel=2,
            )
        return avail[-1]
    # ValueError (not TypeError) is kept for backwards compatibility.
    if not isinstance(name, type) or not issubclass(name, Component):
        raise ValueError(f"{name!r} must be str or a Component subclass")  # noqa: TRY004
    return name


def pluggable(cls):
    """A decorator that adds pluggable capabilities."""
    cls._plugins = {}

    @classmethod
    def init_sc(kls, abstract=False):
        """Provide plugin capablity."""
        # Plugin framework
        if not abstract:
            existing = kls._plugins.get(kls.__name__)
            # Re-defining a class in the same module (e.g. a module reload) is not a
            # clash; a class of the same name from another module silently replacing
            # a registered model is.
            if existing is not None and existing.__module__ != kls.__module__:
                warnings.warn(
                    f"Registering {_class_path(kls)} as the {cls.__name__} model "
                    f"'{kls.__name__}' replaces {_class_path(existing)}, which was "
                    f"registered under the same name. Rename one of the classes, or pass "
                    f"the class itself (or its import path) rather than its name.",
                    UserWarning,
                    stacklevel=2,
                )
            kls._plugins[kls.__name__] = kls

    cls.__init_subclass__ = init_sc
    return cls


def _is_import_path(name: str) -> bool:
    """Whether a model name looks like an import path rather than a registry name."""
    return "." in name or ":" in name


def _import_from_path(path: str) -> Any:
    """Import an object given as ``package.module:Name`` or ``package.module.Name``."""
    if ":" in path:
        modname, _, attr = path.partition(":")
    else:
        modname, _, attr = path.rpartition(".")

    if not modname or not attr:
        raise ValueError(
            f"Could not interpret '{path}' as an import path. Use the form "
            "'package.module:Class' or 'package.module.Class'."
        )

    try:
        obj = importlib.import_module(modname)
    except ImportError as e:
        raise ValueError(
            f"Could not import module '{modname}' (from model '{path}'). Is it installed, "
            "or on your PYTHONPATH?"
        ) from e

    for part in attr.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as e:
            raise ValueError(
                f"Module '{modname}' has no attribute '{attr}' (from model '{path}')."
            ) from e
    return obj


def _not_found_message(name: str, available: list[str]) -> str:
    """The part of a model-not-found error that says what *could* be used."""
    msg = ""
    close = difflib.get_close_matches(name, available, n=3)
    if close:
        msg += f" Did you mean {' or '.join(repr(c) for c in close)}?"
    msg += f" Available: {tuple(sorted(available))}."
    msg += (
        " To use a model defined in another package, give its import path, "
        "e.g. 'package.module:Class', or list its module under 'plugins' in a "
        "TOML config."
    )
    return msg


def get_mdl(
    name: str | type[Component],
    kind: str | type[Component] | None = None,
) -> type[Component]:
    """Return a defined model with given name.

    Parameters
    ----------
    name
        The name of the model to return. Can be the actual model class itself.
        A string is first looked up as a registered model name (the name of the
        class). If it is not registered and contains a ``.`` or ``:``, it is taken
        as an import path, either ``package.module:Class`` or
        ``package.module.Class``, and that class is imported. Importing it also
        registers it, so afterwards it can be found by its class name too.
    kind
        The kind of component to search for.

    Returns
    -------
    model
        The actual model class (not instantiated).

    Raises
    ------
    ValueError
        If no model is registered under ``name`` and it can not be imported.
    TypeError
        If ``name`` is an import path to something that is not a subclass of
        ``kind`` (or of :class:`Component` if ``kind`` is not given).

    Examples
    --------
    >>> from hmf import get_mdl
    >>> get_mdl("PS", "BaseFittingFunction")
    <class 'hmf.mass_function.fitting_functions.PS'>
    >>> get_mdl("hmf.mass_function.fitting_functions:PS", "BaseFittingFunction")
    <class 'hmf.mass_function.fitting_functions.PS'>
    """
    if kind is not None:
        kind = get_base_component(kind)

    if isinstance(name, str):
        if kind is not None:
            if name in kind._plugins:
                return kind._plugins[name]
            if _is_import_path(name):
                return _check_imported_model(name, _import_from_path(name), kind)
            raise ValueError(
                f"The model {name} is not a defined {kind.__name__} model."
                + _not_found_message(name, list(kind._plugins))
            )
        # Try to get *any* model called by this name.
        avail_models = [
            (key, cls)
            for cmp in get_base_components()
            for key, cls in getattr(cmp, "_plugins", {}).items()
            if key == name
        ]
        if len(avail_models) > 1:
            warnings.warn(
                f"More than one model was found with name '{name}' "
                f"({', '.join(_class_path(m) for _, m in avail_models)}). Returning "
                f"{_class_path(avail_models[-1][1])}. Pass `kind` to choose one.",
                stacklevel=2,
            )
        if avail_models:
            return avail_models[-1][1]
        if _is_import_path(name):
            return _check_imported_model(name, _import_from_path(name), Component)
        raise ValueError(
            f"No model found with name '{name}'."
            + _not_found_message(
                name,
                list({k for cmp in get_base_components() for k in getattr(cmp, "_plugins", {})}),
            )
        )
    # ValueError (not TypeError) is kept for backwards compatibility.
    base = kind or Component
    if not isinstance(name, type):
        raise ValueError(  # noqa: TRY004
            f"{name!r} must be str or Component subclass ({base.__name__})"
        )
    if not issubclass(name, base):
        raise ValueError(  # noqa: TRY004
            f"{_class_path(name)} is not a {base.__name__} model (it must be a subclass "
            f"of {_class_path(base)})."
        )
    return name


def _check_imported_model(path: str, obj: Any, kind: type[Component]) -> type[Component]:
    """Check that an object imported for a model is a subclass of ``kind``."""
    if not isinstance(obj, type) or not issubclass(obj, kind):
        raise TypeError(
            f"'{path}' is not a subclass of {kind.__name__}, so it can not be used as a "
            f"{kind.__name__} model (got "
            f"{_class_path(obj) if isinstance(obj, type) else repr(obj)})."
        )
    return obj


@deprecation.deprecated("3.3.0", removed_in="4.0.0", details="Use get_mdl instead of get_model_")
def get_model_(name, mod):
    """
    Returns a class ``name`` from the module ``mod``.

    Parameters
    ----------
    name : str
        The class name of the appropriate model

    mod : str
        The module name of the appropriate module
    """
    return getattr(sys.modules[mod], name)


@deprecation.deprecated(
    "3.3.0", removed_in="4.0.0", details="Use get_mdl and pass **kwargs yourself."
)
def get_model(name, mod, **kwargs):
    r"""
    Returns an instance of ``name`` from the module ``mod``, with given params.

    Parameters
    ----------
    name : str
        The class name of the appropriate model
    mod : str
        The module name of the appropriate module

    \*\*kwargs :
        Any parameters for the instantiated model (including model parameters)
    """
    return get_model_(name, mod)(**kwargs)


class _Validator(type):
    def __call__(cls, *args, **kwargs):
        """Called when you call MyNewClass()."""
        obj = type.__call__(cls, *args, **kwargs)
        obj.validate()
        return obj


class Framework(metaclass=_Validator):
    """
    Class representing a coherent framework of component models.

    The specific subclasses of this class should be composed of methods that are
    decorated with either ``@_cache.parameter`` for things that are parameters,
    or ``@_cache.cached_property`` for derived quantities.

    Other methods are permissable, but may complicate matters if a derived
    quantity uses the non-``cached_property`` method. Reserve these for utility
    methods.

    Importantly, any parameter that may be passed to the constructor, *must* be
    defined as a ``parameter`` within the class so it may be set properly.
    """

    _validate = True

    #: References for the framework itself (not its component models), each a
    #: formatted citation string. References of parent framework classes are
    #: included too.
    references: ClassVar[tuple[str, ...]] = ()

    def validate(self):
        """Perform validation of the input parameters as they relate to each other."""

    def update(self, **kwargs):
        """Update parameters of the framework with kwargs.

        The update is atomic: if setting any parameter, or the validation of the
        new set of parameters, fails, all parameters are restored to their values
        from before the call, and the error is re-raised.

        Parameters
        ----------
        **kwargs
            New values of parameters of the framework. A key ``<name>_params``,
            where ``<name>`` is a sub-framework, takes a dict of parameters with
            which to update that sub-framework.

        Raises
        ------
        ValueError
            If a key is not a parameter of the framework (or a sub-framework's
            ``_params``), or if the new parameters fail validation.
        """
        params = self._parameter_index()
        invalid = [
            k for k in kwargs if k not in params and self._subframework_for_params(k) is None
        ]
        if invalid:
            msg = f"Invalid arguments to {type(self).__name__}.update(): {invalid}."
            for k in invalid:
                close = difflib.get_close_matches(k, list(params), n=3)
                if close:
                    msg += f" For '{k}', did you mean {' or '.join(repr(c) for c in close)}?"
            msg += f" Valid parameters: {tuple(sorted(params))}."
            raise ValueError(msg)

        undo = self._snapshot_parameters(kwargs)
        self._validate = False
        try:
            for k, v in kwargs.items():
                if k in params:
                    setattr(self, k, v)
                else:
                    self._subframework_for_params(k).update(**v)
            self._validate = True
            self.validate()
        except BaseException:
            self._validate = False
            for fmwork, name, old in reversed(undo):
                fmwork._restore_parameter(name, old)
            raise
        finally:
            self._validate = True

    def _parameter_index(self) -> dict[str, set]:
        """The parameters set on this instance, mapped to their dependent quantities."""
        return getattr(self, hidden_loc(self, "recalc_par_prop"), {})

    def _subframework_for_params(self, key: str) -> "Framework | None":
        """The sub-framework that ``key`` (``<name>_params``) updates, if any.

        ``<name>`` must be a ``@subframework``, or an instance attribute holding a
        :class:`Framework`. Other properties are not evaluated, so that a key like
        ``dndm_params`` does not compute ``dndm``.
        """
        if not key.endswith("_params"):
            return None
        name = key[: -len("_params")]
        prop = getattr(type(self), name, None)
        if isinstance(prop, property):
            return getattr(self, name) if getattr(prop.fget, "_is_subframework", False) else None
        sub = vars(self).get(name)
        return sub if isinstance(sub, Framework) else None

    def _snapshot_parameters(self, kwargs: dict[str, Any]) -> list[tuple["Framework", str, Any]]:
        """Record the current values of the parameters that ``kwargs`` would change."""
        params = self._parameter_index()
        undo = []
        for k, v in kwargs.items():
            if k in params:
                # Read the stored value directly, so that taking the snapshot does not
                # register a dependency for a quantity that is being computed.
                undo.append((self, k, getattr(self, hidden_loc(self, k))))
            elif (sub := self._subframework_for_params(k)) is not None and isinstance(v, dict):
                undo.extend(sub._snapshot_parameters(v))
        return undo

    def _restore_parameter(self, name: str, old: Any) -> None:
        """Set a parameter back to a value it had before, without validation."""
        validate = self._validate
        self._validate = False
        try:
            # Setting a non-empty dict merges it into the stored one, so clear the
            # stored dict first, to drop keys the failed update added.
            if isinstance(old, dict):
                setattr(self, name, {})
            setattr(self, name, old)
        finally:
            self._validate = validate

    def clone(self, **kwargs):
        """Create and return an updated clone of the current object."""
        clone = copy.deepcopy(self)
        clone.update(**kwargs)
        return clone

    def _model_references(self, name: str, model: Any) -> tuple[str, ...]:
        """Return the references for the model set on the ``name`` parameter.

        Override in subclasses whose models are not :class:`Component` classes
        (e.g. an astropy cosmology). The default uses the model's ``references``.
        """
        return tuple(getattr(model, "references", ()))

    @overload
    def get_acknowledgments(self, flat: Literal[False] = False) -> dict[str, tuple[str, ...]]: ...

    @overload
    def get_acknowledgments(self, flat: Literal[True]) -> list[str]: ...

    def get_acknowledgments(self, flat: bool = False) -> dict[str, tuple[str, ...]] | list[str]:
        """Get the references to cite for the current setup of the framework.

        The references are grouped by where they come from, in this order:

        - ``"hmf"``: the paper describing hmf itself.
        - The framework's class name (e.g. ``"MassFunction"``): the framework's own
          ``references``, including those of its parent framework classes. Only
          present if there are any.
        - Each ``*_model`` parameter (e.g. ``"hmf_model"``, ``"transfer_model"``):
          the references of the model currently set on it. A model with nothing to
          cite gives an empty tuple; a parameter set to ``None`` is left out.
        - Each sub-framework, with its keys prefixed by the sub-framework's name
          (e.g. ``"tracer.transfer_model"``).

        The keys say which model each citation belongs to, not whether that model
        was used in a quantity you computed. All references are read from the model
        *classes*, so this does not compute anything.

        Parameters
        ----------
        flat
            If True, return a single list of all the references with duplicates
            removed (keeping the first occurrence), e.g. for a paper's
            bibliography.

        Returns
        -------
        dict or list
            A dict mapping each source to a tuple of formatted citations, or a flat
            list of citations if ``flat`` is True.

        Examples
        --------
        >>> from hmf import MassFunction
        >>> mf = MassFunction(hmf_model="Tinker08")
        >>> by_model = mf.get_acknowledgments()
        >>> bibliography = mf.get_acknowledgments(flat=True)
        """
        refs = {"hmf": (HMF_REFERENCE,)}
        self._collect_references(refs, prefix="", seen=set())
        if flat:
            return list(dict.fromkeys(ref for group in refs.values() for ref in group))
        return refs

    def _collect_references(
        self, refs: dict[str, tuple[str, ...]], prefix: str, seen: set[int]
    ) -> None:
        """Add the references of this framework and its sub-frameworks to refs."""
        if id(self) in seen:
            return
        seen.add(id(self))

        own = tuple(
            ref
            for kls in reversed(type(self).__mro__)
            for ref in kls.__dict__.get("references", ())
        )
        if own:
            refs[prefix.rstrip(".") or type(self).__name__] = own

        for name in getattr(self, "_" + self.__class__.__name__ + "__recalc_par_prop"):
            if not name.endswith("_model"):
                continue
            model = getattr(self, name)
            if model is not None:
                refs[prefix + name] = self._model_references(name, model)

        for name in dir(type(self)):
            prop = getattr(type(self), name, None)
            if isinstance(prop, property) and getattr(prop.fget, "_is_subframework", False):
                getattr(self, name)._collect_references(refs, prefix + name + ".", seen)

    @classmethod
    def get_all_parameter_names(cls):
        """Yield all parameter names in the class."""
        K = cls()
        return getattr(K, "_" + K.__class__.__name__ + "__recalc_par_prop")

    @classmethod
    def get_all_parameter_defaults(cls, recursive=True):
        """Dictionary of all parameters and defaults."""
        K = cls()
        out = {name: getattr(K, name) for name in cls.get_all_parameter_names()}

        if recursive:
            for name, default in out.items():
                if default == {} and name.endswith("_params"):
                    try:
                        out[name] = getattr(K, name.replace("_params", "_model"))._defaults

                    except Exception:
                        logger.info(
                            f"Exception caught in getting defaults for {name}",
                            exc_info=True,
                        )

        return out

    @property
    def parameter_values(self):
        """Dictionary of all parameters and their current values."""
        return {
            name: getattr(self, name)
            for name in getattr(self, "_" + self.__class__.__name__ + "__recalc_par_prop")
        }

    @classmethod
    def quantities_available(cls):
        """Obtain a list of all available output quantities."""
        all_names = cls.get_all_parameter_names()
        return [
            name
            for name in dir(cls)
            if name not in all_names and not name.startswith("__") and name not in dir(Framework)
        ]

    @classmethod
    def _get_all_parameters(cls):
        """Yield all parameters as tuples of (name,obj)."""
        for name in cls.get_all_parameter_names():
            yield name, getattr(cls, name)

    def get_dependencies(self, *q):
        """
        Determine all parameter dependencies of the quantities in q.

        Parameters
        ----------
        q : str
            String(s) labelling a quantity

        Returns
        -------
        deps : set
            A set containing all parameters on which quantities in q are dependent.
        """
        recalc_prpa = getattr(self, "_" + self.__class__.__name__ + "__recalc_prop_par")

        deps = set()
        for quant in q:
            # Accessing the quantity populates its entry in the dependency index.
            getattr(self, quant)
            deps.update(recalc_prpa[quant])

        return deps

    @classmethod
    def parameter_info(cls, names=None):
        """
        Prints information about each parameter in the class.

        Optionally, restrict printed parameters to those found in the list of names
        provided.
        """
        docs = ""
        for name, obj in cls._get_all_parameters():
            if names and name not in names:
                continue

            docs += name + " : "
            objdoc = obj.__doc__.split("\n")

            if len(objdoc[0]) == len("**Parameter**: "):
                del objdoc[0]
            else:
                objdoc[0] = objdoc[0][len("**Parameter**: ") :]

            objdoc = [o.strip() for o in objdoc]

            while "" in objdoc:
                objdoc.remove("")

            for i, line in enumerate(objdoc):
                if ":type:" in line:
                    docs += line.split(":type:")[-1].strip() + "\n    "
                    del objdoc[i]
                    break

            docs += "\n    ".join(objdoc) + "\n\n"

        docs = re.sub(r"\n{3,}", "\n\n", docs)
        print(docs[:-1])  # noqa
