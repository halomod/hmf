"""Classes defining the overall structure of the hmf framework."""

import copy
import difflib
import importlib
import inspect
import logging
import re
import sys
import warnings
from typing import Any, ClassVar, Literal, overload

import deprecation

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
    def get_models(cls) -> dict[str, type]:
        """Get a dictionary of all implemented models for this component."""
        return cls._plugins


#: The reference for hmf itself, always included by :meth:`Framework.get_acknowledgments`.
HMF_REFERENCE = (
    "Murray, S. G., Power, C., Robotham, A. S. G., 2013. Astronomy and Computing 3, 23. "
    "arXiv:1306.6721"
)


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
                f"There are no components called '{name}'. Available: {get_base_components()}"
            )
        if len(avail) > 1:
            warnings.warn(
                f"More than one component called '{name}'. Returning {avail[-1]}.", stacklevel=2
            )
        return avail[-1]
    try:
        assert issubclass(name, Component)
        return name
    except TypeError as e:
        raise ValueError(f"{name} must be str or a Component subclass") from e


def pluggable(cls):
    """A decorator that adds pluggable capabilities."""
    cls._plugins = {}

    @classmethod
    def init_sc(kls, abstract=False):
        """Provide plugin capablity."""
        # Plugin framework
        if not abstract:
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
                f"More than one model was found with name '{name}'. Returning "
                f"{avail_models[-1][1]}.",
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
    try:
        assert issubclass(name, kind or Component)
        return name
    except TypeError as e:
        raise ValueError(f"{name} must be str or Component subclass") from e


def _check_imported_model(path: str, obj: Any, kind: type[Component]) -> type[Component]:
    """Check that an object imported for a model is a subclass of ``kind``."""
    if not isinstance(obj, type) or not issubclass(obj, kind):
        raise TypeError(
            f"'{path}' is not a subclass of {kind.__name__}, so it can not be used as a "
            f"{kind.__name__} model (got {obj!r})."
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
        """Update parameters of the framework with kwargs."""
        self._validate = False
        try:
            for k in list(kwargs.keys()):
                # If key is just a parameter to the class, just update it.
                if hasattr(self, k):
                    setattr(self, k, kwargs.pop(k))

                # If key is a dictionary of parameters to a sub-framework,
                # update the sub-framework
                elif k.endswith("_params") and isinstance(getattr(self, k[:-7]), Framework):
                    getattr(self, k[:-7]).update(**kwargs.pop(k))
            self._validate = True
            self.validate()
        except Exception:
            self._validate = True
            raise

        if kwargs:
            raise ValueError(f"Invalid arguments: {kwargs}")

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
    def _descriptor_names(cls, marker: str) -> list[str]:
        """Names of the class's properties whose getter carries ``marker``.

        Names are ordered by the class that first defines them, base classes first,
        then by definition order within each class.
        """
        names = {}
        for kls in reversed(cls.__mro__):
            for name in vars(kls):
                prop = inspect.getattr_static(cls, name, None)
                if isinstance(prop, property) and getattr(prop.fget, marker, False):
                    names[name] = None
        return list(names)

    @classmethod
    def get_all_parameter_names(cls) -> list[str]:
        """Return all parameter names in the class.

        These are read from the class's ``@parameter`` definitions, so the class is
        not instantiated.
        """
        return cls._descriptor_names("_is_parameter")

    @classmethod
    def get_all_parameter_defaults(cls, recursive=True):
        """Dictionary of all parameters and defaults.

        Defaults may be set with some logic in ``__init__``, so they are read from an
        instance. The instance is not validated, so no quantities are computed.
        """
        # type.__call__ skips the validation done by the _Validator metaclass.
        K = type.__call__(cls)
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
    def quantities_available(cls) -> list[str]:
        """Obtain a list of all available (public) output quantities.

        These are the class's ``@cached_quantity`` and ``@subframework`` definitions,
        read without instantiating the class.
        """
        quantities = cls._descriptor_names("_is_cached_quantity")
        quantities += cls._descriptor_names("_is_subframework")
        return sorted(name for name in quantities if not name.startswith("_"))

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
