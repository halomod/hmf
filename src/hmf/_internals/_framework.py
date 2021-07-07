"""Classes defining the overall structure of the hmf framework."""
import copy
import deprecation
import logging
import sys
import warnings
from typing import Dict, List, Optional, Type, Union

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

    _defaults = {}

    def __init__(self, **model_params):
        # Check that all parameters passed are valid
        for k in model_params:
            if k not in self._defaults:
                raise ValueError(
                    f"{k} is not a valid argument for {self.__class__.__name__}."
                )

        # Gather model parameters
        self.params = copy.copy(self._defaults)
        self.params.update(model_params)

    @classmethod
    def get_models(cls) -> Dict[str, Type]:
        """Get a dictionary of all implemented models for this component."""
        return cls._plugins


def get_base_components() -> List[Type[Component]]:
    """Get a list of classes defining base components."""
    return Component.__subclasses__()


def get_base_component(name: [str, Type[Component]]) -> Type[Component]:
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
                f"{get_base_components()}"
            )
        if len(avail) > 1:
            warnings.warn(
                f"More than one component called '{name}'. Returning {avail[-1]}."
            )
        return avail[-1]
    else:
        try:
            assert issubclass(name, Component)
            return name
        except TypeError:
            raise ValueError(f"{name} must be str or a Component subclass")


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


def get_mdl(
    name: Union[str, Type[Component]],
    kind: Optional[Union[str, Type[Component]]] = None,
) -> Type[Component]:
    """Return a defined model with given name.

    Parameters
    ----------
    name
        The name of the model to return. Can be the actual model class itself.
    kind
        The kind of component to search for.

    Returns
    -------
    model
        The actual model class (not instantiated).
    """
    if kind is not None:
        kind = get_base_component(kind)

    if isinstance(name, str):
        if kind is not None:
            try:
                return kind._plugins[name]
            except KeyError:
                raise ValueError(
                    f"The model {name} is not a defined {kind} model. Available: "
                    f"{tuple(kind._plugins.keys())}"
                )
        else:
            # Try to get *any* model called by this name.
            avail_models = [
                (key, cls)
                for cmp in get_base_components()
                for key, cls in cmp._plugins.items()
                if key == name
            ]
            if len(avail_models) > 1:
                warnings.warn(
                    f"More than one model was found with name '{name}'. Returning "
                    f"{avail_models[-1][1]}."
                )
            if not avail_models:
                raise ValueError(f"No model found with name '{name}'.")
            return avail_models[-1][1]
    else:
        try:
            assert issubclass(name, kind or Component)
            return name
        except TypeError:
            raise ValueError(f"{name} must be str or Component subclass")


@deprecation.deprecated(
    "3.3.0", removed_in="4.0.0", details="Use get_mdl instead of get_model_"
)
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
        """Called when you call MyNewClass()"""
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
    _validate_every_param_set = False

    def validate(self):
        """Perform validation of the input parameters as they relate to each other."""
        pass

    def update(self, **kwargs):
        """
        Update parameters of the framework with kwargs.
        """
        self._validate = False
        try:
            for k in list(kwargs.keys()):
                # If key is just a parameter to the class, just update it.
                if hasattr(self, k):
                    setattr(self, k, kwargs.pop(k))

                # If key is a dictionary of parameters to a sub-framework,
                # update the sub-framework
                elif k.endswith("_params") and isinstance(
                    getattr(self, k[:-7]), Framework
                ):
                    getattr(self, k[:-7]).update(**kwargs.pop(k))
            self._validate = True
            self.validate()
        except Exception:
            self._validate = True
            raise

        if kwargs:
            raise ValueError("Invalid arguments: %s" % kwargs)

    def clone(self, **kwargs):
        """Create and return an updated clone of the current object."""
        clone = copy.deepcopy(self)
        clone.update(**kwargs)
        return clone

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
                        out[name] = getattr(
                            K, name.replace("_params", "_model")
                        )._defaults

                    except Exception as e:
                        logger.info(e)
        return out

    @property
    def parameter_values(self):
        """Dictionary of all parameters and their current values"""
        return {
            name: getattr(self, name)
            for name in getattr(
                self, "_" + self.__class__.__name__ + "__recalc_par_prop"
            )
        }

    @classmethod
    def quantities_available(cls):
        """Obtain a list of all available output quantities."""
        all_names = cls.get_all_parameter_names()
        return [
            name
            for name in dir(cls)
            if name not in all_names
            and not name.startswith("__")
            and name not in dir(Framework)
        ]

    @classmethod
    def _get_all_parameters(cls):
        """Yield all parameters as tuples of (name,obj)"""
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
        deps = set()
        for quant in q:
            getattr(self, quant)

            deps.update(
                getattr(
                    self, "_" + self.__class__.__name__ + "__recalc_prop_par_static"
                )[quant]
            )

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
            while "\n\n\n" in docs:
                docs.replace("\n\n\n", "\n\n")
        print(docs[:-1])  # noqa
