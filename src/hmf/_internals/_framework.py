"""
Classes defining the overall structure of the hmf framework.
"""
import copy
import sys


# from _cache import Cache


class Component(object):
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
                    f"{k} is not a valid argument for the {self.__class__.__name__} model"
                )

        # Gather model parameters
        self.params = copy.copy(self._defaults)
        self.params.update(model_params)


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


class Framework(object):
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

    def __init__(self):
        super(Framework, self).__init__()

    def update(self, **kwargs):
        """
        Update parameters of the framework with kwargs.
        """
        for k, v in list(kwargs.items()):
            # If key is just a parameter to the class, just update it.
            if hasattr(self, k):
                setattr(self, k, kwargs.pop(k))

            # If key is a dictionary of parameters to a sub-framework, update the sub-framework
            elif k.endswith("_params") and isinstance(getattr(self, k[:-7]), Framework):
                getattr(self, k[:-7]).update(**kwargs.pop(k))

        if kwargs:
            raise ValueError("Invalid arguments: %s" % kwargs)

    def clone(self, **kwargs):
        """Create and return an updated clone of the current object."""
        clone = copy.deepcopy(self)
        clone.update(**kwargs)
        return clone

    @classmethod
    def get_all_parameter_names(cls):
        "Yield all parameter names in the class."
        K = cls()
        return getattr(K, "_" + K.__class__.__name__ + "__recalc_par_prop")

    @classmethod
    def get_all_parameter_defaults(cls, recursive=True):
        "Dictionary of all parameters and defaults"
        K = cls()
        out = {}
        for name in cls.get_all_parameter_names():
            out[name] = getattr(K, name)

        if recursive:
            for name, default in out.items():
                if default == {} and name.endswith("_params"):
                    try:
                        out[name] = getattr(
                            getattr(K, name.replace("_params", "_model")), "_defaults"
                        )
                    except Exception as e:
                        print(e)
        return out

    @property
    def parameter_values(self):
        "Dictionary of all parameters and their current values"
        out = {}
        for name in getattr(self, "_" + self.__class__.__name__ + "__recalc_par_prop"):
            out[name] = getattr(self, name)
        return out

    @classmethod
    def quantities_available(cls):
        return [
            name
            for name in dir(cls)
            if name not in cls.get_all_parameter_names()
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

        Optionally, restrict printed parameters to those found in the list of names provided.
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
        print((docs[:-1]))
