'''
Classes defining the overall structure of the hmf framework.
'''
import copy
import sys
from _cache import Cache
class Model(object):
    """
    Class representing a component model.

    All component models in the framework should be subclassed from this. The
    features of this class are that it contains a class variable called _defaults
    which contains the defaults for the parameters of the specific model which
    subclasses this. These are checked and updated with passed parameters by
    the __init__ method.
    """

    _defaults = {}

    def __init__(self, **model_params):
        '''
        Constructor
        '''
        # Check that all parameters passed are valid
        for k in model_params:
            if k not in self._defaults:
                raise ValueError("%s is not a valid argument for the %s model" % (k, self.__class__.__name__))

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
    """
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
    return get_model_(name,mod)(**kwargs)

class Framework(Cache):
    """
    Class representing a coherent framework of component models.

    The specific subclasses of this class should be composed of methods that are
    decorated with either ``@_cache.parameter`` for things that are parameters,
    or ``@_cache.cached_property`` for derived quantities.

    Other methods are permissable, but may complicate matters if a derived
    quantity uses the non-cached_property method. Reserve these for utility
    methods.
    """
    def __init__(self):
        super(Framework, self).__init__()

    def update(self, **kwargs):
        """
        Update parameters of the framework with kwargs.
        """
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            del kwargs[k]

        if kwargs:
            raise ValueError("Invalid arguments: %s" % kwargs)
