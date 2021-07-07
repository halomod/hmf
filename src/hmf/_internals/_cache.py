"""
This module defines two decorators, based on the code from
http://forrst.com/posts/Yet_another_caching_property_decorator_for_Pytho-PBy

They are both designed to cache class properties, but have the added
functionality of being automatically updated when a parent property is
updated.
"""
import warnings
from copy import deepcopy
from functools import update_wrapper


def hidden_loc(obj, name):
    """
    Generate the location of a hidden attribute.

    Importantly deals with attributes beginning with an underscore.
    """
    return ("_" + obj.__class__.__name__ + "__" + name).replace("___", "__")


def cached_quantity(f):
    """
    A robust property caching decorator.

    This decorator is intended for use with the complementary `parameter` decorator.
    It caches the decorated quantity, contingent on the parameters it depends on.
    When those parameters are modified, a further call to the quantity will result
    in a recalculation.

    Examples
    --------
    >>>   class CachedClass:
    >>>      @parameter
    >>>      def a_param(self,val):
    >>>         return val
    >>>
    >>>      @cached_quantity
    >>>      def a_quantity(self):
    >>>         return 2*self.a_param
    >>>
    >>>      @cached_quantity
    >>>      def a_child_quantity(self):
    >>>         return self.a_quantity**3

    This code will calculate ``a_child_quantity`` on the first call, but return the cached
    value on all subsequent calls. If `a_param` is modified, the
    calculation of either `a_quantity` and `a_child_quantity` will be re-performed when requested.
    """
    name = f.__name__

    def _get_property(self):
        # Location of the property to be accessed
        prop = hidden_loc(self, name)

        # Locations of indexes [they are set up in the parameter decorator]
        _recalc = hidden_loc(self, "recalc")
        _recalc_prpa = hidden_loc(self, "recalc_prop_par")
        _activeq = hidden_loc(self, "active_q")
        _recalc_papr = hidden_loc(self, "recalc_par_prop")
        _subframeworks = hidden_loc(self, "subframeworks")

        # actual objects
        recalc = getattr(self, _recalc)
        recalc_prpa = getattr(self, _recalc_prpa)
        activeq = getattr(self, _activeq)
        recalc_papr = getattr(self, _recalc_papr)
        subframeworks = [getattr(self, s) for s in getattr(self, _subframeworks, set())]

        # First, if this property has already been indexed,
        # we must ensure all dependent parameters are copied into active indexes,
        # otherwise they will be lost to their parents.
        if name in recalc:
            for pr in activeq:
                try:
                    recalc_prpa[pr].update(recalc_prpa[name])
                except KeyError:
                    raise KeyError(
                        f"When getting {name}, couldn't find {pr} in recalc_prpa. Had {list(recalc_prpa.keys())}."
                    )

            # check all quantities for dependence on subframeworks and update their entries
            for s in subframeworks:
                if getattr(s, hidden_loc(s, "recalc")).get(":" + name, False):
                    recalc[name] = True
                    getattr(s, hidden_loc(s, "recalc"))[":" + name] = False

        # If this property already in recalc and doesn't need updating, just return.
        if not recalc.get(name, True):
            return getattr(self, prop)

        # Otherwise, if its in recalc, and needs updating, just update it
        elif name in recalc:
            value = f(self)
            setattr(self, prop, value)

            # Ensure it doesn't need to be recalculated again
            recalc[name] = False

            return value

        # Otherwise, we need to create its index for caching.
        # if name is already there, can only be because the method has been supered.
        supered = name in activeq
        if not supered:
            recalc_prpa[
                name
            ] = set()  # Empty set to which parameter names will be added
            activeq.add(name)

        # Go ahead and calculate the value -- each parameter accessed will add itself to the index.
        value = f(self)
        setattr(self, prop, value)

        # Invert the index
        for par in recalc_prpa[name]:
            recalc_papr[par].add(name)

        # Copy index to static dict, and remove the index (so that parameters don't keep
        # on trying to add themselves)
        if not supered:  # If super, don't want to remove the name just yet.
            recalc_prpa[name] = deepcopy(recalc_prpa[name])
            activeq.remove(name)

        # Add entry to master recalc list
        recalc[name] = False

        # Invert sub-framework indices
        subframeworks = [
            getattr(self, s) for s in getattr(self, _subframeworks, set())
        ]  # have to get it again, because it's been updated

        for s in subframeworks:
            if ":" + name in getattr(s, hidden_loc(s, "recalc_prop_par")):
                for par in getattr(s, hidden_loc(s, "recalc_prop_par"))[":" + name]:
                    getattr(s, hidden_loc(s, "recalc_par_prop"))[par].add(":" + name)

            if ":" + name in getattr(s, hidden_loc(s, "active_q")):
                getattr(s, hidden_loc(s, "active_q")).remove(":" + name)

        return value

    update_wrapper(_get_property, f)

    def _del_property(self):
        # Locations of indexes
        recalc = hidden_loc(self, "recalc")
        recalc_prpa = hidden_loc(self, "recalc_prop_par")

        # Delete the property AND its recalc dicts
        try:
            prop = hidden_loc(self, name)
            delattr(self, prop)
        except AttributeError:
            pass

        try:
            del getattr(self, recalc)[name]
        except KeyError:
            warnings.warn(f"{name} not found in recalc cache.")

        try:
            del getattr(self, recalc_prpa)[name]
        except KeyError:
            warnings.warn(f"{name} not found in recalc_prop_par cache")

    return property(_get_property, None, _del_property)


def obj_eq(ob1, ob2):
    """Test equality of objects that is numpy-aware."""
    try:
        return bool(ob1 == ob2)
    except ValueError:
        # Could be a numpy array.
        return (ob1 == ob2).all()


def parameter(kind):
    """
    A decorator which indicates a parameter of a calculation.

    This decorator is intended for use with the complementary `cached_quantity` decorator.
    It provides the mechanisms by which the quantities are re-calculated intelligently.
    Parameters should be set by the `__init__` call in any class, so that they are set before any
    dependent quantity is accessed.

    Parameters
    ----------
    kind : str
        Either "param", "option", "model", "switch" or "res". Changes the behaviour of the parameter.
        "param", "option", "model" and "res" all behave the same currently, while when a "switch" is modified,
        all dependent quantities have their dependencies re-indexed.

    Examples
    --------
    >>>   class CachedClass(object):
    >>>      @parameter
    >>>      def a_param(self,val):
    >>>         return val
    >>>
    >>>      @cached_quantity
    >>>      def a_quantity(self):
    >>>         return 2*self.a_param
    >>>
    >>>      @cached_quantity
    >>>      def a_child_quantity(self):
    >>>         return self.a_quantity**3

    This code will calculate ``a_child_quantity`` on the first call, but return the cached
    value on all subsequent calls. If `a_param` is modified, the
    calculation of either `a_quantity` and `a_child_quantity` will be re-performed when requested.
    """

    def param(f):
        name = f.__name__

        def _set_property(self, val):

            prop = hidden_loc(self, name)

            # The following does any complex setting that is written into the code
            val = f(self, val)

            # Here put any custom code that should be run, dependent on the type of parameter
            if (
                name.endswith("_params")
                and not isinstance(val, dict)
                and val is not None
            ):
                raise ValueError(f"{name} must be a dictionary")

            # Locations of indexes
            recalc = hidden_loc(self, "recalc")
            activeq = hidden_loc(self, "active_q")
            recalc_papr = hidden_loc(self, "recalc_par_prop")
            recalc_prpa = hidden_loc(self, "recalc_prop_par")

            try:
                # If the property has already been set, we can grab its old value
                old_val = _get_property(self)
                doset = False
            except AttributeError:
                # Otherwise, it has no old value.
                old_val = None
                doset = True

                # It's not been set before, so add it to our list of parameters
                try:
                    # Only works if something has been set before
                    getattr(self, recalc_papr)[name] = set()

                except AttributeError:
                    # Given that *at least one* parameter must be set before properties
                    # are calculated, we can define the original empty indexes here.
                    setattr(self, recalc, {})
                    setattr(self, activeq, set())
                    setattr(self, recalc_prpa, {})
                    setattr(self, recalc_papr, {name: set()})

            # If either the new value is different from the old, or we never set it before
            if not obj_eq(val, old_val) or doset:
                # Then if its a dict, we update it
                if isinstance(val, dict) and hasattr(self, prop) and val:
                    getattr(self, prop).update(val)
                # Otherwise, just overwrite it. Note if dict is passed empty, it clears
                # the whole dict.
                else:
                    setattr(self, prop, val)

                # Make sure children are updated
                if kind != "switch" or doset:
                    # Normal parameters just update dependencies
                    for pr in getattr(self, recalc_papr)[name]:
                        getattr(self, recalc)[pr] = True
                else:
                    # Switches mean that dependencies could depend on new parameters,
                    # so need to re-index
                    for pr in getattr(self, recalc_papr)[name]:
                        delattr(self, pr)

                if not doset and self._validate:
                    if self._validate_every_param_set:
                        self.validate()
                    else:
                        warnings.warn(
                            f"You are setting {name} directly. This is unstable, as less "
                            f"validation is performed. You can turn on extra validation "
                            f"for directly set parameters by setting framework._validate_every_param_set=True."
                            f"However, this can be brittle, since intermediate states may not be valid.",
                            category=DeprecationWarning,
                        )

        update_wrapper(_set_property, f)

        def _get_property(self):
            prop = hidden_loc(self, name)
            activeq = getattr(self, hidden_loc(self, "active_q"))
            prpa = getattr(self, hidden_loc(self, "recalc_prop_par"))

            # Add parameter to any active quantity
            for pr in activeq:
                prpa[pr].add(name)

            return getattr(self, prop)

        # Here we set the documentation
        doc = (f.__doc__ or "").strip()
        if doc.startswith("\n"):
            doc = doc[1:]

        return property(_get_property, _set_property, None, "**Parameter**: " + doc)

    return param


def subframework(f):
    """
    A quantity that is essentially a sub-framework
    Parameters
    ----------
    f
    Returns
    -------
    """
    name = f.__name__

    def _get_property(self):
        # Location of the property to be accessed
        prop = hidden_loc(self, name)

        # Locations of indexes [they are set up in the parameter decorator]
        activeq = getattr(self, hidden_loc(self, "active_q"))

        def copy_index(fmwork):
            # Every time it's gotten, we update the overall papr and prpa dicts with the
            # relavant sub-items
            fmwork_activeq = getattr(fmwork, hidden_loc(fmwork, "active_q"))
            fmwork_prpa = getattr(fmwork, hidden_loc(fmwork, "recalc_prop_par"))
            fmwork_recalc = getattr(fmwork, hidden_loc(fmwork, "recalc"))

            # Copy all open properties of top-level, into framework prpa.
            # This only runs the first time a property is accessed, since after that,
            # prpa is not populated.
            for k in activeq:
                fmwork_activeq.add(":" + k)
                if ":" + k not in fmwork_prpa:
                    fmwork_prpa[":" + k] = set()
                fmwork_recalc[":" + k] = False

        try:
            fmwork = getattr(self, prop)
            copy_index(fmwork)
            return fmwork

        except AttributeError:
            value = f(self)
            setattr(self, prop, value)
            cls = getattr(self, prop)

            copy_index(cls)

            try:
                getattr(self, hidden_loc(self, "subframeworks")).add(name)
            except AttributeError:
                setattr(self, hidden_loc(self, "subframeworks"), {name})

            return value

    update_wrapper(_get_property, f)

    def _del_property(self):
        try:
            prop = hidden_loc(self, name)
            delattr(self, prop)
        except AttributeError:
            pass

    return property(_get_property, None, _del_property)
