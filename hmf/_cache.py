"""
This module defines two decorators, based on the code from
http://forrst.com/posts/Yet_another_caching_property_decorator_for_Pytho-PBy

They are both designed to cache class properties, but have the added
functionality of being automatically updated when a parent property is
updated.
"""
from functools import update_wrapper
from copy import copy


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

    name = f.__name__

    def _get_property(self):
        # Location of the property to be accessed
        prop = hidden_loc(self, name)

        # Locations of indexes [they are set up in the parameter decorator]
        recalc = hidden_loc(self, "recalc")
        recalc_prpa = hidden_loc(self, "recalc_prop_par")
        recalc_prpa_static = hidden_loc(self, "recalc_prop_par_static")
        recalc_papr = hidden_loc(self, "recalc_par_prop")

        # First, if this property has already been indexed,
        # we must ensure all dependent parameters are copied into active indexes,
        # otherwise they will be lost to their parents.
        if name in getattr(self, recalc):
            for pr, v in getattr(self, recalc_prpa).items():
                v.update(getattr(self, recalc_prpa_static)[name])

        # If this property already in recalc and doesn't need updating, just return.
        if not getattr(self, recalc).get(name, True):
            return getattr(self, prop)

        # Otherwise, if its in recalc, and needs updating, just update it
        elif name in getattr(self, recalc):
            value = f(self)
            setattr(self, prop, value)

            # Ensure it doesn't need to be recalculated again
            getattr(self, recalc)[name] = False

            return value

        # Otherwise, we need to create its index for caching.
        supered = name in getattr(self, recalc_prpa) # if name is already there, can only be because the method has been supered.
        if not supered:
            getattr(self, recalc_prpa)[name] = set()  # Empty set to which parameter names will be added

        # Go ahead and calculate the value -- each parameter accessed will add itself to the index.
        value = f(self)
        setattr(self, prop, value)

        # Invert the index
        for par in getattr(self, recalc_prpa)[name]:
            getattr(self, recalc_papr)[par].add(name)


        # Copy index to static dict, and remove the index (so that parameters don't keep on trying to add themselves)
        if not supered: # If super, don't want to remove the name just yet.
            getattr(self, recalc_prpa_static)[name] = copy(getattr(self, recalc_prpa)[name])
            del getattr(self, recalc_prpa)[name]

        # Add entry to master recalc list
        getattr(self, recalc)[name] = False

        return value

    update_wrapper(_get_property, f)

    def _del_property(self):
        # Locations of indexes
        recalc = hidden_loc(self, "recalc")
        recalc_prpa = hidden_loc(self, "recalc_prop_par_static")
        recalc_papr = hidden_loc(self, "recalc_par_prop")

        # Delete the property AND its recalc dicts
        try:
            prop = hidden_loc(self, name)
            delattr(self, prop)
            del getattr(self, recalc)[name]
            del getattr(self, recalc_prpa)[name]
            # for e in getattr(self, recalc_papr):
            #     if name in getattr(self, recalc_papr)[e]:
            #         getattr(self, recalc_papr)[e].remove(name)
        except AttributeError:
            pass

    return property(_get_property, None, _del_property)


def obj_eq(ob1, ob2):
    try:
        if ob1 == ob2:
            return True
        else:
            return False
    except ValueError:
        if (ob1 == ob2).all():
            return True
        else:
            return False


def parameter(kind):
    """
    A decorator which indicates a parameter of a calculation (i.e. something that must be input by user).

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

            # Locations of indexes
            recalc = hidden_loc(self, "recalc")
            recalc_prpa = hidden_loc(self, "recalc_prop_par")
            recalc_papr = hidden_loc(self, "recalc_par_prop")
            recalc_prpa_static = hidden_loc(self, "recalc_prop_par_static")

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
                    # Given that *at least one* parameter must be set before properties are calculated,
                    # we can define the original empty indexes here.
                    setattr(self, recalc, {})
                    setattr(self, recalc_prpa, {})
                    setattr(self, recalc_prpa_static, {})
                    setattr(self, recalc_papr, {name: set()})

            # If either the new value is different from the old, or we never set it before
            if not obj_eq(val, old_val) or doset:
                # Then if its a dict, we update it
                if isinstance(val, dict) and hasattr(self, prop) and val:
                    getattr(self, prop).update(val)
                # Otherwise, just overwrite it. Note if dict is passed empty, it clears the whole dict.
                else:
                    setattr(self, prop, val)

                # Make sure children are updated
                if kind != "switch" or doset:  # Normal parameters just update dependencies
                    for pr in getattr(self, recalc_papr).get(name):
                        getattr(self, recalc)[pr] = True
                else:  # Switches mean that dependencies could depend on new parameters, so need to re-index
                    for pr in getattr(self, recalc_papr)[name]:
                        delattr(self, pr)

        update_wrapper(_set_property, f)

        def _get_property(self):
            prop = hidden_loc(self, name)
            recalc_prpa = hidden_loc(self, "recalc_prop_par")

            # Add parameter to any index that hasn't been finalised
            for pr, v in getattr(self, recalc_prpa).items():
                v.add(name)

            return getattr(self, prop)

        # Here we set the documentation
        doc = (f.__doc__ or "").strip()
        if doc.startswith("\n"):
            doc = doc[1:]

        return property(_get_property, _set_property, None, "**Parameter**: " + doc)

    return param
