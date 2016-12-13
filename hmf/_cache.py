"""
This module defines two decorators, based on the code from
http://forrst.com/posts/Yet_another_caching_property_decorator_for_Pytho-PBy

They are both designed to cache class properties, but have the added
functionality of being automatically updated when a parent property is
updated.
"""
from functools import update_wrapper
from copy import copy

def hidden_loc(obj,name):
    """
    Generate the location of a hidden attribute.
    Importantly deals with attributes beginning with an underscore.
    """
    return ("_" + obj.__class__.__name__ + "__"+ name).replace("___", "__")

def cached_quantity(f):
    """
    A robust property caching decorator.

    This decorator only works when used with the entire system here....

    Usage::
       class CachedClass(Cache):


           @cached_property("parent_parameter")
           def amethod(self):
              ...calculations...
              return final_value

           @cached_property("amethod")
           def a_child_method(self): #method dependent on amethod
              final_value = 3*self.amethod
              return final_value

    This code will calculate ``amethod`` on the first call, but return the cached
    value on all subsequent calls. If any parent parameter is modified, the
    calculation will be re-performed.
    """

    name = f.__name__

    def _get_property(self):
        # Location of the property to be accessed
        prop = hidden_loc(self,name)

        # Locations of indexes [they are set up in the parameter decorator]
        recalc = hidden_loc(self,"recalc")
        recalc_prpa = hidden_loc(self,"recalc_prop_par")
        recalc_prpa_static = hidden_loc(self, "recalc_prop_par_static")
        recalc_papr = hidden_loc(self,"recalc_par_prop")

        # First, if this property has already been indexed,
        # we must ensure all dependent parameters are copied into active indexes,
        # otherwise they will be lost to their parents.
        if name in getattr(self,recalc):
            for pr, v in getattr(self, recalc_prpa).iteritems():
                v.update(getattr(self, recalc_prpa_static)[name])


        # If this property already in recalc and doesn't need updating, just return.
        if not getattr(self, recalc).get(name, True):
            return getattr(self, prop)

        # Otherwise, if its in recalc, and needs updating, just update it
        elif name in getattr(self,recalc):
            value = f(self)
            setattr(self, prop, value)

            # Ensure it doesn't need to be recalculated again
            getattr(self, recalc)[name] = False

            return value

        # Otherwise, we need to create its index for caching.
        getattr(self,recalc_prpa)[name] = set()  # Empty set to which parameter names will be added

        # Go ahead and calculate the value -- each parameter accessed will add itself to the index.
        value = f(self)
        setattr(self,prop, value)

        # Invert the index
        for par in getattr(self,recalc_prpa)[name]:
            getattr(self,recalc_papr)[par].append(name)

        # Copy index to static dict, and remove the index (so that parameters don't keep on trying to add themselves)
        getattr(self,recalc_prpa_static)[name] = copy(getattr(self,recalc_prpa)[name])
        del getattr(self,recalc_prpa)[name]

        # Add entry to master recalc list
        getattr(self,recalc)[name] = False

        return value

    update_wrapper(_get_property, f)

    def _del_property(self):
        # Locations of indexes
        recalc = hidden_loc(self,"recalc")
        recalc_prpa = hidden_loc(self,"recalc_prop_par")
        recalc_papr = hidden_loc(self,"recalc_par_prop")

        # Delete the property AND its recalc dicts
        try:
            prop = hidden_loc(self,name)
            delattr(self, prop)
            del getattr(self, recalc)[name]
            del getattr(self, recalc_prpa)[name]
            for e in getattr(self, recalc_papr):
                if name in getattr(self, recalc_papr)[e]:
                    getattr(self, recalc_papr)[e].remove(name)
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
        if  (ob1 == ob2).all():
            return True
        else:
            return False


def parameter(f):
    """
    A simple cached property which acts more like an input value.

    This cached property is intended to be used on values that are passed in
    ``__init__``, and can possibly be reset later. It provides the opportunity
    for complex setters, and also the ability to update dependent properties
    whenever the value is modified.

    Usage::
       @set_property("amethod")
       def parameter(self,val):
           if isinstance(int,val):
              return val
           else:
              raise ValueError("parameter must be an integer")

       @cached_property()
       def amethod(self):
          return 3*self.parameter

    Note that the definition of the setter merely returns the value to be set,
    it doesn't set it to any particular instance attribute. The decorator
    automatically sets ``self.__parameter = val`` and defines the get method
    accordingly
    """

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
                getattr(self,recalc_papr)[name] = []

            except AttributeError:
                # Given that *at least one* parameter must be set before properties are calculated,
                # we can define the original empty indexes here.
                setattr(self, recalc, {})
                setattr(self, recalc_prpa, {})
                setattr(self, recalc_prpa_static, {})
                setattr(self, recalc_papr, {name: []})

        # If either the new value is different from the old, or we never set it before
        if not obj_eq(val, old_val) or doset:
            # Then if its a dict, we update it
            if isinstance(val, dict) and hasattr(self, prop) and val:
                getattr(self, prop).update(val)
            # Otherwise, just overwrite it. Note if dict is passed empty, it clears the whole dict.
            else:
                setattr(self, prop, val)

            # Make sure children are updated
            for pr in getattr(self, recalc_papr).get(name):
                getattr(self, recalc)[pr] = True

    update_wrapper(_set_property, f)

    def _get_property(self):
        prop = hidden_loc(self,name)
        recalc_prpa = hidden_loc(self,"recalc_prop_par")

        # Add parameter to any index that hasn't been finalised
        for pr,v in getattr(self,recalc_prpa).iteritems():
            v.add(name)

        return getattr(self, prop)

    # Here we set the documentation
    doc = (f.__doc__ or "").strip()
    if doc.startswith("\n"):
        doc = doc[1:]

    return  property(_get_property, _set_property, None,"**Parameter**: "+doc)#