"""
This module defines two decorators, based on the code from 
http://forrst.com/posts/Yet_another_caching_property_decorator_for_Pytho-PBy

They are both designed to cache class properties, but have the added
functionality of being automatically updated when a parent property is 
updated.
"""
from functools import update_wrapper
from copy import copy

class Cache(object):
    __recalc_proto = {}
    __recalc = {}
    __params = {}

def cached_property(*parents):
    """
    A simple cached_property implementation which contains the functionality
    to delete dependent properties when updated.
    
    Usage::
       @cached_property("a_child_method")
       def amethod(self):
          ...calculations...
          return final_value
          
       @cached_property()
       def a_child_method(self): #method dependent on amethod
          final_value = 3*self.amethod
          return final_value
          
    This code will calculate ``amethod`` on the first call, but return the cached
    value on all subsequent calls. If it is deleted at any stage (perhaps in 
    order to be updated), then ``a_child_method`` will also be deleted, and 
    upon next call will be re-calculated.
    """
    recalc = "_Cache__recalc"
    recalc_proto = "_Cache__recalc_proto"
    params = "_Cache__params"

    def cache(f):
        name = f.__name__

        prop_ext = '__%s' % name

        def _get_property(self):
            prop = ("_" + self.__class__.__name__ + prop_ext).replace("___", "__")

            if name in getattr(self, recalc_proto) and name not in getattr(self, recalc):
                print "INIT RECALC"
                # This should only happen on second-call of a property
                extra = set(getattr(self, recalc_proto)[name])
                final = set()
                while extra:
                    ex_tmp = set()
                    for e in extra:
                        if e not in getattr(self, recalc_proto):
                            # e is a parameter, add it to top-level
                            final.add(e)
                        elif e in getattr(self, recalc):
                            # e's elements are all parameters, add them to final
                            final |= set(getattr(self, recalc_proto)[e])
                        else:
                            # e's elements are arbitrary, keep going.
                            ex_tmp |= set(getattr(self, recalc_proto)[e])
                    ex_tmp -= extra
                    extra = copy(ex_tmp)

                # Set the recalc parameters
                getattr(self, recalc)[name] = {k:getattr(self, params)[k] for k in final}

            if name not in getattr(self, recalc_proto):
                # Initialisation of property
                getattr(self, recalc_proto)[name] = parents  # {p:False for p in parents}
                print "INIT RECALC_PROTO"
            calc = False
            if name not in getattr(self, recalc):
                calc = True
            elif any(getattr(self, recalc)[name].values()):
                calc = True

            if calc:
                print "RECALC!"
                # Recalculate the value
                value = f(self)
                setattr(self, prop, value)

                # Reset recalc so that this is not recalculated
                if name in getattr(self, recalc):
                    for k in getattr(self, recalc)[name]:
                        getattr(self, recalc)[name][k] = False

#                 # Reset recalc so children of this ARE recalculated
#                 for k, v in getattr(self, recalc).iteritems():
#                     if v == name:
#                         getattr(self, recalc)[k][v] = True

            else:
                print "GRAB RES"
                value = getattr(self, prop)

            return value

        update_wrapper(_get_property, f)

        def _del_property(self):
            # Delete the property AND its recalc dict
            try:
                prop = ("_" + self.__class__.__name__ + prop_ext).replace("___", "__")
                delattr(self, prop)
                del getattr(self, recalc)[name]
            except AttributeError:
                pass

        return property(_get_property, None, _del_property)
    return cache


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
    prop_ext = '__%s' % name
    recalc = "_Cache__recalc"
#     recalc_proto = "_Cache__recalc_proto"
    def _set_property(self, val):
        prop = ("_" + self.__class__.__name__ + prop_ext).replace("___", "__")
        val = f(self, val)
        try:
            old_val = getattr(self, prop)
            doset = False
        except AttributeError:
            old_val = None
            doset = True

        if val != old_val or doset:
            if isinstance(val, dict) and hasattr(self, prop):
                getattr(self, prop).update(val)
            else:
                setattr(self, prop, val)

            # Reset recalc so children of this are recalculated
            if getattr(self, recalc):
                print "RESETTING RECALC"
                for k, v in getattr(self, recalc).iteritems():
                    for k1 in v:
                        if k1 == name:
                            getattr(self, recalc)[k][k1] = True
            elif name not in getattr(self, "_Cache__params"):
                getattr(self, "_Cache__params")[name] = False
                print "INIT PARAMS"
            else:
                getattr(self, "_Cache__params")[name] = True
                print "MODIFY PARAMS"
#                 for k, v in getattr(self, recalc_proto).iteritems():
#                     if v == name:
#                         getattr(self, recalc_proto)[k][v] = True
    update_wrapper(_set_property, f)

    def _get_property(self):
        prop = ("_" + self.__class__.__name__ + prop_ext).replace("___", "__")
        return getattr(self, prop)

#     def _del_property(self):
#         prop = ("_" + self.__class__.__name__ + prop_ext).replace("___", "__")
#         print "deleting " + prop
#         delattr(self, prop)

    return property(_get_property, _set_property, None)

def simproperty(f):
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
    prop_ext = '__%s_sp' % name

#     def _set_property(self, val):
#         prop = ("_" + self.__class__.__name__ + prop_ext).replace("___", "__")
#         val = f(self, val)
#         setattr(self, prop, val)


    def _get_property(self):
        prop = ("_" + self.__class__.__name__ + prop_ext).replace("___", "__")
        try:
            return getattr(self, prop)
        except:
            val = f(self)
            setattr(self, prop, val)
            return val

    def _del_property(self):
        prop = ("_" + self.__class__.__name__ + prop_ext).replace("___", "__")
        delattr(self, prop)

    update_wrapper(_get_property, f)

    return property(_get_property, None, _del_property)


def simparameter(f):
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
    prop_ext = '__%s' % name

    def _set_property(self, val):
        prop = ("_" + self.__class__.__name__ + prop_ext).replace("___", "__")
        val = f(self, val)
        setattr(self, prop, val)

    update_wrapper(_set_property, f)

    def _get_property(self):
        prop = ("_" + self.__class__.__name__ + prop_ext).replace("___", "__")
        return getattr(self, prop)

    return property(_get_property, _set_property, None)

# class A(object):
#
#     def __init__(self, a):
#         self.a = a
#
#     @set_property("amethod")
#     def a(self, val):
#         return val
#
#     @cached_property("child_method")
#     def amethod(self):
#         return 3 * self.a
#
#     @cached_property()
#     def child_method(self):
#         return 7 * self.amethod
#
# if __name__ == "__main__":
#     ins = A(4)
#     print ins.amethod
#     print ins.child_method
#     print ins.child_method
#     del ins.amethod
#     print ins.child_method
#     print "SETTING a"
#     ins.a = 5
#     print ins.amethod
#     print ins.child_method
