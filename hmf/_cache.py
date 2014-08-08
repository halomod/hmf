"""
This module defines two decorators, based on the code from 
http://forrst.com/posts/Yet_another_caching_property_decorator_for_Pytho-PBy

They are both designed to cache class properties, but have the added
functionality of being automatically updated when a parent property is 
updated.
"""
from functools import update_wrapper

class Cache(object):
    def __init__(self):
        self.__recalc_prop_par = {}
        self.__recalc_par_prop = {}
        self.__recalc = {}

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
    recalc_prpa = "_Cache__recalc_prop_par"
    recalc_papr = "_Cache__recalc_par_prop"

    def cache(f):
        name = f.__name__

        prop_ext = '__%s' % name

        def _get_property(self):
            prop = ("_" + self.__class__.__name__ + prop_ext).replace("___", "__")

            if getattr(self, recalc).get(name, True):
                value = f(self)
                setattr(self, prop, value)

            else:
                return  getattr(self, prop)


            if name not in getattr(self, recalc):
                final = set()
                for p in parents:
                    if p in getattr(self, recalc_prpa):
                        final |= set(getattr(self, recalc_prpa)[p])
                    else:
                        final.add(p)

                getattr(self, recalc_prpa)[name] = final

                for e in final:
                    if e in getattr(self, recalc_papr):
                        getattr(self, recalc_papr)[e].add(name)
                    else:
                        getattr(self, recalc_papr)[e] = set([name])

            getattr(self, recalc)[name] = False
            return value
        update_wrapper(_get_property, f)

        def _del_property(self):
            # Delete the property AND its recalc dicts
            try:
                prop = ("_" + self.__class__.__name__ + prop_ext).replace("___", "__")
                delattr(self, prop)
                del getattr(self, recalc)[name]
                del getattr(self, recalc_prpa)[name]
                for e in getattr(self, recalc_papr):
                    if name in getattr(self, recalc_papr)[e]:
                        getattr(self, recalc_papr)[e].remove(name)
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
    recalc_papr = "_Cache__recalc_par_prop"

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

            # Make sure children are updated
            for pr in getattr(self, recalc_papr).get(name, []):
                getattr(self, recalc)[pr] = True

    update_wrapper(_set_property, f)

    def _get_property(self):
        prop = ("_" + self.__class__.__name__ + prop_ext).replace("___", "__")
        return getattr(self, prop)


    return property(_get_property, _set_property, None)
#
# def simproperty(f):
#     """
#     A simple cached property which acts more like an input value.
#
#     This cached property is intended to be used on values that are passed in
#     ``__init__``, and can possibly be reset later. It provides the opportunity
#     for complex setters, and also the ability to update dependent properties
#     whenever the value is modified.
#
#     Usage::
#        @set_property("amethod")
#        def parameter(self,val):
#            if isinstance(int,val):
#               return val
#            else:
#               raise ValueError("parameter must be an integer")
#
#        @cached_property()
#        def amethod(self):
#           return 3*self.parameter
#
#     Note that the definition of the setter merely returns the value to be set,
#     it doesn't set it to any particular instance attribute. The decorator
#     automatically sets ``self.__parameter = val`` and defines the get method
#     accordingly
#     """
#     name = f.__name__
#     prop_ext = '__%s_sp' % name
#
# #     def _set_property(self, val):
# #         prop = ("_" + self.__class__.__name__ + prop_ext).replace("___", "__")
# #         val = f(self, val)
# #         setattr(self, prop, val)
#
#
#     def _get_property(self):
#         prop = ("_" + self.__class__.__name__ + prop_ext).replace("___", "__")
#         try:
#             return getattr(self, prop)
#         except:
#             val = f(self)
#             setattr(self, prop, val)
#             return val
#
#     def _del_property(self):
#         prop = ("_" + self.__class__.__name__ + prop_ext).replace("___", "__")
#         delattr(self, prop)
#
#     update_wrapper(_get_property, f)
#
#     return property(_get_property, None, _del_property)
#
#
# def simparameter(f):
#     """
#     A simple cached property which acts more like an input value.
#
#     This cached property is intended to be used on values that are passed in
#     ``__init__``, and can possibly be reset later. It provides the opportunity
#     for complex setters, and also the ability to update dependent properties
#     whenever the value is modified.
#
#     Usage::
#        @set_property("amethod")
#        def parameter(self,val):
#            if isinstance(int,val):
#               return val
#            else:
#               raise ValueError("parameter must be an integer")
#
#        @cached_property()
#        def amethod(self):
#           return 3*self.parameter
#
#     Note that the definition of the setter merely returns the value to be set,
#     it doesn't set it to any particular instance attribute. The decorator
#     automatically sets ``self.__parameter = val`` and defines the get method
#     accordingly
#     """
#     name = f.__name__
#     prop_ext = '__%s' % name
#
#     def _set_property(self, val):
#         prop = ("_" + self.__class__.__name__ + prop_ext).replace("___", "__")
#         val = f(self, val)
#         setattr(self, prop, val)
#
#     update_wrapper(_set_property, f)
#
#     def _get_property(self):
#         prop = ("_" + self.__class__.__name__ + prop_ext).replace("___", "__")
#         return getattr(self, prop)
#
#     return property(_get_property, _set_property, None)

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
