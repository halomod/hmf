"""
This module defines two decorators, based on the code from 
http://forrst.com/posts/Yet_another_caching_property_decorator_for_Pytho-PBy

They are both designed to cache class properties, but have the added
functionality of being automatically updated when a parent property is 
updated.
"""
from functools import update_wrapper
import numpy as np

def cached_property(*children):
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
    def cache(f):
        name = f.__name__

        prop_ext = '__%s' % name

        def _get_property(self):
            prop = ("_" + self.__class__.__name__ + prop_ext).replace("___", "__")
            try:
                value = getattr(self, prop)
            except AttributeError:
                value = f(self)
                setattr(self, prop, value)

            return value

        update_wrapper(_get_property, f)

        def _del_property(self):
            try:
                prop = ("_" + self.__class__.__name__ + prop_ext).replace("___", "__")
                delattr(self, prop)

            except AttributeError:
                pass

            for child in children:
                delattr(self, child)

        return property(_get_property, None, _del_property)
    return cache

def set_property(*children):
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
    def setcache(f):
        name = f.__name__
        prop_ext = '__%s' % name

        def _set_property(self, val):
            prop = ("_" + self.__class__.__name__ + prop_ext).replace("___", "__")
            val = f(self, val)
            try:
                old_val = getattr(self, prop)
                doset = False
            except AttributeError:
                old_val = None
                doset = True
            if np.any(val != old_val) or doset:
                setattr(self, prop, val)

                for child in children:
                    delattr(self, child)

        update_wrapper(_set_property, f)

        def _get_property(self):
            prop = ("_" + self.__class__.__name__ + prop_ext).replace("___", "__")
            return getattr(self, prop)

        return property(_get_property, _set_property, None)
    return setcache


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
