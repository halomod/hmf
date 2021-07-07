from inspect import getmembers, ismethod


def inherit_docstrings(cls):
    """Make docstrings inheritable for a class."""
    for name, func in getmembers(cls, ismethod):
        if func.__doc__:
            continue
        if name.startswith("__"):
            continue
        for parent in cls.__mro__[1:]:
            if hasattr(parent, name):
                func.__func__.__doc__ = getattr(parent, name).__doc__
    return cls
