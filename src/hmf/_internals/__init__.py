"""
A subpackage containing internal definitions and utilities to create the structure of the entire library.
"""
from ._cache import cached_quantity, parameter
from ._framework import Framework, Component
from ._utils import inherit_docstrings
