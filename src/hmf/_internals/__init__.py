"""Subpackage with internal definitions and utilities.

These define the structure of the entire library.
"""
from ._cache import cached_quantity, parameter
from ._framework import (
    Component,
    Framework,
    get_base_component,
    get_base_components,
    get_mdl,
    pluggable,
)
from ._utils import inherit_docstrings
