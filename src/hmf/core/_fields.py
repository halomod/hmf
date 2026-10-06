"""Helpers for the ``attrs`` fields of models and stages.

Fields carry their documentation in their metadata (see :func:`field`), so that
:class:`~hmf.core.stage.Stage`, :class:`~hmf.core.model.Model` and the other
:class:`Documented` classes can list their fields (:func:`fields_info`) and generate
the "Parameters" section of their docstrings (:func:`parameters_section`) from the
class alone, without creating an instance.
"""

from __future__ import annotations

import contextlib
from typing import Any

import attrs

#: The metadata key holding a field's documentation.
DOC_KEY = "hmf_doc"

#: The ``default`` of a :class:`FieldInfo` for a field without a default.
NO_DEFAULT: Any = attrs.NOTHING


def field(*, doc: str, **kwargs: Any) -> Any:
    """Define an ``attrs`` field with documentation.

    Parameters
    ----------
    doc
        One or more sentences describing the field. They become its entry in the
        class's generated "Parameters" docstring section and in ``fields_info()``.
    **kwargs
        Passed on to :func:`attrs.field` (``default``, ``validator``,
        ``converter``, ...).

    Returns
    -------
    Any
        The field definition, as :func:`attrs.field` returns it.
    """
    metadata = dict(kwargs.pop("metadata", None) or {})
    metadata[DOC_KEY] = doc
    return attrs.field(metadata=metadata, **kwargs)


@attrs.frozen
class FieldInfo:
    """A description of one field of a model or stage class.

    Parameters
    ----------
    name
        The field's name, i.e. the keyword argument of the constructor.
    type
        The field's annotated type (a string if it could not be resolved).
    default
        Its default value; a :class:`attrs.Factory` for a computed default, and
        :data:`NO_DEFAULT` if it is required.
    doc
        Its documentation (empty if it has none).
    """

    name: str
    type: Any
    default: Any
    doc: str

    @property
    def required(self) -> bool:
        """Whether the field has no default and so must be given."""
        return self.default is NO_DEFAULT


def _resolved_fields(cls: type) -> tuple[attrs.Attribute[Any], ...]:
    """The fields of an attrs class, with string annotations resolved if possible."""
    # A forward reference that can't be resolved stays a string.
    with contextlib.suppress(NameError):
        attrs.resolve_types(cls)
    return tuple(attrs.fields(cls))


def fields_info(cls: type) -> tuple[FieldInfo, ...]:
    """Describe the fields of an ``attrs`` class, without instantiating it.

    Parameters
    ----------
    cls
        An ``attrs`` class.

    Returns
    -------
    tuple of FieldInfo
        One entry per field, in definition order.
    """
    return tuple(
        FieldInfo(
            name=a.alias or a.name,
            type=a.type,
            default=a.default,
            doc=a.metadata.get(DOC_KEY, ""),
        )
        for a in _resolved_fields(cls)
        if a.init
    )


def _type_name(tp: Any) -> str:
    """A short, readable name for a type annotation."""
    if isinstance(tp, str):
        return tp
    if isinstance(tp, type):
        return tp.__qualname__
    return str(tp).replace("typing.", "")


def parameters_section(cls: type) -> str:
    """Return a numpydoc "Parameters" section generated from a class's fields.

    Parameters
    ----------
    cls
        An ``attrs`` class whose fields were defined with :func:`field`.

    Returns
    -------
    str
        The section, or an empty string if the class has no fields.
    """
    infos = fields_info(cls)
    if not infos:
        return ""
    lines = ["Parameters", "----------"]
    for info in infos:
        kind = _type_name(info.type)
        if not info.required:
            default = info.default
            shown = "computed" if isinstance(default, attrs.Factory) else repr(default)  # type: ignore[arg-type]
            kind += f", default {shown}"
        lines.append(f"{info.name} : {kind}")
        lines.extend(f"    {line}" if line else "" for line in info.doc.splitlines() or [""])
    return "\n".join(lines)


def add_parameters_section(cls: type) -> None:
    """Append a generated "Parameters" section to ``cls.__doc__``.

    Nothing is added if the docstring already has a "Parameters" section (it was
    written by hand), or if the class has no fields.

    Parameters
    ----------
    cls
        An ``attrs`` class.
    """
    doc = cls.__doc__ or ""
    if "Parameters\n" in doc:
        return
    section = parameters_section(cls)
    if not section:
        return
    # Match the docstring's indentation, so that inspect.cleandoc still works on it.
    lines = doc.expandtabs().splitlines()
    indents = [len(line) - len(line.lstrip()) for line in lines[1:] if line.strip()]
    indent = " " * min(indents, default=0)
    body = "\n".join(indent + line if line else "" for line in section.splitlines())
    cls.__doc__ = f"{doc.rstrip()}\n\n{body}\n"


class Documented:
    """A mixin for ``attrs`` classes whose fields are defined with :func:`field`.

    Each ``attrs`` subclass gets the "Parameters" section of its docstring generated
    from its fields (:func:`add_parameters_section`), and the :meth:`fields_info`
    class method. A subclass that defines its own ``__attrs_init_subclass__`` must
    call ``super().__attrs_init_subclass__()``.
    """

    __slots__ = ()

    @classmethod
    def __attrs_init_subclass__(cls) -> None:
        """Generate the subclass's docstring "Parameters" section from its fields."""
        add_parameters_section(cls)

    @classmethod
    def fields_info(cls) -> tuple[FieldInfo, ...]:
        """Describe this class's fields, without creating an instance.

        Returns
        -------
        tuple of FieldInfo
            The name, type, default and documentation of each field.
        """
        return fields_info(cls)
