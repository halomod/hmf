from hmf._internals._utils import inherit_docstrings


@inherit_docstrings
class _Base:
    @classmethod
    def foo(cls):
        """Base foo."""
        return "base"

    @classmethod
    def bar(cls):
        """Base bar."""
        return "base bar"

    @classmethod
    def __str__(cls):
        """Base str."""
        return "base str"


@inherit_docstrings
class _Child(_Base):
    @classmethod
    def foo(cls):
        return "child"

    @classmethod
    def bar(cls):
        """Child bar."""
        return "child bar"

    @classmethod
    def __str__(cls):
        return "child str"


def test_inherit_docstrings_sets_missing():
    assert _Child.foo.__doc__ == "Base foo."


def test_inherit_docstrings_preserves_existing():
    assert _Child.bar.__doc__ == "Child bar."


def test_inherit_docstrings_skips_dunder():
    assert _Child.__str__.__doc__ is None
