"""Tests of hmf.core.stage: atomic evolve(), cached properties and fields_info()."""

import copy
import inspect
import pickle
from functools import cached_property

import attrs
import pytest

from hmf.core import field
from hmf.core.stage import Stage

N_INIT = 0
N_COMPUTE = 0


@attrs.frozen(kw_only=True)
class Squares(Stage):
    """A toy stage."""

    n: int = field(default=3, validator=attrs.validators.ge(0), doc="How many squares.")
    offset: float = field(default=0.0, doc="Added to each square.")

    def __attrs_post_init__(self):
        global N_INIT
        N_INIT += 1

    @cached_property
    def values(self) -> list[float]:
        global N_COMPUTE
        N_COMPUTE += 1
        return [i**2 + self.offset for i in range(self.n)]


@attrs.frozen(kw_only=True)
class Upstream(Stage):
    """A stage holding another stage."""

    squares: Squares = field(factory=Squares, doc="The stage this one depends on.")
    scale: float = field(default=1.0, doc="Multiplies the squares.")

    @cached_property
    def scaled(self) -> list[float]:
        return [self.scale * v for v in self.squares.values]


def test_cached_property_on_frozen_slotted_stage():
    global N_COMPUTE
    s = Squares()
    assert not hasattr(s, "__dict__")
    assert "values" in Squares.__slots__
    N_COMPUTE = 0
    assert s.values == [0, 1, 4]
    assert s.values is s.values
    assert N_COMPUTE == 1
    with pytest.raises(attrs.exceptions.FrozenInstanceError):
        s.n = 4


def test_attrs_version():
    # 23.2: cached_property on slotted classes. 24.1: __attrs_init_subclass__.
    major, minor = (int(x) for x in attrs.__version__.split(".")[:2])
    assert (major, minor) >= (24, 1)


def test_evolve_returns_new_stage():
    s = Squares()
    t = s.evolve(n=4)
    assert t is not s
    assert t.values == [0, 1, 4, 9]
    assert s.n == 3


def test_evolve_is_atomic():
    s = Squares(n=3, offset=1.0)
    values = s.values
    with pytest.raises(ValueError, match="'n' must be >= 0"):
        # offset is valid, n is not: nothing changes.
        s.evolve(offset=2.0, n=-1)
    assert (s.n, s.offset) == (3, 1.0)
    assert s.values is values


def test_evolve_unknown_field():
    with pytest.raises(TypeError, match=r"no field\(s\) \['ofset'\].*Did you mean 'offset'"):
        Squares().evolve(ofset=1.0)


def test_evolve_shares_unchanged_substages():
    up = Upstream()
    up.scaled
    new = up.evolve(scale=2.0)
    assert new.squares is up.squares
    global N_COMPUTE
    N_COMPUTE = 0
    assert new.scaled == [0, 2, 8]
    # The sub-stage's cache carried over.
    assert N_COMPUTE == 0


def test_fields_info_does_not_instantiate():
    global N_INIT
    N_INIT = 0
    info = Squares.fields_info()
    assert N_INIT == 0
    assert [f.name for f in info] == ["n", "offset"]
    assert info[0].type is int
    assert info[0].default == 3
    assert info[0].doc == "How many squares."
    assert not info[0].required
    up = Upstream.fields_info()
    assert isinstance(up[0].default, attrs.Factory)


def test_required_field():
    @attrs.frozen(kw_only=True)
    class Needs(Stage):
        x: float = field(doc="Required.")

    (info,) = Needs.fields_info()
    assert info.required
    with pytest.raises(TypeError):
        Needs()


def test_docstring_generated():
    doc = inspect.getdoc(Squares)
    assert "Parameters\n----------\nn : int, default 3\n    How many squares." in doc
    assert "squares : Squares, default computed" in inspect.getdoc(Upstream)


def test_copy_and_pickle():
    s = Squares(n=5)
    s.values
    for t in (copy.copy(s), copy.deepcopy(s), pickle.loads(pickle.dumps(s))):
        assert t == s
        assert t.values == s.values


def test_equality_ignores_cache():
    a, b = Squares(), Squares()
    a.values
    assert a == b
    assert hash(a) == hash(b)


def test_unresolvable_annotation_kept_as_string():
    # A forward reference that can't be resolved stays a string, in fields_info() and
    # in the generated docstring, rather than breaking class creation.
    @attrs.frozen(kw_only=True)
    class Forward(Stage):
        thing: "NotDefinedAnywhere" = field(default=None, doc="Forward reference.")  # noqa: F821

    (info,) = Forward.fields_info()
    assert info.type == "NotDefinedAnywhere"
    assert "thing : NotDefinedAnywhere, default None" in inspect.getdoc(Forward)
