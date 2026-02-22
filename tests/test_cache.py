import numpy as np
import pytest

import hmf._internals._cache as cache
from hmf._internals._cache import cached_quantity, hidden_loc, obj_eq, parameter, subframework


class _CacheBase:
    def __init__(self):
        self._validate = False
        self.validate_calls = 0
        self.a = 1
        self.flag = False
        self.opts_params = {"a": 1}
        self._validate = True

    def validate(self):
        self.validate_calls += 1

    @parameter("param")
    def a(self, val):
        return val

    @parameter("switch")
    def flag(self, val):
        return val

    @parameter("param")
    def opts_params(self, val):
        return val

    @cached_quantity
    def q(self):
        return self.a + (1 if self.flag else 0)

    @cached_quantity
    def qchild(self):
        return self.q + 1


class _Child:
    def __init__(self):
        self._validate = False
        self.a = 2

    @parameter("param")
    def a(self, val):
        return val

    @cached_quantity
    def child_q(self):
        return self.a + 1


class _Parent:
    def __init__(self):
        self._validate = False
        self.p = 3
        self._counter = 0

    @parameter("param")
    def p(self, val):
        return val

    @subframework
    def sub(self):
        return _Child()

    @cached_quantity
    def q(self):
        self._counter += 1
        return self.p + self.sub.child_q


class _DictLike:
    def __init__(self, data):
        self._data = data

    def __eq__(self, other):
        raise ValueError("no bool")

    def __array__(self, *args, **kwargs):
        raise ValueError("no array")

    def keys(self):
        return self._data.keys()

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)


class _DocStr:
    def __init__(self, value):
        self.value = value

    def strip(self):
        return self.value

    def __bool__(self):
        return True


def _doc_param(self, val):
    return val


_doc_param.__doc__ = _DocStr("\nDoc with leading newline.")


class _DocParam:
    def __init__(self):
        self._validate = False
        self.doc = 1

    doc = parameter("param")(_doc_param)


def test_hidden_loc_underscores():
    obj = _CacheBase()

    assert hidden_loc(obj, "_x") == "__CacheBase__x"


def test_obj_eq_numpy():
    assert obj_eq(np.array([1, 2]), np.array([1, 2]))


def test_obj_eq_dictlike_keys(monkeypatch):
    def fake_array_equal(a, b):
        if isinstance(a, _DictLike) or isinstance(b, _DictLike):
            raise ValueError("boom")
        return np.array_equal(a, b)

    monkeypatch.setattr(cache, "array_equal", fake_array_equal)

    left = _DictLike({"a": np.array([1, 2])})
    right = _DictLike({"b": np.array([1, 2])})
    assert cache.obj_eq(left, right) is False

    left = _DictLike({"a": np.array([1, 2])})
    right = _DictLike({"a": np.array([1, 2])})
    assert cache.obj_eq(left, right) is True


def test_parameter_docstring_trim():
    assert _DocParam.doc.__doc__.startswith("**Parameter**: Doc with leading")


def test_parameter_requires_dict_for_params():
    obj = _CacheBase()

    with pytest.raises(ValueError, match="opts_params must be a dictionary"):
        obj.opts_params = 3


def test_parameter_dict_update():
    obj = _CacheBase()

    with pytest.warns(UserWarning, match="setting opts_params"):
        obj.opts_params = {"b": 2}

    assert obj.opts_params == {"a": 1, "b": 2}


def test_parameter_validate_warning():
    obj = _CacheBase()

    with pytest.warns(UserWarning, match="You are setting a directly"):
        obj.a = 2

    assert obj.validate_calls == 1


def test_parameter_switch_reindexes():
    obj = _CacheBase()
    assert obj.q == 1

    recalc = getattr(obj, hidden_loc(obj, "recalc"))
    assert "q" in recalc

    with pytest.warns(UserWarning, match="setting flag"):
        obj.flag = True

    recalc = getattr(obj, hidden_loc(obj, "recalc"))
    assert "q" not in recalc
    assert obj.q == 2


def test_parameter_recalc_updates():
    obj = _CacheBase()
    _ = obj.q

    recalc = getattr(obj, hidden_loc(obj, "recalc"))
    assert recalc["q"] is False

    obj._validate = False
    obj.a = 5

    assert recalc["q"] is True


def test_cached_quantity_del_warnings():
    obj = _CacheBase()
    _ = obj.q

    recalc = getattr(obj, hidden_loc(obj, "recalc"))
    recalc_prpa = getattr(obj, hidden_loc(obj, "recalc_prop_par"))
    del recalc["q"]
    del recalc_prpa["q"]

    with pytest.warns(UserWarning) as record:
        del obj.q

    assert len(record) == 2


def test_cached_quantity_del_without_value():
    obj = _CacheBase()

    with pytest.warns(UserWarning) as record:
        del obj.q

    assert len(record) == 2


def test_cached_quantity_keyerror_missing_prpa():
    obj = _CacheBase()
    recalc = getattr(obj, hidden_loc(obj, "recalc"))
    _ = obj.q
    activeq = getattr(obj, hidden_loc(obj, "active_q"))
    activeq.add("qchild")
    recalc["q"] = True

    with pytest.raises(KeyError, match="couldn't find qchild"):
        _ = obj.q


def test_subframework_copy_and_recalc():
    obj = _Parent()

    _ = obj.sub
    _ = obj.sub

    value = obj.q
    assert value == 6
    assert obj._counter == 1

    child = obj.sub
    child_recalc = getattr(child, hidden_loc(child, "recalc"))
    child_activeq = getattr(child, hidden_loc(child, "active_q"))
    assert ":q" not in child_activeq
    child_recalc[":q"] = True

    _ = obj.q
    assert obj._counter == 2

    child_papr = getattr(child, hidden_loc(child, "recalc_par_prop"))
    assert ":q" in child_papr["a"]


def test_subframework_delete_without_instance():
    obj = _Parent()

    del obj.sub
