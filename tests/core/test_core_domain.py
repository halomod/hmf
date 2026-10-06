"""Tests of hmf.core.domain: Domain.contains and every domain policy."""

import functools
import gc
import warnings

import astropy.units as u
import numpy as np
import pytest

from hmf.core.domain import (
    DOMAIN_POLICIES,
    Domain,
    DomainError,
    HMFExtrapolationWarning,
    Interval,
    apply_domain_policy,
    warn_once,
)
from hmf.core.units import Msun_h, UnitBoundaryError
from hmf.exceptions import HMFExtrapolationWarning as V3Warning


@pytest.fixture
def domain():
    return Domain({"z": (0, 2), "sigma": (0.25, None), "m": [1e10, 1e15] * Msun_h})


def test_reuses_v3_warning():
    assert HMFExtrapolationWarning is V3Warning


def test_policies():
    assert DOMAIN_POLICIES == ("ignore", "warn", "mask", "raise")


def test_contains_scalars(domain):
    assert domain.contains(z=1.0) is True
    assert domain.contains(z=2.5) is False
    assert domain.contains(z=1.0, sigma=0.3, m=1e12 * Msun_h) is True
    assert domain.contains(z=1.0, sigma=0.2) is False
    # No variables given: nothing to check.
    assert domain.contains() is True


def test_contains_bounds_inclusive(domain):
    assert domain.contains(z=0.0) is True
    assert domain.contains(z=2.0) is True
    assert domain.contains(z=np.nextafter(2.0, 3.0)) is False


def test_contains_unbounded_side(domain):
    assert domain.contains(sigma=1e30) is True
    assert domain["sigma"].upper == np.inf


def test_contains_nan_is_outside(domain):
    assert domain.contains(z=np.nan) is False


def test_contains_broadcasts(domain):
    z = np.array([0.5, 3.0])[:, None]
    sigma = np.array([0.1, 0.5, 1.0])[None, :]
    inside = domain.contains(z=z, sigma=sigma)
    assert inside.shape == (2, 3)
    np.testing.assert_array_equal(inside, [[False, True, True], [False, False, False]])


def test_contains_units(domain):
    # 1e9 Msun is 7e8 Msun/h for h=0.7, but the domain has no H0 to convert with: a
    # physical mass is a unit error, not a silent comparison.
    assert domain.contains(m=1e14 * Msun_h) is True
    assert domain.contains(m=1e16 * Msun_h) is False
    assert domain.contains(m=[1e9, 1e12] * Msun_h).tolist() == [False, True]
    with pytest.raises(u.UnitConversionError):
        domain.contains(m=1e12 * u.Msun)
    with pytest.raises(UnitBoundaryError, match=r"'m' is dimensional.*Msun_h` \(h-units: "):
        domain.contains(m=1e12)


def test_contains_dimensionless_quantity(domain):
    assert domain.contains(z=1.0 * u.dimensionless_unscaled) is True


def test_contains_unknown_variable(domain):
    with pytest.raises(ValueError, match=r"no variable\(s\) \['zz'\]"):
        domain.contains(zz=1.0)


def test_domain_is_frozen_hashable_and_ordered():
    a = Domain({"z": (0, 1), "sigma": (0.1, 2)}, source="Paper 2008")
    b = Domain({"sigma": (0.1, 2), "z": (0, 1)}, source="Paper 2008")
    assert a == b
    assert hash(a) == hash(b)
    assert a.variables == ("sigma", "z")
    with pytest.raises(KeyError):
        a["m"]


def test_interval_must_be_ordered():
    with pytest.raises(ValueError, match="lower <= upper"):
        Interval(2, 1)
    with pytest.raises(ValueError, match="lower <= upper"):
        Interval(np.nan, 1)


def test_interval_bound_used_as_is():
    interval = Interval(0, 1)
    d = Domain({"z": interval})
    assert d["z"] is interval
    assert d.contains(z=0.5) is True


# ---------------------------------------------------------------------------------
# Open and closed bounds
# ---------------------------------------------------------------------------------
def test_interval_bounds_are_closed_by_default():
    interval = Interval(0, 1)
    assert not interval.lower_open
    assert not interval.upper_open
    assert interval.contains(0.0) is True
    assert interval.contains(1.0) is True


@pytest.mark.parametrize(
    ("lower_open", "upper_open", "expected"),
    [
        (False, False, [False, True, True, True, False]),
        (True, False, [False, False, True, True, False]),
        (False, True, [False, True, True, False, False]),
        (True, True, [False, False, True, False, False]),
    ],
)
def test_open_bounds_exclude_the_bound(lower_open, upper_open, expected):
    interval = Interval(0, 1, lower_open=lower_open, upper_open=upper_open)
    x = np.array([-1e-300, 0.0, 0.5, 1.0, np.nan])
    assert interval.contains(x).tolist() == expected
    # The smallest float above an open bound is inside.
    assert interval.contains(np.nextafter(0.0, 1.0)) is True


def test_open_bound_with_units():
    interval = Interval(0, None, Msun_h, lower_open=True)
    assert interval.contains(0.0 * Msun_h) is False
    assert interval.contains(1e-300 * Msun_h) is True


def test_open_flags_are_keyword_only_bools():
    with pytest.raises(TypeError):
        Interval(0, 1, None, True)
    with pytest.raises(TypeError):
        Interval(0, 1, lower_open=1)


def test_open_interval_can_not_be_empty():
    Interval(1, 1)  # a single point
    with pytest.raises(ValueError, match="empty"):
        Interval(1, 1, lower_open=True)
    with pytest.raises(ValueError, match="empty"):
        Interval(1, 1, upper_open=True)


@pytest.mark.parametrize(
    ("brackets", "lower_open", "upper_open"),
    [("[]", False, False), ("(]", True, False), ("[)", False, True), ("()", True, True)],
)
def test_tuple_shorthand_brackets(brackets, lower_open, upper_open):
    d = Domain({"z": (0, 2, brackets)})
    assert d["z"] == Interval(0, 2, lower_open=lower_open, upper_open=upper_open)
    # A pair is closed.
    assert Domain({"z": (0, 2)})["z"] == Interval(0, 2)


def test_tuple_shorthand_with_quantities():
    d = Domain({"k": (0 * Msun_h, None, "(]")})
    assert d["k"] == Interval(0, None, Msun_h, lower_open=True)


@pytest.mark.parametrize("brackets", ["[", "(}", "open", None])
def test_tuple_shorthand_rejects_other_brackets(brackets):
    with pytest.raises(ValueError, match="third item"):
        Domain({"z": (0, 2, brackets)})


# ---------------------------------------------------------------------------------
# Descriptions and checks
# ---------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("interval", "text"),
    [
        (Interval(0, None, lower_open=True), "x > 0"),
        (Interval(0, None), "x >= 0"),
        (Interval(None, 3, upper_open=True), "x < 3"),
        (Interval(None, 3), "x <= 3"),
        (Interval(75, 3e4, lower_open=True), "75 < x <= 30000"),
        (Interval(0, 2.5, upper_open=True), "0 <= x < 2.5"),
        (Interval(None, None), "any x"),
        (Interval(1e10, 1e15, Msun_h), "1e+10 solMass / littleh <= x <= 1e+15 solMass / littleh"),
    ],
)
def test_interval_describe(interval, text):
    assert interval.describe("x") == text


def test_domain_describe(domain):
    assert domain.describe() == (
        "1e+10 solMass / littleh <= m <= 1e+15 solMass / littleh, sigma >= 0.25, 0 <= z <= 2"
    )
    assert Domain({}).describe() == "unbounded"


def test_check_inside_returns_none(domain):
    assert domain.check({"z": [0.0, 2.0], "m": 1e12 * Msun_h}, where="Toy") is None
    assert domain.check({}, where="Toy") is None


def test_check_raises_a_domain_error_with_the_description():
    d = Domain({"sigma": (0, None, "(]"), "z": (0, 2)})
    with pytest.raises(DomainError) as info:
        d.check({"sigma": [1.0, 0.0, 2.0], "z": 1.0}, where="Toy's valid domain")
    message = str(info.value)
    assert message.startswith("Toy's valid domain: 1 of 3 value(s)")
    assert "(sigma > 0, 0 <= z <= 2)" in message
    assert message.endswith("out of range: sigma.")
    with pytest.raises(DomainError, match=r"2 of 2 value.*out of range: sigma, z"):
        d.check({"sigma": [0.0, 1.0], "z": [1.0, np.nan]}, where="Toy")


def test_check_unknown_variable_is_a_value_error(domain):
    with pytest.raises(ValueError, match="no variable") as info:
        domain.check({"zz": 1.0}, where="Toy")
    assert not isinstance(info.value, DomainError)


def test_quantity_pair_bounds():
    d = Domain({"m": (1e10 * Msun_h, None)})
    assert d["m"].unit == Msun_h
    assert d.contains(m=1e20 * Msun_h) is True
    with pytest.raises(ValueError, match="shape"):
        Domain({"m": [1, 2, 3] * Msun_h})


# ---------------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------------
RESULT = np.array([1.0, 2.0, 3.0])
INSIDE = np.array([True, False, True])


def test_ignore_returns_result_unchanged():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert apply_domain_policy(RESULT, INSIDE, "ignore") is RESULT


@pytest.mark.parametrize("policy", DOMAIN_POLICIES)
def test_all_inside_is_a_no_op(policy):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert apply_domain_policy(RESULT, np.ones(3, bool), policy) is RESULT


def test_warn():
    with pytest.warns(HMFExtrapolationWarning, match="1 of 3 value"):
        out = apply_domain_policy(RESULT, INSIDE, "warn", description="Toy's domain")
    assert out is RESULT


def test_mask_array():
    out = apply_domain_policy(RESULT, INSIDE, "mask")
    np.testing.assert_array_equal(out, [1.0, np.nan, 3.0])
    # The input is not modified.
    np.testing.assert_array_equal(RESULT, [1.0, 2.0, 3.0])


def test_mask_scalar():
    out = apply_domain_policy(2.0, False, "mask")
    assert isinstance(out, float)
    assert np.isnan(out)


def test_mask_quantity():
    out = apply_domain_policy(RESULT * Msun_h, INSIDE, "mask")
    assert out.unit is Msun_h
    np.testing.assert_array_equal(out.value, [1.0, np.nan, 3.0])


def test_raise():
    with pytest.raises(DomainError, match="outside Toy's domain"):
        apply_domain_policy(RESULT, INSIDE, "raise", description="Toy's domain")
    assert issubclass(DomainError, ValueError)


def test_unknown_policy():
    with pytest.raises(ValueError, match="Unknown domain policy"):
        apply_domain_policy(RESULT, INSIDE, "clip")


def test_policy_with_contains(domain):
    z = np.array([0.0, 1.0, 5.0])
    out = apply_domain_policy(z**2, domain.contains(z=z), "mask")
    np.testing.assert_array_equal(out, [0.0, 1.0, np.nan])


def test_warn_policy_warns_once_per_owner():
    """With an owner, "warn" warns once per owner and description (#390)."""

    class Stage:
        pass

    first, second = Stage(), Stage()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for owner in (first, first, second):
            apply_domain_policy(RESULT, INSIDE, "warn", description="Toy's domain", owner=owner)
        apply_domain_policy(RESULT, INSIDE, "warn", description="Other domain", owner=first)
        # Without an owner it warns on every call.
        apply_domain_policy(RESULT, INSIDE, "warn")
        apply_domain_policy(RESULT, INSIDE, "warn")
    assert [str(w.message).split(" are outside ")[1] for w in caught] == [
        "Toy's domain. They are extrapolations.",
        "Toy's domain. They are extrapolations.",
        "Other domain. They are extrapolations.",
        "the model. They are extrapolations.",
        "the model. They are extrapolations.",
    ]


# ---------------------------------------------------------------------------------
# warn_once
# ---------------------------------------------------------------------------------
class _Owner:
    """An object that compares equal to every other: warn_once goes by identity."""

    def __eq__(self, other):
        return isinstance(other, _Owner)

    def __hash__(self):
        return 0


def _record(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    return caught


def test_warn_once_per_owner_and_key():
    a, b = _Owner(), _Owner()

    def calls():
        assert warn_once(a, "k", "a k") is True
        assert warn_once(a, "k", "a k again") is False
        assert warn_once(a, "z", "a z") is True
        assert warn_once(b, "k", "b k") is True  # equal to a, but another object

    caught = _record(calls)
    assert [str(w.message) for w in caught] == ["a k", "a z", "b k"]
    assert all(w.category is HMFExtrapolationWarning for w in caught)


def test_warn_once_category_and_stacklevel():
    owner = _Owner()
    caught = _record(lambda: warn_once(owner, "k", "message", UserWarning, stacklevel=1))
    assert caught[0].category is UserWarning
    assert caught[0].filename == __file__


def test_warn_once_works_with_frozen_slotted_attrs():
    import attrs

    @attrs.frozen
    class Frozen:
        x: int = 0

    one, two = Frozen(), Frozen()
    assert one == two
    caught = _record(lambda: [warn_once(o, "k", "m") for o in (one, one, two)])
    assert len(caught) == 2


def test_warn_once_forgets_collected_owners():
    """The record goes with the object, so a new object at the same id warns again."""
    from hmf.core import domain as domain_module

    owner = _Owner()
    _record(functools.partial(warn_once, owner, "k", "m"))
    ident = id(owner)
    assert ident in domain_module._WARNED
    del owner
    gc.collect()
    assert ident not in domain_module._WARNED


def test_warn_once_without_weak_references_warns_every_time():
    caught = _record(lambda: [warn_once(1.5, "k", "m") for _ in range(2)])
    assert len(caught) == 2
