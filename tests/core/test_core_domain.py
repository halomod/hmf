"""Tests of hmf.core.domain: Domain.contains and every domain policy."""

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
    with pytest.raises(UnitBoundaryError, match="'m' must be a Quantity"):
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
