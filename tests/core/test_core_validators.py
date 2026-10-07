"""Tests of hmf.core._validators, the shared attrs validators and array checkers."""

import math

import attrs
import numpy as np
import pytest

from hmf.core._validators import (
    check_finite_positive,
    check_in_range,
    check_increasing,
    check_table,
    finite,
    less_than,
    one_of,
    positive,
)


@attrs.frozen
class _Settings:
    x: float = attrs.field(default=1.0, validator=positive)
    y: float = attrs.field(default=0.0, validator=finite)
    p: float = attrs.field(default=0.3, validator=less_than(0.5))
    kind: str = attrs.field(default="a", validator=one_of(("a", "b")))


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, math.nan])
def test_positive(value):
    with pytest.raises(ValueError, match=r"_Settings\.x must be > 0"):
        _Settings(x=value)


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_finite(value):
    with pytest.raises(ValueError, match=r"_Settings\.y must be finite"):
        _Settings(y=value)


@pytest.mark.parametrize("value", [0.5, 1.0, math.nan])
def test_less_than(value):
    with pytest.raises(ValueError, match=r"_Settings\.p must be < 0\.5"):
        _Settings(p=value)


@pytest.mark.parametrize("value", ["c", None, "A"])
def test_one_of(value):
    with pytest.raises(
        ValueError, match=rf"_Settings\.kind must be one of 'a', 'b'; got {value!r}"
    ):
        _Settings(kind=value)
    assert _Settings(kind="b").kind == "b"


def test_valid_values_pass():
    assert _Settings(x=1e-300, y=-1e300, p=0.4999) == _Settings(x=1e-300, y=-1e300, p=0.4999)


# ---------------------------------------------------------------------------------
# Array checkers
# ---------------------------------------------------------------------------------


class _CustomError(ValueError):
    pass


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf, -np.inf, np.nan])
def test_check_finite_positive_rejects(bad):
    with pytest.raises(_CustomError, match=r"^Here: x must be finite and > 0, but 1 of 3 value"):
        check_finite_positive("x", [1.0, bad, 2.0], where="Here", error=_CustomError)


def test_check_finite_positive_returns_a_float_array():
    out = check_finite_positive("x", [1, 2], where="Here")
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, [1.0, 2.0])
    assert check_finite_positive("x", 1e-300, where="Here") == 1e-300
    with pytest.raises(ValueError, match="1 of 1 value"):
        check_finite_positive("x", 0.0, where="Here")


def test_check_in_range():
    np.testing.assert_array_equal(
        check_in_range("z", [0.0, 1.0], where="W", low=0.0, high=1.0), [0, 1]
    )
    with pytest.raises(ValueError, match="W: z must be finite and > 0, but 1 of 2 value"):
        check_in_range("z", [0.0, 1.0], where="W", low=0.0, low_open=True)
    with pytest.raises(_CustomError, match="z must be >= 0 and < 1"):
        check_in_range(
            "z",
            [0.5, 1.0],
            where="W",
            low=0.0,
            high=1.0,
            high_open=True,
            error=_CustomError,
        )
    with pytest.raises(ValueError, match="out of range or NaN"):
        check_in_range("z", [np.nan], where="W")
    # An unbounded end is open: infinite values are out of range.
    with pytest.raises(ValueError, match="k must be finite and > 0, but 1 of 2"):
        check_in_range("k", [1.0, np.inf], where="W", low=0.0, low_open=True)
    with pytest.raises(ValueError, match="z must be finite and <= 1, but 1 of 2"):
        check_in_range("z", [0.5, -np.inf], where="W", high=1.0)
    with pytest.raises(ValueError, match="z must be finite, but 2 of 3"):
        check_in_range("z", [0.0, np.inf, -np.inf], where="W")
    assert check_in_range("z", 1e300, where="W", low=0.0) == 1e300


@pytest.mark.parametrize("bad", [[1.0, 1.0, 2.0], [2.0, 1.0, 3.0], [1.0, np.nan, 3.0]])
def test_check_increasing(bad):
    with pytest.raises(_CustomError, match="x must be strictly increasing"):
        check_increasing("x", bad, where="W", error=_CustomError)
    np.testing.assert_array_equal(check_increasing("x", [1, 2, 3], where="W"), [1, 2, 3])


def test_check_table():
    k, p = check_table({"k": [1, 2, 3, 4], "p": [5, 6, 7, 8]}, where="W", min_size=4)
    assert k.dtype == p.dtype == np.float64
    for columns in (
        {"k": [1, 2, 3, 4], "p": [5, 6, 7]},  # different lengths
        {"k": [[1, 2], [3, 4]], "p": [[1, 2], [3, 4]]},  # not 1D
        {"k": [1, 2, 3], "p": [5, 6, 7]},  # too short
    ):
        with pytest.raises(_CustomError, match=r"W: k and p must be 1D, of the same length.*>= 4"):
            check_table(columns, where="W", min_size=4, error=_CustomError)
