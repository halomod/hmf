"""Tests of hmf.core._validators, the shared attrs validators."""

import math

import attrs
import pytest

from hmf.core._validators import finite, less_than, positive


@attrs.frozen
class _Settings:
    x: float = attrs.field(default=1.0, validator=positive)
    y: float = attrs.field(default=0.0, validator=finite)
    p: float = attrs.field(default=0.3, validator=less_than(0.5))


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


def test_valid_values_pass():
    assert _Settings(x=1e-300, y=-1e300, p=0.4999) == _Settings(x=1e-300, y=-1e300, p=0.4999)
