"""Tests of hmf.core.accuracy: defaults, presets and validation."""

import math

import attrs
import pytest

from hmf.core.accuracy import KAccuracy, MassAccuracy


def test_mass_defaults():
    acc = MassAccuracy()
    assert acc.dlog10_m == 0.02
    assert acc.log10_m_min == 0.0
    assert acc.log10_m_max == 17.5
    assert acc.extension == "auto"
    assert acc.second_derivative is True


def test_k_defaults():
    acc = KAccuracy()
    assert acc.dln_k == 0.02
    assert acc.ln_k_min == pytest.approx(math.log(1e-8), rel=1e-15)
    assert acc.k_max_r_min == 20.0


def test_fast_presets():
    assert MassAccuracy.fast() == MassAccuracy(dlog10_m=0.05, second_derivative=False)
    assert KAccuracy.fast() == KAccuracy(dln_k=0.05)


def test_high_presets():
    assert MassAccuracy.high() == MassAccuracy(dlog10_m=0.01)
    assert KAccuracy.high().dln_k <= 0.005
    # high() is at least as fine as the default, which is at least as fine as fast().
    assert MassAccuracy.high().dlog10_m < MassAccuracy().dlog10_m < MassAccuracy.fast().dlog10_m
    assert KAccuracy.high().dln_k < KAccuracy().dln_k < KAccuracy.fast().dln_k


def test_preset_overrides():
    acc = MassAccuracy.fast(second_derivative=True, log10_m_max=16)
    assert acc.dlog10_m == 0.05
    assert acc.second_derivative is True
    assert acc.log10_m_max == 16.0


def test_frozen_and_hashable():
    acc = MassAccuracy()
    with pytest.raises(attrs.exceptions.FrozenInstanceError):
        acc.dlog10_m = 0.1
    assert hash(acc) == hash(MassAccuracy())


def test_keyword_only():
    with pytest.raises(TypeError):
        MassAccuracy(0.05)


@pytest.mark.parametrize("value", [0, -0.01, math.inf, math.nan])
def test_spacings_positive(value):
    with pytest.raises(ValueError, match="dlog10_m must be > 0"):
        MassAccuracy(dlog10_m=value)
    with pytest.raises(ValueError, match="dln_k must be > 0"):
        KAccuracy(dln_k=value)
    with pytest.raises(ValueError, match="k_max_r_min must be > 0"):
        KAccuracy(k_max_r_min=value)


@pytest.mark.parametrize(("lo", "hi"), [(10, 10), (12, 8)])
def test_mass_range_ordered(lo, hi):
    with pytest.raises(ValueError, match="must be less than"):
        MassAccuracy(log10_m_min=lo, log10_m_max=hi)


def test_bounds_finite():
    with pytest.raises(ValueError, match="finite"):
        MassAccuracy(log10_m_min=-math.inf)
    with pytest.raises(ValueError, match="finite"):
        KAccuracy(ln_k_min=math.nan)


def test_extension_values():
    assert MassAccuracy(extension="raise").extension == "raise"
    with pytest.raises(ValueError, match="extension"):
        MassAccuracy(extension="clip")


def test_second_derivative_is_bool():
    with pytest.raises(TypeError):
        MassAccuracy(second_derivative=1)


def test_fields_info_and_docstring():
    names = [f.name for f in MassAccuracy.fields_info()]
    assert names == ["dlog10_m", "log10_m_min", "log10_m_max", "extension", "second_derivative"]
    assert all(f.doc for f in KAccuracy.fields_info())
    assert "dln_k : float, default 0.02" in KAccuracy.__doc__
