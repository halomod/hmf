"""Tests of hmf.core.power_source: the PowerSource protocol and TabulatedPower."""

import warnings

import astropy.units as u
import numpy as np
import pytest
from power_models import RHO_CRIT0, AnalyticPower, EisensteinHuNoWiggle

from hmf.core.domain import DomainError
from hmf.core.power_source import PowerSource, TabulatedPower
from hmf.core.units import H0_unit, UnitBoundaryError, h_Mpc, power_unit, rho_unit
from hmf.exceptions import HMFExtrapolationWarning


def _power_law_table(n=-1.5, amplitude=3.0, **kwargs):
    k = np.logspace(-3, 1, 50)
    return TabulatedPower(
        k=k * h_Mpc,
        pk=amplitude * k**n * power_unit,
        mean_density=0.3 * RHO_CRIT0 * rho_unit,
        **kwargs,
    )


def test_satisfies_the_protocol():
    assert isinstance(_power_law_table(), PowerSource)
    assert isinstance(AnalyticPower(EisensteinHuNoWiggle()), PowerSource)
    assert not isinstance(object(), PowerSource)


def test_power_law_is_reproduced_and_extrapolated_exactly():
    """Ln P is linear in ln k, so the spline and the power-law extrapolation are exact."""
    src = _power_law_table()
    k = np.logspace(-8, 5, 40)
    with pytest.warns(HMFExtrapolationWarning, match="extrapolated as a power law"):
        np.testing.assert_allclose(src._power(k), 3.0 * k**-1.5, rtol=1e-12)


def test_spline_is_accurate_for_a_smooth_spectrum():
    """Inside a reasonably dense table, the log-log spline matches the function to 1e-6."""
    eh = EisensteinHuNoWiggle()
    k = np.logspace(-4, 2, 300)
    src = TabulatedPower(k=k * h_Mpc, pk=eh(k) * power_unit, mean_density=1.0 * rho_unit)
    kk = np.logspace(-3.9, 1.9, 777)
    np.testing.assert_allclose(src._power(kk), eh(kk), rtol=1e-6)


def test_extension_raise_raises_a_domain_error():
    src = _power_law_table(extension="raise")
    src._power(np.array([1e-3, 1.0, 10.0]))
    with pytest.raises(DomainError, match=r"above the table.*extension='raise'"):
        src._power(np.array([20.0]))
    with pytest.raises(DomainError, match="below the table"):
        src.power(1e-4 * h_Mpc)


def test_extension_values():
    assert _power_law_table().extension == "auto"
    with pytest.raises(ValueError, match="extension"):
        _power_law_table(extension="clip")


def test_extrapolation_warns_once_per_instance_and_end():
    """Each end of the table warns once per instance; an equal instance warns again."""
    src = _power_law_table()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        src._power(np.array([1.0]))  # inside: no warning
        src.power(np.array([20.0, 30.0]) * h_Mpc)
        src.power(40.0 * h_Mpc)
        src._power(np.array([1e-5]))
        src._power(np.array([1e-6, 100.0]))
        _power_law_table().power(20.0 * h_Mpc)
    messages = [str(w.message) for w in caught]
    assert all(w.category is HMFExtrapolationWarning for w in caught)
    assert len(messages) == 3, messages
    assert "2 value(s) of k above the table [0.001, 10] h/Mpc" in messages[0]
    assert "below the table" in messages[1]
    assert "above the table" in messages[2]


def test_inside_the_table_does_not_warn():
    src = _power_law_table()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        src.power(np.logspace(-3, 1, 9) * h_Mpc)


@pytest.mark.parametrize("k", [0.0, -1.0, np.nan])
def test_non_positive_k_is_a_domain_error(k):
    with pytest.raises(DomainError, match="k must be > 0"):
        _power_law_table().power(k * h_Mpc)


def test_public_power_is_unit_checked():
    src = _power_law_table(H0=70 * H0_unit)
    p = src.power(np.array([0.1, 1.0]) * h_Mpc)
    assert p.unit is power_unit
    np.testing.assert_allclose(p.value, 3.0 * np.array([0.1, 1.0]) ** -1.5, rtol=1e-12)
    # 0.07 / Mpc is 0.1 h/Mpc with h = 0.7.
    np.testing.assert_allclose(src.power(0.07 / u.Mpc).value, 3.0 * 0.1**-1.5, rtol=1e-12)
    with pytest.raises(UnitBoundaryError):
        src.power(0.1)


def test_physical_units_are_converted_with_h0():
    h = 0.7
    k = np.logspace(-3, 1, 50)
    a = _power_law_table(n=1.0, amplitude=1.0)
    b = TabulatedPower(
        k=k * h / u.Mpc,
        pk=k * u.Mpc**3 / h**3,
        mean_density=0.3 * RHO_CRIT0 * h**2 * u.Msun / u.Mpc**3,
        H0=100 * h * H0_unit,
    )
    np.testing.assert_allclose(b._power(k), a._power(k), rtol=1e-12)
    assert b._rho_mean0 == pytest.approx(a._rho_mean0, rel=1e-12)
    with pytest.raises(u.UnitConversionError):
        TabulatedPower(k=k / u.Mpc, pk=k * u.Mpc**3, mean_density=1 * rho_unit)._table


@pytest.mark.parametrize(
    ("changes", "error", "match"),
    [
        ({"k": np.logspace(-3, 1, 50)}, UnitBoundaryError, "Quantity"),
        ({"mean_density": 1.0}, UnitBoundaryError, "Quantity"),
        ({"pk": -np.ones(50) * power_unit}, ValueError, "pk must be finite"),
        ({"k": np.logspace(1, -3, 50) * h_Mpc}, ValueError, "strictly increasing"),
        ({"k": np.logspace(-3, 1, 3) * h_Mpc, "pk": np.ones(3) * power_unit}, ValueError, ">= 4"),
        ({"mean_density": -1 * rho_unit}, ValueError, "mean_density"),
    ],
)
def test_validation(changes, error, match):
    with pytest.raises(error, match=match):
        _power_law_table().evolve(**changes)


def test_table_is_read_only_copy():
    k = np.logspace(-3, 1, 50)
    src = TabulatedPower(k=k * h_Mpc, pk=k * power_unit, mean_density=1 * rho_unit)
    k[0] = 99.0
    assert src.k[0].value == pytest.approx(1e-3)
    with pytest.raises(ValueError, match="read-only"):
        src.k.value[0] = 1.0


def test_equality_and_hash():
    a, b = _power_law_table(), _power_law_table()
    assert a == b
    assert hash(a) == hash(b)
    assert a != _power_law_table(n=-1.4)
    assert a != _power_law_table(H0=70 * H0_unit)


def test_h0_validated():
    with pytest.raises(UnitBoundaryError, match="H0 must be a Quantity"):
        _power_law_table(H0=70.0)
    with pytest.raises(u.UnitConversionError, match="H0 must be in"):
        _power_law_table(H0=70 * u.km)
