"""The units boundary, applied consistently across every model and public method.

* every registered model of every kind round-trips through ``attrs.evolve``, and its
  ``content_hash`` is stable (unit-carrying fields are stored as Quantities in the
  canonical unit, and compared by their value in it);
* a numeric parameter given as an int or as a float is the same model, with the
  same content hash;
* the scalar rule: a scalar input gives a scalar output (a numpy scalar if
  dimensionless, a 0-d Quantity if dimensional), and arrays keep their shape;
* errors name the class a method was called on, not the class that defines it.
"""

from __future__ import annotations

import inspect
import math

import astropy.units as u
import attrs
import numpy as np
import pytest

from hmf.core import (
    filters,
    fits,
    growth,
    growth_models,
    mass_variance,
    power_source,
    transfer,
    transfer_models,
)
from hmf.core._serialise import content_hash, quantity_key
from hmf.core.growth import Growth
from hmf.core.mass_variance import MassVariance
from hmf.core.model import Model
from hmf.core.power_source import TabulatedPower
from hmf.core.transfer import Transfer
from hmf.core.units import (
    H0_unit,
    Mpc_h,
    Msun_h,
    UnitBoundaryError,
    UnitContext,
    dndm_unit,
    h_Mpc,
    littleh,
    number_density_unit,
    power_unit,
    rho_unit,
    unit_boundary,
)

_K = np.geomspace(1e-4, 1e2, 50)

#: Constructor arguments of the registered models that have required fields.
_REQUIRED = {
    transfer_models.FromArray: {"k": _K * h_Mpc, "t": 1 / (1 + (_K / 0.1) ** 2)},
    transfer_models.FromFile: {"fname": "transfer.dat"},
    growth_models.FromArray: {"z": [0.0, 1.0, 2.0, 3.0], "d": [1.0, 0.6, 0.4, 0.3]},
    growth_models.FromFile: {"fname": "growth.dat"},
}


def _kinds():
    """Every model kind, found from the class hierarchy (so a new kind is not missed)."""
    found, stack = [], list(Model.__subclasses__())
    while stack:
        cls = stack.pop()
        if "_registry" in cls.__dict__:
            found.append(cls)
        stack.extend(cls.__subclasses__())
    return found


def _registered_models():
    return sorted(
        {m for kind in _kinds() for m in kind.get_models().values()},
        key=lambda m: m.qualified_name(),
    )


MODELS = _registered_models()


def _make(cls, **kwargs):
    return cls(**{**_REQUIRED.get(cls, {}), **kwargs})


def _id(cls):
    return cls.qualified_name()


def test_every_kind_has_models():
    kinds = {k.__name__ for k in _kinds()}
    assert {"FittingFunction", "Filter", "GrowthModel", "TransferModel"} <= kinds
    assert len(MODELS) > 40


@pytest.mark.parametrize("cls", MODELS, ids=_id)
def test_evolve_round_trips_every_field(cls):
    """``attrs.evolve`` passes each field back to the constructor: it must accept it."""
    model = _make(cls)
    assert attrs.evolve(model) == model
    for a in attrs.fields(cls):
        if not a.init:
            continue
        name = a.alias or a.name
        evolved = attrs.evolve(model, **{name: getattr(model, a.name)})
        assert evolved == model, name
        assert hash(evolved) == hash(model), name
        assert content_hash(evolved) == content_hash(model), name


@pytest.mark.parametrize("cls", MODELS, ids=_id)
def test_content_hash_is_stable(cls):
    """Equal constructions, and evolve, give the same content hash, in a fixed text."""
    first, second = _make(cls), _make(cls)
    assert first == second
    assert content_hash(first) == content_hash(second) == content_hash(attrs.evolve(first))
    # And it is a hash of the values, not of object identities.
    assert content_hash(first) == content_hash(_make(cls))


def test_quantity_fields_are_stored_in_the_canonical_unit():
    camb = transfer_models.CAMB(k_max=0.02 * littleh / u.kpc)
    assert camb.k_max.unit is h_Mpc
    np.testing.assert_allclose(camb.k_max.value, 20.0, rtol=1e-15)
    # Compared by the value in the canonical unit, not the unit it was given in.
    same = transfer_models.CAMB(k_max=camb.k_max.value * h_Mpc)
    assert camb == same
    assert hash(camb) == hash(same)
    assert content_hash(camb) == content_hash(same)
    assert content_hash(transfer_models.CAMB(k_max=20 * h_Mpc)) == content_hash(
        transfer_models.CAMB()
    )
    # Changing another field keeps it (the evolve that used to raise).
    ppf = attrs.evolve(camb, dark_energy_model="ppf")
    assert ppf.k_max == camb.k_max
    table = _make(transfer_models.FromArray)
    assert table.k.unit is h_Mpc
    assert not table.k.flags.writeable
    other = attrs.evolve(table, t=np.ones(_K.size))
    np.testing.assert_array_equal(other.k.value, table.k.value)


def _integral_candidates(value):
    return [round(value), math.floor(value), math.ceil(value), 1, 2, 0]


def _as_int_and_float(cls, name, value):
    """A pair of models with field ``name`` given as an int and as the equal float."""
    is_quantity = isinstance(value, u.Quantity)
    number = float(value.value) if is_quantity else float(value)
    for n in _integral_candidates(number):
        as_int = n * value.unit if is_quantity else n
        as_float = float(n) * value.unit if is_quantity else float(n)
        try:
            return _make(cls, **{name: as_int}), _make(cls, **{name: as_float})
        except (ValueError, TypeError):
            continue
    return None


@pytest.mark.parametrize("cls", MODELS, ids=_id)
def test_int_and_float_parameters_are_the_same_model(cls):
    """``M(x=1)`` and ``M(x=1.0)`` are equal, with the same content hash."""
    model = _make(cls)
    numeric = [
        a
        for a in attrs.fields(cls)
        if a.init
        and not isinstance(getattr(model, a.name), bool)
        and (
            isinstance(getattr(model, a.name), float)
            or (isinstance(getattr(model, a.name), u.Quantity) and getattr(model, a.name).ndim == 0)
        )
    ]
    if not numeric:
        pytest.skip(f"{cls.__name__} has no scalar numeric fields")
    checked = []
    for a in numeric:
        pair = _as_int_and_float(cls, a.alias or a.name, getattr(model, a.name))
        if pair is None:
            continue
        as_int, as_float = pair
        assert as_int == as_float, a.name
        assert content_hash(as_int) == content_hash(as_float), a.name
        value = getattr(as_int, a.name)
        assert isinstance(value.value if isinstance(value, u.Quantity) else value, float), a.name
        checked.append(a.name)
    # A field that accepts no integer at all (e.g. 0 < a_min < 1) can't be given one.
    assert checked or all(
        _as_int_and_float(cls, a.name, getattr(model, a.name)) is None for a in numeric
    )


def test_optional_fit_parameter_keeps_none():
    assert fits.SMT().A is None
    assert fits.SMT(A=1) == fits.SMT(A=1.0)
    assert type(fits.SMT(A=1).A) is float


def test_bool_fit_parameter_is_not_a_float():
    assert fits.Bhattacharya(normed=True).normed is True
    with pytest.raises(TypeError, match="normed"):
        fits.Bhattacharya(normed=1)


# ---------------------------------------------------------------------------------
# The scalar rule
# ---------------------------------------------------------------------------------

SHAPES = [(), (3,), (2, 3)]


def _shaped(values, shape):
    """``values`` (a 1-d list of at least 6) as an array of ``shape`` (a scalar for ())."""
    flat = np.asarray(values, dtype=float)
    return flat[0] if shape == () else flat[: math.prod(shape)].reshape(shape)


def _power_source():
    # Wider than MassVariance's default k grid, so that it is never extrapolated.
    k = np.geomspace(1e-9, 1e7, 640)
    return TabulatedPower(
        k=k * h_Mpc,
        pk=1e4 * k / (1 + (k / 0.02) ** 2.5) ** 1.6 * power_unit,
        mean_density=8.5e10 * rho_unit,
    )


_SOURCE = _power_source()
_MV = MassVariance(power=_SOURCE)
_TRANSFER = Transfer(model="EH")
_GROWTH = Growth()

_M = [1e11, 1e12, 1e13, 2e13, 5e13, 1e14]
_SIGMA = [2.0, 1.5, 1.2, 1.0, 0.8, 0.6]
_KS = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]

#: (id, bound method, function of the input shape giving (args, kwargs), the
#: output unit or None if dimensionless).
CASES = [
    ("transfer_function", _TRANSFER.transfer_function,
     lambda s: ((_shaped(_KS, s) * h_Mpc,), {}), None),
    ("unnormalised_power", _TRANSFER.unnormalised_power,
     lambda s: ((_shaped(_KS, s) * h_Mpc,), {}), None),
    ("growth_factor", _GROWTH.growth_factor, lambda s: ((_shaped(_KS, s),), {}), None),
    ("growth_rate", _GROWTH.growth_rate, lambda s: ((_shaped(_KS, s),), {}), None),
    ("power", _SOURCE.power, lambda s: ((_shaped(_KS, s) * h_Mpc,), {}), power_unit),
    ("sigma", _MV.sigma, lambda s: ((_shaped(_M, s) * Msun_h,), {}), None),
    ("dlnsigma_dlnm", _MV.dlnsigma_dlnm, lambda s: ((_shaped(_M, s) * Msun_h,), {}), None),
    ("m_from_sigma", _MV.m_from_sigma,
     lambda s: ((_MV.sigma(_shaped(_M, s) * Msun_h),), {}), Msun_h),
    ("m_from_radius", _MV.m_from_radius, lambda s: ((_shaped(_KS, s) * Mpc_h,), {}), Msun_h),
    ("radius_from_m", _MV.radius_from_m, lambda s: ((_shaped(_M, s) * Msun_h,), {}), Mpc_h),
    ("window", filters.TopHat().window, lambda s: ((_shaped(_KS, s),), {}), None),
    ("dwindow_dlnx", filters.SmoothK().dwindow_dlnx, lambda s: ((_shaped(_KS, s),), {}), None),
    ("modify_dndm", fits.Behroozi().modify_dndm,
     lambda s: (
         (_shaped(_M, s) * Msun_h, _shaped(_M, s) ** -1.9 * dndm_unit),
         {"z": 4.0, "ngtm": _shaped(_M, s) ** -0.9 * number_density_unit, "H0": 70 * H0_unit},
     ), dndm_unit),
] + [
    (f"{cls.__name__}.mass_ratio_to_200m", cls().mass_ratio_to_200m,
     lambda s: ((_shaped(_M, s) * 100 * Msun_h,), {"z": 0.5, "omega_m0": 0.3, "H0": 70 * H0_unit}),
     None)
    for cls in (fits.Bocquet200mDMOnly, fits.Bocquet200cHydro, fits.Bocquet500cDMOnly)
]  # fmt: skip


def _fit_inputs(cls, shape):
    """Inputs inside the fit's calibration domain, with sigma of the given shape."""
    kwargs = {
        "z": 7.0 if cls is fits.Yung24 else 0.0,
        "omega_m_z": 0.5,
        "delta_halo": 178.0 if cls is fits.Watson else 200.0,
        "delta_c": 1.686,
        "n_eff": -2.0,
        "m": 1e12 * Msun_h,
    }
    needed = cls.requires | fits._domain_inputs(cls.valid_domain)
    return (_shaped(_SIGMA, shape),), {k: v for k, v in kwargs.items() if k in needed}


FITS = sorted(fits.FittingFunction.get_models().values(), key=lambda m: m.__name__)
CASES += [
    (f"{cls.__name__}.fsigma", cls().fsigma, lambda s, cls=cls: _fit_inputs(cls, s), None)
    for cls in FITS
]


def _check_scalar_rule(out, shape, unit):
    if unit is None:
        assert not isinstance(out, u.Quantity)
        if shape == ():
            assert type(out) is np.float64
            assert isinstance(out, float)
        else:
            assert type(out) is np.ndarray
            assert out.shape == shape
    else:
        assert type(out) is u.Quantity
        assert out.unit is unit
        assert out.shape == shape


@pytest.mark.parametrize("shape", SHAPES, ids=["scalar", "1d", "2d"])
@pytest.mark.parametrize(("name", "method", "inputs", "unit"), CASES, ids=[c[0] for c in CASES])
def test_scalar_rule(name, method, inputs, unit, shape):
    args, kwargs = inputs(shape)
    _check_scalar_rule(method(*args, **kwargs), shape, unit)


FITS_WITH_PARAMETERS = [m for m in FITS if hasattr(m, "parameters")]


@pytest.mark.parametrize("shape", SHAPES, ids=["scalar", "1d", "2d"])
@pytest.mark.parametrize("cls", FITS_WITH_PARAMETERS, ids=_id)
def test_fit_parameters_follow_the_scalar_rule(cls, shape):
    """The ``parameters`` of a fit (a tuple of dimensionless values) follow the rule."""
    z = _shaped([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], shape) + (6.5 if cls is fits.Yung24 else 0.0)
    values = {"z": z, "delta_halo": 200.0, "omega_m_z": 0.3}
    names = list(inspect.signature(cls.parameters).parameters)[1:]
    for value in cls().parameters(*(values[n] for n in names)):
        _check_scalar_rule(value, shape, None)


#: Public boundary methods that return something other than numbers.
_NOT_NUMERIC = {"FittingFunction.inputs"}


def test_scalar_rule_covers_every_public_boundary_method():
    """Every public method decorated with unit_boundary is in CASES (or not numeric)."""
    modules = (filters, fits, growth, mass_variance, power_source, transfer)
    decorated = {
        f"{cls.__name__}.{name}"
        for module in modules
        for cls in vars(module).values()
        if isinstance(cls, type) and cls.__module__ == module.__name__
        for name, attr in vars(cls).items()
        if not name.startswith("_") and hasattr(attr, "__unit_boundary__")
    }
    covered = {method.__func__.__qualname__ for _, method, _, _ in CASES}
    covered |= {cls.parameters.__qualname__ for cls in FITS_WITH_PARAMETERS}
    assert decorated - _NOT_NUMERIC <= covered, decorated - _NOT_NUMERIC - covered


# ---------------------------------------------------------------------------------
# Error messages
# ---------------------------------------------------------------------------------


def test_errors_name_the_class_called():
    """An inherited method's errors name the subclass, not the class defining it."""
    model = fits.Bocquet200cDMOnly()
    with pytest.raises(UnitBoundaryError, match=r"^Bocquet200cDMOnly\.mass_ratio_to_200m\(\)"):
        model.mass_ratio_to_200m(1e14, z=0.0, omega_m0=0.3, H0=70 * H0_unit)
    with pytest.raises(u.UnitConversionError, match=r"^Bocquet200cDMOnly\.mass_ratio_to_200m\(\)"):
        model.mass_ratio_to_200m(1e14 * u.s, z=0.0, omega_m0=0.3, H0=70 * H0_unit)
    with pytest.raises(UnitBoundaryError, match=r"^Bocquet200cDMOnly\.fsigma\(\)"):
        model.fsigma(1.0, z=0.0, m=1e12)
    with pytest.raises(UnitBoundaryError, match=r"^Tinker10\.modify_dndm\(\)"):
        fits.Tinker10().modify_dndm(
            1e12, 1 * dndm_unit, z=0.0, ngtm=1 * number_density_unit, H0=70 * H0_unit
        )
    with pytest.raises(UnitBoundaryError, match=r"^Transfer\.transfer_function\(\)"):
        _TRANSFER.transfer_function(0.1)
    with pytest.raises(u.UnitConversionError, match=r"^MassVariance\.sigma\(\)"):
        _MV.sigma(1e12 * u.s)


# ---------------------------------------------------------------------------------
# Edge cases of the helpers
# ---------------------------------------------------------------------------------


def test_unit_context_rejects_a_non_scalar_h0():
    with pytest.raises(ValueError, match="H0 must be a scalar"):
        UnitContext([70, 71] * H0_unit)


def test_quantity_field_requires_a_value():
    with pytest.raises(UnitBoundaryError, match="FromArray: argument 'k'"):
        transfer_models.FromArray(k=None, t=np.ones(3))


def test_quantity_key_needs_a_quantity():
    assert quantity_key(None) is None
    with pytest.raises(TypeError, match="expected a Quantity"):
        quantity_key(1.0)


def test_python_float_output_becomes_a_numpy_scalar():
    class Toy:
        @unit_boundary()
        def half(self, x):
            return float(x) / 2

    out = Toy().half(3)
    assert type(out) is np.float64
    assert out == 1.5
