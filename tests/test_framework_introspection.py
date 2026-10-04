"""Tests that Framework introspection is static: it neither computes nor runs CAMB."""

import io
from contextlib import redirect_stdout

import pytest

import hmf
from hmf import MassFunction
from hmf._internals._cache import cached_quantity, parameter
from hmf._internals._framework import Framework
from hmf.alternatives.wdm import MassFunctionWDM
from hmf.density_field import transfer_models as tm
from hmf.mass_function import fitting_functions as ff

FRAMEWORKS = [hmf.Cosmology, hmf.Transfer, MassFunction, MassFunctionWDM]


class ExtendedMassFunction(MassFunction):
    """A halomod-style subclass adding parameters and quantities."""

    ERROR_ON_SOMETHING = True

    def __init__(self, bias_scale: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.bias_scale = bias_scale

    @parameter("param")
    def bias_scale(self, val):
        """A made-up scale for the bias.

        :type: float
        """
        return val

    @cached_quantity
    def scaled_bias(self):
        """A public quantity."""
        return self.bias_scale * self.nu

    @cached_quantity
    def _private_quantity(self):
        return self.bias_scale

    def helper(self):
        return 1


@pytest.fixture
def no_transfer_calls(monkeypatch):
    """Fail if any transfer function (in particular CAMB) is evaluated."""
    calls = []

    def counted(self, lnk):
        calls.append(type(self).__name__)
        raise AssertionError("Introspection must not evaluate the transfer function")

    for model in tm.TransferComponent._plugins.values():
        monkeypatch.setattr(model, "lnt", counted)

    try:
        import camb
    except ImportError:  # pragma: no cover
        pass
    else:
        for name in ["get_results", "get_transfer_functions", "get_background"]:

            def counted_camb(*args, _name=name, **kwargs):
                calls.append(_name)
                raise AssertionError("Introspection must not run CAMB")

            monkeypatch.setattr(camb, name, counted_camb)

    yield calls
    assert calls == []


# Deprecated aliases are not quantities in their own right, so quantities_available()
# leaves them out (listing them would also make anything that evaluates every quantity
# emit their DeprecationWarning).
_DEPRECATED_ALIASES = {"nu"}


def _old_quantities_available(cls):
    """The pre-optimisation result, minus its instantiation (see test docstrings)."""
    params = set(cls.get_all_parameter_names())
    return [
        name
        for name in dir(cls)
        if name not in params and not name.startswith("__") and name not in dir(Framework)
    ]


def _index_parameter_names(cls):
    """Parameter names as recorded by an (unvalidated) instance, as before."""
    obj = type.__call__(cls)
    return set(getattr(obj, "_" + cls.__name__ + "__recalc_par_prop"))


@pytest.mark.parametrize("cls", [*FRAMEWORKS, ExtendedMassFunction], ids=lambda c: c.__name__)
def test_parameter_names_match_instance_index(cls, no_transfer_calls):
    names = cls.get_all_parameter_names()
    assert len(names) == len(set(names))
    assert set(names) == _index_parameter_names(cls)


@pytest.mark.parametrize("cls", [*FRAMEWORKS, ExtendedMassFunction], ids=lambda c: c.__name__)
def test_quantities_available_are_the_public_quantities(cls, no_transfer_calls):
    quantities = cls.quantities_available()

    # The same public names as before, minus constants, methods and deprecated aliases
    # (plain properties that only forward to a quantity, e.g. ``MassFunction.nu``).
    expected = {
        name
        for name in _old_quantities_available(cls)
        if not name.startswith("_")
        and isinstance(getattr(cls, name), property)
        and name not in _DEPRECATED_ALIASES
    }
    assert set(quantities) == expected
    assert quantities == sorted(quantities)
    assert not any(name.startswith("_") for name in quantities)


def test_quantities_available_excludes_junk(no_transfer_calls):
    quantities = MassFunction.quantities_available()
    assert {"dndm", "ngtm", "sigma", "power", "growth_factor", "cosmo", "filter"} <= set(quantities)
    for junk in ["ERROR_ON_BAD_MDEF", "_gtm", "_dlnsdlnm", "_converted_dndm"]:
        assert junk not in quantities
    assert not set(quantities) & set(MassFunction.get_all_parameter_names())


def test_subclass_introspection(no_transfer_calls):
    names = ExtendedMassFunction.get_all_parameter_names()
    assert "bias_scale" in names
    assert set(MassFunction.get_all_parameter_names()) < set(names)

    quantities = ExtendedMassFunction.quantities_available()
    assert "scaled_bias" in quantities
    assert set(MassFunction.quantities_available()) < set(quantities)
    for junk in ["_private_quantity", "helper", "ERROR_ON_SOMETHING", "bias_scale"]:
        assert junk not in quantities

    defaults = ExtendedMassFunction.get_all_parameter_defaults()
    assert defaults["bias_scale"] == 2.0
    assert defaults["Mmin"] == 10.0

    out = io.StringIO()
    with redirect_stdout(out):
        ExtendedMassFunction.parameter_info(names=["bias_scale"])
    assert "bias_scale : float" in out.getvalue()
    assert "A made-up scale for the bias." in out.getvalue()


def test_parameter_defaults(no_transfer_calls):
    defaults = MassFunction.get_all_parameter_defaults()
    assert list(defaults) == MassFunction.get_all_parameter_names()
    assert defaults["z"] == 0
    assert defaults["Mmin"] == 10.0
    assert defaults["hmf_model"] is ff.Tinker08
    assert defaults["hmf_params"] == ff.Tinker08._defaults
    assert defaults["filter_model"] is hmf.density_field.filters.TopHat
    assert defaults["transfer_params"] == defaults["transfer_model"]._defaults

    defaults = MassFunction.get_all_parameter_defaults(recursive=False)
    assert defaults["hmf_params"] == {}


@pytest.mark.parametrize("cls", FRAMEWORKS, ids=lambda c: c.__name__)
def test_parameter_info_is_static(cls, no_transfer_calls):
    out = io.StringIO()
    with redirect_stdout(out):
        assert cls.parameter_info() is None
    for name in cls.get_all_parameter_names():
        assert f"{name} : " in out.getvalue()


def test_get_all_parameter_names_does_not_instantiate(monkeypatch):
    def fail(self, *args, **kwargs):
        raise AssertionError("must not instantiate")

    monkeypatch.setattr(MassFunction, "__init__", fail)
    assert "z" in MassFunction.get_all_parameter_names()
    assert "dndm" in MassFunction.quantities_available()
    with redirect_stdout(io.StringIO()):
        MassFunction.parameter_info()
