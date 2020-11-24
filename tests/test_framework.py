from pytest import raises
import hmf
import pytest
from hmf.density_field.transfer_models import TransferComponent
from hmf._internals import pluggable, get_base_components, get_base_component
from hmf._internals._framework import get_model_
from deprecation import fail_if_not_removed
from hmf import GrowthFactor


def test_incorrect_argument():
    with raises(TypeError):
        hmf.MassFunction(wrong_arg=3)


def test_incorrect_update_arg():
    with raises(ValueError):
        t = hmf.MassFunction()
        t.update(wrong_arg=3)


@pytest.fixture(scope="class")
def cls():
    return hmf.MassFunction


@pytest.fixture(scope="class")
def inst(cls):
    return cls(z=10)


def test_parameter_names(cls):
    assert "cosmo_model" in cls.get_all_parameter_names()


def test_parameter_defaults(cls):
    assert type(cls.get_all_parameter_defaults(recursive=False)) is dict

    assert cls.get_all_parameter_defaults()["z"] == 0


def test_parameter_default_rec(cls):
    pd = cls.get_all_parameter_defaults(recursive=True)
    assert type(pd["cosmo_params"]) is dict


def test_param_values(inst):
    assert type(inst.parameter_values) is dict
    assert inst.parameter_values["z"] == 10


def test_qnt_avail(cls):
    assert "dndm" in cls.quantities_available()


def test_parameter_info(cls):
    assert cls.parameter_info() is None
    assert cls.parameter_info(names=["z"]) is None


def test_pluggable():
    class A:
        pass

    @pluggable
    class B(A):
        pass

    class C(B):
        pass

    assert not hasattr(A, "_plugins")
    assert "C" in B._plugins
    assert "C" in C._plugins


def test_get_base_components():
    assert TransferComponent in get_base_components()


def test_get_base_component():
    assert get_base_component("TransferComponent") == TransferComponent


@fail_if_not_removed
def test_get_model():
    assert get_model_("GrowthFactor", "hmf.cosmology.growth_factor") == GrowthFactor


def test_growth_plugins():
    assert "GenMFGrowth" in GrowthFactor._plugins
