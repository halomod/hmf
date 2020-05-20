from pytest import raises
import hmf
import pytest


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
