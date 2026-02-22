import sys
import typing

import pytest
from deprecation import fail_if_not_removed

import hmf
from hmf import GrowthFactor, MassFunction
from hmf._internals import get_base_component, get_base_components, pluggable
from hmf._internals._cache import cached_quantity, parameter
from hmf._internals._framework import Component, Framework, get_mdl, get_model, get_model_
from hmf.density_field.transfer_models import TransferComponent


def test_incorrect_argument():
    with pytest.raises(TypeError):
        hmf.MassFunction(wrong_arg=3)


def test_incorrect_update_arg():
    with pytest.raises(ValueError):
        t = hmf.MassFunction(transfer_model="EH")
        t.update(wrong_arg=3)


@pytest.fixture(scope="class")
def cls():
    return hmf.MassFunction


@pytest.fixture(scope="class")
def inst(cls):
    return cls(z=10, transfer_model="EH")


@pytest.mark.filterwarnings("ignore:'extrapolate_with_eh' was not set")
def test_parameter_names(cls):
    assert "cosmo_model" in cls.get_all_parameter_names()


@pytest.mark.filterwarnings("ignore:'extrapolate_with_eh' was not set")
def test_parameter_defaults(cls):
    assert type(cls.get_all_parameter_defaults(recursive=False)) is dict

    assert cls.get_all_parameter_defaults()["z"] == 0


@pytest.mark.filterwarnings("ignore:'extrapolate_with_eh' was not set")
def test_parameter_default_rec(cls):
    pd = cls.get_all_parameter_defaults(recursive=True)
    assert type(pd["cosmo_params"]) is dict


def test_param_values(inst):
    assert type(inst.parameter_values) is dict
    assert inst.parameter_values["z"] == 10


@pytest.mark.filterwarnings("ignore:'extrapolate_with_eh' was not set")
def test_qnt_avail(cls):
    assert "dndm" in cls.quantities_available()


@pytest.mark.filterwarnings("ignore:'extrapolate_with_eh' was not set")
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


def test_validate_inputs():
    with pytest.raises(AssertionError, match=r"Mmin > Mmax: 10, 9"):
        MassFunction(Mmin=10, Mmax=9, transfer_model="EH")

    m = MassFunction(Mmin=10, Mmax=11, transfer_model="EH")
    with pytest.raises(AssertionError, match="Mmin > Mmax: 10, 9"):
        m.update(Mmax=9)

    # Without checking on, we can still manually set it, but it will warn us
    with pytest.warns(UserWarning, match="You are setting Mmin directly."):
        m.Mmin = 8

    # Ensure that validation still runs.
    with (
        pytest.warns(UserWarning, match="You are setting Mmax directly."),
        pytest.raises(AssertionError, match="Mmin > Mmax: 8, 7"),
    ):
        m.Mmax = 7


def test_component_invalid_param():
    @pluggable
    class Dummy(Component):
        _defaults: typing.ClassVar = {"a": 1}

    with pytest.raises(ValueError, match="not a valid argument"):
        Dummy(b=1)


def test_get_base_component_missing():
    with pytest.raises(ValueError, match="There are no components called"):
        get_base_component("DoesNotExist")


def test_get_base_component_multiple_warns():
    @pluggable
    class DupComponent(Component):
        pass

    def construct():
        @pluggable
        class DupComponent(Component):
            pass

        return DupComponent

    construct()

    with pytest.warns(UserWarning, match="More than one component called"):
        get_base_component("DupComponent")


def test_get_base_component_bad_type():
    with pytest.raises(ValueError, match="must be str or a Component subclass"):
        get_base_component(123)


def test_get_mdl_kind_lookup_and_error():
    @pluggable
    class DummyBase(Component):
        pass

    class DummyModel(DummyBase):
        pass

    assert get_mdl("DummyModel", DummyBase) is DummyModel

    with pytest.raises(ValueError, match="not a defined"):
        get_mdl("MissingModel", DummyBase)


def test_get_mdl_ambiguous_and_missing(monkeypatch):
    @pluggable
    class BaseA(Component):
        pass

    @pluggable
    class BaseB(Component):
        pass

    type("SharedModel", (BaseA,), {})
    SharedB = type("SharedModel", (BaseB,), {})

    monkeypatch.setattr(
        sys.modules["hmf._internals._framework"],
        "get_base_components",
        lambda: [BaseA, BaseB],
    )

    with pytest.warns(UserWarning, match="More than one model was found"):
        model = get_mdl("SharedModel")
    assert model is SharedB

    with pytest.raises(ValueError, match="No model found"):
        get_mdl("NoModelFound")


def test_get_mdl_invalid_class():
    with pytest.raises(ValueError, match="must be str or Component subclass"):
        get_mdl(object(), Component)


def test_get_model_deprecated():
    with pytest.deprecated_call():
        inst = get_model("Component", "hmf._internals._framework")

    assert isinstance(inst, Component)


def test_component_get_models():
    @pluggable
    class ModelBase(Component):
        pass

    class ModelImpl(ModelBase):
        pass

    models = ModelBase.get_models()
    assert models["ModelImpl"] is ModelImpl


def test_validator_calls_validate():
    class Validated(Framework):
        def __init__(self):
            self._validate = False
            self.validated = 0
            self.x = 1
            self._validate = True

        @parameter("param")
        def x(self, val):
            return val

        def validate(self):
            self.validated += 1

    obj = Validated()
    assert obj.validated == 1


def test_framework_update_and_defaults():
    class Child(Framework):
        def __init__(self, y=1):
            self._validate = False
            self.validated = 0
            self.y = y
            self._validate = True

        @parameter("param")
        def y(self, val):
            return val

        def validate(self):
            self.validated += 1

    class ChildModel:
        _defaults: typing.ClassVar = {"y": 2}

    class Parent(Framework):
        def __init__(self, x=1, child=None):
            self._validate = False
            self.validated = 0
            self.x = x
            self.child = child or Child()
            self.other_params = {}
            self.missing_params = {}
            self._validate = True

        @parameter("param")
        def x(self, val):
            return val

        @parameter("param")
        def missing_params(self, val):
            return val

        @parameter("param")
        def other_params(self, val):
            return val

        @property
        def other_model(self):
            return ChildModel

        @cached_quantity
        def q(self):
            return self.x + self.child.y

        def validate(self):
            self.validated += 1

    parent = Parent()
    parent.update(x=3, child_params={"y": 4})

    assert parent.x == 3
    assert parent.child.y == 4
    assert parent.validated >= 1

    defaults = Parent.get_all_parameter_defaults(recursive=True)
    assert defaults["other_params"] == {"y": 2}
    assert defaults["missing_params"] == {}

    with pytest.raises(ValueError, match="Invalid arguments"):
        parent.update(bad=1)

    clone = parent.clone(x=5)
    assert clone.x == 5
    assert parent.x == 3


def test_get_dependencies_and_parameter_info(capsys):
    class Simple(Framework):
        def __init__(self):
            self._validate = False
            self.x = 2
            self._validate = True

        @parameter("param")
        def x(self, val):
            r"""Value for x."""
            return val

        @cached_quantity
        def q(self):
            return self.x + 1

    obj = Simple()
    setattr(obj, "_" + obj.__class__.__name__ + "__recalc_prop_par_static", {"q": {"x"}})

    assert obj.get_dependencies("q") == {"x"}

    assert Simple.parameter_info() is None
    output = capsys.readouterr().out
    assert "x" in output


def test_parameter_info_empty_doc(capsys):
    class EmptyDoc(Framework):
        def __init__(self):
            self._validate = False
            self.empty = 1
            self._validate = True

        @parameter("param")
        def empty(self, val):
            return val

    assert EmptyDoc.parameter_info() is None
    output = capsys.readouterr().out
    assert "empty" in output
