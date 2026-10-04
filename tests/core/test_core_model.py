"""Tests of hmf.core.model: the Model base class and the per-kind registry."""

import abc
import importlib.metadata
import inspect
import sys
import textwrap
import types

import attrs
import pytest

from hmf.core import field, model
from hmf.core.model import DuplicateAliasError, Model, ModelNotFoundError


def make_kind():
    """A fresh model kind, so that each test has its own registry."""

    @attrs.frozen(kw_only=True)
    class Fit(Model, kind=True):
        """A toy model kind."""

        @abc.abstractmethod
        def f(self, x: float) -> float:
            """Evaluate the model."""

    return Fit


@pytest.fixture
def Fit():
    return make_kind()


@pytest.fixture
def modules(tmp_path, monkeypatch):
    """Write importable modules: ``modules(name=source)``. Removed afterwards."""
    monkeypatch.syspath_prepend(str(tmp_path))
    written = []

    def write(**sources):
        for name, source in sources.items():
            (tmp_path / f"{name}.py").write_text(textwrap.dedent(source))
            written.append(name)

    yield write
    for name in written:
        sys.modules.pop(name, None)


# Plugin modules use a kind that they can import.
KIND_SOURCE = """
import attrs
from hmf.core.model import Model

@attrs.frozen(kw_only=True)
class PluginKind(Model, kind=True):
    pass
"""

MODELS_SOURCE = """
import attrs
from {kind_module} import PluginKind

@attrs.frozen(kw_only=True)
class Remote(PluginKind, alias="remote"):
    a: float = 1.0
"""


def test_qualified_name_registration(Fit):
    @attrs.frozen(kw_only=True)
    class Tinker(Fit):
        def f(self, x):
            return x

    name = f"{__name__}:test_qualified_name_registration.<locals>.Tinker"
    assert Tinker.qualified_name() == name
    assert Fit.get_models() == {name: Tinker}
    assert Fit.get(name) is Tinker
    assert Tinker.alias is None
    # The registered class is the final (slotted) one that attrs built.
    assert "__slots__" in Fit.get(name).__dict__


def test_alias(Fit):
    @attrs.frozen(kw_only=True)
    class Tinker(Fit, alias="Tinker08"):
        def f(self, x):
            return x

    assert Fit.get("Tinker08") is Tinker
    assert Tinker.alias == "Tinker08"
    assert dict(Fit.get_aliases()) == {"Tinker08": Tinker.qualified_name()}
    assert Tinker().f(2.0) == 2.0


def test_alias_not_inherited(Fit):
    @attrs.frozen(kw_only=True)
    class A(Fit, alias="a"):
        def f(self, x):
            return x

    @attrs.frozen(kw_only=True)
    class B(A):
        pass

    assert B.alias is None
    assert Fit.get("a") is A
    assert Fit.get(B.qualified_name()) is B


def test_duplicate_alias_is_an_error(Fit):
    @attrs.frozen(kw_only=True)
    class A(Fit, alias="dup"):
        def f(self, x):
            return x

    with pytest.raises(DuplicateAliasError, match=r"'dup'.*already belongs to.*override=True"):

        @attrs.frozen(kw_only=True)
        class B(Fit, alias="dup"):
            def f(self, x):
                return 2 * x

    assert Fit.get("dup") is A


def test_override(Fit):
    @attrs.frozen(kw_only=True)
    class A(Fit, alias="dup"):
        def f(self, x):
            return x

    @attrs.frozen(kw_only=True)
    class B(Fit, alias="dup", override=True):
        def f(self, x):
            return 2 * x

    assert Fit.get("dup") is B
    # A is still reachable by its qualified name.
    assert Fit.get(A.qualified_name()) is A


def test_same_class_name_elsewhere_does_not_shadow(Fit):
    @attrs.frozen(kw_only=True)
    class Tinker(Fit, alias="Tinker"):
        def f(self, x):
            return x

    builtin = Tinker

    def user_code():
        # A user class with the same __name__, in other code, without an alias.
        @attrs.frozen(kw_only=True)
        class Tinker(Fit):
            def f(self, x):
                return -x

        return Tinker

    user = user_code()
    assert Fit.get("Tinker") is builtin
    assert len(Fit.get_models()) == 2
    assert Fit.get(user.qualified_name()) is user


def test_redefinition_replaces(Fit):
    classes = []
    for _ in range(2):

        @attrs.frozen(kw_only=True)
        class Again(Fit, alias="again"):
            def f(self, x):
                return x

        classes.append(Again)
    assert Fit.get("again") is classes[1]
    assert len(Fit.get_models()) == 1


def test_abstract_not_registered(Fit):
    @attrs.frozen(kw_only=True)
    class Base(Fit, abstract=True):
        scale: float = 1.0

    @attrs.frozen(kw_only=True)
    class Concrete(Base, alias="c"):
        def f(self, x):
            return self.scale * x

    assert list(Fit.get_models().values()) == [Concrete]
    with pytest.raises(TypeError, match="abstract"):
        Fit.get(Base)


def test_abstract_methods_enforced(Fit):
    @attrs.frozen(kw_only=True)
    class Incomplete(Fit, alias="incomplete"):
        a: float = 1.0

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()
    with pytest.raises(TypeError, match="abstract"):
        Fit.get("incomplete")


def test_kind_is_not_a_model(Fit):
    with pytest.raises(TypeError, match="abstract"):
        Fit.get(Fit)
    assert Fit not in Fit.get_models().values()


def test_kind_rules():
    with pytest.raises(TypeError, match="can not have an alias"):

        class Kind(Model, kind=True, alias="k"):
            pass

    with pytest.raises(TypeError, match="subclasses Model directly"):

        @attrs.frozen(kw_only=True)
        class Orphan(Model):
            pass

    with pytest.raises(TypeError, match="no registry"):
        Model.get("anything")


@pytest.mark.parametrize("alias", ["", "a.b", "pkg:Cls"])
def test_bad_alias(Fit, alias):
    with pytest.raises(ValueError, match="alias"):

        class Bad(Fit, alias=alias):
            pass


def test_registries_isolated_per_kind():
    FitA, FitB = make_kind(), make_kind()

    @attrs.frozen(kw_only=True)
    class A(FitA, alias="same"):
        def f(self, x):
            return x

    @attrs.frozen(kw_only=True)
    class B(FitB, alias="same"):
        def f(self, x):
            return x

    assert FitA.get("same") is A
    assert FitB.get("same") is B
    assert list(FitA.get_models().values()) == [A]
    with pytest.raises(TypeError, match="not a Fit"):
        FitA.get(B)


def test_nested_kind_registers_in_both():
    Fit = make_kind()

    @attrs.frozen(kw_only=True)
    class SubFit(Fit, kind=True):
        pass

    @attrs.frozen(kw_only=True)
    class A(SubFit, alias="a"):
        def f(self, x):
            return x

    assert SubFit.get("a") is A
    assert Fit.get("a") is A


def test_duplicate_alias_in_outer_kind_registers_nothing():
    Fit = make_kind()

    @attrs.frozen(kw_only=True)
    class SubFit(Fit, kind=True):
        pass

    @attrs.frozen(kw_only=True)
    class Outer(Fit, alias="taken"):
        def f(self, x):
            return x

    with pytest.raises(DuplicateAliasError):

        @attrs.frozen(kw_only=True)
        class Inner(SubFit, alias="taken"):
            def f(self, x):
                return x

    assert len(SubFit.get_models()) == 0
    assert Fit.get("taken") is Outer


def test_get_models_read_only_and_live(Fit):
    models = Fit.get_models()
    assert isinstance(models, types.MappingProxyType)
    with pytest.raises(TypeError):
        models["x"] = object

    @attrs.frozen(kw_only=True)
    class Later(Fit):
        def f(self, x):
            return x

    assert Later.qualified_name() in models


def test_get_models_on_subclass_filters(Fit):
    @attrs.frozen(kw_only=True)
    class Base(Fit, abstract=True):
        pass

    @attrs.frozen(kw_only=True)
    class A(Base):
        def f(self, x):
            return x

    @attrs.frozen(kw_only=True)
    class B(Fit):
        def f(self, x):
            return x

    assert list(Base.get_models().values()) == [A]
    assert len(Fit.get_models()) == 2


def test_get_class(Fit):
    @attrs.frozen(kw_only=True)
    class A(Fit):
        def f(self, x):
            return x

    assert Fit.get(A) is A
    with pytest.raises(TypeError, match="not a Fit"):
        Fit.get(int)


def test_did_you_mean(Fit):
    @attrs.frozen(kw_only=True)
    class Tinker(Fit, alias="Tinker08"):
        def f(self, x):
            return x

    with pytest.raises(ModelNotFoundError, match=r"Did you mean 'Tinker08'\?") as e:
        Fit.get("Tinkr08")
    assert "Available aliases: ('Tinker08',)" in str(e.value)
    assert isinstance(e.value, LookupError)


def test_import_path_resolution(modules):
    modules(plugkind_a=KIND_SOURCE, plugmodels_a=MODELS_SOURCE.format(kind_module="plugkind_a"))
    from plugkind_a import PluginKind

    assert "plugmodels_a" not in sys.modules
    with pytest.raises(ModelNotFoundError):
        PluginKind.get("remote")

    remote = PluginKind.get("plugmodels_a:Remote")
    assert remote.__name__ == "Remote"
    # Importing it registered it.
    assert PluginKind.get("remote") is remote
    assert PluginKind.get("plugmodels_a.Remote") is remote


def test_import_path_wrong_kind(modules, Fit):
    modules(plugkind_b=KIND_SOURCE, plugmodels_b=MODELS_SOURCE.format(kind_module="plugkind_b"))
    with pytest.raises(TypeError, match="not a Fit"):
        Fit.get("plugmodels_b:Remote")


def test_import_path_not_found(Fit):
    with pytest.raises(ModelNotFoundError, match="import path"):
        Fit.get("no_such_module_xyz:Cls")
    # Malformed paths: no module, or no attribute.
    for path in (":Cls", "hmf.core.model:", ".Cls"):
        with pytest.raises(ModelNotFoundError):
            Fit.get(path)
    with pytest.raises(ModelNotFoundError):
        Fit.get("hmf.core.model:NoSuchClass")


def test_entry_point_discovery(modules, monkeypatch):
    modules(plugkind_c=KIND_SOURCE, plugmodels_c=MODELS_SOURCE.format(kind_module="plugkind_c"))
    from plugkind_c import PluginKind

    eps = importlib.metadata.EntryPoints(
        [importlib.metadata.EntryPoint(name="toy", value="plugmodels_c", group="hmf.models")]
    )
    requested = []

    def entry_points(**kwargs):
        requested.append(kwargs)
        return eps.select(**kwargs)

    monkeypatch.setattr(importlib.metadata, "entry_points", entry_points)
    monkeypatch.setattr(model, "_loaded_entry_points", set())

    # Nothing is loaded until a lookup needs it.
    assert "plugmodels_c" not in sys.modules
    remote = PluginKind.get("remote")
    assert remote.__module__ == "plugmodels_c"
    assert requested == [{"group": "hmf.models"}]
    # Each entry point is loaded once.
    PluginKind.get_models()
    assert len(model._loaded_entry_points) == 1


def test_entry_point_loaded_by_get_models(modules, monkeypatch):
    modules(plugkind_d=KIND_SOURCE, plugmodels_d=MODELS_SOURCE.format(kind_module="plugkind_d"))
    from plugkind_d import PluginKind

    ep = importlib.metadata.EntryPoint(name="toy", value="plugmodels_d", group="hmf.models")
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda **kw: [ep])
    monkeypatch.setattr(model, "_loaded_entry_points", set())
    assert [m.__name__ for m in PluginKind.get_models().values()] == ["Remote"]


def test_broken_entry_point_warns(monkeypatch, Fit):
    ep = importlib.metadata.EntryPoint(name="broken", value="no_such_pkg_xyz", group="hmf.models")
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda **kw: [ep])
    monkeypatch.setattr(model, "_loaded_entry_points", set())
    with (
        pytest.warns(RuntimeWarning, match="'broken'"),
        pytest.raises(ModelNotFoundError),
    ):
        Fit.get("anything")


def test_cooperative_mixin(Fit):
    class Tagged:
        """A mixin with its own class keyword."""

        def __init_subclass__(cls, /, tag=None, **kwargs):
            super().__init_subclass__(**kwargs)
            # attrs calls this again without keywords; keep the first tag.
            if tag is not None:
                cls.tag = tag

    @attrs.frozen(kw_only=True)
    class A(Tagged, Fit, tag="first", alias="a"):
        def f(self, x):
            return x

    @attrs.frozen(kw_only=True)
    class B(Fit, Tagged, alias="b", tag="last"):
        def f(self, x):
            return x

    assert (A.tag, B.tag) == ("first", "last")
    assert Fit.get("a") is A
    assert Fit.get("b") is B


def test_class_keyword_typo_is_an_error(Fit):
    with pytest.raises(TypeError):

        class A(Fit, alais="a"):
            pass


def test_parameters_docstring(Fit):
    @attrs.frozen(kw_only=True)
    class Documented(Fit):
        """A documented model."""

        A: float = field(default=0.186, doc="The amplitude.")
        a: float = field(default=1.47, doc="The slope.\nOn two lines.")

        def f(self, x):
            return x

    doc = inspect.getdoc(Documented)
    assert doc.startswith("A documented model.")
    assert "Parameters\n----------\nA : float, default 0.186\n    The amplitude." in doc
    assert "a : float, default 1.47\n    The slope.\n    On two lines." in doc
    assert [f.name for f in Documented.fields_info()] == ["A", "a"]


def test_hand_written_parameters_kept(Fit):
    @attrs.frozen(kw_only=True)
    class Handwritten(Fit):
        """A model.

        Parameters
        ----------
        A : float
            Written by hand.
        """

        A: float = field(default=1.0, doc="Generated.")

        def f(self, x):
            return x

    assert "Generated." not in Handwritten.__doc__


def test_model_class_variables(Fit):
    @attrs.frozen(kw_only=True)
    class Cited(Fit, alias="cited"):
        references = ("Someone et al. 2020",)
        parameter_source = "Someone et al. 2020, Table 1 (published version)"

        def f(self, x):
            return x

    assert Cited.references == ("Someone et al. 2020",)
    assert Cited.parameter_source.startswith("Someone")
    assert Model.references == ()


def test_models_are_frozen_and_keyword_only(Fit):
    @attrs.frozen(kw_only=True)
    class A(Fit):
        a: float = 1.0

        def f(self, x):
            return self.a * x

    with pytest.raises(TypeError):
        A(2.0)
    m = A(a=2.0)
    with pytest.raises(attrs.exceptions.FrozenInstanceError):
        m.a = 3.0
    assert m == A(a=2.0)
