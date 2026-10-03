"""Tests of the references and get_acknowledgments machinery."""

import warnings
from typing import ClassVar

import numpy as np
import pytest

from hmf import MassFunction, Transfer
from hmf._internals import _references as refs
from hmf._internals._cache import parameter, subframework
from hmf._internals._framework import HMF_REFERENCE, Component, Framework
from hmf.cosmology.cosmo import Cosmology
from hmf.density_field import transfer_models as tm
from hmf.mass_function import fitting_functions as ff


@pytest.fixture(scope="module")
def mf():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return MassFunction(transfer_model="EH", hmf_model="Tinker08")


def test_default_contents(mf):
    acks = mf.get_acknowledgments()

    assert acks[0] == HMF_REFERENCE
    assert len(acks) == len(set(acks))
    assert any(ref.startswith("Tinker, J., et al., 2008. ApJ 688") for ref in acks)
    assert refs.EH98 in acks
    for ref in (*mf.growth_model.references, *mf.filter_model.references):
        assert ref in acks

    # The default cosmology (Planck18) carries its own reference.
    assert any("Planck" in ref for ref in acks)


def test_default_massfunction_has_no_duplicates():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mf = MassFunction()
    acks = mf.get_acknowledgments()
    assert acks[0] == HMF_REFERENCE
    assert len(acks) == len(set(acks))
    assert all(ref in acks for ref in mf.hmf_model.references)
    assert all(ref in acks for ref in mf.transfer_model.references)
    assert all(ref in acks for ref in mf.growth_model.references)


def test_hmf_model_changes_fit_reference(mf):
    (tinker,) = ff.Tinker08.references
    (st,) = ff.ST.references

    other = mf.clone(hmf_model="ST")
    assert st in other.get_acknowledgments()
    assert tinker not in other.get_acknowledgments()
    assert tinker in mf.get_acknowledgments()


def test_transfer_model_changes_reference(mf):
    other = mf.clone(transfer_model="BBKS")
    assert refs.BBKS86 in other.get_acknowledgments()
    assert refs.EH98 not in other.get_acknowledgments()


def test_takahashi_switch():
    t = Transfer(transfer_model="EH")
    assert refs.SMITH03 in t.get_acknowledgments()
    assert refs.TAKAHASHI12 in t.get_acknowledgments()

    t.update(takahashi=False)
    assert refs.SMITH03 in t.get_acknowledgments()
    assert refs.TAKAHASHI12 not in t.get_acknowledgments()


def test_does_not_compute_transfer(mf, monkeypatch):
    calls = []

    def counting_lnt(self, lnk):
        calls.append(lnk)
        return np.zeros_like(lnk)

    monkeypatch.setattr(tm.EH_BAO, "lnt", counting_lnt)
    fresh = mf.clone()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fresh.get_acknowledgments()
    assert calls == []


@pytest.mark.skipif(not tm.HAVE_CAMB, reason="CAMB is not installed")
def test_does_not_run_camb(monkeypatch):
    import camb

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mf = MassFunction(transfer_model="CAMB")

    def fail(*args, **kwargs):
        raise AssertionError("CAMB was run")

    monkeypatch.setattr(camb, "get_transfer_functions", fail)
    monkeypatch.setattr(camb, "get_results", fail)

    assert refs.CAMB in mf.get_acknowledgments()


def test_user_component_references(mf):
    class MyFit(ff.PS):
        references: ClassVar[tuple[str, ...]] = ("Me, A., 2026. Nowhere 1, 1.",)

    acks = mf.clone(hmf_model=MyFit).get_acknowledgments()
    assert "Me, A., 2026. Nowhere 1, 1." in acks
    # A subclass's own references replace those it would inherit.
    assert ff.PS.references[0] not in acks


@pytest.mark.parametrize("name", sorted(ff.BaseFittingFunction.get_models()))
def test_every_fitting_function_has_references(name):
    fit = ff.BaseFittingFunction.get_models()[name]
    assert fit.references
    assert all(isinstance(ref, str) and ref == ref.strip() for ref in fit.references)
    # The first reference is the one cited in the docstring.
    assert " ".join(fit.references[0].split()) in " ".join(fit.__doc__.split())


def test_inherited_references():
    # EH_NoBAO is an EH fit too, so it inherits EH_BAO's references.
    assert tm.EH_NoBAO.references == (refs.EH98,)


def test_framework_references_include_parents():
    class Child(Cosmology):
        references: ClassVar[tuple[str, ...]] = ("Child, C., 2020.",)

    acks = Child().get_acknowledgments()
    assert acks[0] == HMF_REFERENCE
    assert "Child, C., 2020." in acks


def test_custom_cosmology_without_reference():
    from astropy.cosmology import FlatLambdaCDM

    acks = Cosmology(cosmo_model=FlatLambdaCDM(H0=70, Om0=0.3)).get_acknowledgments()
    assert acks == [HMF_REFERENCE]


class _SubComponent(Component):
    references: ClassVar[tuple[str, ...]] = ("Sub, S., 2010.",)


class _Inner(Framework):
    references: ClassVar[tuple[str, ...]] = ("Inner, I., 2000.",)

    def __init__(self, sub_model=_SubComponent):
        self.sub_model = sub_model

    @parameter("model")
    def sub_model(self, val):
        return val


class _Outer(Framework):
    def __init__(self, x=1):
        self.x = x

    @parameter("param")
    def x(self, val):
        return val

    @subframework
    def inner(self):
        return _Inner()


def test_subframework_references():
    acks = _Outer().get_acknowledgments()
    assert acks == [HMF_REFERENCE, "Inner, I., 2000.", "Sub, S., 2010."]


class _Shared(_Outer):
    _inner = _Inner()

    @subframework
    def inner(self):
        return self._inner

    @subframework
    def also_inner(self):
        return self._inner


def test_shared_subframework_visited_once():
    acks = _Shared().get_acknowledgments()
    assert acks == [HMF_REFERENCE, "Inner, I., 2000.", "Sub, S., 2010."]


def test_component_without_references():
    class NoRefs(Component):
        pass

    assert NoRefs.references == ()
    assert _Inner(sub_model=NoRefs).get_acknowledgments() == [HMF_REFERENCE, "Inner, I., 2000."]


# Generic methods or user-supplied data, which have nothing to cite.
_UNCITED_MODELS = {
    "BaseGrowthFactor": {"FromFile", "FromArray"},
    "BaseFilter": {"TopHat", "Gaussian", "SharpK"},
    "TransferComponent": {"FromFile", "FromArray"},
    "BaseMassDefinition": {"SphericalOverdensity", "SOGeneric", "SOMean", "SOCritical"},
}


def test_every_model_has_references():
    import hmf.alternatives.wdm  # noqa: F401  (registers the WDM models)
    from hmf._internals._framework import get_base_components

    missing = [
        f"{base.__name__}.{name}"
        for base in get_base_components()
        for name, model in getattr(base, "_plugins", {}).items()
        if not model.references and name not in _UNCITED_MODELS.get(base.__name__, set())
    ]
    assert missing == []
