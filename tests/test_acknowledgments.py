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

    assert next(iter(acks)) == "hmf"
    assert acks["hmf"] == (HMF_REFERENCE,)
    assert len(acks["hmf_model"]) == 1
    assert acks["hmf_model"][0].startswith("Tinker, J., et al., 2008. ApJ 688")
    assert acks["transfer_model"] == (refs.EH98,)
    assert acks["growth_model"] == mf.growth_model.references
    assert acks["filter_model"] == mf.filter_model.references

    # The default cosmology (Planck18) carries its own reference.
    assert len(acks["cosmo_model"]) == 1
    assert "Planck" in acks["cosmo_model"][0]

    # No mass definition is set, so there is nothing to cite for it.
    assert "mdef_model" not in acks


def test_model_without_references_gives_empty_tuple(mf):
    assert mf.clone(filter_model="TopHat").get_acknowledgments()["filter_model"] == ()


def test_subframework_keys_are_prefixed():
    assert _Outer().get_acknowledgments() == {
        "hmf": (HMF_REFERENCE,),
        "inner": ("Inner, I., 2000.",),
        "inner.sub_model": ("Sub, S., 2010.",),
    }


def test_flat_list(mf):
    acks = mf.get_acknowledgments()
    flat = mf.get_acknowledgments(flat=True)

    assert flat[0] == HMF_REFERENCE
    assert len(flat) == len(set(flat))
    assert set(flat) == {ref for group in acks.values() for ref in group}


def test_default_massfunction_has_no_duplicates():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mf = MassFunction()
    acks = mf.get_acknowledgments(flat=True)
    assert acks[0] == HMF_REFERENCE
    assert len(acks) == len(set(acks))
    assert all(ref in acks for ref in mf.hmf_model.references)
    assert all(ref in acks for ref in mf.transfer_model.references)
    assert all(ref in acks for ref in mf.growth_model.references)


def test_hmf_model_changes_fit_reference(mf):
    (tinker,) = ff.Tinker08.references
    (st,) = ff.ST.references

    assert mf.clone(hmf_model="ST").get_acknowledgments()["hmf_model"] == (st,)
    assert mf.get_acknowledgments()["hmf_model"] == (tinker,)


def test_transfer_model_changes_reference(mf):
    other = mf.clone(transfer_model="BBKS")
    assert other.get_acknowledgments()["transfer_model"] == (refs.BBKS86,)
    assert refs.EH98 not in other.get_acknowledgments(flat=True)


def test_halofit_not_cited():
    # HALOFIT is only used for the non-linear power spectrum, which the
    # mass function never needs, so it is cited in those docstrings instead.
    acks = Transfer(transfer_model="EH").get_acknowledgments(flat=True)
    assert not any("Smith, R. E." in ref or "Takahashi, R." in ref for ref in acks)


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

    assert mf.get_acknowledgments()["transfer_model"] == (refs.CAMB,)


def test_user_component_references(mf):
    class MyFit(ff.PS):
        references: ClassVar[tuple[str, ...]] = ("Me, A., 2026. Nowhere 1, 1.",)

    acks = mf.clone(hmf_model=MyFit).get_acknowledgments()
    # A subclass's own references replace those it would inherit.
    assert acks["hmf_model"] == ("Me, A., 2026. Nowhere 1, 1.",)


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
    assert list(acks)[:2] == ["hmf", "Child"]
    assert acks["Child"] == ("Child, C., 2020.",)


def test_custom_cosmology_without_reference():
    from astropy.cosmology import FlatLambdaCDM

    acks = Cosmology(cosmo_model=FlatLambdaCDM(H0=70, Om0=0.3)).get_acknowledgments()
    assert acks == {"hmf": (HMF_REFERENCE,), "cosmo_model": ()}


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
    acks = _Outer().get_acknowledgments(flat=True)
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
    # Sub-frameworks are visited in alphabetical order, so the shared one is
    # listed once, under the first name.
    assert _Shared().get_acknowledgments() == {
        "hmf": (HMF_REFERENCE,),
        "also_inner": ("Inner, I., 2000.",),
        "also_inner.sub_model": ("Sub, S., 2010.",),
    }


def test_component_without_references():
    class NoRefs(Component):
        pass

    assert NoRefs.references == ()
    assert _Inner(sub_model=NoRefs).get_acknowledgments() == {
        "hmf": (HMF_REFERENCE,),
        "_Inner": ("Inner, I., 2000.",),
        "sub_model": (),
    }


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

    # Only hmf's own models: other packages (e.g. halomod, which hmf imports for
    # mass conversions) register their components here too.
    missing = [
        f"{base.__name__}.{name}"
        for base in get_base_components()
        for name, model in getattr(base, "_plugins", {}).items()
        if model.__module__.startswith("hmf.")
        and not model.references
        and name not in _UNCITED_MODELS.get(base.__name__, set())
    ]
    assert missing == []
