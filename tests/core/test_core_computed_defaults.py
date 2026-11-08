"""Tests of computed defaults under evolve.

A mass function's sigma_8 and n_s default to the cosmology's (its ``meta``). They are
computed from the cosmology alone (``computed_defaults_inputs``), so an ``evolve`` that
changes the cosmology takes both from the new one, unless they are given in the same
call; any other change leaves them as they are.
"""

import copy
import pickle
from typing import ClassVar

import attrs
import numpy as np
import pytest
from astropy.cosmology import WMAP7, WMAP9, FlatLambdaCDM, Planck13, Planck15, Planck18

from hmf.core import config, field, routing
from hmf.core.mass_function import MassFunction
from hmf.core.stage import Stage, undeclared_inputs
from hmf.core.transfer import Transfer
from hmf.core.units import Msun_h

EH = {"transfer.model": "EH"}
M = np.logspace(10, 15, 11) * Msun_h
Z = np.array([0.0, 1.0])[:, None]
#: The flat names of the parameters a mass function computes from the cosmology.
DEFAULTED = ("sigma_8", "n_s")
#: A cosmology whose meta has neither sigma8 nor n.
BARE = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)


@attrs.frozen(kw_only=True)
class HaloModel(Stage):
    """A toy extension stage holding a mass function."""

    mass_function: MassFunction = field(
        factory=lambda: MassFunction.build(**EH), doc="The mass function."
    )
    bias: str = field(default="Tinker10", doc="The bias model.")


@pytest.fixture(scope="module")
def default():
    """The default mass function, with the fast transfer model."""
    return MassFunction.build(**EH)


def _dndm(mf: MassFunction) -> np.ndarray:
    return mf.dndm(m=M[None, :], z=Z).value


def _sigma_8(mf: MassFunction) -> float:
    return mf.linear_power.sigma_8


def _n_s(mf: MassFunction) -> float:
    return mf.linear_power.transfer.n_s


def _rebuilt(mf: MassFunction, **changes) -> MassFunction:
    """``mf``'s parameters, without those computed from the cosmology, and ``changes``."""
    flat = {k: v for k, v in mf.to_flat().items() if k not in DEFAULTED}
    return MassFunction.from_flat({**flat, **changes})


@pytest.mark.parametrize("cosmology", [Planck13, Planck15, WMAP7, WMAP9])
def test_evolve_cosmology_equals_build(default, cosmology):
    built = MassFunction.build(cosmology=cosmology, **EH)
    evolved = default.evolve(cosmology=cosmology)
    # The defaults are the cosmology's own (astropy's realisations give them).
    assert _sigma_8(built) == cosmology.meta["sigma8"]
    assert _n_s(built) == cosmology.meta["n"]
    assert evolved == built
    assert MassFunction.from_flat({**EH, "cosmology": cosmology}) == built
    assert np.array_equal(_dndm(evolved), _dndm(built))
    # From any starting cosmology, and back.
    assert MassFunction.build(cosmology=Planck15, **EH).evolve(cosmology=cosmology) == built
    assert evolved.evolve(cosmology=Planck18) == default


@pytest.mark.parametrize(
    "start",
    [
        lambda: MassFunction.build(sigma_8=0.7, n_s=0.95, **EH),
        lambda: MassFunction.build(**EH).evolve(sigma_8=0.7, n_s=0.95),
        lambda: MassFunction.build(cosmology=WMAP9, **EH).evolve(sigma_8=0.7),
    ],
)
def test_a_cosmology_change_takes_the_new_cosmologys_defaults(start):
    """However sigma_8 and n_s were set before, the new cosmology gives them."""
    mf = start()
    new = mf.evolve(cosmology=Planck15)
    assert _sigma_8(new) == Planck15.meta["sigma8"]
    assert _n_s(new) == Planck15.meta["n"]
    assert new == _rebuilt(mf, cosmology=Planck15)
    assert new == MassFunction.build(cosmology=Planck15, **EH)


def test_values_given_with_the_cosmology_are_kept(default):
    new = default.evolve(cosmology=Planck15, sigma_8=0.7)
    assert _sigma_8(new) == 0.7
    assert _n_s(new) == Planck15.meta["n"]
    assert new == MassFunction.build(cosmology=Planck15, sigma_8=0.7, **EH)
    both = default.evolve(cosmology=Planck15, sigma_8=0.7, n_s=0.95)
    assert (_sigma_8(both), _n_s(both)) == (0.7, 0.95)


def test_other_changes_compute_no_default(default):
    """Only the cosmology's change recomputes: n_s, set before, survives a sigma_8 change."""
    mf = default.evolve(n_s=0.95)
    lower = mf.evolve(sigma_8=0.8)
    assert _n_s(lower) == 0.95
    assert lower.linear_power.transfer is mf.linear_power.transfer
    assert _sigma_8(mf.evolve(n_s=0.9)) == _sigma_8(mf)
    new = default.evolve(fit="ST")
    assert new.linear_power is default.linear_power


def test_a_cosmology_with_the_same_defaults_rebuilds_only_the_cosmology(default):
    # Same parameters and meta: an equal tree.
    assert default.evolve(cosmology=Planck18.clone(H0=Planck18.H0)) == default
    # The defaults are recomputed but equal the old ones: only the cosmology changes.
    other = Planck18.clone(Om0=0.32)
    new = default.evolve(cosmology=other)
    assert (_sigma_8(new), _n_s(new)) == (_sigma_8(default), _n_s(default))


def test_a_transfer_stage_given_whole(default):
    """A stage holding the cosmology is a change to it; n_s is the stage's own."""
    transfer = Transfer(model="EH", cosmology=Planck15, n_s=0.95)
    evolved = default.evolve(**{"linear_power.transfer": transfer})
    assert evolved == MassFunction.from_flat({"linear_power.transfer": transfer})
    assert _sigma_8(evolved) == Planck15.meta["sigma8"]
    assert _n_s(evolved) == 0.95


def test_a_linear_power_given_whole_is_kept(default):
    lp = default.evolve(sigma_8=0.7, cosmology=Planck15).linear_power
    new = default.evolve(linear_power=lp)
    assert new.linear_power is lp
    assert _sigma_8(new) == 0.7


def test_evolve_without_a_source_raises(default):
    with pytest.raises(ValueError, match=r"'FlatLambdaCDM' does not give n_s.*set n_s explicitly"):
        default.evolve(cosmology=BARE)
    with pytest.raises(ValueError, match=r"does not give sigma_8.*set sigma_8 explicitly"):
        default.evolve(cosmology=BARE, n_s=0.96)
    # Values set before do not help: the new cosmology gives the defaults.
    with pytest.raises(ValueError, match="does not give n_s"):
        default.evolve(sigma_8=0.8, n_s=0.96).evolve(cosmology=BARE)
    named = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05, name="mine")
    with pytest.raises(ValueError, match="the cosmology 'mine' does not give n_s"):
        default.evolve(cosmology=named)
    # build raises the same, and the original is unchanged.
    with pytest.raises(ValueError, match=r"does not give sigma_8.*set sigma_8 explicitly"):
        MassFunction.build(cosmology=BARE, n_s=0.96, **EH)
    assert _sigma_8(default) == Planck18.meta["sigma8"]


def test_evolve_without_a_source_works_when_given():
    mf = MassFunction.build(**EH)
    new = mf.evolve(cosmology=BARE, sigma_8=0.8, n_s=0.96)
    assert new == MassFunction.build(cosmology=BARE, sigma_8=0.8, n_s=0.96, **EH)
    # And from it, any other change computes nothing (BARE gives no defaults).
    assert _sigma_8(new.evolve(fit="ST")) == 0.8
    # Back to a cosmology with defaults.
    assert new.evolve(cosmology=Planck15) == MassFunction.build(cosmology=Planck15, **EH)


@pytest.mark.parametrize(
    "copier",
    [
        lambda s: pickle.loads(pickle.dumps(s)),
        copy.deepcopy,
        lambda s: config.from_config(config.to_config(s)),
    ],
)
def test_evolve_depends_only_on_the_value(copier):
    """No hidden state: a copy, a pickle or a configuration evolves as the original."""
    mf = MassFunction.build(sigma_8=0.7, **EH)
    same = copier(mf)
    assert same == mf
    assert same.evolve(cosmology=Planck15) == mf.evolve(cosmology=Planck15)
    assert same.evolve(sigma_8=0.8) == mf.evolve(sigma_8=0.8)


def test_extension_stage_recomputes_the_mass_functions_defaults(default):
    built = HaloModel.from_flat({**EH, "cosmology": Planck15})
    assert built.mass_function == MassFunction.build(cosmology=Planck15, **EH)
    assert HaloModel.from_flat(EH).evolve(cosmology=Planck15) == built
    held = HaloModel(mass_function=default.evolve(sigma_8=0.7))
    assert held.evolve(cosmology=Planck15) == built
    assert _sigma_8(held.evolve(bias="SMT").mass_function) == 0.7


def test_extension_stage_with_a_mass_function_given_whole(default):
    explicit = default.evolve(sigma_8=0.7, cosmology=Planck15)
    hm = HaloModel.from_flat({"mass_function": default})
    assert hm.mass_function is default
    # Replacing the stage whole computes nothing in it.
    replaced = hm.evolve(mass_function=explicit)
    assert replaced.mass_function is explicit


def test_mass_function_declares_its_inputs(default):
    assert MassFunction.computed_defaults_inputs == ("linear_power.transfer.cosmology",)
    assert undeclared_inputs(default) == {}


# ---------------------------------------------------------------------------------
# Toy trees: shared and optional stages with computed defaults, and bad declarations
# ---------------------------------------------------------------------------------


@attrs.frozen(kw_only=True)
class Part(Stage):
    """A toy stage with a setting."""

    acc: int = field(default=1, doc="A setting.")


@attrs.frozen(kw_only=True)
class Pair(Stage):
    """A toy stage sharing its parts' setting, whose default is computed from ``base``."""

    shared_parameters: ClassVar = frozenset({"acc"})
    computed_defaults_inputs: ClassVar = ("base",)

    left: Part = field(factory=Part, doc="One.")
    right: Part = field(factory=Part, doc="Two.")
    base: int = field(default=3, doc="What acc's default is computed from.")

    @classmethod
    def computed_defaults(cls, given):
        return {"acc": 2 * given.get("base", 3)}


@attrs.frozen(kw_only=True)
class Top(Stage):
    """A toy stage holding an optional Pair."""

    pair: Pair | None = field(default=None, doc="An optional pair.")
    other: int = field(default=0, doc="Something else.")


def test_toy_shared_computed_default_is_recomputed():
    built = Pair.from_flat({})
    assert built.left.acc == built.right.acc == 6
    new = built.evolve(base=4)
    assert new.left.acc == new.right.acc == 8
    assert new == Pair.from_flat({"base": 4})
    # Set by its shared name, then recomputed by a change of base.
    assert built.evolve(acc=5).left.acc == 5
    assert built.evolve(acc=5).evolve(base=4) == new
    assert built.evolve(acc=5, base=4).right.acc == 5
    assert undeclared_inputs(built) == {}


def test_toy_optional_stage():
    top = Top.from_flat({})
    assert top.pair is None
    assert top.evolve(acc=5).pair is None  # a shared name reaching only an absent stage
    with pytest.raises(ValueError, match="pair is None"):
        top.evolve(**{"pair.base": 5})  # an input of an absent stage: nothing to recompute
    with_pair = Top.from_flat({"pair.base": 2})
    assert with_pair.pair.left.acc == 4
    assert with_pair.evolve(**{"pair.base": 5}).pair == Pair.from_flat({"base": 5})


def test_undeclared_inputs_of_computed_defaults():
    @attrs.frozen(kw_only=True)
    class Undeclared(Stage):
        a: int = field(default=1, doc="Read, not declared.")
        b: int = field(default=2, doc="Computed.")
        c: int = field(default=0, doc="Declared, not read.")
        computed_defaults_inputs: ClassVar = ("c",)

        @classmethod
        def computed_defaults(cls, given):
            return {} if "b" in given else {"b": 2 * given.get("a", 1)}

    @attrs.frozen(kw_only=True)
    class Iterates(Stage):
        a: int = field(default=1, doc="A parameter.")
        computed_defaults_inputs: ClassVar = ("a",)

        @classmethod
        def computed_defaults(cls, given):
            return {"a": len(given)}

    assert undeclared_inputs(Undeclared()) == {"computed_defaults": ("a",)}
    assert undeclared_inputs(Iterates()) == {"computed_defaults": ("*",)}
    # An undeclared input leaves the default stale: what the check is for.
    assert Undeclared.from_flat({}).evolve(a=5).b == 2


def test_computed_defaults_need_declared_inputs():
    @attrs.frozen(kw_only=True)
    class NoInputs(Stage):
        a: int = field(default=1, doc="A parameter.")

        @classmethod
        def computed_defaults(cls, given):
            return {"a": 2}

    with pytest.raises(TypeError, match="NoInputs defines computed_defaults but no"):
        NoInputs.from_flat({})
    with pytest.raises(TypeError, match="NoInputs defines computed_defaults but no"):
        NoInputs().evolve(a=3)


def test_declared_input_must_be_a_field():
    @attrs.frozen(kw_only=True)
    class BadInput(Stage):
        a: int = field(default=1, doc="A parameter.")
        computed_defaults_inputs: ClassVar = ("nope",)

        @classmethod
        def computed_defaults(cls, given):
            return {}

    with pytest.raises(TypeError, match=r"BadInput.computed_defaults_inputs: 'nope'"):
        BadInput.parameter_names()


def test_base_stage_computes_no_defaults():
    assert Stage.computed_defaults(routing.GivenParameters({})) == {}
    assert routing.router(Part).computing == {}
    assert undeclared_inputs(Part()) == {}
