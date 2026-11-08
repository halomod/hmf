"""Tests of computed defaults under evolve: build(**a).evolve(**b) == build(**a, **b).

A mass function's sigma_8 and n_s default to the cosmology's (its ``meta``). A
parameter that took such a default is computed again when ``evolve`` changes what it is
computed from; one set explicitly keeps its value.
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
from hmf.core.routing import ComputedRecord
from hmf.core.stage import Stage
from hmf.core.transfer import Transfer
from hmf.core.units import Msun_h

EH = {"transfer.model": "EH"}
M = np.logspace(10, 15, 11) * Msun_h
Z = np.array([0.0, 1.0])[:, None]
COMPUTED = ("linear_power.sigma_8", "linear_power.transfer.n_s")
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


@pytest.mark.parametrize("cosmology", [Planck13, Planck15, WMAP7, WMAP9])
def test_evolve_cosmology_equals_build(default, cosmology):
    built = MassFunction.build(cosmology=cosmology, **EH)
    evolved = default.evolve(cosmology=cosmology)
    flat = MassFunction.from_flat({**EH, "cosmology": cosmology})
    # The defaults are the cosmology's own (astropy's realisations give them).
    assert built.linear_power.sigma_8 == cosmology.meta["sigma8"]
    assert built.linear_power.transfer.n_s == cosmology.meta["n"]
    assert evolved == built
    assert flat == built
    assert evolved.computed_parameters() == built.computed_parameters() == COMPUTED
    assert np.array_equal(_dndm(evolved), _dndm(built))
    assert np.array_equal(_dndm(flat), _dndm(built))
    # And from a non-default cosmology, and back.
    other = MassFunction.build(cosmology=Planck15, **EH).evolve(cosmology=cosmology)
    assert other == built
    assert evolved.evolve(cosmology=Planck18) == default


def test_evolve_transfer_stage_given_whole_equals_build(default):
    transfer = Transfer(model="EH", cosmology=Planck15, n_s=0.95)
    evolved = default.evolve(**{"linear_power.transfer": transfer})
    built = MassFunction.from_flat({"linear_power.transfer": transfer})
    assert evolved == built
    assert evolved.linear_power.sigma_8 == Planck15.meta["sigma8"]
    assert evolved.linear_power.transfer.n_s == 0.95
    # n_s, in the stage given whole, is explicit; sigma_8 is still computed.
    assert evolved.computed_parameters() == built.computed_parameters()
    assert evolved.computed_parameters() == ("linear_power.sigma_8",)


@pytest.mark.parametrize(
    "start",
    [
        lambda: MassFunction.build(sigma_8=0.7, **EH),
        lambda: MassFunction.build(**EH).evolve(sigma_8=0.7),
        lambda: MassFunction.build(**EH).evolve(sigma_8=0.7).evolve(cosmology=WMAP9),
    ],
)
def test_explicit_sigma_8_survives_a_cosmology_change(start):
    new = start().evolve(cosmology=Planck15)
    assert new.linear_power.sigma_8 == 0.7
    # n_s was not set, so it is still the cosmology's.
    assert new.linear_power.transfer.n_s == Planck15.meta["n"]
    assert new.computed_parameters() == ("linear_power.transfer.n_s",)
    assert new == MassFunction.build(sigma_8=0.7, cosmology=Planck15, **EH)


def test_explicit_value_equal_to_the_default_is_kept(default):
    # Set to the value it had: still explicit from then on.
    pinned = default.evolve(sigma_8=default.linear_power.sigma_8)
    assert pinned.evolve(cosmology=Planck15).linear_power.sigma_8 == Planck18.meta["sigma8"]


def test_a_stage_given_whole_is_explicit(default):
    lp = default.linear_power
    replaced = default.evolve(linear_power=lp)
    assert replaced.computed_parameters() == ()
    new = replaced.evolve(cosmology=Planck15)
    assert new.linear_power.sigma_8 == Planck18.meta["sigma8"]
    assert new.linear_power.transfer.n_s == Planck18.meta["n"]


def test_evolve_without_a_source_raises_when_computed(default):
    with pytest.raises(ValueError, match=r"'FlatLambdaCDM' does not give n_s.*set n_s explicitly"):
        default.evolve(cosmology=BARE)
    with pytest.raises(ValueError, match=r"does not give sigma_8.*set sigma_8 explicitly"):
        default.evolve(n_s=0.96).evolve(cosmology=BARE)
    # build raises the same.
    with pytest.raises(ValueError, match=r"does not give sigma_8.*set sigma_8 explicitly"):
        MassFunction.build(cosmology=BARE, n_s=0.96, **EH)
    named = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05, name="mine")
    with pytest.raises(ValueError, match="the cosmology 'mine' does not give n_s"):
        default.evolve(cosmology=named)
    # The original is unchanged.
    assert default.linear_power.sigma_8 == Planck18.meta["sigma8"]


def test_evolve_without_a_source_works_when_explicit():
    mf = MassFunction.build(sigma_8=0.8, n_s=0.96, **EH)
    new = mf.evolve(cosmology=BARE)
    assert new.linear_power.sigma_8 == 0.8
    assert new.linear_power.transfer.n_s == 0.96
    assert new == MassFunction.build(cosmology=BARE, sigma_8=0.8, n_s=0.96, **EH)


def test_provenance_is_not_part_of_the_value(default):
    explicit = MassFunction.build(
        sigma_8=default.linear_power.sigma_8, n_s=default.linear_power.transfer.n_s, **EH
    )
    assert explicit.computed_parameters() == ()
    assert default.computed_parameters() == COMPUTED
    assert explicit == default
    assert hash(explicit) == hash(default)
    assert config.content_hash(explicit) == config.content_hash(default)
    assert config.to_config(explicit) == config.to_config(default)
    assert "_computed" not in repr(default)
    assert "_computed" not in MassFunction.parameter_names()


@pytest.mark.parametrize("copier", [lambda s: pickle.loads(pickle.dumps(s)), copy.deepcopy])
def test_pickling_keeps_provenance(copier):
    mf = MassFunction.build(sigma_8=0.7, **EH)
    same = copier(mf)
    assert same == mf
    assert same.computed_parameters() == mf.computed_parameters()
    assert same.linear_power._computed == mf.linear_power._computed
    assert same._computed == mf._computed
    assert same.evolve(cosmology=Planck15) == mf.evolve(cosmology=Planck15)


def test_a_configuration_records_no_provenance(default):
    read = config.from_config(config.to_config(default))
    assert read == default
    assert read.computed_parameters() == ()
    # So every value it gives is kept.
    assert read.evolve(cosmology=Planck15).linear_power.sigma_8 == Planck18.meta["sigma8"]
    # A value a partial configuration leaves out takes a computed default, and stays one.
    partial = config.from_config(
        {"hmf": {"schema_version": 1}, "sigma_8": 0.8, **EH}, stage=MassFunction
    )
    assert partial.computed_parameters() == ("linear_power.transfer.n_s",)


def test_evolve_own_makes_the_fields_it_sets_explicit(default):
    assert default.evolve_own(delta_c=1.7).computed_parameters() == COMPUTED
    assert default.evolve_own(linear_power=default.linear_power).computed_parameters() == ()


def test_unrelated_changes_share_the_stages(default):
    new = default.evolve(fit="ST")
    assert new.linear_power is default.linear_power
    assert new.computed_parameters() == COMPUTED
    lower = default.evolve(sigma_8=0.7)
    assert lower.linear_power.transfer is default.linear_power.transfer
    assert lower.computed_parameters() == ("linear_power.transfer.n_s",)
    # A cosmology change with the same sigma_8 and n_s rebuilds only what it must.
    same_meta = Planck18.clone(H0=Planck18.H0)
    assert default.evolve(cosmology=same_meta) == default


def test_extension_stage_recomputes_the_mass_functions_defaults(default):
    built = HaloModel.from_flat({**EH, "cosmology": Planck15})
    assert built.mass_function == MassFunction.build(cosmology=Planck15, **EH)
    evolved = HaloModel.from_flat(EH).evolve(cosmology=Planck15)
    assert evolved == built
    # A mass function given whole brings its own record.
    held = HaloModel(mass_function=default).evolve(cosmology=Planck15)
    assert held == built
    assert held.computed_parameters() == (
        "mass_function.linear_power.sigma_8",
        "mass_function.linear_power.transfer.n_s",
    )
    explicit = HaloModel(mass_function=default.evolve(sigma_8=0.7)).evolve(cosmology=Planck15)
    assert explicit.mass_function.linear_power.sigma_8 == 0.7


def test_computed_record():
    record = ComputedRecord(paths=frozenset({("a", "b"), ("c",)}), reads=frozenset({("d",)}))
    assert record.explicit([("a",)]).paths == {("c",)}
    assert record.explicit([("e",)]) is record
    assert record.explicit([("c",)]).reads == record.reads


def test_extension_stage_with_a_mass_function_given_whole(default):
    explicit = default.evolve(sigma_8=0.7)
    built = HaloModel.from_flat({"mass_function": default})
    assert built.computed_parameters() == tuple(f"mass_function.{p}" for p in COMPUTED)
    # Replacing the stage whole brings the new stage's own record, untouched.
    replaced = built.evolve(mass_function=explicit)
    assert replaced.mass_function is explicit
    assert replaced.computed_parameters() == ("mass_function.linear_power.transfer.n_s",)


# ---------------------------------------------------------------------------------
# Toy trees: shared and optional stages with computed defaults
# ---------------------------------------------------------------------------------


@attrs.frozen(kw_only=True)
class Part(Stage):
    """A toy stage with a setting."""

    acc: int = field(default=1, doc="A setting.")


@attrs.frozen(kw_only=True)
class Pair(Stage):
    """A toy stage sharing its parts' setting, whose default is computed from ``base``."""

    shared_parameters: ClassVar = frozenset({"acc"})

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
    assert new.computed_parameters() == ("left.acc", "right.acc")
    # Set by its shared name: explicit for both holders.
    pinned = built.evolve(acc=5)
    assert pinned.computed_parameters() == ()
    assert pinned.evolve(base=4).left.acc == 5


def test_toy_optional_stage_left_out_has_no_record():
    top = Top.from_flat({})
    assert top.pair is None
    assert top.computed_parameters() == ()
    # A shared name that only reaches the absent stage changes nothing.
    assert top.evolve(acc=5).pair is None
    with_pair = Top.from_flat({"pair.base": 2})
    assert with_pair.pair.left.acc == 4
    assert with_pair.evolve(**{"pair.base": 5}).pair == Pair.from_flat({"base": 5})


def test_tree_parameters_view_of_an_evolving_tree(default):
    computed = frozenset({("linear_power", "sigma_8")})
    given = routing._TreeParameters(default, {("delta_c",): 1.7}, computed)
    assert given["delta_c"] == 1.7
    assert given["linear_power.transfer.n_s"] == default.linear_power.transfer.n_s
    assert given["fit.A_200"] == default.fit.A_200  # into a model field
    assert "linear_power.sigma_8" not in given  # still computed
    assert "linear_power" not in given  # a stage, not changed whole
    assert "fit.nope" not in given
    names = list(given)
    assert "delta_c" in names
    assert "linear_power.sigma_8" not in names
    assert len(given) == len(names)
    assert () in given.reads


def test_same_value_when_eq_cannot_answer():
    a = np.array([1.0, 2.0])
    assert routing._same_value(a, a)
    assert not routing._same_value(a, np.array([1.0, 3.0]))
    assert routing._same_value(1, 1.0)


def test_base_stage_computes_no_defaults():
    assert Stage.computed_defaults(routing.GivenParameters({})) == {}
    assert routing.router(Part).computing == ()
