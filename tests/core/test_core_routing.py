"""Tests of hmf.core.routing: parameters by name through a stage tree, and introspection."""

from functools import cached_property
from typing import ClassVar

import attrs
import numpy as np
import pytest
from astropy.cosmology import Planck15, Planck18

from hmf.core import _boltzmann, field, routing
from hmf.core import stage as stage_module
from hmf.core.accuracy import KAccuracy, MassAccuracy
from hmf.core.growth import Growth
from hmf.core.linear_power import LinearPower
from hmf.core.mass_function import MassFunction
from hmf.core.mass_variance import MassVariance
from hmf.core.routing import GivenParameters
from hmf.core.stage import Stage, derived_property
from hmf.core.transfer import Transfer
from hmf.core.transfer_models import FromArray
from hmf.core.units import Msun_h, h_Mpc

M = np.logspace(10, 15, 6) * Msun_h

#: The parameter names of a MassFunction: build's keywords before routing, with
#: "transfer.model" for transfer_model, and the variance's truncation_rtol.
BUILD_KEYWORDS = {
    "cosmology",
    "transfer.model",
    "growth_model",
    "n_s",
    "sigma_8",
    "species",
    "sigma_8_species",
    "fit",
    "filter",
    "delta_c",
    "domain_policy",
    "k_accuracy",
    "mass_accuracy",
    "disk_cache",
    "truncation_rtol",
    "mass_definition",
    "mass_conversion",
}


@pytest.fixture(scope="module")
def mf():
    """A fast mass function (Eisenstein & Hu), with the quantities of the tests cached."""
    stage = MassFunction.build(**{"transfer.model": "EH"})
    stage.dndm(m=M, z=0.0)
    return stage


def _stage_classes():
    """Every Stage subclass defined so far."""
    out, todo = set(), [Stage]
    while todo:
        cls = todo.pop()
        for sub in cls.__subclasses__():
            if sub not in out:
                out.add(sub)
                todo.append(sub)
    return out


# ---------------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------------


def test_parameter_names_are_build_keywords():
    """build()'s keywords and evolve()'s flat names are the same set."""
    assert set(MassFunction.parameter_names()) == BUILD_KEYWORDS
    # build accepts every one of them.
    assert MassFunction.build(**MassFunction.parameter_defaults()) == MassFunction.build()


def test_parameter_info():
    info = MassFunction.parameter_info()
    assert info["sigma_8"].path == "linear_power.sigma_8"
    assert not info["sigma_8"].shared
    assert info["sigma_8"].required  # computed from the cosmology by build
    assert info["transfer.model"].paths == ("linear_power.transfer.model",)
    assert "model" not in info  # Transfer lists it qualified ...
    # ... but a bare model still routes to it while it is unique.
    assert routing.router(MassFunction).paths_of("model")[0] == (
        ("linear_power", "transfer", "model"),
    )
    # A transfer stage on its own has nothing to qualify it with.
    assert Transfer.parameter_names()[1] == "model"
    assert info["growth_model"].paths == ("linear_power.growth_model",)
    assert info["filter"].path == "filter"
    assert info["delta_c"].default == 1.686
    assert info["delta_c"].type is float
    assert "critical overdensity" in info["delta_c"].doc
    # Each value lives in one stage: the growth and variance take them from the transfer.
    assert info["cosmology"].paths == ("linear_power.transfer.cosmology",)
    assert info["cosmology"].default is Planck18
    assert info["k_accuracy"].paths == ("linear_power.transfer.k_accuracy",)
    assert info["disk_cache"].paths == ("linear_power.transfer.disk_cache",)
    assert not any(i.shared for i in info.values())


def test_shared_parameters_of_a_growth_stage():
    """Growth declares the names it holds twice (with a transfer stage) as shared."""
    info = Growth.parameter_info()
    assert info["cosmology"].shared
    assert set(info["cosmology"].paths) == {"cosmology", "transfer.cosmology"}
    assert set(info["k_accuracy"].paths) == {"k_accuracy", "transfer.k_accuracy"}
    assert set(info["disk_cache"].paths) == {"disk_cache", "transfer.disk_cache"}
    # The growth model is the root's own field; the transfer's is reached by its path.
    assert info["model"].paths == ("model",)
    assert "transfer.model" in info


def test_parameter_defaults():
    defaults = MassFunction.parameter_defaults()
    assert list(defaults) == list(MassFunction.parameter_names())
    assert defaults["sigma_8"] == Planck18.meta["sigma8"]
    assert defaults["n_s"] == Planck18.meta["n"]
    assert defaults["k_accuracy"] == KAccuracy()
    assert defaults["fit"].__class__.__name__ == "Tinker08"
    assert MassFunction.from_flat(defaults) == MassFunction.from_flat({}) == MassFunction.build()
    # A required parameter without a computed default is left out.
    assert "sigma_8" not in LinearPower.parameter_defaults()


def test_every_field_is_routable():
    """Every field of every stage of the MassFunction tree is a parameter, or a stage."""
    info = MassFunction.parameter_info()
    routed = {path for i in info.values() for path in i.paths}
    table = routing.router(MassFunction)
    fields = set()
    for path, cls in table.stages.items():
        for a in attrs.fields(cls):
            full = (*path, a.name)
            if full not in table.stages:
                fields.add(".".join(full))
    assert fields == routed
    assert set(table.stages) == {(), ("linear_power",), ("linear_power", "transfer")}
    # The derived properties are known from the classes, with their stage classes.
    assert {".".join(p) for p in table.derived} == {
        "variance",
        "linear_power.growth",
        "linear_power.unnormalised_sigma_8",
    }
    assert table.derived_stages == {
        ("variance",): MassVariance,
        ("linear_power", "growth"): Growth,
    }


def test_dotted_paths_and_flat_names_route_alike(mf):
    for name, info in MassFunction.parameter_info().items():
        if info.shared:
            continue
        assert routing.router(MassFunction).paths_of(info.path)[0] == (tuple(info.path.split(".")),)
        assert routing.router(MassFunction).paths_of(name)[0] == (tuple(info.path.split(".")),)


# ---------------------------------------------------------------------------------
# evolve through every parameter: the stale-dependency guarantee
# ---------------------------------------------------------------------------------

#: A new value of each parameter, and the quantity that must change with it: a
#: (method, z) of the mass function, or None for a parameter that changes no result
#: here (checked to reach its field instead).
CHANGES = {
    "cosmology": (Planck15, ("dndm", 0.0)),
    "transfer.model": ("BBKS", ("dndm", 0.0)),
    "n_s": (0.95, ("dndm", 0.0)),
    "k_accuracy": (KAccuracy.fast(), ("dndm", 0.0)),
    # Where to cache Boltzmann runs: EH makes none, and the cache changes no result.
    "disk_cache": (None, None),
    "growth_model": ("Carroll92", ("dndm", 1.0)),
    "sigma_8": (0.7, ("dndm", 0.0)),
    # On the tree of _two_species(), whose species differ (EH's are the same).
    "species": ("tot", ("dndm", 0.0)),
    "sigma_8_species": ("cb", ("dndm", 0.0)),
    "filter": ("SharpK", ("dndm", 0.0)),
    "mass_accuracy": (MassAccuracy(log10_m_max=16.0), ("ngtm", 0.0)),
    # A tighter bound on the truncation error: no mass here is near it.
    "truncation_rtol": (1e-4, None),
    "fit": ("ST", ("dndm", 0.0)),
    # Tinker08's f(sigma) does not depend on delta_c; the peak height does.
    "delta_c": (1.6, ("peak_height", 0.0)),
    # z = 4 is outside Tinker08's calibration (z <= 2.5): masked to NaN.
    "domain_policy": ("mask", ("dndm", 4.0)),
    # Tinker08 accepts every SO definition: its overdensity changes, nothing is converted.
    "mass_definition": ("200c", ("dndm", 0.0)),
    # Tinker08 accepts the (default) mass definition, so nothing is converted.
    "mass_conversion": ("profile", None),
}


def test_changes_cover_every_parameter():
    assert set(CHANGES) == set(MassFunction.parameter_names())


def _two_species(mf):
    """``mf`` with a tabulated transfer function whose total matter is suppressed at small scales.

    EH has one transfer function for every species, so the species parameters change no
    result with it.
    """
    k = np.logspace(-9, 5, 600)
    t = np.asarray(Transfer(model="EH").transfer_function(k * h_Mpc))
    return mf.evolve(model=FromArray(k=k * h_Mpc, t=t, t_tot=t * (1 - 0.1 * k / (k + 1))))


@pytest.mark.filterwarnings("ignore::hmf.exceptions.HMFExtrapolationWarning")
@pytest.mark.parametrize("name", sorted(CHANGES))
def test_evolve_every_parameter(mf, tmp_path, name):
    """Each parameter, by its flat name, changes what depends on it, and gives a fresh build."""
    value, quantity = CHANGES[name]
    if name == "disk_cache":
        value = tmp_path
    if name in ("species", "sigma_8_species"):
        mf = _two_species(mf)
    new = mf.evolve(**{name: value})
    for path in MassFunction.parameter_info()[name].paths:
        got, old = new, mf
        for part in path.split("."):
            got, old = getattr(got, part), getattr(old, part)
        assert got is not old, path
    # A fresh tree, with every other parameter the same.
    fresh = MassFunction.from_flat({**mf.to_flat(), name: value})
    assert new == fresh
    if quantity is not None:
        method, z = quantity
        before = np.asarray(getattr(mf, method)(m=M, z=z))
        after = np.asarray(getattr(new, method)(m=M, z=z))
        assert not np.array_equal(before, after, equal_nan=True)
        np.testing.assert_array_equal(after, np.asarray(getattr(fresh, method)(m=M, z=z)))


@pytest.mark.parametrize(
    "name", sorted(n for n, i in MassFunction.parameter_info().items() if not i.shared)
)
def test_evolve_by_dotted_path(mf, name):
    value, _ = CHANGES[name]
    path = MassFunction.parameter_info()[name].path
    assert mf.evolve(**{path: value}) == mf.evolve(**{name: value})


def test_evolve_nested_mapping(mf):
    nested = mf.evolve(linear_power={"sigma_8": 0.7, "transfer": {"n_s": 0.95}})
    assert nested == mf.evolve(sigma_8=0.7, n_s=0.95)


def test_evolve_nothing_returns_the_stage(mf):
    assert mf.evolve() is mf


def test_evolve_on_a_sub_stage_routes_its_own_tree(mf):
    lp = mf.linear_power.evolve(**{"transfer.model": "BBKS"})
    assert lp.growth.transfer is lp.transfer
    assert lp.transfer.model.__class__.__name__ == "BBKS"
    assert Transfer().evolve(n_s=0.9).n_s == 0.9


# ---------------------------------------------------------------------------------
# Structural sharing, derived properties and shared parameters
# ---------------------------------------------------------------------------------


def test_structural_sharing(mf):
    lower = mf.evolve(sigma_8=0.7)
    assert lower.variance is mf.variance
    assert lower.linear_power.transfer is mf.linear_power.transfer
    assert lower.linear_power.growth is mf.linear_power.growth

    sharp = mf.evolve(filter="SharpK")
    assert sharp.linear_power is mf.linear_power

    st = mf.evolve(fit="ST")
    assert st.linear_power is mf.linear_power
    assert st.variance is mf.variance

    growth = mf.evolve(growth_model="Carroll92")
    assert growth.variance is mf.variance
    assert growth.linear_power.transfer is mf.linear_power.transfer
    assert growth.linear_power.unnormalised_sigma_8 == mf.linear_power.unnormalised_sigma_8


def test_sigma_8_change_carries_the_normalisation(mf, monkeypatch):
    """sigma_8,raw is a derived property of the transfer and sigma_8_species: carried."""
    raw = mf.linear_power.unnormalised_sigma_8
    monkeypatch.setattr(MassVariance, "ln_sigma_at_radius_kernel", None)  # must not run
    lower = mf.evolve(sigma_8=0.7)
    assert lower.linear_power.unnormalised_sigma_8 == raw


def test_cosmology_change_is_consistent(mf):
    new = mf.evolve(cosmology=Planck15)
    lp = new.linear_power
    for stage in (lp, lp.transfer, lp.growth):
        assert stage.cosmology is Planck15
    assert lp.growth.transfer is lp.transfer
    assert new.variance.power.transfer is lp.transfer
    assert new == MassFunction.from_flat({**mf.to_flat(), "cosmology": Planck15})


def test_k_accuracy_change_is_consistent(mf):
    high = KAccuracy.high()
    new = mf.evolve(k_accuracy=high)
    lp = new.linear_power
    for stage in (lp, lp.transfer, lp.growth, new.variance):
        assert stage.k_accuracy == high
    assert new.variance.power.transfer is lp.transfer


def test_derived_stages_are_recomputed(mf):
    new = mf.evolve(**{"transfer.model": "BBKS"})
    lp = new.linear_power
    assert new.variance.power.transfer is lp.transfer
    assert lp.growth.transfer is lp.transfer
    assert new.variance.power == lp.power_source
    # Replacing the linear power whole recomputes the variance of its power ...
    whole = mf.evolve(linear_power=lp)
    assert whole.variance.power.transfer is lp.transfer
    # ... and carries it when the transfer and species do not change.
    assert mf.evolve(linear_power=mf.linear_power.evolve(sigma_8=0.7)).variance is mf.variance


def test_species_recomputes_the_variance(mf):
    new = mf.evolve(species="tot")
    assert new.variance.power.species == "tot"
    assert new.variance.power.transfer is mf.linear_power.transfer
    assert new.linear_power.growth is mf.linear_power.growth


def test_derived_values_are_carried_only_once_computed():
    """evolve_own carries a derived value already computed; one not computed is not."""
    fresh = MassFunction.build(**{"transfer.model": "EH"})
    lower = fresh.evolve(sigma_8=0.7)
    variance = lower.variance  # computed on the new stage, not the old one
    assert fresh.evolve(sigma_8=0.7).variance is not variance
    assert lower.evolve(sigma_8=0.8).variance is variance


def test_derived_property_cannot_be_set(mf):
    with pytest.raises(TypeError, match=r"'variance' is computed .*'linear_power.transfer'"):
        mf.evolve(variance=mf.variance)
    with pytest.raises(TypeError, match=r"'variance.power' is computed \(in 'variance'"):
        mf.evolve(**{"variance.power": mf.variance.power})
    with pytest.raises(TypeError, match=r"'linear_power.growth.model' is computed"):
        mf.evolve(**{"linear_power.growth.model": "ODE"})


def test_shared_name_skips_a_stage_given_whole():
    """A stage given whole has its own value of a shared parameter."""
    growth = Growth.from_transfer(Transfer(model="EH"))
    transfer = Transfer(model="EH", cosmology=Planck15)
    new = growth.evolve(transfer=transfer, cosmology=Planck15)
    assert new.cosmology is Planck15
    assert new.transfer is transfer


def test_shared_name_changes_every_holder():
    growth = Growth.from_transfer(Transfer(model="EH"))
    new = growth.evolve(cosmology=Planck15, k_accuracy=KAccuracy.fast())
    assert new.cosmology is new.transfer.cosmology is Planck15
    assert new.k_accuracy == new.transfer.k_accuracy == KAccuracy.fast()


def test_optional_stage():
    """A shared name skips an absent optional stage; other changes to it raise."""
    growth = Growth(model="Carroll92")
    assert growth.transfer is None
    new = growth.evolve(cosmology=Planck15)
    assert new.cosmology is Planck15
    assert new.transfer is None
    with pytest.raises(ValueError, match="transfer is None"):
        growth.evolve(**{"transfer.n_s": 0.9})
    with_transfer = growth.evolve(transfer=Transfer(model="EH"))
    assert with_transfer.evolve(**{"transfer.n_s": 0.9}).transfer.n_s == 0.9
    assert Growth.from_flat({}).transfer is None
    assert Growth.from_flat({"transfer.n_s": 0.9}).transfer.n_s == 0.9


# ---------------------------------------------------------------------------------
# Model fields by dotted path
# ---------------------------------------------------------------------------------


def test_model_field_by_dotted_path(mf):
    st = mf.evolve(fit="ST")
    changed = st.evolve(**{"fit.a": 0.75})
    assert changed.fit.a == 0.75
    assert changed.variance is mf.variance
    assert mf.evolve(fit="ST", **{"fit.a": 0.75}).fit == changed.fit
    growth = mf.evolve(**{"growth_model.dln_a": 0.02})
    assert growth.linear_power.growth.model.dln_a == 0.02
    assert growth.evolve(**{"linear_power.growth_model.dln_a": 0.01}).linear_power.growth_model == (
        mf.linear_power.growth_model
    )
    built = MassFunction.build(**{"transfer.model": "EH"}, fit="ST", **{"fit.a": 0.75})
    assert built.fit == changed.fit
    assert MassFunction.build(**{"transfer.model": "EH"}, **{"fit.A_200": 0.2}).fit.A_200 == 0.2


def test_model_field_errors(mf):
    with pytest.raises(TypeError, match=r"Tinker08 \(at 'fit'\) has no field 'A_20'.*'A_200'"):
        mf.evolve(**{"fit.A_20": 0.2})
    with pytest.raises(TypeError, match="not a model with fields"):
        mf.evolve(**{"sigma_8.x": 1.0})
    with pytest.raises(TypeError, match="given twice"):
        mf.evolve(**{"fit.A_200": 0.2, "fit": mf.fit, "linear_power.sigma_8": 0.7, "sigma_8": 0.7})
    with pytest.raises(TypeError, match="given twice"):
        mf.evolve(**{"growth_model.dln_a": 0.2, "linear_power.growth_model.dln_a": 0.1})
    with pytest.raises(TypeError, match="ambiguous"):
        HaloModel.from_flat({"delta_c.real": 1.0})
    with pytest.raises(TypeError, match="no default to change"):
        LinearPower.from_flat({"transfer.model": "EH", "sigma_8.x": 1.0})
    with pytest.raises(TypeError, match="'delta_c' is a float, not a model"):
        MassFunction.from_flat({"delta_c.real": 1.0})


# ---------------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------------


def test_unknown_name_suggests(mf):
    with pytest.raises(TypeError, match=r"no parameter 'sigma8'.*Did you mean 'sigma_8'"):
        mf.evolve(sigma8=0.8)
    with pytest.raises(
        TypeError, match=r"no parameter 'linear_power.sigma8'.*'linear_power.sigma_8'"
    ):
        mf.evolve(**{"linear_power.sigma8": 0.8})
    with pytest.raises(TypeError, match=r"no parameter 'Mminn'"):
        MassFunction.build(Mminn=3)
    with pytest.raises(TypeError, match=r"no parameter 'tranfer.model'.*'transfer.model'"):
        MassFunction.invalidated_by("tranfer.model")
    with pytest.raises(TypeError, match=r"no parameter 'transfer_model'.*'transfer.model'"):
        MassFunction.build(transfer_model="EH")


def test_ambiguous_name_lists_paths(mf):
    hm = HaloModel(mass_function=mf)
    with pytest.raises(TypeError, match="ambiguous") as err:
        hm.evolve(delta_c=1.6)
    message = str(err.value)
    assert "give one of 'mass_function.delta_c', 'profile.delta_c'" in message


def test_conflicts(mf):
    with pytest.raises(TypeError, match="given twice"):
        mf.evolve(sigma_8=0.7, **{"linear_power.sigma_8": 0.75})
    with pytest.raises(TypeError, match="given twice"):
        mf.evolve(cosmology=Planck15, **{"linear_power.transfer.cosmology": Planck15})
    with pytest.raises(TypeError, match="which is also given"):
        mf.evolve(linear_power=mf.linear_power, sigma_8=0.7)


def test_unknown_names_raise_before_anything_is_built(mf, monkeypatch):
    built = []
    real = routing._rebuild
    monkeypatch.setattr(routing, "_rebuild", lambda *args: built.append(1) or real(*args))
    with pytest.raises(TypeError):
        mf.evolve(sigma_8=0.7, sigma8=0.7)
    with pytest.raises(TypeError):
        MassFunction.build(sigma_8=0.7, fit_a=1.0)
    assert built == []


def test_missing_required_parameter():
    with pytest.raises(TypeError, match=r"missing required parameter\(s\) \['sigma_8'\]"):
        LinearPower.from_flat({"transfer.model": "EH"})
    with pytest.raises(TypeError, match=r"missing required parameter\(s\) \['power'\]"):
        MassVariance.from_flat({})
    lp = LinearPower.from_flat({"transfer.model": "EH", "sigma_8": 0.8})
    assert lp.growth.transfer is lp.transfer


def test_evolve_is_atomic(mf):
    before = mf.dndm(m=M, z=0.0)
    lp, variance = mf.linear_power, mf.variance
    for changes in (
        {"sigma_8": 0.7, "delta_c": -1.0},  # the root fails after linear_power is rebuilt
        {"transfer.model": "BBKS", "filter": "NotAFilter"},
        {"n_s": 0.95, "k_accuracy": "high"},
    ):
        with pytest.raises((ValueError, TypeError, LookupError)):
            mf.evolve(**changes)
    assert mf.linear_power is lp
    assert mf.variance is variance
    assert mf.linear_power.sigma_8 == pytest.approx(Planck18.meta["sigma8"])
    np.testing.assert_array_equal(mf.dndm(m=M, z=0.0), before)


# ---------------------------------------------------------------------------------
# from_flat and round trips
# ---------------------------------------------------------------------------------


def test_round_trip(mf):
    flat = mf.to_flat()
    assert list(flat) == list(MassFunction.parameter_names())
    assert MassFunction.from_flat(flat) == mf
    assert flat["sigma_8"] == mf.linear_power.sigma_8
    assert flat["transfer.model"] == mf.linear_power.transfer.model


def test_from_flat_nested_and_dotted():
    nested = MassFunction.from_flat(
        {"transfer.model": "EH", "linear_power": {"sigma_8": 0.7, "transfer": {"n_s": 0.95}}}
    )
    dotted = MassFunction.from_flat(
        {"transfer.model": "EH", "linear_power.sigma_8": 0.7, "linear_power.transfer.n_s": 0.95}
    )
    assert nested == dotted == MassFunction.build(**{"transfer.model": "EH"}, sigma_8=0.7, n_s=0.95)
    assert nested.linear_power.growth.transfer is nested.linear_power.transfer
    assert nested.variance.power.transfer is nested.linear_power.transfer
    assert MassFunction.from_flat({"linear_power": {"sigma_8": 0.7}}) == MassFunction.build(
        sigma_8=0.7
    )


def test_from_flat_with_a_stage_given_whole(mf):
    """A stage given whole is used; build's computed defaults below it are not."""
    built = MassFunction.from_flat({"linear_power": mf.linear_power, "fit": "ST"})
    assert built.linear_power is mf.linear_power
    assert built.variance.power.transfer is mf.linear_power.transfer
    transfer = Transfer(model="EH", cosmology=Planck15, n_s=0.9)
    lp_built = MassFunction.from_flat({"linear_power.transfer": transfer})
    assert lp_built.linear_power.transfer is transfer
    assert lp_built.linear_power.growth.transfer is transfer
    # sigma_8 comes from the given transfer's cosmology.
    assert lp_built.linear_power.sigma_8 == Planck15.meta["sigma8"]


def test_build_none_is_not_the_default():
    """A parameter left out takes its default; None is a value like any other."""
    with pytest.raises(TypeError):
        MassFunction.build(**{"transfer.model": "EH"}, n_s=None)


def test_given_parameters():
    transfer = Transfer(model="EH", n_s=0.9)
    given = GivenParameters({("linear_power", "transfer"): transfer, ("delta_c",): 1.6})
    assert given["linear_power.transfer.n_s"] == 0.9
    assert given["delta_c"] == 1.6
    assert "linear_power.sigma_8" not in given
    assert "linear_power.transfer.nope" not in given
    assert sorted(given) == ["delta_c", "linear_power.transfer"]
    assert len(given) == 2


# ---------------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------------


def test_invalidated_by():
    assert MassFunction.invalidated_by("sigma_8") == ("", "linear_power")
    assert MassFunction.invalidated_by("filter") == ("", "variance")
    assert MassFunction.invalidated_by("fit") == ("",)
    assert MassFunction.invalidated_by("fit.A_200") == ("",)
    assert MassFunction.invalidated_by("delta_c") == ("",)
    assert MassFunction.invalidated_by("growth_model") == (
        "",
        "linear_power",
        "linear_power.growth",
    )
    assert MassFunction.invalidated_by("sigma_8_species") == (
        "",
        "linear_power",
        "linear_power.unnormalised_sigma_8",
    )
    everything = (
        "",
        "linear_power",
        "variance",
        "linear_power.growth",
        "linear_power.transfer",
        "linear_power.unnormalised_sigma_8",
    )
    assert MassFunction.invalidated_by("model") == everything
    assert MassFunction.invalidated_by("transfer.model") == everything
    assert MassFunction.invalidated_by("n_s") == everything
    assert MassFunction.invalidated_by("cosmology") == everything
    assert MassFunction.invalidated_by("species") == ("", "linear_power", "variance")
    assert MassFunction.invalidated_by("linear_power.transfer") == everything
    with pytest.raises(TypeError, match="computed"):
        MassFunction.invalidated_by("variance")


def test_invalidated_by_matches_evolve(mf):
    """What invalidated_by names is exactly what evolve rebuilds or recomputes."""
    table = routing.router(MassFunction)
    for name, (value, _) in CHANGES.items():
        if name == "disk_cache":
            continue
        new = mf.evolve(**{name: value})
        rebuilt = set()
        for path in (*table.stages, *table.derived):
            a, b = mf, new
            for part in path:
                a, b = getattr(a, part), getattr(b, part)
            if a is not b:
                rebuilt.add(".".join(path))
        assert rebuilt == set(MassFunction.invalidated_by(name)), name


def test_quantities_available():
    q = MassFunction.quantities_available()
    assert set(q) == {
        "",
        "linear_power",
        "linear_power.transfer",
        "linear_power.growth",
        "variance",
    }
    assert {"dndm", "ngtm", "sigma", "fsigma", "m_from_sigma", "m_top"} <= set(q[""])
    # Methods outside the units boundary are not outputs: at() gives a view, and
    # Transfer.power_source() builds a power source.
    assert "at" not in q[""]
    assert "power_source" not in q["linear_power.transfer"]
    assert {"power", "amplitude", "power_source", "unnormalised_sigma_8"} <= set(q["linear_power"])
    assert "growth_factor" in q["linear_power.growth"]
    # A derived stage has its own entry, not one in its parent's.
    assert "variance" not in q[""]
    assert "growth" not in q["linear_power"]
    for names in q.values():
        for name in names:
            assert not name.startswith("_")
            assert not name.endswith("_kernel")
            assert name not in {
                "build",
                "from_flat",
                "evolve",
                "evolve_own",
                "parameter_info",
                "fields_info",
                "same_cosmology",
                "shared_parameters",
            }
            assert not name.isupper()
    # Fields are parameters, not quantities.
    assert "sigma_8" not in q["linear_power"]
    assert "delta_c" not in q[""]


def test_introspection_constructs_nothing(monkeypatch):
    """No introspection call constructs a stage or runs a Boltzmann code (#383).

    The routing tables are cleared first, so that building them is counted too.
    """
    monkeypatch.setattr(routing, "_ROUTERS", {})
    monkeypatch.setattr(routing, "_NODES", {})
    monkeypatch.setattr(routing, "_QUANTITIES", {})
    inits = []
    for cls in _stage_classes():
        real = cls.__init__

        def counting(self, *args, _real=real, **kwargs):
            inits.append(type(self).__name__)
            _real(self, *args, **kwargs)

        monkeypatch.setattr(cls, "__init__", counting)
    _boltzmann.clear_memo()
    runs = dict(_boltzmann.run_counts())

    for cls in (MassFunction, LinearPower, Transfer, Growth, MassVariance):
        cls.fields_info()
        cls.parameter_info()
        cls.parameter_names()
        cls.parameter_defaults()
        cls.quantities_available()
        for name in cls.parameter_names():
            cls.invalidated_by(name)
    assert inits == []
    assert dict(_boltzmann.run_counts()) == runs
    # The counter does count construction.
    Transfer(model="EH")
    assert inits == ["Transfer"]


# ---------------------------------------------------------------------------------
# Extension stages (the hooks halomod-style stages use)
# ---------------------------------------------------------------------------------


@attrs.frozen(kw_only=True)
class Profile(Stage):
    """A toy stage, with a field named like one of MassFunction's."""

    concentration: float = field(default=5.0, doc="A concentration.")
    delta_c: float = field(default=1.0, doc="Not the mass function's delta_c.")


@attrs.frozen(kw_only=True)
class HaloModel(Stage):
    """A toy extension stage holding a MassFunction and a Profile."""

    mass_function: MassFunction = field(
        factory=lambda: MassFunction.build(**{"transfer.model": "EH"}), doc="The mass function."
    )
    profile: Profile = field(factory=Profile, doc="The profile.")
    bias: str = field(default="Tinker10", doc="The bias model.")

    @derived_property("mass_function.delta_c", "profile")
    def boosted(self) -> Profile:
        """A derived stage."""
        return self.profile.evolve(concentration=3 * self.mass_function.delta_c)

    @cached_property
    def power(self) -> float:
        """A toy output."""
        return self.mass_function.delta_c * self.boosted.concentration

    def halo_power_kernel(self) -> float:
        """A kernel entry point: not a quantity."""
        return 0.0


def test_extension_stage(mf):
    hm = HaloModel(mass_function=mf)
    names = set(HaloModel.parameter_names())
    assert BUILD_KEYWORDS - {"delta_c"} <= names
    assert {"bias", "profile.delta_c", "mass_function.delta_c"} <= names
    lower = hm.evolve(sigma_8=0.7)
    assert lower.mass_function.variance is mf.variance
    assert lower.profile is hm.profile
    # The derived stage follows the mass function's delta_c, and is carried otherwise.
    boosted = hm.boosted
    assert hm.evolve(bias="SMT").boosted is boosted
    assert hm.evolve(sigma_8=0.7).boosted is boosted
    new = hm.evolve(**{"mass_function.delta_c": 1.6})
    assert new.boosted.concentration == pytest.approx(4.8)
    assert hm.evolve(**{"profile.delta_c": 2.0}).profile.delta_c == 2.0
    assert HaloModel.invalidated_by("mass_function.delta_c") == ("", "boosted", "mass_function")
    q = HaloModel.quantities_available()
    assert q[""] == ("power",)
    assert q["boosted"] == ()
    assert q["mass_function.variance"]
    # The mass function's computed defaults (sigma_8, n_s) apply inside the tree too.
    fresh = HaloModel.from_flat({"transfer.model": "EH"})
    assert fresh.mass_function == MassFunction.build(**{"transfer.model": "EH"})


def test_bad_derived_input_raises_when_the_table_is_built():
    @attrs.frozen(kw_only=True)
    class BadInput(Stage):
        profile: Profile = field(factory=Profile, doc="The profile.")

        @derived_property("profile.nope")
        def derived(self) -> int:
            return 0

    with pytest.raises(TypeError, match=r"BadInput.derived: the input 'profile.nope'"):
        BadInput.parameter_names()


def test_dotted_suffixes():
    @attrs.frozen(kw_only=True)
    class Pair(Stage):
        x: Profile = field(factory=Profile, doc="One.")
        y: Profile = field(factory=Profile, doc="Two.")

    @attrs.frozen(kw_only=True)
    class Nest(Stage):
        pair: Pair = field(factory=Pair, doc="A pair.")
        concentration: float = field(default=1.0, doc="The root's own.")

    with pytest.raises(TypeError, match=r"'concentration' is ambiguous.*'x.concentration'"):
        Pair().evolve(concentration=1.0)
    assert Pair().evolve(**{"x.concentration": 2.0}).x.concentration == 2.0
    assert Pair.parameter_names() == (
        "x.concentration",
        "x.delta_c",
        "y.concentration",
        "y.delta_c",
    )
    # A field's full path means that field: the root's own name beats the suffixes.
    assert Nest().evolve(concentration=3.0).concentration == 3.0
    assert Nest().evolve(**{"x.concentration": 3.0}).pair.x.concentration == 3.0
    assert Nest.parameter_info()["concentration"].paths == ("concentration",)


# ---------------------------------------------------------------------------------
# Toy trees: the less common shapes of a tree
# ---------------------------------------------------------------------------------


@attrs.frozen(kw_only=True)
class Inner:
    """A toy model field value, with a field."""

    x: float = 1.0


@attrs.frozen(kw_only=True)
class Outer:
    """A toy model holding another model."""

    inner: Inner = attrs.field(factory=Inner)
    y: float = 2.0


@attrs.frozen(kw_only=True)
class Leafy(Stage):
    """A toy stage with a setting that a stage composing two of them shares."""

    acc: int = field(default=1, doc="A setting.")
    thing: Outer = field(factory=Outer, doc="A model.")
    count: int = field(default=0, converter=attrs.Converter(int), doc="A count.")


@attrs.frozen(kw_only=True)
class Holder(Stage):
    """A toy stage composing two Leafy stages, sharing their setting, with a computed default."""

    shared_parameters: ClassVar = frozenset({"acc"})

    left: Leafy = field(factory=Leafy, doc="One.")
    right: Leafy = field(factory=Leafy, doc="Two.")
    base: int = field(default=3, doc="A base.")

    @derived_property("base", "left.acc")
    def double(self) -> int:
        """Twice the base, times the setting."""
        return 2 * self.base * self.left.acc

    @derived_property("base")
    def unannotated(self):
        """No return annotation."""
        return self.base

    @derived_property("base")
    def unresolved(self) -> "NotAType":  # noqa: F821 - an annotation that can't be resolved
        """A return annotation that can't be resolved."""
        return self.base

    @classmethod
    def computed_defaults(cls, given):
        return {"acc": 2} if "base" not in given else {}


def test_toy_shared_parameter_and_derived_properties():
    built = Holder.from_flat({})
    assert built.left.acc == built.right.acc == 2  # a shared computed default
    assert Holder.from_flat({"base": 5}).left.acc == 1
    assert Holder.parameter_info()["acc"].paths == ("left.acc", "right.acc")
    new = built.evolve(acc=4)
    assert new.left.acc == new.right.acc == 4
    double = built.double
    assert built.evolve(**{"right.count": 2}).double is double
    assert built.evolve(acc=3).double == 18
    assert built.evolve(base=4).double == 16
    assert Holder.invalidated_by("acc") == ("", "double", "left", "right")
    assert Holder.invalidated_by("base") == ("", "double", "unannotated", "unresolved")
    # Without the composing stage's declaration, the name is ambiguous.
    with pytest.raises(TypeError, match=r"'acc' is ambiguous.*'a.acc'"):

        @attrs.frozen(kw_only=True)
        class Unshared(Stage):
            a: Leafy = field(factory=Leafy, doc="One.")
            b: Leafy = field(factory=Leafy, doc="Two.")

        Unshared().evolve(acc=2)


def test_toy_nested_model_fields():
    stage = Leafy()
    assert stage.evolve(**{"thing.inner.x": 5.0}).thing == Outer(inner=Inner(x=5.0))
    both = stage.evolve(**{"thing": Outer(y=3.0), "thing.inner.x": 5.0})
    assert both.thing == Outer(inner=Inner(x=5.0), y=3.0)
    assert stage.evolve(**{"thing.inner": Inner(x=2.0), "thing.inner.x": 5.0}).thing.inner.x == 5.0
    # A model field of a field with a converter: the given value is converted first.
    with pytest.raises(TypeError, match="not a model with fields"):
        stage.evolve(**{"count": "3", "count.real": 1})
    with pytest.raises(TypeError, match="not a model with fields"):
        stage.evolve(**{"acc": 3, "acc.real": 1})
    with pytest.raises(TypeError, match="ambiguous"):
        Holder().evolve(**{"count.real": 1})


def test_toy_unknown_dotted_names():
    with pytest.raises(TypeError, match=r"no parameter 'nothing\.at\.all'"):
        Leafy().evolve(**{"nothing.at.all": 1})
    with pytest.raises(TypeError, match=r"the input 'base.x' is not a field"):

        @attrs.frozen(kw_only=True)
        class Through(Stage):
            base: int = field(default=3, doc="Not a stage.")

            @derived_property("base.x")
            def derived(self) -> int:
                return 0

        Through.parameter_names()


def test_same():
    assert stage_module._same(1, 1)
    assert not stage_module._same(np.ones(2), np.ones(2))  # an eq that can't answer


# ---------------------------------------------------------------------------------
# Derived properties declare what they read (undeclared_inputs)
# ---------------------------------------------------------------------------------


def test_hmf_derived_properties_declare_their_inputs(mf):
    """Every field a derived property of hmf.core reads is one of its declared inputs."""
    checked = {
        MassFunction: MassFunction.build(**{"transfer.model": "EH"}),
        LinearPower: LinearPower(transfer=Transfer(model="EH"), sigma_8=0.8),
    }
    # Every stage class of hmf.core with derived properties is checked here.
    with_derived = {
        cls
        for cls in _stage_classes()
        if cls.__module__.startswith("hmf.core") and cls._derived_properties
    }
    assert with_derived == set(checked)
    for stage in checked.values():
        assert stage_module.undeclared_inputs(stage) == {}
    assert stage_module.undeclared_inputs(HaloModel(mass_function=mf)) == {}


@attrs.frozen(kw_only=True)
class Reads(Stage):
    """A toy stage whose derived properties declare too little, or too much."""

    profile: Profile = field(factory=Profile, doc="A stage it holds.")
    a: int = field(default=1, doc="One.")
    b: int = field(default=2, doc="Two.")

    @derived_property("a")
    def too_little(self) -> int:
        return self.a + self.b + int(self.profile.concentration)

    @derived_property("a", "b", "profile")
    def too_much(self) -> int:
        # b and the profile are only read on another branch: declaring them is safe.
        return self.a if self.a > 0 else self.b + int(self.profile.delta_c)

    @derived_property("a")
    def holds_the_profile(self) -> tuple:
        # The profile is held, not read: a value built from it may read it later.
        return ({"stage": [self.profile]}, self.a)


def test_undeclared_inputs_reports_reads_and_held_stages():
    assert stage_module.undeclared_inputs(Reads()) == {
        "too_little": ("b", "profile.concentration"),
        "holds_the_profile": ("profile",),
    }
