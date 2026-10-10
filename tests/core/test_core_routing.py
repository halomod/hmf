"""Tests of hmf.core.routing: parameters by name through a stage tree, and introspection."""

from functools import cached_property
from types import MappingProxyType
from typing import ClassVar

import attrs
import numpy as np
import pytest
from astropy.cosmology import Planck15, Planck18

from hmf.core import _boltzmann, field, routing
from hmf.core.accuracy import KAccuracy, MassAccuracy
from hmf.core.growth import Growth
from hmf.core.linear_power import LinearPower
from hmf.core.mass_function import MassFunction
from hmf.core.mass_variance import MassVariance
from hmf.core.routing import Derivation, GivenParameters
from hmf.core.stage import Stage
from hmf.core.transfer import Transfer
from hmf.core.transfer_models import FromArray
from hmf.core.units import Msun_h, h_Mpc

M = np.logspace(10, 15, 6) * Msun_h

#: MassFunction.build's keywords before routing, plus the variance's truncation_rtol,
#: which build did not take: build's keywords and evolve's flat names are one set.
BUILD_KEYWORDS = {
    "cosmology",
    "transfer_model",
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
}


@pytest.fixture(scope="module")
def mf():
    """A fast mass function (Eisenstein & Hu), with the quantities of the tests cached."""
    stage = MassFunction.build(transfer_model="EH")
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
    assert info["transfer_model"].paths == ("linear_power.transfer.model",)
    assert info["growth_model"].paths == ("linear_power.growth.model",)
    assert info["filter"].path == "variance.filter"
    assert info["delta_c"].default == 1.686
    assert info["delta_c"].type is float
    assert "critical overdensity" in info["delta_c"].doc
    assert info["cosmology"].shared
    assert set(info["cosmology"].paths) == {
        "linear_power.transfer.cosmology",
        "linear_power.growth.cosmology",
        "linear_power.cosmology",
    }
    assert info["cosmology"].default is Planck18
    assert set(info["k_accuracy"].paths) == {
        "linear_power.transfer.k_accuracy",
        "linear_power.growth.k_accuracy",
        "linear_power.k_accuracy",
        "variance.k_accuracy",
    }
    assert set(info["disk_cache"].paths) == {
        "linear_power.transfer.disk_cache",
        "linear_power.growth.disk_cache",
    }
    assert {name for name, i in info.items() if i.shared} == {
        "cosmology",
        "k_accuracy",
        "disk_cache",
    }


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
    """Every field of every stage of the MassFunction tree has a name, or is derived."""
    info = MassFunction.parameter_info()
    routed = {path for i in info.values() for path in i.paths}
    table = routing.router(MassFunction)
    fields = set()
    for path, cls in table.stages.items():
        for a in attrs.fields(cls):
            full = (*path, a.name)
            if full in table.stages or full in table.derived:
                continue
            fields.add(".".join(full))
    assert fields == routed
    # The derived fields are the declared ones, and the stages are the tree's.
    assert {".".join(p) for p in table.derived} == {
        "variance.power",
        "linear_power.growth.transfer",
    }
    assert set(table.stages) == {
        (),
        ("linear_power",),
        ("linear_power", "transfer"),
        ("linear_power", "growth"),
        ("variance",),
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
    "transfer_model": ("BBKS", ("dndm", 0.0)),
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
    return mf.evolve(transfer_model=FromArray(k=k * h_Mpc, t=t, t_tot=t * (1 - 0.1 * k / (k + 1))))


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
    lp = mf.linear_power.evolve(transfer_model="BBKS")
    assert lp.growth.transfer is lp.transfer
    assert lp.transfer.model.__class__.__name__ == "BBKS"
    assert Transfer().evolve(n_s=0.9).n_s == 0.9


# ---------------------------------------------------------------------------------
# Structural sharing, shared parameters and derived fields
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


def test_sigma_8_change_shares_the_normalisation(mf):
    """Routing rebuilds LinearPower with its own evolve_own, which shares sigma_8,raw."""
    raw = mf.linear_power.unnormalised_sigma_8
    lower = mf.evolve(sigma_8=0.7)
    assert lower.linear_power._normalisation_memo is mf.linear_power._normalisation_memo
    assert lower.linear_power.unnormalised_sigma_8 == raw


def test_shared_cosmology_is_consistent(mf):
    new = mf.evolve(cosmology=Planck15)
    lp = new.linear_power
    for stage in (lp, lp.transfer, lp.growth):
        assert stage.cosmology is Planck15
    assert lp.growth.transfer is lp.transfer
    assert new.variance.power.transfer is lp.transfer
    assert new == MassFunction.from_flat({**mf.to_flat(), "cosmology": Planck15})


def test_shared_k_accuracy_is_consistent(mf):
    high = KAccuracy.high()
    new = mf.evolve(k_accuracy=high)
    lp = new.linear_power
    for stage in (lp, lp.transfer, lp.growth, new.variance):
        assert stage.k_accuracy == high
    assert new.variance.power.transfer is lp.transfer


def test_shared_name_skips_a_stage_given_whole(mf):
    """A stage given whole has its own value of a shared parameter."""
    transfer = mf.linear_power.transfer.evolve(cosmology=Planck15)
    lp = mf.linear_power.evolve(
        transfer=transfer, growth=Growth.from_transfer(transfer), cosmology=Planck15
    )
    assert lp.cosmology is Planck15
    assert lp.transfer is transfer


def test_derived_fields_are_rederived(mf):
    new = mf.evolve(transfer_model="BBKS")
    lp = new.linear_power
    assert new.variance.power.transfer is lp.transfer
    assert lp.growth.transfer is lp.transfer
    assert new.variance.power == lp.power_source
    # Replacing the linear power whole re-derives the variance's power too.
    whole = mf.evolve(linear_power=lp)
    assert whole.variance.power.transfer is lp.transfer
    # ... and keeps the variance when the power does not change.
    assert mf.evolve(linear_power=mf.linear_power.evolve(sigma_8=0.7)).variance is mf.variance


def test_species_rederives_the_variance(mf):
    new = mf.evolve(species="tot")
    assert new.variance.power.species == "tot"
    assert new.variance.power.transfer is mf.linear_power.transfer


def test_a_link_the_user_did_not_make_is_left():
    """A Growth without a transfer stage keeps none when the transfer changes."""
    transfer = Transfer(model="EH")
    lp = LinearPower(transfer=transfer, growth=Growth(), sigma_8=0.8)
    new = lp.evolve(n_s=0.9)
    assert new.growth.transfer is None
    assert new.growth is lp.growth


def test_derived_field_cannot_be_set(mf):
    with pytest.raises(TypeError, match=r"'variance.power' is derived from"):
        mf.evolve(**{"variance.power": mf.variance.power})
    with pytest.raises(TypeError, match=r"derived from 'linear_power.transfer'"):
        mf.evolve(**{"linear_power.growth.transfer": None})


def test_optional_stage(mf):
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
    assert growth.evolve(**{"linear_power.growth.model.dln_a": 0.01}).linear_power.growth.model == (
        mf.linear_power.growth.model
    )
    built = MassFunction.build(transfer_model="EH", fit="ST", **{"fit.a": 0.75})
    assert built.fit == changed.fit
    assert MassFunction.build(transfer_model="EH", **{"fit.A_200": 0.2}).fit.A_200 == 0.2


def test_model_field_errors(mf):
    with pytest.raises(TypeError, match=r"Tinker08 \(at 'fit'\) has no field 'A_20'.*'A_200'"):
        mf.evolve(**{"fit.A_20": 0.2})
    with pytest.raises(TypeError, match="not a model with fields"):
        mf.evolve(**{"sigma_8.x": 1.0})
    with pytest.raises(TypeError, match="given twice"):
        mf.evolve(**{"fit.A_200": 0.2, "fit": mf.fit, "linear_power.sigma_8": 0.7, "sigma_8": 0.7})
    with pytest.raises(TypeError, match="given twice"):
        mf.evolve(**{"growth_model.dln_a": 0.2, "linear_power.growth.model.dln_a": 0.1})
    with pytest.raises(TypeError, match="ambiguous"):
        mf.evolve(**{"model.z_max": 10.0})
    with pytest.raises(TypeError, match="no default to change"):
        LinearPower.from_flat({"transfer_model": "EH", "sigma_8.x": 1.0})


# ---------------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------------


def test_unknown_name_suggests(mf):
    with pytest.raises(TypeError, match=r"no parameter 'sigma8'.*Did you mean 'sigma_8'"):
        mf.evolve(sigma8=0.8)
    with pytest.raises(TypeError, match=r"no parameter 'variance.filtr'.*'variance.filter'"):
        mf.evolve(**{"variance.filtr": "SharpK"})
    with pytest.raises(TypeError, match=r"no parameter 'Mminn'"):
        MassFunction.build(Mminn=3)
    with pytest.raises(TypeError, match=r"no parameter 'tranfer_model'.*'transfer_model'"):
        MassFunction.invalidated_by("tranfer_model")


def test_ambiguous_name_lists_paths(mf):
    with pytest.raises(TypeError, match="ambiguous") as err:
        mf.evolve(model="EH")
    message = str(err.value)
    assert "'linear_power.transfer.model'" in message
    assert "'linear_power.growth.model'" in message
    assert "'transfer_model'" in message
    assert "'growth_model'" in message


def test_conflicts(mf):
    with pytest.raises(TypeError, match="given twice"):
        mf.evolve(sigma_8=0.7, **{"linear_power.sigma_8": 0.75})
    with pytest.raises(TypeError, match="given twice"):
        mf.evolve(cosmology=Planck15, **{"linear_power.cosmology": Planck15})
    with pytest.raises(TypeError, match="which is also given"):
        mf.evolve(linear_power=mf.linear_power, sigma_8=0.7)


def test_unknown_names_raise_before_anything_is_built(mf, monkeypatch):
    built = []
    real = routing._Rebuild.run
    monkeypatch.setattr(routing._Rebuild, "run", lambda self: built.append(1) or real(self))
    with pytest.raises(TypeError):
        mf.evolve(sigma_8=0.7, sigma8=0.7)
    with pytest.raises(TypeError):
        MassFunction.build(sigma_8=0.7, fit_a=1.0)
    assert built == []


def test_missing_required_parameter():
    with pytest.raises(TypeError, match=r"missing required parameter\(s\) \['sigma_8'\]"):
        LinearPower.from_flat({"transfer_model": "EH"})
    with pytest.raises(TypeError, match=r"missing required parameter\(s\) \['power'\]"):
        MassVariance.from_flat({})
    lp = LinearPower.from_flat({"transfer_model": "EH", "sigma_8": 0.8})
    assert lp.growth.transfer is lp.transfer


def test_evolve_is_atomic(mf):
    before = mf.dndm(m=M, z=0.0)
    lp, variance = mf.linear_power, mf.variance
    for changes in (
        {"sigma_8": 0.7, "delta_c": -1.0},  # the root fails after linear_power is rebuilt
        {"transfer_model": "BBKS", "filter": "NotAFilter"},
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
    assert flat["transfer_model"] == mf.linear_power.transfer.model


def test_from_flat_nested_and_dotted():
    nested = MassFunction.from_flat(
        {"transfer_model": "EH", "linear_power": {"sigma_8": 0.7, "transfer": {"n_s": 0.95}}}
    )
    dotted = MassFunction.from_flat(
        {"transfer_model": "EH", "linear_power.sigma_8": 0.7, "linear_power.transfer.n_s": 0.95}
    )
    assert nested == dotted == MassFunction.build(transfer_model="EH", sigma_8=0.7, n_s=0.95)
    assert nested.linear_power.growth.transfer is nested.linear_power.transfer
    assert nested.variance.power.transfer is nested.linear_power.transfer


def test_from_flat_with_a_stage_given_whole(mf):
    """A stage given whole is used; build's computed defaults below it are not."""
    built = MassFunction.from_flat({"linear_power": mf.linear_power, "fit": "ST"})
    assert built.linear_power is mf.linear_power
    assert built.variance.power.transfer is mf.linear_power.transfer
    transfer = Transfer(model="EH", cosmology=Planck15, n_s=0.9)
    lp_built = MassFunction.from_flat({"linear_power.transfer": transfer, "cosmology": Planck15})
    assert lp_built.linear_power.transfer is transfer
    assert lp_built.linear_power.growth.transfer is transfer
    # sigma_8 comes from the given transfer's cosmology.
    assert lp_built.linear_power.sigma_8 == Planck15.meta["sigma8"]


def test_build_none_is_not_the_default():
    """A parameter left out takes its default; None is a value like any other."""
    with pytest.raises(TypeError):
        MassFunction.build(transfer_model="EH", n_s=None)


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
    everything = ("", "linear_power", "variance", "linear_power.growth", "linear_power.transfer")
    assert MassFunction.invalidated_by("transfer_model") == everything
    assert MassFunction.invalidated_by("n_s") == everything
    assert MassFunction.invalidated_by("cosmology") == everything
    assert MassFunction.invalidated_by("species") == ("", "linear_power", "variance")
    assert MassFunction.invalidated_by("linear_power.transfer") == everything
    with pytest.raises(TypeError, match="ambiguous"):
        MassFunction.invalidated_by("model")


def test_invalidated_by_matches_evolve(mf):
    """The stages invalidated_by names are exactly the ones evolve rebuilds."""
    table = routing.router(MassFunction)
    for name, (value, _) in CHANGES.items():
        if name == "disk_cache":
            continue
        new = mf.evolve(**{name: value})
        rebuilt = set()
        for path in table.stages:
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
                "derivations",
                "parameter_aliases",
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

    parameter_aliases: ClassVar = MappingProxyType({"profile_delta_c": "profile.delta_c"})
    derivations: ClassVar = (
        Derivation(
            field="profile.concentration",
            sources=("mass_function.delta_c",),
            derive=lambda get: 3 * get("mass_function.delta_c"),
        ),
    )

    mass_function: MassFunction = field(doc="The mass function.")
    profile: Profile = field(factory=Profile, doc="The profile.")
    bias: str = field(default="Tinker10", doc="The bias model.")

    @cached_property
    def power(self) -> float:
        """A toy output."""
        return self.mass_function.delta_c * self.profile.concentration

    def halo_power_kernel(self) -> float:
        """A kernel entry point: not a quantity."""
        return 0.0


def test_extension_stage(mf):
    hm = HaloModel(mass_function=mf, profile=Profile(concentration=3 * mf.delta_c))
    names = set(HaloModel.parameter_names())
    assert BUILD_KEYWORDS - {"delta_c"} <= names
    assert {"bias", "profile_delta_c", "mass_function.delta_c"} <= names
    with pytest.raises(TypeError, match="ambiguous"):
        hm.evolve(delta_c=1.6)
    lower = hm.evolve(sigma_8=0.7)
    assert lower.mass_function.variance is mf.variance
    assert lower.profile is hm.profile
    # The derivation follows the mass function's delta_c.
    new = hm.evolve(**{"mass_function.delta_c": 1.6})
    assert new.profile.concentration == pytest.approx(4.8)
    assert hm.evolve(profile_delta_c=2.0).profile.delta_c == 2.0
    assert HaloModel.invalidated_by("mass_function.delta_c") == ("", "mass_function", "profile")
    assert HaloModel.quantities_available()[""] == ("power",)
    built = HaloModel.from_flat({"mass_function": mf, "bias": "SMT"})
    assert built.profile.concentration == pytest.approx(3 * mf.delta_c)
    # The mass function's computed defaults (sigma_8, n_s) apply inside the tree too.
    fresh = HaloModel.from_flat({"transfer_model": "EH"})
    assert fresh.mass_function == MassFunction.build(transfer_model="EH")


def test_bad_hooks_raise_when_the_table_is_built():
    @attrs.frozen(kw_only=True)
    class BadAlias(Stage):
        parameter_aliases: ClassVar = MappingProxyType({"x": "profile.nope"})
        profile: Profile = field(factory=Profile, doc="The profile.")

    with pytest.raises(TypeError, match="not a parameter field"):
        BadAlias.parameter_names()

    @attrs.frozen(kw_only=True)
    class ClashingAlias(Stage):
        parameter_aliases: ClassVar = MappingProxyType({"concentration": "profile.delta_c"})
        profile: Profile = field(factory=Profile, doc="The profile.")

    with pytest.raises(TypeError, match="also the name of a field"):
        ClashingAlias.parameter_names()

    @attrs.frozen(kw_only=True)
    class BadDerivation(Stage):
        derivations: ClassVar = (
            Derivation(field="profile.nope", sources=(), derive=lambda get: 0),
        )
        profile: Profile = field(factory=Profile, doc="The profile.")

    with pytest.raises(TypeError, match="not a field of the stage tree"):
        BadDerivation.parameter_names()


def test_repeated_alias_is_ambiguous():
    @attrs.frozen(kw_only=True)
    class Twice(Stage):
        a: Profile = field(factory=Profile, doc="One.")
        b: Profile = field(factory=Profile, doc="Two.")

    @attrs.frozen(kw_only=True)
    class Aliased(Stage):
        parameter_aliases: ClassVar = MappingProxyType({"c": "concentration"})
        concentration: float = field(default=1.0, doc="A concentration.")

    @attrs.frozen(kw_only=True)
    class Both(Stage):
        x: Aliased = field(factory=Aliased, doc="One.")
        y: Aliased = field(factory=Aliased, doc="Two.")

    with pytest.raises(TypeError, match=r"'concentration' is ambiguous.*'a.concentration'"):
        Twice().evolve(concentration=1.0)
    with pytest.raises(TypeError, match=r"'c' is ambiguous.*'x.concentration'"):
        Both().evolve(c=2.0)
    assert Both().evolve(**{"x.concentration": 2.0}).x.concentration == 2.0
    assert Both.parameter_names() == ("x.concentration", "y.concentration")


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
    """A toy stage with a shared field."""

    acc: int = field(default=1, shared=True, doc="A shared setting.")
    thing: Outer = field(factory=Outer, doc="A model.")
    count: int = field(default=0, converter=attrs.Converter(int), doc="A count.")


@attrs.frozen(kw_only=True)
class Holder(Stage):
    """A toy stage with derivations from its own fields, and a computed default."""

    derivations: ClassVar = (
        Derivation(field="double", sources=("base",), derive=lambda get: 2 * get("base")),
        # A chain: derived from a derived field of the stage.
        Derivation(field="right.count", sources=("double",), derive=lambda get: get("double")),
        Derivation(
            field="left.count",
            sources=("base", "right.acc"),
            derive=lambda get: get("base") + get("right.acc"),
        ),
    )

    left: Leafy = field(factory=Leafy, doc="One.")
    right: Leafy = field(factory=Leafy, doc="Two.")
    base: int = field(default=3, doc="A base.")
    double: int = field(default=6, doc="Twice the base.")

    @classmethod
    def computed_defaults(cls, given):
        return {"acc": 2} if "base" not in given else {}


def test_toy_derivations_from_own_fields():
    built = Holder.from_flat({})
    assert (built.base, built.double, built.right.count, built.left.count) == (3, 6, 6, 5)
    assert built.left.acc == built.right.acc == 2  # a shared computed default
    assert Holder.from_flat({"base": 5}).left.acc == 1
    new = built.evolve(base=4)
    assert (new.double, new.right.count, new.left.count) == (8, 8, 6)
    assert built.evolve(**{"right.acc": 4}).left.count == 7
    assert Holder.invalidated_by("base") == ("", "left", "right")
    # The right count is derived: its link holds until it is set by hand.
    with pytest.raises(TypeError, match="derived from"):
        built.evolve(**{"right.count": 2})
    other = built.evolve(**{"left.acc": 5})
    assert other.right is built.right
    # A derived field set by hand (the link does not hold) is left alone.
    by_hand = Holder(double=7)
    assert by_hand.evolve(base=4).double == 7


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


def test_toy_unknown_dotted_names():
    with pytest.raises(TypeError, match=r"no parameter 'nothing\.at\.all'"):
        Leafy().evolve(**{"nothing.at.all": 1})
    with pytest.raises(TypeError, match="not a field of the stage tree"):

        @attrs.frozen(kw_only=True)
        class Through(Stage):
            derivations: ClassVar = (Derivation(field="base.x", sources=(), derive=lambda get: 0),)
            base: int = field(default=3, doc="Not a stage.")

        Through.parameter_names()


def test_same():
    assert routing._same(1, 1)
    assert not routing._same(np.ones(2), np.ones(2))  # an eq that can't answer
