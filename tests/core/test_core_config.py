"""Tests of hmf.core.config: configurations, their TOML form, content hashes and pickling."""

from __future__ import annotations

import copy
import logging
import pickle
import pickletools
import tomllib
import typing
import warnings
from pathlib import Path

import astropy.units as u
import attrs
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM, LambdaCDM, Planck15, Planck18

from hmf.core import _boltzmann, config, field
from hmf.core.accuracy import KAccuracy, MassAccuracy
from hmf.core.concentration import ConcentrationModel, Duffy08
from hmf.core.filters import Filter
from hmf.core.fits import SMT, FittingFunction, Tinker08
from hmf.core.growth import Growth
from hmf.core.growth_models import GrowthModel
from hmf.core.linear_power import LinearPower
from hmf.core.mass_conversion import MassConversion, ProfileConversion
from hmf.core.mass_definitions import FOF, SOCritical, SOGeneric, SOMean
from hmf.core.mass_function import MassFunction
from hmf.core.mass_variance import MassVariance
from hmf.core.model import ModelNotFoundError
from hmf.core.power_source import TabulatedPower
from hmf.core.profiles import DensityProfile
from hmf.core.stage import Stage
from hmf.core.transfer import Transfer
from hmf.core.transfer_models import CAMB, FromArray, TransferModel
from hmf.core.units import H0_unit, Msun_h, h_Mpc, littleh, rho_unit
from hmf.exceptions import HMFExtrapolationWarning

GOLDEN = Path(__file__).parent / "data" / "config_schema_1_default_mass_function.toml"

M = np.logspace(10, 15, 11) * Msun_h
K = np.logspace(-3, 1, 60)
T = 1 / (1 + (K / 0.05) ** 2) ** 0.8

HEADER = "[hmf]\nschema_version = 1\nstage = 'hmf.core.mass_function:MassFunction'\n"


def roundtrip(stage: Stage) -> tuple[Stage, Stage]:
    """Check that config and TOML round trips are the identity; return both copies."""
    cfg = config.to_config(stage)
    from_mapping = config.from_config(cfg)
    assert from_mapping == stage
    assert config.to_config(from_mapping) == cfg
    text = config.dumps_toml(stage)
    from_toml = config.loads_toml(text)
    assert from_toml == stage
    assert config.dumps_toml(from_toml) == text
    assert config.to_config(from_toml) == cfg
    return from_mapping, from_toml


def _eh_linear_power(**kwargs) -> LinearPower:
    return LinearPower(transfer=Transfer(model="EH", **kwargs), sigma_8=0.8)


# ---------------------------------------------------------------------------------
# Every registered model of every kind
# ---------------------------------------------------------------------------------

#: The arguments of the models that have required fields.
REQUIRED = {
    "hmf.core.transfer_models:FromArray": {"k": K * h_Mpc, "t": T},
    "hmf.core.transfer_models:FromFile": {"fname": "transfer.dat"},
    "hmf.core.growth_models:FromArray": {"z": [0, 1, 2, 3], "d": [1, 0.6, 0.4, 0.3]},
    "hmf.core.growth_models:FromFile": {"fname": "growth.dat"},
}

KINDS = (
    FittingFunction,
    TransferModel,
    GrowthModel,
    Filter,
    ConcentrationModel,
    DensityProfile,
    MassConversion,
)


def _holder(kind: type, model: object) -> Stage:
    """A stage tree holding ``model`` (nothing is computed to build it)."""
    lp = _eh_linear_power()
    if kind is FittingFunction:
        return MassFunction(linear_power=lp, fit=model)
    if kind is Filter:
        return MassFunction(linear_power=lp, filter=model)
    if kind is TransferModel:
        return Transfer(model=model)
    if kind is GrowthModel:
        try:
            return Growth(model=model)
        except ValueError:  # e.g. Heath77Growth: a cosmology without dark energy
            return Growth(model=model, cosmology=LambdaCDM(H0=70, Om0=0.3, Ode0=0.0))
    conversion = (
        model
        if kind is MassConversion
        else ProfileConversion(
            **{"concentration" if kind is ConcentrationModel else "profile": model}
        )
    )
    return MassFunction(
        linear_power=lp, fit=SMT(), mass_definition="200c", mass_conversion=conversion
    )


def _changed(model: object) -> object | None:
    """``model`` with one field changed from its default (None if it has no field)."""
    for a in attrs.fields(type(model)):
        if not a.init:
            continue
        value = getattr(model, a.name)
        options = typing.get_args(a.type) if typing.get_origin(a.type) is typing.Literal else ()
        if isinstance(value, bool):
            new = not value
        elif isinstance(value, float):
            new = value * 1.1 + 0.01
        elif value is None:
            new = 0.5
        elif isinstance(value, u.Quantity):
            new = value * 2
        elif options:
            new = next(o for o in options if o != value)
        else:
            continue
        try:
            return attrs.evolve(model, **{a.alias or a.name: new})
        except (TypeError, ValueError):
            continue
    return None


MODELS = [
    pytest.param(kind, name, id=name.split(":")[1] + f"-{kind.__name__}")
    for kind in KINDS
    for name in kind.get_models()
]


@pytest.mark.parametrize(("kind", "name"), MODELS)
def test_every_registered_model_round_trips(kind, name):
    model = kind.get(name)(**REQUIRED.get(name, {}))
    variants = [model]
    changed = _changed(model)
    if changed is not None:
        assert changed != model
        variants.append(changed)
    for value in variants:
        data = config.unstructure(value)
        assert data["model"] == name
        assert config.structure(data, kind) == value
        assert config.unstructure(config.structure(data, kind)) == data
        roundtrip(_holder(kind, value))


def test_a_model_by_alias_or_partial_table():
    assert config.structure("ST", FittingFunction) == FittingFunction.get("ST")()
    st = FittingFunction.get("ST")
    assert config.structure({"model": "ST", "a": 0.75}, FittingFunction) == st(a=0.75)
    assert config.structure({"a": 0.75}, SMT) == SMT(a=0.75)


@attrs.frozen(kw_only=True)
class OptionalStage(Stage):
    """A stage with an optional field whose default is not None."""

    x: float | None = field(default=1.0, doc="A number, or None.")


def test_none_that_is_not_the_default_is_explicit_in_toml():
    stage = OptionalStage(x=None)
    assert config.to_config(stage)["x"] is None
    assert "x = { none = true }" in config.dumps_toml(stage)
    _, from_toml = roundtrip(stage)
    assert from_toml.x is None
    # A None that is the default is left out of TOML.
    mf = MassFunction(linear_power=_eh_linear_power())
    assert config.to_config(mf)["mass_definition"] is None
    assert "mass_definition" not in config.dumps_toml(mf)


# ---------------------------------------------------------------------------------
# Accuracy, Quantities, cosmologies, tables, mass definitions
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize("preset", ["default", "fast", "high"])
def test_every_accuracy_class_and_preset_round_trips(preset):
    def make(cls):
        return cls() if preset == "default" else getattr(cls, preset)()

    mf = MassFunction.build(
        **{"transfer.model": "EH"}, k_accuracy=make(KAccuracy), mass_accuracy=make(MassAccuracy)
    )
    cfg = config.to_config(mf)
    assert cfg["mass_accuracy"] == attrs.asdict(make(MassAccuracy))
    assert cfg["linear_power"]["transfer"]["k_accuracy"] == attrs.asdict(make(KAccuracy))
    roundtrip(mf)


def test_quantities_are_written_with_littleh():
    stage = Transfer(model=CAMB(k_max=5 * h_Mpc))
    model = config.to_config(stage)["model"]
    assert model["k_max"] == {"value": 5.0, "unit": "littleh / Mpc"}
    # A different spelling of the same unit, and a convertible unit, read back the same.
    for unit in ("littleh / Mpc", "littleh Mpc^-1", "littleh / kpc"):
        scale = 1e-3 if "kpc" in unit else 1.0
        data = {**model, "k_max": {"value": 5.0 * scale, "unit": unit}}
        k_max = config.structure(data, TransferModel).k_max
        assert k_max.unit is h_Mpc
        assert k_max.value == pytest.approx(5.0, rel=1e-15)
    roundtrip(stage)


def test_wrong_littleh_power_raises():
    data = config.unstructure(FromArray(k=K * h_Mpc, t=T))
    data["k"]["unit"] = "littleh2 / Mpc"
    with pytest.raises(u.UnitConversionError, match="littleh") as info:
        config.structure(data, TransferModel, where="transfer.model")
    assert "transfer.model" in "".join(info.value.__notes__)
    # Physical units for a model (which has no H0) raise too.
    data["k"]["unit"] = "1 / Mpc"
    with pytest.raises(u.UnitConversionError):
        config.structure(data, TransferModel)


def test_massive_neutrino_cosmology_round_trips():
    cosmo = Planck18.clone(m_nu=[0.0, 0.02, 0.06] * u.eV, name="nu", meta={"Mmin": 1e10 * Msun_h})
    transfer = Transfer(model="EH", cosmology=cosmo)
    data = config.to_config(transfer)["cosmology"]
    assert data["m_nu"] == {"value": [0.0, 0.02, 0.06], "unit": "eV"}
    assert data["meta"]["Mmin"] == {"value": 1e10, "unit": "solMass / littleh"}
    for copy_ in roundtrip(transfer):
        assert copy_.cosmology == cosmo
        assert copy_.cosmology.name == "nu"
        assert copy_.cosmology.meta["Mmin"] == 1e10 * Msun_h
        assert np.array_equal(copy_.cosmology.m_nu, cosmo.m_nu)


def test_massive_neutrinos_give_bit_identical_results():
    cosmo = Planck18.clone(m_nu=[0.0, 0.0, 0.1] * u.eV)
    mf = MassFunction.build(cosmology=cosmo, sigma_8=0.8, n_s=0.96)
    for other in roundtrip(mf):
        assert np.array_equal(other.dndm(m=M, z=0.5), mf.dndm(m=M, z=0.5))


def test_from_array_and_tabulated_power_round_trip_bit_identically():
    transfer = Transfer(model=FromArray(k=K * h_Mpc, t=T, t_tot=T * 0.99))
    lp = LinearPower(transfer=transfer, sigma_8=0.8)
    pk = 1e4 * K / (1 + (K / 0.02) ** 2.5) ** 1.6
    source = TabulatedPower(
        k=K / 0.7 / u.Mpc, pk=pk * 0.7**3 * u.Mpc**3, mean_density=8.5e10 * rho_unit,
        H0=70 * H0_unit,
    )  # fmt: skip
    mv = MassVariance(power=source, filter="SharpK")
    assert config.to_config(mv)["power"]["class"] == "hmf.core.power_source:TabulatedPower"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", HMFExtrapolationWarning)
        expected_p = lp.power(k=[0.01, 0.1, 1.0] * h_Mpc, z=1.0)
        expected_s = mv.sigma(M)
        for other in roundtrip(lp):
            assert np.array_equal(other.power(k=[0.01, 0.1, 1.0] * h_Mpc, z=1.0), expected_p)
        for other in roundtrip(mv):
            assert np.array_equal(other.sigma(M), expected_s)


def test_a_mass_variance_of_a_transfer_stage_round_trips():
    mv = MassVariance(power=Transfer(model="EH").power_source("tot"))
    cfg = config.to_config(mv)
    assert cfg["power"]["class"] == "hmf.core.transfer:UnnormalisedPower"
    assert cfg["power"]["species"] == "tot"
    for other in roundtrip(mv):
        assert np.array_equal(other.sigma(M), mv.sigma(M))


@pytest.mark.parametrize(
    "definition", [SOMean(), SOCritical(overdensity=500), FOF(linking_length=0.17)]
)
def test_mass_definitions_round_trip(definition):
    generic = SOGeneric(preferred=SOMean(overdensity=300))
    assert config.structure(config.unstructure(generic), SOGeneric) == generic
    mf = MassFunction.build(
        **{"transfer.model": "EH"}, fit="SMT", mass_definition=definition,
        mass_conversion=ProfileConversion(concentration=Duffy08(sample="relaxed")),
    )  # fmt: skip
    data = config.to_config(mf)["mass_definition"]
    assert data["class"] == f"hmf.core.mass_definitions:{type(definition).__name__}"
    text = f"fit = 'SMT'\nmass_conversion = 'profile'\nmass_definition = '{definition.name}'\n"
    assert config.loads_toml(text + HEADER).mass_definition == definition
    roundtrip(mf)


def test_default_and_converted_mass_functions_give_bit_identical_results():
    for mf in (
        MassFunction.build(),
        MassFunction.build(
            **{"transfer.model": "EH"}, fit="SMT", mass_definition="200c",
            mass_conversion=ProfileConversion(),
        ),
    ):  # fmt: skip
        expected = (mf.dndm(m=M, z=0.5), mf.ngtm(m=M, z=1.0))
        for other in roundtrip(mf):
            assert np.array_equal(other.dndm(m=M, z=0.5), expected[0])
            assert np.array_equal(other.ngtm(m=M, z=1.0), expected[1])


def test_growth_with_a_shared_transfer_round_trips():
    growth = Growth.from_transfer(Transfer(model="EH"), model="Integral")
    cfg = config.to_config(growth)
    assert cfg["cosmology"] == cfg["transfer"]["cosmology"]
    roundtrip(growth)
    roundtrip(Growth(model="ODE"))


def test_files_and_disk_cache(tmp_path):
    mf = MassFunction.build(**{"transfer.model": "EH"}, disk_cache=tmp_path)
    assert config.to_config(mf)["linear_power"]["transfer"]["disk_cache"] == {
        "directory": str(tmp_path)
    }
    path = config.dump_toml(mf, tmp_path / "mf.toml")
    loaded = config.load_toml(path)
    assert loaded == mf
    assert loaded.linear_power.transfer.disk_cache == mf.linear_power.transfer.disk_cache


# ---------------------------------------------------------------------------------
# Partial configurations
# ---------------------------------------------------------------------------------


def test_partial_config_takes_the_defaults_of_build():
    text = (
        """
sigma_8 = 0.8
fit = "ST"
"transfer.model" = {model = "CAMB", k_max = {value = 10.0, unit = "littleh / Mpc"}}
"fit.a" = 0.75

[linear_power.transfer.k_accuracy]
dln_k = 0.05

[mass_accuracy]
dlog10_m = 0.05

"""
        + HEADER
    )
    expected = MassFunction.build(
        sigma_8=0.8,
        fit=FittingFunction.get("ST")(a=0.75),
        **{"transfer.model": CAMB(k_max=10 * h_Mpc)},
        k_accuracy=KAccuracy(dln_k=0.05),
        mass_accuracy=MassAccuracy(dlog10_m=0.05),
    )
    assert config.loads_toml(text) == expected
    # Only the header: every default, n_s and sigma_8 from the cosmology.
    assert config.loads_toml(HEADER) == MassFunction.build()


def test_partial_model_table_changes_the_default_model():
    text = "[fit]\nA_200 = 0.2\n[filter]\nmodel = 'SharpK'\n" + HEADER
    expected = MassFunction.build(fit=Tinker08(A_200=0.2), filter="SharpK")
    assert config.loads_toml(text) == expected
    doc = {"hmf": {"schema_version": 1}, "fit.A_200": 0.2, "filter": "SharpK"}
    assert config.from_config(doc, stage=MassFunction) == expected


def test_partial_config_without_n_s_needs_it_in_the_cosmology_meta():
    cosmo = config.unstructure(FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05))
    with pytest.raises(ValueError, match="does not give n_s"):
        config.from_config({"hmf": {"schema_version": 1}, "cosmology": cosmo}, stage=MassFunction)
    with pytest.raises(ValueError, match="does not give n_s"):
        MassFunction.build(cosmology=FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05))


# ---------------------------------------------------------------------------------
# Bad configurations
# ---------------------------------------------------------------------------------


def test_unknown_names_raise_with_suggestions():
    with pytest.raises(TypeError, match="Did you mean 'sigma_8'"):
        config.loads_toml("sigma8 = 0.8\n" + HEADER)
    with pytest.raises(TypeError, match=r"no field 'aa'\. Did you mean 'a'"):
        config.loads_toml("[fit]\nmodel = 'ST'\naa = 1.0\n" + HEADER)
    with pytest.raises(ModelNotFoundError, match="Did you mean 'Tinker08'"):
        config.loads_toml("fit = 'Tinker8'\n" + HEADER)
    with pytest.raises(TypeError, match="given twice"):
        config.from_config({"hmf": {"schema_version": 1}, "sigma_8": 0.8,
                            "linear_power": {"sigma_8": 0.7}}, stage=MassFunction)  # fmt: skip
    with pytest.raises(TypeError, match="Expected a number"):
        config.loads_toml("delta_c = '1.686'\n" + HEADER)
    with pytest.raises(TypeError, match="Expected a boolean"):
        config.loads_toml("[mass_accuracy]\nsecond_derivative = 'false'\n" + HEADER)


@pytest.mark.parametrize(
    ("header", "match"),
    [
        ("", "not an hmf configuration"),
        ("[hmf]\nschema_version = 2\nstage = 'hmf.core.transfer:Transfer'", "newer"),
        ("[hmf]\nschema_version = '1'\nstage = 'hmf.core.transfer:Transfer'", "integer"),
        ("[hmf]\nschema_version = 0\nstage = 'hmf.core.transfer:Transfer'", "no migration"),
        ("[hmf]\nschema_version = 1", "class of its stage"),
        ("[hmf]\nschema_version = 1\nstage = 'hmf.core.fits:SMT'", "not a stage class"),
    ],
)
def test_bad_headers_raise(header, match):
    with pytest.raises(ValueError, match=match):
        config.loads_toml(header)


def test_stage_class_must_agree():
    with pytest.raises(ValueError, match="not a Transfer"):
        config.loads_toml(HEADER, stage=Transfer)


# ---------------------------------------------------------------------------------
# Schema versions, migrations and package versions
# ---------------------------------------------------------------------------------


def test_migration_from_an_old_schema_is_logged(monkeypatch, caplog):
    def rename(document):
        document["delta_c"] = document.pop("critical_overdensity")
        return document

    monkeypatch.setitem(
        config.MIGRATIONS, 0, config.Migration(0, "critical_overdensity is now delta_c.", rename)
    )
    old = {
        "hmf": {"schema_version": 0, "stage": "hmf.core.mass_function:MassFunction"},
        "critical_overdensity": 1.7,
        "transfer.model": "EH",
    }
    with caplog.at_level(logging.INFO, logger="hmf.core.config"):
        mf = config.from_config(old)
    assert mf == MassFunction.build(delta_c=1.7, **{"transfer.model": "EH"})
    assert "from schema 0 to 1: critical_overdensity is now delta_c." in caplog.text
    assert old["hmf"]["schema_version"] == 0  # the input is not modified
    with pytest.raises(ValueError, match="migrate=True"):
        config.from_config(old, migrate=False)


def test_register_migration(monkeypatch):
    monkeypatch.setattr(config, "MIGRATIONS", {})

    @config.register_migration(0, "Nothing.")
    def nothing(document):
        return document

    assert config.MIGRATIONS[0].function is nothing
    with pytest.raises(ValueError, match="already registered"):
        config.register_migration(0, "Again.")(nothing)


def test_other_package_versions_are_logged_or_raise(monkeypatch, caplog):
    cfg = config.to_config(Transfer(model="EH"))
    cfg["hmf"]["versions"]["camb"] = "0.1"
    with caplog.at_level(logging.WARNING, logger="hmf.core.config"):
        assert config.from_config(cfg) == Transfer(model="EH")
    assert "camb 0.1" in caplog.text
    with pytest.raises(ValueError, match=r"camb 0\.1"):
        config.from_config(cfg, strict=True)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="hmf.core.config"):
        config.from_config(config.to_config(Transfer(model="EH")), strict=True)
    assert not caplog.text


def test_header_records_package_versions():
    versions = config.to_config(Transfer(model="EH"))["hmf"]["versions"]
    assert set(versions) == set(config.PACKAGES)
    assert versions["camb"] is not None
    assert "versions" not in config.to_config(Transfer(model="EH"), versions=False)["hmf"]
    toml = tomllib.loads(config.dumps_toml(Transfer(model="EH")))["hmf"]["versions"]
    assert toml == {k: v for k, v in versions.items() if v is not None}


# ---------------------------------------------------------------------------------
# Content hashes
# ---------------------------------------------------------------------------------


def test_content_hash_is_stable_and_physics_only(tmp_path, monkeypatch):
    mf = MassFunction.build(**{"transfer.model": "EH"})
    key = config.content_hash(mf)
    for other in roundtrip(mf):
        assert config.content_hash(other) == key
    assert config.content_hash(mf.evolve(disk_cache=tmp_path)) == key
    assert config.content_hash(mf.evolve(cosmology=Planck18.clone(name="x", meta={}))) == key
    assert config.content_hash(mf.evolve(sigma_8=0.81)) != key
    assert config.content_hash(mf.evolve(k_accuracy=KAccuracy.fast())) != key
    assert config.content_hash(mf.evolve(cosmology=Planck15, n_s=0.96, sigma_8=0.8)) != key
    # The versions of the packages that compute results are part of it; others not.
    versions = dict(config.package_versions())
    monkeypatch.setattr(config, "package_versions", lambda: {**versions, "numpy": "0.0"})
    assert config.content_hash(mf) == key
    monkeypatch.setattr(config, "package_versions", lambda: {**versions, "camb": "0.0"})
    assert config.content_hash(mf) != key


def test_content_hash_of_a_value_is_its_configuration():
    from hmf.core._serialise import canonical

    assert canonical(CAMB()) == config.unstructure(CAMB())
    assert canonical(KAccuracy()) == attrs.asdict(KAccuracy())
    with pytest.raises(TypeError, match="Expected a stage"):
        config.content_hash(CAMB())


def test_disk_cache_keys_are_unchanged(monkeypatch):
    """The key of a run is a hash of plain data, which the converter leaves as it is.

    A change to this key makes every disk-cache entry miss: change it deliberately.
    """
    monkeypatch.setattr(_boltzmann, "_version", lambda distribution: "1.0")
    inputs = {"cosmology": {"H0": 67.66, "m_nu": [0.06]}, "settings": {}, "x": True}
    assert _boltzmann.run_key("camb", inputs) == (
        "80dd886fa851857c09ca2c928762f05d1818d009669a04caa08fa947355dd61d"
    )


# ---------------------------------------------------------------------------------
# Pickling
# ---------------------------------------------------------------------------------


def _pickled_modules(blob: bytes) -> set[str]:
    """The modules of every global a (protocol 2) pickle refers to."""
    return {arg.split()[0] for op, arg, _ in pickletools.genops(blob) if op.name == "GLOBAL"}


def test_pickle_is_the_config_and_runs_no_boltzmann_code():
    mf = MassFunction.build()  # CAMB
    expected = mf.dndm(m=M, z=0.5)
    for protocol in (2, pickle.HIGHEST_PROTOCOL):
        blob = pickle.dumps(mf, protocol=protocol)
        if protocol == 2:
            modules = _pickled_modules(blob)
            assert "hmf.core.config" in modules
            assert not any(m.split(".")[0] in {"camb", "classy"} for m in modules)
        _boltzmann.clear_memo()
        before = dict(_boltzmann.run_counts())
        other = pickle.loads(blob)
        assert other == mf
        assert np.array_equal(other.dndm(m=M, z=0.5), expected)
        assert dict(_boltzmann.run_counts()) == before


def test_pickle_without_computed_runs_runs_the_code_again():
    mf = MassFunction.build(sigma_8=0.79)
    blob = pickle.dumps(mf)  # nothing computed yet: no runs in it
    _boltzmann.clear_memo()
    before = _boltzmann.run_counts().get("camb", 0)
    _ = pickle.loads(blob).linear_power.transfer.solution
    assert _boltzmann.run_counts()["camb"] == before + 1


def test_deepcopy_is_an_equal_stage():
    mf = MassFunction.build(**{"transfer.model": "EH"}, fit="SMT")
    other = copy.deepcopy(mf)
    assert other == mf
    assert np.array_equal(other.dndm(m=M, z=0.0), mf.dndm(m=M, z=0.0))


# ---------------------------------------------------------------------------------
# The default configuration: a change to a default is a schema change
# ---------------------------------------------------------------------------------

GOLDEN_MESSAGE = f"""
The configuration of MassFunction.build() differs from {GOLDEN.name}: a default changed.

A change to a default changes results, so it is a schema change. To make it:
  1. bump hmf.core.config.SCHEMA_VERSION;
  2. register a migration from the previous schema (hmf.core.config.register_migration),
     which sets the old default in configurations that relied on it;
  3. write the new default configuration to a new golden file, e.g.
       from hmf.core import config
       from hmf.core.mass_function import MassFunction
       config.dump_toml(MassFunction.build(), "<file>", versions=False)
  4. label the PR "changes results".
"""


def test_default_config_is_the_golden_file():
    golden = tomllib.loads(GOLDEN.read_text())
    assert golden["hmf"]["schema_version"] == config.SCHEMA_VERSION, GOLDEN_MESSAGE
    current = tomllib.loads(config.dumps_toml(MassFunction.build(), versions=False))
    assert current == golden, GOLDEN_MESSAGE
    assert config.load_toml(GOLDEN) == MassFunction.build()


# ---------------------------------------------------------------------------------
# The converter's edge cases
# ---------------------------------------------------------------------------------


def test_arrays_and_dtypes_round_trip():
    assert config.unstructure(np.array([1.0, 2.0])) == [1.0, 2.0]
    ints = config.unstructure(np.arange(3))
    assert ints == {"value": [0, 1, 2], "dtype": "int64"}
    assert np.array_equal(config.structure(ints, np.ndarray), np.arange(3))
    assert config.structure(ints, np.ndarray).dtype == np.int64
    assert config.structure([1, 2], np.ndarray).dtype == np.float64
    q = config.unstructure(u.Quantity(np.arange(3), u.Mpc, dtype=np.int64))
    assert q == {"value": [0, 1, 2], "unit": "Mpc", "dtype": "int64"}
    assert config.structure(q, u.Quantity).dtype == np.int64
    assert config.unstructure(np.float32(0.5)) == 0.5
    assert config.unstructure(Path("a/b")) == "a/b"
    assert config.unstructure(h_Mpc) == {"unit": "littleh / Mpc"}


def test_quantity_edge_cases():
    q = 3 * Msun_h
    assert config.structure(q, u.Quantity) == q
    assert config.structure(3.0, u.Quantity) == 3.0  # left to the field's converter
    # A unit with littleh that is not a physical kind of hmf.core is left as it is.
    assert config.structure({"value": 2.0, "unit": "littleh"}, u.Quantity) == 2.0 * littleh
    with pytest.raises(ValueError, match=r"value = \.\.\., unit"):
        config.structure({"value": 2.0}, u.Quantity)
    with pytest.raises(ValueError, match=r"value = \.\.\., unit"):
        config.structure({"value": 2.0, "unit": "Mpc", "scale": 1}, u.Quantity)


def test_plain_value_edge_cases():
    import enum

    class Colour(enum.Enum):
        RED = "red"

    assert config.unstructure(Colour.RED) == "red"
    assert config.structure("red", Colour) is Colour.RED
    assert config.structure(Colour.RED, Colour) is Colour.RED
    with pytest.raises(ValueError, match="'blue' is not a valid"):
        config.structure("blue", Colour)
    with pytest.raises(ValueError, match="Expected one of"):
        config.structure("x", typing.Literal["a", "b"])
    assert config.structure("a", typing.Literal["a", "b"]) == "a"
    assert config.structure("abc", tuple[float, ...]) == "abc"  # the converter rejects it
    assert config.structure([1, "a"], tuple[float, str]) == (1.0, "a")
    assert config.structure([1, 2], tuple) == (1, 2)
    assert config.structure({"a": 1}, tuple[tuple[str, typing.Any], ...]) == (("a", 1),)
    assert config.structure(3, typing.Any) == 3
    assert config.structure(3, "AnUnresolvedAnnotation") == 3
    assert config.structure(3.0, np.ndarray) == 3.0
    assert config.structure(SMT(), FittingFunction) == SMT()
    assert config.structure(None, float | None) is None
    assert config.structure({"none": True}, float | None) is None
    assert config.structure(2, int | str | None) == 2
    with pytest.raises(TypeError, match="Expected an integer"):
        config.structure(True, int)
    with pytest.raises(TypeError, match="Expected a string"):
        config.structure(1, str)
    with pytest.raises(TypeError, match="must be strings"):
        config.unstructure({1: 2})
    with pytest.raises(TypeError, match="can't be written"):
        config.unstructure(object())


def test_class_key_edge_cases():
    from hmf.core.mass_definitions import MassDefinition

    with pytest.raises(TypeError, match="must be an import path"):
        config.structure({"class": 3}, MassDefinition)
    with pytest.raises(ValueError, match="no class can be imported"):
        config.structure({"class": "no.such.module:Thing"}, MassDefinition)
    with pytest.raises(TypeError, match="is not a MassDefinition"):
        config.structure({"class": "hmf.core.fits:SMT"}, MassDefinition)
    with pytest.raises(TypeError, match="not an attrs class"):
        config.structure({"class": "pathlib:PurePath"}, object)
    assert config.structure({"class": "hmf.core.fits:SMT", "a": 0.75}, object) == SMT(a=0.75)
    assert config.structure(KAccuracy.fast(), KAccuracy) == KAccuracy.fast()
    assert config.structure("200m", MassDefinition | None) == "200m"  # read by the field
    assert config.structure(3, object) == 3
    with pytest.raises(ValueError, match="needs its 'model'"):
        config.structure({"a": 0.75}, FittingFunction)
    with pytest.raises(TypeError, match="abstract"):
        config.structure({"model": "hmf.core.fits:FittingFunction"}, FittingFunction)
    assert config.structure(42, FittingFunction) == 42  # the field's converter rejects it


def test_cosmology_edge_cases():
    from astropy.cosmology import FlatwCDM

    data = config.unstructure(Planck18)
    assert data["class"] == "FlatLambdaCDM"
    with pytest.raises(TypeError, match="Expected a FlatwCDM"):
        config.structure(data, FlatwCDM)
    with pytest.raises(ValueError, match="needs its 'class'"):
        config.structure({k: v for k, v in data.items() if k != "class"}, FlatLambdaCDM)
    assert config.structure("Planck18", FlatLambdaCDM) == "Planck18"  # the validator rejects it
    meta = {"masses": [1e10 * Msun_h, 2e10 * Msun_h], "flag": None}
    cosmo = Planck18.clone(meta=meta)
    back = config.loads_toml(config.dumps_toml(Transfer(model="EH", cosmology=cosmo)))
    assert back.cosmology.meta["masses"] == meta["masses"]
    assert back.cosmology.meta["flag"] is None


def test_model_field_paths():
    text = (
        "fit = 'SMT'\nmass_definition = '200c'\nmass_conversion = 'profile'\n"
        "'mass_conversion.concentration.sample' = 'relaxed'\n" + HEADER
    )
    mf = config.loads_toml(text)
    assert mf.mass_conversion == ProfileConversion(concentration=Duffy08(sample="relaxed"))
    with pytest.raises(TypeError, match=r"no field 'A_20'\. Did you mean 'A_200'"):
        config.loads_toml("'fit.A_20' = 0.2\n" + HEADER)
    with pytest.raises(TypeError, match="not a model"):
        config.loads_toml("'fit.A_200.x' = 0.2\n" + HEADER)
    with pytest.raises(ValueError, match="needs its 'model'"):
        config.loads_toml("[mass_conversion.concentration]\nsample = 'relaxed'\n" + HEADER)
    doc = {"hmf": {"schema_version": 1}, "power.species": "tot"}
    with pytest.raises(TypeError, match="no default: give it"):
        config.from_config(doc, stage=MassVariance)


@attrs.frozen(kw_only=True)
class HeaderClash(Stage):
    """A stage with a field named like the header."""

    hmf: float = field(default=1.0, doc="A number.")


def test_header_errors():
    with pytest.raises(TypeError, match="the header's key"):
        config.to_config(HeaderClash())
    with pytest.raises(TypeError, match="Expected a stage"):
        config.to_config(CAMB())
    with pytest.raises(TypeError, match="Did you mean 'stage'"):
        config.from_config({"hmf": {"schema_version": 1, "stag": "x"}})
    with pytest.raises(ValueError, match="must be a table"):
        config.from_config({"hmf": {"schema_version": 1, "versions": "1.0"}}, stage=Transfer)
    with pytest.raises(TypeError, match="must be a string"):
        config.from_config({"hmf": {"schema_version": 1}, 3: 1.0}, stage=Transfer)


def test_from_paths():
    from hmf.core import routing

    lp = _eh_linear_power()
    assert routing.from_paths(MassFunction, {"linear_power": lp, "fit": SMT()}) == (
        MassFunction(linear_power=lp, fit=SMT())
    )
    with pytest.raises(TypeError, match="has no parameter 'sigma8'"):
        routing.from_paths(MassFunction, {"sigma8": 0.8})
    with pytest.raises(TypeError, match="above it"):
        routing.from_paths(MassFunction, {"linear_power": lp, "linear_power.sigma_8": 0.7})
    # A shared value inside a stage given whole is ignored.
    growth = routing.from_paths(
        Growth,
        {"transfer": None, "cosmology": Planck15, "transfer.cosmology": Planck15},
        shared=["cosmology", "transfer.cosmology"],
    )
    assert growth.transfer is None
