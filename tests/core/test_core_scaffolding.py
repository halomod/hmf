"""Tests of the scaffolding shared by stages and models.

* ``Model.coerce``, the converter of every model field, for each kind;
* :class:`~hmf.core.stage.CosmologyStage`, the base of the stages with a cosmology;
* the model-kind mixins, and the registries they must leave unchanged;
* the explicit precision thresholds of the Boltzmann codes;
* the tables (transfer, growth, power) built on ``FrozenSpline``, against a direct
  :class:`scipy.interpolate.CubicSpline`.
"""

import math

import astropy.units as u
import attrs
import numpy as np
import pytest
from astropy.cosmology import WMAP9, Planck18, w0waCDM
from scipy.interpolate import CubicSpline

from hmf.core import transfer_models as tm
from hmf.core._kernels import growth as kg
from hmf.core._kernels import transfer as kt
from hmf.core.accuracy import KAccuracy
from hmf.core.filters import Filter, SharpK, TopHat
from hmf.core.fits import PS, FittingFunction, Tinker08
from hmf.core.growth import Growth
from hmf.core.growth_models import CambGrowth, ClassGrowth, GrowthModel, ODEGrowth
from hmf.core.mass_variance import MassVariance
from hmf.core.model import ModelNotFoundError
from hmf.core.power_source import TabulatedPower
from hmf.core.stage import CosmologyStage
from hmf.core.transfer import Transfer
from hmf.core.transfer_models import CAMB, CLASS, EH, TransferModel
from hmf.core.units import h_Mpc, power_unit, rho_unit

#: (kind, a registered name, the class it names, a model of another kind).
KINDS = [
    (TransferModel, "EH", EH, ODEGrowth()),
    (GrowthModel, "ODE", ODEGrowth, EH()),
    (Filter, "SharpK", SharpK, EH()),
    (FittingFunction, "PS", PS, TopHat()),
]
KIND_IDS = [k[0].__name__ for k in KINDS]


# ---------------------------------------------------------------------------------
# Model.coerce
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize(("kind", "name", "cls", "other"), KINDS, ids=KIND_IDS)
def test_coerce_returns_an_instance_unchanged(kind, name, cls, other):
    model = cls()
    assert kind.coerce(model) is model


@pytest.mark.parametrize(("kind", "name", "cls", "other"), KINDS, ids=KIND_IDS)
def test_coerce_instantiates_a_name_an_import_path_or_a_class(kind, name, cls, other):
    for value in (name, cls.qualified_name(), f"{cls.__module__}.{cls.__qualname__}", cls):
        got = kind.coerce(value)
        assert type(got) is cls
        assert got == cls()


@pytest.mark.parametrize(("kind", "name", "cls", "other"), KINDS, ids=KIND_IDS)
def test_coerce_rejects_other_types(kind, name, cls, other):
    for value in (other, 3.0, None, ["EH"]):
        with pytest.raises(TypeError) as info:
            kind.coerce(value)
        assert str(info.value) == (
            f"Expected a {kind.__name__}: an instance, a model class or a registered "
            f"name, not {type(value).__name__}."
        )


@pytest.mark.parametrize(("kind", "name", "cls", "other"), KINDS, ids=KIND_IDS)
def test_coerce_rejects_a_class_of_another_kind(kind, name, cls, other):
    with pytest.raises(TypeError, match=f"which is not a {kind.__name__}"):
        kind.coerce(type(other))


@pytest.mark.parametrize(("kind", "name", "cls", "other"), KINDS, ids=KIND_IDS)
def test_coerce_rejects_an_unknown_name(kind, name, cls, other):
    with pytest.raises(ModelNotFoundError, match=f"No {kind.__name__} model is known as"):
        kind.coerce(name + "_typo")


def test_coerce_on_a_model_class_checks_against_that_class():
    """Coercing through a subclass of a kind only accepts that subclass's models."""
    assert type(EH.coerce("EH")) is EH
    with pytest.raises(TypeError, match="not a EH"):
        EH.coerce("BBKS")


@pytest.mark.parametrize(
    ("make", "field", "kind", "name"),
    [
        (lambda v: Transfer(model=v), "model", TransferModel, "BBKS"),
        (lambda v: Growth(model=v), "model", GrowthModel, "Integral"),
        (lambda v: MassVariance(power=_power(), filter=v), "filter", Filter, "SmoothK"),
    ],
    ids=["Transfer.model", "Growth.model", "MassVariance.filter"],
)
def test_stage_model_fields_use_coerce(make, field, kind, name):
    """Each model field converts with its kind's coerce, and rejects a bad value."""
    with pytest.raises(TypeError, match=f"Expected a {kind.__name__}"):
        make(1.5)
    assert type(getattr(make(name), field)) is kind.get(name)


# ---------------------------------------------------------------------------------
# CosmologyStage
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize("stage", [Transfer, Growth])
def test_cosmology_stages_share_the_base(stage):
    assert issubclass(stage, CosmologyStage)
    (cosmology,) = (a for a in attrs.fields(stage) if a.name == "cosmology")
    assert cosmology.default is Planck18
    assert stage.fields_info()[0].name == "cosmology"
    assert stage.fields_info()[-1].name == "disk_cache"


def test_field_order_and_names():
    assert [f.name for f in Transfer.fields_info()] == [
        "cosmology",
        "model",
        "n_s",
        "k_accuracy",
        "disk_cache",
    ]
    assert [f.name for f in Growth.fields_info()] == [
        "cosmology",
        "model",
        "transfer",
        "k_accuracy",
        "disk_cache",
    ]
    assert [f.name for f in MassVariance.fields_info()] == [
        "power",
        "filter",
        "mass_accuracy",
        "k_accuracy",
        "truncation_rtol",
    ]


def test_cosmology_is_compared_by_value_not_name():
    unnamed = Planck18.clone(name="not Planck18")
    assert Transfer(model="EH") == Transfer(model="EH", cosmology=unnamed)
    assert hash(Transfer(model="EH")) == hash(Transfer(model="EH", cosmology=unnamed))
    assert Transfer(model="EH") != Transfer(model="EH", cosmology=WMAP9)


def test_disk_cache_is_not_part_of_the_value(tmp_path):
    assert Transfer(model="EH") == Transfer(model="EH", disk_cache=tmp_path)
    assert Transfer(model="EH", disk_cache=True).disk_cache is not None
    assert Growth(disk_cache=False).disk_cache is None


def test_same_cosmology():
    t = Transfer(model="EH")
    assert t.same_cosmology(Growth())
    assert t.same_cosmology(Growth(cosmology=Planck18.clone(name="x")))
    assert not t.same_cosmology(Growth(cosmology=WMAP9))


def test_growth_rejects_a_transfer_with_another_cosmology():
    with pytest.raises(ValueError, match="the cosmology differs from the transfer stage's"):
        Growth(cosmology=WMAP9, transfer=Transfer(model="EH"))


@pytest.mark.parametrize("stage", [Transfer(model="EH"), Growth()])
def test_unit_context_has_the_cosmology_h0(stage):
    """The units context converts physical units with the cosmology's H0."""
    assert stage._unit_context.H0 == stage.cosmology.H0
    # k = 1/Mpc is 1/h h/Mpc.
    if isinstance(stage, Transfer):
        h = stage.cosmology.h
        np.testing.assert_allclose(
            stage.transfer_function(h / u.Mpc), stage.transfer_function(1 * h_Mpc), rtol=1e-14
        )


def test_model_must_apply_to_the_cosmology():
    """The shared model validator calls the model's check_cosmology."""
    cosmo = Planck18.clone(Ob0=0)
    with pytest.raises(ValueError, match="needs a cosmology with baryons"):
        Transfer(model="EH_BAO", cosmology=cosmo)
    with pytest.raises(ValueError, match="set the baryon density"):
        Growth(model="CAMB", cosmology=cosmo)


# ---------------------------------------------------------------------------------
# Model-kind mixins and registries
# ---------------------------------------------------------------------------------

#: Every alias of the kinds whose scaffolding moved into the mixins, and the class it
#: names. Registries, aliases and qualified names must not change.
REGISTERED = {
    TransferModel: {
        "BBKS": "hmf.core.transfer_models:BBKS",
        "BondEfs": "hmf.core.transfer_models:BondEfs",
        "CAMB": "hmf.core.transfer_models:CAMB",
        "CLASS": "hmf.core.transfer_models:CLASS",
        "EH": "hmf.core.transfer_models:EH",
        "EH_BAO": "hmf.core.transfer_models:EH_BAO",
        "EH_NoBAO": "hmf.core.transfer_models:EH_NoBAO",
        "FromArray": "hmf.core.transfer_models:FromArray",
        "FromFile": "hmf.core.transfer_models:FromFile",
    },
    GrowthModel: {
        "CAMB": "hmf.core.growth_models:CambGrowth",
        "CLASS": "hmf.core.growth_models:ClassGrowth",
        "Carroll92": "hmf.core.growth_models:Carroll92Growth",
        "Eisenstein97": "hmf.core.growth_models:Eisenstein97Growth",
        "FromArray": "hmf.core.growth_models:FromArray",
        "FromFile": "hmf.core.growth_models:FromFile",
        "GenMF": "hmf.core.growth_models:GenMFGrowth",
        "Heath77": "hmf.core.growth_models:Heath77Growth",
        "Integral": "hmf.core.growth_models:IntegralGrowth",
        "ODE": "hmf.core.growth_models:ODEGrowth",
    },
}


@pytest.mark.parametrize("kind", list(REGISTERED), ids=lambda k: k.__name__)
def test_registry_is_unchanged(kind):
    assert dict(kind.get_aliases()) == REGISTERED[kind]
    assert sorted(kind.get_models()) == sorted(REGISTERED[kind].values())


@pytest.mark.parametrize("kind", [k[0] for k in KINDS], ids=KIND_IDS)
def test_every_registered_name_resolves(kind):
    for alias, qualname in kind.get_aliases().items():
        assert kind.get(alias).qualified_name() == qualname
    for qualname, cls in kind.get_models().items():
        assert kind.get(qualname) is cls
        assert cls.qualified_name() == qualname


@pytest.mark.parametrize("kind", [TransferModel, GrowthModel], ids=lambda k: k.__name__)
def test_kind_scaffolding(kind):
    assert kind.backend is None
    assert kind.calibration_domain is None
    assert kind.valid_domain is not None
    # The default check accepts any cosmology, even one without baryons.
    kind.get("EH_NoBAO" if kind is TransferModel else "ODE")().check_cosmology(
        Planck18.clone(Ob0=0)
    )


@pytest.mark.parametrize(
    ("model", "backend"),
    [(CAMB(), "camb"), (CLASS(), "class"), (CambGrowth(), "camb"), (ClassGrowth(), "class")],
    ids=["CAMB", "CLASS", "CambGrowth", "ClassGrowth"],
)
def test_boltzmann_backed_models_check_the_cosmology(model, backend):
    """Transfer and growth models of a Boltzmann code share their cosmology checks."""
    name = type(model).__name__
    assert model.backend == backend
    model.check_cosmology(Planck18)
    with pytest.raises(ValueError, match=f"To use {name}, set the baryon density"):
        model.check_cosmology(Planck18.clone(Ob0=0))
    with pytest.raises(ValueError, match=f"To use {name}, set the CMB temperature"):
        model.check_cosmology(Planck18.clone(Tcmb0=0, m_nu=0))
    model.check_cosmology(w0waCDM(H0=70, Om0=0.3, Ode0=0.7, Ob0=0.05, Tcmb0=2.7255))


def test_models_have_no_instance_dict():
    """The mixins are slotted, so the models stay slotted."""
    for model in (CAMB(), CambGrowth(), ODEGrowth(), EH()):
        assert not hasattr(model, "__dict__")


def test_z_max_is_shared_by_camb_and_camb_growth():
    """CambGrowth(z_max) makes the run of CAMB(z_max)."""
    assert CAMB(z_max=5.0).run_input(Planck18, KAccuracy())["growth"]["z_max"] == 5.0
    assert CLASS(z_max=5.0).run_input(Planck18, KAccuracy())["class"]["z_max_pk"] == 5.0
    with pytest.raises(TypeError):
        CAMB(z_max_growth=5.0)


# ---------------------------------------------------------------------------------
# The Boltzmann codes' precision thresholds
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dln_k", "k_per_logint", "high_precision", "k_per_decade"),
    [
        (0.05, 0, False, None),  # KAccuracy.fast()
        (0.02, 0, False, None),  # the threshold itself: the codes' own sampling
        (0.019, 11, True, 25),
        (0.01, 20, True, 47),
        (0.005, 40, True, 93),  # KAccuracy.high()
    ],
)
def test_boltzmann_precision_is_pinned(dln_k, k_per_logint, high_precision, k_per_decade):
    """CAMB and CLASS sampling depend on dln_k through fixed thresholds only.

    They do not follow the KAccuracy defaults: these values hold whatever those are.
    """
    acc = KAccuracy(dln_k=dln_k)
    transfer = CAMB().run_input(Planck18, acc)["transfer"]
    assert transfer["k_per_logint"] == k_per_logint
    assert transfer["high_precision"] is high_precision
    params = CLASS().run_input(Planck18, acc)["class"]
    assert params.get("k_per_decade_for_pk") == k_per_decade


def test_boltzmann_precision_thresholds():
    assert tm._CAMB_FINE_K_BELOW_DLN_K == 0.02
    assert tm._CLASS_FINE_K_BELOW_DLN_K == 0.02
    # k_per_logint is a fifth of 1/dln_k per e-fold, and k_per_decade the same per
    # decade.
    assert tm._camb_k_per_logint(KAccuracy(dln_k=0.004)) == math.ceil(0.2 / 0.004)
    assert tm._class_k_per_decade(KAccuracy(dln_k=0.004)) == math.ceil(0.2 * math.log(10) / 0.004)


# ---------------------------------------------------------------------------------
# Tables on FrozenSpline, against a direct CubicSpline
# ---------------------------------------------------------------------------------


def _power():
    k = np.logspace(-4, 2, 200)
    return TabulatedPower(
        k=k * h_Mpc,
        pk=1e4 * k / (1 + (k / 0.02) ** 2.5) ** 1.6 * power_unit,
        mean_density=8.5e10 * rho_unit,
    )


def test_tabulated_power_matches_a_direct_cubic_spline():
    """TabulatedPower's interpolation and power-law tails, bit for bit."""
    source = _power()
    ln_k_tab, ln_p_tab = source._table
    spline = CubicSpline(ln_k_tab, ln_p_tab)
    lo, hi = ln_k_tab[0], ln_k_tab[-1]
    k = np.logspace(-9, 6, 3001)
    ln_k = np.log(k)
    ln_p = spline(np.clip(ln_k, lo, hi))
    ln_p = np.where(ln_k < lo, ln_p + spline(lo, 1) * (ln_k - lo), ln_p)
    ln_p = np.where(ln_k > hi, ln_p + spline(hi, 1) * (ln_k - hi), ln_p)
    got = np.exp(source.ln_power_kernel(np.log(k)))
    np.testing.assert_array_equal(got, np.exp(ln_p))


def test_growth_table_matches_a_direct_cubic_spline():
    ln_a = np.linspace(-5, 0, 101)
    d = np.exp(ln_a) * (1 - 0.2 * np.exp(3 * ln_a))
    table = kg.tabulate_growth(ln_a, d)
    ln_d = np.log(d) - np.log(d[-1])
    ln_d_spline = CubicSpline(ln_a, ln_d)
    f_spline = CubicSpline(ln_a, ln_d_spline(ln_a, 1))
    z = np.linspace(0, table.z_max, 997)
    np.testing.assert_array_equal(table.growth_factor(z), np.exp(ln_d_spline(-np.log1p(z))))
    np.testing.assert_array_equal(table.growth_rate(z), f_spline(-np.log1p(z)))


def test_tabulated_transfer_matches_a_direct_cubic_spline():
    scales = tm._eh98_scales(Planck18)
    k = np.logspace(-4, 1.5, 120)
    t = np.exp(kt.ln_t_eh98_no_wiggle(k, scales)) * (1 + 0.05 * np.sin(k / 0.01) * k / (1 + k))
    table = kt.tabulate_transfer(k, t, scales)
    residual = np.log(t) - kt.ln_t_eh98_no_wiggle(k, scales)
    residual = residual - residual[0]
    spline = CubicSpline(
        np.log(k), residual, bc_type=((1, 0.0), (1, table.slope_high)), extrapolate=False
    )
    ln_k = np.linspace(np.log(k[0]), np.log(k[-1]), 1501)
    np.testing.assert_array_equal(table.ln_residual(ln_k), spline(ln_k))


def test_tinker_tables():
    """The tabulated parameters are read from the fields, in the order of delta_tab."""
    from hmf.core.fits import Tinker10, _delta_table

    t08 = Tinker08(A_300=0.5)
    assert _delta_table(t08, "A")[1] == 0.5
    np.testing.assert_array_equal(
        _delta_table(t08, "c"), [getattr(t08, f"c_{d}") for d in Tinker08.delta_tab]
    )
    t10 = Tinker10()
    np.testing.assert_array_equal(
        _delta_table(t10, "alpha"), [getattr(t10, f"alpha_{d}") for d in Tinker10.delta_tab]
    )
