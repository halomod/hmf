"""Tests of the error, domain and extrapolation policy of hmf.core.

* The exception rule: a :class:`DomainError` for an input value outside what can be
  evaluated, a :class:`ValueError` for a bad option, configuration or combination
  (including a model that does not apply to a cosmology), a :class:`TypeError` for an
  object of the wrong kind. One case per site that raises.
* Every registered model declares both domains, and the stages check their inputs
  against their model's (class's) valid domain.
* Extrapolating beyond a table the user supplied warns, once per instance; the
  default configurations warn about nothing.
"""

import warnings

import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM, Flatw0wzCDM, LambdaCDM, Planck18, wCDM
from power_models import RHO_CRIT0, AnalyticPower, EisensteinHuNoWiggle

from hmf.core import fits, growth_models, transfer_models
from hmf.core.accuracy import MassAccuracy
from hmf.core.domain import Domain, DomainError
from hmf.core.filters import Filter, SharpK, TopHat
from hmf.core.fits import FitInputs, FittingFunction, evaluate_fsigma
from hmf.core.growth import Growth
from hmf.core.growth_models import GrowthModel
from hmf.core.mass_variance import MassVariance
from hmf.core.power_source import TabulatedPower
from hmf.core.transfer import Transfer
from hmf.core.transfer_models import TransferModel
from hmf.core.units import Msun_h, h_Mpc, power_unit, rho_unit
from hmf.exceptions import HMFExtrapolationWarning

DELTA_C = 1.686
EH = AnalyticPower(EisensteinHuNoWiggle())
OPEN_LAMBDA = LambdaCDM(H0=70, Om0=0.3, Ode0=0.6, Ob0=0.05, Tcmb0=2.7255)
WCDM = wCDM(H0=70, Om0=0.3, Ode0=0.7, w0=-0.9)


def _table_power(k_min=1e-3, k_max=10.0, **kwargs):
    k = np.geomspace(k_min, k_max, 60)
    return TabulatedPower(
        k=k * h_Mpc,
        pk=EisensteinHuNoWiggle()(k) * power_unit,
        mean_density=0.3 * RHO_CRIT0 * rho_unit,
        **kwargs,
    )


def _eh_table():
    """A FromArray transfer model: EH98's transfer function, tabulated."""
    k = np.geomspace(1e-3, 10.0, 80)
    t = Transfer(model="EH_NoBAO").transfer_function(k * h_Mpc)
    return transfer_models.FromArray(k=k * h_Mpc, t=t)


# ---------------------------------------------------------------------------------
# The exception rule, site by site
# ---------------------------------------------------------------------------------
def _mv(**kwargs):
    return MassVariance(power=EH, **kwargs)


#: (id, a call that raises, the type it must raise). "domain" is a DomainError,
#: "value" a ValueError that is not a DomainError, "type" a TypeError.
SITES = [
    # MassVariance: masses, sigma and the lattice.
    ("mv-mass-zero", lambda: _mv().sigma(0.0 * Msun_h), "domain"),
    ("mv-mass-nan", lambda: _mv().dlnsigma_dlnm(np.nan * Msun_h), "domain"),
    (
        "mv-outside-lattice",
        lambda: _mv(mass_accuracy=MassAccuracy(extension="raise")).sigma(1e18 * Msun_h),
        "domain",
    ),
    ("mv-low-mass-end", lambda: _mv().sigma(1e-12 * Msun_h), "domain"),
    (
        "mv-unresolved",
        lambda: MassVariance(
            power=AnalyticPower(lambda k: np.ones_like(k)),
            mass_accuracy=MassAccuracy(second_derivative=False),
        ).sigma(1e12 * Msun_h),
        "domain",
    ),
    ("mv-sigma-out-of-range", lambda: _mv().m_from_sigma(1e6), "domain"),
    ("mv-sigma-negative", lambda: _mv().m_from_sigma(-1.0), "domain"),
    (
        "mv-power-not-positive",
        lambda: MassVariance(power=AnalyticPower(lambda k: -np.ones_like(k))).sigma(1e12 * Msun_h),
        "value",
    ),
    ("mv-not-a-power-source", lambda: MassVariance(power=object()), "type"),
    # TabulatedPower.
    ("power-k-outside", lambda: _table_power(extension="raise").power(100.0 * h_Mpc), "domain"),
    ("power-k-zero", lambda: _table_power().power(0.0 * h_Mpc), "domain"),
    ("power-extension-option", lambda: _table_power(extension="clip"), "value"),
    # Transfer and its models.
    ("transfer-k-zero", lambda: Transfer(model="EH").transfer_function(0.0 * h_Mpc), "domain"),
    (
        "transfer-k-negative",
        lambda: Transfer(model="BBKS").unnormalised_power(-1.0 * h_Mpc),
        "domain",
    ),
    (
        "transfer-no-baryons",
        lambda: Transfer(cosmology=FlatLambdaCDM(H0=70, Om0=0.3), model="EH"),
        "value",
    ),
    (
        "transfer-boltzmann-cosmology-class",
        lambda: Transfer(cosmology=Flatw0wzCDM(H0=70, Om0=0.3, Ob0=0.05, Tcmb0=2.7), model="CAMB"),
        "value",
    ),
    (
        "transfer-fromarray-t-length",
        lambda: transfer_models.FromArray(k=[0.1, 1.0] * h_Mpc, t=[1.0]),
        "value",
    ),
    (
        "transfer-fromarray-t_tot-length",
        lambda: transfer_models.FromArray(k=[0.1, 1.0] * h_Mpc, t=[1.0, 0.5], t_tot=[1.0]),
        "value",
    ),
    ("transfer-class-params", lambda: transfer_models.CLASS(class_params={"h": 0.7}), "value"),
    ("transfer-camb-setting-kind", lambda: transfer_models.CAMB(settings={"a": [1]}), "type"),
    ("transfer-model-kind", lambda: Transfer(model=3), "type"),
    # Growth and its models.
    ("growth-z-negative", lambda: Growth().growth_factor(-0.1), "domain"),
    ("growth-z-nan", lambda: Growth().growth_rate(np.nan), "domain"),
    (
        "growth-z-beyond-table",
        lambda: Growth(
            model=growth_models.FromArray(z=[0, 1, 2, 3], d=[1, 0.6, 0.4, 0.3])
        ).growth_factor(3.5),
        "domain",
    ),
    ("growth-integral-wcdm", lambda: Growth(cosmology=WCDM, model="Integral"), "value"),
    ("growth-genmf-wcdm", lambda: Growth(cosmology=WCDM, model="GenMF"), "value"),
    (
        "growth-eisenstein97-open",
        lambda: Growth(cosmology=OPEN_LAMBDA, model="Eisenstein97"),
        "value",
    ),
    ("growth-heath77-lambda", lambda: Growth(cosmology=Planck18, model="Heath77"), "value"),
    (
        "growth-genmf-closed",
        lambda: Growth(cosmology=LambdaCDM(H0=70, Om0=0.5, Ode0=0.7), model="GenMF"),
        "value",
    ),
    (
        "growth-fromarray-no-z0",
        lambda: growth_models.FromArray(z=[1, 2, 3, 4], d=[1, 0.6, 0.4, 0.3]),
        "value",
    ),
    (
        "growth-cosmology-differs",
        lambda: Growth(cosmology=OPEN_LAMBDA, transfer=Transfer(model="EH")),
        "value",
    ),
    ("growth-model-kind", lambda: Growth(model=3.0), "type"),
    # Filters.
    ("filter-x-negative", lambda: TopHat().window(-1.0), "domain"),
    ("filter-x-nan", lambda: TopHat().dwindow_dlnx(np.nan), "domain"),
    ("filter-sharpk-derivative", lambda: SharpK().dwindow_dlnx(0.5), "value"),
    # Fits.
    ("fit-sigma-zero", lambda: fits.ST().fsigma(0.0, delta_c=DELTA_C), "domain"),
    ("fit-delta-75", lambda: fits.Tinker08().fsigma(1.0, z=0.0, delta_halo=75.0), "domain"),
    ("fit-n_eff--3", lambda: fits.Reed07().fsigma(1.0, delta_c=DELTA_C, n_eff=-3.0), "domain"),
    (
        "fit-unphysical-parameters",
        lambda: fits.Tinker08(A_3200=-0.1).fsigma(1.0, z=0.0, delta_halo=3200.0),
        "domain",
    ),
    (
        "fit-calibration-raise",
        lambda: evaluate_fsigma(
            fits.Tinker08(), FitInputs(sigma=10.0, z=0.0, delta_halo=200.0), policy="raise"
        ),
        "domain",
    ),
    ("fit-missing-input", lambda: fits.Tinker08().fsigma(1.0, z=0.0), "value"),
    (
        "fit-unknown-policy",
        lambda: evaluate_fsigma(fits.PS(), FitInputs(sigma=1.0, delta_c=DELTA_C), policy="no"),
        "value",
    ),
    ("fit-bad-parameters", lambda: fits.Bhattacharya(p=1.0, q=1.0), "value"),
    # Domains.
    ("domain-unknown-variable", lambda: Domain({"z": (0, 1)}).contains(zz=0.5), "value"),
    ("domain-bad-brackets", lambda: Domain({"z": (0, 1, "{}")}), "value"),
]

_EXPECTED = {"domain": DomainError, "value": ValueError, "type": TypeError}


@pytest.mark.filterwarnings("ignore::hmf.exceptions.HMFExtrapolationWarning")
@pytest.mark.parametrize(("call", "kind"), [s[1:] for s in SITES], ids=[s[0] for s in SITES])
def test_exception_rule(call, kind):
    """Each site raises exactly the type the rule gives it."""
    with pytest.raises(_EXPECTED[kind]) as info:
        call()
    if kind == "value":
        assert not isinstance(info.value, DomainError)
    # A DomainError is a ValueError, so ``except ValueError`` catches every bad input.
    if kind == "domain":
        assert isinstance(info.value, ValueError)


def test_open_bound_message_says_greater_than():
    """Tinker08's Delta > 75 is an open bound, and the error describes it as one."""
    with pytest.raises(DomainError, match="75 < delta_halo <= 30000") as info:
        fits.Tinker08().fsigma(1.0, z=0.0, delta_halo=75.0)
    assert ">= 75" not in str(info.value)
    assert np.isfinite(fits.Tinker08().fsigma(1.0, z=0.0, delta_halo=np.nextafter(75.0, 76.0)))


# ---------------------------------------------------------------------------------
# Domains of models and stages
# ---------------------------------------------------------------------------------
KINDS = [TransferModel, GrowthModel, Filter, FittingFunction]


@pytest.mark.parametrize("kind", KINDS, ids=lambda k: k.__name__)
def test_every_registered_model_declares_both_domains(kind):
    models = kind.get_models()
    assert models
    for name, cls in models.items():
        assert isinstance(cls.valid_domain, Domain), name
        assert cls.valid_domain.variables, name
        # No calibration domain is None; a calibration domain bounds something.
        calibration = cls.calibration_domain
        assert calibration is None or (isinstance(calibration, Domain) and calibration.variables)


def test_transfer_domain_is_open_at_zero():
    """T(k) is defined for k > 0: the declared domain agrees with what the stage does."""
    domain = TransferModel.valid_domain
    assert not domain.contains(k=0.0 * h_Mpc)
    assert domain.contains(k=1e-300 * h_Mpc)
    assert domain.describe() == "k > 0 littleh / Mpc"


@pytest.mark.parametrize("model", ["EH", "BBKS"])
def test_transfer_checks_its_models_domain(model):
    t = Transfer(model=model)
    assert t.transfer_function(1e-10 * h_Mpc) > 0
    with pytest.raises(DomainError, match=rf"Transfer \({model}\).*k > 0"):
        t.transfer_function(np.array([0.1, 0.0]) * h_Mpc)


class _NarrowEH(transfer_models.EH_NoBAO, abstract=True):
    """EH_NoBAO, valid only for k <= 1 h/Mpc (not registered: abstract)."""

    valid_domain = Domain({"k": (0 * h_Mpc, 1 * h_Mpc, "(]")})


class _NarrowODE(growth_models.ODEGrowth, abstract=True):
    """ODEGrowth, valid only for z <= 5 (not registered: abstract)."""

    valid_domain = Domain({"z": (0, 5)})


def test_a_transfer_model_can_narrow_its_domain():
    t = Transfer(model=_NarrowEH())
    t.transfer_function(1.0 * h_Mpc)
    with pytest.raises(DomainError, match="k <= 1"):
        t.transfer_function(2.0 * h_Mpc)
    with pytest.raises(DomainError):
        t.unnormalised_power(2.0 * h_Mpc)


def test_a_growth_model_can_narrow_its_domain():
    g = Growth(model=_NarrowODE())
    g.growth_factor(5.0)
    with pytest.raises(DomainError, match="0 <= z <= 5"):
        g.growth_factor(6.0)
    with pytest.raises(DomainError):
        g.growth_rate(6.0)


def test_growth_checks_the_tables_redshifts():
    g = Growth(model=growth_models.FromArray(z=[0, 1, 2, 3], d=[1, 0.6, 0.4, 0.3]))
    g.growth_factor(3.0)
    with pytest.raises(DomainError, match=r"solution's table.*z <= 3"):
        g.growth_factor(3.01)


# ---------------------------------------------------------------------------------
# Extrapolation warnings
# ---------------------------------------------------------------------------------
def _caught(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    return [str(w.message) for w in caught if issubclass(w.category, HMFExtrapolationWarning)]


def test_user_transfer_table_warns_once_per_stage_and_end():
    model = _eh_table()
    t = Transfer(model=model)

    def calls():
        t.transfer_function(np.array([0.01, 1.0]) * h_Mpc)  # inside: no warning
        t.transfer_function(np.array([20.0, 50.0]) * h_Mpc)
        t.unnormalised_power(100.0 * h_Mpc)
        t.transfer_function(1e-5 * h_Mpc)
        t.unnormalised_power(1e-6 * h_Mpc)

    messages = _caught(calls)
    assert len(messages) == 2, messages
    assert "above the table's largest wavenumber, 10 h/Mpc" in messages[0]
    assert "below the table's smallest wavenumber, 0.001 h/Mpc" in messages[1]
    # Another stage, even an equal one, warns again.
    assert len(_caught(lambda: Transfer(model=model).transfer_function(20.0 * h_Mpc))) == 1


def test_user_transfer_file_warns(tmp_path):
    k = np.geomspace(1e-3, 10.0, 80)
    t = Transfer(model="EH_NoBAO").transfer_function(k * h_Mpc)
    fname = tmp_path / "transfer.dat"
    np.savetxt(fname, np.column_stack([k, t]))
    stage = Transfer(model=transfer_models.FromFile(fname=fname))
    with pytest.warns(HMFExtrapolationWarning, match=r"Transfer \(FromFile\): k above"):
        stage.transfer_function(30.0 * h_Mpc)


def test_boltzmann_tail_does_not_warn():
    """Extrapolation beyond a Boltzmann code's k_max is by design: no warning."""
    t = Transfer(model="CAMB")
    assert t.k_max_table is not None
    k = np.geomspace(1e-7, 1e4, 50) * h_Mpc
    assert np.any(k > t.k_max_table)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t.transfer_function(k)
        t.unnormalised_power(k, species="tot")


def test_tabulated_power_warns_once_in_mass_variance():
    """A TabulatedPower narrower than MassVariance's k grid warns once for each end."""
    mv = MassVariance(power=_table_power())
    messages = _caught(lambda: (mv.sigma(1e12 * Msun_h), mv.dlnsigma_dlnm(1e13 * Msun_h)))
    assert len(messages) == 2, messages
    assert "below the table" in messages[0]
    assert "above the table" in messages[1]


# ---------------------------------------------------------------------------------
# Default configurations warn about nothing
# ---------------------------------------------------------------------------------
@pytest.fixture
def no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


K = np.geomspace(1e-7, 1e4, 200) * h_Mpc
Z = np.linspace(0.0, 20.0, 41)


@pytest.mark.parametrize("model", ["EH", "CAMB"])
def test_default_transfer_and_growth_emit_no_warnings(no_warnings, model):
    transfer = Transfer(model=model)
    for species in ("cb", "tot"):
        transfer.transfer_function(K, species=species)
        transfer.unnormalised_power(K, species=species)
    for growth in (Growth(), Growth.from_transfer(transfer, model="CAMB")):
        growth.growth_factor(Z)
        growth.growth_rate(Z)


def test_default_mass_variance_emits_no_warnings(no_warnings):
    """Default MassVariance settings, on an analytic source and on a wide enough table."""
    m = np.geomspace(1.0, 10**17.5, 200) * Msun_h
    k = np.geomspace(1e-9, 1e7, 1000)
    table = TabulatedPower(
        k=k * h_Mpc,
        pk=EisensteinHuNoWiggle()(k) * power_unit,
        mean_density=0.3 * RHO_CRIT0 * rho_unit,
    )
    for power in (EH, table):
        mv = MassVariance(power=power)
        s = mv.sigma(m)
        mv.dlnsigma_dlnm(m)
        mv.m_from_sigma(s[10:-10])
