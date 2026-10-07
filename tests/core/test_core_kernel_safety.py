"""Tests of the kernel-level entry points of hmf.core and of extrapolation warnings.

* Every public ``*_kernel`` checks its input against its owner's domain, from the
  input's extent, and raises a :class:`DomainError` rather than extrapolate silently.
* The growth kernels apply the checks of the public growth methods; the growth
  solution itself is data, NaN outside its table.
* Every path that evaluates a user-supplied table outside its range warns through one
  mechanism (``TableRange.warn_outside``), once per owner; designed extrapolation and
  the rounding of the k lattice's ends do not warn.
"""

import inspect
import math
import warnings

import numpy as np
import pytest
from power_models import RHO_CRIT0, AnalyticPower, EisensteinHuNoWiggle

import hmf.core.fits
import hmf.core.growth
import hmf.core.mass_variance
import hmf.core.power_source
import hmf.core.transfer
from hmf.core import fits, growth_models, transfer_models
from hmf.core._kernels import growth as kg
from hmf.core.accuracy import KAccuracy
from hmf.core.domain import Domain, DomainError, Interval, check_extent, warn_once
from hmf.core.fits import FitInputs, evaluate_fsigma
from hmf.core.growth import Growth
from hmf.core.mass_variance import MassVariance, n_eff_kernel
from hmf.core.power_source import TableRange, TabulatedPower
from hmf.core.transfer import Transfer
from hmf.core.units import Msun_h, h_Mpc, power_unit, rho_unit
from hmf.exceptions import HMFExtrapolationWarning

DELTA_C = 1.686


def _caught(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    return [str(w.message) for w in caught if issubclass(w.category, HMFExtrapolationWarning)]


def _eds_growth():
    """A growth table on 0 <= z <= 3 of Einstein-de Sitter: D = a, f = 1."""
    z = np.linspace(0.0, 3.0, 31)
    return Growth(model=growth_models.FromArray(z=z, d=1 / (1 + z)))


def _eh_transfer_table(k_min=1e-3, k_max=10.0):
    """A FromArray transfer model: EH98's transfer function, tabulated."""
    k = np.geomspace(k_min, k_max, 80)
    t = Transfer(model="EH_NoBAO").transfer_function(k * h_Mpc)
    return transfer_models.FromArray(k=k * h_Mpc, t=t)


def _eh_power_table(k_min, k_max, n=2000):
    k = np.geomspace(k_min, k_max, n)
    return TabulatedPower(
        k=k * h_Mpc,
        pk=EisensteinHuNoWiggle()(k) * power_unit,
        mean_density=0.3 * RHO_CRIT0 * rho_unit,
    )


# ---------------------------------------------------------------------------------
# The growth kernels
# ---------------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["growth_factor_kernel", "growth_rate_kernel"])
@pytest.mark.parametrize("z", [5.0, -0.5, np.nan, np.inf, np.array([0.5, 5.0])])
def test_growth_kernels_raise_outside_the_table(method, z):
    """Beyond a FromArray table (0 <= z <= 3), or below z = 0, the kernels raise."""
    with pytest.raises(DomainError, match=f"Growth.{method}"):
        getattr(_eds_growth(), method)(z)


@pytest.mark.parametrize("z", [5.0, -0.5])
def test_growth_solution_is_nan_outside_its_table(z):
    """The solution does not extrapolate: bypassing the check gives NaN, not a number."""
    solution = _eds_growth().solution
    for species in ("cb", "tot"):
        assert np.isnan(solution.growth_factor(z, species))
        assert np.isnan(solution.growth_rate(z, species))


@pytest.mark.parametrize("species", ["cb", "tot"])
def test_growth_kernels_match_the_public_methods_bit_for_bit(species):
    g = _eds_growth()
    z = np.linspace(0.0, 3.0, 97)
    np.testing.assert_array_equal(g.growth_factor_kernel(z, species), g.growth_factor(z, species))
    np.testing.assert_array_equal(g.growth_rate_kernel(z, species), g.growth_rate(z, species))
    assert g.growth_factor_kernel(1.0, species).shape == ()


def test_growth_kernels_reproduce_einstein_de_sitter():
    """A table of D = a gives D = 1/(1 + z) and f = 1 between its nodes (physical)."""
    g = _eds_growth()
    z = np.linspace(0.0, 3.0, 101)
    np.testing.assert_allclose(g.growth_factor_kernel(z), 1 / (1 + z), rtol=1e-10)
    np.testing.assert_allclose(g.growth_rate_kernel(z), 1.0, rtol=1e-8)


@pytest.mark.parametrize(
    ("valid", "lower", "lower_open", "upper"),
    [
        (Domain({"z": (0.5, None, "(]")}), 0.5, True, 3.0),  # the model narrows z
        (Domain({"z": (0.0, 2.0)}), 0.0, False, 2.0),  # below the table's z_max
        (Domain({}), 0.0, False, 3.0),  # the model does not bound z: the table does
    ],
)
def test_growth_kernels_check_the_model_domain_and_the_table(
    monkeypatch, valid, lower, lower_open, upper
):
    """The kernels' z interval is the intersection of the valid domain and the table."""
    monkeypatch.setattr(growth_models.FromArray, "valid_domain", valid)
    g = _eds_growth()
    interval = g._z_interval
    assert interval.lower == lower
    assert interval.lower_open is lower_open
    assert interval.upper == pytest.approx(upper, rel=1e-11)
    g.growth_factor_kernel(np.array([lower + 0.01, upper - 0.01]))
    for z in (lower if lower_open else lower - 0.01, upper + 0.01):
        with pytest.raises(DomainError, match=r"Growth\.growth_factor_kernel"):
            g.growth_factor_kernel(z)


def test_growth_kernels_accept_the_last_redshift_of_the_table():
    """Z = z_max (to the stage's round-off tolerance) is in the table, so it is finite."""
    g = Growth()
    z_max = g.solution.z_max
    for z in (z_max, z_max * (1 + 1e-13)):
        d = g.growth_factor_kernel(z)
        assert np.isfinite(d)
        assert d == g.growth_factor(z)
        assert np.isfinite(g.growth_rate_kernel(z))
    # Below the snapping tolerance the table is not extrapolated.
    first = g.solution.tables["cb"].ln_a[0]
    assert np.isnan(g.solution.growth_factor(np.expm1(-(first - 10 * kg.LN_A_ATOL))))


def test_growth_splines_are_bit_identical_inside_the_table():
    """Not extrapolating changes no value inside the table."""
    ln_a = np.linspace(-5, 0, 101)
    d = np.exp(ln_a) * (1 - 0.2 * np.exp(3 * ln_a))
    table = kg.tabulate_growth(ln_a, d)
    assert not table.ln_d_spline.extrapolate
    assert not table.f_spline.extrapolate
    z = np.linspace(0, table.z_max, 997)
    ln_a_z = -np.log1p(z)
    extrapolating = kg.FrozenSpline(table.ln_d_spline.x, table.ln_d_spline.c)
    np.testing.assert_array_equal(table.growth_factor(z), np.exp(extrapolating(ln_a_z)))


# ---------------------------------------------------------------------------------
# One mechanism for extrapolation warnings
# ---------------------------------------------------------------------------------
def test_mass_variance_on_a_user_transfer_table_warns_exactly_once():
    """The path MassFunction will use: Transfer(FromArray).power_kernel() in MassVariance."""
    mv = MassVariance(power=Transfer(model=_eh_transfer_table()).power_kernel())

    def calls():
        mv.sigma(1e12 * Msun_h)
        mv.sigma(np.array([1e10, 1e14]) * Msun_h)
        mv.ln_sigma_and_slope_kernel(1e13)

    messages = _caught(calls)
    assert len(messages) == 1, messages
    assert messages[0].startswith("Transfer (FromArray): k below the table's smallest")
    assert "above the table's largest wavenumber, 10 h/Mpc" in messages[0]
    assert "with the shape of EH98" in messages[0]


def test_the_transfer_stage_and_its_power_kernel_warn_through_the_same_table_range():
    t = Transfer(model=_eh_transfer_table())
    power = t.power_kernel()
    assert power.table_range is t._table_range
    table = power.table_range
    assert table is not None
    assert table.k_min == pytest.approx(1e-3, rel=1e-12)
    assert table.k_max == pytest.approx(10.0, rel=1e-12)
    # The kernel itself is silent.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        power.ln_power_kernel(np.log([1e-6, 1e3]))
        power.power_kernel(np.array([1e-6, 1e3]))
    # The stage warns once per end, for itself.
    assert len(_caught(lambda: t.transfer_function([1e-6, 1e3] * h_Mpc))) == 1
    assert _caught(lambda: t.transfer_function([1e-6, 1e3] * h_Mpc)) == []


@pytest.mark.parametrize("model", ["EH", "CAMB"])
def test_default_transfer_and_mass_variance_are_silent(model):
    """Designed extrapolation (a Boltzmann tail, a fitting formula) never warns."""
    t = Transfer(model=model)
    assert t.power_kernel().table_range is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mv = MassVariance(power=t.power_kernel())
        mv.sigma(np.geomspace(1e2, 1e17, 50) * Msun_h)
        mv.dlnsigma_dlnm(1e12 * Msun_h)


def _k_grid_ends():
    """The unrounded ends of the default MassVariance k grid, in h/Mpc."""
    mv = MassVariance(power=AnalyticPower(EisensteinHuNoWiggle()))
    grid = mv._k_grid
    k_min = math.exp(KAccuracy().ln_k_min)
    assert grid.k[0] < k_min  # the lattice rounds its end outwards: 9.8e-9 h/Mpc
    return k_min, grid.k[-1]


def test_a_table_reaching_the_grid_ends_does_not_warn():
    """A table from exactly 1e-8 h/Mpc covers the k grid: its first node is rounding."""
    k_min, k_top = _k_grid_ends()
    mv = MassVariance(power=_eh_power_table(k_min, 2 * k_top))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mv.sigma(1e12 * Msun_h)


def test_a_table_short_of_the_grid_ends_warns():
    k_min, k_top = _k_grid_ends()
    mv = MassVariance(power=_eh_power_table(k_min * (1 + 1e-6), 2 * k_top))
    messages = _caught(lambda: mv.sigma(1e12 * Msun_h))
    assert len(messages) == 1, messages
    assert "TabulatedPower: k below the table's smallest wavenumber, 1e-08 h/Mpc" in messages[0]
    assert "above" not in messages[0]


def test_table_range_warns_once_per_owner_and_end():
    table = TableRange(k_min=1.0, k_max=10.0, where="Test", how="somehow")

    class Owner:
        pass

    owner = Owner()

    def calls():
        table.warn_outside(owner, 2.0, 5.0)  # inside
        table.warn_outside(owner, 0.5, 5.0)  # below
        table.warn_outside(owner, 0.1, 50.0)  # above is new
        table.warn_outside(owner, 0.1, 50.0)  # nothing new
        table.warn_outside(Owner(), 0.1, 50.0)  # another owner: both, in one warning
        table.warn_outside(Owner(), 1.0 - 1e-12, 10.0 + 1e-11, rtol=1e-11)  # tolerated

    both = (
        "Test: k below the table's smallest wavenumber, 1 h/Mpc, and k above the table's "
        "largest wavenumber, 10 h/Mpc, is extrapolated somehow."
    )
    messages = _caught(calls)
    assert messages == [
        "Test: k below the table's smallest wavenumber, 1 h/Mpc, is extrapolated somehow.",
        both,
        both,
    ]


def test_warn_once_with_several_keys():
    class Owner:
        pass

    owner = Owner()

    def calls():
        assert warn_once(owner, "a", "a")
        assert warn_once(owner, ("a", "b"), "a and b")
        assert not warn_once(owner, ("b", "a"), "again")
        assert not warn_once(owner, "b", "b")

    assert _caught(calls) == ["a", "a and b"]


# ---------------------------------------------------------------------------------
# evaluate_fsigma's owner
# ---------------------------------------------------------------------------------
def _warren_outside():
    return FitInputs(sigma=[1.0, 1.0], z=0.0, m=[1e12, 1e16])


def test_evaluate_fsigma_warns_once_per_owner():
    class Owner:
        pass

    model = fits.Warren()
    owner = Owner()

    def calls():
        evaluate_fsigma(model, _warren_outside(), policy="warn", owner=owner)
        evaluate_fsigma(model, _warren_outside(), policy="warn", owner=owner)
        evaluate_fsigma(model, _warren_outside(), policy="warn", owner=Owner())

    messages = _caught(calls)
    assert len(messages) == 2, messages
    assert all("Warren's calibration domain" in m for m in messages)


def test_evaluate_fsigma_warns_once_per_model_by_default():
    model = fits.Warren()

    def calls():
        evaluate_fsigma(model, _warren_outside(), policy="warn")
        evaluate_fsigma(model, _warren_outside(), policy="warn")
        evaluate_fsigma(fits.Warren(), _warren_outside(), policy="warn")

    assert len(_caught(calls)) == 2


# ---------------------------------------------------------------------------------
# check_extent
# ---------------------------------------------------------------------------------
def test_check_extent():
    positive = Interval(0, None, lower_open=True)
    x = np.array([1.0, 2.0])
    assert check_extent("x", x, positive, where="T") is x
    assert check_extent("x", np.array([]), positive, where="T").size == 0
    assert check_extent("x", 3.0, where="T").shape == ()
    for bad in (0.0, -1.0, np.nan, np.inf, [1.0, np.nan]):
        with pytest.raises(DomainError, match="T: x"):
            check_extent("x", bad, positive, where="T")
    # ln values are compared with the bounds of the variable.
    closed = Interval(1.0, 10.0)
    check_extent("ln_k", np.log([1.5, 9.5]), closed, where="T", ln_values=True)
    with pytest.raises(DomainError, match=r"k in \[1, 20\] is outside the domain"):
        check_extent("ln_k", np.log([1.0, 20.0]), closed, where="T", ln_values=True)


# ---------------------------------------------------------------------------------
# Every public kernel checks its input
# ---------------------------------------------------------------------------------
def _fsigma_kernel():
    return fits.PS().fsigma_kernel(FitInputs(sigma=[1.0, -1.0], delta_c=DELTA_C))


def _modify_dndm(fit, **bad):
    args = {"m": np.array([1e12, 1e13]), "z": 0.5, "h": 0.7, "omega_m0": 0.3} | bad
    return fit.modify_dndm_kernel(
        args["m"], np.ones(2), z=args["z"], ngtm=np.ones(2), h=args["h"], omega_m0=args["omega_m0"]
    )


def _mass_ratio(**bad):
    args = {"m": np.array([1e13, 1e14]), "z": 0.5, "h": 0.7, "omega_m0": 0.3} | bad
    return fits.Bocquet200cDMOnly().mass_ratio_to_200m_kernel(
        args["m"], z=args["z"], omega_m0=args["omega_m0"], h=args["h"]
    )


def _ups(model="EH"):
    return Transfer(model=model).power_kernel()


#: One out-of-domain call for each public kernel: (its qualified name, the call).
KERNEL_CASES = [
    ("Growth.growth_factor_kernel", lambda: _eds_growth().growth_factor_kernel(5.0)),
    ("Growth.growth_rate_kernel", lambda: _eds_growth().growth_rate_kernel(-0.5)),
    (
        "MassVariance.ln_sigma_and_slope_kernel",
        lambda: MassVariance(power=_ups()).ln_sigma_and_slope_kernel([1e12, -1.0]),
    ),
    ("n_eff_kernel", lambda: n_eff_kernel([0.1, np.nan])),
    (
        "TabulatedPower.ln_power_kernel",
        lambda: _eh_power_table(1e-3, 10.0, n=50).ln_power_kernel(np.array([0.0, np.nan])),
    ),
    ("UnnormalisedPower.ln_power_kernel", lambda: _ups().ln_power_kernel(np.array([np.inf]))),
    ("UnnormalisedPower.power_kernel", lambda: _ups().power_kernel(np.array([1.0, 0.0]))),
    ("FittingFunction.fsigma_kernel", _fsigma_kernel),
    (
        "MeasuredMassDefinition.delta_halo_mean_kernel",
        lambda: fits.Tinker08.measured_mass_definition.delta_halo_mean_kernel([0.3, -0.1]),
    ),
    ("FittingFunction.modify_dndm_kernel", lambda: _modify_dndm(fits.PS(), m=-1.0)),
    ("Behroozi.modify_dndm_kernel z", lambda: _modify_dndm(fits.Behroozi(), z=-0.5)),
    ("Behroozi.modify_dndm_kernel h", lambda: _modify_dndm(fits.Behroozi(), h=0.0)),
    (
        "Bocquet200cDMOnly.modify_dndm_kernel",
        lambda: _modify_dndm(fits.Bocquet200cDMOnly(), omega_m0=np.nan),
    ),
    ("Bocquet200mDMOnly.mass_ratio_to_200m_kernel", lambda: _mass_ratio(m=0.0)),
    ("Bocquet200mDMOnly.mass_ratio_to_200m_kernel z", lambda: _mass_ratio(z=np.inf)),
]


@pytest.mark.parametrize(("name", "call"), KERNEL_CASES, ids=[c[0] for c in KERNEL_CASES])
def test_every_kernel_raises_outside_its_domain(name, call):
    with pytest.raises(DomainError):
        call()


def test_tabulated_power_kernel_with_extension_raise():
    source = TabulatedPower(
        k=np.geomspace(1e-3, 10.0, 50) * h_Mpc,
        pk=np.ones(50) * power_unit,
        mean_density=1.0 * rho_unit,
        extension="raise",
    )
    with pytest.raises(DomainError, match=r"1 value\(s\) of k above the table"):
        source.ln_power_kernel(np.log([1.0, 20.0]))


#: Kernels whose argument is not a domain value: Transfer.power_kernel takes a species
#: (a bad one is a ValueError) and returns a PowerSource.
_FACTORIES = {"Transfer.power_kernel"}


def _public_kernels():
    """The qualified names of every public *_kernel of the stage and model modules."""
    names = set()
    modules = (hmf.core.fits, hmf.core.growth, hmf.core.mass_variance, hmf.core.power_source)
    for module in (*modules, hmf.core.transfer):
        for obj_name, obj in vars(module).items():
            if obj_name.startswith("_"):
                continue
            if inspect.isfunction(obj) and obj_name.endswith("_kernel"):
                if obj.__module__ == module.__name__:
                    names.add(obj_name)
            elif inspect.isclass(obj) and obj.__module__ == module.__name__:
                for attr, value in vars(obj).items():
                    if attr.endswith("_kernel") and not attr.startswith("_") and callable(value):
                        names.add(f"{obj.__name__}.{attr}")
    return names


def test_every_public_kernel_has_an_out_of_domain_case():
    covered = {name.split(" ")[0] for name, _ in KERNEL_CASES}
    kernels = _public_kernels() - _FACTORIES
    # The PowerSource protocol only declares the method.
    kernels.discard("PowerSource.ln_power_kernel")
    # Overrides are covered through the checking method of their base class.
    overrides = {
        "Bocquet200cDMOnly.mass_ratio_to_200m_kernel",
        "Bocquet500cDMOnly.mass_ratio_to_200m_kernel",
    }
    assert not (kernels & overrides), "overrides must go through the checked base method"
    missing = kernels - covered
    assert not missing, f"public kernels without an out-of-domain test: {sorted(missing)}"


def test_kernel_checks_do_not_change_values():
    """Inside the domain the checks are transparent: kernels match the public methods."""
    out = _modify_dndm(fits.Behroozi())
    assert out.shape == (2,)
    assert np.all(np.isfinite(out))
    mdef = fits.Tinker08.measured_mass_definition
    om = np.array([0.3, 0.5])
    np.testing.assert_array_equal(mdef.delta_halo_mean_kernel(om), np.full(2, 200.0))
    np.testing.assert_array_equal(n_eff_kernel([-0.5, 0.0]), [0.0, -3.0])
