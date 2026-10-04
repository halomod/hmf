"""Regression tests for API-hygiene fixes: update(), validation, the registry, defaults and nu."""

import types
import warnings

import numpy as np
import pytest
from astropy.cosmology import Planck15, Planck18

from hmf import MassFunction, Transfer, get_mdl
from hmf._internals._cache import cached_quantity, parameter, subframework
from hmf._internals._framework import Component, Framework, get_base_component, pluggable
from hmf.alternatives import wdm
from hmf.cosmology import Cosmology
from hmf.cosmology.cosmo import DEFAULT_COSMOLOGY
from hmf.halos import mass_definitions as md
from hmf.mass_function import fitting_functions as ff

# A coarse grid keeps these tests fast.
FAST = {"transfer_model": "EH", "lnk_min": -8, "lnk_max": 3, "dlnk": 0.1, "dlog10m": 0.25}


@pytest.fixture
def mf():
    return MassFunction(Mmin=10, Mmax=15, **FAST)


# ---------------------------------------------------------------------------
# update() is atomic and only accepts parameters
# ---------------------------------------------------------------------------
def test_update_failed_validation_restores_parameters(mf):
    dndm = mf.dndm.copy()
    with pytest.raises(ValueError, match="Mmin must be less than Mmax"):
        mf.update(Mmin=16)

    assert mf.Mmin == 10
    assert mf.Mmax == 15
    np.testing.assert_allclose(mf.dndm, dndm, rtol=1e-12)


def test_update_failed_validation_restores_every_parameter(mf):
    with pytest.raises(ValueError, match="Mmin must be less than Mmax"):
        mf.update(z=1.0, Mmax=12, Mmin=13)

    assert (mf.z, mf.Mmin, mf.Mmax) == (0.0, 10, 15)


def test_update_failed_setter_restores_earlier_parameters(mf):
    # z=-1 is rejected by the z setter, after Mmin has already been set.
    with pytest.raises(ValueError, match="z must be"):
        mf.update(Mmin=11, z=-1)

    assert mf.Mmin == 10
    assert mf.z == 0.0


def test_update_rollback_restores_dicts_exactly():
    mf = MassFunction(hmf_model="ST", hmf_params={"a": 0.75}, **FAST)
    with pytest.raises(ValueError, match="Mmin must be less than Mmax"):
        # A non-empty dict is merged into the stored one. The rollback must also drop
        # the key that the failed update added, not just re-merge the old dict.
        mf.update(hmf_params={"p": 0.2}, Mmin=20)

    assert mf.hmf_params == {"a": 0.75}


def test_update_rejects_methods(mf):
    with pytest.raises(ValueError, match=r"Invalid arguments.*'validate'"):
        mf.update(validate=None)

    # The object still works: validate() was not replaced.
    mf.validate()
    mf.update(Mmin=11)
    assert mf.Mmin == 11


def test_update_rejects_quantities(mf):
    with pytest.raises(ValueError, match=r"Invalid arguments.*'dndm'"):
        mf.update(dndm=1)


def test_update_rejects_unknown_names_before_setting_anything(mf):
    with pytest.raises(ValueError, match="did you mean 'Mmin'"):
        mf.update(Mmax=14, Mmn=11)

    assert mf.Mmax == 15


def test_update_subclass_parameters():
    """Parameters added by subclasses (as halomod does) can be updated."""

    class _MFWithExtra(MassFunction):
        def __init__(self, extra=1.0, **kwargs):
            super().__init__(**kwargs)
            self.extra = extra

        @parameter("param")
        def extra(self, val):
            """An extra parameter."""
            return float(val)

        @cached_quantity
        def scaled_dndm(self):
            return self.extra * self.dndm

        def validate(self):
            super().validate()
            if self.extra < 0:
                raise ValueError("extra must be non-negative")

    mf = _MFWithExtra(extra=2.0, **FAST)
    first = mf.scaled_dndm.copy()

    mf.update(extra=3.0)
    np.testing.assert_allclose(mf.scaled_dndm, 1.5 * first, rtol=1e-12)

    with pytest.raises(ValueError, match="extra must be non-negative"):
        mf.update(extra=-1.0, z=1.0)
    assert mf.extra == 3.0
    assert mf.z == 0.0


class _Inner(Framework):
    def __init__(self, a=1.0):
        super().__init__()
        self.a = a

    @parameter("param")
    def a(self, val):
        """A parameter."""
        return val


class _Outer(Framework):
    def __init__(self, b=1.0):
        super().__init__()
        self.b = b

    @parameter("param")
    def b(self, val):
        """A parameter."""
        return val

    @subframework
    def inner(self) -> _Inner:
        return _Inner()

    def validate(self):
        super().validate()
        if self.inner.a > self.b:
            raise ValueError("inner.a must not exceed b")


def test_update_subframework_params():
    outer = _Outer(b=5.0)
    outer.update(inner_params={"a": 2.0})
    assert outer.inner.a == 2.0


def test_update_rolls_back_subframework_params():
    outer = _Outer(b=5.0)
    with pytest.raises(ValueError, match=r"inner\.a must not exceed b"):
        outer.update(inner_params={"a": 10.0})
    assert outer.inner.a == 1.0


# ---------------------------------------------------------------------------
# Validation raises ValueError, not AssertionError (asserts vanish under -O)
# ---------------------------------------------------------------------------
def test_mass_range_validation_is_value_error():
    with pytest.raises(ValueError, match="Mmin must be less than Mmax"):
        MassFunction(Mmin=15, Mmax=10, **FAST)


def test_k_range_validation_is_value_error():
    with pytest.raises(ValueError, match="lnk_min must be less than lnk_max"):
        Transfer(transfer_model="EH", lnk_min=3, lnk_max=-3)


def test_k_length_validation_is_value_error():
    with pytest.raises(ValueError, match="at least 2 entries"):
        Transfer(transfer_model="EH", lnk_min=0, lnk_max=0.05, dlnk=0.1)


# ---------------------------------------------------------------------------
# Error messages are strings, and say what the check does
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("cls", "kwargs", "expected"),
    [
        (Transfer, {"z": -1}, "z must be ≥ 0, got -1.0"),
        (Transfer, {"z": "high"}, "z must be a number, got 'high'"),
        (MassFunction, {"delta_c": -1}, "delta_c must be > 0, got -1.0"),
        (MassFunction, {"delta_c": 11}, "delta_c must be ≤ 10, got 11.0"),
        (MassFunction, {"delta_c": "x"}, "delta_c must be a number, got 'x'"),
        (wdm.MassFunctionWDM, {"wdm_mass": 0}, "wdm_mass must be > 0, got 0.0"),
        (wdm.MassFunctionWDM, {"wdm_mass": "x"}, "wdm_mass must be a number, got 'x'"),
    ],
)
def test_parameter_error_messages(cls, kwargs, expected):
    with pytest.raises(ValueError) as exc:
        cls(transfer_model="EH", **kwargs)
    # A tuple-shaped message would be "('z must be > 0 (', -1.0, ')')".
    assert str(exc.value) == expected


def test_redshift_zero_is_allowed():
    assert Transfer(transfer_model="EH", z=0).z == 0.0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_get_models_is_read_only():
    models = ff.BaseFittingFunction.get_models()
    assert isinstance(models, types.MappingProxyType)
    assert models["PS"] is ff.PS
    with pytest.raises(TypeError):
        models["PS"] = ff.ST
    assert ff.BaseFittingFunction.get_models()["PS"] is ff.PS


def test_registering_a_name_from_another_module_warns():
    try:
        with pytest.warns(UserWarning) as record:
            type("Tinker08", (ff.PS,), {"__module__": "my_package.models"})
        msg = str(record[0].message)
        assert "my_package.models.Tinker08" in msg
        assert "hmf.mass_function.fitting_functions.Tinker08" in msg
    finally:
        ff.BaseFittingFunction._plugins["Tinker08"] = ff.Tinker08


def test_reregistering_from_the_same_module_does_not_warn():
    @pluggable
    class _Kind(Component):
        pass

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        first = type("Model", (_Kind,), {"__module__": "my_package.models"})
        # As happens when a module is reloaded.
        second = type("Model", (_Kind,), {"__module__": "my_package.models"})

    assert first is not second
    assert _Kind.get_models()["Model"] is second


def test_get_mdl_errors_use_readable_names():
    with pytest.raises(ValueError, match=r"builtins\.dict is not a BaseFittingFunction") as exc:
        get_mdl(dict, "BaseFittingFunction")
    assert "<class" not in str(exc.value)

    with pytest.raises(ValueError) as exc:
        get_base_component("NotAComponent")
    assert "<class" not in str(exc.value)
    assert "BaseFittingFunction" in str(exc.value)


def test_get_mdl_ambiguous_name_warning_is_readable():
    # FromFile is both a growth-factor and a transfer model.
    with pytest.warns(UserWarning, match="More than one model") as record:
        get_mdl("FromFile")
    msg = str(record[0].message)
    assert "<class" not in msg
    assert "hmf.cosmology.growth_factor.FromFile" in msg
    assert "hmf.density_field.transfer_models.FromFile" in msg


# ---------------------------------------------------------------------------
# Components default to the same cosmology as the frameworks
# ---------------------------------------------------------------------------
def test_default_cosmology_is_framework_default():
    assert DEFAULT_COSMOLOGY is Planck18
    assert Cosmology().cosmo_model is DEFAULT_COSMOLOGY


def test_mass_definition_default_cosmology_matches_framework():
    mf = MassFunction(**FAST)
    mdef = md.SOMean(overdensity=200)
    # Physical check: the default mean density is that of the framework's cosmology,
    # and not that of Planck15 (which differs by ~1%).
    assert np.isclose(mdef.mean_density(0), mf.mean_density0, rtol=1e-10)
    assert not np.isclose(mdef.mean_density(0), mdef.mean_density(0, cosmo=Planck15), rtol=1e-3)
    assert np.isclose(mdef.m_to_r(1e12), mdef.m_to_r(1e12, cosmo=mf.cosmo), rtol=1e-12)


def test_fitting_function_default_cosmology_matches_framework():
    # Tinker08 depends on the cosmology through the mean-density overdensity of a
    # critical-density mass definition, Delta_c / Omega_m(z).
    mf = MassFunction(hmf_model="Tinker08", mdef_model="SOCritical", z=0.5, **FAST)
    fit = ff.Tinker08(nu2=mf.nu2, m=mf.m, z=0.5, mass_definition=md.SOCritical(overdensity=200))
    np.testing.assert_allclose(fit.fsigma, mf.fsigma, rtol=1e-10)


def test_wdm_default_cosmology_matches_framework():
    np.testing.assert_allclose(
        wdm.Viel05(mx=1.0).m_hm, wdm.Viel05(mx=1.0, cosmo=Planck18).m_hm, rtol=1e-12
    )
    assert not np.isclose(
        wdm.Viel05(mx=1.0).m_hm, wdm.Viel05(mx=1.0, cosmo=Planck15).m_hm, rtol=1e-3
    )


# ---------------------------------------------------------------------------
# nu is deprecated in favour of peak_height and nu2
# ---------------------------------------------------------------------------
def test_peak_height_is_delta_c_over_sigma(mf):
    np.testing.assert_allclose(mf.peak_height, mf.delta_c / mf.sigma, rtol=1e-12)
    np.testing.assert_allclose(mf.nu2, mf.peak_height**2, rtol=1e-12)
    # It is the same nu as the fitting function's.
    np.testing.assert_allclose(mf.peak_height, mf.hmf.nu, rtol=1e-12)
    # The peak height grows with mass, and is of order unity around M* at z=0.
    assert np.all(np.diff(mf.peak_height) > 0)
    assert mf.peak_height[0] < 1 < mf.peak_height[-1]


def test_mass_function_nu_is_deprecated(mf):
    with pytest.warns(DeprecationWarning, match=r"squared.*v4.*nu2.*peak_height"):
        nu = mf.nu
    np.testing.assert_allclose(nu, mf.nu2, rtol=1e-12)


def test_filter_nu_is_deprecated(mf):
    r = mf.radii
    with pytest.warns(DeprecationWarning, match=r"squared.*v4.*nu2"):
        nu = mf.filter.nu(r, mf.delta_c)
    np.testing.assert_allclose(nu, mf.filter.nu2(r, mf.delta_c), rtol=1e-12)


@pytest.mark.parametrize("filter_model", ["TopHat", "SharpKEllipsoid"])
def test_internal_quantities_do_not_use_deprecated_nu(filter_model):
    mf = MassFunction(filter_model=filter_model, **FAST)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        mf.dndm
        mf.nu_fn
        mf.n_eff_at_collapse
        mf.mass_nonlinear


def test_fitting_function_nu_is_not_deprecated(mf):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        np.testing.assert_allclose(mf.hmf.nu, np.sqrt(mf.hmf.nu2), rtol=1e-12)
