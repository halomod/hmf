"""Tests of the LinearPower stage: sigma_8 normalisation and P(k, z)."""

import math
import pickle
import warnings

import astropy.units as u
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM, Planck18
from power_models import UnitTransfer
from scipy.integrate import simpson

from hmf.core import linear_power as lp_module
from hmf.core.accuracy import KAccuracy
from hmf.core.domain import DomainError
from hmf.core.growth import Growth
from hmf.core.linear_power import SIGMA_8_RADIUS, LinearPower
from hmf.core.mass_variance import MassVariance
from hmf.core.transfer import Transfer
from hmf.core.transfer_models import FromArray
from hmf.core.units import Mpc_h, UnitBoundaryError, h_Mpc, power_unit
from hmf.exceptions import HMFExtrapolationWarning


def _linear_power(transfer: Transfer, **kwargs) -> LinearPower:
    return LinearPower(transfer=transfer, growth=Growth.from_transfer(transfer), **kwargs)


@pytest.fixture(scope="module")
def eh():
    return Transfer(model="EH")


@pytest.fixture(scope="module")
def lp(eh):
    return _linear_power(eh, sigma_8=0.8)


def _tophat_sigma(k, pk, r):
    """sigma(R) from P(k) by direct quadrature with the top-hat window (independent of hmf)."""
    x = k * r
    w = 3 * (np.sin(x) - x * np.cos(x)) / x**3
    return math.sqrt(simpson(k**3 * pk * w**2, x=np.log(k)) / (2 * math.pi**2))


# ---------------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize("sigma_8", [0.7, 0.8102, 0.95])
def test_power_has_the_sigma_8_it_is_normalised_to(eh, sigma_8):
    """Integrating the normalised P(k, 0) with a top-hat of 8 Mpc/h gives sigma_8.

    The integral is an independent quadrature (scipy's Simpson on a fine grid), not the
    MassVariance the amplitude comes from. EH has the same T for both species.
    """
    stage = _linear_power(eh, sigma_8=sigma_8)
    k = np.exp(np.arange(np.log(1e-6), np.log(1e3), 0.002))
    pk = stage.power(k=k * h_Mpc, z=0.0).to_value(power_unit)
    assert _tophat_sigma(k, pk, SIGMA_8_RADIUS) == pytest.approx(sigma_8, rel=1e-5)


def test_amplitude_scales_as_sigma_8_squared(lp):
    """A = (sigma_8 / sigma_8,raw)^2: doubling sigma_8 quadruples the power, exactly."""
    doubled = lp.evolve(sigma_8=2 * lp.sigma_8)
    assert doubled.amplitude == pytest.approx(4 * lp.amplitude, rel=1e-15)
    k = [0.01, 0.1, 1.0] * h_Mpc
    np.testing.assert_allclose(doubled.power(k=k, z=0.5), 4 * lp.power(k=k, z=0.5), rtol=1e-14)


def test_power_grows_as_the_growth_factor_squared(lp):
    """P(k, z) = D(z)^2 P(k, 0), the scale-independent growth of the default."""
    k = np.logspace(-3, 1, 9) * h_Mpc
    z = np.array([0.5, 1.0, 3.0])
    ratio = lp.power(k=k[None, :], z=z[:, None]) / lp.power(k=k, z=0.0)[None, :]
    d = lp.growth.growth_factor(z)
    np.testing.assert_allclose(ratio, np.broadcast_to(d[:, None] ** 2, ratio.shape), rtol=1e-14)


def test_power_law_shape():
    """With T = 1 the power is A k^n: its log-slope is n_s everywhere."""
    transfer = Transfer(model=UnitTransfer(), n_s=-2.0)
    stage = _linear_power(transfer, sigma_8=0.8)
    k = np.logspace(-3, 2, 6)
    p = stage.power_kernel(k=k, z=0.0)
    np.testing.assert_allclose(np.diff(np.log(p)) / np.diff(np.log(k)), -2.0, rtol=1e-12)


def test_sigma_8_species_with_massive_neutrinos():
    """With massive neutrinos, the CDM + baryon power normalised to the total sigma_8.

    T_cb > T_tot on small scales (neutrinos do not cluster there), so with the same
    total-matter sigma_8 the CDM + baryon field has more power, by a fraction set by
    f_nu = Omega_nu / Omega_m (here 3 x 0.1 eV, f_nu ~ 0.02).
    """
    cosmo = FlatLambdaCDM(H0=67.66, Om0=0.30966, Ob0=0.04897, Tcmb0=2.7255, m_nu=[0.1] * 3 * u.eV)
    transfer = Transfer(cosmology=cosmo, model="CAMB")
    tot = _linear_power(transfer, sigma_8=0.8, species="tot", sigma_8_species="tot")
    cb = _linear_power(transfer, sigma_8=0.8, species="cb", sigma_8_species="tot")
    assert cb.unnormalised_sigma_8 == tot.unnormalised_sigma_8  # both normalise to tot
    k = np.exp(np.arange(np.log(1e-6), np.log(1e3), 0.002))
    s_tot = _tophat_sigma(k, tot.power_kernel(k=k, z=0.0), SIGMA_8_RADIUS)
    s_cb = _tophat_sigma(k, cb.power_kernel(k=k, z=0.0), SIGMA_8_RADIUS)
    assert s_tot == pytest.approx(0.8, rel=1e-5)
    assert 0.8 < s_cb < 0.8 * 1.05
    # On large scales neutrinos cluster like CDM: the powers agree there.
    assert cb.power_kernel(k=1e-4, z=0.0) == pytest.approx(
        tot.power_kernel(k=1e-4, z=0.0), rel=1e-3
    )


# ---------------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------------


def test_defaults_come_from_the_transfer_stage(eh):
    stage = _linear_power(eh, sigma_8=0.8)
    assert stage.cosmology is eh.cosmology
    assert stage.k_accuracy == eh.k_accuracy
    assert stage.species == "cb"
    assert stage.sigma_8_species == "tot"
    assert stage.power_source == eh.power_source("cb")


def test_units_and_scalar_rule(lp):
    p = lp.power(k=0.1 * h_Mpc, z=0.0)
    assert p.unit == power_unit
    assert p.shape == ()
    p_phys = lp.power(k=0.1 * Planck18.h / u.Mpc, z=0.0)
    assert p_phys.to_value(power_unit) == pytest.approx(p.to_value(power_unit), rel=1e-14)
    assert lp.power(k=[0.1, 1.0] * h_Mpc, z=0.0).shape == (2,)
    with pytest.raises(UnitBoundaryError):
        lp.power(k=0.1, z=0.0)
    assert np.ndim(lp.sigma_scale_kernel(0.5)) == 0
    assert lp.sigma_scale_kernel(0.0) == pytest.approx(math.sqrt(lp.amplitude), rel=1e-14)


def test_kernel_equals_method(lp):
    k = np.logspace(-3, 1, 7)
    np.testing.assert_array_equal(lp.power(k=k * h_Mpc, z=1.0).value, lp.power_kernel(k=k, z=1.0))


def test_domains_raise(lp):
    with pytest.raises(DomainError):
        lp.power(k=-1.0 * h_Mpc, z=0.0)
    with pytest.raises(DomainError):
        lp.power(k=0.1 * h_Mpc, z=-0.5)
    with pytest.raises(DomainError):
        lp.power_kernel(k=np.array([0.1, np.nan]), z=0.0)
    with pytest.raises(DomainError):
        lp.sigma_scale_kernel(np.inf)


def test_inconsistent_stages_raise(eh):
    other = Growth(cosmology=Planck18.clone(Om0=0.25))
    with pytest.raises(ValueError, match="cosmology"):
        LinearPower(transfer=eh, growth=other, sigma_8=0.8)
    with pytest.raises(ValueError, match="KAccuracy"):
        LinearPower(
            transfer=eh, growth=Growth.from_transfer(eh, k_accuracy=KAccuracy.fast()), sigma_8=0.8
        )
    with pytest.raises(ValueError, match="KAccuracy"):
        _linear_power(eh, sigma_8=0.8, k_accuracy=KAccuracy.high())
    with pytest.raises(ValueError, match="sigma_8"):
        _linear_power(eh, sigma_8=-0.8)
    with pytest.raises(ValueError, match="species"):
        _linear_power(eh, sigma_8=0.8, species="nu")


def test_evolve_shares_the_unnormalised_sigma_8(monkeypatch, eh):
    """Changing sigma_8 (or species) reuses sigma_8,raw; changing what it depends on doesn't."""
    calls = []
    real = MassVariance.ln_sigma_at_radius_kernel

    def counting(self, r):
        calls.append(r)
        return real(self, r)

    monkeypatch.setattr(MassVariance, "ln_sigma_at_radius_kernel", counting)
    stage = _linear_power(eh, sigma_8=0.8)
    raw = stage.unnormalised_sigma_8
    assert len(calls) == 1
    for changed in (stage.evolve(sigma_8=0.7), stage.evolve(species="tot")):
        assert changed.unnormalised_sigma_8 == raw
    assert len(calls) == 1

    # A different transfer (n_s) changes sigma_8,raw: it is computed again.
    tilted = eh.evolve(n_s=0.9)
    new = stage.evolve(transfer=tilted, growth=Growth.from_transfer(tilted))
    assert new.unnormalised_sigma_8 != raw
    assert len(calls) == 2
    assert new.unnormalised_sigma_8 == _linear_power(tilted, sigma_8=0.8).unnormalised_sigma_8


def test_evolve_is_atomic(lp):
    before = lp.power(k=0.1 * h_Mpc, z=0.0)
    with pytest.raises(ValueError):
        lp.evolve(sigma_8=-1.0)
    with pytest.raises(TypeError, match=r"no parameter 'sigma8'\. Did you mean 'sigma_8'"):
        lp.evolve(sigma8=0.7)
    assert lp.sigma_8 == 0.8
    assert lp.power(k=0.1 * h_Mpc, z=0.0) == before


def test_pickle_and_copy(lp):
    import copy

    _ = lp.amplitude
    for other in (pickle.loads(pickle.dumps(lp)), copy.deepcopy(lp)):
        assert other == lp
        assert other.amplitude == lp.amplitude
        assert other.power(k=0.3 * h_Mpc, z=1.0) == lp.power(k=0.3 * h_Mpc, z=1.0)


def test_user_table_warns_once():
    k = np.logspace(-3, 1, 50)
    t = 1 / (1 + (k / 0.05) ** 2)
    transfer = Transfer(model=FromArray(k=k * h_Mpc, t=t))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", HMFExtrapolationWarning)  # the sigma_8 k grid
        stage = _linear_power(transfer, sigma_8=0.8)
        _ = stage.amplitude
    with pytest.warns(HMFExtrapolationWarning, match="above the table"):
        stage.power(k=[0.1, 100.0] * h_Mpc, z=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stage.power(k=[0.1, 100.0] * h_Mpc, z=0.0)  # once per stage


def test_sigma_8_radius_is_8_mpc_h():
    assert SIGMA_8_RADIUS == 8.0
    assert lp_module.__all__ == ["SIGMA_8_RADIUS", "LinearPower"]
    assert (SIGMA_8_RADIUS * Mpc_h).unit == Mpc_h


def test_power_arguments_are_keyword_only(lp):
    with pytest.raises(TypeError):
        lp.power(0.1 * h_Mpc, 0.0)
    with pytest.raises(TypeError):
        lp.power_kernel(np.array([0.1]), 0.0)
