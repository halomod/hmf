"""Physical tests of the warm dark matter models.

References: the defining property of the half-mode scale (Schneider et al. 2012:
the WDM/CDM transfer function equals 1/2 there), the large-scale CDM limit, the
CDM limit of an infinitely heavy particle, and the free-streaming and half-mode
masses tabulated by Schneider et al. (2012).
"""

import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

from hmf import MassFunction
from hmf.alternatives import wdm

MODELS = [wdm.Viel05, wdm.Bode01]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("mx", [0.5, 1.0, 3.0, 10.0])
def test_transfer_is_one_half_at_half_mode_scale(model, mx):
    """The half-mode wavenumber k_hm = 2 pi / lambda_hm is where T_WDM/T_CDM = 1/2."""
    w = model(mx=mx)
    # Tolerance: closed form inversion; floating-point only.
    assert w.transfer(2 * np.pi / w.lam_hm) == pytest.approx(0.5, rel=1e-10)


@pytest.mark.parametrize("model", MODELS)
def test_transfer_shape(model):
    """Free-streaming only erases small-scale power.

    T -> 1 on large scales, falls monotonically, and tends to zero far below the
    free-streaming length.
    """
    w = model(mx=1.0)
    k = np.logspace(-4, 3, 200)
    t = w.transfer(k)
    # Tolerance: at k = 1e-4 h/Mpc, (alpha k)^{2 mu} ~ 1e-13 so T = 1 - O(1e-12).
    assert t[0] == pytest.approx(1.0, abs=1e-10)
    assert np.all(np.diff(t) < 0)
    assert t[-1] < 1e-6


# Schneider et al. (2012) Table 1: (m_WDM [keV], M_fs, M_hm [Msun/h]).
_SCHNEIDER12_TABLE1 = [
    (1.25, 2.3e6, 6.3e9),
    (1.0, 4.9e6, 1.3e10),
    (0.75, 1.3e7, 3.4e10),
    (0.5, 4.9e7, 1.3e11),
    (0.25, 5.0e8, 1.3e12),
]
# Their WMAP7 cosmology (Sect. 3).
_SCHNEIDER12_COSMO = FlatLambdaCDM(H0=70.4, Om0=0.2726, Ob0=0.046, Tcmb0=2.725)


@pytest.mark.parametrize(("mx", "m_fs", "m_hm"), _SCHNEIDER12_TABLE1)
def test_free_streaming_and_half_mode_masses_match_schneider12(mx, m_fs, m_hm):
    """M_fs and M_hm reproduce Table 1 of Schneider et al. (2012) for their cosmology."""
    w = wdm.Viel05(mx=mx, cosmo=_SCHNEIDER12_COSMO, z=0)
    # Tolerance: the table quotes 2 significant figures (up to 4% rounding) and does
    # not state the exact mean density used; measured deviations are 3-8%.
    assert w.m_fs == pytest.approx(m_fs, rel=0.1)
    assert w.m_hm == pytest.approx(m_hm, rel=0.1)


def test_half_mode_to_free_streaming_ratio():
    """Schneider et al. (2012) Eq. 8: lambda_hm ~ 13.93 lambda_fs for mu = 1.12."""
    w = wdm.Viel05(mx=1.0)
    # Tolerance: the paper quotes 4 significant figures.
    assert w.lam_hm / w.lam_eff_fs == pytest.approx(13.93, abs=5e-3)


def test_half_mode_scale_shrinks_with_particle_mass():
    """Heavier (colder) particles free-stream less: lambda_hm and M_hm fall with m_x."""
    mx = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    lam = np.array([wdm.Viel05(mx=m).lam_hm for m in mx])
    mhm = np.array([wdm.Viel05(mx=m).m_hm for m in mx])
    assert np.all(np.diff(lam) < 0)
    assert np.all(np.diff(mhm) < 0)


@pytest.fixture(scope="module")
def mf_pair():
    common = {"transfer_model": "EH", "Mmin": 7, "Mmax": 15, "dlog10m": 0.05}
    cdm = MassFunction(**common)
    warm = wdm.MassFunctionWDM(wdm_mass=1.0, **common)
    return cdm, warm


def test_wdm_sigma_suppressed_and_converges_to_cdm(mf_pair):
    """WDM removes small-scale power, so sigma_WDM <= sigma_CDM, converging at high M."""
    cdm, warm = mf_pair
    # Before the sigma_8 normalisation, T_WDM <= 1 at every k, so the variance can only
    # be lower, strictly so at every mass.
    assert np.all(warm._unn_sigma0 < cdm._unn_sigma0)
    # Both are then normalised to the same sigma_8. WDM slightly suppresses
    # sigma(8 Mpc/h) itself, so its amplitude is scaled up by a factor just above 1
    # (measured 1 + 1.9e-4 for m_x = 1 keV).
    renorm = warm._normalisation / cdm._normalisation
    assert 1 < renorm < 1 + 1e-3
    # The suppression weakens monotonically with scale, so sigma_WDM/sigma_CDM rises
    # with mass towards that renormalisation factor.
    ratio = warm.sigma / cdm.sigma
    assert np.all(np.diff(ratio) > 0)
    assert np.all(ratio <= renorm)
    # Tolerance: at M = 1e15 (R ~ 13 Mpc/h >> lambda_hm ~ 0.7 Mpc/h) only the sigma_8
    # renormalisation remains; measured 1.1e-4.
    assert warm.sigma[-1] == pytest.approx(cdm.sigma[-1], rel=1e-3)
    # Well below M_hm (~1e10) sigma is strongly suppressed.
    assert warm.sigma[0] < 0.7 * cdm.sigma[0]


def test_wdm_mass_function_suppressed_below_half_mode_mass(mf_pair):
    """Haloes below M_hm are suppressed relative to CDM; massive haloes are not."""
    cdm, warm = mf_pair
    low = warm.m < warm.wdm.m_hm / 10
    high = warm.m > warm.wdm.m_hm * 1000
    assert np.any(low)
    assert np.any(high)
    assert np.all(warm.dndm[low] < cdm.dndm[low])
    # Tolerance: above 1000 M_hm the transfer function deviates from 1 by < 1e-4.
    np.testing.assert_allclose(warm.dndm[high], cdm.dndm[high], rtol=1e-2)


def test_wdm_tends_to_cdm_for_heavy_particle():
    """As m_x -> infinity the particle is cold and WDM must reduce to CDM."""
    common = {"transfer_model": "EH", "Mmin": 7, "Mmax": 15, "dlog10m": 0.1}
    cdm = MassFunction(**common)
    warm = wdm.MassFunctionWDM(wdm_mass=1e4, **common)
    # Tolerance: for m_x = 10^4 keV, lambda_hm ~ 1e-5 Mpc/h, far below the smallest
    # radius (R ~ 0.04 Mpc/h at 1e7 Msun/h); measured max difference 3e-7.
    np.testing.assert_allclose(warm.dndm, cdm.dndm, rtol=1e-4)


@pytest.mark.parametrize("model", ["Schneider12_vCDM", "Schneider12", "Lovell14"])
def test_recalibrations_only_suppress_and_vanish_at_high_mass(model, mf_pair):
    """Empirical WDM recalibrations only suppress, and vanish at high mass.

    They multiply dn/dm by a factor in (0, 1] that rises monotonically with mass and
    tends to 1 for M >> M_hm.
    """
    _, warm = mf_pair
    cls = getattr(wdm, model)
    m = np.logspace(6, 16, 100)
    dndm0 = np.ones_like(m)
    factor = cls(m=m, dndm0=dndm0, wdm=warm.wdm).dndm_alter()
    assert np.all(factor > 0)
    assert np.all(factor <= 1)
    assert np.all(np.diff(factor) > 0)
    # Tolerance: at M = 1e6 M_hm the factor is 1 - O(beta gamma 1e-6) ~ 3e-6.
    assert factor[-1] == pytest.approx(1.0, abs=1e-4)
