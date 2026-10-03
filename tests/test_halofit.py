import importlib

import numpy as np
import pytest

from hmf.density_field import transfer
from hmf.density_field.halofit import halofit as hmf_halofit


def test_takahashi():
    t = transfer.Transfer(transfer_model="EH", takahashi=False, lnk_max=7)
    tt = transfer.Transfer(transfer_model="EH", takahashi=True, lnk_max=7)

    assert np.isclose(t.nonlinear_power[0], tt.nonlinear_power[0], rtol=1e-4)
    print(t.nonlinear_power[-1] / tt.nonlinear_power[-1])
    assert np.logical_not(np.isclose(t.nonlinear_power[-1] / tt.nonlinear_power[-1], 1, rtol=0.4))


def test_takahashi_hiz():
    # This test should do the HALOFIT WARNING
    t = transfer.Transfer(transfer_model="EH", takahashi=False, lnk_max=7, z=8.0)
    tt = transfer.Transfer(transfer_model="EH", takahashi=True, lnk_max=7, z=8.0)

    assert np.isclose(t.nonlinear_power[0], tt.nonlinear_power[0], rtol=1e-4)
    print(t.nonlinear_power[-1] / tt.nonlinear_power[-1])
    assert np.logical_not(np.isclose(t.nonlinear_power[-1] / tt.nonlinear_power[-1], 1, rtol=0.4))

    t.update(z=0)

    assert np.logical_not(np.isclose(t.nonlinear_power[0] / tt.nonlinear_power[0], 0.9, rtol=0.1))
    assert np.logical_not(
        np.isclose(t.nonlinear_power[-1] / tt.nonlinear_power[-1], 0.99, rtol=0.1)
    )


def test_halofit_high_s8():
    t = transfer.Transfer(transfer_model="EH", lnk_max=7, sigma_8=0.999)
    thi = transfer.Transfer(transfer_model="EH", lnk_max=7, sigma_8=1.001)  # just above threshold

    print(
        t.nonlinear_power[0] / thi.nonlinear_power[0] - 1,
        t.nonlinear_power[-1] / thi.nonlinear_power[-1] - 1,
    )
    assert np.isclose(t.nonlinear_power[0], thi.nonlinear_power[0], rtol=2e-2)
    assert np.isclose(t.nonlinear_power[-1], thi.nonlinear_power[-1], rtol=5e-2)


@pytest.mark.skipif(
    not transfer.HAVE_CAMB,
    reason="CAMB not installed; cannot compare hmf halofit against CAMB halofit.",
)
def test_halofit_vs_camb():
    """
    Validate hmf's halofit implementation against CAMB's Takahashi+2012 halofit.

    When the **same** linear power spectrum is fed to both implementations the
    non-linear power spectra must agree to within 0.5 % for
    0.01 ≤ k ≤ 30 h/Mpc.  This confirms that any larger discrepancy observed
    when comparing full pipelines (e.g. hmf with an EH transfer function vs.
    nbodykit/CLASS) originates from the *input* linear spectrum, not from the
    halofit algorithm.
    """
    import camb
    from astropy.cosmology import FlatLambdaCDM

    from hmf.cosmology.cosmo import Cosmology as HMFCosmology

    # ---- reference cosmology ----
    h = 0.6774
    ombh2 = 0.02230
    omch2 = 0.1188

    # ---- CAMB linear power spectrum ----
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100 * h, ombh2=ombh2, omch2=omch2, mnu=0.0, omk=0)
    pars.InitPower.set_params(As=2.215e-9, ns=0.9667)
    pars.set_matter_power(redshifts=[0.0], kmax=200.0)
    pars.NonLinear = camb.model.NonLinear_none
    results_lin = camb.get_results(pars)
    kh_lin, _, pk_lin = results_lin.get_matter_power_spectrum(minkh=1e-4, maxkh=100.0, npoints=500)
    delta_k_lin = kh_lin**3 * pk_lin[0] / (2 * np.pi**2)

    # ---- CAMB nonlinear power (Takahashi) ----
    pars_nl = camb.CAMBparams()
    pars_nl.set_cosmology(H0=100 * h, ombh2=ombh2, omch2=omch2, mnu=0.0, omk=0)
    pars_nl.InitPower.set_params(As=2.215e-9, ns=0.9667)
    pars_nl.set_matter_power(redshifts=[0.0], kmax=200.0)
    pars_nl.NonLinear = camb.model.NonLinear_both
    nl_model = camb.nonlinear.Halofit()
    nl_model.set_params(halofit_version="takahashi")
    pars_nl.NonLinearModel = nl_model
    results_nl = camb.get_results(pars_nl)
    kh_nl, _, pk_nl_camb = results_nl.get_matter_power_spectrum(
        minkh=1e-4, maxkh=100.0, npoints=500
    )
    delta_k_nl_camb = kh_nl**3 * pk_nl_camb[0] / (2 * np.pi**2)

    # ---- hmf halofit on the same CAMB linear power spectrum ----
    Om0 = (ombh2 + omch2) / h**2
    Ob0 = ombh2 / h**2
    cosmo_astropy = FlatLambdaCDM(H0=100 * h, Om0=Om0, Ob0=Ob0)
    cosmo_hmf = HMFCosmology(cosmo_model=cosmo_astropy)
    delta_k_nl_hmf = hmf_halofit(kh_lin, delta_k_lin, z=0.0, cosmo=cosmo_hmf.cosmo, takahashi=True)

    # ---- comparison over 0.01 ≤ k ≤ 30 h/Mpc ----
    mask = (kh_lin >= 0.01) & (kh_lin <= 30.0)
    ratio = delta_k_nl_camb[mask] / delta_k_nl_hmf[mask]
    max_dev = np.max(np.abs(ratio - 1))
    assert np.all(np.abs(ratio - 1) < 5e-3), (
        f"hmf halofit differs from CAMB by more than 0.5%: max ratio deviation = {max_dev:.4f}"
    )


# ---------------------------------------------------------------------------
# Regression tests for the vectorised _get_spec (#24) and default cosmology (#111)
# ---------------------------------------------------------------------------


def _get_spec_reference(
    k: np.ndarray, delta_k: np.ndarray, exact_root: bool = False
) -> tuple[float, float, float]:
    """Copy of the pre-#24 ``_get_spec`` (hmf main @ 3ffd4cc).

    Kept as the reference that the faster implementation must reproduce. With
    ``exact_root=False`` it is verbatim. The old Nelder-Mead stops once
    ``|ln sigma^2| < ~1e-4``, leaving ``ln k_nl`` off by 1e-5 to a few 1e-3,
    depending on the spectrum (e.g. 3e-3 for BBKS at z=3); with
    ``exact_root=True`` the root is instead solved to machine precision and only
    the (slow) spline derivatives are kept, isolating the derivative calculation.
    """
    from scipy.integrate import simpson
    from scipy.interpolate import InterpolatedUnivariateSpline
    from scipy.optimize import brentq, minimize

    def get_log_sigma2(lnr):
        R = np.exp(lnr)
        integrand = delta_k * np.exp(-((k * R) ** 2))
        return np.log(simpson(integrand, x=np.log(k)))

    def get_sigma_abs(lnr):
        return np.abs(get_log_sigma2(lnr))

    if exact_root:
        with np.errstate(divide="ignore"):  # sigma^2 underflows to 0 at huge R
            rnl = np.exp(brentq(get_log_sigma2, -15, 15, xtol=1e-14))
    else:
        res = minimize(
            get_sigma_abs, x0=[1.0], options={"xatol": np.log(1.1)}, method="Nelder-Mead"
        )
        rnl = np.exp(res.x)
    knl = 1 / rnl

    lnr = np.linspace(np.log(0.75 * rnl), np.log(1.25 * rnl), 20)
    lnsig = [get_log_sigma2(r) for r in lnr]
    sig_of_r = InterpolatedUnivariateSpline(lnr, lnsig, k=5)
    dev1, dev2 = sig_of_r.derivatives(np.log(rnl))[1:3]

    return float(np.squeeze(knl)), -dev1 - 3.0, -dev2


@pytest.mark.parametrize("z", [0.0, 1.0, 3.0])
@pytest.mark.parametrize("takahashi", [True, False])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"transfer_model": "EH", "lnk_max": 7},
        {"transfer_model": "EH"},
        {"transfer_model": "EH", "lnk_max": 7, "sigma_8": 1.001},
        {"transfer_model": "BBKS", "lnk_min": -10, "lnk_max": 5},
    ],
)
def test_nonlinear_power_matches_reference(monkeypatch, z, takahashi, kwargs):
    """The fast ``_get_spec`` reproduces the old algorithm.

    Against the old algorithm with an exactly-solved root the agreement is ~1e-10.
    Against the verbatim old code it is limited to <5e-3 by the old minimiser's
    tolerance on the non-linear scale (the new root is the more accurate one).
    """
    # ``hmf.density_field.halofit`` is shadowed by the function of the same name.
    halofit_module = importlib.import_module("hmf.density_field.halofit")

    t = transfer.Transfer(z=z, takahashi=takahashi, **kwargs)
    new = hmf_halofit(t.k, t.delta_k, z=z, cosmo=t.cosmo, takahashi=takahashi)
    # The framework, which is what users actually call, goes through the same path.
    np.testing.assert_allclose(t.nonlinear_delta_k, new, rtol=1e-12, atol=0)

    def run_with(**kw):
        monkeypatch.setattr(
            halofit_module, "_get_spec", lambda k, dk: _get_spec_reference(k, dk, **kw)
        )
        return hmf_halofit(t.k, t.delta_k, z=z, cosmo=t.cosmo, takahashi=takahashi)

    np.testing.assert_allclose(new, run_with(exact_root=True), rtol=1e-8, atol=0)
    np.testing.assert_allclose(new, run_with(exact_root=False), rtol=5e-3, atol=0)


@pytest.mark.parametrize("z", [0.0, 1.0, 3.0])
def test_get_spec_matches_reference(z):
    """knl, n_eff and the curvature each agree with the old implementation."""
    from hmf.density_field.halofit import _get_spec

    t = transfer.Transfer(transfer_model="EH", lnk_max=7, z=z)
    knl, neff, ncur = _get_spec(t.k, t.delta_k)

    knl_ref, neff_ref, ncur_ref = _get_spec_reference(t.k, t.delta_k, exact_root=True)
    assert knl == pytest.approx(knl_ref, rel=1e-10)
    assert neff == pytest.approx(neff_ref, rel=1e-8)
    assert ncur == pytest.approx(ncur_ref, rel=1e-6)

    knl_ref, neff_ref, ncur_ref = _get_spec_reference(t.k, t.delta_k)
    assert knl == pytest.approx(knl_ref, rel=5e-4)
    assert neff == pytest.approx(neff_ref, rel=1e-4)
    assert ncur == pytest.approx(ncur_ref, rel=1e-3)


@pytest.mark.parametrize("takahashi", [True, False])
def test_default_cosmo(takahashi):
    """#111: no cosmo, an hmf Cosmology framework, and its FLRW all agree."""
    from hmf.cosmology.cosmo import Cosmology

    t = transfer.Transfer(transfer_model="EH", lnk_max=7)
    k, dk = t.k, t.delta_k

    from_flrw = hmf_halofit(k, dk, z=1.0, cosmo=Cosmology().cosmo, takahashi=takahashi)
    from_none = hmf_halofit(k, dk, z=1.0, takahashi=takahashi)
    from_framework = hmf_halofit(k, dk, z=1.0, cosmo=Cosmology(), takahashi=takahashi)

    np.testing.assert_allclose(from_none, from_flrw, rtol=1e-12, atol=0)
    np.testing.assert_allclose(from_framework, from_flrw, rtol=1e-12, atol=0)


def test_framework_cosmo_params_respected():
    """An hmf Cosmology framework with custom params uses those params, not the default."""
    from hmf.cosmology.cosmo import Cosmology

    t = transfer.Transfer(transfer_model="EH", lnk_max=7)
    k, dk = t.k, t.delta_k

    cosmo = Cosmology(cosmo_params={"Om0": 0.2})
    from_framework = hmf_halofit(k, dk, z=1.0, cosmo=cosmo)
    from_flrw = hmf_halofit(k, dk, z=1.0, cosmo=cosmo.cosmo)
    default = hmf_halofit(k, dk, z=1.0, cosmo=Cosmology().cosmo)

    np.testing.assert_allclose(from_framework, from_flrw, rtol=1e-12, atol=0)
    assert not np.allclose(from_framework, default, rtol=1e-3, atol=0)


def test_get_spec_warns_without_nonlinear_scale():
    """If sigma(R) never crosses unity on the k-range, warn and keep going."""
    from hmf.density_field.halofit import _get_spec

    t = transfer.Transfer(transfer_model="EH", lnk_max=7, z=500.0)
    with pytest.warns(UserWarning, match="non-linear scale"):
        knl, neff, ncur = _get_spec(t.k, t.delta_k)
    assert np.isfinite([knl, neff, ncur]).all()


# ---------------------------------------------------------------------------
# Physical tests: limits and bounds that any correct HALOFIT must satisfy,
# independent of how _get_spec is implemented.
# ---------------------------------------------------------------------------


def _nl_to_lin_ratio(z: float, takahashi: bool) -> tuple[np.ndarray, np.ndarray, float]:
    """Return k, P_nl/P_lin and k_nl for an EH Transfer at redshift z."""
    from hmf.density_field.halofit import _get_spec

    t = transfer.Transfer(transfer_model="EH", z=z, takahashi=takahashi)
    knl = _get_spec(t.k, t.delta_k)[0]
    return t.k, t.nonlinear_delta_k / t.delta_k, knl


@pytest.mark.parametrize("z", [0.0, 1.0, 3.0])
@pytest.mark.parametrize("takahashi", [True, False])
def test_low_k_reduces_to_linear(z, takahashi):
    """On large scales the non-linear power reduces to the linear power.

    For k << k_nl the halo term is negligible and the quasi-linear term is
    P_lin * exp(-y/4 - y^2/8) with y = k/k_nl, so to leading order
    P_nl/P_lin = 1 - k/(4 k_nl). (k <= 0.005 is excluded because halofit
    returns the linear power there by construction.)
    """
    k, ratio, knl = _nl_to_lin_ratio(z, takahashi)
    m = (k > 0.005) & (k <= 0.01)
    suppression = 1 - ratio[m]

    # Converges to linear: k_nl >~ 0.36 h/Mpc at z=0 (and grows with z), so the
    # leading-order suppression is at most 0.01/(4*0.36) ~ 7e-3. Measured <= 6.5e-3.
    assert np.all(suppression > 0)
    assert np.all(suppression < 1e-2)

    # ...and the deviation is the expected leading-order damping. The (positive)
    # halo term and the O(Delta_lin) quasi-linear corrections make the measured
    # suppression slightly smaller than k/(4 k_nl): 0.88-0.995 of it here.
    np.testing.assert_array_less(0.8, suppression / (k[m] / (4 * knl)))
    np.testing.assert_array_less(suppression / (k[m] / (4 * knl)), 1.05)


@pytest.mark.parametrize("z", [0.0, 1.0, 3.0])
def test_nonlinear_scale_satisfies_sigma_unity(z):
    """k_nl is defined by sigma^2(R = 1/k_nl) = 1 with a Gaussian window."""
    from scipy.integrate import simpson

    from hmf.density_field.halofit import _get_spec

    t = transfer.Transfer(transfer_model="EH", z=z)
    knl = _get_spec(t.k, t.delta_k)[0]
    sigma2 = simpson(t.delta_k * np.exp(-((t.k / knl) ** 2)), x=np.log(t.k))
    assert sigma2 == pytest.approx(1.0, rel=1e-8)


def test_nonlinear_scale_grows_with_redshift():
    """Structure is less evolved at higher z, so the non-linear scale moves to higher k."""
    from hmf.density_field.halofit import _get_spec

    knls = []
    for z in [0.0, 0.5, 1.0, 2.0, 3.0]:
        t = transfer.Transfer(transfer_model="EH", z=z)
        knls.append(_get_spec(t.k, t.delta_k)[0])
    assert np.all(np.diff(knls) > 0)


@pytest.mark.parametrize("takahashi", [True, False])
def test_high_k_boost_at_z0(takahashi):
    """At z=0 and 1 <~ k <~ 10 h/Mpc, scales are deep in the non-linear regime.

    Gravitational collapse (the one-halo term) boosts the power well above
    linear there. Measured P_nl/P_lin >= 5.8 over this range; we require > 4.
    """
    k, ratio, _ = _nl_to_lin_ratio(0.0, takahashi)
    m = (k >= 1) & (k <= 10)
    assert np.all(ratio[m] > 4)


@pytest.mark.parametrize("takahashi", [True, False])
def test_nonlinear_boost_decreases_with_redshift(takahashi):
    """At fixed high k, the non-linear boost shrinks as z grows (less collapse)."""
    redshifts = [0.0, 0.5, 1.0, 2.0, 3.0]
    kvals = np.array([1.0, 3.0, 10.0])
    boosts = []
    for z in redshifts:
        k, ratio, _ = _nl_to_lin_ratio(z, takahashi)
        boosts.append(np.interp(np.log(kvals), np.log(k), ratio))
    boosts = np.array(boosts)  # shape (z, k)

    assert np.all(np.diff(boosts, axis=0) < 0)
    # Still non-linear (boost > 1) at the highest redshift for these k.
    assert np.all(boosts[-1] > 1)
