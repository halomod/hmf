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
    ``|ln sigma^2| < ~1e-4``, leaving ``ln k_nl`` off by up to a few 1e-4; with
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
    Against the verbatim old code it is limited to <1e-3 by the old minimiser's
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
    np.testing.assert_allclose(new, run_with(exact_root=False), rtol=1e-3, atol=0)


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
