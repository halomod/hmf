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
