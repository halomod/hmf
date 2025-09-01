import pytest

import camb
import numpy as np
from astropy.cosmology import FlatLambdaCDM, LambdaCDM, w0waCDM, wCDM

from hmf.density_field.transfer import Transfer


@pytest.fixture
def transfers():
    return Transfer(transfer_model="EH"), Transfer(transfer_model="EH")


@pytest.mark.parametrize(
    [
        "name",
        "val",
    ],
    [("z", 0.1), ("sigma_8", 0.82), ("n", 0.95), ("cosmo_params", {"H0": 68.0})],
)
def test_updates(transfers, name, val):
    t, t2 = transfers
    t.update(**{name: val})
    assert (
        np.mean(np.abs((t.power - t2.power) / t.power)) < 1
        and np.mean(np.abs((t.power - t2.power) / t.power)) > 1e-6
    )


def test_updates_from_file_array(datadir):
    tdata = np.genfromtxt(f"{datadir}/transfer_for_hmf_tests.dat")
    t = Transfer(
        transfer_model="FromArray", transfer_params={"k": tdata[:, 0], "T": tdata[:, 1]}
    )
    t2 = Transfer(
        transfer_model="FromArray", transfer_params={"k": tdata[:, 0], "T": tdata[:, 1]}
    )
    t2.update(transfer_params={"k": tdata[::2, 0], "T": tdata[::2, 1]})
    # This test for both FromArray transfer model and caching of dictionaries
    assert (
        np.mean(np.abs((t.power - t2.power) / t.power)) < 1
        and np.mean(np.abs((t.power - t2.power) / t.power)) > 1e-6
    )


def test_halofit():
    t = Transfer(lnk_min=-20, lnk_max=20, dlnk=0.05, transfer_model="EH")
    assert np.isclose(t.power[0], t.nonlinear_power[0])
    assert 5 * t.power[-1] < t.nonlinear_power[-1]


def test_ehnobao():
    t = Transfer(transfer_model="EH")
    tnobao = Transfer(transfer_model="EH_NoBAO")
    assert np.isclose(t._unnormalised_lnT[0], tnobao._unnormalised_lnT[0], rtol=1e-5)


def test_bondefs():
    t = Transfer(transfer_model="BondEfs")
    print(np.exp(t._unnormalised_lnT))
    assert np.isclose(np.exp(t._unnormalised_lnT[0]), 1, rtol=1e-5)


@pytest.mark.skip("Too slow and needs to be constantly updated.")
def test_data(datadir):
    import camb
    from astropy.cosmology import LambdaCDM

    cp = camb.CAMBparams()
    cp.set_matter_power(kmax=100.0)
    t = Transfer(
        cosmo_model=LambdaCDM(Om0=0.3, Ode0=0.7, H0=70.0, Ob0=0.05),
        sigma_8=0.8,
        n=1,
        transfer_params={"camb_params": cp},
        lnk_min=np.log(1e-11),
        lnk_max=np.log(1e11),
    )
    pdata = np.genfromtxt(datadir / "power_for_hmf_tests.dat")

    assert np.sqrt(np.mean(np.square(t.power - pdata[:, 1]))) < 0.001


def test_camb_extrapolation():
    t = Transfer(transfer_params={"extrapolate_with_eh": True}, transfer_model="CAMB")

    k = np.logspace(1.5, 2, 20)
    eh = t.transfer._eh.lnt(np.log(k))
    camb = t.transfer.lnt(np.log(k))

    eh += eh[0] - camb[0]

    assert np.isclose(eh[-1], camb[-1], rtol=1e-1)


def test_camb_neutrinos():
    # Correct parameter settings:
    cosmo_model = FlatLambdaCDM(
        Om0=0.3, H0=70.0, Ob0=0.05, m_nu=[0, 0, 0.06], Tcmb0=2.7255
    )

    t_nu = Transfer(
        cosmo_model=cosmo_model,
        sigma_8=0.8,
        n=1,
        transfer_model="CAMB",
        transfer_params={"extrapolate_with_eh": True},
        lnk_min=np.log(1e-11),
        lnk_max=np.log(1e11),
    )

    pars = camb.CAMBparams(
        DoLensing=False,
        Want_CMB=False,
        Want_CMB_lensing=False,
        WantCls=False,
        WantDerivedParameters=False,
        WantTransfer=True,
    )
    pars.Transfer.high_precision = False
    pars.Transfer.k_per_logint = 0
    pars.set_cosmology(
        H0=cosmo_model.H0.value,
        ombh2=cosmo_model.Ob0 * cosmo_model.h**2,
        omch2=(cosmo_model.Om0 - cosmo_model.Ob0) * cosmo_model.h**2,
        mnu=sum(cosmo_model.m_nu.value),
        neutrino_hierarchy="degenerate",
        omk=cosmo_model.Ok0,
        nnu=cosmo_model.Neff,
        standard_neutrino_neff=cosmo_model.Neff,
        TCMB=cosmo_model.Tcmb0.value,
    )

    t_nu_camb = t_nu.clone()
    t_nu_camb.transfer.params["camb_params"] = pars

    k = np.logspace(-4, 2, 10)
    hmf_t = t_nu.transfer.lnt(np.log(k))[0]
    camb_t = t_nu_camb.transfer.lnt(np.log(k))[0]

    diff = np.abs((camb_t - hmf_t) / camb_t)

    camb_cosmo = camb.get_background(t_nu.transfer.params["camb_params"])
    sum_omega_astropy = (
        t_nu.cosmo_model.Odm0 + t_nu.cosmo_model.Ob0 + t_nu.cosmo_model.Onu0
    )
    sum_omega_camb = (
        camb_cosmo.get_Omega("tot")
        - camb_cosmo.get_Omega("photon")
        - camb_cosmo.omega_de
    )

    assert diff <= 1e-3
    assert np.isclose(sum_omega_astropy, sum_omega_camb, rtol=1e-2)


def test_setting_kmax():
    t = Transfer(
        transfer_params={"extrapolate_with_eh": True, "kmax": 1.0},
        transfer_model="CAMB",
    )
    assert t.transfer.params["camb_params"].Transfer.kmax == 1.0
    camb_transfers = camb.get_transfer_functions(t.transfer.params["camb_params"])
    T = camb_transfers.get_matter_transfer_data().transfer_data
    assert np.max(T[0]) < 2.0


def test_camb_w0wa():
    """Essentially just test that CAMB doesn't fall over with a w0wa model."""
    t = Transfer(
        transfer_model="CAMB",
        cosmo_model=w0waCDM(
            Om0=0.3, Ode0=0.7, w0=-1, wa=0.03, Ob0=0.05, H0=70.0, Tcmb0=2.7
        ),
        transfer_params={"extrapolate_with_eh": True},
    )
    assert t.transfer_function.shape == t.k.shape


def test_camb_wCDM():
    """Essentially just test that CAMB doesn't fall over with a w0wa model."""
    t = Transfer(
        transfer_model="CAMB",
        cosmo_model=wCDM(Om0=0.3, Ode0=0.7, w0=-1, Ob0=0.05, H0=70.0, Tcmb0=2.7),
        transfer_params={"extrapolate_with_eh": True},
    )

    t2 = Transfer(
        transfer_model="CAMB",
        cosmo_model=LambdaCDM(Om0=0.3, Ode0=0.7, Ob0=0.05, H0=70.0, Tcmb0=2.7),
        transfer_params={"extrapolate_with_eh": True},
    )
    np.testing.assert_array_almost_equal(t.transfer_function, t2.transfer_function)


def test_camb_unset_params():
    with pytest.raises(ValueError):
        Transfer(
            transfer_model="CAMB",
            cosmo_model=w0waCDM(Om0=0.3, Ode0=0.7, w0=-1, wa=0.03, Ob0=0.05, H0=70.0),
        ).transfer

    with pytest.raises(ValueError):
        Transfer(
            transfer_model="CAMB",
            cosmo_model=w0waCDM(Om0=0.3, Ode0=0.7, w0=-1, wa=0.03, H0=70.0, Tcmb0=2.7),
        ).transfer


def test_bbks_sugiyama():
    t = Transfer(transfer_model="BBKS", transfer_params={"use_sugiyama_baryons": True})
    t2 = Transfer(transfer_model="BBKS")

    assert not np.allclose(t.transfer_function, t2.transfer_function)
