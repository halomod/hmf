import warnings

import camb
import numpy as np
import pytest
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM, LambdaCDM, w0waCDM, wCDM

from hmf.density_field.filters import TopHat
from hmf.density_field.transfer import Transfer


@pytest.fixture
def transfers():
    return Transfer(transfer_model="EH"), Transfer(transfer_model="EH")


@pytest.mark.parametrize(
    ("name", "val"),
    [("z", 0.1), ("sigma_8", 0.82), ("n", 0.95), ("cosmo_params", {"H0": 68.0})],
)
def test_updates(transfers, name, val):
    t, t2 = transfers
    t.update(**{name: val})
    assert np.mean(np.abs((t.power - t2.power) / t.power)) < 1
    assert np.mean(np.abs((t.power - t2.power) / t.power)) > 1e-6


def test_updates_from_file_array(datadir):
    tdata = np.genfromtxt(f"{datadir}/transfer_for_hmf_tests.dat")
    t = Transfer(transfer_model="FromArray", transfer_params={"k": tdata[:, 0], "T": tdata[:, 1]})
    t2 = Transfer(transfer_model="FromArray", transfer_params={"k": tdata[:, 0], "T": tdata[:, 1]})
    t2.update(transfer_params={"k": tdata[::2, 0], "T": tdata[::2, 1]})
    # This test for both FromArray transfer model and caching of dictionaries
    assert np.mean(np.abs((t.power - t2.power) / t.power)) < 1
    assert np.mean(np.abs((t.power - t2.power) / t.power)) > 1e-6


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


@pytest.mark.filterwarnings("ignore:matter_species was not set")
def test_camb_extrapolation():
    t = Transfer(transfer_params={"extrapolate_with_eh": True}, transfer_model="CAMB")

    k = np.logspace(1.5, 2, 20)
    eh = t.transfer._eh.lnt(np.log(k))
    camb = t.transfer.lnt(np.log(k))

    eh += eh[0] - camb[0]

    assert np.isclose(eh[-1], camb[-1], rtol=1e-1)


@pytest.mark.filterwarnings("ignore:matter_species was not set")
def test_camb_neutrinos():
    # Correct parameter settings:
    cosmo_model = FlatLambdaCDM(Om0=0.3, H0=70.0, Ob0=0.05, m_nu=[0, 0, 0.06], Tcmb0=2.7255)

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
        omch2=cosmo_model.Odm0 * cosmo_model.h**2,
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
    sum_omega_astropy = t_nu.cosmo_model.Odm0 + t_nu.cosmo_model.Ob0 + t_nu.cosmo_model.Onu0
    sum_omega_camb = (
        camb_cosmo.get_Omega("tot") - camb_cosmo.get_Omega("photon") - camb_cosmo.omega_de
    )

    assert diff <= 1e-3
    assert np.isclose(sum_omega_astropy, sum_omega_camb, rtol=1e-2)


@pytest.mark.filterwarnings("ignore:matter_species was not set")
def test_camb_massive_neutrinos_affect_transfer():
    """Verify that massive neutrinos produce a different transfer function from massless ones.

    The neutrino mass is passed to CAMB via the ``mnu`` parameter (sum of masses in eV).
    The CDM density (``omch2``) is set from ``cosmo.Odm0 * h**2``, which is the CDM-only
    density (astropy's ``Om0`` does not include massive neutrinos). This ensures the
    total matter budget in CAMB correctly separates CDM, baryons, and neutrinos.
    """
    base_kwargs = {
        "H0": 70.0,
        "Ob0": 0.05,
        "Om0": 0.3,
        "Tcmb0": 2.7255,
    }
    cosmo_massless = FlatLambdaCDM(m_nu=[0, 0, 0] * u.eV, **base_kwargs)
    cosmo_massive = FlatLambdaCDM(m_nu=[0, 0, 0.3] * u.eV, **base_kwargs)

    transfer_kwargs = {
        "transfer_model": "CAMB",
        "transfer_params": {"extrapolate_with_eh": False},
    }
    t_massless = Transfer(cosmo_model=cosmo_massless, **transfer_kwargs)
    t_massive = Transfer(cosmo_model=cosmo_massive, **transfer_kwargs)

    # At small scales (large k), neutrino free-streaming suppresses the power spectrum.
    # The transfer functions should differ significantly at k ~ 1 h/Mpc and above.
    k = np.logspace(-1, 1, 20)
    lnt_massless = t_massless.transfer.lnt(np.log(k))
    lnt_massive = t_massive.transfer.lnt(np.log(k))

    # Massive neutrinos suppress structure at small scales: T_massive < T_massless at large k
    assert not np.allclose(lnt_massless, lnt_massive, rtol=1e-3), (
        "Transfer functions for massless and massive neutrinos should differ"
    )
    assert np.all(lnt_massless[-5:] > lnt_massive[-5:]), (
        "Massive neutrinos should suppress the transfer function at small scales (large k)"
    )


def _camb_species_powers(m_nu):
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3, Ob0=0.05, Tcmb0=2.7255, m_nu=m_nu * u.eV)
    powers = {}
    for species in ("tot", "cb"):
        t = Transfer(
            cosmo_model=cosmo,
            transfer_model="CAMB",
            transfer_params={"extrapolate_with_eh": False, "matter_species": species},
            lnk_min=np.log(1e-3),
            lnk_max=np.log(10.0),
        )
        powers[species] = t._unnormalised_power
    camb_params = t.transfer.params["camb_params"]
    f_nu = camb_params.omnuh2 / (camb_params.omnuh2 + camb_params.omch2 + camb_params.ombh2)
    return t.k, powers, f_nu


def test_camb_matter_species_agree_for_massless_neutrinos():
    """With massless neutrinos, P_cb and P_tot are the same spectrum."""
    _, powers, _ = _camb_species_powers([0.0, 0.0, 0.0])
    np.testing.assert_allclose(powers["cb"], powers["tot"], rtol=1e-6)


def test_camb_matter_species_neutrino_suppression():
    """P_tot is suppressed relative to P_cb by neutrino free-streaming.

    On scales much larger than the free-streaming length neutrinos cluster like CDM,
    so P_tot = P_cb. Well inside it, delta_nu -> 0 and delta_tot -> (1 - f_nu) delta_cb
    with f_nu = Omega_nu / Omega_m, so P_tot / P_cb -> (1 - f_nu)^2.
    """
    previous_ratio = 1.0
    for m_nu in (0.06, 0.15, 0.3):
        k, powers, f_nu = _camb_species_powers([0.0, 0.0, m_nu])
        ratio = powers["tot"] / powers["cb"]

        # Large scales: the two fields agree.
        np.testing.assert_allclose(ratio[k < 2e-3], 1.0, rtol=1e-3)

        # Small scales: P_cb > P_tot, approaching the analytic free-streaming limit.
        small = (k > 1.0) & (k < 10.0)
        assert np.all(ratio[small] < 1)
        np.testing.assert_allclose(ratio[small], (1 - f_nu) ** 2, rtol=1e-2)

        # More massive neutrinos give stronger suppression.
        assert np.max(ratio[small]) < previous_ratio
        previous_ratio = np.min(ratio[small])


def test_camb_default_matter_species_is_cb_with_warning():
    """The default changed from "tot" to "cb"; users with massive neutrinos are told."""
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3, Ob0=0.05, Tcmb0=2.7255, m_nu=[0, 0, 0.06] * u.eV)
    t = Transfer(
        cosmo_model=cosmo, transfer_model="CAMB", transfer_params={"extrapolate_with_eh": False}
    )
    with pytest.warns(UserWarning, match="matter_species was not set for CAMB"):
        species = t.transfer.params["matter_species"]
    assert species == "cb"


@pytest.mark.parametrize(
    ("m_nu", "transfer_params"),
    [
        ([0, 0, 0], {"extrapolate_with_eh": False}),
        ([0, 0, 0.06], {"extrapolate_with_eh": False, "matter_species": "cb"}),
        ([0, 0, 0.06], {"extrapolate_with_eh": False, "matter_species": "tot"}),
    ],
)
def test_camb_matter_species_no_warning(m_nu, transfer_params):
    """No warning when the species is set, or when it can't matter (massless neutrinos)."""
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3, Ob0=0.05, Tcmb0=2.7255, m_nu=m_nu * u.eV)
    t = Transfer(cosmo_model=cosmo, transfer_model="CAMB", transfer_params=transfer_params)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="matter_species was not set")
        t.transfer


def _sigma_8_of(t):
    return TopHat(t.k, t.power).sigma(8.0)[0]


def test_sigma_8_species_tot_normalises_total_field():
    """sigma_8_species='tot' normalises P_cb so that the total field has the given sigma_8.

    The result must be the same universe as normalising P_tot directly: P_cb = P_tot on
    large scales, P_tot/P_cb -> (1 - f_nu)^2 on small scales, and since
    delta_tot / delta_cb lies in [1 - f_nu, 1], sigma_8 <= sigma_8,cb <= sigma_8 / (1 - f_nu).
    """
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3, Ob0=0.05, Tcmb0=2.7255, m_nu=[0, 0, 0.3] * u.eV)
    kwargs = {"cosmo_model": cosmo, "transfer_model": "CAMB", "sigma_8": 0.8}
    t_tot = Transfer(
        transfer_params={"extrapolate_with_eh": True, "matter_species": "tot"}, **kwargs
    )
    t_cb = Transfer(
        transfer_params={"extrapolate_with_eh": True, "matter_species": "cb"},
        sigma_8_species="tot",
        **kwargs,
    )
    camb_params = t_cb.transfer.params["camb_params"]
    f_nu = camb_params.omnuh2 / (camb_params.omnuh2 + camb_params.omch2 + camb_params.ombh2)

    k = t_tot.k
    ratio = t_tot.power / t_cb.power
    np.testing.assert_allclose(ratio[(k > 1e-5) & (k < 1e-3)], 1.0, rtol=1e-3)
    np.testing.assert_allclose(ratio[(k > 1) & (k < 10)], (1 - f_nu) ** 2, rtol=1e-2)

    assert _sigma_8_of(t_tot) == pytest.approx(0.8, rel=1e-6)
    assert 0.8 * (1 + 1e-3) < _sigma_8_of(t_cb) <= 0.8 / (1 - f_nu)


@pytest.mark.parametrize("sigma_8_species", ["cb", "tot"])
def test_sigma_8_species_irrelevant_cases(sigma_8_species):
    """sigma_8_species changes nothing for massless neutrinos or species-blind models."""
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3, Ob0=0.05, Tcmb0=2.7255, m_nu=[0, 0, 0] * u.eV)
    camb_kwargs = {
        "cosmo_model": cosmo,
        "transfer_model": "CAMB",
        "transfer_params": {"extrapolate_with_eh": True, "matter_species": "cb"},
    }
    np.testing.assert_allclose(
        Transfer(sigma_8_species=sigma_8_species, **camb_kwargs).power,
        Transfer(**camb_kwargs).power,
        rtol=1e-6,
    )
    np.testing.assert_array_equal(
        Transfer(transfer_model="EH", sigma_8_species=sigma_8_species).power,
        Transfer(transfer_model="EH").power,
    )


def test_sigma_8_species_default_and_none():
    """By default sigma_8 is that of the total field; None means the computed field."""
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3, Ob0=0.05, Tcmb0=2.7255, m_nu=[0, 0, 0.3] * u.eV)
    kwargs = {
        "cosmo_model": cosmo,
        "transfer_model": "CAMB",
        "sigma_8": 0.8,
        "transfer_params": {"extrapolate_with_eh": True, "matter_species": "cb"},
    }
    np.testing.assert_array_equal(
        Transfer(**kwargs).power, Transfer(sigma_8_species="tot", **kwargs).power
    )
    t_none = Transfer(sigma_8_species=None, **kwargs)
    assert _sigma_8_of(t_none) == pytest.approx(0.8, rel=1e-6)


def test_bad_sigma_8_species():
    with pytest.raises(ValueError, match="sigma_8_species must be"):
        Transfer(transfer_model="EH", sigma_8_species="nu")


def test_camb_bad_matter_species():
    with pytest.raises(ValueError, match="matter_species must be one of"):
        Transfer(
            transfer_model="CAMB",
            transfer_params={"extrapolate_with_eh": False, "matter_species": "nu"},
        ).transfer


@pytest.mark.filterwarnings("ignore:matter_species was not set")
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
        cosmo_model=w0waCDM(Om0=0.3, Ode0=0.7, w0=-1, wa=0.03, Ob0=0.05, H0=70.0, Tcmb0=2.7),
        transfer_params={"extrapolate_with_eh": True},
    )
    assert t.transfer_function.shape == t.k.shape


def test_camb_wcdm():
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
    t = Transfer(
        transfer_model="CAMB",
        cosmo_model=w0waCDM(Om0=0.3, Ode0=0.7, w0=-1, wa=0.03, Ob0=0.05, H0=70.0),
    )
    with pytest.raises(ValueError, match="the CMB temperature must be set explicitly"):
        t.transfer

    t = Transfer(
        transfer_model="CAMB",
        cosmo_model=w0waCDM(Om0=0.3, Ode0=0.7, w0=-1, wa=0.03, H0=70.0, Tcmb0=2.7),
    )
    with pytest.raises(ValueError, match="you must set the baryon density"):
        t.transfer


def test_bbks_sugiyama():
    t = Transfer(transfer_model="BBKS", transfer_params={"use_sugiyama_baryons": True})
    t2 = Transfer(transfer_model="BBKS")

    assert not np.allclose(t.transfer_function, t2.transfer_function)
