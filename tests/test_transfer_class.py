"""Tests of the CLASS transfer model, which needs the optional classy package."""

import pickle
import warnings

import numpy as np
import pytest
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import (
    FlatLambdaCDM,
    Flatw0waCDM,
    FlatwCDM,
    LambdaCDM,
    Planck18,
    w0wzCDM,
)

from hmf import MassFunction, Transfer
from hmf.density_field import transfer_models
from hmf.density_field.transfer_models import CLASS, class_cosmology

classy = pytest.importorskip("classy")

MASSLESS = FlatLambdaCDM(H0=67.66, Om0=0.31, Ob0=0.049, Tcmb0=2.7255, m_nu=0 * u.eV)
MASSIVE = FlatLambdaCDM(H0=67.66, Om0=0.31, Ob0=0.049, Tcmb0=2.7255, m_nu=[0.1, 0.1, 0.1] * u.eV)

BACKGROUND_COSMOLOGIES = {
    "planck18": Planck18,
    "massive": MASSIVE,
    "curved": LambdaCDM(
        H0=70, Om0=0.3, Ode0=0.65, Ob0=0.05, Tcmb0=2.7255, m_nu=[0, 0, 0.06] * u.eV
    ),
    "wcdm": FlatwCDM(H0=67.66, Om0=0.31, Ob0=0.049, Tcmb0=2.7255, w0=-0.9),
    "w0wa": Flatw0waCDM(H0=67.66, Om0=0.31, Ob0=0.049, Tcmb0=2.7255, w0=-0.9, wa=0.2),
}


def _run_class(params):
    cl = classy.Class()
    cl.set(params)
    cl.compute()
    return cl


@pytest.mark.parametrize("cosmo", BACKGROUND_COSMOLOGIES.values(), ids=BACKGROUND_COSMOLOGIES)
def test_class_cosmology_matches_astropy_background(cosmo):
    """CLASS's expansion history from class_cosmology is astropy's, at all redshifts."""
    cl = _run_class(class_cosmology(cosmo))
    z = np.array([0.0, 0.5, 1.0, 3.0, 10.0, 1000.0])
    h_class = np.array([cl.Hubble(zi) for zi in z])  # 1/Mpc
    h_astropy = (cosmo.H(z) / const.c).to_value(1 / u.Mpc)
    np.testing.assert_allclose(h_class, h_astropy, rtol=1e-4)
    assert cl.Neff() == pytest.approx(cosmo.Neff, rel=1e-4)


def test_class_cosmology_neutrino_density():
    """Three 0.1 eV neutrinos: CLASS's neutrino density is astropy's (all massive)."""
    cl = _run_class(class_cosmology(MASSIVE))
    omega_ncdm = cl.Omega_m() - cl.Omega0_cdm() - cl.Omega_b()
    assert omega_ncdm == pytest.approx(MASSIVE.Onu0, rel=1e-3)
    assert cl.Omega0_cdm() == pytest.approx(MASSIVE.Odm0, rel=1e-6)


@pytest.mark.parametrize("species", ["tot", "cb"])
@pytest.mark.parametrize("cosmo", [Planck18, MASSIVE], ids=["planck18", "massive"])
def test_power_shape_matches_class_pk(cosmo, species):
    """k^n T(k)^2 has the shape of CLASS's own linear power spectrum of that field."""
    n_s = 0.9667
    model = CLASS(cosmo, matter_species=species)
    params = {**model._class_input(), "output": "mPk, mTk", "n_s": n_s}
    cl = _run_class(params)

    h = cl.h()
    k = np.logspace(-4, -0.2, 40)  # h/Mpc, within CLASS's kmax
    pk_fn = cl.pk_lin if species == "tot" else cl.pk_cb_lin
    pk_class = np.array([pk_fn(ki * h, 0.0) for ki in k])

    ratio = k**n_s * np.exp(2 * model.lnt(np.log(k))) / pk_class
    np.testing.assert_allclose(ratio / ratio[0], 1.0, rtol=1e-4)


@pytest.mark.parametrize(
    "cosmo",
    [Planck18, FlatwCDM(H0=67.66, Om0=0.31, Ob0=0.049, Tcmb0=2.7255, w0=-0.9)],
    ids=["planck18", "wcdm"],
)
@pytest.mark.parametrize("species", ["tot", "cb"])
def test_agrees_with_camb(cosmo, species):
    """CLASS and CAMB are independent codes, and agree on P(k) to better than 1%.

    This includes k well above CLASS's and CAMB's kmax, where both are extrapolated.
    """
    kw = {
        "cosmo_model": cosmo,
        "lnk_min": np.log(1e-3),
        "lnk_max": np.log(100),
        "transfer_params": {"matter_species": species},
    }
    camb_tr = Transfer(
        transfer_model="CAMB",
        **{**kw, "transfer_params": {**kw["transfer_params"], "extrapolate_with_eh": True}},
    )
    class_tr = Transfer(transfer_model="CLASS", **kw)
    np.testing.assert_allclose(class_tr.power, camb_tr.power, rtol=1e-2)


def test_mass_function_agrees_with_camb():
    camb_mf = MassFunction(
        transfer_model="CAMB",
        transfer_params={"extrapolate_with_eh": True, "matter_species": "cb"},
    )
    class_mf = MassFunction(transfer_model="CLASS", transfer_params={"matter_species": "cb"})
    np.testing.assert_allclose(class_mf.dndm, camb_mf.dndm, rtol=1e-2)


def test_matter_species_free_streaming_limits():
    """P_tot = P_cb on large scales, and (1 - f_nu)^2 P_cb well inside free-streaming."""
    lnk = np.log(np.logspace(-4, 0, 50))
    k = np.exp(lnk)
    lnt_tot = CLASS(MASSIVE, matter_species="tot").lnt(lnk)
    lnt_cb = CLASS(MASSIVE, matter_species="cb").lnt(lnk)
    ratio = np.exp(lnt_tot - lnt_cb)

    f_nu = MASSIVE.Onu0 / (MASSIVE.Om0 + MASSIVE.Onu0)
    np.testing.assert_allclose(ratio[k < 1e-3], 1.0, rtol=1e-3)
    assert np.all(ratio[k > 0.5] < 1)
    np.testing.assert_allclose(ratio[k > 0.5], 1 - f_nu, rtol=5e-3)


def test_matter_species_identical_for_massless_neutrinos():
    lnk = np.log(np.logspace(-4, 2, 50))
    np.testing.assert_allclose(
        CLASS(MASSLESS, matter_species="tot").lnt(lnk),
        CLASS(MASSLESS, matter_species="cb").lnt(lnk),
        atol=1e-4,
    )


def test_default_matter_species_warns_for_massive_neutrinos():
    with pytest.warns(UserWarning, match="matter_species was not set for CLASS"):
        model = CLASS(MASSIVE)
    assert model.params["matter_species"] == "cb"


def test_default_matter_species_silent_for_massless_neutrinos():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = CLASS(MASSLESS)
    assert model.params["matter_species"] == "cb"


def test_eh_extrapolation_matches_direct_calculation():
    """Extrapolating above CLASS's default kmax agrees with computing up to there.

    The EH shape is not exact, so the error grows with k: it is ~1.5% in T by
    k = 20 h/Mpc, an order of magnitude above the default kmax. Below kmax the two
    differ only by interpolation between CLASS's (different) grids of k.
    """
    lnk = np.log(np.logspace(-3, np.log10(20), 100))
    k = np.exp(lnk)
    extrapolated = np.exp(CLASS(Planck18, matter_species="cb").lnt(lnk))
    direct = np.exp(
        CLASS(Planck18, matter_species="cb", kmax=30.0, extrapolate_with_eh=False).lnt(lnk)
    )
    np.testing.assert_allclose(extrapolated[k < 1], direct[k < 1], rtol=5e-3)
    np.testing.assert_allclose(extrapolated, direct, rtol=2e-2)


def test_unity_at_low_k():
    lnk = np.log(np.logspace(-8, -5, 10))
    np.testing.assert_allclose(CLASS(MASSLESS).lnt(lnk), 0.0, atol=1e-4)


def test_kmax_sets_class_kmax():
    model = CLASS(MASSLESS, kmax=5.0)
    assert model._class_input()["P_k_max_h/Mpc"] == 5.0
    assert model._transfers()["kh"].max() >= 5.0
    assert "P_k_max_h/Mpc" not in CLASS(MASSLESS)._class_input()


def test_output_always_includes_mtk():
    model = CLASS(MASSLESS, class_params={"output": "mPk"})
    assert model._class_input()["output"] == "mPk, mTk"
    assert np.all(np.isfinite(model.lnt(np.log(np.logspace(-3, 1, 10)))))


def test_class_params_passed_to_class():
    """A non-cosmological CLASS parameter, the helium fraction, reaches CLASS."""
    lnk = np.log(np.logspace(-2, 0, 20))
    default = CLASS(MASSLESS).lnt(lnk)
    more_helium = CLASS(MASSLESS, class_params={"YHe": 0.4}).lnt(lnk)
    # Helium changes the baryon physics, so the transfer function on small scales.
    assert np.max(np.abs(more_helium - default)) > 1e-3


@pytest.mark.parametrize("key", ["h", "H0", "omega_cdm", "m_ncdm", "Omega_fld", "gauge"])
def test_class_params_cannot_set_fixed_parameters(key):
    with pytest.raises(ValueError, match="class_params cannot set"):
        CLASS(MASSLESS, class_params={key: 1})


def test_unknown_class_parameter_raises():
    """CLASS itself rejects parameters it does not read (e.g. typos)."""
    model = CLASS(MASSLESS, class_params={"not_a_class_parameter": 1})
    with pytest.raises(classy.CosmoSevereError):
        model.lnt(np.log(np.array([0.1, 1.0])))


def test_rejects_unsupported_cosmology():
    cosmo = w0wzCDM(H0=70, Om0=0.3, Ode0=0.7, Ob0=0.05, Tcmb0=2.7255, w0=-1, wz=0.1)
    with pytest.raises(ValueError, match="CLASS will only work with LCDM or wCDM"):
        CLASS(cosmo)


def test_needs_baryons_and_cmb_temperature():
    with pytest.raises(ValueError, match="baryon density"):
        CLASS(FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.7255))
    with pytest.raises(ValueError, match="CMB temperature"):
        CLASS(FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05))


def test_sigma_8_species_reuses_class_run(monkeypatch):
    """Normalising sigma_8 of the total field needs no second CLASS run."""
    calls = []
    original = CLASS._compute_transfers

    def counting(self):
        calls.append(1)
        return original(self)

    monkeypatch.setattr(CLASS, "_compute_transfers", counting)
    t = Transfer(
        cosmo_model=MASSIVE,
        transfer_model="CLASS",
        transfer_params={"matter_species": "cb"},
        sigma_8_species="tot",
    )
    t.power
    assert t._sigma_8_transfer is not t.transfer
    assert len(calls) == 1


def test_sigma_8_normalises_total_field():
    """With sigma_8_species='tot', the total matter field has r.m.s. sigma_8."""
    t = Transfer(
        cosmo_model=MASSIVE,
        transfer_model="CLASS",
        transfer_params={"matter_species": "cb"},
        sigma_8_species="tot",
    )
    tot = Transfer(
        cosmo_model=MASSIVE, transfer_model="CLASS", transfer_params={"matter_species": "tot"}
    )
    # Both are normalised to the same total-matter field, so they agree on large scales.
    large = t.k < 1e-3
    np.testing.assert_allclose(t.power[large], tot.power[large], rtol=1e-3)


def test_pickle_roundtrip():
    model = CLASS(MASSLESS)
    lnk = np.log(np.logspace(-3, 1, 10))
    expected = model.lnt(lnk)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.lnt(lnk), expected)


def test_transfer_by_name_and_update():
    t = Transfer(transfer_model="CLASS", transfer_params={"matter_species": "tot"})
    assert isinstance(t.transfer, CLASS)
    power = t.power.copy()
    t.update(cosmo_params={"H0": 70.0})
    assert not np.allclose(t.power, power)


def test_missing_classy(monkeypatch):
    monkeypatch.setattr(transfer_models, "HAVE_CLASS", False)
    with pytest.raises(ImportError, match=r"hmf\[class\]"):
        CLASS(MASSLESS)

    import hmf.density_field.transfer as transfer_module

    monkeypatch.setattr(transfer_module, "HAVE_CLASS", False)
    with pytest.raises(ValueError, match="classy isn't installed"):
        Transfer(transfer_model="CLASS")


def test_cannot_share_results_with_other_code():
    camb_model = transfer_models.CAMB(MASSLESS, extrapolate_with_eh=True)
    with pytest.raises(TypeError, match="Can only share results with another CLASS"):
        CLASS(MASSLESS)._share_results(camb_model)
