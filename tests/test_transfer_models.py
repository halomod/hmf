import warnings

import camb
import numpy as np
import pytest
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM, LambdaCDM, Planck18

from hmf.density_field import transfer_models
from hmf.density_field.transfer_models import CAMB, EH_BAO, FromArray, FromFile


@pytest.fixture
def base_cosmo():
    return LambdaCDM(Om0=0.3, Ode0=0.7, H0=70.0, Ob0=0.05, Tcmb0=2.7)


def test_fromfile_low_k_branch(tmp_path, base_cosmo):
    data = np.array(
        [
            [1.0, 1.0],
            [2.0, 1.0],
            [3.0, 1.0],
            [4.0, 1.0],
        ]
    )
    fname = tmp_path / "transfer.dat"
    np.savetxt(fname, data)

    model = FromFile(base_cosmo, fname=str(fname))
    lnk = np.log(np.array([0.1, 0.5, 1.0]))
    out = model.lnt(lnk)

    assert out.shape == lnk.shape


@pytest.mark.parametrize(
    ("k", "t", "match"),
    [
        (None, None, "must supply an array"),
        (np.array([1.0, 2.0]), np.array([1.0]), "must have same length"),
    ],
)
def test_fromarray_validation(base_cosmo, k, t, match):
    model = FromArray(base_cosmo, k=k, T=t)
    with pytest.raises(ValueError, match=match):
        model.lnt(np.log(np.array([0.5, 1.0])))


def test_fromarray_low_k_branch(base_cosmo):
    k = np.array([1.0, 2.0, 3.0])
    t = np.array([1.0, 1.0, 1.0])
    model = FromArray(base_cosmo, k=k, T=t)
    lnk = np.log(np.array([0.1, 1.0, 2.0]))

    out = model.lnt(lnk)

    assert out.shape == lnk.shape


def test_eh_bao_k_peak_property(base_cosmo):
    model = EH_BAO(base_cosmo)

    assert np.isfinite(model.k_peak)


def test_camb_rejects_non_lcdm_cosmology():
    with pytest.raises(ValueError, match="CAMB will only work with LCDM or wCDM"):
        CAMB(object(), extrapolate_with_eh=False)


def test_camb_no_extrapolation_branch(base_cosmo):
    model = CAMB(base_cosmo, extrapolate_with_eh=False)
    lnk = np.log(np.logspace(-3, -2, 4))

    out = model.lnt(lnk)

    assert out.shape == lnk.shape


def test_camb_getstate_and_setstate(base_cosmo):
    model = CAMB(base_cosmo, extrapolate_with_eh=False)
    state = model.__getstate__()

    restored = CAMB.__new__(CAMB)
    restored.__setstate__(state)

    assert isinstance(restored.params["camb_params"], camb.CAMBparams)


def test_camb_getstate_warns_for_missing_and_unpickleable(base_cosmo):
    class DummyCambParams:
        def __init__(self):
            self.WantCls = lambda: None

        def __getattr__(self, name):
            raise AttributeError(name)

    model = CAMB(base_cosmo, extrapolate_with_eh=False)
    model.params["camb_params"] = DummyCambParams()

    with pytest.warns(UserWarning, match="CAMB key"):
        model.__getstate__()


def _write_camb_transfer_file(path, m_nu):
    """Write CAMB's own transfer output in CAMB's file format; return the file and f_nu."""
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=70.0, ombh2=0.05 * 0.7**2, omch2=0.25 * 0.7**2, mnu=m_nu, neutrino_hierarchy="degenerate"
    )
    pars.WantTransfer = True
    pars.Transfer.kmax = 10.0
    transfer = camb.get_transfer_functions(pars).get_matter_transfer_data().transfer_data
    np.savetxt(path, transfer[:, :, 0].T)
    f_nu = pars.omnuh2 / (pars.omnuh2 + pars.omch2 + pars.ombh2)
    return str(path), f_nu


def test_camb_file_columns_match_camb_constants():
    assert {
        "tot": camb.model.Transfer_tot - 1,
        "cb": camb.model.Transfer_nonu - 1,
    } == transfer_models._CAMB_FILE_COLUMNS


@pytest.mark.parametrize("m_nu", [0.0, 0.3])
def test_fromfile_matter_species(tmp_path, base_cosmo, m_nu):
    """Reading a CAMB file, P_cb and P_tot obey the neutrino free-streaming limits.

    Large scales: delta_tot = delta_cb. Well inside the free-streaming scale:
    delta_tot -> (1 - f_nu) delta_cb. With massless neutrinos the two are identical.
    """
    fname, f_nu = _write_camb_transfer_file(tmp_path / "camb_transfer.dat", m_nu)
    lnk = np.log(np.logspace(-3.5, 0.9, 100))
    k = np.exp(lnk)

    lnt_tot = FromFile(base_cosmo, fname=fname, matter_species="tot").lnt(lnk)
    lnt_cb = FromFile(base_cosmo, fname=fname, matter_species="cb").lnt(lnk)
    ratio = np.exp(lnt_tot - lnt_cb)

    if m_nu == 0:
        np.testing.assert_allclose(ratio, 1.0, rtol=1e-6)
    else:
        np.testing.assert_allclose(ratio[k < 1e-3], 1.0, rtol=1e-3)
        small = k > 1.0
        assert np.all(ratio[small] < 1)
        np.testing.assert_allclose(ratio[small], 1 - f_nu, rtol=5e-3)


def test_fromfile_default_matter_species(tmp_path):
    """By default a CAMB file's CDM+baryon column is read, with a warning if m_nu > 0."""
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3, Ob0=0.05, Tcmb0=2.7255, m_nu=[0, 0, 0.3] * u.eV)
    fname, _ = _write_camb_transfer_file(tmp_path / "camb_transfer.dat", 0.3)
    lnk = np.log(np.logspace(-3, 0.5, 20))
    with pytest.warns(UserWarning, match="matter_species was not set for FromFile"):
        lnt_default = FromFile(cosmo, fname=fname).lnt(lnk)
    np.testing.assert_array_equal(
        lnt_default, FromFile(cosmo, fname=fname, matter_species="cb").lnt(lnk)
    )


@pytest.mark.parametrize("ncols", [2, 7])
def test_fromfile_default_without_cb_column(tmp_path, ncols):
    """Files with no CDM+baryon column keep their old behaviour, without a warning."""
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3, Ob0=0.05, Tcmb0=2.7255, m_nu=[0, 0, 0.3] * u.eV)
    k = np.logspace(-3, 1, 20)
    data = np.column_stack([k] + [1 / (1 + k**2)] * (ncols - 1))
    fname = tmp_path / "transfer.dat"
    np.savetxt(fname, data)
    lnk = np.log(np.array([0.01, 0.1, 1.0]))
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="matter_species was not set")
        lnt_default = FromFile(cosmo, fname=str(fname)).lnt(lnk)
    np.testing.assert_array_equal(
        lnt_default, FromFile(cosmo, fname=str(fname), matter_species="tot").lnt(lnk)
    )


def test_fromfile_cb_needs_nonu_column(tmp_path, base_cosmo):
    """A pre-``Transfer_nonu`` CAMB file (7 columns) has no CDM+baryon column."""
    data = np.column_stack([np.logspace(-3, 1, 20)] + [np.ones(20)] * 6)
    fname = tmp_path / "old_camb.dat"
    np.savetxt(fname, data)
    lnk = np.log(np.array([0.01, 0.1]))

    # "tot" still reads column 6.
    assert FromFile(base_cosmo, fname=str(fname)).lnt(lnk).shape == lnk.shape
    with pytest.raises(ValueError, match="no CDM\\+baryon column"):
        FromFile(base_cosmo, fname=str(fname), matter_species="cb").lnt(lnk)


def test_fromfile_two_column_file_ignores_matter_species(tmp_path, base_cosmo):
    k = np.logspace(-3, 1, 20)
    fname = tmp_path / "two_col.dat"
    np.savetxt(fname, np.column_stack([k, 1 / (1 + k**2)]))
    lnk = np.log(np.array([0.01, 0.1, 1.0]))
    np.testing.assert_array_equal(
        FromFile(base_cosmo, fname=str(fname)).lnt(lnk),
        FromFile(base_cosmo, fname=str(fname), matter_species="cb").lnt(lnk),
    )


def test_fromfile_bad_matter_species(base_cosmo):
    with pytest.raises(ValueError, match="matter_species must be one of"):
        FromFile(base_cosmo, matter_species="nu")


def _direct_camb_lnt(cosmo, **neutrinos):
    """Ln T_tot(k) from CAMB run directly, with CAMB's own neutrino options; returns (kh, lnT)."""
    pars = camb.CAMBparams(DoLensing=False, Want_CMB=False, Want_CMB_lensing=False, WantCls=False)
    pars.set_cosmology(
        H0=cosmo.H0.value,
        ombh2=cosmo.Ob0 * cosmo.h**2,
        omch2=cosmo.Odm0 * cosmo.h**2,
        nnu=cosmo.Neff,
        standard_neutrino_neff=cosmo.Neff,
        TCMB=cosmo.Tcmb0.value,
        **neutrinos,
    )
    pars.WantTransfer = True
    pars.Transfer.high_precision = False
    pars.Transfer.k_per_logint = 0
    pars.Transfer.kmax = 10.0
    transfer = camb.get_transfer_functions(pars).get_matter_transfer_data().transfer_data
    kh = transfer[camb.model.Transfer_kh - 1, :, 0]
    t = transfer[camb.model.Transfer_tot - 1, :, 0]
    return kh, np.log(t / t[0])


def _hmf_camb(cosmo):
    return CAMB(cosmo, matter_species="tot", extrapolate_with_eh=False, kmax=10.0)


def test_camb_three_degenerate_neutrinos_match_direct_camb():
    """m_nu=[0.1, 0.1, 0.1] eV is three massive species, not one of 0.3 eV.

    Both have the same neutrino density, but lighter neutrinos free-stream on larger
    scales, which changes T(k) at k ~ 0.01-1 h/Mpc by ~2%.
    """
    cosmo = Planck18.clone(m_nu=[0.1, 0.1, 0.1] * u.eV)
    model = _hmf_camb(cosmo)

    p = model.params["camb_params"]
    assert p.num_nu_massive == 3
    assert p.nu_mass_eigenstates == 1

    kh, lnt_three = _direct_camb_lnt(cosmo, mnu=0.3, num_massive_neutrinos=3)
    _, lnt_one = _direct_camb_lnt(cosmo, mnu=0.3, num_massive_neutrinos=1)

    lnt = model.lnt(np.log(kh))
    np.testing.assert_allclose(lnt, lnt_three, rtol=0, atol=1e-5)
    # The one-species run is measurably different, so the test can tell them apart.
    assert np.max(np.abs(lnt - lnt_one)) > 1e-2


def test_camb_split_neutrino_masses_match_camb_normal_hierarchy():
    """Non-degenerate masses become separate mass eigenstates.

    CAMB's own ``neutrino_hierarchy="normal"`` uses two eigenstates (two light
    species of equal mass and one heavy one); giving hmf those same three masses
    must reproduce CAMB's normal-hierarchy transfer function.
    """
    mnu = 0.1
    ref = camb.CAMBparams()
    ref.set_cosmology(H0=67.66, mnu=mnu, neutrino_hierarchy="normal")
    assert list(ref.nu_mass_numbers[:2]) == [2, 1]
    m_light, m_heavy = (f * mnu / n for f, n in zip(ref.nu_mass_fractions[:2], [2, 1], strict=True))

    cosmo = Planck18.clone(m_nu=[m_light, m_light, m_heavy] * u.eV)
    model = _hmf_camb(cosmo)

    p = model.params["camb_params"]
    assert p.num_nu_massive == 3
    assert p.nu_mass_eigenstates == 2

    kh, lnt_normal = _direct_camb_lnt(cosmo, mnu=mnu, neutrino_hierarchy="normal")
    np.testing.assert_allclose(model.lnt(np.log(kh)), lnt_normal, rtol=0, atol=1e-5)


@pytest.mark.parametrize(
    ("m_nu", "n_massive"),
    [([0.0, 0.0, 0.0], 0), ([0.0, 0.0, 0.06], 1), ([0.0, 0.05, 0.05], 2), ([0.01, 0.02, 0.05], 3)],
)
def test_camb_neutrino_species_conserve_neff_and_mass(m_nu, n_massive):
    """Each massive species keeps its share of Neff, and each eigenstate its share of mass."""
    cosmo = Planck18.clone(m_nu=m_nu * u.eV)
    p = CAMB(cosmo, matter_species="tot", extrapolate_with_eh=False).params["camb_params"]

    n_eig = p.nu_mass_eigenstates
    degeneracies = np.array(p.nu_mass_degeneracies[:n_eig])
    numbers = np.array(p.nu_mass_numbers[:n_eig])
    fractions = np.array(p.nu_mass_fractions[:n_eig])

    assert p.num_nu_massive == n_massive
    assert numbers.sum() == n_massive
    assert p.num_nu_massless + degeneracies.sum() == pytest.approx(cosmo.Neff, rel=1e-10)

    masses = np.array(m_nu)[np.array(m_nu) > 0]
    if n_massive:
        # Equal effective number of relativistic species per massive neutrino.
        np.testing.assert_allclose(degeneracies / numbers, degeneracies[0] / numbers[0])
        # Mass per species in each eigenstate is the input mass.
        np.testing.assert_allclose(
            np.sort(fractions * sum(m_nu) / numbers), np.unique(masses), rtol=1e-10
        )


def test_camb_getstate_keeps_neutrino_species():
    """The pickle round trip re-applies the per-species neutrino masses."""
    cosmo = Planck18.clone(m_nu=[0.01, 0.02, 0.05] * u.eV)
    model = CAMB(cosmo, matter_species="tot", extrapolate_with_eh=False)

    restored = CAMB.__new__(CAMB)
    restored.__setstate__(model.__getstate__())

    p, q = model.params["camb_params"], restored.params["camb_params"]
    assert q.nu_mass_eigenstates == p.nu_mass_eigenstates == 3
    np.testing.assert_allclose(q.nu_mass_fractions[:3], p.nu_mass_fractions[:3], rtol=1e-12)
