import warnings

import camb
import numpy as np
import pytest
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM, LambdaCDM

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
