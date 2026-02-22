import camb
import numpy as np
import pytest
from astropy.cosmology import LambdaCDM

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
