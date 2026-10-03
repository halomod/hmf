"""Tests of HMF."""

import numpy as np
import pytest

from hmf import MassFunction


def test_wrong_filter():
    with pytest.raises(ValueError, match=r"2 must be str or Component subclass"):
        MassFunction(filter_model=2)


def test_string_dc():
    with pytest.raises(ValueError, match=r"delta_c must be a number"):
        MassFunction(delta_c="this")


def test_neg_dc():
    with pytest.raises(ValueError, match=r"delta_c must be > 0"):
        MassFunction(delta_c=-1)


def test_big_dc():
    with pytest.raises(ValueError, match=r"delta_c must be < 10.0"):
        MassFunction(delta_c=20.0)


def test_wrong_fit():
    with pytest.raises(ValueError, match=r"must be str or Component subclass"):
        MassFunction(hmf_model=1)


def test_wrong_mf_par():
    with pytest.raises(ValueError, match=r"hmf_params must be a dictionary"):
        MassFunction(hmf_params=2)


def test_str_filter():
    h = MassFunction(filter_model="TopHat", transfer_model="EH")
    h_ = MassFunction(filter_model="TopHat", transfer_model="EH")

    assert np.allclose(h.sigma, h_.sigma)


def test_mass_nonlinear_outside_range():
    h = MassFunction(Mmin=8, Mmax=9, transfer_model="EH")
    with pytest.warns(UserWarning, match="Nonlinear mass outside mass range"):
        assert h.mass_nonlinear > 0


def test_nu():
    h = MassFunction(Mmin=8, Mmax=18, transfer_model="EH")
    assert np.allclose(h.nu_fn(h.m), h.nu)


def test_sigma8z():
    h = MassFunction(z=0.0, sigma_8=0.8, Mmin=8, Mmax=18, transfer_model="EH")
    assert np.allclose(h.sigma8_z, 0.8)


def test_neff_at_collapse():
    h = MassFunction(Mmin=8, Mmax=18, transfer_model="EH")
    assert np.allclose(h.n_eff_at_collapse, h.n_eff[np.argmin(np.abs(h.nu - 1.0))], rtol=0.05)


class _LntCounter:
    """Wrap ``BondEfs.lnt`` so tests can count transfer-function evaluations."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch):
        from hmf.density_field.transfer_models import BondEfs

        self.calls = 0
        original = BondEfs.lnt

        def counting_lnt(model, lnk):
            self.calls += 1
            return original(model, lnk)

        monkeypatch.setattr(BondEfs, "lnt", counting_lnt)


def _bondefs_mf(**kwargs) -> MassFunction:
    return MassFunction(
        transfer_model="BondEfs",
        transfer_params={"a": 37.1, "b": 21.1},
        cosmo_params={"Om0": 0.3, "H0": 70.0},
        **kwargs,
    )


@pytest.mark.parametrize(
    "update",
    [
        {"cosmo_params": {"Om0": 0.3}},
        {"transfer_params": {"a": 37.1}},
        {"cosmo_params": {"Om0": 0.3}, "transfer_params": {"b": 21.1}},
    ],
)
def test_noop_subset_dict_update_does_not_recompute_transfer(monkeypatch, update):
    """Regression test for #109: a no-op partial dict update must not invalidate."""
    counter = _LntCounter(monkeypatch)
    mf = _bondefs_mf()
    dndm = mf.dndm.copy()
    assert counter.calls > 0

    counter.calls = 0
    mf.update(**update)
    np.testing.assert_allclose(mf.dndm, dndm, rtol=0, atol=0)
    assert counter.calls == 0

    # The stored dicts are untouched by the no-op merge.
    assert mf.cosmo_params == {"Om0": 0.3, "H0": 70.0}
    assert mf.transfer_params == {"a": 37.1, "b": 21.1}


@pytest.mark.parametrize(
    ("update", "expected"),
    [
        (
            {"cosmo_params": {"Om0": 0.32}},
            {"cosmo_params": {"Om0": 0.32, "H0": 70.0}},
        ),
        (
            {"transfer_params": {"a": 30.0}},
            {"transfer_params": {"a": 30.0, "b": 21.1}},
        ),
    ],
)
def test_changing_subset_dict_update_invalidates(monkeypatch, update, expected):
    """A partial dict update that changes a value must recompute and merge."""
    counter = _LntCounter(monkeypatch)
    mf = _bondefs_mf()
    dndm_old = mf.dndm.copy()

    counter.calls = 0
    mf.update(**update)
    dndm_new = mf.dndm
    assert counter.calls > 0
    assert not np.allclose(dndm_new, dndm_old, rtol=1e-6, atol=0)

    fresh = _bondefs_mf()
    fresh.update(**expected)
    for key, value in expected.items():
        assert getattr(mf, key) == value
    np.testing.assert_allclose(dndm_new, fresh.dndm, rtol=1e-12, atol=0)


def test_empty_dict_update_clears_params():
    """Passing an empty dict still clears a ``*_params`` dict."""
    mf = _bondefs_mf()
    mf.dndm
    mf.update(cosmo_params={})
    assert mf.cosmo_params == {}

    fresh = MassFunction(transfer_model="BondEfs", transfer_params={"a": 37.1, "b": 21.1})
    np.testing.assert_allclose(mf.dndm, fresh.dndm, rtol=1e-12, atol=0)
