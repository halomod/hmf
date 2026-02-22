import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from hmf.helpers.sample import _choose_halo_masses_num, dndm_from_sample, sample_mf


def test_circular():
    rng = np.random.default_rng(1234)
    m, h = sample_mf(1e5, 11, transfer_model="EH", rng=rng)
    centres, hist = dndm_from_sample(m, 1e5 / h.ngtm[0])

    s = Spline(np.log10(h.m), np.log10(h.dndm))

    print(hist, 10 ** s(centres))
    assert np.allclose(hist, 10 ** s(centres), rtol=0.05)


def test_mmax_big():
    rng = np.random.default_rng(12345)

    m, h = sample_mf(1e5, 11, transfer_model="EH", Mmax=18, rng=rng)
    dndm_from_sample(m, 1e5 / h.ngtm[0])


def test_sample_mf_sort_descending():
    rng = np.random.default_rng(4)
    m, _ = sample_mf(20, 11, transfer_model="EH", sort=True, rng=rng)

    assert np.all(np.diff(m) <= 0)


def test_choose_halo_masses_default_rng():
    def icdf(x):
        return np.zeros_like(x) + 1.0

    m = _choose_halo_masses_num(5, icdf, xmin=0.2, rng=None)

    assert m.shape == (5,)
    assert np.allclose(m, 10.0)


def test_dndm_from_sample_nan_edges():
    m = np.array([10**2.5])
    bins = np.array([1.0, 2.0, 3.0, 4.0])

    _, hist = dndm_from_sample(m, 1.0, bins=bins)

    assert np.isnan(hist[1])
    assert hist[0] == 0
    assert hist[-1] == 0


def test_dndm_from_sample_empty():
    m = np.array([])

    _, hist = dndm_from_sample(m, 1.0, bins=3)

    assert np.all(hist == 0)
