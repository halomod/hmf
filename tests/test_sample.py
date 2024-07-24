import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline

from hmf.helpers.sample import dndm_from_sample, sample_mf


def test_circular():
    rng = np.random.default_rng(1234)
    m, h = sample_mf(1e5, 11, transfer_model="EH", rng=rng)
    centres, hist = dndm_from_sample(m, 1e5 / h.ngtm[0])

    s = spline(np.log10(h.m), np.log10(h.dndm))

    print(hist, 10 ** s(centres))
    assert np.allclose(hist, 10 ** s(centres), rtol=0.05)


def test_mmax_big():
    rng = np.random.default_rng(12345)

    m, h = sample_mf(1e5, 11, transfer_model="EH", Mmax=18, rng=rng)
    # centres,hist = dndm_from_sample(m,1e5/h.ngtm[0])
    # print centres,hist
    dndm_from_sample(m, 1e5 / h.ngtm[0])
