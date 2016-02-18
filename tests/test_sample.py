import numpy as np
import inspect
import os
LOCATION = "/".join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split("/")[:-1])
import sys
sys.path.insert(0, LOCATION)
from hmf.sample import sample_mf, dndm_from_sample
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from nose.tools import assert_raises

def test_circular():
    np.random.seed(1234)
    m,h = sample_mf(1e5,11,transfer_model="EH")
    centres,hist = dndm_from_sample(m,1e5/h.ngtm[0])

    s = spline(np.log10(h.m),np.log10(h.dndm))

    print hist, 10**s(centres)
    assert np.allclose(hist,10**s(centres),rtol=0.05)


def test_mmax_big():
    #raises ValueError because ngtm=0 exactly at m=18
    #due to hard limit of integration in integrate_hmf.
    np.random.seed(12345)
    m,h = sample_mf(1e5,11,transfer_model="EH",Mmax=18)
    #centres,hist = dndm_from_sample(m,1e5/h.ngtm[0])
    #print centres,hist
    assert_raises(ValueError,dndm_from_sample,m,1e5/h.ngtm[0])

