import numpy as np
import inspect
import os
LOCATION = "/".join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split("/")[:-1])
import sys
sys.path.insert(0, LOCATION)
from mpmath import gammainc
from hmf.integrate_hmf import hmf_integral_gtm

class TestAnalyticIntegral(object):
    def __init__(self):
        pass

    def tggd(self,m,loghs,alpha,beta):
        return beta*(m/10**loghs)**alpha * np.exp(-(m/10**loghs)**beta)

    def anl_int(self,m,loghs,alpha,beta):
        return 10**loghs * gammainc((alpha+1)/beta,(m/10**loghs)**beta)

    def anl_m_int(self,m,loghs,alpha,beta):
        return 10**(2*loghs) * gammainc((alpha+2)/beta,(m/10**loghs)**beta)

    def test_basic(self):
        m = np.logspace(10,18,500)
        dndm = self.tggd(m,14.0,-1.9,0.8)
        ngtm = self.anl_int(m,14.0,-1.9,0.8)

        assert np.allclose(ngtm,hmf_integral_gtm(m,dndm))

    def test_basic_mgtm(self):
        m = np.logspace(10,18,500)
        dndm = self.tggd(m,14.0,-1.9,0.8)
        ngtm = self.anl_m_int(m,14.0,-1.9,0.8)

        assert np.allclose(ngtm,hmf_integral_gtm(m,dndm,True))

    def test_high_z(self):
        m = np.logspace(10,18,500)
        dndm = self.tggd(m,9.0,-1.93,0.4)
        ngtm = self.anl_int(m,9.0,-1.93,0.4)

        assert np.allclose(ngtm,hmf_integral_gtm(m,dndm))

    def test_low_mmax_z0(self):
        m = np.logspace(10,14,500)
        dndm = self.tggd(m,14.0,-1.9,0.8)
        ngtm = self.anl_int(m,14.0,-1.9,0.8)

        assert np.allclose(ngtm,hmf_integral_gtm(m,dndm))

    def test_low_mmax_high_z(self):
        m = np.logspace(10,14,500)
        dndm = self.tggd(m,9.0,-1.93,0.4)
        ngtm = self.anl_int(m,9.0,-1.93,0.4)

        assert np.allclose(ngtm,hmf_integral_gtm(m,dndm))
