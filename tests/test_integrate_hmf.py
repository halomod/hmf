import numpy as np
from mpmath import gammainc as _mp_ginc

from hmf.mass_function.integrate_hmf import hmf_integral_gtm


def _flt(a):
    try:
        return a.astype("float")
    except AttributeError:
        return float(a)


_ginc_ufunc = np.frompyfunc(lambda z, x: _mp_ginc(z, x), 2, 1)


def gammainc(z, x):
    return _flt(_ginc_ufunc(z, x))


class TestAnalyticIntegral:
    def tggd(self, m, loghs, alpha, beta):
        return beta * (m / 10 ** loghs) ** alpha * np.exp(-((m / 10 ** loghs) ** beta))

    def anl_int(self, m, loghs, alpha, beta):
        return 10 ** loghs * gammainc((alpha + 1) / beta, (m / 10 ** loghs) ** beta)

    def anl_m_int(self, m, loghs, alpha, beta):
        return 10 ** (2 * loghs) * gammainc(
            (alpha + 2) / beta, (m / 10 ** loghs) ** beta
        )

    # def test_basic(self):
    #     m = np.logspace(10,18,500)
    #     dndm = self.tggd(m,14.0,-1.9,0.8)
    #     ngtm = self.anl_int(m,14.0,-1.9,0.8)
    #
    #     print ngtm/hmf_integral_gtm(m,dndm)
    #     assert np.allclose(ngtm,hmf_integral_gtm(m,dndm),rtol=0.03)
    #
    # def test_basic_mgtm(self):
    #     m = np.logspace(10,18,500)
    #     dndm = self.tggd(m,14.0,-1.9,0.8)
    #     ngtm = self.anl_m_int(m,14.0,-1.9,0.8)
    #
    #     print ngtm/hmf_integral_gtm(m,dndm,True)
    #     assert np.allclose(ngtm,hmf_integral_gtm(m,dndm,True),rtol=0.03)

    def test_high_z(self):
        m = np.logspace(10, 18, 500)
        dndm = self.tggd(m, 9.0, -1.93, 0.4)
        ngtm = self.anl_int(m, 9.0, -1.93, 0.4)

        print(ngtm / hmf_integral_gtm(m, dndm))
        assert np.allclose(ngtm, hmf_integral_gtm(m, dndm), rtol=0.03)

    # def test_low_mmax_z0(self):
    #     m = np.logspace(10,15,500)
    #     dndm = self.tggd(m,14.0,-1.9,0.8)
    #     ngtm = self.anl_int(m,14.0,-1.9,0.8)
    #
    #     print ngtm/hmf_integral_gtm(m,dndm)
    #     assert np.allclose(ngtm,hmf_integral_gtm(m,dndm),rtol=0.03)

    def test_low_mmax_high_z(self):
        m = np.logspace(10, 15, 500)
        dndm = self.tggd(m, 9.0, -1.93, 0.4)
        ngtm = self.anl_int(m, 9.0, -1.93, 0.4)

        print(ngtm / hmf_integral_gtm(m, dndm))
        assert np.allclose(ngtm, hmf_integral_gtm(m, dndm), rtol=0.03)
