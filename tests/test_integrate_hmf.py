import warnings

import numpy as np
import pytest
from mpmath import gammainc as _mp_ginc

from hmf import HMFExtrapolationWarning
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
        return beta * (m / 10**loghs) ** alpha * np.exp(-((m / 10**loghs) ** beta))

    def anl_int(self, m, loghs, alpha, beta):
        return 10**loghs * gammainc((alpha + 1) / beta, (m / 10**loghs) ** beta)

    def anl_m_int(self, m, loghs, alpha, beta):
        return 10 ** (2 * loghs) * gammainc((alpha + 2) / beta, (m / 10**loghs) ** beta)

    def test_high_z(self):
        m = np.logspace(10, 18, 500)
        dndm = self.tggd(m, 9.0, -1.93, 0.4)
        ngtm = self.anl_int(m, 9.0, -1.93, 0.4)
        mask = m < 10**15
        np.testing.assert_allclose(
            ngtm[mask], hmf_integral_gtm(m, dndm)[mask], rtol=0.03, atol=1e-8
        )

    def test_low_mmax_high_z(self):
        m = np.logspace(10, 15, 500)
        dndm = self.tggd(m, 9.0, -1.93, 0.4)
        ngtm = self.anl_int(m, 9.0, -1.93, 0.4)

        print(ngtm / hmf_integral_gtm(m, dndm))
        assert np.allclose(ngtm, hmf_integral_gtm(m, dndm), rtol=0.03)


class TestExtrapolationWarning:
    """The power-law extrapolation above m[-1] warns only when it matters."""

    def test_warns_when_tail_is_significant(self):
        # For dn/dm = m^-2 the tail above 1e12 is 1% of n(>1e10).
        m = np.logspace(10, 12, 500)
        with pytest.warns(HMFExtrapolationWarning, match="power law"):
            ngtm = hmf_integral_gtm(m, m**-2.0)
        # The extrapolation is exact for a power law: n(>m) = 1/m (up to the 1e18 cutoff).
        np.testing.assert_allclose(ngtm, 1 / m - 1e-18, rtol=1e-4)

    def test_no_warning_when_tail_is_negligible(self):
        # Same exponentially cut-off mass function as TestAnalyticIntegral, truncated at
        # 1e15: the tail is a negligible fraction of the total, so no warning.
        m = np.logspace(10, 15, 500)
        dndm = TestAnalyticIntegral().tggd(m, 9.0, -1.93, 0.4)
        with warnings.catch_warnings():
            warnings.simplefilter("error", HMFExtrapolationWarning)
            hmf_integral_gtm(m, dndm)

    def test_no_extrapolation_to_1e18(self):
        m = np.logspace(10, 18, 500)
        with warnings.catch_warnings():
            warnings.simplefilter("error", HMFExtrapolationWarning)
            hmf_integral_gtm(m, m**-2.0)
