from scipy.interpolate import InterpolatedUnivariateSpline as spline
import numpy as np
from scipy.integrate import simps, cumtrapz

def hmf_integral_gtm(M, dndm, mass_density=False):
    """
    Integrate dn/dm to get number or mass density above M
    
    Parameters
    ----------
    M : array_like
        Array of masses
        
    dndm : array_like
        Array of dn/dm (corresponding to M)
        
    mass_density : bool, `False`
        Whether to calculate mass density (or number density).
    """
    # Eliminate NaN's
    m = M[np.logical_not(np.isnan(dndm))]
    dndm = dndm[np.logical_not(np.isnan(dndm))]
    dndlnm = m * dndm
    # Calculate the mass function (and its integral) from the highest M up to 10**18
    if m[-1] < m[0] * 10 ** 18 / m[3]:
        m_upper = np.arange(np.log(m[-1]), np.log(10 ** 18), np.log(m[1]) - np.log(m[0]))
        mf_func = spline(np.log(m), np.log(dndlnm), k=1)
        mf = mf_func(m_upper)

        if not mass_density:
            int_upper = simps(np.exp(mf), dx=m_upper[2] - m_upper[1], even='first')
        else:
            int_upper = simps(np.exp(m_upper + mf), dx=m_upper[2] - m_upper[1], even='first')
    else:
        int_upper = 0

    # Calculate the cumulative integral (backwards) of [m*]dndlnm
    if not mass_density:
        ngtm = np.concatenate((cumtrapz(dndlnm[::-1], dx=np.log(m[1]) - np.log(m[0]))[::-1], np.zeros(1)))
    else:
        ngtm = np.concatenate((cumtrapz(m[::-1] * dndlnm[::-1], dx=np.log(m[1]) - np.log(m[0]))[::-1], np.zeros(1)))

    return (ngtm + int_upper) * dndlnm.unit
