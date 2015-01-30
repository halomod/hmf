'''
A collection of functions which do some of the core work of the HMF calculation.

The routines here could be made more 'elegant' by taking `MassFunction` or 
`Transfer` objects as arguments, but we keep them simple for the sake of 
flexibility.
'''

#===============================================================================
# Imports
#===============================================================================
import numpy as np
import scipy.integrate as intg
import cosmolopy as cp
import logging

from filters import TopHat
logger = logging.getLogger('hmf')
#===============================================================================
# Functions
#===============================================================================
def check_kr(min_m, max_m, mean_dens, mink, maxk):
    """
    Check the bounds of the product of k*r
    
    If the bounds are not high/low enough, then there can be information loss
    in the calculation of the mass variance. This routine returns a warning
    indicating the necessary adjustment for requisite accuracy.
    
    See http://arxiv.org/abs/1306.6721 for details. 
    """
    # Define min and max radius
    min_r = mass_to_radius(min_m, mean_dens)
    max_r = mass_to_radius(max_m, mean_dens)

    if np.exp(maxk) * min_r < 3:
        logger.warn("r_min (%s) * k_max (%s) < 3. Mass variance could be inaccurate." % (min_r, np.exp(maxk)))
    elif np.exp(mink) * max_r > 0.1:
        logger.warn("r_max (%s) * k_min (%s) > 0.1. Mass variance could be inaccurate." % (max_r, np.exp(mink)))

def normalize(norm_sigma_8, unn_power, lnk, mean_dens):
    """
    Normalize the power spectrum to a given :math:`\sigma_8`
    
    Parameters
    ----------
    norm_sigma_8 : float
        The value of :math:`\sigma_8` to normalize to.
        
    unn_power : array_like
        The natural logarithm of the unnormalised power spectrum
        
    lnk : array_like
        The natural logarithm of the values of *k/h* at which `unn_power` is 
        defined.
        
    mean_dens : float
        The mean density of the Universe.
        
    Returns
    -------
    power : array_like
        An array of the same length as `unn_power` in which the values are 
        normalised to :math:``sigma_8`
        
    normalisation : float
        The normalisation constant. 
    """
    # Calculate the value of sigma_8 without prior normalization.

    filter = TopHat(mean_dens, None, lnk, unn_power)
    sigma_8 = filter.sigma(8.0)[0]

    # Calculate the normalization factor
    normalization = norm_sigma_8 / sigma_8

    # Normalize the previously calculated power spectrum.
    power = 2 * np.log(normalization) + unn_power

    return power, normalization


def d_plus(z, cdict, getvec=False):
    """
    Finds the factor :math:`D^+(a)`, from Lukic et. al. 2007, eq. 8.
    
    Uses simpson's rule to integrate, with 1000 steps.
    
    Parameters
    ----------
    z : float
        The redshift
        
    cosmo : ``hmf.cosmo.Cosmology()`` object
        Cosmological parameters 
    
    Returns
    -------
    dplus : float
        The un-normalised growth factor.
    """
    a_upper = 1.0 / (1.0 + z)
    lna = np.linspace(np.log(1e-8), np.log(a_upper), 1000)
    z_vec = 1.0 / np.exp(lna) - 1.0

    integrand = 1.0 / (np.exp(lna) * cp.distance.e_z(z_vec, **cdict)) ** 3
    integral = intg.simps(np.exp(lna) * integrand, dx=lna[1] - lna[0])
    dplus = 5.0 * cdict["omega_M_0"] * cp.distance.e_z(z, **cdict) * integral / 2.0

    if getvec:
        lna = np.linspace(lna[-1], 0.0, 1000)
        z_vec = 1.0 / np.exp(lna) - 1.0
        integrand = 1.0 / (np.exp(lna) * cp.distance.e_z(z_vec, **cdict)) ** 3
        integral = intg.cumtrapz(np.exp(lna) * integrand, dx=lna[1] - lna[0], initial=0.0)

        dplus += 5.0 * cdict["omega_M_0"] * cp.distance.e_z(z_vec, **cdict) * integral / 2.0

    if getvec:
        dplus = np.vstack((z_vec, dplus))  # spline(z_vec[1:], dplus)

    return dplus

def growth_factor(z, cdict, getvec=False):
    """
    Calculate :math:`d(a) = D^+(a)/D^+(a=1)`, from Lukic et. al. 2007, eq. 7.
    
    Parameters
    ----------
    z : float
        The redshift
        
    cosmo : ``hmf.cosmo.Cosmology()`` object
        Cosmological parameters 
    
    Returns
    -------
    growth : float
        The normalised growth factor.
    """
    if not getvec:
        growth = d_plus(z, cdict, getvec) / d_plus(0.0, cdict)
    else:
        growth = d_plus(z, cdict, getvec)
        growth[1, :] /= d_plus(0.0, cdict)
#         growth = lambda z: dp(z) / d_plus(0.0, cdict)

    return growth

