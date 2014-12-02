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
import collections
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


def wdm_transfer(m_x, power_cdm, lnk, h, omegac):
    """
    Transform a CDM Power Spectrum into WDM.
    
    Formula from Bode et. al. 2001 eq. A9
    
    Parameters
    ----------
    m_x : float
        The mass of the single-species WDM particle in *keV*
        
    power_cdm : array
        The normalised power spectrum of CDM.
        
    lnk : array
        The wavenumbers *k/h* corresponding to  ``power_cdm``.
        
    h : float
        Hubble parameter
        
    omegac : float
        The dark matter density as a ratio of critical density at the current 
        epoch.
    
    Returns
    -------
    power_wdm : array
        The normalised WDM power spectrum at ``lnk``.
        
    """
    g_x = 1.5
    nu = 1.12

    alpha = 0.049 * (omegac / 0.25) ** 0.11 * (h / 0.7) ** 1.22 * (1 / m_x) ** 1.11 * (1.5 / g_x) ** 0.29

    transfer = (1 + (alpha * np.exp(lnk)) ** (2 * nu)) ** -(5.0 / nu)
    print transfer
    return power_cdm + 2 * np.log(transfer)

def mass_to_radius(m, mean_dens):
        return (3.*m / (4.*np.pi * mean_dens)) ** (1. / 3.)

def dlnsdlnm(M, sigma, power, lnk, mean_dens):
    r"""
    Calculate :math:\frac{d \ln(\sigma)}{d \ln M}`
    
    Parameters
    ----------
    M : array
        The masses 
        
    sigma : array
        Mass variance at M

    power : array
        The logarithmic power spectrum at ``lnk``
        
    lnk : array
        The wavenumbers *k/h* corresponding to the power
        
    mean_dens : float
        Mean density of the universe.
    
    Returns
    -------
    dlnsdlnM : array
    """
    dlnk = lnk[1] - lnk[0]
    R = mass_to_radius(M, mean_dens)
    dlnsdlnM = np.zeros_like(M)
    for i, r in enumerate(R):
        g = np.exp(lnk) * r
        w = dw2dm(M[i], g)  # Derivative of W^2
#         integ = w * np.exp(power - lnk)
        integ = w * np.exp(power + 3 * lnk)
#         dlnsdlnM[i] = (3.0 / (2.0 * sigma[i] ** 2 * np.pi ** 2 * r ** 4)) * intg.simps(integ, dx=dlnk)
        dlnsdlnM[i] = (M[i] / (4.0 * sigma[i] ** 2 * np.pi ** 2)) * intg.simps(integ, dx=dlnk)
    return dlnsdlnM

def dw2dm(m, kR):
    """
    The derivative of the top-hat window function squared
    
    Parameters
    ----------
    kR : array
        Product of wavenumber with R [final product is unitless]
        
    Returns
    -------
    dw2dm : array
        The derivative of the top-hat window function squared.
    """
    return 6 * (np.sin(kR) - kR * np.cos(kR)) * (np.sin(kR) * (1 - 3.0 / (kR ** 2)) + 3.0 * np.cos(kR) / kR) / (m * kR ** 4)

def n_eff(dlnsdlnm):
    """
    The power spectral slope at the scale of the halo radius 
    
    Parameters
    ----------
    dlnsdlnm : array
        The derivative of log sigma with log M
    
    Returns
    -------
    n_eff : float

    Notes
    -----
    Uses eq. 42 in Lukic et. al 2007.
    """

    n_eff = -3.0 * (2.0 * dlnsdlnm + 1.0)

    return n_eff


def d_plus(z, cdict):
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

    return dplus

def growth_factor(z, cdict):
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

    growth = d_plus(z, cdict) / d_plus(0.0, cdict)

    return growth

