'''
A collection of functions which do some of the core work of the HMF calculation.
'''

#===============================================================================
# Imports
#===============================================================================
import numpy as np
import scipy.integrate as intg
import pycamb
import collections
import cosmolopy.perturbation as pert
import cosmolopy.distance as cdist
from scipy.interpolate import InterpolatedUnivariateSpline as spline

#===============================================================================
# Functions
#===============================================================================
def get_transfer(transfer_file, cosmo_params, transfer_fit,
                 camb_options=None, k_bounds=None):
    """
    Calculate a transfer function.
    
    Uses either CAMB or the EH transfer functions.
    
    Parameters
    ----------
    transfer_file : `str` or ``None``
        A full or relative path to a file containing the transfer function.
        The file must be in the same format as CAMB transfer function output,
        ie. 7 columns, the first being k/h and the 7th being the total transfer.
        
    cosmo_params : `Cosmology` instance
        An instance of the `Cosmology` class from this package. Contains all 
        cosmological parameters.
        
    transfer_fit : `str` {``"CAMB"``,``"EH"``}
        A string identifier specifying which transfer function fit to use.
        
    camb_options : `dict`, default ``None``
        A dictionary of options to send to the `pycamb.transfers` routine. Does
        not include cosmological parameters.
        
    k_bounds : sequence, len=2
        Minimum and maximum bounds on *k/h*. Only used for the EH transfer fit.
        
    Returns
    -------
    T : array_like
        An array with the first column containing values of ln*(k/h)* and the second
        the natural log of the total transfer function.
    """
    # If no transfer file uploaded, but it was custom, execute CAMB
    if transfer_file is None:
        if transfer_fit == "CAMB":
            cdict = dict(cosmo_params.pycamb_dict(),
                         **camb_options)
            k, T, sig8 = pycamb.transfers(**cdict)
            T = np.log(T[[0, 6], :, 0])
            del sig8, k
        elif transfer_fit == "EH":
            k = np.exp(np.linspace(np.log(k_bounds[0]), np.log(k_bounds[1]), 250))
            # Since the function natively calculates the transfer based on k in Mpc^-1,
            # we need to multiply by h.
            t, T = pert.transfer_function_EH(k * cosmo_params.h,
                                             **cosmo_params.cosmolopy_dict())
            T = np.vstack((np.log(k), np.log(T)))
            del t
    else:
        # Import the transfer file
        transfer = np.log(np.genfromtxt(transfer_file)[:, [0, 6]].T)
        k = transfer[0, :]
        T = transfer[1, :]
    return T

def check_kr(min_m, max_m, mean_dens, mink, maxk):
    """
    Check the bounds of the product of k*r
    
    If the bounds are not high/low enough, then there can be information loss
    in the calculation of the mass variance.
    
    See http://arxiv.org/abs/1306.6721 
    """
    # Define mass from radius function
    def M(r):
        return 4 * np.pi * r ** 3 * mean_dens / 3

    # Define min and max radius
    min_r = (3 * min_m / (4 * np.pi * mean_dens)) ** (1. / 3.)
    max_r = (3 * max_m / (4 * np.pi * mean_dens)) ** (1. / 3.)

    errmsg1 = \
"""
Please make sure the product of minimum radius and maximum k is > 3.
If it is not, then the mass variance could be extremely inaccurate.
                    
"""

    errmsg2 = \
"""
Please make sure the product of maximum radius and minimum k is < 0.1
If it is not, then the mass variance could be inaccurate.
                    
"""

    if maxk * min_r < 3:
        error1 = errmsg1 + "This means extrapolating k to " + str(3 / min_r) + " or using min_M > " + str(np.log10(M(3.0 / maxk)))
    else:
        error1 = None

    if mink * max_r > 0.1:
        error2 = errmsg2 + "This means extrapolating k to " + str(0.1 / max_r) + " or using max_M < " + str(np.log10(M(0.1 / mink)))
    else:
        error2 = None

    return error1, error2

def interpolate_transfer(k, transfer, tol=0.01):
    """
    Interpolate the given transfer function
    
    Parameters
    ----------
    k : array_like
        Value of lnk
        
    transfer : array_like
        values of lnT
        
    tol : float, default 0.01
        The routine performs a check that at low-k the slope asymptotes to a 
        slope of zero. If this is not true the transfer function is cut when the 
        slope is out by tol. 
              
    Notes
    -----
    Interpolation is linear as we may have to extrapolate.
    """
    # Unfortunately it looks like there's a turn-up at low-k for some CAMB
    # transfers which makes the extrapolation silly.
    # If this is the case, we start when it is nice.

    start = 0
    for i in range(len(k) - 1):
        if abs((transfer[i + 1] - transfer[i]) / (k[i + 1] - k[i])) < tol:
            start = i
            break
    if start > 0:
        transfer = transfer[start:-1]
        k = k[start:-1]
        spline_order = 1
    else:
        spline_order = 1

    transfer_function = spline(k, transfer, k=spline_order)

    return transfer_function

def normalize(norm_sigma_8, unn_power, lnk, mean_dens):
    """
    Normalize the power spectrum to a given sigma_8
    
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


    sigma_8 = mass_variance(4.*np.pi * 8 ** 3 * mean_dens / 3., unn_power, lnk, mean_dens)[0]

    # Calculate the normalization factor
    normalization = norm_sigma_8 / sigma_8

    # Normalize the previously calculated power spectrum.
    power = 2 * np.log(normalization) + unn_power

    return power, normalization

def mass_variance(M, power, lnk, mean_dens):
    """
    Calculate the Mass Variance of M using the top-hat window function.
    
    Parameters
    ----------
    M : float or array_like
        The mass of the sphere at which to calculate the mass variance.
        
    power : array_like
        The (normalised) natural log of the power spectrum
        
    lnk : array_like
        The natural logarithm of the values of *k/h* at which `power` is 
        defined.
        
    mean_dens : float
        Mean density of the Universe
        
    Returns
    sigma : array_like (len(M))
        The mass variance at ``M``
        
    """

    # If we input a scalar as M, then just make it a one-element list.
    if not isinstance(M, collections.Iterable):
        M = [M]

    dlnk = lnk[1] - lnk[0]
    # Calculate the integrand of the function. Note that the power spectrum and k values must be
    # 'un-logged' before use, and we multiply by k because our steps are in logk.
    sigma = np.zeros_like(M)
    rest = np.exp(power + 3 * lnk)
    for i, m in enumerate(M):
        integ = rest * top_hat_window(m, lnk, mean_dens)
        sigma[i] = (0.5 / np.pi ** 2) * intg.trapz(integ, dx=dlnk)

    return np.sqrt(sigma)

def top_hat_window(M, lnk, mean_dens):
    """
    The fourier-space top-hat window function
    
    Parameters
    ----------
    M : float or array of floats
        The masses at which to evaluate the function
        
    lnk : float or array of floats
        The natural log of *k/h* at which to evaluate the function. Only one 
        of lnk or M may be an array.
        
    mean_dens : float
        The mean density of the universe.
       
    Returns
    -------
    W_squared : float or array of floats 
        The square of the top-hat window function in fourier space 
    """

    # Calculate the factor kR, minding to un-log k before use.
    kR = np.exp(lnk) * mass_to_radius(M, mean_dens)
    W_squared = (3 * (np.sin(kR) / kR ** 3 - np.cos(kR) / kR ** 2)) ** 2
    W_squared[kR < 0.01] = 1.0

    return W_squared


def mass_to_radius(M, mean_dens):
    """
    Calculate radius of a region of space from its mass.
    
    Parameters
    ----------
    M : float or array of floats
        Masses
        
    mean_dens : float
        The mean density of the universe
        
    Returns
    ------
    R : float or array of floats
        The corresponding radii to M
    
    Notes
    -----
    The units of ``M`` don't matter as long as they are consistent with 
    ``mean_dens``.
    """
    return (3.*M / (4.*np.pi * mean_dens)) ** (1. / 3.)

def radius_to_mass(R, mean_dens):
    """
    Calculates mass of a region of space from its radius
    
    Parameters
    ----------
    R : float or array of floats
        Radii
        
    mean_dens : float
        The mean density of the universe
        
    Returns
    ------
    M : float or array of floats
        The corresponding masses in R
    
    Notes
    -----
    The units of ``R`` don't matter as long as they are consistent with 
    ``mean_dens``.
    """
    return 4 * np.pi * R ** 3 * mean_dens / 3

def wdm_transfer(m_x, power_cdm, lnk, h, omegac):
    """
    Transform a CDM Power Spectrum into WDM.
    
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
        
    Notes
    -----
    Formula from Bode et. al. 2001 eq. A9
    """
    g_x = 1.5
    nu = 1.12

    alpha = 0.049 * (omegac / 0.25) ** 0.11 * (h / 0.7) ** 1.22 * (1 / m_x) ** 1.11 * (1.5 / g_x) ** 0.29

    transfer = (1 + (alpha * np.exp(lnk)) ** (2 * nu)) ** -(5.0 / nu)

    return power_cdm + 2 * np.log(transfer)

def dlnsdlnm(M, sigma, power, lnk, mean_dens):
    """
    Calculate the derivative of log sigma with log M
    
    Parameters
    ----------
    M : array
        The masses 
        
    sigma : array
        Mass variance at M

    power : array
        The power spectrum at ``lnk``
        
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
        w = dw2dm(g)  # Derivative of W^2
        integ = w * np.exp(power - lnk)
        dlnsdlnM[i] = (3.0 / (2.0 * sigma[i] ** 2 * np.pi ** 2 * r ** 4)) * intg.trapz(integ, dx=dlnk)
    return dlnsdlnM

def dw2dm(kR):
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
    return (np.sin(kR) - kR * np.cos(kR)) * (np.sin(kR) * (1 - 3.0 / (kR ** 2)) + 3.0 * np.cos(kR) / kR)

def n_eff(dlnsdlnm):
    """
    Return the power spectral slope at the scale of the halo radius, 
    
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


def d_plus(z, cosmo):
    """
    Finds the factor :math:`D^+(a)`, from Lukic et. al. 2007, eq. 8.
    
    Uses simpson's rule to integrate, with 1000 steps.
    
    Parameters
    ----------
    z : float
        The redshift
        
    cosmo : ``Cosmology()`` object
        Cosmological parameters 
    
    Returns
    -------
    dplus : float
        The un-normalised growth factor.
    """
    cdict = cosmo.cosmolopy_dict()
    a_upper = 1.0 / (1.0 + z)
    lna = np.linspace(np.log(1e-8), np.log(a_upper), 1000)
    z_vec = 1.0 / np.exp(lna) - 1.0

    integrand = 1.0 / (np.exp(lna) * cdist.e_z(z_vec, **cdict)) ** 3

    integral = intg.simps(np.exp(lna) * integrand, dx=lna[1] - lna[0])
    dplus = 5.0 * cosmo.omegam * cdist.e_z(z, **cdict) * integral / 2.0

    return dplus

def growth_factor(z, cosmo):
    """
    Calculate :math:`d(a) = D^+(a)/D^+(a=1)`, from Lukic et. al. 2007, eq. 7.
    
    Parameters
    ----------
    z : float
        The redshift
        
    cosmo : ``Cosmology()`` object
        Cosmological parameters 
    
    Returns
    -------
    growth : float
        The normalised growth factor.
    """

    growth = d_plus(z, cosmo) / d_plus(0.0, cosmo)

    return growth
