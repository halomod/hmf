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
import itertools

import logging

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


    sigma_8 = mass_variance(4.*np.pi * 8 ** 3 * mean_dens / 3., unn_power, lnk, mean_dens)[0]

    # Calculate the normalization factor
    normalization = norm_sigma_8 / sigma_8

    # Normalize the previously calculated power spectrum.
    power = 2 * np.log(normalization) + unn_power

    return power, normalization

def mass_variance(M, power, lnk, mean_dens, scheme='trapz'):
    """
    Calculate the mass variance, :math:`\sigma(M)` using the top-hat window function.
    
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
    sigma : array_like ( ``len=len(M)`` )
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
        if scheme == 'trapz':
            sigma[i] = (0.5 / np.pi ** 2) * intg.trapz(integ, dx=dlnk)
        elif scheme == "simps":
            sigma[i] = (0.5 / np.pi ** 2) * intg.simps(integ, dx=dlnk)
        elif scheme == 'romb':
            sigma[i] = (0.5 / np.pi ** 2) * intg.romb(integ, dx=dlnk)
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

    kR = np.exp(lnk) * mass_to_radius(M, mean_dens)
    # # The following 2 lines cut the integral at small scales to prevent numerical error.
    W_squared = np.ones(len(kR))
    kR = kR[kR > 1.4e-6]

    W_squared[-len(kR):] = (3 * (np.sin(kR) / kR ** 3 - np.cos(kR) / kR ** 2)) ** 2
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
    
    .. note :: The units of ``M`` don't matter as long as they are consistent with 
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

    return power_cdm + 2 * np.log(transfer)

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
        w = dw2dm(g)  # Derivative of W^2
        integ = w * np.exp(power - lnk)
        dlnsdlnM[i] = (3.0 / (2.0 * sigma[i] ** 2 * np.pi ** 2 * r ** 4)) * intg.simps(integ, dx=dlnk)
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


def get_hmf(required_attrs, get_label=True, **kwargs):
    """
    Yield :class:`hmf.MassFunction` objects for all combinations of parameters supplied.
    """
    from hmf import MassFunction
    import re

    order = ["delta_h", "delta_wrt", "delta_c", "user_fit", "mf_fit", "cut_fit",
             "z2", "nz", "z", "M", "wdm_mass", "sigma_8", "n", "lnk", "transfer_fit"][::-1]

    ordered_kwargs = collections.OrderedDict([])
    for item in order:
        try:
            if isinstance(kwargs[item], (list, tuple)):
                ordered_kwargs[item] = kwargs.pop(item)
        except KeyError:
            pass
    # # add the rest in any order
    for k in kwargs:
        if isinstance(kwargs[k], (list, tuple)):
            ordered_kwargs[k] = kwargs.pop(k)

    ordered_list = [ordered_kwargs[k] for k in ordered_kwargs]

    # # Now ordered_kwargs contains an ordered dict of list values, and kwargs
    # # has an unordered dict of singular values

    final_list = [dict(zip(ordered_kwargs.keys(), v)) for v in itertools.product(*ordered_list)]

    # We want the highest possible wanted attribute
    if not isinstance(required_attrs, (list, tuple)):
        attribute = required_attrs
    else:
        attribute = "dndm"
        mattrs = ["ngtm", "mgtm", "mltm", "nltm", "how_big", "dndlnm", "dndlog10m", "dndm",
                 "fsigma", "n_eff", "lnsigma", "sigma", "_dlnsdlnm", "_sigma_0", "M"]
        kattrs = ["nonlinear_power", "delta_k", "power", "transfer", "lnk",
                   "_lnP_0", "_lnP_cdm_0", "_lnT_cdm", "_unnormalised_lnP",
                   "_unnormalised_lnT"]
        for a in mattrs + kattrs:
            if a in required_attrs:
                attribute = a
                break

    h = MassFunction(**kwargs)

    for vals in final_list:
        h.update(**vals)
        if attribute in mattrs:
            getattr(h, attribute)
        elif attribute in kattrs:
            getattr(h.transfer, attribute)
        if get_label:
            if len(final_list) > 1:
                label = str(vals)
            elif kwargs:
                label = str(kwargs)
            else:
                label = h.mf_fit


            label = label.replace("{", "").replace("}", "").replace("'", "")
            label = label.replace("_", "").replace(": ", "").replace(", ", "_")
            label = label.replace("mffit", "").replace("transferfit", "").replace("delta_wrt", "").replace("\n", "")

            # The following lines transform the M and lnk parts
            while "[" in label:
                label = re.sub("[\[].*?[\]]", "", label)
            label = label.replace("array", "")
            label = label.replace("M()", "M(" + str(np.log10(h.M[0])) + ", " + str(np.log10(h.M[-1])) + ", " +
                          str(np.log10(h.M[1]) - np.log10(h.M[0])) + ")")
            label = label.replace("lnk()", "lnk(" + str(h.transfer.lnk[0]) + ", " + str(h.transfer.lnk[-1]) + ", " +
                          str(h.transfer.lnk[1] - h.transfer.lnk[0]) + ")")
            yield h, label
        else:
            yield h
