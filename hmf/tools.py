'''
Created on Apr 20, 2012

@author: Steven Murray
@contact: steven.murray@uwa.edu.au

Last Updated: 12 March 2013

This module contains 4 functions which all have the purpose of importing the
transfer function (or running CAMB on the fly) for
input to the Perturbations class (in Perturbations.py)

The functions are:
    1. SetParameter: modifies the params.ini file for use in CAMB with specified parameters
    2. ImportTransferFunction: Merely reads a CAMB transfer file and outputs ln(k) and ln(T).
    3. CAMB: runs CAMB through the OS (camb must be compiled previously)
    4. Setup: A driver for the above functions. Returns ln(k) and ln(T)
'''

###############################################################################
# Some simple imports
###############################################################################
import numpy as np
import scipy.integrate as intg
import pycamb
import collections
import cosmolopy.perturbation as pert

from scipy.interpolate import InterpolatedUnivariateSpline as spline

###############################################################################
# The function definitions
###############################################################################
def get_transfer(transfer_file, camb_dict, transfer_fit, k_bounds=None):
    """
    A convenience function used to fully setup the workspace in the 'usual' way
    
    We use either CAMB or the EH approximation to get the transfer function.
    The transfer function is in terms of the wavenumber IN UNITS OF h/Mpc!!
    """
    #If no transfer file uploaded, but it was custom, execute CAMB
    if transfer_file is None:
        if transfer_fit == "CAMB":
            k, T, sig8 = pycamb.transfers(**camb_dict)
            T = np.log(T[[0, 6], :, 0])
            del sig8, k
        elif transfer_fit == "EH":
            k = np.exp(np.linspace(np.log(k_bounds[0]), np.log(k_bounds[1]), 4097))
            #Since the function natively calculates the transfer based on k in Mpc^-1,
            # we need to multiply by h.
            t, T = pert.transfer_function_EH(k * camb_dict['H0'] / 100,
                                             omega_M_0=(camb_dict['omegac'] + camb_dict['omegab']), omega_lambda_0=camb_dict['omegav'],
                                             h=camb_dict["H0"] / 100, omega_b_0=camb_dict['omegab'], omega_n_0=camb_dict['omegan'],
                                             N_nu=camb_dict['Num_Nu_massive'])
            T = np.vstack((np.log(k), np.log(T)))
            del t
    else:
        #Import the transfer file wherever it is.
        T = read_transfer(transfer_file)

    return T

def read_transfer(transfer_file):
    """
    Imports the Transfer Function file to be analysed, and returns the pair ln(k), ln(T)
    
    Input: "transfer_file": full path to the file containing the transfer function (from camb).
    
    Output: ln(k), ln(T)
    """

    transfer = np.loadtxt(transfer_file)
    T = np.log(transfer[:, [0, 6]]).T

    return T

def check_kr(min_m, max_m, mean_dens, mink, maxk):

    #Define mass from radius function
    def M(r):
        return 4 * np.pi * r ** 3 * mean_dens / 3

    #Define min and max radius
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
    Interpolates the given Transfer function and transforms it into a Power Spectrum.
    
    Input: k        : an array of values of lnk
           Transfer : an array of values of lnT
           step_size: the step_size between values of lnk required in interpolation.
              
    Notes: the power spectrum is calculated using P(k) =  A.k^n.T^2, with A=1 in this method
            (which leaves the power spectrum un-normalized). The interpolation is done using
            cubic splines.
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
    
    Input: norm_sigma_8: The sigma_8 value to normalize to.
    """
    # Calculate the value of sigma_8 without prior normalization.


    sigma_8 = mass_variance(4.*np.pi * 8 ** 3 * mean_dens / 3., unn_power, lnk, mean_dens)[0]

    # Calculate the normalization factor
    normalization = norm_sigma_8 / sigma_8

    # Normalize the previously calculated power spectrum.
    power = 2 * np.log(normalization) + unn_power
    return power

def mass_variance(M, power, lnk, mean_dens):
    """
    Finds the Mass Variance of M using the top-hat window function.
    
    Input: M: the radius(mass) of the top-hat function (vector).
    Output: sigma: the mass variance.
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
        sigma[i] = (0.5 / np.pi ** 2) * intg.simps(integ, dx=dlnk, even='first')

    return np.sqrt(sigma)

def top_hat_window(M, lnk, mean_dens):
    """
    Constructs the window function squared in Fourier space for given radii
    
    Input: R: The radius of the top-hat function
    Output: W_squared: The square of the top-hat window function in Fourier space.
    """

    # Calculate the factor kR, minding to un-log k before use.
    kR = np.exp(lnk) * mass_to_radius(M, mean_dens)
    W_squared = (3 * (np.sin(kR) / kR ** 3 - np.cos(kR) / kR ** 2)) ** 2
    W_squared[kR < 0.01] = 1.0

    return W_squared


def mass_to_radius(M, mean_dens):
    """
    Calculates radius from mass given mean density.
    """
    return (3.*M / (4.*np.pi * mean_dens)) ** (1. / 3.)

def radius_to_mass(R, mean_dens):
    """
    Calculates mass from radius given mean density
    """
    return 4 * np.pi * R ** 3 * mean_dens / 3
def wdm_transfer(m_x, power_cdm, lnk, H0, omegac):
    """
    Tansforms the CDM Power Spectrum into a WDM power spectrum for a given warm particle mass m_x.
    
    NOTE: formula from Bode et. al. 2001 eq. A9
    """

    h = H0 / 100
    g_x = 1.5
    nu = 1.12

    alpha = 0.049 * (omegac / 0.25) ** 0.11 * (h / 0.7) ** 1.22 * (1 / m_x) ** 1.11 * (1.5 / g_x) ** 0.29

    transfer = (1 + (alpha * np.exp(lnk)) ** (2 * nu)) ** -(5.0 / nu)

    return power_cdm + 2 * np.log(transfer)

def dlnsdlnm(M, sigma, power, lnk, mean_dens):
    dlnk = lnk[1] - lnk[0]
    R = mass_to_radius(M, mean_dens)
    dlnsdlnM = np.zeros_like(M)
    for i, r in enumerate(R):
        g = np.exp(lnk) * r
        w = dw2dm(g)  # Derivative of W^2
        integ = w * np.exp(power - lnk)
        dlnsdlnM[i] = (3.0 / (2.0 * sigma[i] ** 2 * np.pi ** 2 * r ** 4)) * intg.romb(integ, dx=dlnk)
    return dlnsdlnM

def dw2dm(kR):
    """
    The derivative of the top-hat window function squared
    """
    return (np.sin(kR) - kR * np.cos(kR)) * (np.sin(kR) * (1 - 3.0 / (kR ** 2)) + 3.0 * np.cos(kR) / kR)

def n_eff(dlnsdlnm):
    """
    Calculates the power spectral slope at the scale of the halo radius, using eq. 42 in Lukic et. al 2007.
    """

    n_eff = -3.0 * (2.0 * dlnsdlnm + 1.0)

    return n_eff

def new_k_grid(k, k_bounds=None):
    """
    Creates a new grid for the transfer function, for application of Romberg integration.
    
    Note: for Romberg integration, the number of steps must be 2**p+1 where p is an integer, which is why this scaling
            should be performed. We choose 4097 bins for ln(k). This could possibly be optimized or made variable.
    """

    # Determine the true k_bounds.
    min_k = np.log(k_bounds[0])
    max_k = np.log(k_bounds[1])

    # Setup the grid and fetch the grid-spacing as well
    k, dlnk = np.linspace(min_k, max_k, 4097, retstep=True)

    return k, dlnk

def power_to_corr(lnP, lnk, R):
    """
    Calculates the correlation function given a power spectrum
    
    NOTE: no check is done to make sure k spans [0,Infinity] - make sure of this before you enter the arguments.
    
    INPUT
        lnP: vector of values for the log power spectrum
        lnk: vector of values (same length as lnP) giving the log wavenumbers for the power (EQUALLY SPACED)
        r:   radi(us)(i) at which to calculate the correlation
    """
    import matplotlib.pyplot as plt
    k = np.exp(lnk)
    P = np.exp(lnP)

    if not isinstance(R, collections.Iterable):
        R = [R]

    corr = np.zeros_like(R)

    #We must do better than using just 4097 logarithmic bins for k, since
    #we are integrating over a fast-oscillating function if r is big.
    integrand_part = spline(lnk, P * k ** 2, k=1)
    def integ(r, lnk):
        return integrand_part(lnk) * np.sin(np.exp(lnk) * r) / r

    for i, r in enumerate(R):
        plt.plot(np.exp(lnk), integ(r, lnk))
        plt.xscale('log')
        plt.savefig("/Users/Steven/Documents/dm_corr_plots/" + str(r) + ".png")
        plt.clf()
        corr[i] = (0.5 / np.pi ** 2) * intg.quad(lambda k: integ(r, k), -10, 10, epsabs=0, epsrel=10 ** -2, limit=2000)[0]

    return corr

