'''
Created on Jan 21, 2013

@author: Steven
'''
import numpy as np
import scipy.integrate as intg
import cosmolopy.density as cden
import cosmolopy.distance as cdist


def mean_dens(z, omegam, omegav, omegak, crit_dens):
    return omegam_z(z, omegam, omegav, omegak) * crit_dens

def a_from_z(z):
    """
    Calculates the scale factor at a given redshift
    
    Input: z: the redshift.
    Output a: the scale factor at the given redshift.
    """

    a = 1.0 / (1.0 + z)

    return a

def z_from_a(a):
    """
    Returns the redshift of a particular scale factor
    """

    z = 1.0 / a - 1.0

    return z

def hubble_z(z, omegam, omegak, omegav):
    """
    Finds the hubble parameter at z normalized by the current hubble constant: E(z).
    
    Source: Hogg 1999, Eq. 14
    
    Input: z: redshift
           omegam: matter density
           omegak: curvature density
           omegav: dark energy density
    Output: hubble: the hubble parameter at z divided by H_0 (ie. we don't need H_0)
    """

    hubble = np.sqrt(omegam * (1 + z) ** 3 + omegak * (1 + z) ** 2 + omegav)

    return hubble


def omegam_z(z, omegam, omegav, omegak):
    """
    Finds Omega_M as a function of redshift
    """
    top = omegam * (1 + z) ** 3
    bottom = omegam * (1 + z) ** 3 + omegav + omegak * (1 + z) ** 2

    return top / bottom

def d_plus(z, **cosmo):
    """
    Finds the factor D+(a), from Lukic et. al. 2007, eq. 8.
    
    Uses romberg integration with a suitable step-size. 
    
    Input: z: redshift.
    
    Output: dplus: the factor.
    """
    lna = np.linspace(np.log(1e-8), 0, 1000)
    z_vec = 1.0 / np.exp(lna) - 1.0

    integrand = 1.0 / (np.exp(lna) * cdist.e_z(z_vec, **cosmo)) ** 3

    integral = intg.simps(np.exp(lna) * integrand, dx=lna[1] - lna[0])
    dplus = 5.0 * cosmo["omega_M_0"] * cdist.e_z(z, **cosmo) * integral / 2.0

    return dplus

def growth_factor(z, **cosmo):
    """
    Finds the factor d(a) = D+(a)/D+(a=1), from Lukic et. al. 2007, eq. 8.
    
    Input: z: redshift.
    
    Output: growth: the growth factor.
    """

    growth = d_plus(z, **cosmo) / d_plus(0.0, **cosmo)

    return growth
