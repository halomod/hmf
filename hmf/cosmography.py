'''
Created on Jan 21, 2013

@author: Steven
'''
import numpy as np
import scipy.integrate as intg
#import cosmolopy.density as cd
#import cosmolopy.distance as cd
#import cosmolopy.constants as cc



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

def d_plus(z, omegam, omegak, omegav):
    """
    Finds the factor D+(a), from Lukic et. al. 2007, eq. 8.
    
    Uses romberg integration with a suitable step-size. 
    
    Input: z: redshift.
    
    Output: dplus: the factor.
    """

    stepsize = step_size(0.0000001, a_from_z(z))
    a_vector = np.arange(0.0000001, a_from_z(z), stepsize)
    integrand = 1.0 / (a_vector * hubble_z(z_from_a(a_vector), omegam, omegak, omegav)) ** 3

    integral = intg.romb(integrand, dx=stepsize)
    dplus = 5.0 * omegam * hubble_z(z, omegam, omegak, omegav) * integral / 2.0

    return dplus

def growth_factor(z, omegam, omegak, omegav):
    """
    Finds the factor d(a) = D+(a)/D+(a=1), from Lukic et. al. 2007, eq. 8.
    
    Input: z: redshift.
    
    Output: growth: the growth factor.
    """

    growth = d_plus(z, omegam, omegak, omegav) / d_plus(0.0, omegam, omegak, omegav)

    return growth

def step_size(mini, maxi):
    """
    Calculates a suitable step size for romberg integration given data limits
    """

    p = 13

    while (maxi - mini) / (2 ** p + 1) < 10 ** (-5):
        p = p - 1

    step_size = (maxi - mini) / (2 ** p + 1.0)

    return step_size
#
#def z_last_scattering(omega_b, omega_m, h):
#    """
#    Defines the redshift of the surface of last scattering via Hu and White 1997
#    """
#    ob = omega_b * h ** 2
#    om = omega_m * h ** 2
#    b1 = 0.0783 * ob ** -0.238 / (1 + 39.5 * ob ** 0.763)
#    b2 = 0.560 * (1 + 21.1 * ob ** 1.81) ** -1
#    return 1048 * (1 + 0.00124 * ob ** -0.738) * (1 + b1 * om ** b2)
#
#def r_s(omega_m, z_ls, h):
#    age = cd.age(z_ls, use_flat=True, omega_M_0=omega_m, omega_lambda_0=1 - omega_m, h=h)
#    return age * cc.c_light_Mpc_s
#
#def d_ls_dist(omega_m, z_ls, h):
#    ang_size = cd.angular_diameter_distance(z_ls, omega_M_0=omega_m, omega_lambda_0=1 - omega_m, omega_k_0=0, h=h)
#    return ang_size
#
#def theta(omega_m, omega_b, h):
#    z = z_last_scattering(omega_b, omega_m, h=h)
#
#    r = r_s(omega_m, z, h)
#    d = d_ls_dist(omega_m, z, h)
#
#    return r / d
#def d_ls(omega_m, h, z_ls):
#
#    om = omega_m * h ** 2
#    aeq = (4.17 * 10 ** -5 / om) * (2.7255 / 2.728) ** 4
#    a_ls = 1.0 / (1 + z_ls)
#
#    eta_0 = (2.0 / np.sqrt(omega_m * (100 * h) ** 2)) * (np.sqrt(1 + aeq) - np.sqrt(aeq)) * (1 - 0.0841 * np.log(omega_m))
#    eta_star = (2.0 / np.sqrt(omega_m * (100 * h) ** 2)) * (np.sqrt(astar + aeq) - np.sqrt(aeq))
#    return eta_0 - eta_star
#
#def ang_scale(omega_m, omega_b, h):
#    z_ls = z_last_scattering(omega_b, omega_m, h)
#
#    om = omega_m * h ** 2
#    aeq = (4.17 * 10 ** -5 / om) * (2.7255 / 2.728) ** 4
#    a_ls = 1.0 / (1 + z_ls)
#
#    eta_0 = (2.0 / np.sqrt(omega_m * (100 * h) ** 2)) * (np.sqrt(1 + aeq) - np.sqrt(aeq)) * (1 - 0.0841 * np.log(omega_m))
#    eta_star = (2.0 / np.sqrt(omega_m * (100 * h) ** 2)) * (np.sqrt(a_ls + aeq) - np.sqrt(aeq))
#
#    return np.pi * (eta_0 / eta_star - 1)

