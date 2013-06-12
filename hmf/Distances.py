'''
Created on Jan 21, 2013

@author: Steven
'''
import numpy as np
import scipy.integrate as intg
#import cosmolopy.density as cd
import cosmolopy.distance as cd
import cosmolopy.constants as cc
class Distances(object):
    """
    A Class to calculate distances and other cosmological quantities.
    
    The cosmology is input to the init function (at the moment only a flat cosmology)
    """

    def __init__(self, **kwargs):

        self.omega_lambda = kwargs.pop('omegav', 0.7)
        self.omega_cdm = kwargs.pop('omegac', 0.25)
        self.omega_b = kwargs.pop('omegab', 0.05)
        self.omega_mass = kwargs.pop('omegam', self.omega_cdm + self.omega_b)
        self.omega_k = kwargs.pop('omegak', 1 - self.omega_lambda - self.omega_mass)
        self.crit_dens = 27.755 * 10 ** 10
        self.D_h = 9.26 * 10 ** 25  #h^{-1}m Hogg eq.4

    def MeanDens(self, z):
        return self.Omega_M(z) * self.crit_dens

    def ScaleFactor(self, z):
        """
        Calculates the scale factor at a given redshift
        
        Input: z: the redshift.
        Output a: the scale factor at the given redshift.
        """

        a = 1.0 / (1.0 + z)

        return a

    def zofa(self, a):
        """
        Returns the redshift of a particular scale factor
        """

        z = 1.0 / a - 1.0

        return z

    def HubbleFunction(self, z):
        """
        Finds the hubble parameter at z normalized by the current hubble constant: E(z).
        
        Source: Hogg 1999, Eq. 14
        
        Input: z: redshift
        Output: hubble: the hubble parameter at z divided by H_0 (ie. we don't need H_0)
        """

        hubble = np.sqrt(self.omega_mass * (1 + z) ** 3 + self.omega_k * (1 + z) ** 2 + self.omega_lambda)

        return hubble


    def Omega_M(self, z):
        """
        Finds Omega_M as a function of redshift
        """
        top = self.omega_mass * (1 + z) ** 3
        bottom = self.omega_mass * (1 + z) ** 3 + self.omega_lambda + self.omega_k * (1 + z) ** 2

        return top / bottom

    def Dplus(self, z):
        """
        Finds the factor D+(a), from Lukic et. al. 2007, eq. 8.
        
        Uses romberg integration with a suitable step-size. 
        
        Input: z: redshift.
        
        Output: dplus: the factor.
        """

        step_size = self.StepSize(0.0000001, self.ScaleFactor(z))
        a_vector = np.arange(0.0000001, self.ScaleFactor(z), step_size)
        integrand = 1.0 / (a_vector * self.HubbleFunction(self.zofa(a_vector))) ** 3

        integral = intg.romb(integrand, dx=step_size)
        dplus = 5.0 * self.omega_mass * self.HubbleFunction(z) * integral / 2.0

        return dplus

    def GrowthFactor(self, z):
        """
        Finds the factor d(a) = D+(a)/D+(a=1), from Lukic et. al. 2007, eq. 8.
        
        Input: z: redshift.
        
        Output: growth: the growth factor.
        """

        growth = self.Dplus(z) / self.Dplus(0.0)

        return growth

    def StepSize(self, mini, maxi):
        """
        Calculates a suitable step size for romberg integration given data limits
        """

        p = 13

        while (maxi - mini) / (2 ** p + 1) < 10 ** (-5):
            p = p - 1

        step_size = (maxi - mini) / (2 ** p + 1.0)

        return step_size

def z_last_scattering(omega_b, omega_m, h):
    """
    Defines the redshift of the surface of last scattering via Hu and White 1997
    """
    ob = omega_b * h ** 2
    om = omega_m * h ** 2
    b1 = 0.0783 * ob ** -0.238 / (1 + 39.5 * ob ** 0.763)
    b2 = 0.560 * (1 + 21.1 * ob ** 1.81) ** -1
    return 1048 * (1 + 0.00124 * ob ** -0.738) * (1 + b1 * om ** b2)

def r_s(omega_m, z_ls, h):
    age = cd.age(z_ls, use_flat=True, omega_M_0=omega_m, omega_lambda_0=1 - omega_m, h=h)
    return age * cc.c_light_Mpc_s

def d_ls_dist(omega_m, z_ls, h):
    ang_size = cd.angular_diameter_distance(z_ls, omega_M_0=omega_m, omega_lambda_0=1 - omega_m, omega_k_0=0, h=h)
    return ang_size

def theta(omega_m, omega_b, h):
    z = z_last_scattering(omega_b, omega_m, h=h)

    r = r_s(omega_m, z, h)
    d = d_ls_dist(omega_m, z, h)

    return r / d
def d_ls(omega_m, h, z_ls):

    om = omega_m * h ** 2
    aeq = (4.17 * 10 ** -5 / om) * (2.7255 / 2.728) ** 4
    a_ls = 1.0 / (1 + z_ls)

    eta_0 = (2.0 / np.sqrt(omega_m * (100 * h) ** 2)) * (np.sqrt(1 + aeq) - np.sqrt(aeq)) * (1 - 0.0841 * np.log(omega_m))
    eta_star = (2.0 / np.sqrt(omega_m * (100 * h) ** 2)) * (np.sqrt(astar + aeq) - np.sqrt(aeq))
    return eta_0 - eta_star

def ang_scale(omega_m, omega_b, h):
    z_ls = z_last_scattering(omega_b, omega_m, h)

    om = omega_m * h ** 2
    aeq = (4.17 * 10 ** -5 / om) * (2.7255 / 2.728) ** 4
    a_ls = 1.0 / (1 + z_ls)

    eta_0 = (2.0 / np.sqrt(omega_m * (100 * h) ** 2)) * (np.sqrt(1 + aeq) - np.sqrt(aeq)) * (1 - 0.0841 * np.log(omega_m))
    eta_star = (2.0 / np.sqrt(omega_m * (100 * h) ** 2)) * (np.sqrt(a_ls + aeq) - np.sqrt(aeq))

    return np.pi * (eta_0 / eta_star - 1)

