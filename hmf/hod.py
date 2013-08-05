'''
Created on Aug 5, 2013

@author: Steven
'''
'''
A class containing the methods to do HOD modelling
'''
###############################################################################
# Some Imports
###############################################################################
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import scipy.integrate as intg
import numpy as np
import scipy.special as sp

from hmf import Perturbations
import tools
from profiles import profiles

#===============================================================================
# The class itself
#===============================================================================
class HOD(Perturbations):
    '''
    A class containing methods to do HOD modelling

    The reason this is not in the update-property style of Perturbations, is that 
    there is virtually nothing to incrementally update. Any change of cosmology
    affects the whole calculation virtually (except for the HOD n(M), which is
    trivial). Any change in HOD parameters DOES affect everything. There is almost
    no gain by doing it that way, and it is a lot more complicated, so we stick 
    with normal functions.
    
    INPUT PARAMETERS
        R:     The distances at which the dark matter correlation function is calculated in Mpc/h
                Default: np.linspace(1, 200, 200)

        M_min: scalar float
               the minimum mass of a halo containing a galaxy

        sigma: scalar float
               the width of the transition from 0 to 1 galaxies
               if None, is step-function.

        M_1 : scalar float
              the mean mass of a halo which contains 1 galaxy

        alpha: scalar float
               the slope of the satellite term

        M_0:    scalar float
                the minimum mass at which satellites can exist

        **kwargs: anything that can be used in the Perturbations class
    
    '''
    def __init__(self, M_1=10 ** 13.5, alpha=1.27, M_min=10 ** 12.3,
                 sigma=None, M_0=0, fca, fcb, fs, delta, x, HOD_model='zehavi',
                 profile='nfw', cm_relation='duffy', bias_model='tinker',
                 r=np.linspace(1, 200, 200), central=False, ** kwargs):

        #This gives the access to all the methods of Perturbations
        super(HOD, self).__init__(**kwargs)

        #correlation separations
        self.r = r

        #HOD parameters
        self.M_1 = M_1
        self.M_0 = M_0
        self.alpha = alpha
        self.M_min = M_min
        self.sigma = sigma
        self.fca = fca
        self.fcb = fcb
        self.fs = fs
        self.delta = delta
        self.x = x
        self.HOD_model = HOD_model
        self.central = central

        self.bias_model = bias_model
        self.profile = profiles(self.cosmo_params['mean_dens'], self.cosmo_params['delta_vir'], profile=profile, cm_relation=cm_relation)

    def n_tot(self):
        """
        Gets the function N(m), the number of galaxies in a halo of mass m.

        This is a wrapper around invidual methods which use different parameterisations
        """

        #Most parameterisations are the same with a few parameters added/set to 0/1. These are summed
        #up in the following call

        if self.HOD_model == 'geach':
            self.M_0 = 0
            self.x = 1

        elif self.HOD_model == 'contreras':
            self.M_0 = 0

        elif self.HOD_model == 'zehavi':
            self.x = 1
            self.fcb = 0
            self.fca = 0.5
            self.fs = 1
            self.delta = None

        elif self.HOD_model == 'zheng':
            self.x = 1
            self.fcb = 0
            self.fca = 0.5
            self.fs = 1
            self.delta = None
            self.sigma = None
            self.M_0 = 0

        if self.HOD_model in ['geach', 'contreras', 'zehavi', 'zheng']:
            n_c = self._n_cen()
            n_s = self._n_sat()

        else:
            raise ValueError("HOD model not implemented")

        if self.central:
            n_tot = n_c(1 + n_s)
        else:
            n_tot = n_c + n_s

        return n_tot

    def _n_cen(self):
        """
        Defines the central galaxy number
        """
        n_c = np.zeros_like(self.M)

        if self.sigma is None:
            n_c[self.M > self.M_min] = 1

        else:
            n_c = self.fcb * (1 - self.fca) * np.exp(np.log10(self.M / self.M_min) ** 2 / (2 * (self.x * self.sigma) ** 2)) + \
                  self.fca * (1 + sp.erf(np.log10(self.M / self.M_min) / (self.x * self.sigma)))

        return n_c


    def _n_sat(self):

        if self.delta is None:
            n_s = ((self.M - self.M_0) / self.M_1) ** self.alpha
        else:
            n_s = self.fs * (1 + sp.erf(np.log10(self.M / self.M_1) / self.delta)) * (self.M / self.M_1) ** self.alpha

        return n_s

    def mean_gal_den(self):
        """
        Gets the mean number density of galaxies
        """
        integrand = self.dndm * self.n_tot()

        #Now take the spline of the integrand and extrapolate to simulate integration
        # from 0 to infinity
        M = self.M[np.logical_not(np.isnan(self.dndm))]
        integrand = np.log(integrand[np.logical_not(np.isnan(self.dndm))])

        integ = spline(np.log10(M), integrand, k=1)

        M_new, dlogM = np.linspace(8, 18, 4097, retstep=True)
        integrand = integ(M_new)

        return intg.romb(np.exp(integrand) * 10 ** M_new, dx=dlogM)


    def galaxy_power(self):
        """
        Returns the overall two-term galaxy power spectrum
        """
        pg1h = self._power_1h()
        pg2h = self._power_2h()

        return pg1h + pg2h

    def _power_1h(self):
        """
        Returns the one-halo galaxy power spectrum
        """
        pg1h = np.zeros_like(self.lnk)

        #Restrict quantities in M to where they have real values
        m = self.M[np.logical_not(np.isnan(self.dndm))]
        dndm = self.dndm[np.logical_not(np.isnan(self.dndm))]
        n_c = self._n_cen()[np.logical_not(np.isnan(self.dndm))]
        n_s = self._n_sat()[np.logical_not(np.isnan(self.dndm))]
        M_new, dlogM = np.linspace(8, 18, 4097, retstep=True)

        for i, lnk in enumerate(self.lnk):
            if self.central:
                integrand = n_c * (2 * self.n_s * self.profile.u(np.exp(lnk), m, self.z) + \
                                   (n_s * self.profile.u(np.exp(lnk), m, self.z)) ** 2) * dndm
            else:
                integrand = (n_c * 2 * self.n_s * self.profile.u(np.exp(lnk), m, self.z) + \
                             (n_s * self.profile.u(np.exp(lnk), m, self.z)) ** 2) * dndm

            integ = spline(np.log10(m), integrand, k=1)


            integrand = integ(M_new)
            pg1h[i] = intg.romb(integrand * 10 ** M_new, dx=dlogM)

        return pg1h / self.mean_gal_den() ** 2

    def _power_2h(self):
        """
        Returns the two-halo galaxy power spectrum
        """
        pg2h = np.zeros_like(self.lnk)

        m = self.M[np.logical_not(np.isnan(self.dndm))]
        dndm = self.dndm[np.logical_not(np.isnan(self.dndm))]
        M_new, dlogM = np.linspace(8, 18, 4097, retstep=True)
        n_tot = self.n_tot()[np.logical_not(np.isnan(self.dndm))]
        bias = self.bias()[np.logical_not(np.isnan(self.dndm))]

        for i, lnk in enumerate(self.lnk):
            integrand = n_tot * bias * dndm * self.profile.u * (np.exp(lnk), m, self.z)
            integ = spline(np.log10(m), integrand, k=1)
            integrand = integ(M_new)

            pg2h[i] = intg.romb(integrand * 10 ** M_new, dx=dlogM)

        return pg2h ** 2 / self.mean_gal_den() ** 2

    def galaxy_corr(self):
        """
        The galaxy-galaxy correlation function
        """

        power = self.galaxy_power()
        corr = tools.power_to_corr(np.log(power), self.lnk, self.r)

        return corr


    def bias_large_scale(self):
        """
        Large scale bias b(M)
        """

        nu = (self.cosmo_params['delta_c'] / self.sigma) ** 2
        dc = self.cosmo_params['delta_c']
        if self.bias_model is 'seljak':
            return 1 + (nu - 1) / dc + 0.6 / (dc * (1 + (0.707 * nu) ** 0.3))
        elif self.bias_model is 'ma':
            return (1 + (nu - 1) / dc) * (1 / (2 * nu ** 2) + 1) ** (0.06 - 0.02 * self.cosmo_params['n'])
        elif self.bias_model is 'seljak_warren':
            siginv = spline(self.sigma, self.M)
            mstar = siginv(dc)
            x = self.M / mstar
            bias_ls = 0.53 + 0.39 * x ** 0.45 + 0.13 / (40 * x + 1) + 5E-4 * x ** 1.5

            #TODO: what the monkeys is alpha_s??
            return bias_ls + np.log10(x) * (0.4 * (self.cosmo_params['omegam'] - 0.3 + self.cosmo_params['n'] - 1) + 0.3 * (self.cosmo_params['sigma_8'] - 0.9 + self.cosmo_params['H0'] / 100 - 0.7) + 0.8)

        elif self.bias_model is 'tinker':
            a = 0.707
            sa = np.sqrt(a)
            b = 0.35
            c = 0.8

            return 1 + 1 / (sa * dc) * (sa * (a * nu) + sa * b * (a * nu) ** (1 - c) - (a * nu) ** c / ((a * nu) ** c + b * (1 - c) * (1 - c / 2)))

    def bias_scale_dep(self):
        """
        Scale-dependent bias
        """
        b = self.bias_large_scale()

        return b ** 2 * (1 + 1.17 * self.dm_correlation()) ** 1.49 / (1 + 0.69 * self.dm_correlation()) ** 2.09

    @property
    def dm_correlation(self):
        """
        The dark-matter only two-point correlation function of the given cosmology
        """

        return tools.power_to_corr(self.power, self.lnk, self.r)

