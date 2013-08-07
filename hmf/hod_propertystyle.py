#'''
#Created on Jul 15, 2013
#
#@author: Steven
#'''
#import numpy as np
#import scipy.special as sp
#from hmf import Perturbations
#import tools
#from scipy.interpolate import InterpolatedUnivariateSpline as spline
#import scipy.integrate as intg
#
#class HOD(Perturbations):
#    """
#    A class to add HOD methods onto the existing Perturbations object
#
#
#    INPUT PARAMETERS
#        R:             The distances at which the dark matter correlation function is calculated in Mpc/h
#                       Default: np.linspace(1, 200, 200)
#
#        M_min: scalar float
#               the minimum mass of a halo containing a galaxy
#
#        sigma: scalar float
#               the width of the transition from 0 to 1 galaxies
#               if None, is step-function.
#
#        M_1 : scalar float
#              the mean mass of a halo which contains 1 galaxy
#
#        alpha: scalar float
#               the slope of the satellite term
#
#        M_0:scalar float
#            the minimum mass at which satellites can exist
#
#        **kwargs: anything that can be used in the Perturbations class
#
#    """
#
#    def __init__(self, M_1=10 ** 13.5, alpha=1.27, M_min=10 ** 12.3,
#                 sigma=None, M_0=0, fca, fcb, fs, delta, x, HOD_model='zehavi',
#                 profile='nfw', cm_relation='duffy', bias_model='tinker',
#                 r=np.linspace(1, 200, 200), ** kwargs):
#        """
#        NOTE: names of input arguments MUST NOT CONFLICT with Perturbations names
#        """
#        super(HOD, self).__init__(**kwargs)
#
#        self.update(M_min=M_min, sigma=sigma, M_1=M_1, alpha=alpha, M_0=M_0, fca=fca, fcb=fcb, fs=fs, delta=delta,
#                    x=x, HOD_model=HOD_model, profile=profile, cm_relation=cm_relation, bias_model=bias_model,
#                    r=r, central=True)
#
#    def update(self, **kwargs):
#        """
#        Over-rides the base Perturbations update() method with a couple of extra bits and pieces.
#        """
#
#        super(HOD, self).update(**kwargs)
#
#        for arg in kwargs:
#            if arg is "M_min":
#                self.M_min = kwargs[arg]
#            elif arg is "sigma":
#                self.sigma = kwargs[arg]
#            elif arg is "M_1":
#                self.M_1 = kwargs[arg]
#            elif arg is "alpha":
#                self.alpha = kwargs[arg]
#            elif arg is "M_0":
#                self.M_0 = kwargs[arg]
#            elif arg is "fca":
#                self.fca = kwargs[arg]
#            elif arg is 'fcb':
#                self.fcb = kwargs[arg]
#            elif arg is 'fs':
#                self.fs = kwargs[arg]
#            elif arg is 'delta':
#                self.delta = kwargs[arg]
#            elif arg is 'x':
#                self.x = kwargs[arg]
#            elif arg is 'HOD_model':
#                self.HOD_model = kwargs[arg]
#            elif arg is 'profile':
#                self.profile = kwargs[arg]
#            elif arg is 'cm_relation':
#                self.cm_relation = kwargs[arg]
#            elif arg is 'bias_model':
#                self.bias_model = kwargs[arg]
#            elif arg is 'r':
#                self.r = kwargs[arg]
#            elif arg is 'central':
#                self.central = kwargs[arg]
#
#    @property
#    def r(self):
#        return self.__r
#
#    @r.setter
#    def r(self, val):
#        try:
#            if len(val) == 1:
#                raise ValueError("R must be a sequence of length > 1")
#        except TypeError:
#            raise TypeError("R must be a sequence of length > 1")
#
#        #Delete stuff dependent on it
#        del self.dm_correlation
#
#        self.__r = val
#
#    @property
#    def M_min(self):
#        return self.__M_min
#
#    @M_min.setter
#    def M_min(self, val):
#        self.__M_min = val
#        del self._n_cen
#
#    @property
#    def sigma(self):
#        return self.__sigma
#
#    @sigma.setter
#    def sigma(self, val):
#        self.__sigma = val
#        del self._n_cen
#
#    @property
#    def M_1(self):
#        return self.__M_1
#
#    @M_1.setter
#    def M_1(self, val):
#        self.__M_1 = val
#        del self._n_sat
#
#    @property
#    def alpha(self):
#        return self.__alpha
#
#    @alpha.setter
#    def alpha(self, val):
#        self.__alpha = val
#        del self._n_sat
#
#    @property
#    def M_0(self):
#        return self.__M_0
#
#    @M_0.setter
#    def M_0(self, val):
#        self.__M_0 = val
#        del self._n_sat
#
#    @property
#    def fca(self):
#        return self.__fca
#
#    @fca.setter
#    def fca(self, val):
#        self.__fca = val
#        del self._n_cen
#
#    @property
#    def fcb(self):
#        return self.__fcb
#
#    @fcb.setter
#    def fcb(self, val):
#        self.__fcb = val
#        del self._n_cen
#
#    @property
#    def fs(self):
#        return self.__fs
#
#    @fs.setter
#    def fs(self, val):
#        self.__fs = val
#        del self._n_sat
#
#    @property
#    def delta(self):
#        return self.__delta
#
#    @delta.setter
#    def delta(self, val):
#        self.__delta = val
#        del self._n_sat
#
#    @property
#    def x(self):
#        return self.__x
#
#    @x.setter
#    def x(self, val):
#        self.__x = val
#        del self._n_cen
#
#    @property
#    def HOD_model(self):
#        return self.__HOD_model
#
#    @HOD_model.setter
#    def HOD_model(self, val):
#        self.__HOD_model = val
#
#        if self.HOD_model == 'geach':
#            self.M_0 = 0
#            self.x = 1
#
#        elif self.HOD_model == 'contreras':
#            self.M_0 = 0
#
#        elif self.HOD_model == 'zehavi':
#            self.x = 1
#            self.fcb = 0
#            self.fca = 0.5
#            self.fs = 1
#            self.delta = None
#
#        elif self.HOD_model == 'zheng':
#            self.x = 1
#            self.fcb = 0
#            self.fca = 0.5
#            self.fs = 1
#            self.delta = None
#            self.sigma = None
#            self.M_0 = 0
#
#        del self._n_sat
#        del self._n_cen
#
#    @property
#    def cm_relation(self):
#        return self.__cm_relation
#
#    @cm_relation.setter
#    def cm_relation(self, val):
#        self.__cm_relation = val
#
#    @property
#    def bias_model(self):
#        return self.__bias_model
#
#    @bias_model.setter
#    def bias_model(self, val):
#        self.__bias_model = val
#
#    @property
#    def profile(self):
#        return self.__profile
#
#    @profile.setter
#    def profile(self, val):
#        self.__profile = val
#
#    @property
#    def central(self):
#        return self.__central
#
#    @central.setter
#    def central(self, val):
#        self.__central = val
#
##--------------------------------  START NON-SET PROPERTIES ----------------------------------------------
##--------------------------------  NUMBER DENSITIES ----------------------------------------------
#    @property
#    def _n_cen(self):
#        """
#        The number of galaxies in the centre of a halo (0-1)
#        """
#        try:
#            return self.__n_cen
#        except:
#            if self.sigma is not None:
#                self.__n_cen = self.fcb * (1 - self.fca) * np.exp(np.log10(self.M / self.M_min) ** 2 / (2 * (self.x * self.sigma) ** 2)) + self.fca * (1 + sp.erf((np.log10(self.M) - np.log10(self.M_min)) / self.sigma))
#            else:
#                self.__n_cen = np.zeros_like(self.M)
#                self.__n_cen[self.M > self.M_min] = 1
#
#            return self.__n_cen
#
#    @_n_cen.deleter
#    def _n_cen(self):
#        try:
#            del self.__n_cen
#            del self.n_of_m
#        except:
#            pass
#
#    @property
#    def _n_sat(self):
#        """
#        The number of satellite galaxies in a halo
#        """
#        try:
#            return self.__n_sat
#        except:
#            if self.delta is None:
#                self.__n_sat = ((self.M - self.M_0) / self.M_1) ** self.alpha
#            else:
#                self.__n_sat = self.fs * (1 + sp.erf(np.log10(self.M / self.M_1) / self.delta)) * (self.M / self.M_1) ** self.alpha
#
#            return self.__n_sat
#
#    @_n_sat.deleter
#    def _n_sat(self):
#        try:
#            del self.__n_sat
#            del self.n_of_m
#        except:
#            pass
#
#    @property
#    def n_of_m(self):
#        """
#        Calculates the mean number of galaxies in a halo of mass M
#
#        """
#        try:
#            return self.__n_of_m
#        except:
#            self.__n_of_m = self._n_cen + self._n_sat
#            return self.__n_of_m
#
#    @n_of_m.deleter
#    def n_of_m(self):
#        try:
#            del self.__n_of_m
#        except:
#            pass
#
#
#
#    #--------------------------------  CORRELATION FUNCTIONS ----------------------------------------------
#    @property
#    def _power_1h(self):
#        """
#        The 1-halo term of the correlation function with one central galaxy and one satellite
#        """
#        try:
#            return self.__power_1h
#        except:
#            self.__power_1h = np.zeros_like(self.lnk)
#            m = self.M[np.logical_not(np.isnan(self.dndm))]
#            dndm = self.dndm[np.logical_not(np.isnan(self.dndm))]
#            n_c = self._n_cen[np.logical_not(np.isnan(self.dndm))]
#            n_s = self._n_sat[np.logical_not(np.isnan(self.dndm))]
#
#            M_new, dlogM = np.linspace(8, 18, 4097, retstep=True)
#
#            for i, lnk in enumerate(self.lnk):
#                if self.central:
#                    integrand = n_c * (2 * n_s * self.profile.u(np.exp(lnk), m, self.z) + (n_s * self.profile.u(np.exp(lnk), m, self.z)) ** 2) * dndm
#                else:
#                    integrand = (n_c * 2 * n_s * self.profile.u(np.exp(lnk), m, self.z) + (n_s * self.profile.u(np.exp(lnk), m, self.z)) ** 2) * dndm
#
#                integ = spline(np.log10(m), integrand, k=1)
#
#
#                integrand = integ(M_new)
#                self.__power_1h[i] = intg.romb(integrand * 10 ** M_new, dx=dlogM) / self.mean_gal_den() ** 2
#
#            return self.__power_1h
#
#
#    @_power_1h.deleter
#    def _corr_1_cs(self):
#        try:
#            del self.__power_1h
#            del self.galaxy_power
#        except:
#            pass
#
##    @property
##    def _power_1_ss(self):
##        """
##        The power spectrum of 1-halo term with satellite galaxy pairs
##        """
##        try:
##            return self.__power_1_ss
##        except:
##            self.__power_1_ss = np.zeros_like(self.lnk)
##
##            # set M and mass_function within computed range
##            M = self.M[np.logical_not(np.isnan(self.dndm))]
##            galaxy_function = self.dndm * self._n_cen * self._n_sat ** 2
##            galaxy_function = np.log(galaxy_function[np.logical_not(np.isnan(self.dndm))])
##
##            # Define max_M as either 18 or the maximum set by user
##            max_M = np.log10(np.max([10 ** 18, M[-1]]))
##
##            # Interpolate the galaxy_function - this is in log-log space.
##            gf = spline(np.log(M), galaxy_function, k=1)
##            M_new, dlogM = np.linspace(8, max_M, 4097, retstep=True)
##            semi_integrand = gf(M_new) * 10 ** M_new * np.log(10)
##
##            for i, k in enumerate(self.lnk):
##                integrand = semi_integrand * self.rho_ft(k, M_new) ** 2
##                self.__power_1_ss[i] = intg.romb(integrand, dx=dlogM) / (2 * n_g ** 2)
##
##            return self.__power_1_ss
#
##    @_power_1_ss.deleter
##    def _power_1_ss(self):
##        try:
##            del self.__power_1_ss
##            del self._corr_1
##        except:
##            pass
##
##    @property
##    def _corr_1_ss(self):
##        """
##        The correlation function of 1-halo term satallite pairs
##        """
##        try:
##            return self.__corr_1_ss
##        except:
##
##            self.__corr_1_ss = np.zeros_like(self.R)
##            semi_integrand = self._power_1_ss * np.exp(self.lnk) ** 2
##            for i, r in enumerate(self.R):
##                integrand = semi_integrand * np.sin(np.exp(self.lnk) * r) / r
##                self.__corr_1_ss[i] = intg.romb(integrand, self.lnk[1] - self.lnk[0]) / (2 * np.pi ** 2)
##
##            return self.__corr_1_ss
##
##    @_corr_1_ss.deleter
##    def _corr_1_ss(self):
##        try:
##            del self.__cor_1_ss
##            del self._corr_1
##        except:
##            pass
##
##    @property
##    def _corr_1(self):
##        """
##        The 1-halo correlation term
##        """
##        try:
##            return self.__corr_1
##        except:
##            self.__corr_1 = self._corr_1_cs + self._corr_1_ss - 1
#
#    @property
#    def _power_2h(self):
#        """
#        The power spectrum for the 2-halo term
#        """
#        try:
#            return self.__power_2h
#        except:
#
#            self.__power_2h = np.zeros_like(self.lnk)
#
#            m = self.M[np.logical_not(np.isnan(self.dndm))]
#            dndm = self.dndm[np.logical_not(np.isnan(self.dndm))]
#            M_new, dlogM = np.linspace(8, 18, 4097, retstep=True)
#            n_tot = self.n_of_m[np.logical_not(np.isnan(self.dndm))]
#            bias = self.bias[np.logical_not(np.isnan(self.dndm))]
#
#            for i, lnk in enumerate(self.lnk):
#                integrand = n_tot * bias * dndm * self.profile.u * (np.exp(lnk), m, self.z)
#                integ = spline(np.log10(m), integrand, k=1)
#                integrand = integ(M_new)
#
#                self.__power_2h[i] = intg.romb(integrand * 10 ** M_new, dx=dlogM) ** 2 / self.mean_gal_den() ** 2
#
#            return self.__power_2h
#
#    @_power_2h.deleter
#    def _power_2h(self):
#        try:
#            del self.__power_2h
#            del self.galaxy_power
#        except:
#            pass
#
#    @property
#    def galaxy_power(self):
#        try:
#            return self.__galaxy_power
#        except:
#            self.__galaxy_power = self._power_1h + self._power_2h
#            return self.__galaxy_power
#
#    @galaxy_power.deleter
#    def galaxy_power(self):
#        try:
#            del self.__galaxy_power
#            del self.galaxy_corr
#        except:
#            pass
#
#    @property
#    def galaxy_corr(self):
#        try:
#            return self.__galaxy_corr
#        except:
#            self.__galaxy_corr = tools.power_to_corr(np.log(self.galaxy_power), self.lnk, self.r)
#            return self.__galaxy_corr
#
#    @galaxy_corr.deleter
#    def galaxy_corr(self):
#        try:
#            del self.__galaxy_corr
#        except:
#            pass
#
#    @property
#    def bias_ls(self):
#        """
#        Large scale bias b(M)
#        """
#        try:
#            return self.__bias_ls
#        except:
#            nu = (self.cosmo_params['delta_c'] / self.sigma) ** 2
#            dc = self.cosmo_params['delta_c']
#            if self.bias_model is 'seljak':
#                self.__bias_ls = 1 + (nu - 1) / dc + 0.6 / (dc * (1 + (0.707 * nu) ** 0.3))
#            elif self.bias_model is 'ma':
#                self.__bias_ls = (1 + (nu - 1) / dc) * (1 / (2 * nu ** 2) + 1) ** (0.06 - 0.02 * self.cosmo_params['n'])
#            elif self.bias_model is 'seljak_warren':
#                siginv = spline(self.sigma, self.M)
#                mstar = siginv(dc)
#                x = self.M / mstar
#                self.__bias_ls = 0.53 + 0.39 * x ** 0.45 + 0.13 / (40 * x + 1) + 5E-4 * x ** 1.5
#
#                #TODO: what the monkeys is alpha_s??
#                self.__bias_ls = self.__bias_ls + np.log10(x) * (0.4 * (self.cosmo_params['omegam'] - 0.3 + self.cosmo_params['n'] - 1) + 0.3 * (self.cosmo_params['sigma_8'] - 0.9 + self.cosmo_params['H0'] / 100 - 0.7) + 0.8)
#
#            elif self.bias_model is 'tinker':
#                a = 0.707
#                sa = np.sqrt(a)
#                b = 0.35
#                c = 0.8
#
#                self.__bias_ls = 1 + 1 / (sa * dc) * (sa * (a * nu) + sa * b * (a * nu) ** (1 - c) - (a * nu) ** c / ((a * nu) ** c + b * (1 - c) * (1 - c / 2)))
#
#            return self.__bias_ls
#
#    @bias_ls.deleter
#    def bias_ls(self):
#        try:
#            del self.__bias_ls
#            del self.bias_sd
#        except:
#            pass
#
#
#    def bias_sd(self):
#        """
#        Scale-dependent bias
#        """
#
#        b = self.bias()
#
#        return b ** 2 * (1 + 1.17 * self.dm_correlation) ** 1.49 / (1 + 0.69 * self.dm_correlation) ** 2.09
#
#    @property
#    def dm_correlation(self):
#        """
#        The dark-matter only two-point correlation function of the given cosmology
#        """
#        try:
#            return self.__dm_correlation
#        except:
#            self.__dm_correlation = tools.power_to_corr(self.power, self.lnk, self.R)
#            return self.__dm_correlation
#
#    @dm_correlation.deleter
#    def dm_correlation(self):
#        try:
#            del self.__dm_correlation
#        except:
#            pass
