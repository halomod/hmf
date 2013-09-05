'''
Created on Aug 29, 2013

@author: Steven
'''
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import cosmography

class fits(object):
    def __init__(self, M, n_eff, mf_fit, sigma, delta_c, z, delta_halo, cosmo_params, user_fit=None, cut_fit=True):
        self.sigma = sigma
        self.delta_c = delta_c
        self.z = z
        self.delta_halo = delta_halo
        self.lnsigma = np.log(1.0 / sigma)
        self.user_fit = user_fit
        self.mf_fit = mf_fit
        self.cut_fit = cut_fit
        self.n_eff = n_eff
        self.M = M
        self.cosmo_params = cosmo_params

        self.mf_fits = {
            "PS":self._nufnu_PS,
            "ST":self._nufnu_ST,
            "Warren":self._nufnu_Warren,
            "Jenkins":self._nufnu_Jenkins,
            "Reed03":self._nufnu_Reed03,
            "Reed07":self._nufnu_Reed07,
            "Angulo":self._nufnu_Angulo,
            "Angulo_Bound":self._nufnu_Angulo_Bound,
            "Tinker":self._nufnu_Tinker,
            "Watson_FoF":self._nufnu_Watson_FoF,
            "Watson":self._nufnu_Watson,
            "Crocce":self._nufnu_Crocce,
            "Courtin":self._nufnu_Courtin,
            "Bhattacharya": self._nufnu_Bhattacharya,
            "Behroozi": self._nufnu_Tinker,
            "user_model":self._nufnu_user_model
            }

    def nufnu(self):
        """
        Merely chooses the correct function from those available
        """
        return self.mf_fits[self.mf_fit]

    def _nufnu_PS(self):
        """
        Computes the function nu*f(nu) for the Press-Schechter approach at a given radius.
        
        Input R: radius of the top-hat function
        Output: f_of_nu: the function nu*f(nu) for the PS approach.
        """

        vfv = np.sqrt(2.0 / np.pi) * (self.delta_c / self.sigma) * np.exp(-0.5 * (self.delta_c / self.sigma) ** 2)

        return vfv

    def _nufnu_ST(self):
        """
        Finds the Sheth Tormen vf(v) 
        
        Input R: radius of the top-hat function
        Output: vfv: the Sheth-Tormen mass function fit.
        """

        nu = self.delta_c / self.sigma
        a = 0.707

        vfv = 0.3222 * np.sqrt(2.0 * a / np.pi) * nu * np.exp(-(a * nu ** 2) / 2.0) * (1 + (1.0 / (a * nu ** 2)) ** 0.3)

        return vfv

    def _nufnu_Jenkins(self):
        """
        Finds the Jenkins empirical vf(v) 
        
        Output: vfv: the Jenkins mass function fit.
        """

        vfv = 0.315 * np.exp(-np.abs(self.lnsigma + 0.61) ** 3.8)
        # Conditional on sigma range.
        if self.cut_fit:
            vfv[np.logical_or(self.lnsigma < -1.2, self.lnsigma > 1.05)] = np.NaN

        return vfv

    def _nufnu_Warren(self):
        """
        Finds the Warren empirical vf(v) 
        
        Input R: radius of the top-hat function
        Output: vfv: the Warren mass function fit.
        """

        vfv = 0.7234 * ((1.0 / self.sigma) ** 1.625 + 0.2538) * np.exp(-1.1982 / self.sigma ** 2)
        print len(self.M)
        print len(self.sigma)
        if self.cut_fit:
            vfv[np.logical_or(self.M < 10 ** 10, self.M > 10 ** 15)] = np.NaN
        return vfv

    def _nufnu_Reed03(self):
        """
        Finds the Reed 2003 empirical vf(v) 
        
        Input R: radius of the top-hat function
        Output: vfv: the Reed 2003 mass function fit.
        
        NOTE: Only valid from -1.7 < ln sigma^-1 < 0.9
        """

        ST_Fit = self._nufnu_ST()

        vfv = ST_Fit * np.exp(-0.7 / (self.sigma * np.cosh(2.0 * self.sigma) ** 5))

        if self.cut_fit:
            vfv[np.logical_or(self.lnsigma < -1.7, self.lnsigma > 0.9)] = np.NaN
        return vfv

    def _nufnu_Reed07(self):
        """
        Finds the Reed 2007 empirical vf(v) 
        
        Input R: radius of the top-hat function
        Output: vfv: the Reed 2003 mass function fit.
        
        NOTE: Only valid from -1.7 < ln sigma^-1 < 0.9
        """
        nu = self.delta_c / self.sigma

        G_1 = np.exp(-(self.lnsigma - 0.4) ** 2 / (2 * 0.6 ** 2))
        G_2 = np.exp(-(self.lnsigma - 0.75) ** 2 / (2 * 0.2 ** 2))

        c = 1.08
        a = 0.764 / c
        A = 0.3222
        p = 0.3

        vfv = A * np.sqrt(2.0 * a / np.pi) * (1.0 + (1.0 / (a * nu ** 2)) ** p + 0.6 * G_1 + 0.4 * G_2) * nu * np.exp(-c * a * nu ** 2 / 2.0 - 0.03 * nu ** 0.6 / (self.n_eff + 3) ** 2)

        if self.cut_fit:
            vfv[np.logical_or(self.lnsigma < -0.5, self.lnsigma > 1.2)] = np.NaN

        return vfv


    def _nufnu_Angulo(self):

        vfv = 0.201 * ((2.08 / self.sigma) ** 1.7 + 1) * np.exp(-1.172 / self.sigma ** 2)
        return vfv

    def _nufnu_Angulo_Bound(self):
        vfv = 0.265 * ((1.675 / self.sigma) ** 1.9 + 1) * np.exp(-1.4 / self.sigma ** 2)
        return vfv

    def _nufnu_Tinker(self):

        #The Tinker function is a bit tricky - we use the code from http://cosmo.nyu.edu/~tinker/massfunction/MF_code.tar
        #to aide us.
        delta_virs = np.array([200, 300, 400, 600, 800, 1200, 1600, 2400, 3200])
        A_array = np.array([ 1.858659e-01,
                            1.995973e-01,
                            2.115659e-01,
                            2.184113e-01,
                            2.480968e-01,
                            2.546053e-01,
                            2.600000e-01,
                            2.600000e-01,
                            2.600000e-01])

        a_array = np.array([1.466904e+00,
                            1.521782e+00,
                            1.559186e+00,
                            1.614585e+00,
                            1.869936e+00,
                            2.128056e+00,
                            2.301275e+00,
                            2.529241e+00,
                            2.661983e+00])

        b_array = np.array([2.571104e+00 ,
                            2.254217e+00,
                            2.048674e+00,
                            1.869559e+00,
                            1.588649e+00,
                            1.507134e+00,
                            1.464374e+00,
                            1.436827e+00,
                            1.405210e+00])

        c_array = np.array([1.193958e+00,
                            1.270316e+00,
                            1.335191e+00,
                            1.446266e+00,
                            1.581345e+00,
                            1.795050e+00,
                            1.965613e+00,
                            2.237466e+00,
                            2.439729e+00])
        A_func = spline(delta_virs, A_array)
        a_func = spline(delta_virs, a_array)
        b_func = spline(delta_virs, b_array)
        c_func = spline(delta_virs, c_array)

        A_0 = A_func(self.delta_halo)
        a_0 = a_func(self.delta_halo)
        b_0 = b_func(self.delta_halo)
        c_0 = c_func(self.delta_halo)

        A = A_0 * (1 + self.z) ** (-0.14)
        a = a_0 * (1 + self.z) ** (-0.06)
        alpha = np.exp(-(0.75 / np.log(self.delta_halo / 75)) ** 1.2)
        b = b_0 * (1 + self.z) ** (-alpha)
        c = c_0


        vfv = A * ((self.sigma / b) ** (-a) + 1) * np.exp(-c / self.sigma ** 2)

        if self.cut_fit:
            if self.z == 0.0:
                vfv[np.logical_or(self.lnsigma / np.log(10) < -0.6 , self.lnsigma / np.log(10) > 0.4)] = np.nan
            else:
                vfv[np.logical_or(self.lnsigma / np.log(10) < -0.2 , self.lnsigma / np.log(10) > 0.4)] = np.nan
        return vfv

    def _watson_gamma(self):
        C = np.exp(0.023 * (self.delta_halo / 178 - 1))
        d = -0.456 * cosmography.omegam_z(self.z, self.cosmo_params['omegam'], self.cosmo_params['omegav'], self.cosmo_params['omegak']) - 0.139
        p = 0.072
        q = 2.13

        return C * (self.delta_halo / 178) ** d * np.exp(p * (1 - self.delta_halo / 178) / self.sigma ** q)


    def _nufnu_Watson_FoF(self):
        vfv = 0.282 * ((1.406 / self.sigma) ** 2.163 + 1) * np.exp(-1.21 / self.sigma ** 2)
        if self.cut_fit:
            vfv[np.logical_or(self.lnsigma < -0.55 , self.lnsigma > 1.31)] = np.NaN
        return vfv

    def _nufnu_Watson(self):

        if self.z == 0:
            A = 0.194
            alpha = 2.267
            beta = 1.805
            gamma = 1.287
        elif self.z > 6:
            A = 0.563
            alpha = 0.874
            beta = 3.810
            gamma = 1.453
        else:
            A = cosmography.omegam_z(self.z, self.cosmo_params['omegam'], self.cosmo_params['omegav'], self.cosmo_params['omegak']) * (1.097 * (1 + self.z) ** (-3.216) + 0.074)
            alpha = cosmography.omegam_z(self.z, self.cosmo_params['omegam'], self.cosmo_params['omegav'], self.cosmo_params['omegak']) * (3.136 * (1 + self.z) ** (-3.058) + 2.349)
            beta = cosmography.omegam_z(self.z, self.cosmo_params['omegam'], self.cosmo_params['omegav'], self.cosmo_params['omegak']) * (5.907 * (1 + self.z) ** (-3.599) + 2.344)
            gamma = 1.318

        vfv = self._watson_gamma() * A * ((beta / self.sigma) ** alpha + 1) * np.exp(-gamma / self.sigma ** 2)

        if self.cut_fit:
            vfv[np.logical_or(self.lnsigma < -0.55, self.lnsigma > 1.05)] = np.NaN

        return vfv

    def _nufnu_Crocce(self):

        A = 0.58 * (1 + self.z) ** (-0.13)
        a = 1.37 * (1 + self.z) ** (-0.15)
        b = 0.3 * (1 + self.z) ** (-0.084)
        c = 1.036 * (1 + self.z) ** (-0.024)

        vfv = A * (self.sigma ** (-a) + b) * np.exp(-c / self.sigma ** 2)
        return vfv

    def _nufnu_Courtin(self):
        A = 0.348
        a = 0.695
        p = 0.1
        d_c = self.delta_c  # Note for WMAP5 they find delta_c = 1.673

        vfv = A * np.sqrt(2 * a / np.pi) * (d_c / self.sigma) * (1 + (d_c / (self.sigma * np.sqrt(a))) ** (-2 * p)) * np.exp(-d_c ** 2 * a / (2 * self.sigma ** 2))
        return vfv

    def _nufnu_Bhattacharya(self):
        A = 0.333 * (1 + self.z) ** -0.11
        a = 0.788 * (1 + self.z) ** -0.01
        p = 0.807
        q = 1.795

        nu = self.delta_c / self.sigma

        vfv = A * np.sqrt(2.0 / np.pi) * np.exp(-(a * nu ** 2) / 2.0) * (1 + (1.0 / (a * nu ** 2)) ** p) * (nu * np.sqrt(a)) ** q
        if self.cut_fit:
            vfv[np.logical_or(self.M < 6 * 10 ** 11, self.M > 3 * 10 ** 15)] = np.NaN

        return vfv

    def _nufnu_user_model(self):
        """
        Calculates vfv based on a user-input model.
        """
        from scitools.StringFunction import StringFunction

        f = StringFunction(self.user_fit, globals=globals())


        return f(self.sigma)
