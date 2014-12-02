'''
Created on 28/11/2014

@author: Steven

A module containing various smoothing filter models, including the popular 
top-hat in real space.
'''
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import scipy.integrate as intg
import collections
import sys
import copy

def get_filter(name, **kwargs):
    """
    Returns the correct subclass of :class:`Filter`.
    
    Parameters
    ----------
    name : str
        The class name of the appropriate filter
        
    \*\*kwargs : 
        Any parameters for the instantiated filter (including model parameters)
    """
    try:
        return getattr(sys.modules[__name__], name)(**kwargs)
    except AttributeError:
        raise AttributeError(str(name) + "  is not a valid Filter class")


class Filter(object):
    _defaults = {}

    def __init__(self, rho_mean, **model_parameters):
        self.rho_mean = rho_mean

        # Check that all parameters passed are valid
        for k in model_parameters:
            if k not in self._defaults:
                raise ValueError("%s is not a valid argument for the %s Filter" % (k, self.__class__.__name__))

        # Gather model parameters
        self.params = copy.copy(self._defaults)
        self.params.update(model_parameters)

    def real_space(self):
        pass

    def k_space(self):
        """
        Fourier-transform of the filter.
        
        Parameters
        ----------
        m : float or array of floats
            The mass-scales at which to evaluate the function
            
        k : float or array of floats
            The wavenumbers at which to evaluate the function. The units must 
            match the (inverse) units of m. Only one of ``k`` or ``m`` may be 
            an array.
           
        Returns
        -------
        w : float or array of floats 
            The top-hat filter in fourier space, ``len(m|k)`` 
        """
        pass

    def mass_to_radius(self, m):
        """
        Calculate radius of a region of space from its mass.
        
        Parameters
        ----------
        m : float or array of floats
            Masses
            
        Returns
        ------
        r : float or array of floats
            The corresponding radii to m
        
        .. note :: The units of ``m`` don't matter as long as they are consistent with 
                ``rho_mean``.
        """
        pass

    def radius_to_mass(self):
        """
        Calculates mass of a region of space from its radius
    
        Parameters
        ----------
        r : float or array of floats
            Radii

        Returns
        ------
        m : float or array of floats
            The corresponding masses to r
    
        Notes
        -----
        The units of ``r`` don't matter as long as they are consistent with
        ``rho_mean``.
        """
        pass

    def dw2dm(self):
        """
        The derivative of the filter squared with mass
        """
        pass

    def dw_dlnkr(self, kr):
        """
        The derivative of the filter with log kr.
        
        In terms of dw^2/dm, which is a commonly used quantity, this has the 
        relationship :math:`w\frac{dw}{d\ln r} = \frac{2}{r}\frac{dw^2}{dm}\frac{dm}{dr}`. 
        """
        pass

    def dlnss_dlnr(self, m, sigma, lnk, lnp):
        """
        The derivative of log variance with log scale. 
        """
        dlnk = lnk[1] - lnk[0]
        r = self.mass_to_radius(m)
        out = np.zeros_like(m)
        for i, rr in enumerate(r):
            w = self.k_space(m[i], np.exp(lnk))
            dw = self.dw_dlnkr(np.exp(lnk) * rr)
            integ = w * dw * np.exp(lnp + 3 * lnk)
            out[i] = (1 / np.pi ** 2 * sigma[i] ** 2) * intg.simps(integ, dx=dlnk)
        return out

    def dlnr_dlnm(self):
        """
        The derivative of log scale with log mass.
        
        For the usual :math:`m\propto r^3` mass assignment, this is just 1/3.
        """
        return 1. / 3.

    def dlnss_dlnm(self, m, sigma, lnk, lnp):
        """
        The logarithmic slope of mass variance with mass, used directly for n(m).
        
        Note this is :math:`\frac{d\ln \sigma^2}{d\ln m} = 2\frac{d\ln \sigma}{d\ln m}`
        """
        return self.dlnss_dlnr(m, sigma, lnk, lnp) * self.dlnr_dlnm()

    def sigma(self, m, lnk, lnp):
        """
        Calculate the mass variance, :math:`\sigma(m)`.
        
        .. note :: This is not :math:`\sigma^2(m)`!
        
        Parameters
        ----------
        m : float or array_like
            The mass of the sphere at which to calculate the mass variance.
            
        lnk, lnp : array_like
            The associated log wavenumber and power spectrum
            
        
        Returns
        -------
        sigma : array_like ( ``len=len(m)`` )
            The square root of the mass variance at ``m``
        """
        # If we input a scalar as M, then just make it a one-element list.
        if not isinstance(m, collections.Iterable):
            m = [m]

        dlnk = lnk[1] - lnk[0]
        # Calculate the integrand of the function. Note that the power spectrum and k values must be
        # 'un-logged' before use, and we multiply by k because our steps are in logk.
        sigma = np.zeros_like(m)
        rest = np.exp(lnp + 3 * lnk)
        for i, mm in enumerate(m):
            integ = rest * self.k_space(mm, np.exp(lnk)) ** 2
            sigma[i] = (0.5 / np.pi ** 2) * intg.simps(integ, dx=dlnk)

        return np.sqrt(sigma)

class TopHat(Filter):
    """
    Real-space top-hat window function.
    """

    def real_space(self, m, r):
        # TODO: not sure if this is right?
        r_m = self.mass_to_radius(m)
        if r < r_m:
            return 1.0
        elif r == r_m:
            return 0.5
        else:
            return 0.0

    def k_space(self, m , k):
        kr = k * self.mass_to_radius(m)
        w = np.ones(len(kr))
        K = kr[kr > 1.4e-6]
        # Truncate the filter at small kr for numerical reasons
        w[kr > 1.4e-6] = (3 / K ** 3) * (np.sin(K) - K * np.cos(K))
        return w

    def mass_to_radius(self, m):
        return (3.*m / (4.*np.pi * self.rho_mean)) ** (1. / 3.)

    def radius_to_mass(self, r):
        return 4 * np.pi * r ** 3 * self.rho_mean / 3

#     def dw2dm(self, m, k):
#         kr = k * self.mass_to_radius(m)
#         return (np.sin(kr) - kr * np.cos(kr)) * \
#             (np.sin(kr) * (1 - 3.0 / (kr ** 2)) + 3.0 * np.cos(kr) / kr)

    def dw_dlnkr(self, kr):
        out = np.zeros_like(kr)
        y = kr[kr > 1e-3]
        out[kr > 1e-3] = (9 * y * np.cos(y) + 3 * (y ** 2 - 3) * np.sin(y)) / y ** 3
        return out

    def dlnss_dlnr(self, m, sigma, lnk, lnp):
        dlnk = lnk[1] - lnk[0]
        r = self.mass_to_radius(m)
        out = np.zeros_like(m)
        for i, rr in enumerate(r):
            w = self.k_space(m[i], np.exp(lnk))
            dw = self.dw_dlnkr(np.exp(lnk) * rr)
            integ = w * dw * np.exp(lnp + 3 * lnk)
            out[i] = (1 / np.pi ** 2 * sigma[i] ** 2) * intg.simps(integ, dx=dlnk)
        return out

class SharpK(Filter):
    """
    Fourier-space top-hat window function
    """
    _defaults = {"c":2.7}
    def k_space(self, m, k):
        kr = k * self.mass_to_radius(m)
        w = np.ones(len(kr))
        w[kr > 1] = 0.0
        w[kr == 1] = 0.5
        return w

    def real_space(self, m, r):
        # TODO: write this
        pass

    def dw_dlnkr(self, kr):
        out = np.zeros_like(kr)
        out[kr == 1] = 1.0
        return out

    def dlnss_dlnr(self, m, sigma, lnk, lnp):
        r = self.mass_to_radius(m)
        power = np.exp(spline(lnk, lnp)(np.log(1 / r)))
        return (1.0 / 2 * np.pi ** 2 * sigma ** 2) * (power / r ** 3)

    def mass_to_radius(self, m):
        return (1. / self.params['c']) * (3.*m / (4.*np.pi * self.rho_mean)) ** (1. / 3.)

    def radius_to_mass(self, r):
        return 4 * np.pi * (self.params['c'] * r) ** 3 * self.rho_mean / 3

#     def sigma(self, m, lnk, lnp):
#         # If we input a scalar as M, then just make it a one-element list.
#         if not isinstance(m, collections.Iterable):
#             m = [m]
#
#         p = spline(lnk, lnp)
# #         dlnk = lnk[1] - lnk[0]
#         sigma = np.zeros_like(m)
#         kmin = np.exp(lnk.min())
#         kmax = np.exp(lnk.max())
#
# #         rest = np.exp(lnp + 3 * lnk)
#         for i, r in enumerate(self.mass_to_radius(m)):
#             xmin = max(0, 1 - r * kmax)
#             xmax = 1 - r * kmin
# #             x = np.logspace(np.log(xmin), np.log(xmax), 300, base=np.e)
#             x = np.linspace(xmin, xmax, 300)
#             dlnx = x[1] - x[0]
#             integ = (1 - x) ** 2 * np.exp(p(np.log((1 - x) / r)))  # * x
#             sigma[i] = (0.5 / (np.pi ** 2 * r ** 3)) * intg.simps(integ, dx=dlnx)
#
#         return np.sqrt(sigma)

    def sigma(self, m, lnk, lnp):
        # If we input a scalar as M, then just make it a one-element list.
        if not isinstance(m, collections.Iterable):
            m = [m]

        # # Need to re-define this because the integral needs to go exactly kr=1
        # # or else the function 'jitters'
        sigma = np.zeros_like(m)
        power = spline(lnk, lnp)
        for i, r in enumerate(self.mass_to_radius(m)):
            lnk = np.linspace(lnk[0], np.log(1.0 / r), len(lnk))
            p = power(lnk)
            dlnk = lnk[1] - lnk[0]
            integ = np.exp(p + 3 * lnk)
            sigma[i] = (0.5 / (np.pi ** 2)) * intg.simps(integ, dx=dlnk)

        return np.sqrt(sigma)
