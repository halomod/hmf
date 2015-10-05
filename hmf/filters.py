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
import astropy.units as u
from _framework import Model

class Filter(Model):

    def __init__(self,  k, power, **model_parameters):
        self.k = k
        self.power = power

        super(Filter, self).__init__(**model_parameters)

    def real_space(self, R, r):
        """
        Filter definition in real space.

        Parameters
        ----------
        R : float
            The smoothing scale

        r : array_like
            The radial co-ordinate
        """
        pass

    def k_space(self, kr):
        """
        Fourier-transform of the filter.

        Parameters
        ----------
        kr : float or array of floats
            The scales at which to return the filter function

        Returns
        -------
        w : float or array of floats
            The filter in fourier space, ``len(kr)``
        """
        pass

    def mass_to_radius(self, m,rho_mean):
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

    def radius_to_mass(self, r,rho_mean):
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

    def dw_dlnkr(self, kr):
        """
        The derivative of the filter with log kr.

        In terms of dw^2/dm, which is a commonly used quantity, this has the
        relationship :math:`w\frac{dw}{d\ln r} = \frac{2}{r}\frac{dw^2}{dm}\frac{dm}{dr}`.
        """
        pass

    def dlnss_dlnr(self, r):
        if hasattr(self.k,"value"):
            k = self.k.value
        else:
            k = self.k

        if hasattr(r,"value"):
            r = r.value

        dlnk = np.log(self.k[1] / self.k[0])
        #out = np.zeros(len(r))
        s = self.sigma(r)

        rest = self.power * k ** 3
        w = self.k_space(np.outer(r,k))
        dw = self.dw_dlnkr(np.outer(r,k))
        integ = w*dw*rest
        # for i, rr in enumerate(r):
        #     y = rr * k
        #     w = self.k_space(y)
        #     dw = self.dw_dlnkr(y)
        #     integ = w * dw * rest
        return intg.simps(integ, dx=dlnk,axis=-1) / (np.pi ** 2 * s ** 2)
        # return out

    def dlnr_dlnm(self, r):
        """
        The derivative of log scale with log mass.

        For the usual :math:`m\propto r^3` mass assignment, this is just 1/3.
        """
        return 1. / 3.

    def dlnss_dlnm(self, r):
        """
        The logarithmic slope of mass variance with mass, used directly for n(m).

        Note this is :math:`\frac{d\ln \sigma^2}{d\ln m} = 2\frac{d\ln \sigma}{d\ln m}`
        """
        return self.dlnss_dlnr(r) * self.dlnr_dlnm(r)

    def sigma(self, r, order=0):
        """
        Calculate the mass variance, :math:`\sigma(m)`.

        .. note :: This is not :math:`\sigma^2(m)`!

        Parameters
        ----------
        r : float or array_like
            The radii of the spheres at which to calculate the mass variance.

        Returns
        -------
        sigma : array_like ( ``len=len(m)`` )
            The square root of the mass variance at ``m``
        """
        if hasattr(self.k,"unit"):
            k = self.k.value
        else:
            k = self.k

        if hasattr(r,"unit"):
            r = r.value

        dlnk = np.log(self.k[1] / self.k[0])
        #sigma = np.zeros(len(r))

        # we multiply by k because our steps are in logk.
        rest = self.power * self.k ** (3 + order * 2)
        integ = rest*self.k_space(np.outer(r,k))**2
        sigma = (0.5/np.pi**2) * intg.simps(integ,dx=dlnk,axis=-1)
        # return np.
        # for i, rr in enumerate(r):
        #     integ = rest * self.k_space(rr * k) ** 2
        #     sigma[i] = (0.5 / np.pi ** 2) * intg.simps(integ, dx=dlnk)
        return np.sqrt(sigma)

    def nu(self, r,delta_c):
        return (delta_c / self.sigma(r)) ** 2

class TopHat(Filter):
    """
    Real-space top-hat window function.
    """

    def real_space(self, R, r):
        a = np.where(r<R,1,0)
        return np.where(r==R,0.5,a)

    def k_space(self,kr):
        return np.where(kr>1.4e-6,(3 / kr ** 3) * (np.sin(kr*u.rad ) - kr * np.cos(kr*u.rad )),1)

    def mass_to_radius(self, m,rho_mean):
        return (3.*m / (4.*np.pi * rho_mean)) ** (1. / 3.)

    def radius_to_mass(self, r,rho_mean):
        return 4 * np.pi * r ** 3 * rho_mean / 3

    def dw_dlnkr(self, kr):
        return np.where(kr>1e-3,(9 * kr * np.cos(kr*u.rad ) + 3 * (kr ** 2 - 3) * np.sin(kr*u.rad)) / kr ** 3,0)

class Gaussian(Filter):
    """
    Gaussian window function.
    """

    def real_space(self, R, r):
        return np.exp(-r**2/2/R**2)/(2*np.pi)**1.5/R**3

    def k_space(self, kr):
        return np.exp(-kr**2/2.0)

    def mass_to_radius(self, m,rho_mean):
        return (m/rho_mean)**(1./3.)/np.sqrt(2*np.pi)

    def radius_to_mass(self, r,rho_mean):
        return (2*np.pi)**1.5 * r**3 * rho_mean

    def dw_dlnkr(self, kr):
        return -kr * self.k_space(kr)

class SharpK(Filter):
    """
    Fourier-space top-hat window function
    """
    _defaults = {"c":2.5}

    def k_space(self, kr):
        a = np.where(kr>1,0,1)
        return np.where(kr==1,0.5,a)

    def real_space(self, R, r):
        return (np.sin(r/R * u.rad) - (r/R)*np.cos(r/R*u.rad))/(2*np.pi**2 * r**3)

    def dw_dlnkr(self, kr):
        return np.where(kr==1,1.0,0.0)

    def dlnss_dlnr(self, r):
        sigma = self.sigma(r)
        power = spline(self.k, self.power)(1 / r) * self.power.unit
        return -power / (2 * np.pi ** 2 * sigma ** 2 * r ** 3)

    def mass_to_radius(self, m,rho_mean):
        return (1. / self.params['c']) * (3.*m / (4.*np.pi * rho_mean)) ** (1. / 3.)

    def radius_to_mass(self, r,rho_mean):
        return 4 * np.pi * (self.params['c'] * r) ** 3 * rho_mean / 3

    def sigma(self, r, order=0):
        # If we input a scalar as M, then just make it a one-element list.
        if not isinstance(r, collections.Iterable):
            r = [r]

        # # Need to re-define this because the integral needs to go exactly kr=1
        # # or else the function 'jitters'
        sigma = np.zeros(len(r))
        power = spline(self.k, self.power)
        for i, rr in enumerate(r):
            k = np.logspace(np.log(self.k[0].value), np.log(1.0 / rr.value), len(self.k), base=np.e)
            p = power(k)
            dlnk = np.log(k[1] / k[0])
            integ = p * k ** (3 + 2 * order)
            sigma[i] = (0.5 / (np.pi ** 2)) * intg.simps(integ, dx=dlnk)

        return np.sqrt(sigma)


class SharpKEllipsoid(SharpK):
    """
    Fourier-space top-hat window function with ellipsoidal correction
    """
    _defaults = {"c":2.0}

    def xm(self, g, v):
        """
        Peak of the distribution of x, where x is the sum of the eigenvalues
        of the inertia tensor (?) of an ellipsoidal peak, divided by the second
        spectral moment.

        Equation A6. in Schneider et al. 2013
        """
        top = 3 * (1 - g ** 2) + (1.1 - 0.9 * g ** 4) * np.exp(-g * (1 - g ** 2) * (g * v / 2) ** 2)
        bot = (3 * (1 - g ** 2) + 0.45 + (g * v / 2) ** 2) ** 0.5 + g * v / 2
        return g * v + top / bot

    def em(self, xm):
        """
        The average ellipticity of a patch as a function of peak tensor
        """
        return 1 / np.sqrt(5 * xm ** 2 + 6)

    def pm(self, xm):
        """
        The average prolateness of a patch as a function of peak tensor
        """
        return 30.0 / (5 * xm ** 2 + 6) ** 2

    def a3a1(self, e, p):
        """
        The short to long axis ratio of an ellipsoid given its ellipticity and
        prolateness
        """
        return np.sqrt((1 - 3 * e + p) / (1 + 3 * e + p))

    def a3a2(self, e, p):
        """
        The short to medium axis ratio of an ellipsoid given its ellipticity and
        prolateness
        """
        return np.sqrt((1 - 2 * p) / (1 + 3 * e + p))

    def gamma(self, r):
        """
        Bardeen et al. 1986 equation 6.17
        """
        sig_0 = self.sigma(r)
        sig_1 = self.sigma(r, order=1)
        sig_2 = self.sigma(r, order=2)
        return sig_1 ** 2 / (sig_0 * sig_2)

    def xi(self, pm, em):
        return ((1 + 4 * pm) ** 2 / (1 - 3 * em + pm) / (1 - 2 * pm)) ** (1. / 6.)

    def a3(self, r):
        g = self.gamma(r)
        xm = self.xm(g, self.nu(r))
        em = self.em(xm)
        pm = self.pm(xm)
        return r / self.xi(pm, em)

    def r_a3(self, rmin, rmax):
        r = np.logspace(np.log(rmin), np.log(rmax), 200, base=np.e)
        a3 = self.a3(r)
        s = spline(a3, r)
        return s

    def dlnss_dlnr(self, r):
        a3 = self.a3(r)
        sigma = self.sigma(a3)
        power = np.exp(spline(self.lnk, self.lnp)(np.log(1 / a3)))
        return -power / (2 * np.pi ** 2 * sigma ** 2 * a3 ** 3)

    def dlnr_dlnm(self, r):
        a3 = self.a3(r)
        xi = r / a3
        drda = spline(a3, r).derivative()(a3)
        return xi / 3 / drda
