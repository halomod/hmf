'''
Module for calculating the growth factor (in various ways)

The main class, which is a numerical calculator, is extensible to provide simpler
fitting functions
'''
import sys
import numpy as np
from scipy import integrate as intg
import copy

def get_growth(name, cosmo, **kwargs):
    """
    Returns the correct subclass of :class:`GrowthFactor`.
    
    Parameters
    ----------
    name : str
        The class name of the appropriate fit
        
    \*\*kwargs : 
        Any parameters for the instantiated fit (including model parameters)
    """
    try:
        return getattr(sys.modules[__name__], name)(cosmo, **kwargs)
    except AttributeError:
        raise AttributeError(str(name) + "  is not a valid GrowthFactor class")

class GrowthFactor(object):
    _defaults = {}
    r"""
    General class for a growth factor calculation
     
    """
    def __init__(self, cosmo, **model_parameters):
        """
        cosmo : ``astropy.cosmology.FLRW()`` object or subclass
            Cosmological model
            
        model_parameters : 
            Other parameters of the specific model
        """
        # Check that all parameters passed are valid
        for k in model_parameters:
            if k not in self._defaults:
                raise ValueError("%s is not a valid argument for %s" % (k, self.__class__.__name__))

        # Gather model parameters
        self.params = copy.copy(self._defaults)
        self.params.update(model_parameters)

        # Set simple parameters
        self.cosmo = cosmo

    def d_plus(self, z):
        """
        Finds the factor :math:`D^+(a)`, from Lukic et. al. 2007, eq. 8.
        
        Uses simpson's rule to integrate, with 1000 steps.
        
        Parameters
        ----------
        z : float
            The redshift
            
        cosmo : ``astropy.cosmology.FLRW()`` object or subclass
            Cosmological model
        
        Returns
        -------
        dplus : float
            The un-normalised growth factor.
        """
        a_upper = 1.0 / (1.0 + z)
        lna = np.linspace(np.log(1e-8), np.log(a_upper), 1000)
        z_vec = 1.0 / np.exp(lna) - 1.0

        integrand = 1.0 / (np.exp(lna) * self.cosmo.efunc(z_vec)) ** 3
        integral = intg.simps(np.exp(lna) * integrand, dx=lna[1] - lna[0])
        dplus = 5.0 * self.cosmo.Om0 * self.cosmo.efunc(z) * integral / 2.0

#         if getvec:
#             lna = np.linspace(lna[-1], 0.0, 1000)
#             z_vec = 1.0 / np.exp(lna) - 1.0
#             integrand = 1.0 / (np.exp(lna) * cosmo.efunc(z)) ** 3
#             integral = intg.cumtrapz(np.exp(lna) * integrand, dx=lna[1] - lna[0], initial=0.0)
#
#             dplus += 5.0 * cosmo.Om0 * cosmo.efunc(z) * integral / 2.0
#
#         if getvec:
#             dplus = np.vstack((z_vec, dplus))  # spline(z_vec[1:], dplus)

        return dplus

    def growth_factor(self, z):
        """
        Calculate :math:`d(a) = D^+(a)/D^+(a=1)`, from Lukic et. al. 2007, eq. 7.
        
        Parameters
        ----------
        z : float
            The redshift
            
        Returns
        -------
        float
            The normalised growth factor.
        """
#         if not getvec:
        growth = self.d_plus(z) / self.d_plus(0.0)
#         else:
#             growth = d_plus(z, cosmo, getvec)
#             growth[1, :] /= d_plus(0.0, cosmo)

        return growth

    def growth_rate(self, z):
        """
        Growth rate, dln(d)/dln(a) from Hamilton 2000 eq. 4
        """

        return (-1 - self.cosmo.Om(z) / 2 + self.Ode(z) +
                5 * self.Om(z) / (2 * self.growth_factor(z)))


class GenMFGrowth(GrowthFactor):
    """
    Port of growth factor routines found in the genmf code.
    """
    def growth_factor(self, z):
        a = 1 / (1 + z)
        w = 1 / self.cosmo.Om0 - 1.0
        sum = self.cosmo.Om0 + self.cosmo.Ode0
        if (sum > 1 or self.cosmo.Om0 < 0 or (sum != 1 and self.cosmo.Ode0 > 0)):
            if np.abs(sum - 1.0) > 1.e-10:
                raise ValueError('Cannot cope with this cosmology!')

        if self.cosmo.Om0 == 1:
            return a
        elif self.cosmo.Ode0 > 0:
            xn = (2.0 * w) ** (1.0 / 3)
            xn_vec = np.linspace(0, xn, 1000)
            func2 = (xn_vec / (xn_vec ** 3 + 2)) ** 1.5
            g = intg.simps(func2, dx=xn_vec[1 - xn_vec[0]])
            aofxn = ((xn ** 3.0 + 2.0) ** 0.5) * (g / xn ** 1.5)
            x = a * xn
            x_vec = np.linspace(0, x, 1000)
            func2 = (x_vec / (x_vec ** 3 + 2)) ** 1.5
            g = intg.simps(func2, dx=x_vec[1 - x_vec[0]])
            aofx = ((x ** 3 + 2) ** 0.5) * (g / x ** 1.5)
            return aofx / aofxn
        else:
            dn = 1 + 3 / w + (3 * ((1 + w) ** 0.5) / w ** 1.5) * log((1 + w) ** 0.5 - w ** 0.5)
            x = w * a
            return (1 + 3 / x + (3 * ((1 + x) ** 0.5) / x ** 1.5) * log((1 + x) ** 0.5 - x ** 0.5)) / dn
