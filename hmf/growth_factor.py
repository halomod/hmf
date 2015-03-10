'''
Module for calculating the growth factor (in various ways)

The main class, which is a numerical calculator, is extensible to provide simpler
fitting functions
'''
import numpy as np
from scipy import integrate as intg
from _framework import Model
from scipy.interpolate import InterpolatedUnivariateSpline as spline

class GrowthFactor(Model):
    r"""
    General class for a growth factor calculation
     
    """
    _defaults = {"dlna":0.01, "amin":1e-8}
    def __init__(self, cosmo, **model_parameters):
        """
        cosmo : ``astropy.cosmology.FLRW()`` object or subclass
            Cosmological model
            
        model_parameters : 
            Other parameters of the specific model
        """

        # Set simple parameters
        self.cosmo = cosmo
        super(GrowthFactor, self).__init__(**model_parameters)

    def d_plus(self, z, getvec=False):
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
        lna = np.arange(np.log(self.params["amin"]), np.log(a_upper) + self.params['dlna'] / 2, self.params['dlna'])
        self._zvec = 1.0 / np.exp(lna) - 1.0

        integrand = 1.0 / (np.exp(lna) * self.cosmo.efunc(self._zvec)) ** 3

        if not getvec:
            integral = intg.simps(np.exp(lna) * integrand, dx=self.params['dlna'])
            dplus = 5.0 * self.cosmo.Om0 * self.cosmo.efunc(z) * integral / 2.0
        else:
            integral = intg.cumtrapz(np.exp(lna) * integrand, dx=self.params['dlna'], initial=0.0)
            dplus = 5.0 * self.cosmo.Om0 * self.cosmo.efunc(self._zvec) * integral / 2.0

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
        growth = self.d_plus(z) / self.d_plus(0.0)
        return growth

    def growth_factor_fn(self, zmin=0.0, inverse=False):
        """
        Calculate :math:`d(a) = D^+(a)/D^+(a=1)`, from Lukic et. al. 2007, eq. 7.
        
        Returns a function G(z).
        
        Parameters
        ----------
        zmin : float, optional
            The minimum redshift of the function. Default 0.0
            
        inverse: bool, optional
            Whether to return the inverse relationship [z(g)]. Default False.
            
        Returns
        -------
        callable
            The normalised growth factor as a function of redshift, or
            redshift as a function of growth factor if ``inverse`` is True.
        """
        growth = self.d_plus(zmin, True) / self.d_plus(0.0)
        if not inverse:
            s = spline(self._zvec[::-1], growth[::-1])
        else:
            s = spline(growth, self._zvec)
        return s

    def growth_rate(self, z):
        """
        Growth rate, dln(d)/dln(a) from Hamilton 2000 eq. 4
        
        Parameters
        ----------
        z : float
            The redshift
        """
        return (-1 - self.cosmo.Om(z) / 2 + self.Ode(z) +
                5 * self.cosmo.Om(z) / (2 * self.growth_factor(z)))


    def growth_rate_fn(self, zmin):
        """
        Growth rate, dln(d)/dln(a) from Hamilton 2000 eq. 4
        
        Parameters
        ----------
        zmin : float, optional
            The minimum redshift of the function. Default 0.0
            
        Returns
        -------
        callable
            The normalised growth rate as a function of redshift.
        """
        gfn = self.growth_factor_fn(zmin)

        return lambda z: (-1 - self.cosmo.Om(z) / 2 + self.cosmo.Ode(z) +
                          5 * self.cosmo.Om(z) / (2 * gfn(z)))

class GenMFGrowth(GrowthFactor):
    """
    Port of growth factor routines found in the genmf code.
    """
    _defaults = {"dz":0.01, "zmax":1000.0}
    def d_plus(self, z, getvec=False):
        raise NotImplementedError()

    def _general_case(self, w, x):
        xn_vec = np.linspace(0, x, 1000)
        func = (xn_vec / (xn_vec ** 3 + 2)) ** 1.5
        g = intg.simps(func, dx=xn_vec[1] - xn_vec[0])
        return ((x ** 3.0 + 2.0) ** 0.5) * (g / x ** 1.5)

    def growth_factor(self, z):
        a = 1 / (1 + z)
        w = 1 / self.cosmo.Om0 - 1.0
        s = self.cosmo.Om0 + self.cosmo.Ode0
        if (s > 1 or self.cosmo.Om0 < 0 or (s != 1 and self.cosmo.Ode0 > 0)):
            if np.abs(s - 1.0) > 1.e-10:
                raise ValueError('Cannot cope with this cosmology!')

        if self.cosmo.Om0 == 1:
            return a
        elif self.cosmo.Ode0 > 0:
            xn = (2.0 * w) ** (1.0 / 3)
            aofxn = self._general_case(w, xn)
            x = a * xn
            aofx = self._general_case(w, x)
            return aofx / aofxn
        else:
            dn = 1 + 3 / w + (3 * ((1 + w) ** 0.5) / w ** 1.5) * np.log((1 + w) ** 0.5 - w ** 0.5)
            x = w * a
            return (1 + 3 / x + (3 * ((1 + x) ** 0.5) / x ** 1.5) * np.log((1 + x) ** 0.5 - x ** 0.5)) / dn

    def growth_factor_fn(self, zmin=0.0, inverse=False):
        if not inverse:
            return self.growth_factor
        else:
            self._zvec = np.arange(zmin, self.params['zmax'], self.params['dz'])
            gf = self.growth_factor(self._zvec)
            return spline(gf, self._zvec)
