import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import cosmolopy as cp
import sys
import copy
from scipy.special import gamma as Gam
from . import cosmo
_allfits = ["ST", "SMT", 'Jenkins', "Warren", "Reed03", "Reed07", "Peacock",
            "Angulo", "AnguloBound", "Tinker", "Watson_FoF", "Watson", "Crocce",
            "Courtin", "Bhattacharya", "Behroozi", "Tinker08", "Tinker10"]

# TODO: check out units for boundaries (ie. whether they should be log or ln 1/sigma or M/h or M)
def get_fit(name, **kwargs):
    """
    Returns the correct subclass of :class:`FittingFunction`.
    
    Parameters
    ----------
    name : str
        The class name of the appropriate fit
        
    \*\*kwargs : 
        Any parameters for the instantiated fit (including model parameters)
    """
    try:
        return getattr(sys.modules[__name__], name)(**kwargs)
    except AttributeError:
        raise AttributeError(str(name) + "  is not a valid FittingFunction class")

def _makedoc(pdocs, lname, sname, eq, ref):
    return \
    r"""
    Class representing a %s mass function fit
    """ % lname + pdocs + \
    r"""
    Notes
    -----
    The %s [1]_ form is:
    
    .. math:: f_{\rm %s}(\sigma) = %s
    
    References
    ----------
    .. [1] %s
    """ % (lname, sname, eq, ref)


class FittingFunction(object):
    r"""
    Base-class for a halo mass function fit
    
    This class should not be called directly, rather use a subclass which is 
    specific to a certain fitting formula. 
    """
    _pdocs = \
    """
    
    Parameters
    ----------
    M   : array
        A vector of halo masses [units M_sun/h]
        
    nu2  : array
        A vector of peak-heights, :math:`\delta_c^2/\sigma^2` corresponding to ``M``
        
    z   : float, optional
        The redshift. 
        
    delta_halo : float, optional
        The overdensity of the halo w.r.t. the mean density of the universe.
        
    cosmo : :class:`cosmo.Cosmology` instance, optional
        A cosmology. Default is the default provided by the :class:`cosmo.Cosmology`
        class. Not required if ``omegam_z`` is passed.
         
    omegam_z : float, optional
        A value for the mean matter density at the given redshift ``z``. If not
        provided, will be calculated using the value of ``cosmo``. 
        
    \*\*model_parameters : unpacked-dictionary
        These parameters are model-specific. For any model, list the available
        parameters (and their defaults) using ``<model>._defaults``
        
    """
    __doc__ += _pdocs
    _defaults = {}

    use_cosmo = False
    def __init__(self, M, nu2, z=0, delta_halo=200, cosmo=None, omegam_z=None,
                  **model_parameters):
        # Check that all parameters passed are valid
        for k in model_parameters:
            if k not in self._defaults:
                raise ValueError("%s is not a valid argument for the %s Fitting Function" % (k, self.__class__.__name__))

        # Gather model parameters
        self.params = copy.copy(self._defaults)
        self.params.update(model_parameters)

        # Save instance variables
        self.M = M
        self.nu2 = nu2
        self.nu = np.sqrt(nu2)
        self.z = z
        self.delta_halo = delta_halo

        if omegam_z is None and self.use_cosmo:
            if cosmo is None:
                cosmo = cosmo.Cosmology()
            self.omegam_z = cp.density.omega_M_z(self.z, **cosmo.cosmolopy_dict)
        elif self.use_cosmo:
            self.omegam_z = omegam_z

    def fsigma(self, cut_fit):
        r"""
        Calculate :math:`f(\sigma)\equiv\nu f(\nu)`.
        
        Parameters
        ----------
        cut_fit : bool
            Whether to cut the fit at bounds corresponding to the fitted range
            (in mass or corresponding unit, not redshift). If so, values outside
            this range will be set to ``NaN``. 
        
        Returns
        -------
        vfv : array_like, ``len=len(self.M)``
            The function f(sigma).
        """
        raise NotImplementedError("Please use a subclass")

class PS(FittingFunction):
    _eq = r"\sqrt{\frac{2}{\pi}}\nu\exp(-0.5\nu^2)"
    _ref = r"""Press, W. H., Schechter, P., 1974. ApJ 187, 425-438.
    http://adsabs.harvard.edu/full/1974ApJ...187..425P"""

    __doc__ = _makedoc(FittingFunction._pdocs, "Press-Schechter", "PS", _eq, _ref)

    def fsigma(self, cut_fit):
        return np.sqrt(2.0 / np.pi) * self.nu * np.exp(-0.5 * self.nu2)

class SMT(FittingFunction):
    _eq = r"A\sqrt{2a/\pi}\nu\exp(-a\nu^2/2)(1+(a\nu^2)^{-p})"
    _ref = r"""Sheth, R. K., Mo, H. J., Tormen, G., May 2001. MNRAS 323 (1), 1-12.
    http://doi.wiley.com/10.1046/j.1365-8711.2001.04006.x"""
    __doc__ = _makedoc(FittingFunction._pdocs, "Sheth-Mo-Tormen", "SMT", _eq, _ref)

    _defaults = {"a":0.707, "p":0.3, "A":0.3222}

    def fsigma(self, cut_fit):
        A = self.params["A"]
        a = self.params["a"]
        p = self.params['p']

        vfv = A * np.sqrt(2.0 * a / np.pi) * self.nu * np.exp(-(a * self.nu2) / 2.0)\
                 * (1 + (1.0 / (a * self.nu2)) ** p)

        return vfv

class ST(SMT):
    pass

class Jenkins(FittingFunction):
    _eq = r"A\exp(-|\ln\sigma^{-1}+b|^c)"
    _ref = r"""Jenkins, A. R., et al., Feb. 2001. MNRAS 321 (2), 372-384.
    http://doi.wiley.com/10.1046/j.1365-8711.2001.04029.x"""
    __doc__ = _makedoc(FittingFunction._pdocs, "Jenkins", "Jenkins", _eq, _ref)
    _defaults = {"A":0.315, "b":0.61, "c":3.8}

    def fsigma(self, cut_fit):
        A = self.params["A"]
        b = self.params["b"]
        c = self.params['c']
        vfv = A * np.exp(-np.abs(self.hmf.lnsigma + b) ** c)

        if cut_fit:
            vfv[np.logical_or(self.hmf.lnsigma < -1.2, self.hmf.lnsigma > 1.05)] = np.NaN

        return vfv

class Warren(FittingFunction):
    _eq = r"A\left[\left(\frac{e}{\sigma}\right)^b + c\right]\exp(\frac{d}{\sigma^2})"
    _ref = r"""Warren, M. S., et al., Aug. 2006. ApJ 646 (2), 881-885.
    http://adsabs.harvard.edu/abs/2006ApJ...646..881W"""
    __doc__ = _makedoc(FittingFunction._pdocs, "Warren", "Warren", _eq, _ref)

    _defaults = {"A":0.7234, "b":1.625, "c":0.2538, "d":1.1982, "e":1}
    def fsigma(self, cut_fit):
        A = self.params["A"]
        b = self.params["b"]
        c = self.params['c']
        d = self.params['d']
        e = self.params['e']

        vfv = A * ((e / self.hmf.sigma) ** b + c) * np.exp(-d / self.hmf.sigma ** 2)

        if cut_fit:
            vfv[np.logical_or(self.hmf.M < 10 ** 10, self.hmf.M > 10 ** 15)] = np.NaN

        return vfv

class Reed03(ST):
    _defaults = {"a":0.707, "p":0.3, "A":0.3222, "c":0.7}
    def fsigma(self, cut_fit):
        """
        Calculate :math:`f(\sigma)` for Reed (2003) form.

        Reed, D., et al., Dec. 2003. MNRAS 346 (2), 565-572.
        http://adsabs.harvard.edu/abs/2003MNRAS.346..565R
        
        .. note:: valid for :math:`-1.7 < \ln \sigma{^-1} < 0.9`
        
        Returns
        -------
        vfv : array_like, len=len(pert.M)
            The function :math:`f(\sigma)\equiv\nu f(\nu)` defined on ``pert.M``
        """

        vfv = super(Reed03, self).fsigma(False)


        vfv *= np.exp(-self.params['c'] / (self.hmf.sigma * np.cosh(2.0 * self.hmf.sigma) ** 5))

        if cut_fit:
            vfv[np.logical_or(self.hmf.lnsigma < -1.7, self.hmf.lnsigma > 0.9)] = np.NaN
        return vfv

class Reed07(FittingFunction):
    _defaults = {"A":0.3222, "p":0.3, "c":1.08, "a":0.764}
    def fsigma(self, cut_fit):
        """
        Calculate :math:`f(\sigma)` for Reed (2007) form.

        Reed, D. S., et al., Jan. 2007. MNRAS 374 (1), 2-15.
        http://adsabs.harvard.edu/abs/2007MNRAS.374....2R
        
        .. note:: valid for :math:`-1.7 < \ln \sigma{^-1} < 0.9`
       
        Returns
        -------
        vfv : array_like, len=len(pert.M)
            The function :math:`f(\sigma)\equiv\nu f(\nu)` defined on ``pert.M``
        """
        G_1 = np.exp(-(self.hmf.lnsigma - 0.4) ** 2 / (2 * 0.6 ** 2))
        G_2 = np.exp(-(self.hmf.lnsigma - 0.75) ** 2 / (2 * 0.2 ** 2))

        c = self.params['c']
        a = self.params['a'] / self.params['c']
        A = self.params['A']
        p = self.params['p']

        vfv = A * np.sqrt(2.0 * a / np.pi) * \
            (1.0 + (1.0 / (a * self.nu ** 2)) ** p + 0.6 * G_1 + 0.4 * G_2) * self.nu * \
            np.exp(-c * a * self.nu ** 2 / 2.0 - 0.03 * self.nu ** 0.6 / (self.hmf.n_eff + 3) ** 2)

        if cut_fit:
            vfv[np.logical_or(self.hmf.lnsigma < -0.5, self.hmf.lnsigma > 1.2)] = np.NaN

        return vfv

class Peacock(FittingFunction):
    _defaults = {"a":1.529, "b":0.704, 'c':0.412}
    def fsigma(self, cut_fit):
        """
        Calculate :math:`f(\sigma)` for Peacock form.

        Peacock, J. A., Aug. 2007. MNRAS 379 (3), 1067-1074.
        http://adsabs.harvard.edu/abs/2007MNRAS.379.1067P
        
        The Peacock fit is a fit to the Warren function, but sets the derivative
        to 0 at small `M`. The paper defines it as f_coll=(1+a*nu**b)**-1 * exp(-c*nu**2)
        
        .. note:: valid for :math:`10^{10}M_\odot < M <10^{15}M_\odot`
       
        Returns
        -------
        vfv : array_like, len=len(pert.M)
            The function :math:`f(\sigma)\equiv\nu f(\nu)` defined on ``pert.M``
        """
        a = self.params['a']
        b = self.params['b']
        c = self.params['c']

        d = 1 + a * self.nu ** b
        vfv = self.nu * np.exp(-c * self.nu2) * (2 * c * d * self.nu + b * a * self.nu ** (b - 1)) / d ** 2

        if cut_fit:
            vfv[np.logical_or(self.hmf.M < 10 ** 10, self.hmf.M > 10 ** 15)] = np.NaN

        return vfv

class Angulo(Warren):
    _defaults = {"A":0.201, "b":1.7, "c":1, "d":1.172, "e":2.08}
    def fsigma(self, cut_fit):
        """
        Calculate :math:`f(\sigma)` for Angulo form.

        Angulo, R. E., et al., 2012.
        arXiv:1203.3216v1
                
        .. note:: valid for :math:`10^{8}M_\odot < M <10^{16}M_\odot`
       
        Returns
        -------
        vfv : array_like, len=len(pert.M)
            The function :math:`f(\sigma)\equiv\nu f(\nu)` defined on ``pert.M``
        """
        vfv = super(Angulo, self).fsigma(False)

        if cut_fit:
            vfv[np.logical_or(self.hmf.M < 10 ** 8, self.hmf.M > 10 ** 16)] = np.NaN
        return vfv

class AnguloBound(Angulo):
    _defaults = {"A":0.265, "b":1.9, "c":1, "d":1.4, "e":1.675}

class Watson_FoF(Warren):
    _defaults = {"A":0.282, "b":2.163, "c":1, "d":1.21, "e":1.406}
    def fsigma(self, cut_fit):
        """
        Calculate :math:`f(\sigma)` for Watson (FoF) form.

        Watson, W. A., et al., Dec. 2012.
        http://arxiv.org/abs/1212.0095
                
        .. note:: valid for :math:`-0.55<\ln\sigma^{-1}<1.31`
       
        Returns
        -------
        vfv : array_like, len=len(pert.M)
            The function :math:`f(\sigma)\equiv\nu f(\nu)` defined on ``pert.M``
        """
        vfv = super(Watson_FoF, self).fsigma(cut_fit)

        if cut_fit:
            vfv[np.logical_or(self.hmf.lnsigma < -0.55 , self.hmf.lnsigma > 1.31)] = np.NaN
        return vfv

class Watson(FittingFunction):
    _defaults = {"C_a":0.023, "d_a":0.456, "d_b":0.139, "p":0.072, "q":2.13,
                 "A_0":0.194, "alpha_0":2.267, "beta_0":1.805, "gamma_0":1.287,
                 "z_hi":6, "A_hi":0.563, "alpha_hi":0.874, "beta_hi":3.810, "gamma_hi":1.453,
                 "A_a":1.097, "A_b":3.216, "A_c":0.074,
                 "alpha_a":3.136, "alpha_b":3.058, "alpha_c":2.349,
                 "beta_a":5.907, "beta_b":3.599, "beta_c":2.344,
                 "gamma_z":1.318}
    def gamma(self):
        """
        Calculate :math:`\Gamma` for the Watson fit.
        """
        C = np.exp(self.params["C_a"] * (self.hmf.delta_halo / 178 - 1))
        d = -self.params["d_a"] * cp.density.omega_M_z(self.hmf.z, **self.hmf.cosmolopy_dict) - self.params["d_b"]
        p = self.params["p"]
        q = self.params['q']

        return C * (self.hmf.delta_halo / 178) ** d * np.exp(p * (1 - self.hmf.delta_halo / 178) / self.hmf.sigma ** q)

    def fsigma(self, cut_fit):
        """
        Calculate :math:`f(\sigma)` for Watson (SO) form.

        Watson, W. A., et al., Dec. 2012.
        http://arxiv.org/abs/1212.0095
                
        .. note:: valid for :math:`-0.55<\ln\sigma^{-1}<1.05` at ``z=0``
                  valid for :math:`-0.06<\ln\sigma^{-1}<1.024` at ``z>0``
       
        Returns
        -------
        vfv : array_like, len=len(pert.M)
            The function :math:`f(\sigma)\equiv\nu f(\nu)` defined on ``pert.M``
        """
        if self.hmf.z == 0:
            A = self.params["A_0"]
            alpha = self.params["alpha_0"]
            beta = self.params["beta_0"]
            gamma = self.params["gamma_0"]
        elif self.hmf.z > self.params['z_hi']:
            A = self.params["A_hi"]
            alpha = self.params["alpha_hi"]
            beta = self.params["beta_hi"]
            gamma = self.params["gamma_hi"]
        else:
            omz = cp.density.omega_M_z(self.hmf.z, **self.hmf.cosmolopy_dict)
            A = omz * (self.params["A_a"] * (1 + self.hmf.z) ** (-self.params["A_b"]) + self.params["A_c"])
            alpha = omz * (self.params["alpha_a"] * (1 + self.hmf.z) ** (-self.params["alpha_b"]) + self.params["alpha_c"])
            beta = omz * (self.params["beta_a"] * (1 + self.hmf.z) ** (-self.params["beta_b"]) + self.params["beta_c"])
            gamma = self.params["gamma_z"]

        vfv = self.gamma() * A * ((beta / self.hmf.sigma) ** alpha + 1) * \
                 np.exp(-gamma / self.hmf.sigma ** 2)

        if cut_fit:
            vfv[np.logical_or(self.hmf.lnsigma < -0.55, self.hmf.lnsigma > 1.05)] = np.NaN

        return vfv

class Crocce(Warren):
    _defaults = {"A_a":0.58, "A_b":0.13,
                 "b_a":1.37, "b_b":0.15,
                 "c_a":0.3, "c_b":0.084,
                 "d_a":1.036, "d_b":0.024,
                 "e":1}
    def __init__(self, hmf, **model_parameters):
        super(Crocce, self).__init__(hmf, **model_parameters)

        self.params["A"] = self.params["A_a"] * (1 + self.hmf.z) ** (-self.params["A_b"])
        self.params['b'] = self.params["b_a"] * (1 + self.hmf.z) ** (-self.params["b_b"])
        self.params['c'] = self.params["c_a"] * (1 + self.hmf.z) ** (-self.params["c_b"])
        self.params['d'] = self.params["d_a"] * (1 + self.hmf.z) ** (-self.params["d_b"])

#     def fsigma(self, cut_fit):
#         """
#         Calculate :math:`f(\sigma)` for Crocce form.
#
#         Crocce, M., et al. MNRAS 403 (3), 1353-1367.
#         http://doi.wiley.com/10.1111/j.1365-2966.2009.16194.x
#
#         .. note:: valid for :math:`10^{10.5}M_\odot < M <10^{15.5}M_\odot`
#
#         Returns
#         -------
#         vfv : array_like, len=len(pert.M)
#             The function :math:`f(\sigma)\equiv\nu f(\nu)` defined on ``pert.M``
#         """
#
#
#         vfv = A * (self.hmf.sigma ** (-a) + b) * np.exp(-c / self.hmf.sigma ** 2)
#         return vfv

class Courtin(SMT):
    _defaults = {"A":0.348, "a":0.695, "p":0.1}
#     def fsigma(self, cut_fit):
#         """
#         Calculate :math:`f(\sigma)` for Courtin form.
#
#         Courtin, J., et al., Oct. 2010. MNRAS 1931
#         http://doi.wiley.com/10.1111/j.1365-2966.2010.17573.x
#
#         .. note:: valid for :math:`-0.8<\ln\sigma^{-1}<0.7`
#
#         Returns
#         -------
#         vfv : array_like, len=len(pert.M)
#             The function :math:`f(\sigma)\equiv\nu f(\nu)` defined on ``pert.M``
#         """
#         A = 0.348
#         a = 0.695
#         p = 0.1
#         # d_c = self.hmf.delta_c  # Note for WMAP5 they find delta_c = 1.673
#
#         vfv = A * np.sqrt(2.0 * a / np.pi) * self.nu * np.exp(-(a * self.nu2) / 2.0)\
#                  * (1 + (1.0 / (a * self.nu2)) ** p)
#         return vfv

class Bhattacharya(SMT):
    _defaults = {"A_a":0.333, "A_b":0.11, "a_a":0.788, "a_b":0.01, "p":0.807, "q":1.795}

    def __init__(self, hmf, **model_parameters):
        super(Bhattacharya, self).__init__(hmf, **model_parameters)
        self.params["A"] = self.params["A_a"] * (1 + self.hmf.z) ** -self.params["A_b"]
        self.params["a"] = self.params["a_a"] * (1 + self.hmf.z) ** -self.params["a_b"]

    def fsigma(self, cut_fit):
        """
        Calculate :math:`f(\sigma)` for Bhattacharya form.

        Bhattacharya, S., et al., May 2011. ApJ 732 (2), 122.
        http://labs.adsabs.harvard.edu/ui/abs/2011ApJ...732..122B
                
        .. note:: valid for :math:`10^{11.8}M_\odot < M <10^{15.5}M_\odot`
       
        Returns
        -------
        vfv : array_like, len=len(pert.M)
            The function :math:`f(\sigma)\equiv\nu f(\nu)` defined on ``pert.M``
        """
        vfv = super(Bhattacharya, self).fsigma(cut_fit)
        vfv *= (self.nu * np.sqrt(self.params['a'])) ** self.params['q']

        if cut_fit:
            vfv[np.logical_or(self.hmf.M < 6 * 10 ** 11,
                              self.hmf.M > 3 * 10 ** 15)] = np.NaN

        return vfv

class Tinker08(FittingFunction):
    _defaults = {"A_array":np.array([ 1.858659e-01, 1.995973e-01, 2.115659e-01, 2.184113e-01,
                                     2.480968e-01, 2.546053e-01, 2.600000e-01, 2.600000e-01,
                                     2.600000e-01]),
                 "a_array": np.array([1.466904, 1.521782, 1.559186, 1.614585, 1.869936,
                                     2.128056, 2.301275, 2.529241, 2.661983e+00]),
                 "b_array": np.array([2.571104, 2.254217, 2.048674, 1.869559, 1.588649,
                                     1.507134, 1.464374, 1.436827, 1.405210]),
                 "c_array": np.array([1.193958, 1.270316, 1.335191, 1.446266, 1.581345,
                                      1.795050, 1.965613, 2.237466, 2.439729]),
                 "A_exp":0.14, "a_exp":0.06
                 }

    delta_virs = np.array([200, 300, 400, 600, 800, 1200, 1600, 2400, 3200])

    def __init__(self, hmf, **model_parameters):
        super(Tinker08, self).__init__(hmf, **model_parameters)

        if self.hmf.delta_halo not in self.delta_virs:
            A_func = spline(self.delta_virs, self.params['A_array'])
            a_func = spline(self.delta_virs, self.params['a_array'])
            b_func = spline(self.delta_virs, self.params['b_array'])
            c_func = spline(self.delta_virs, self.params['c_array'])

            A_0 = A_func(self.hmf.delta_halo)
            a_0 = a_func(self.hmf.delta_halo)
            b_0 = b_func(self.hmf.delta_halo)
            c_0 = c_func(self.hmf.delta_halo)
        else:
            ind = np.where(self.delta_virs == self.hmf.delta_halo)[0][0]
            A_0 = self.params["A_array"][ind]
            a_0 = self.params["a_array"][ind]
            b_0 = self.params["b_array"][ind]
            c_0 = self.params["c_array"][ind]


        self.A = A_0 * (1 + self.hmf.z) ** (-self.params["A_exp"])
        self.a = a_0 * (1 + self.hmf.z) ** (-self.params["a_exp"])
        alpha = 10 ** (-(0.75 / np.log10(self.hmf.delta_halo / 75)) ** 1.2)
        self.b = b_0 * (1 + self.hmf.z) ** (-alpha)
        self.c = c_0
    def fsigma(self, cut_fit):
        """
        Calculate :math:`f(\sigma)` for Tinker form.

        Tinker, J., et al., 2008. ApJ 688, 709-728.
        http://iopscience.iop.org/0004-637X/688/2/709
                
        .. note:: valid for :math:`-0.6<\log_{10}\sigma^{-1}<0.4`
       
        Returns
        -------
        vfv : array_like, len=len(pert.M)
            The function :math:`f(\sigma)\equiv\nu f(\nu)` defined on ``pert.M``
        """

        vfv = self.A * ((self.hmf.sigma / self.b) ** (-self.a) + 1) * np.exp(-self.c / self.hmf.sigma ** 2)

        if cut_fit:
            if self.hmf.z == 0.0:
                vfv[np.logical_or(self.hmf.lnsigma / np.log(10) < -0.6 ,
                                  self.hmf.lnsigma / np.log(10) > 0.4)] = np.nan
            else:
                vfv[np.logical_or(self.hmf.lnsigma / np.log(10) < -0.2 ,
                                  self.hmf.lnsigma / np.log(10) > 0.4)] = np.nan
        return vfv

class Tinker10(FittingFunction):
    _defaults = {"alpha_array":np.array([ 0.368, 0.363, 0.385, 0.389,
                                         0.393, 0.365, 0.379, 0.355, 0.327]),
                 "beta_array":np.array([0.589, 0.585, 0.544, 0.543, 0.564,
                                        0.623, 0.637, 0.673, 0.702]),
                 "gamma_array":np.array([0.864, 0.922, 0.987, 1.09, 1.20,
                                          1.34, 1.50, 1.68, 1.81]),
                 "phi_array": np.array([-0.729, -0.789, -0.910, -1.05, -1.20,
                                        - 1.26, -1.45, -1.50, -1.49]),
                 "eta_array":np.array([-0.243, -0.261, -0.261, -0.273, -0.278,
                                       - 0.301, -0.301, -0.319, -0.336]),
                 "beta_exp":0.2, "phi_exp":-0.08, "eta_exp":0.27, "gamma_exp":-0.01
                 }

    delta_virs = np.array([200, 300, 400, 600, 800, 1200, 1600, 2400, 3200])
    def __init__(self, hmf, **model_parameters):
        super(Tinker10, self).__init__(hmf, **model_parameters)

        if self.hmf.delta_halo not in self.delta_virs:
            beta_func = spline(self.delta_virs, self.params['beta_array'])
            gamma_func = spline(self.delta_virs, self.params['gamma_array'])
            phi_func = spline(self.delta_virs, self.params['phi_array'])
            eta_func = spline(self.delta_virs, self.params['eta_array'])

            beta_0 = beta_func(self.hmf.delta_halo)
            gamma_0 = gamma_func(self.hmf.delta_halo)
            phi_0 = phi_func(self.hmf.delta_halo)
            eta_0 = eta_func(self.hmf.delta_halo)
        else:
            ind = np.where(self.delta_virs == self.hmf.delta_halo)[0][0]
            beta_0 = self.params['beta_array'][ind]
            gamma_0 = self.params['gamma_array'][ind]
            phi_0 = self.params['phi_array'][ind]
            eta_0 = self.params['eta_array'][ind]

        self.beta = beta_0 * (1 + min(self.hmf.z, 3)) ** self.params["beta_exp"]
        self.phi = phi_0 * (1 + min(self.hmf.z, 3)) ** self.params['phi_exp']
        self.eta = eta_0 * (1 + min(self.hmf.z, 3)) ** self.params['eta_exp']
        self.gamma = gamma_0 * (1 + min(self.hmf.z, 3)) ** self.params['gamma_exp']

    @property
    def normalise(self):
        if self.hmf.delta_halo in self.delta_virs and self.hmf.z == 0:
            ind = np.where(self.delta_virs == self.hmf.delta_halo)[0][0]
            return self.params['alpha_array'][ind]
        else:
            return 1 / (2 ** (self.eta - self.phi - 0.5) * self.beta ** (-2 * self.phi) \
                      * self.gamma ** (-0.5 - self.eta) * (2 ** self.phi * self.beta ** (2 * self.phi)\
                      * Gam(self.eta + 0.5) + self.gamma ** self.phi * Gam(0.5 + self.eta - self.phi)))

    def fsigma(self, cut_fit):
        """
        Calculate :math:`f(\sigma)` for Tinker+10 form.

        Tinker, J., et al., 2010. ApJ 724, 878.
        http://iopscience.iop.org/0004-637X/724/2/878/pdf/apj_724_2_878.pdf
                
        .. note:: valid for :math:`-0.6<\log_{10}\sigma^{-1}<0.4`
       
        Returns
        -------
        vfv : array_like, len=len(pert.M)
            The function :math:`f(\sigma)\equiv\nu f(\nu)` defined on ``M``
        """
        fv = (1 + (self.beta * self.nu) ** (-2 * self.phi)) * \
        self.nu ** (2 * self.eta) * np.exp(-self.gamma * (self.nu ** 2) / 2)

        fv *= self.normalise * self.nu

        if cut_fit:
            if self.hmf.z == 0.0:
                fv[np.logical_or(self.hmf.lnsigma / np.log(10) < -0.6 ,
                                  self.hmf.lnsigma / np.log(10) > 0.4)] = np.nan
            else:
                fv[np.logical_or(self.hmf.lnsigma / np.log(10) < -0.2 ,
                                  self.hmf.lnsigma / np.log(10) > 0.4)] = np.nan
        return fv

class Behroozi(Tinker08):
    def _modify_dndm(self, m, dndm, z, ngtm_tinker):
        a = 1 / (1 + z)
        theta = 0.144 / (1 + np.exp(14.79 * (a - 0.213))) * (m / 10 ** 11.5) ** (0.5 / (1 + np.exp(6.5 * a)))
        ngtm_behroozi = 10 ** (theta + np.log10(ngtm_tinker))
        dthetadM = 0.144 / (1 + np.exp(14.79 * (a - 0.213))) * \
            (0.5 / (1 + np.exp(6.5 * a))) * (m / 10 ** 11.5) ** \
            (0.5 / (1 + np.exp(6.5 * a)) - 1) / (10 ** 11.5)
        return dndm * 10 ** theta - ngtm_behroozi * np.log(10) * dthetadM


class Tinker(Tinker08):
    """ Alias for Tinker08 """
    pass
