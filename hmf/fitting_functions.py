import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import copy
from scipy.special import gamma as Gam
import cosmo
from _framework import Model
_allfits = ["ST", "SMT", 'Jenkins', "Warren", "Reed03", "Reed07", "Peacock",
            "Angulo", "AnguloBound", "Tinker", "Watson_FoF", "Watson", "Crocce",
            "Courtin", "Bhattacharya", "Behroozi", "Tinker08", "Tinker10"]

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


class FittingFunction(Model):
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
    def __init__(self, M, nu2, delta_c, sigma=None, n_eff=None, lnsigma=None, z=0,
                 delta_halo=200, cosmo=None, omegam_z=None,
                  **model_parameters):
        # Save instance variables
        self.M = M.value
        self.nu2 = nu2
        self.nu = np.sqrt(nu2)
        self.z = z
        self.delta_halo = delta_halo
        self.n_eff = n_eff
        if sigma is None:
            self.sigma = delta_c / self.nu
        else:
            self.sigma = sigma
        if lnsigma is None:
            self.lnsigma = np.log(1 / self.sigma)
        else:
            self.lnsigma = lnsigma

        self.delta_c = delta_c
        if omegam_z is None and self.use_cosmo:
            if cosmo is None:
                cosmo = cosmo.Cosmology()
            self.omegam_z = cosmo.cosmo.Om(self.z)
        elif self.use_cosmo:
            self.omegam_z = omegam_z

        super(FittingFunction, self).__init__(**model_parameters)

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
        vfv = A * np.exp(-np.abs(self.lnsigma + b) ** c)

        if cut_fit:
            vfv[np.logical_or(self.lnsigma < -1.2, self.lnsigma > 1.05)] = np.NaN

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

        vfv = A * ((e / self.sigma) ** b + c) * np.exp(-d / self.sigma ** 2)

        if cut_fit:
            vfv[np.logical_or(self.M < 10 ** 10, self.M > 10 ** 15)] = np.NaN

        return vfv

class Reed03(ST):
    _eq = r"f_{\rm SMT}(\sigma)\exp\left(-\frac{c}{\sigma \cosh^5(2\sigma)}\right)"
    _ref = r"""Reed, D., et al., Dec. 2003. MNRAS 346 (2), 565-572.
    http://adsabs.harvard.edu/abs/2003MNRAS.346..565R"""
    __doc__ = _makedoc(FittingFunction._pdocs, "Reed03", "R03", _eq, _ref)

    _defaults = {"a":0.707, "p":0.3, "A":0.3222, "c":0.7}

    def fsigma(self, cut_fit):
        vfv = super(Reed03, self).fsigma(False)
        vfv *= np.exp(-self.params['c'] / (self.sigma * np.cosh(2.0 * self.sigma) ** 5))

        if cut_fit:
            vfv[np.logical_or(self.lnsigma < -1.7, self.lnsigma > 0.9)] = np.NaN
        return vfv

class Reed07(FittingFunction):
    _eq = r"A\sqrt{2a/\pi}\left[1+(\frac{1}{a\nu^2})^p+0.6G_1+0.4G_2\right]\nu\exp(-ca\nu^2/2-\frac{0.03\nu^{0.6}}{(n_{\rm eff}+3)^2})"
    _ref = """Reed, D. S., et al., Jan. 2007. MNRAS 374 (1), 2-15.
    http://adsabs.harvard.edu/abs/2007MNRAS.374....2R"""
    __doc__ = _makedoc(FittingFunction._pdocs, "Reed07", "R07", _eq, _ref)

    _defaults = {"A":0.3222, "p":0.3, "c":1.08, "a":0.764}
    def fsigma(self, cut_fit):
        G_1 = np.exp(-(self.lnsigma - 0.4) ** 2 / (2 * 0.6 ** 2))
        G_2 = np.exp(-(self.lnsigma - 0.75) ** 2 / (2 * 0.2 ** 2))

        c = self.params['c']
        a = self.params['a'] / self.params['c']
        A = self.params['A']
        p = self.params['p']

        vfv = A * np.sqrt(2.0 * a / np.pi) * \
            (1.0 + (1.0 / (a * self.nu ** 2)) ** p + 0.6 * G_1 + 0.4 * G_2) * self.nu * \
            np.exp(-c * a * self.nu ** 2 / 2.0 - 0.03 * self.nu ** 0.6 / (self.n_eff + 3) ** 2)

        if cut_fit:
            vfv[np.logical_or(self.lnsigma < -0.5, self.lnsigma > 1.2)] = np.NaN

        return vfv

class Peacock(FittingFunction):
    _eq = r"\nu\exp(-c\nu^2)(2cd\nu+ba\nu^{b-1})/d^2"
    _ref = """Peacock, J. A., Aug. 2007. MNRAS 379 (3), 1067-1074.
    http://adsabs.harvard.edu/abs/2007MNRAS.379.1067P"""
    __doc__ = _makedoc(FittingFunction._pdocs, "Peacock", "Pck", _eq, _ref)
    _defaults = {"a":1.529, "b":0.704, 'c':0.412}

    def fsigma(self, cut_fit):
        a = self.params['a']
        b = self.params['b']
        c = self.params['c']

        d = 1 + a * self.nu ** b
        vfv = self.nu * np.exp(-c * self.nu2) * (2 * c * d * self.nu + b * a * self.nu ** (b - 1)) / d ** 2

        if cut_fit:
            vfv[np.logical_or(self.M < 10 ** 10, self.M > 10 ** 15)] = np.NaN

        return vfv

class Angulo(Warren):
    _ref = """Angulo, R. E., et al., 2012.
    arXiv:1203.3216v1"""
    __doc__ = _makedoc(FittingFunction._pdocs, "Angulo", "Ang", Warren._eq, _ref)
    _defaults = {"A":0.201, "b":1.7, "c":1, "d":1.172, "e":2.08}
    def fsigma(self, cut_fit):
        vfv = super(Angulo, self).fsigma(False)

        if cut_fit:
            vfv[np.logical_or(self.M < 10 ** 8, self.M > 10 ** 16)] = np.NaN
        return vfv

class AnguloBound(Angulo):
    _defaults = {"A":0.265, "b":1.9, "c":1, "d":1.4, "e":1.675}

class Watson_FoF(Warren):
    _ref = """Watson, W. A., et al., Dec. 2012.
    http://arxiv.org/abs/1212.0095"""
    __doc__ = _makedoc(FittingFunction._pdocs, "WatsonFoF", "WatF", Warren._eq, _ref)
    _defaults = {"A":0.282, "b":2.163, "c":1, "d":1.21, "e":1.406}
    def fsigma(self, cut_fit):
        vfv = super(Watson_FoF, self).fsigma(cut_fit)

        if cut_fit:
            vfv[np.logical_or(self.lnsigma < -0.55 , self.lnsigma > 1.31)] = np.NaN
        return vfv

class Watson(FittingFunction):
    _eq = r"\Gamma A \left((\frac{\beta}{\sigma}^\alpha+1\right)\exp(-\gamma/\sigma^2)"
    __doc__ = _makedoc(FittingFunction._pdocs, "Watson", "WatS", _eq, Watson_FoF._ref)
    _defaults = {"C_a":0.023, "d_a":0.456, "d_b":0.139, "p":0.072, "q":2.13,
                 "A_0":0.194, "alpha_0":2.267, "beta_0":1.805, "gamma_0":1.287,
                 "z_hi":6, "A_hi":0.563, "alpha_hi":0.874, "beta_hi":3.810, "gamma_hi":1.453,
                 "A_a":1.097, "A_b":3.216, "A_c":0.074,
                 "alpha_a":3.136, "alpha_b":3.058, "alpha_c":2.349,
                 "beta_a":5.907, "beta_b":3.599, "beta_c":2.344,
                 "gamma_z":1.318}
    use_cosmo = True
    def gamma(self):
        """
        Calculate :math:`\Gamma` for the Watson fit.
        """
        C = np.exp(self.params["C_a"] * (self.delta_halo / 178 - 1))
        d = -self.params["d_a"] * self.omegam_z - self.params["d_b"]
        p = self.params["p"]
        q = self.params['q']

        return C * (self.delta_halo / 178) ** d * np.exp(p * (1 - self.delta_halo / 178) / self.sigma ** q)

    def fsigma(self, cut_fit):
        if self.z == 0:
            A = self.params["A_0"]
            alpha = self.params["alpha_0"]
            beta = self.params["beta_0"]
            gamma = self.params["gamma_0"]
        elif self.z >= self.params['z_hi']:
            A = self.params["A_hi"]
            alpha = self.params["alpha_hi"]
            beta = self.params["beta_hi"]
            gamma = self.params["gamma_hi"]
        else:
            omz = self.omegam_z
            A = omz * (self.params["A_a"] * (1 + self.z) ** (-self.params["A_b"]) + self.params["A_c"])
            alpha = omz * (self.params["alpha_a"] * (1 + self.z) ** (-self.params["alpha_b"]) + self.params["alpha_c"])
            beta = omz * (self.params["beta_a"] * (1 + self.z) ** (-self.params["beta_b"]) + self.params["beta_c"])
            gamma = self.params["gamma_z"]

        vfv = self.gamma() * A * ((beta / self.sigma) ** alpha + 1) * \
                 np.exp(-gamma / self.sigma ** 2)

        if cut_fit:
            vfv[np.logical_or(self.lnsigma < -0.55, self.lnsigma > 1.05)] = np.NaN

        return vfv

class Crocce(Warren):
    _ref = """Crocce, M., et al. MNRAS 403 (3), 1353-1367.
    http://doi.wiley.com/10.1111/j.1365-2966.2009.16194.x"""
    __doc__ = _makedoc(FittingFunction._pdocs, "Crocce", "Cro", Warren._eq, _ref)
    _defaults = {"A_a":0.58, "A_b":0.13,
                 "b_a":1.37, "b_b":0.15,
                 "c_a":0.3, "c_b":0.084,
                 "d_a":1.036, "d_b":0.024,
                 "e":1}
    def __init__(self, **model_parameters):
        super(Crocce, self).__init__(**model_parameters)

        self.params["A"] = self.params["A_a"] * (1 + self.z) ** (-self.params["A_b"])
        self.params['b'] = self.params["b_a"] * (1 + self.z) ** (-self.params["b_b"])
        self.params['c'] = self.params["c_a"] * (1 + self.z) ** (-self.params["c_b"])
        self.params['d'] = self.params["d_a"] * (1 + self.z) ** (-self.params["d_b"])

#         .. note:: valid for :math:`10^{10.5}M_\odot < M <10^{15.5}M_\odot`


class Courtin(SMT):
    _ref = """Courtin, J., et al., Oct. 2010. MNRAS 1931
    http://doi.wiley.com/10.1111/j.1365-2966.2010.17573.x"""
    __doc__ = _makedoc(FittingFunction._pdocs, "Courtin", "Ctn", SMT._eq, _ref)
    _defaults = {"A":0.348, "a":0.695, "p":0.1}

#         .. note:: valid for :math:`-0.8<\ln\sigma^{-1}<0.7`


class Bhattacharya(SMT):
    _eq = r"f_{\rm SMT}(\sigma) (\nu\sqrt{a})^q"
    _ref = """Bhattacharya, S., et al., May 2011. ApJ 732 (2), 122.
    http://labs.adsabs.harvard.edu/ui/abs/2011ApJ...732..122B"""
    __doc__ = _makedoc(FittingFunction._pdocs, "Bhattacharya", "Btc", _eq, _ref)
    _defaults = {"A_a":0.333, "A_b":0.11, "a_a":0.788, "a_b":0.01, "p":0.807, "q":1.795}

    def __init__(self, **model_parameters):
        super(Bhattacharya, self).__init__(**model_parameters)
        self.params["A"] = self.params["A_a"] * (1 + self.z) ** -self.params["A_b"]
        self.params["a"] = self.params["a_a"] * (1 + self.z) ** -self.params["a_b"]

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
            vfv[np.logical_or(self.M < 6 * 10 ** 11,
                              self.M > 3 * 10 ** 15)] = np.NaN

        return vfv

class Tinker08(FittingFunction):
    _eq = r"A(\frac{\sigma}{b}^{-a}+1)\exp(-c/\sigma^2)"
    _ref = """Tinker, J., et al., 2008. ApJ 688, 709-728.
    http://iopscience.iop.org/0004-637X/688/2/709"""
    __doc__ = _makedoc(FittingFunction._pdocs, "Tinker08", "Tkr", _eq, _ref)
    _defaults = {  # -- A
                "A_200":1.858659e-01,
                "A_300":1.995973e-01,
                "A_400":2.115659e-01,
                "A_600": 2.184113e-01,
                "A_800":2.480968e-01,
                "A_1200":2.546053e-01,
                "A_1600":2.600000e-01,
                "A_2400":2.600000e-01,
                "A_3200":2.600000e-01,
                # -- a
                "a_200":1.466904,
                "a_300":1.521782,
                "a_400":1.559186,
                "a_600": 1.614585,
                "a_800":1.869936,
                "a_1200":2.128056,
                "a_1600":2.301275,
                "a_2400":2.529241,
                "a_3200":2.661983,
                #--- b
                "b_200":2.571104,
                "b_300":2.254217,
                "b_400":2.048674,
                "b_600": 1.869559,
                "b_800":1.588649,
                "b_1200":1.507134,
                "b_1600":1.464374,
                "b_2400":1.436827,
                "b_3200":1.405210,
                #--- c
                "c_200":1.193958,
                "c_300":1.270316,
                "c_400":1.335191,
                "c_600": 1.446266,
                "c_800":1.581345,
                "c_1200": 1.795050,
                "c_1600":1.965613,
                "c_2400":2.237466,
                "c_3200":2.439729,
                # -- others
                "A_exp":0.14, "a_exp":0.06}

    delta_virs = np.array([200, 300, 400, 600, 800, 1200, 1600, 2400, 3200])

    def __init__(self, **model_parameters):
        super(Tinker08, self).__init__(**model_parameters)


        if self.delta_halo not in self.delta_virs:
            A_array = np.array([self.params["A_%s" % d] for d in self.delta_virs])
            a_array = np.array([self.params["a_%s" % d] for d in self.delta_virs])
            b_array = np.array([self.params["b_%s" % d] for d in self.delta_virs])
            c_array = np.array([self.params["c_%s" % d] for d in self.delta_virs])

            A_func = spline(self.delta_virs, A_array)
            a_func = spline(self.delta_virs, a_array)
            b_func = spline(self.delta_virs, b_array)
            c_func = spline(self.delta_virs, c_array)

            A_0 = A_func(self.delta_halo)
            a_0 = a_func(self.delta_halo)
            b_0 = b_func(self.delta_halo)
            c_0 = c_func(self.delta_halo)
        else:
            A_0 = self.params["A_%s" % (int(self.delta_halo))]
            a_0 = self.params["a_%s" % (int(self.delta_halo))]
            b_0 = self.params["b_%s" % (int(self.delta_halo))]
            c_0 = self.params["c_%s" % (int(self.delta_halo))]


        self.A = A_0 * (1 + self.z) ** (-self.params["A_exp"])
        self.a = a_0 * (1 + self.z) ** (-self.params["a_exp"])
        alpha = 10 ** (-(0.75 / np.log10(self.delta_halo / 75)) ** 1.2)
        self.b = b_0 * (1 + self.z) ** (-alpha)
        self.c = c_0

    def fsigma(self, cut_fit):
        vfv = self.A * ((self.sigma / self.b) ** (-self.a) + 1) * np.exp(-self.c / self.sigma ** 2)

        if cut_fit:
            if self.z == 0.0:
                vfv[np.logical_or(self.lnsigma / np.log(10) < -0.6 ,
                                  self.lnsigma / np.log(10) > 0.4)] = np.nan
            else:
                vfv[np.logical_or(self.lnsigma / np.log(10) < -0.2 ,
                                  self.lnsigma / np.log(10) > 0.4)] = np.nan
        return vfv

class Tinker10(FittingFunction):
    _eq = r"(1+(\beta\nu)^{-2\phi})\nu^{2\eta+1}\exp(-\gamma\nu^2/2)"
    _ref = """Tinker, J., et al., 2010. ApJ 724, 878.
    http://iopscience.iop.org/0004-637X/724/2/878/pdf/apj_724_2_878.pdf"""
    _defaults = {  # --- alpha
                  "alpha_200": 0.368, "alpha_300":0.363, "alpha_400":0.385,
                  "alpha_600":0.389, "alpha_800":0.393, "alpha_1200":0.365,
                  "alpha_1600":0.379, "alpha_2400":0.355, "alpha_3200":0.327,
                  #--- beta
                  "beta_200": 0.589, "beta_300":0.585, "beta_400":0.544, "beta_600":0.543,
                  "beta_800":0.564, "beta_1200":0.623, "beta_1600":0.637, "beta_2400":0.673,
                  "beta_3200":0.702,
                  # --- gamma
                  "gamma_200": 0.864, "gamma_300":0.922, "gamma_400":0.987,
                  "gamma_600":1.09, "gamma_800":1.2, "gamma_1200":1.34,
                  "gamma_1600":1.5, "gamma_2400":1.68, "gamma_3200":1.81,
                  # --- phi
                  "phi_200":-0.729, "phi_300":-0.789, "phi_400":-0.910,
                  "phi_600":-1.05, "phi_800":-1.2, "phi_1200":-1.26,
                  "phi_1600":-1.45, "phi_2400":-1.5, "phi_3200":-1.49,
                  # -- eta
                  "eta_200":-0.243, "eta_300":-0.261, "eta_400":-0.261,
                  "eta_600":-0.273, "eta_800":-0.278, "eta_1200":-0.301,
                  "eta_1600":-0.301, "eta_2400":-0.319, "eta_3200":-0.336,
                  # --others
                  "beta_exp":0.2, "phi_exp":-0.08, "eta_exp":0.27, "gamma_exp":-0.01,
                  "max_z":3
                  }

    delta_virs = np.array([200, 300, 400, 600, 800, 1200, 1600, 2400, 3200])
    terminate = True

    def __init__(self, **model_parameters):
        super(Tinker10, self).__init__(**model_parameters)


        if self.delta_halo not in self.delta_virs:
            beta_array = np.array([self.params["beta_%s" % d] for d in self.delta_virs])
            gamma_array = np.array([self.params["gamma_%s" % d] for d in self.delta_virs])
            phi_array = np.array([self.params["phi_%s" % d] for d in self.delta_virs])
            eta_array = np.array([self.params["eta_%s" % d] for d in self.delta_virs])

            beta_func = spline(self.delta_virs, beta_array)
            gamma_func = spline(self.delta_virs, gamma_array)
            phi_func = spline(self.delta_virs, phi_array)
            eta_func = spline(self.delta_virs, eta_array)

            beta_0 = beta_func(self.delta_halo)
            gamma_0 = gamma_func(self.delta_halo)
            phi_0 = phi_func(self.delta_halo)
            eta_0 = eta_func(self.delta_halo)
        else:
            beta_0 = self.params['beta_%s' % (int(self.delta_halo))]
            gamma_0 = self.params['gamma_%s' % (int(self.delta_halo))]
            phi_0 = self.params['phi_%s' % (int(self.delta_halo))]
            eta_0 = self.params['eta_%s' % (int(self.delta_halo))]

        self.beta = beta_0 * (1 + min(self.z, self.params["max_z"])) ** self.params["beta_exp"]
        self.phi = phi_0 * (1 + min(self.z, self.params["max_z"])) ** self.params['phi_exp']
        self.eta = eta_0 * (1 + min(self.z, self.params["max_z"])) ** self.params['eta_exp']
        self.gamma = gamma_0 * (1 + min(self.z, self.params["max_z"])) ** self.params['gamma_exp']

        # # The normalisation only works with specific conditions
        # gamma > 0
        if self.gamma <= 0:
            if self.terminate:
                raise ValueError("gamma must be > 0, got " + str(self.gamma))
            else:
                self.gamma = 1e-3
        # eta >-0.5
        if self.eta <= -0.5:
            if self.terminate:
                raise ValueError("eta must be > -0.5, got " + str(self.eta))
            else:
                self.eta = -0.499
        # eta-phi >-0.5
        if self.eta - self.phi <= -0.5:
            if self.terminate:
                raise ValueError("eta-phi must be >-0.5, got " + str(self.eta - self.phi))
            else:
                self.phi = self.eta + 0.499
        if self.beta <= 0:
            if self.terminate:
                raise ValueError("beta must be > 0, got " + str(self.beta))
            else:
                self.beta = 1e-3

    @property
    def normalise(self):
        if self.delta_halo in self.delta_virs and self.z == 0:
            return self.params['alpha_%s' % (int(self.delta_halo))]
        else:
            return 1 / (2 ** (self.eta - self.phi - 0.5) * self.beta ** (-2 * self.phi) \
                      * self.gamma ** (-0.5 - self.eta) * (2 ** self.phi * self.beta ** (2 * self.phi)\
                      * Gam(self.eta + 0.5) + self.gamma ** self.phi * Gam(0.5 + self.eta - self.phi)))

    def fsigma(self, cut_fit):
        fv = (1 + (self.beta * self.nu) ** (-2 * self.phi)) * \
        self.nu ** (2 * self.eta) * np.exp(-self.gamma * (self.nu ** 2) / 2)

        fv *= self.normalise * self.nu

        if cut_fit:
            if self.z == 0.0:
                fv[np.logical_or(self.lnsigma / np.log(10) < -0.6 ,
                                  self.lnsigma / np.log(10) > 0.4)] = np.nan
            else:
                fv[np.logical_or(self.lnsigma / np.log(10) < -0.2 ,
                                  self.lnsigma / np.log(10) > 0.4)] = np.nan
        return fv

class Behroozi(Tinker10):
    def _modify_dndm(self, m, dndm, z, ngtm_tinker):
        a = 1 / (1 + z)
        theta = 0.144 / (1 + np.exp(14.79 * (a - 0.213))) * (m.value / 10 ** 11.5) ** (0.5 / (1 + np.exp(6.5 * a)))
        ngtm_behroozi = 10 ** (theta + np.log10(ngtm_tinker.value))
        dthetadM = 0.144 / (1 + np.exp(14.79 * (a - 0.213))) * \
            (0.5 / (1 + np.exp(6.5 * a))) * (m.value / 10 ** 11.5) ** \
            (0.5 / (1 + np.exp(6.5 * a)) - 1) / (10 ** 11.5)
        return (dndm.value * 10 ** theta - ngtm_behroozi * np.log(10) * dthetadM) * dndm.unit


class Tinker(Tinker08):
    pass
