import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import cosmolopy as cp
import sys

_allfits = ["ST", "SMT", 'Jenkins', "Warren", "Reed03", "Reed07", "Peacock",
            "Angulo", "AnguloBound", "Tinker", "Watson_FoF", "Watson", "Crocce",
            "Courtin", "Bhattacharya", "Behroozi", "Tinker08", "Tinker10"]

# TODO: check out units for boundaries (ie. whether they should be log or ln 1/sigma or M/h or M)

def get_fit(name, h):
    """
    A function that chooses the correct Profile class and returns it
    """
    try:
        return getattr(sys.modules[__name__], name)(h)
    except AttributeError:
        raise AttributeError(str(name) + "  is not a valid FittingFunction class")

class FittingFunction(object):
    """
    Calculates :math:`f(\sigma)` given a `MassFunction` instance.
    
    The class simplifies the choosing of the fitting function through a simple
    mapping of string identifiers.
    
    Parameters
    ----------
    hmf : `hmf.MassFunction` instance
        This object contains everything that is needed to 
        calculate :math:`f(\sigma)` -- the mass variance, redshift etc.
    
    """
    # This is a full list of available string identifiers. Aliases may also
    # be included here (eg. SMT and ST)
#     mf_fits = ["PS", "SMT", "ST", "Warren", "Jenkins", "Reed03", "Reed07", "Peacock",
#                "Angulo", "Angulo_Bound", "Tinker", "Watson_FoF", "Watson", "Crocce",
#                "Courtin", "Bhattacharya", "user_model", "Behroozi"]


    def __init__(self, hmf):
        self.hmf = hmf
        self.nu2 = self.hmf.nu
        self.nu = np.sqrt(self.hmf.nu)

    def fsigma(self, cut_fit):
        pass

class PS(FittingFunction):
    def fsigma(self, cut_fit):
        """
        Calculate :math:`f(\sigma)` for Press-Schechter form.

        Press, W. H., Schechter, P., 1974. ApJ 187, 425-438.
        http://adsabs.harvard.edu/full/1974ApJ...187..425P
        
        Returns
        -------
        vfv : array_like, len=len(pert.M)
            The function :math:`f(\sigma)\equiv\nu f(\nu)` defined on ``pert.M``
        """
        return np.sqrt(2.0 / np.pi) * self.nu * np.exp(-0.5 * self.nu2)

class ST(FittingFunction):
    def fsigma(self, cut_fit):
        """
        Calculate :math:`f(\sigma)` for Sheth-Mo-Tormen form.

        Sheth, R. K., Mo, H. J., Tormen, G., May 2001. MNRAS 323 (1), 1-12.
        http://doi.wiley.com/10.1046/j.1365-8711.2001.04006.x
        
        Returns
        -------
        vfv : array_like, len=len(pert.M)
            The function :math:`f(\sigma)\equiv\nu f(\nu)` defined on ``pert.M``
        """
        a = 0.707
        p = 0.3
        A = 0.3222

        vfv = A * np.sqrt(2.0 * a / np.pi) * self.nu * np.exp(-(a * self.nu2) / 2.0)\
                 * (1 + (1.0 / (a * self.nu2)) ** p)

        return vfv

class SMT(ST):
    pass

class Jenkins(FittingFunction):
    def fsigma(self, cut_fit):
        """
        Calculate :math:`f(\sigma)` for Jenkins form.

        Jenkins, A. R., et al., Feb. 2001. MNRAS 321 (2), 372-384.
        http://doi.wiley.com/10.1046/j.1365-8711.2001.04029.x
        
        .. note:: valid for :math: -1.2 < \ln \sigma^{-1} < 1.05
        
        Returns
        -------
        vfv : array_like, len=len(pert.M)
            The function :math:`f(\sigma)\equiv\nu f(\nu)` defined on ``pert.M``
        """

        vfv = 0.315 * np.exp(-np.abs(self.hmf.lnsigma + 0.61) ** 3.8)

        if cut_fit:
            vfv[np.logical_or(self.hmf.lnsigma < -1.2, self.hmf.lnsigma > 1.05)] = np.NaN

        return vfv

class Warren(FittingFunction):
    def fsigma(self, cut_fit):
        """
        Calculate :math:`f(\sigma)` for Warren form.

        Warren, M. S., et al., Aug. 2006. ApJ 646 (2), 881-885.
        http://adsabs.harvard.edu/abs/2006ApJ...646..881W
        
        .. note:: valid for :math:`10^{10}M_\odot < M <10^{15}M_\odot`
        
        Returns
        -------
        vfv : array_like, len=len(pert.M)
            The function :math:`f(\sigma)\equiv\nu f(\nu)` defined on ``pert.M``
        """

        vfv = 0.7234 * ((1.0 / self.hmf.sigma) ** 1.625 + 0.2538) * \
                np.exp(-1.1982 / self.hmf.sigma ** 2)

        if cut_fit:
            vfv[np.logical_or(self.hmf.M < 10 ** 10, self.hmf.M > 10 ** 15)] = np.NaN

        return vfv

class Reed03(ST):
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


        vfv *= np.exp(-0.7 / (self.hmf.sigma * np.cosh(2.0 * self.hmf.sigma) ** 5))

        if cut_fit:
            vfv[np.logical_or(self.hmf.lnsigma < -1.7, self.hmf.lnsigma > 0.9)] = np.NaN
        return vfv

class Reed07(FittingFunction):
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

        c = 1.08
        a = 0.764 / c
        A = 0.3222
        p = 0.3


        vfv = A * np.sqrt(2.0 * a / np.pi) * \
            (1.0 + (1.0 / (a * self.nu ** 2)) ** p + 0.6 * G_1 + 0.4 * G_2) * self.nu * \
            np.exp(-c * a * self.nu ** 2 / 2.0 - 0.03 * self.nu ** 0.6 / (self.hmf.n_eff + 3) ** 2)

        if cut_fit:
            vfv[np.logical_or(self.hmf.lnsigma < -0.5, self.hmf.lnsigma > 1.2)] = np.NaN

        return vfv

class Peacock(FittingFunction):
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
        a = 1.529
        b = 0.704
        c = 0.412

        d = 1 + a * self.nu ** b
        vfv = self.nu * np.exp(-c * self.nu2) * (2 * c * d * self.nu + b * a * self.nu ** (b - 1)) / d ** 2

        if cut_fit:
            vfv[np.logical_or(self.hmf.M < 10 ** 10, self.hmf.M > 10 ** 15)] = np.NaN

        return vfv

class Angulo(FittingFunction):
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

        vfv = 0.201 * ((2.08 / self.hmf.sigma) ** 1.7 + 1) * \
                np.exp(-1.172 / self.hmf.sigma ** 2)

        if cut_fit:
            vfv[np.logical_or(self.hmf.M < 10 ** 8, self.hmf.M > 10 ** 16)] = np.NaN
        return vfv

class AnguloBound(FittingFunction):
    def fsigma(self, cut_fit):
        """
        Calculate :math:`f(\sigma)` for Angulo (subhalo) form.

        Angulo, R. E., et al., 2012.
        arXiv:1203.3216v1
                
        .. note:: valid for :math:`10^{8}M_\odot < M <10^{16}M_\odot`
       
        Returns
        -------
        vfv : array_like, len=len(pert.M)
            The function :math:`f(\sigma)\equiv\nu f(\nu)` defined on ``pert.M``
        """
        vfv = 0.265 * ((1.675 / self.hmf.sigma) ** 1.9 + 1) * \
                np.exp(-1.4 / self.hmf.sigma ** 2)

        if cut_fit:
            vfv[np.logical_or(self.hmf.M < 10 ** 8, self.hmf.M > 10 ** 16)] = np.NaN
        return vfv

class Tinker08(FittingFunction):
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
        # The Tinker function is a bit tricky - we use the code from
        # http://cosmo.nyu.edu/~tinker/massfunction/MF_code.tar to aid us.
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

        b_array = np.array([2.571104e+00,
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

        A_0 = A_func(self.hmf.delta_halo)
        a_0 = a_func(self.hmf.delta_halo)
        b_0 = b_func(self.hmf.delta_halo)
        c_0 = c_func(self.hmf.delta_halo)

        A = A_0 * (1 + self.hmf.z) ** (-0.14)
        a = a_0 * (1 + self.hmf.z) ** (-0.06)
        alpha = 10 ** (-(0.75 / np.log10(self.hmf.delta_halo / 75)) ** 1.2)
        b = b_0 * (1 + self.hmf.z) ** (-alpha)
        c = c_0


        vfv = A * ((self.hmf.sigma / b) ** (-a) + 1) * np.exp(-c / self.hmf.sigma ** 2)

        if cut_fit:
            if self.hmf.z == 0.0:
                vfv[np.logical_or(self.hmf.lnsigma / np.log(10) < -0.6 ,
                                  self.hmf.lnsigma / np.log(10) > 0.4)] = np.nan
            else:
                vfv[np.logical_or(self.hmf.lnsigma / np.log(10) < -0.2 ,
                                  self.hmf.lnsigma / np.log(10) > 0.4)] = np.nan
        return vfv

class Watson_FoF(FittingFunction):
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
        vfv = 0.282 * ((1.406 / self.hmf.sigma) ** 2.163 + 1) * np.exp(-1.21 / self.hmf.sigma ** 2)
        if cut_fit:
            vfv[np.logical_or(self.hmf.lnsigma < -0.55 , self.hmf.lnsigma > 1.31)] = np.NaN
        return vfv

class Watson(FittingFunction):
    def gamma(self):
        """
        Calculate :math:`\Gamma` for the Watson fit.
        """
        C = np.exp(0.023 * (self.hmf.delta_halo / 178 - 1))
        d = -0.456 * cp.density.omega_M_z(self.hmf.z, **self.hmf.cosmolopy_dict) - 0.139
        p = 0.072
        q = 2.13

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
            A = 0.194
            alpha = 2.267
            beta = 1.805
            gamma = 1.287
        elif self.hmf.z > 6:
            A = 0.563
            alpha = 0.874
            beta = 3.810
            gamma = 1.453
        else:
            omz = cp.density.omega_M_z(self.hmf.z, **self.hmf.cosmolopy_dict)
            A = omz * (1.097 * (1 + self.hmf.z) ** (-3.216) + 0.074)
            alpha = omz * (3.136 * (1 + self.hmf.z) ** (-3.058) + 2.349)
            beta = omz * (5.907 * (1 + self.hmf.z) ** (-3.599) + 2.344)
            gamma = 1.318

        vfv = self.gamma() * A * ((beta / self.hmf.sigma) ** alpha + 1) * \
                 np.exp(-gamma / self.hmf.sigma ** 2)

        if cut_fit:
            vfv[np.logical_or(self.hmf.lnsigma < -0.55, self.hmf.lnsigma > 1.05)] = np.NaN

        return vfv

class Crocce(FittingFunction):
    def fsigma(self, cut_fit):
        """
        Calculate :math:`f(\sigma)` for Crocce form.

        Crocce, M., et al. MNRAS 403 (3), 1353-1367.
        http://doi.wiley.com/10.1111/j.1365-2966.2009.16194.x
                
        .. note:: valid for :math:`10^{10.5}M_\odot < M <10^{15.5}M_\odot`
       
        Returns
        -------
        vfv : array_like, len=len(pert.M)
            The function :math:`f(\sigma)\equiv\nu f(\nu)` defined on ``pert.M``
        """
        A = 0.58 * (1 + self.hmf.z) ** (-0.13)
        a = 1.37 * (1 + self.hmf.z) ** (-0.15)
        b = 0.3 * (1 + self.hmf.z) ** (-0.084)
        c = 1.036 * (1 + self.hmf.z) ** (-0.024)

        vfv = A * (self.hmf.sigma ** (-a) + b) * np.exp(-c / self.hmf.sigma ** 2)
        return vfv

class Courtin(FittingFunction):
    def fsigma(self, cut_fit):
        """
        Calculate :math:`f(\sigma)` for Courtin form.

        Courtin, J., et al., Oct. 2010. MNRAS 1931
        http://doi.wiley.com/10.1111/j.1365-2966.2010.17573.x
                
        .. note:: valid for :math:`-0.8<\ln\sigma^{-1}<0.7`
       
        Returns
        -------
        vfv : array_like, len=len(pert.M)
            The function :math:`f(\sigma)\equiv\nu f(\nu)` defined on ``pert.M``
        """
        A = 0.348
        a = 0.695
        p = 0.1
        # d_c = self.hmf.delta_c  # Note for WMAP5 they find delta_c = 1.673

        vfv = A * np.sqrt(2.0 * a / np.pi) * self.nu * np.exp(-(a * self.nu2) / 2.0)\
                 * (1 + (1.0 / (a * self.nu2)) ** p)
        return vfv

class Bhattacharya(FittingFunction):
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
        A = 0.333 * (1 + self.hmf.z) ** -0.11
        a = 0.788 * (1 + self.hmf.z) ** -0.01
        p = 0.807
        q = 1.795

        vfv = A * np.sqrt(2.0 / np.pi) * np.exp(-(a * self.nu ** 2) / 2.0) * \
                 (1 + (1.0 / (a * self.nu ** 2)) ** p) * (self.nu * np.sqrt(a)) ** q
        if cut_fit:
            vfv[np.logical_or(self.hmf.M < 6 * 10 ** 11,
                              self.hmf.M > 3 * 10 ** 15)] = np.NaN

        return vfv

class Behroozi(Tinker08):
    pass


class Tinker10(FittingFunction):
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
        from scipy.special import gamma as G

        delta_virs = np.array([200, 300, 400, 600, 800, 1200, 1600, 2400, 3200])

        alpha_array = np.array([ 0.368, 0.363, 0.385, 0.389, 0.393, 0.365, 0.379, 0.355, 0.327])
        beta_array = np.array([0.589, 0.585, 0.544, 0.543, 0.564, 0.623, 0.637, 0.673, 0.702])
        gamma_array = np.array([0.864, 0.922, 0.987, 1.09, 1.20, 1.34, 1.50, 1.68, 1.81])
        phi_array = np.array([-0.729, -0.789, -0.910, -1.05, -1.20, -1.26, -1.45, -1.50, -1.49])
        eta_array = np.array([-0.243, -0.261, -0.261, -0.273, -0.278, -0.301, -0.301, -0.319, -0.336])

        if self.hmf.delta_halo not in delta_virs:
            beta_func = spline(delta_virs, beta_array)
            gamma_func = spline(delta_virs, gamma_array)
            phi_func = spline(delta_virs, phi_array)
            eta_func = spline(delta_virs, eta_array)

            beta_0 = beta_func(self.hmf.delta_halo)
            gamma_0 = gamma_func(self.hmf.delta_halo)
            phi_0 = phi_func(self.hmf.delta_halo)
            eta_0 = eta_func(self.hmf.delta_halo)
        else:
            ind = np.where(delta_virs == self.hmf.delta_halo)[0][0]
            alpha_0 = alpha_array[ind]
            beta_0 = beta_array[ind]
            gamma_0 = gamma_array[ind]
            phi_0 = phi_array[ind]
            eta_0 = eta_array[ind]

        beta = beta_0 * (1 + min(self.hmf.z, 3)) ** 0.20
        phi = phi_0 * (1 + min(self.hmf.z, 3)) ** -0.08
        eta = eta_0 * (1 + min(self.hmf.z, 3)) ** 0.27
        gamma = gamma_0 * (1 + min(self.hmf.z, 3)) ** -0.01

#         print "Tinker10 params: ", beta_0, phi_0, eta_0, gamma_0
#         print "Tinker10 zparams: ", beta, phi, eta, gamma, 1 / (2 ** (eta - phi - 0.5) * beta ** (-2 * phi) * gamma ** (-0.5 - eta) \
#             * (2 ** phi * beta ** (2 * phi) * G(eta + 0.5) + gamma ** phi * G(0.5 + eta - phi)))

        fv = (1 + (beta * self.nu) ** (-2 * phi)) * self.nu ** (2 * eta) * np.exp(-gamma * (self.nu ** 2) / 2)

        # The following sets alpha (by \int f(\nu) d\nu = 1)
        if self.hmf.z > 0 or self.hmf.delta_halo not in delta_virs:
            fv /= 2 ** (eta - phi - 0.5) * beta ** (-2 * phi) * gamma ** (-0.5 - eta) \
            * (2 ** phi * beta ** (2 * phi) * G(eta + 0.5) + gamma ** phi * G(0.5 + eta - phi))
        else:
            fv /= alpha_0

        vfv = self.nu * fv

        if cut_fit:
            if self.hmf.z == 0.0:
                vfv[np.logical_or(self.hmf.lnsigma / np.log(10) < -0.6 ,
                                  self.hmf.lnsigma / np.log(10) > 0.4)] = np.nan
            else:
                vfv[np.logical_or(self.hmf.lnsigma / np.log(10) < -0.2 ,
                                  self.hmf.lnsigma / np.log(10) > 0.4)] = np.nan
        return vfv

class Tinker(Tinker08):
    pass
