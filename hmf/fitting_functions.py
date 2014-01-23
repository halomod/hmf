import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import cosmolopy as cp

# TODO: check out units for boundaries (ie. whether they should be log or ln 1/sigma or M/h or M)
class Fits(object):
    """
    Calculates :math:`f(\sigma)` given a `MassFunction` instance.
    
    The class simplifies the choosing of the fitting function through a simple
    mapping of string identifiers.
    
    Parameters
    ----------
    hmf : `hmf.MassFunction` instance
        This object contains everything that is needed to 
        calculate :math:`f(\sigma)` -- the mass variance, redshift etc.
        
    cut_fit : bool, optional, default ``True``
        Determines whether the function is cut at appropriate mass limits, 
        given by the respective publication for each fit. Though it is included
        in the `hmf` argument, one can specify it explicitly here for more 
        flexibility.
    
    """
    # This is a full list of available string identifiers. Aliases may also
    # be included here (eg. SMT and ST)
    mf_fits = ["PS", "SMT", "ST", "Warren", "Jenkins", "Reed03", "Reed07", "Peacock",
               "Angulo", "Angulo_Bound", "Tinker", "Watson_FoF", "Watson", "Crocce",
               "Courtin", "Bhattacharya", "user_model", "Behroozi"]

    def __init__(self, hmf, cut_fit=True):
        # We explicitly pass cut fit even though its in the Perturbations object
        # since it may be changed more flexibly.
        self.cut_fit = cut_fit
        self.pert = hmf
        self._cp = hmf.transfer.cosmo

    def nufnu(self):
        """
        Calculate and return :math:`f(\sigma,z)`.
        
        Internally this uses the string identifier to call an appropriate function.
        """
        if self.pert.mf_fit in Fits.mf_fits:
            return getattr(self, "_nufnu_" + self.pert.mf_fit)()

    def _nufnu_PS(self):
        """
        Calculate :math:`f(\sigma)` for Press-Schechter form.

        Press, W. H., Schechter, P., 1974. ApJ 187, 425-438.
        http://adsabs.harvard.edu/full/1974ApJ...187..425P
        
        Returns
        -------
        vfv : array_like, len=len(pert.M)
            The function :math:`f(\sigma)\equiv\nu f(\nu)` defined on ``pert.M``
        """

        vfv = np.sqrt(2.0 / np.pi) * (self._cp.delta_c / self.pert.sigma) * \
                np.exp(-0.5 * (self._cp.delta_c / self.pert.sigma) ** 2)

        return vfv

    def _nufnu_ST(self):
        """
        Calculate :math:`f(\sigma)` for Sheth-Mo-Tormen form.

        Sheth, R. K., Mo, H. J., Tormen, G., May 2001. MNRAS 323 (1), 1-12.
        http://doi.wiley.com/10.1046/j.1365-8711.2001.04006.x
        
        Returns
        -------
        vfv : array_like, len=len(pert.M)
            The function :math:`f(\sigma)\equiv\nu f(\nu)` defined on ``pert.M``
        """

        nu = self._cp.delta_c / self.pert.sigma
        a = 0.707

        vfv = 0.3222 * np.sqrt(2.0 * a / np.pi) * nu * np.exp(-(a * nu ** 2) / 2.0)\
                 * (1 + (1.0 / (a * nu ** 2)) ** 0.3)

        return vfv

    def _nufnu_SMT(self):
        """
        Calculate :math:`f(\sigma)` for Sheth-Mo-Tormen form.

        Sheth, R. K., Mo, H. J., Tormen, G., May 2001. MNRAS 323 (1), 1-12.
        http://doi.wiley.com/10.1046/j.1365-8711.2001.04006.x
        
        Returns
        -------
        vfv : array_like, len=len(pert.M)
            The function :math:`f(\sigma)\equiv\nu f(\nu)` defined on ``pert.M``
        """
        return self._nufnu_ST()

    def _nufnu_Jenkins(self):
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

        vfv = 0.315 * np.exp(-np.abs(self.pert.lnsigma + 0.61) ** 3.8)

        if self.cut_fit:
            vfv[np.logical_or(self.pert.lnsigma < -1.2, self.pert.lnsigma > 1.05)] = np.NaN

        return vfv

    def _nufnu_Warren(self):
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

        vfv = 0.7234 * ((1.0 / self.pert.sigma) ** 1.625 + 0.2538) * \
                np.exp(-1.1982 / self.pert.sigma ** 2)

        if self.cut_fit:
            vfv[np.logical_or(self.pert.M < 10 ** 10, self.pert.M > 10 ** 15)] = np.NaN
        return vfv

    def _nufnu_Reed03(self):
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

        ST_Fit = self._nufnu_ST()

        vfv = ST_Fit * np.exp(-0.7 / (self.pert.sigma * np.cosh(2.0 * self.pert.sigma) ** 5))

        if self.cut_fit:
            vfv[np.logical_or(self.pert.lnsigma < -1.7, self.pert.lnsigma > 0.9)] = np.NaN
        return vfv

    def _nufnu_Reed07(self):
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
        nu = self._cp.delta_c / self.pert.sigma

        G_1 = np.exp(-(self.pert.lnsigma - 0.4) ** 2 / (2 * 0.6 ** 2))
        G_2 = np.exp(-(self.pert.lnsigma - 0.75) ** 2 / (2 * 0.2 ** 2))

        c = 1.08
        a = 0.764 / c
        A = 0.3222
        p = 0.3


        vfv = A * np.sqrt(2.0 * a / np.pi) * \
            (1.0 + (1.0 / (a * nu ** 2)) ** p + 0.6 * G_1 + 0.4 * G_2) * nu * \
            np.exp(-c * a * nu ** 2 / 2.0 - 0.03 * nu ** 0.6 / (self.pert.n_eff + 3) ** 2)

        if self.cut_fit:
            vfv[np.logical_or(self.pert.lnsigma < -0.5, self.pert.lnsigma > 1.2)] = np.NaN

        return vfv

    def _nufnu_Peacock(self):
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
        nu = self._cp.delta_c / self.pert.sigma
        a = 1.529
        b = 0.704
        c = 0.412

        d = 1 + a * nu ** b
        vfv = nu * np.exp(-c * nu ** 2) * (2 * c * d * nu + b * a * nu ** (b - 1)) / d ** 2

        if self.cut_fit:
            vfv[np.logical_or(self.pert.M < 10 ** 10, self.pert.M > 10 ** 15)] = np.NaN

        return vfv

    def _nufnu_Angulo(self):
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

        vfv = 0.201 * ((2.08 / self.pert.sigma) ** 1.7 + 1) * \
                np.exp(-1.172 / self.pert.sigma ** 2)

        if self.cut_fit:
            vfv[np.logical_or(self.pert.M < 10 ** 8, self.pert.M > 10 ** 16)] = np.NaN
        return vfv

    def _nufnu_Angulo_Bound(self):
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
        vfv = 0.265 * ((1.675 / self.pert.sigma) ** 1.9 + 1) * \
                np.exp(-1.4 / self.pert.sigma ** 2)

        if self.cut_fit:
            vfv[np.logical_or(self.pert.M < 10 ** 8, self.pert.M > 10 ** 16)] = np.NaN
        return vfv

    def _nufnu_Tinker(self):
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

        A_0 = A_func(self.pert.delta_halo)
        a_0 = a_func(self.pert.delta_halo)
        b_0 = b_func(self.pert.delta_halo)
        c_0 = c_func(self.pert.delta_halo)

        A = A_0 * (1 + self.pert.transfer.z) ** (-0.14)
        a = a_0 * (1 + self.pert.transfer.z) ** (-0.06)
        alpha = 10 ** (-(0.75 / np.log10(self.pert.delta_halo / 75)) ** 1.2)
        b = b_0 * (1 + self.pert.transfer.z) ** (-alpha)
        c = c_0


        vfv = A * ((self.pert.sigma / b) ** (-a) + 1) * np.exp(-c / self.pert.sigma ** 2)

        if self.cut_fit:
            if self.pert.transfer.z == 0.0:
                vfv[np.logical_or(self.pert.lnsigma / np.log(10) < -0.6 ,
                                  self.pert.lnsigma / np.log(10) > 0.4)] = np.nan
            else:
                vfv[np.logical_or(self.pert.lnsigma / np.log(10) < -0.2 ,
                                  self.pert.lnsigma / np.log(10) > 0.4)] = np.nan
        return vfv

    def _watson_gamma(self):
        """
        Calculate :math:`\Gamma` for the Watson fit.
        """
        C = np.exp(0.023 * (self.pert.delta_halo / 178 - 1))
        d = -0.456 * cp.density.omega_M_z(self.pert.transfer.z, **self.pert.transfer.cosmo.cosmolopy_dict()) - 0.139
        p = 0.072
        q = 2.13

        return C * (self.pert.delta_halo / 178) ** d * np.exp(p * (1 - self.pert.delta_halo / 178) / self.pert.sigma ** q)


    def _nufnu_Watson_FoF(self):
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
        vfv = 0.282 * ((1.406 / self.pert.sigma) ** 2.163 + 1) * np.exp(-1.21 / self.pert.sigma ** 2)
        if self.cut_fit:
            vfv[np.logical_or(self.pert.lnsigma < -0.55 , self.pert.lnsigma > 1.31)] = np.NaN
        return vfv

    def _nufnu_Watson(self):
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
        if self.pert.transfer.z == 0:
            A = 0.194
            alpha = 2.267
            beta = 1.805
            gamma = 1.287
        elif self.pert.transfer.z > 6:
            A = 0.563
            alpha = 0.874
            beta = 3.810
            gamma = 1.453
        else:
            omz = cp.density.omega_M_z(self.pert.transfer.z, **self.pert.transfer.cosmo.cosmolopy_dict())
            A = omz * (1.097 * (1 + self.pert.transfer.z) ** (-3.216) + 0.074)
            alpha = omz * (3.136 * (1 + self.pert.transfer.z) ** (-3.058) + 2.349)
            beta = omz * (5.907 * (1 + self.pert.transfer.z) ** (-3.599) + 2.344)
            gamma = 1.318

        vfv = self._watson_gamma() * A * ((beta / self.pert.sigma) ** alpha + 1) * \
                 np.exp(-gamma / self.pert.sigma ** 2)

        if self.cut_fit:
            vfv[np.logical_or(self.pert.lnsigma < -0.55, self.pert.lnsigma > 1.05)] = np.NaN

        return vfv

    def _nufnu_Crocce(self):
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
        A = 0.58 * (1 + self.pert.transfer.z) ** (-0.13)
        a = 1.37 * (1 + self.pert.transfer.z) ** (-0.15)
        b = 0.3 * (1 + self.pert.transfer.z) ** (-0.084)
        c = 1.036 * (1 + self.pert.transfer.z) ** (-0.024)

        vfv = A * (self.pert.sigma ** (-a) + b) * np.exp(-c / self.pert.sigma ** 2)
        return vfv

    def _nufnu_Courtin(self):
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
        d_c = self._cp.delta_c  # Note for WMAP5 they find delta_c = 1.673

        vfv = A * np.sqrt(2 * a / np.pi) * (d_c / self.pert.sigma) * \
             (1 + (d_c / (self.pert.sigma * np.sqrt(a))) ** (-2 * p)) * \
             np.exp(-d_c ** 2 * a / (2 * self.pert.sigma ** 2))
        return vfv

    def _nufnu_Bhattacharya(self):
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
        A = 0.333 * (1 + self.pert.transfer.z) ** -0.11
        a = 0.788 * (1 + self.pert.transfer.z) ** -0.01
        p = 0.807
        q = 1.795

        nu = self._cp.delta_c / self.pert.sigma

        vfv = A * np.sqrt(2.0 / np.pi) * np.exp(-(a * nu ** 2) / 2.0) * \
                 (1 + (1.0 / (a * nu ** 2)) ** p) * (nu * np.sqrt(a)) ** q
        if self.cut_fit:
            vfv[np.logical_or(self.pert.M < 6 * 10 ** 11,
                              self.pert.M > 3 * 10 ** 15)] = np.NaN

        return vfv

    def _nufnu_Behroozi(self):
        return self._nufnu_Tinker()

    def _nufnu_user_model(self):
        """
        Calculates vfv based on a user-input model.
        """
        from scitools.StringFunction import StringFunction

        f = StringFunction(self.pert.user_fit, globals=globals())


        return f(self.sigma)
