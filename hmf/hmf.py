'''
This is the primary module for user-interaction with the :mod:`hmf` package.

The module contains a single class, `MassFunction`, which wraps almost all the
functionality of :mod:`hmf` in an easy-to-use way.
'''

version = '1.6.0'

###############################################################################
# Some Imports
###############################################################################
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import scipy.integrate as intg
import numpy as np
# from numpy import sin, cos, tan, abs, arctan, arccos, arcsin, exp
import copy
import logging
import cosmolopy as cp
import tools
from fitting_functions import Tinker08, get_fit
from transfer import Transfer
from _cache import parameter, cached_property
logger = logging.getLogger('hmf')






class MassFunction(Transfer):
    """
    An object containing all relevant quantities for the mass function.
    
    The purpose of this class is to calculate many quantities associated with 
    the dark matter halo mass function (HMF). The class is initialized to form a 
    cosmology and takes in various options as to how to calculate all
    further quantities. 
    
    All required outputs are provided as ``@property`` attributes for ease of 
    access.
    
    Contains an update() method which can be passed arguments to update, in the
    most optimal manner. All output quantities are calculated only when needed 
    (but stored after first calculation for quick access).
    
    Quantities related to the transfer function can be accessed through the 
    ``transfer`` property of this object.
    
    Parameters
    ----------   
    Mmin : float
        Minimum mass at which to perform analysis [units :math:`\log_{10}M_\odot h^{-1}`]. 
                    
    Mmax : float
        Maximum mass at which to perform analysis [units :math:`\log_{10}M_\odot h^{-1}`].
        
    dlog10m : float
        log10 interval between mass bins
        
    mf_fit : str or callable, optional, default ``"SMT"``
        A string indicating which fitting function to use for :math:`f(\sigma)`
                       
        Available options:
                                           
        1. ``'PS'``: Press-Schechter form from 1974
        #. ``'ST'``: Sheth-Mo-Tormen empirical fit 2001 (deprecated!)
        #. ``'SMT'``: Sheth-Mo-Tormen empirical fit from 2001
        #. ``'Jenkins'``: Jenkins empirical fit from 2001
        #. ``'Warren'``: Warren empirical fit from 2006
        #. ``'Reed03'``: Reed empirical from 2003
        #. ``'Reed07'``: Reed empirical from 2007
        #. ``'Tinker'``: Tinker empirical from 2008
        #. ``'Watson'``: Watson empirical 2012
        #. ``'Watson_FoF'``: Watson Friend-of-friend fit 2012
        #. ``'Crocce'``: Crocce 2010
        #. ``'Courtin'``: Courtin 2011
        #. ``'Angulo'``: Angulo 2012
        #. ``'Angulo_Bound'``: Angulo sub-halo function 2012
        #. ``'Bhattacharya'``: Bhattacharya empirical fit 2011
        #. ``'Behroozi'``: Behroozi extension to Tinker for high-z 2013

        Alternatively, one may define a callable function, with the signature
        ``func(self)``, where ``self`` is a :class:`MassFunction` object (and
        has access to all its attributes). This may be passed here. 
        
    delta_wrt : str, {``"mean"``, ``"crit"``}
        Defines what the overdensity of a halo is with respect to, mean density
        of the universe, or critical density.
                       
    delta_h : float, optional, default ``200.0``
        The overdensity for the halo definition, with respect to ``delta_wrt``
                       
    user_fit : str, optional, default ``""``
        A string defining a mathematical function in terms of `x`, used as
        the fitting function, where `x` is taken as :math:`\( \sigma \)`. Will only
        be applicable if ``mf_fit == "user_model"``.
                                       
    cut_fit : bool, optional, default ``True``
        Whether to forcibly cut :math:`f(\sigma)` at bounds in literature.
        If false, will use whole range of `M`.
           
    delta_c : float, default ``1.686``
        The critical overdensity for collapse, :math:`\delta_c`
        
    kwargs : keywords
        These keyword arguments are sent to the `hmf.transfer.Transfer` class.
        
        Included are all the cosmological parameters (see the docs for details).
        
    """


    def __init__(self, Mmin=10, Mmax=15, dlog10m=0.01, mf_fit=Tinker08, delta_h=200.0,
                 delta_wrt='mean', cut_fit=True, z2=None, nz=None,
                 delta_c=1.686, **transfer_kwargs):
        """
        Initializes some parameters      
        """
        # # Call super init MUST BE DONE FIRST.
        super(MassFunction, self).__init__(**transfer_kwargs)

        # Set all given parameters.
        self.mf_fit = mf_fit
        self.Mmin = Mmin
        self.Mmax = Mmax
        self.dlog10m = dlog10m
        self.delta_h = delta_h
        self.delta_wrt = delta_wrt
        self.cut_fit = cut_fit
        self.z2 = z2
        self.nz = nz
        self.delta_c = delta_c


    #===========================================================================
    # # --- PARAMETERS -------------------------------------------------------
    #===========================================================================
    @parameter
    def Mmin(self, val):
        return val

    @parameter
    def Mmax(self, val):
        return val

    @parameter
    def dlog10m(self, val):
        return val

    @parameter
    def delta_c(self, val):
        try:
            val = float(val)
        except ValueError:
            raise ValueError("delta_c must be a number: ", val)

        if val <= 0:
            raise ValueError("delta_c must be > 0 (", val, ")")
        if val > 10.0:
            raise ValueError("delta_c must be < 10.0 (", val, ")")

        return val

    @parameter
    def mf_fit(self, val):
        return val

    @parameter
    def delta_h(self, val):
        try:
            val = float(val)
        except ValueError:
            raise ValueError("delta_halo must be a number: ", val)

        if val <= 0:
            raise ValueError("delta_halo must be > 0 (", val, ")")
        if val > 10000:
            raise ValueError("delta_halo must be < 10,000 (", val, ")")

        return val

    @parameter
    def delta_wrt(self, val):
        if val not in ['mean', 'crit']:
            raise ValueError("delta_wrt must be either 'mean' or 'crit' (", val, ")")

        return val


    @parameter
    def z2(self, val):
        if val is None:
            return val

        try:
            val = float(val)
        except ValueError:
            raise ValueError("z must be a number (", val, ")")

        if val <= self.z:
            raise ValueError("z2 must be larger than z")
        else:
            return val

    @parameter
    def nz(self, val):
        if val is None:
            return val

        try:
            val = int(val)
        except ValueError:
            raise ValueError("nz must be an integer")

        if val < 1:
            raise ValueError("nz must be >= 1")
        else:
            return val

    @parameter
    def cut_fit(self, val):
        if not isinstance(val, bool):
            raise ValueError("cut_fit must be a bool, " + str(val))
        return val

    #--------------------------------  START NON-SET PROPERTIES ----------------------------------------------
    @cached_property("Mmin", "Mmax", "dlog10m")
    def M(self):
        return 10 ** np.arange(self.Mmin, self.Mmax, self.dlog10m)


    @cached_property("M", "lnk", "mean_dens")
    def kr_warning(self):
        return tools.check_kr(self.M[0], self.M[-1], self.mean_dens,
                              self.lnk[0], self.lnk[-1])

    @cached_property("delta_wrt", "delta_h", "z", "cosmolopy_dict")
    def delta_halo(self):
        """ Overdensity of a halo w.r.t mean density"""
        if self.delta_wrt == 'mean':
            return self.delta_h

        elif self.delta_wrt == 'crit':
            return self.delta_h / cp.density.omega_M_z(self.z, **self.cosmolopy_dict)

    @cached_property("M", "_lnP_0", "lnk", "mean_dens")
    def _sigma_0(self):
        """
        The normalised mass variance at z=0 :math:`\sigma`
        
        Notes
        -----
        
        .. math:: \sigma^2(R) = \frac{1}{2\pi^2}\int_0^\infty{k^2P(k)W^2(kR)dk}
        
        """
        return tools.mass_variance(self.M, self._lnP_0,
                                   self.lnk,
                                   self.mean_dens, "trapz")

    @cached_property("M", "_sigma_0", "_lnP_0", "lnk", "mean_dens")
    def _dlnsdlnm(self):
        """
        The value of :math:`\left|\frac{\d \ln \sigma}{\d \ln M}\right|`, ``len=len(M)``
        
        Notes
        -----
        
        .. math:: frac{d\ln\sigma}{d\ln M} = \frac{3}{2\sigma^2\pi^2R^4}\int_0^\infty \frac{dW^2(kR)}{dM}\frac{P(k)}{k^2}dk
        
        """
        return tools.dlnsdlnm(self.M, self._sigma_0, self._lnP_0,
                                             self.lnk,
                                             self.mean_dens)

    @cached_property("_sigma_0", "growth")
    def sigma(self):
        """
        The mass variance at `z`, ``len=len(M)``
        """
        return self._sigma_0 * self.growth

    @cached_property("sigma", "delta_c")
    def nu(self):
        """
        The parameter :math:`\nu = \left(\frac{\delta_c}{\sigma}\right)^2`, ``len=len(M)``
        """
        return (self.delta_c / self.sigma) ** 2

    @cached_property("sigma")
    def lnsigma(self):
        """
        Natural log of inverse mass variance, ``len=len(M)``
        """
        return np.log(1 / self.sigma)

    @cached_property("_dlnsdlnm")
    def n_eff(self):
        """
        Effective spectral index at scale of halo radius, ``len=len(M)``
        """
        return tools.n_eff(self._dlnsdlnm)

    @cached_property("mf_fit", "cut_fit", "sigma", "z", "delta_halo")
    def fsigma(self):
        """
        The multiplicity function, :math:`f(\sigma)`, for `mf_fit`. ``len=len(M)``
        """
        try:
            fit = self.mf_fit(self)
        except:
            fit = get_fit(self.mf_fit, self)

        fsigma = fit.fsigma(self.cut_fit)

        if np.sum(np.isnan(fsigma)) > 0.8 * len(fsigma):
            # the input mass range is almost completely outside the cut
            logger.warning("The specified mass-range was almost entirely \
                            outside of the limits from the fit. Ignored fit range...")
            fsigma = fit.fsigma(False)

        return fsigma

    @cached_property("z2", "fsigma", "mean_dens", "_dlnsdlnm", "M", "z",
                     "nz", "cosmolopy_dict")
    def dndm(self):
        """
        The number density of haloes, ``len=len(M)`` [units :math:`h^4 M_\odot^{-1} Mpc^{-3}`]
        """
        if self.z2 is None:  # #This is normally the case
            dndm = self.fsigma * self.mean_dens * np.abs(self._dlnsdlnm) / self.M ** 2
            if self.mf_fit == 'Behroozi' or type(self.mf_fit) == "Behroozi":
                a = 1 / (1 + self.z)
                theta = 0.144 / (1 + np.exp(14.79 * (a - 0.213))) * (self.M / 10 ** 11.5) ** (0.5 / (1 + np.exp(6.5 * a)))
                ngtm_tinker = self._ngtm()
                ngtm_behroozi = 10 ** (theta + np.log10(ngtm_tinker))
                dthetadM = 0.144 / (1 + np.exp(14.79 * (a - 0.213))) * \
                    (0.5 / (1 + np.exp(6.5 * a))) * (self.M / 10 ** 11.5) ** \
                    (0.5 / (1 + np.exp(6.5 * a)) - 1) / (10 ** 11.5)
                dndm = self.__dndm * 10 ** theta - ngtm_behroozi * np.log(10) * dthetadM
        else:  # #This is for a survey-volume weighted calculation
            if self.nz is None:
                self.nz = 10
            zedges = np.linspace(self.z, self.z2, self.nz)
            zcentres = (zedges[:-1] + zedges[1:]) / 2
            dndm = np.zeros_like(zcentres)
            vol = np.zeros_like(zedges)
            vol[0] = cp.distance.comoving_volume(self.z,
                                        **self.cosmolopy_dict)
            for i, zz in enumerate(zcentres):
                self.update(z=zz)
                dndm[i] = self.fsigma * self.mean_dens * np.abs(self._dlnsdlnm) / self.M ** 2
                if self.mf_fit == 'Behroozi' or type(self.mf_fit) == "Behroozi":
                    a = 1 / (1 + self.z)
                    theta = 0.144 / (1 + np.exp(14.79 * (a - 0.213))) * (self.M / 10 ** 11.5) ** (0.5 / (1 + np.exp(6.5 * a)))
                    ngtm_tinker = self._ngtm()
                    ngtm_behroozi = 10 ** (theta + np.log10(ngtm_tinker))
                    dthetadM = 0.144 / (1 + np.exp(14.79 * (a - 0.213))) * (0.5 / (1 + np.exp(6.5 * a))) * (self.M / 10 ** 11.5) ** (0.5 / (1 + np.exp(6.5 * a)) - 1) / (10 ** 11.5)
                    dndm[i] = dndm[i] * 10 ** theta - ngtm_behroozi * np.log(10) * dthetadM

                vol[i + 1] = cp.distance.comoving_volume(z=zedges[i + 1],
                                                **self.cosmolopy_dict)

            vol = vol[1:] - vol[:-1]  # Volume in shells
            integrand = vol * dndm
            numerator = intg.simps(integrand, x=zcentres)
            denom = intg.simps(vol, zcentres)
            dndm = numerator / denom
        return dndm

    @cached_property("M", "dndm")
    def dndlnm(self):
        """
        The differential mass function in terms of natural log of `M`, ``len=len(M)`` [units :math:`h^3 Mpc^{-3}`]
        """
        return self.M * self.dndm

    @cached_property("M", "dndm")
    def dndlog10m(self):
        """
        The differential mass function in terms of log of `M`, ``len=len(M)`` [units :math:`h^3 Mpc^{-3}`]
        """
        return self.M * self.dndm * np.log(10)

    def _upper_ngtm(self, M, mass_function, cut):
        """Calculate the mass function above given range of `M` in order to integrate"""
        # mass_function is logged already (not log10 though)

        if cut:  # since its been cut, the best we can do is a power law
            m_upper = np.linspace(np.log(M[-1]), np.log(10 ** 18), 500)
            mf_func = spline(np.log(M), mass_function, k=1)
            mf = mf_func(m_upper)
        else:
            # We try to calculate the hmf as far as we can normally
            new_pert = copy.deepcopy(self)
            new_pert.update(Mmin=np.log10(M[-1]),
                            Mmax=18)
            mf = np.log(new_pert.M * new_pert.dndm)

            if np.isnan(mf[-1]):  # Then we couldn't get up all the way, so have to do linear ext.
                if np.isnan(mf[1]):  # Then the whole extension is nan and we have to use the original (start at 1 because 1 val won't work either)
                    mf_func = spline(np.log(M), mass_function, k=1)
                    mf = mf_func(m_upper)
                else:
                    mfslice = mf[np.logical_not(np.isnan(mf))]
                    m_nan = np.log(new_pert.M[np.isnan(mf)])
                    m_true = np.log(new_pert.M[np.logical_not(np.isnan(mf))])
                    mf_func = spline(m_true, mfslice, k=1)
                    mf[len(mfslice):] = mf_func(m_nan)
            m_upper = np.log(new_pert.M)
        return m_upper, mf

    def _lower_ngtm(self, M, mass_function, cut):
        ### WE CALCULATE THE MASS FUNCTION BELOW THE COMPUTED RANGE ###
        # mass_function is logged already (not log10 though)
        if cut:  # since its been cut, the best we can do is a power law
            m_lower = np.linspace(np.log(10 ** 3), np.log(M[0]), 500)
            mf_func = spline(np.log(M), mass_function, k=1)
            mf = mf_func(m_lower)
        else:
            # We try to calculate the hmf as far as we can normally
            new_pert = copy.deepcopy(self)
            new_pert.update(Mmin=3,
                            Mmax=np.log10(M[0]))

            mf = np.log(new_pert.M * new_pert.dndm)

            if np.isnan(mf[0]):  # Then we couldn't go down all the way, so have to do linear ext.
                mfslice = mf[np.logical_not(np.isnan(mf))]
                m_nan = np.log(new_pert.M[np.isnan(mf)])
                m_true = np.log(new_pert.M[np.logical_not(np.isnan(mf))])
                mf_func = spline(m_true, mfslice, k=1)
                mf[:len(m_nan)] = mf_func(m_nan)
            m_lower = np.log(new_pert.M)
        return m_lower, mf

    def _ngtm(self):
        """
        Calculate n(>m).
        
        This function is separated from the property because of the Behroozi fit
        """
        # set M and mass_function within computed range
        M = self.M[np.logical_not(np.isnan(self.dndlnm))]
        mass_function = self.dndlnm[np.logical_not(np.isnan(self.dndlnm))]

        # Calculate the mass function (and its integral) from the highest M up to 10**18
        if M[-1] < 0.9 * 10 ** 18:
            m_upper, mf = self._upper_ngtm(M, np.log(mass_function), M[-1] < self.M[-1])

            int_upper = intg.simps(np.exp(mf), dx=m_upper[2] - m_upper[1], even='first')
        else:
            int_upper = 0

        # Calculate the cumulative integral (backwards) of mass_function (Adding on the upper integral)
        ngtm = np.concatenate((intg.cumtrapz(mass_function[::-1], dx=np.log(M[1]) - np.log(M[0]))[::-1], np.zeros(1))) + int_upper

        # We need to set ngtm back in the original length vector with nans where they were originally
        if len(ngtm) < len(self.M):
            ngtm_temp = np.zeros_like(self.dndlnm)
            ngtm_temp[:] = np.nan
            ngtm_temp[np.logical_not(np.isnan(self.dndlnm))] = ngtm
            ngtm = ngtm_temp

        return ngtm

    @cached_property("dndm")
    def ngtm(self):
        """
        The cumulative mass function above `M`, ``len=len(M)`` [units :math:`h^3 Mpc^{-3}`]
        """
        return self._ngtm()

    @cached_property("M", "dndlnm")
    def mgtm(self):
        """
        Mass in haloes `>M`, ``len=len(M)`` [units :math:`M_\odot h^2 Mpc^{-3}`]
        """
        M = self.M[np.logical_not(np.isnan(self.dndlnm))]
        mass_function = self.dndlnm[np.logical_not(np.isnan(self.dndlnm))]

        # Calculate the mass function (and its integral) from the highest M up to 10**18
        if M[-1] < 0.9 * 10 ** 18:
            m_upper, mf = self._upper_ngtm(M, np.log(mass_function), M[-1] < self.M[-1])
            int_upper = intg.simps(np.exp(mf + m_upper) , dx=m_upper[2] - m_upper[1], even='first')
        else:
            int_upper = 0

        # Calculate the cumulative integral (backwards) of mass_function (Adding on the upper integral)
        mgtm = np.concatenate((intg.cumtrapz(mass_function[::-1] * M[::-1], dx=np.log(M[1]) - np.log(M[0]))[::-1], np.zeros(1))) + int_upper

        # We need to set ngtm back in the original length vector with nans where they were originally
        if len(mgtm) < len(self.M):
            mgtm_temp = np.zeros_like(self.dndlnm)
            mgtm_temp[:] = np.nan
            mgtm_temp[np.logical_not(np.isnan(self.dndlnm))] = mgtm
            mgtm = mgtm_temp
        return mgtm

    @cached_property("M", 'dndlnm')
    def nltm(self):
        """
        Inverse cumulative mass function, ``len=len(M)`` [units :math:`h^3 Mpc^{-3}`]
        """
        # set M and mass_function within computed range
        M = self.M[np.logical_not(np.isnan(self.dndlnm))]
        mass_function = self.dndlnm[np.logical_not(np.isnan(self.dndlnm))]

        # Calculate the mass function (and its integral) from 10**3 up to lowest M
        if M[0] > 1.1 * 10 ** 3:
            m_lower, mf = self._lower_ngtm(M, np.log(mass_function), M[0] > self.M[0])

            int_lower = intg.simps(np.exp(mf), dx=m_lower[2] - m_lower[1], even='first')
        else:
            int_lower = 0

        # Calculate the cumulative integral of mass_function (Adding on the lower integral)
        nltm = np.concatenate((np.zeros(1), intg.cumtrapz(mass_function, dx=np.log(M[1]) - np.log(M[0])))) + int_lower

        # We need to set ngtm back in the original length vector with nans where they were originally
        if len(nltm) < len(self.M):
            nltm_temp = np.zeros_like(self.dndlnm)
            nltm_temp[:] = np.nan
            nltm_temp[np.logical_not(np.isnan(self.dndlnm))] = nltm
            nltm = nltm_temp

        return nltm

    @cached_property("M", 'dndm')
    def mltm(self):
        """
        Total mass in haloes `<M`, ``len=len(M)`` [units :math:`M_\odot h^2 Mpc^{-3}`]
        """
        # Set M within calculated range
        M = self.M[np.logical_not(np.isnan(self.dndlnm))]
        mass_function = self.dndlnm[np.logical_not(np.isnan(self.dndlnm))]

        # Calculate the mass function (and its integral) from 10**3 up to lowest M
        if M[0] > 1.1 * 10 ** 3:
            m_lower, mf = self._lower_ngtm(M, np.log(mass_function), M[0] > self.M[0])

            int_lower = intg.simps(np.exp(mf + m_lower), dx=m_lower[2] - m_lower[1], even='first')
        else:
            int_lower = 0

        # Calculate the cumulative integral of mass_function (Adding on the lower integral)
        mltm = np.concatenate((np.zeros(1), intg.cumtrapz(mass_function * M, dx=np.log(M[1]) - np.log(M[0])))) + int_lower

        # We need to set ngtm back in the original length vector with nans where they were originally
        if len(mltm) < len(self.M):
            nltm_temp = np.zeros_like(self.dndlnm)
            nltm_temp[:] = np.nan
            nltm_temp[np.logical_not(np.isnan(self.dndlnm))] = mltm
            mltm = nltm_temp

        return mltm

    @cached_property("ngtm")
    def how_big(self):
        """ 
        Size of simulation volume in which to expect one halo of mass M, ``len=len(M)`` [units :math:`Mpch^{-1}`]
        """

        return self.ngtm ** (-1. / 3.)
