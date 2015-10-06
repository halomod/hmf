'''
This is the primary module for user-interaction with the :mod:`hmf` package.

The module contains a single class, `MassFunction`, which wraps almost all the
functionality of :mod:`hmf` in an easy-to-use way.
'''

version = '2.0.0'

###############################################################################
# Some Imports
###############################################################################
import numpy as np
import copy
import logging
from fitting_functions import Tinker08, Behroozi
from transfer import Transfer
from _cache import parameter, cached_property
from integrate_hmf import hmf_integral_gtm
from fitting_functions import FittingFunction
from numpy import issubclass_
logger = logging.getLogger('hmf')
from filters import TopHat, Filter
from _framework import get_model
from scipy.optimize import minimize
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import warnings

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
                 delta_wrt='mean', cut_fit=True, z2=None, nz=None, fsig_params=None,
                 delta_c=1.686, filter=TopHat, filter_params=None,
                 **transfer_kwargs):
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
        self.fsig_params = fsig_params or {}
        self.filter = filter
        self.filter_params = filter_params or {}

    #===========================================================================
    # PARAMETERS
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
    def filter(self, val):
        if not issubclass_(val, Filter) and not isinstance(val, basestring):
            raise ValueError("filter must be a Filter or string, got %s" % type(val))
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
        if not issubclass_(val, FittingFunction) and not isinstance(val, basestring):
            raise ValueError("mf_fit must be a FittingFunction or string, got %s" % type(val))
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
    def filter_params(self, val):
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

    @parameter
    def fsig_params(self, val):
        if not isinstance(val, dict):
            raise ValueError("fsig_params must be a dictionary")
        return val

    #--------------------------------  PROPERTIES ------------------------------
    @cached_property("z", "mean_density0")
    def mean_density(self):
        """
        Mean density of universe at redshift z
        """
        return self.mean_density0 * (1 + self.z) ** 3

    @cached_property("mf_fit", "sigma", "z", "delta_halo", "nu", "M", "fsig_params",
                     "cosmo", "delta_c")
    def _fit(self):
        """The actual fitting function class (as opposed to string identifier)"""
        if issubclass_(self.mf_fit, FittingFunction):
            fit = self.mf_fit(M=self.M, nu2=self.nu, z=self.z,
                              delta_halo=self.delta_halo, omegam_z=self.cosmo.Om(self.z),
                              delta_c=self.delta_c, sigma=self.sigma, n_eff=self.n_eff,
                              ** self.fsig_params)
        elif isinstance(self.mf_fit, basestring):
            fit = get_model(self.mf_fit, "hmf.fitting_functions",
                            M=self.M, nu2=self.nu, z=self.z,
                            delta_halo=self.delta_halo, omegam_z=self.cosmo.Om(self.z),
                            delta_c=self.delta_c, sigma=self.sigma, n_eff=self.n_eff,
                            ** self.fsig_params)
        return fit

    @cached_property("mean_density0", "filter", "delta_c", "k", "_unnormalised_power", "filter_params")
    def filter_mod(self):

        if issubclass_(self.filter, Filter):
            filter = self.filter(self.k,self._unnormalised_power, **self.filter_params)
        elif isinstance(self.filter, basestring):
            filter = get_model(self.filter, "hmf.filters", k=self.k,
                                power=self._unnormalised_power, **self.filter_params)

        return filter

    @cached_property("Mmin", "Mmax", "dlog10m")
    def M(self):
        return (10 ** np.arange(self.Mmin, self.Mmax, self.dlog10m))


#     @cached_property("M", "lnk", "mean_density0")
#     def kr_warning(self):
#         return tools.check_kr(self.M[0], self.M[-1], self.mean_density0,
#                               self.lnk[0], self.lnk[-1])

    @cached_property("delta_wrt", "delta_h", "z", "cosmo")
    def delta_halo(self):
        """ Overdensity of a halo w.r.t mean density"""
        if self.delta_wrt == 'mean':
            return self.delta_h

        elif self.delta_wrt == 'crit':
            return self.delta_h / self.cosmo.Om(self.z)

    @cached_property("radii", "filter_mod")
    def _unn_sigma0(self):
        return self.filter_mod.sigma(self.radii)

    @cached_property("_normalisation", "_unn_sigma0")
    def _sigma_0(self):
        """
        The normalised mass variance at z=0 :math:`\sigma`

        Notes
        -----

        .. math:: \sigma^2(R) = \frac{1}{2\pi^2}\int_0^\infty{k^2P(k)W^2(kR)dk}

        """
        return self._normalisation * self._unn_sigma0

    @cached_property("M", "filter_mod")
    def radii(self):
        """
        The radii corresponding to the masses `M`
        """
        return self.filter_mod.mass_to_radius(self.M,self.mean_density)

    @cached_property("radii", "filter_mod")
    def _dlnsdlnm(self):
        """
        The value of :math:`\left|\frac{\d \ln \sigma}{\d \ln M}\right|`, ``len=len(M)``

        Notes
        -----

        .. math:: frac{d\ln\sigma}{d\ln M} = \frac{3}{2\sigma^2\pi^2R^4}\int_0^\infty \frac{dW^2(kR)}{dM}\frac{P(k)}{k^2}dk

        """
        return 0.5 * self.filter_mod.dlnss_dlnm(self.radii)

    @cached_property("_sigma_0", "growth_factor")
    def sigma(self):
        """
        The mass variance at `z`, ``len=len(M)``
        """
        return self._sigma_0 * self.growth_factor

    @cached_property("sigma", "delta_c")
    def nu(self):
        """
        The parameter :math:`\nu = \left(\frac{\delta_c}{\sigma}\right)^2`, ``len=len(M)``
        """
        return (self.delta_c / self.sigma) ** 2

    @cached_property("nu")
    def mass_nonlinear(self):
        """
        The nonlinear mass, nu(M_\star) = 1.
        """
        if self.nu.min() >1 or self.nu.max()<1:
            warnings.warn("Nonlinear mass outside mass range")
            if self.nu.min() > 1:
                startr = np.log(self.radii.min())
            else:
                startr = np.log(self.radii.max())

            model = lambda lnr : (self.filter_mod.sigma(np.exp(lnr)*self.radii)*self._normalisation*self.growth_factor - self.delta_c)**2

            res = minimize(model,[startr,])

            if res.success:
                r = np.exp(res.x[0])
                return self.filter_mod.radius_to_mass(r)
            else:
                warnings.warn("Minimization failed :(")
                return 0
        else:
            nu = spline(self.nu,self.M,k=5)
            return nu(1)

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

        Notes
        -----
        This function, and any derived quantities, can show small non-physical
        'wiggles' at the 0.1% level, if too coarse a grid in ln(k) is used. If
        applications are sensitive at this level, please use a very fine k-space
        grid.

        Uses eq. 42 in Lukic et. al 2007.
        """
        return -3.0 * (2.0 * self._dlnsdlnm + 1.0)

    @cached_property("_fit", "cut_fit", "sigma", "z", "delta_halo", "nu", "M")
    def fsigma(self):
        """
        The multiplicity function, :math:`f(\sigma)`, for `mf_fit`. ``len=len(M)``
        """
        fsigma = self._fit.fsigma(self.cut_fit)

        if np.sum(np.logical_not(np.isnan(fsigma))) < 2:
            # the input mass range is almost completely outside the cut
            logger.warning("The specified mass-range was almost entirely \
                            outside of the limits from the fit. Ignored fit range...")
            fsigma = self._fit.fsigma(False)

        return fsigma

    @cached_property("z2", "fsigma", "mean_density0", "_dlnsdlnm", "M", "z", "nz")
    def dndm(self):
        """
        The number density of haloes, ``len=len(M)`` [units :math:`h^4 M_\odot^{-1} Mpc^{-3}`]
        """
        if self.z2 is None:  # #This is normally the case
            dndm = self.fsigma * self.mean_density0 * np.abs(self._dlnsdlnm) / self.M ** 2
            if isinstance(self._fit, Behroozi):
                ngtm_tinker = self._gtm(dndm)
                dndm = self._fit._modify_dndm(self.M, dndm, self.z, ngtm_tinker)

        else:  # #This is for a survey-volume weighted calculation
            raise NotImplementedError()
#             if self.nz is None:
#                 self.nz = 10
#             zedges = np.linspace(self.z, self.z2, self.nz)
#             zcentres = (zedges[:-1] + zedges[1:]) / 2
#             dndm = np.zeros_like(zcentres)
#             vol = np.zeros_like(zedges)
#             vol[0] = cp.distance.comoving_volume(self.z,
#                                         **self.cosmolopy_dict)
#             for i, zz in enumerate(zcentres):
#                 self.update(z=zz)
#                 dndm[i] = self.fsigma * self.mean_dens * np.abs(self._dlnsdlnm) / self.M ** 2
#                 if isinstance(self.mf_fit, "Behroozi"):
#                     ngtm_tinker = self._gtm(dndm[i])
#                     dndm[i] = self.mf_fit._modify_dndm(self.M, dndm[i], self.z, ngtm_tinker)
#
#                 vol[i + 1] = cp.distance.comoving_volume(z=zedges[i + 1],
#                                                 **self.cosmolopy_dict)
#
#             vol = vol[1:] - vol[:-1]  # Volume in shells
#             integrand = vol * dndm[i]
#             numerator = intg.simps(integrand, x=zcentres)
#             denom = intg.simps(vol, zcentres)
#             dndm = numerator / denom

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

    def _gtm(self, dndm, mass_density=False):
        """
        Calculate number or mass density above mass thresholds in ``self.M``

        This function is here, separate from the properties, due to its need
        of being passed ``dndm` in the case of the Behroozi fit only, in which
        case an infinite recursion would occur otherwise.

        Parameters
        ----------
        dndm : array_like, ``len(self.M)``
            Should usually just be exactly ``self.dndm``, except in Behroozi fit.

        mass_density : bool, ``False``
            Whether to get the mass density, or number density.
        """
        # Get required local variables
        size = len(dndm)
        m = self.M
        # If the highest mass is very low, we try calculating it to higher masses
        # The dlog10m is NOT CHANGED, so the input needs to be finely spaced.
        # If the top value of dndm is NaN, don't try calculating higher masses.
        if m[-1] < 10 ** 16.5 and not np.isnan(dndm[-1]):
            # Behroozi function won't work here.
            if isinstance(self._fit, Behroozi):
                pass
            else:
                new_mf = copy.deepcopy(self)
                new_mf.update(Mmin=np.log10(self.M[-1]) + self.dlog10m, Mmax=18)
                dndm = np.concatenate((dndm, new_mf.dndm))

                m = np.concatenate((m, new_mf.M))

        ngtm = hmf_integral_gtm(m, dndm, mass_density)

        # We need to set ngtm back in the original length vector with nans where they were originally
        if len(ngtm) < len(m):  # Will happen if some dndlnm are NaN
            ngtm_temp = np.zeros(len(dndm))
            ngtm_temp[:] = np.nan
            ngtm_temp[np.logical_not(np.isnan(dndm))] = ngtm
            ngtm = ngtm_temp

        # Since ngtm may have been extended, we cut it back
        return ngtm[:size]

    @cached_property("M", "dndm")
    def ngtm(self):
        """
        The cumulative mass function above `M`, ``len=len(M)`` [units :math:`h^3 Mpc^{-3}`]

        In the case that `M` does not extend to sufficiently high masses, this
        routine will auto-generate ``dndm`` for an extended mass range. If
        ``cut_fit`` is True, and this extension is invalid, then a power-law fit
        is applied to extrapolate to sufficient mass.

        In the case of the Behroozi fit, it is impossible to auto-extend the mass
        range except by the power-law fit, thus one should be careful to supply
        appropriate mass ranges in this case.
        """
        return self._gtm(self.dndm)

    @cached_property("M", "dndm")
    def rho_gtm(self):
        """
        Mass density in haloes `>M`, ``len=len(M)`` [units :math:`M_\odot h^2 Mpc^{-3}`]

        In the case that `M` does not extend to sufficiently high masses, this
        routine will auto-generate ``dndm`` for an extended mass range. If
        ``cut_fit`` is True, and this extension is invalid, then a power-law fit
        is applied to extrapolate to sufficient mass.

        In the case of the Behroozi fit, it is impossible to auto-extend the mass
        range except by the power-law fit, thus one should be careful to supply
        appropriate mass ranges in this case.
        """
        return self._gtm(self.dndm, mass_density=True)


    @cached_property("mean_density0", 'rho_gtm')
    def rho_ltm(self):
        """
        Mass density in haloes `<M`, ``len=len(M)`` [units :math:`M_\odot h^2 Mpc^{-3}`]

        .. note :: As of v1.6.2, this assumes that the entire mass density of
                   halos is encoded by the ``mean_density`` parameter (ie. all
                   mass is found in halos). This is not explicitly true of all
                   fitting functions (eg. Warren), in which case the definition
                   of this property is somewhat inconsistent, but will still
                   work.

        In the case that `M` does not extend to sufficiently high masses, this
        routine will auto-generate ``dndm`` for an extended mass range. If
        ``cut_fit`` is True, and this extension is invalid, then a power-law fit
        is applied to extrapolate to sufficient mass.

        In the case of the Behroozi fit, it is impossible to auto-extend the mass
        range except by the power-law fit, thus one should be careful to supply
        appropriate mass ranges in this case.
        """
        return self.mean_density0 - self.rho_gtm


    @cached_property("ngtm")
    def how_big(self):
        """
        Size of simulation volume in which to expect one halo of mass M, ``len=len(M)`` [units :math:`Mpch^{-1}`]
        """

        return self.ngtm ** (-1. / 3.)
