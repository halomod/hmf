'''
This is the primary module for user-interaction with the `hmf` package.

The module contains a single class, `Perturbations`, which wraps almost all the
functionality of `hmf` in an easy-to-use way.
'''

version = '1.3.0'

###############################################################################
# Some Imports
###############################################################################
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import scipy.integrate as intg
import numpy as np
from numpy import sin, cos, tan, abs, arctan, arccos, arcsin, exp
import copy

# from scitools.std import sin,cos,tan,abs,arctan,arccos,arcsin #Must be in this form to work for some reason.
from cosmolopy import distance as cd
from cosmolopy import density as cden
# import cosmography
from cosmo import Cosmology
import tools
from fitting_functions import Fits

class Perturbations(object):
    r"""
    An object containing all relevant quantities for the mass function.
    
    The purpose of this class is to calculate many quantities associated with 
    perturbations in the early Universe. The class is initialized to form a 
    cosmology and takes in various options as to how to calculate all
    further quantities. 
    
    All required outputs are provided as ``@property`` attributes for ease of 
    access.
    
    Contains an update() method which can be passed arguments to update, in the
    most optimal manner. All output quantities are calculated only when needed 
    (but stored after first calculation for quick access).
    
    Parameters
    ----------
    M : array_like, optional, Default ``np.linspace(10,15,501)``
        The masses at which to perform analysis [units :math:`\log_{10}M_\odot h^{-1}`].    
        
        
    transfer_file : str, optional, default ``None``
        Either a string pointing to a file with a CAMB-produced transfer function,
        or ``None``. If ``None``, will use CAMB on the fly to produce the function.
                       
    z : float, optional, default ``0.0``
        The redshift of the analysis.
                   
    wdm_mass : float, optional, default ``None``
        The warm dark matter particle size in *keV*, or ``None`` for CDM.
                                                       
    k_bounds : sequence (``len=2``), optional, default ``[1e-8,2e4]``
        Defines the lower and upper limit of the wavenumber, *k* [units :math:`h^3Mpc^{-3}`]. 
        Used to truncate/extend the power spectrum. 
                     
    mf_fit : str, optional, default ``"SMT"``
        A string indicating which fitting function to use for :math:`f(\sigma)`
                       
        Available options:
                                           
        1. ``'PS'``: Press-Schechter form from 1974
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
        #. ``'user_model'``: A user-input string function
        
    delta_wrt : str, {``"mean"``,``"crit"``}
        Defines what the overdensity of a halo is with respect to.
        Can take ``'mean'`` or ``'crit'``
                       
    delta_halo : float, optional, default ``200.0``
        The overdensity for the halo definition, with respect to ``delta_wrt``
                       
    user_fit : str, optional, default ``""``
        A string defining a mathematical function in terms of `x`, used as
        the fitting function, where `x` is taken as :math:`\( \sigma \)`. Will only
        be applicable if ``mf_fit == "user_model"``.
                       
    transfer_fit : str, {``"CAMB"``, ``"EH"``} 
        Defines which transfer function fit to use. 
                    
    cut_fit : bool, optional, default ``True``
        Whether to forcibly cut :math:`f(\sigma)` at bounds given by respective papers.
        If `False`, will use whole range of `M`(may give ridiculous results).
           
           
    Other Parameters (for CAMB)
    ---------------------------
    Scalar_initial_condition : int, {1,2,3,4,5}
        Initial scalar perturbation mode (adiabatic=1, CDM iso=2, Baryon iso=3, 
        neutrino density iso =4, neutrino velocity iso = 5) 
        
    lAccuracyBoost : float, optional, default ``1.0``
        Larger to keep more terms in the hierarchy evolution
    
    AccuracyBoost : float, optional, default ``1.0``
        Increase accuracy_boost to decrease time steps, use more k values,  etc.
        Decrease to speed up at cost of worse accuracy. Suggest 0.8 to 3.
        
    w_perturb : bool, optional, default ``False``
        
    transfer__k_per_logint : int, optional, default ``11``
        Number of wavenumbers estimated per log interval by CAMB
        Default of 11 gets best performance for requisite accuracy of mass function.
        
    transfer__kmax : float, optional, default ``0.25``
        Maximum value of the wavenumber.
        Default of 0.25 is high enough for requisite accuracy of mass function.
        
    ThreadNum : int, optional, default ``0``
        Number of threads to use for calculation of transfer function by CAMB
        Default 0 automatically determines the number.
                       
    Available \*\*kwargs
    --------------------
    The ``**kwargs`` takes any cosmological parameters desired, which are input 
    to the `hmf.cosmo.Cosmology` class. `hmf.Perturbations` uses a default 
    parameter set from the first-year PLANCK mission, with optional modifications
    by the user.       
                 
    sigma_8 : default 0.8344
        The normalisation. Mass variance in top-hat spheres with :math:`R=8`Mpc:math:`h^{-1}`
        
    n : default 0.9624
        The spectral index
        
    w : default -1
        The dark-energy equation of state
        
    cs2_lam : default 1
        The constant comoving sound speed of dark energy
        
    t_cmb : default 2.725 
        Temperature of the CMB
        
    y_he : default 0.24 
        Helium fraction
        
        
    N_nu : default 3.04 
        Number of massless neutrino species
        
    N_nu_massive : default 0
        Number of massive neutrino species
        
    delta_c : default 1.686
        The critical overdensity for collapse
        
    h : default ``H0/100.0``
        The hubble parameter
        
    H0 : default 67.11
        The hubble constant
        
    omegan : default 0 
        The normalised density of neutrinos
        
    omegam : default ``(omegab_h2 + omegac_h2)/h**2`` 
        The normalised density of matter
        
    omegav : default 0.6825 
        The normalised density of dark energy
        
    omegab: default ``omegab_h2/h**2`` 
        The normalised baryon density

    omegac : default ``omegac_h2/h**2``
        The normalised CDM density
        
    omegab_h2 : default 0.022068 
        The normalised baryon density by ``h**2``
        
    omegac_h2 : default 0.12029 
        The normalised CDM density by ``h**2``
           
    force_flat : bool, default False
        Whether to force the cosmology to be flat (affects only ``omegav``)
    """


    def __init__(self, M=None, mf_fit="ST", transfer_file=None, z=0.0,
                 wdm_mass=None, k_bounds=[1e-8, 2e4], delta_halo=200.0,
                 delta_wrt='mean', user_fit='', transfer_fit='CAMB',
                 cut_fit=True, Scalar_initial_condition=1, lAccuracyBoost=1,
                 AccuracyBoost=1, w_perturb=False, transfer__k_per_logint=11,
                 transfer__kmax=0.25, ThreadNum=0, z2=None, nz=None, **kwargs):
        """
        Initializes the cosmology for which to perform the perturbation analysis.      
        """
        # A list of all available kwargs (these are sent to the Cosmology class)
        self._cp = ["sigma_8", "n", "w", "cs2_lam", "t_cmb", "y_he", "N_nu",
                    "omegan", "delta_c", "H0", "h", "omegab",
                    "omegac", "omegav", "omegab_h2", "omegac_h2",
                    "force_flat"]

        # Set up a simple dictionary of kwargs which can be later updated
        self._cpdict = {k:v for k, v in kwargs.iteritems() if k in self._cp}
        if M is None:
            M = np.linspace(10, 15, 501)

        self._camb_options = {'Scalar_initial_condition' : Scalar_initial_condition,
                              'scalar_amp'      : 1E-9,
                              'lAccuracyBoost' : lAccuracyBoost,
                              'AccuracyBoost'  : AccuracyBoost,
                              'w_perturb'      : w_perturb,
                              'transfer__k_per_logint': transfer__k_per_logint,
                              'transfer__kmax':transfer__kmax,
                              'ThreadNum':ThreadNum}


        # Set all given parameters -- annoying but has to be done for best updating
        self.M = M
        self.mf_fit = mf_fit
        self.k_bounds = k_bounds
        self._transfer_file = transfer_file
        self.wdm_mass = wdm_mass
        self._delta_halo_base = delta_halo
        self.delta_wrt = delta_wrt
        self.user_fit = user_fit
        self.z = z
        self.transfer_fit = transfer_fit
        self.cut_fit = cut_fit
        self.z2 = z2
        self.nz = nz
        self.cosmo_params = Cosmology(default="planck1_base", **self._cpdict)

    def update(self, **kwargs):
        """
        Update the class with the given arguments in an optimal manner.
        
        Accepts any argument that the constructor takes.
        """

        # First update the cosmology
        cp = {k:v for k, v in kwargs.iteritems() if k in self._cp}
        if cp:
            true_cp = {}
            for k, v in cp.iteritems():
                if k not in self._cpdict:
                    true_cp[k] = v
                elif k in self._cpdict:
                    if v != self._cpdict[k]:
                        true_cp[k] = v

            self._cpdict.update(true_cp)
            # Delete the entries we've used from kwargs
            for k in cp:
                del kwargs[k]

            # Now actually update the Cosmology class
            self.cosmo_params = Cosmology(default="planck1_base", **self._cpdict)
            if "n" in true_cp:
                del self._unnormalized_power
            if "sigma_8" in true_cp:
                del self._power_cdm_0
            if "delta_c" in true_cp:
                del self.fsigma

            for item in ["omegab", "omegac", "h", "H0", "omegab_h2", "omegac_h2"]:
                if item in true_cp:
                    del self._transfer_original

        # Now do rest of the parameters
        for key, val in kwargs.iteritems():

            if key in self._camb_options:
                if self._camb_options[key] != val:
                    self._camb_options.update({key:val})
                    if key != "ThreadNum":
                        del self._transfer_original

            elif key is 'M':
                if np.any(self.M != kwargs["M"]):
                    self.M = kwargs['M']
            elif key is "mf_fit":
                if self.mf_fit != kwargs["mf_fit"]:
                    self.mf_fit = kwargs["mf_fit"]
            elif key is "k_bounds":
                if self.k_bounds[0] != kwargs["k_bounds"][0] or self.k_bounds[1] != kwargs["k_bounds"][1]:
                    self.k_bounds = kwargs['k_bounds']
            elif key is "transfer_file":
                if self._transfer_file != kwargs['transfer_file']:
                    self._transfer_file = kwargs['transfer_file']
            elif key is 'z':
                if self.z != kwargs['z']:
                    self.z = kwargs['z']
            elif key is "wdm_mass":
                if self.wdm_mass != kwargs["wdm_mass"]:
                    self.wdm_mass = kwargs['wdm_mass']
            elif key is 'delta_halo':
                if self._delta_halo_base != kwargs['delta_halo']:
                    self._delta_halo_base = kwargs['delta_halo']
            elif key is 'delta_wrt':
                if self.delta_wrt != kwargs['delta_wrt']:
                    self.delta_wrt = kwargs['delta_wrt']
            elif key is "user_fit":
                if self.user_fit != kwargs['user_fit']:
                    self.user_fit = kwargs['user_fit']
            elif key is "transfer_fit":
                if self.transfer_fit != kwargs['transfer_fit']:
                    self.transfer_fit = kwargs['transfer_fit']
            elif key is "cut_fit":
                if self.cut_fit != kwargs['cut_fit']:
                    self.cut_fit = kwargs['cut_fit']
            elif key is "z2":  # must be set after z for comparison
                if self.z2 != kwargs['z2']:
                    self.z2 = kwargs["z2"]
            elif key is "nz":
                if self.nz != kwargs['nz']:
                    self.nz = kwargs["nz"]

            else:
                print "WARNING: ", key, " is not a valid parameter for the Perturbations class"

        # Some extra logic for deletes
        if ('omegab' in cp or 'omegac' in cp or 'omegav' in cp) and self.z > 0:
            del self.growth
        elif 'z' in kwargs:
            if kwargs['z'] == 0:
                del self.growth

    @property
    def M(self):
        """
        Mass vector [units of :math:`M_\odot h^{-1}`]
        """
        return self.__M

    @M.setter
    def M(self, val):
        try:
            if len(val) == 1:
                raise ValueError("M must be a sequence of length > 1")
        except TypeError:
            raise TypeError("M must be a sequence of length > 1")

        # Delete stuff dependent on it
        del self._sigma_0

        self.__M = 10 ** val

    @property
    def mf_fit(self):
        """
        String identifier of the fitting function used.
        """
        return self.__mf_fit

    @mf_fit.setter
    def mf_fit(self, val):

        try:
            val = str(val)
        except:
            raise ValueError("mf_fit must be a string, got ", val)

        if val not in Fits.mf_fits + ["Behroozi"]:
            raise ValueError("mf_fit is not in the list of available fitting functions: ", val)

        # Also delete stuff dependent on it
        del self.fsigma

        self.__mf_fit = val

    @property
    def k_bounds(self):
        """
        The bounds of the final transfer function in wavenumber *k* [units *h*/Mpc]
        """
        return self.__k_bounds

    @k_bounds.setter
    def k_bounds(self, val):
        try:
            if len(val) != 2:
                raise ValueError("k_bounds must be a sequence of length 2 (lower, upper)")
        except TypeError:
            raise TypeError("k_bounds must be a sequence of length 2 (lower, upper)")

        if val[0] > val[1]:
            raise ValueError("k_bounds must be in the form (lower, upper)")

        # We delete stuff directly dependent on it
        del self.lnk
        # Wrap the following in a try: except: because self.transfer_fit may not be set yet (at instantiation)
        try:
            if self.transfer_fit == "EH":
                del self._transfer_original
        except:
            pass


        self.__k_bounds = val

    @property
    def wdm_mass(self):
        """
        The WDM particle mass for a single-species model [units keV]
        """
        return self.__wdm_mass

    @wdm_mass.setter
    def wdm_mass(self, val):
        if val is None:
            self.__wdm_mass = val
            return
        try:
            val = float(val)
        except ValueError:
            raise ValueError("wdm_mass must be a number (", val, ")")

        if val <= 0:
            raise ValueError("wdm_mass must be > 0 (", val, ")")

        # Also delete stuff dependent on it
        del self._power_0

        self.__wdm_mass = val

    @property
    def _delta_halo_base(self):
        """
        The overdensity given by user: undefined until it is given with respect to something
        """
        return self.__delta_halo_base

    @_delta_halo_base.setter
    def _delta_halo_base(self, val):
        try:
            val = float(val)
        except ValueError:
            raise ValueError("delta_halo must be a number: ", val)

        if val <= 0:
            raise ValueError("delta_halo must be > 0 (", val, ")")
        if val > 10000:
            raise ValueError("delta_halo must be < 10,000 (", val, ")")

        self.__delta_halo_base = val

        # Delete stuff dependent on it
        del self.delta_halo

    @property
    def delta_wrt(self):
        """
        Defines what `delta_halo` is with respect to (``mean`` or ``crit``)
        """
        return self.__delta_wrt

    @delta_wrt.setter
    def delta_wrt(self, val):
        if val not in ['mean', 'crit']:
            raise ValueError("delta_wrt must be either 'mean' or 'crit' (", val, ")")

        self.__delta_wrt = val
        del self.delta_halo

    @property
    def z(self):
        """
        The redshift
        """
        return self.__z

    @z.setter
    def z(self, val):
        try:
            val = float(val)
        except ValueError:
            raise ValueError("z must be a number (", val, ")")

        if val < 0:
            raise ValueError("z must be > 0 (", val, ")")

        # Delete stuff dependent on it
        del self.growth
        self.__z = val

    @property
    def z2(self):
        """ Upper redshift for survey volume weighting """
        return self.__z2

    @z2.setter
    def z2(self, val):
        if val is None:
            self.__z2 = val
            return

        try:
            val = float(val)
        except ValueError:
            raise ValueError("z must be a number (", val, ")")

        if val <= self.z:
            raise ValueError("z2 must be larger than z")
        else:
            self.__z2 = val

    @property
    def nz(self):
        """ Number of redshift bins (if using survey volume weighting) """
        return self.__nz

    @nz.setter
    def nz(self, val):
        if val is None:
            self.__nz = val
            return

        try:
            val = int(val)
        except ValueError:
            raise ValueError("nz must be an integer")

        if val < 1:
            raise ValueError("nz must be >= 1")
        else:
            self.__nz = val

    @property
    def _transfer_file(self):
        """
        The path to the file where the transfer function is.
        
        If ``None``, triggers on-the-fly calculation by CAMB
        """
        return self.__transfer_file

    @_transfer_file.setter
    def _transfer_file(self, val):
        if val is None:
            self.__transfer_file = val
            return
        try:
            open(val)
        except IOError:
            raise IOError("The file " + val + " does not exist or cannot be opened")

        # Here we delete properties that are directly dependent on transfer_file
        del self._transfer_original

        self.__transfer_file = val

    @property
    def user_fit(self):
        """
        User-specified string equation for the fitting function.
        """
        return self.__user_fit

    @user_fit.setter
    def user_fit(self, val):
        self.__user_fit = val

        del self.fsigma

    @property
    def transfer_fit(self):
        """
        A string specifying a method with which to calculate the transfer function.
        """
        return self.__transfer_fit

    @transfer_fit.setter
    def transfer_fit(self, val):
        if val not in ["CAMB", "EH"]:
            raise ValueError("Sorry the transfer_fit has not been implemented for " + val + ". Please choose 'CAMB' or 'EH'")

        del self._transfer_original
        self.__transfer_fit = val

    @property
    def cut_fit(self):
        return self.__cut_fit

    @cut_fit.setter
    def cut_fit(self, val):
        if not isinstance(val, bool):
            raise ValueError("cut_fit must be a bool, " + str(val))

        del self.fsigma
        self.__cut_fit = val



    #--------------------------------  START NON-SET PROPERTIES ----------------------------------------------
    @property
    def delta_halo(self):
        """ Overdensity of a halo w.r.t mean density"""
        try:
            return self.__delta_halo
        except:
            if self.delta_wrt == 'mean':
                self.__delta_halo = self._delta_halo_base

            elif self.delta_wrt == 'crit':
                self.__delta_halo = self._delta_halo_base / cden.omega_M_z(self.z, **self.cosmo_params.cosmolopy_dict())
            return self.__delta_halo

    @delta_halo.deleter
    def delta_halo(self):
        try:
            del self.__delta_halo
            del self.fsigma
        except:
            pass

    @property
    def dlogM(self):
        """Logarithmic intervals between values of `M` (should be constant)"""
        return self.M[1] - self.M[0]

    @property
    def _transfer_original(self):
        """
        Original values of the transfer function from the call to CAMB/EH
        
        Length of matrix unknown prior to computation if CAMB used (0th index has len 2).
        If EH used, length is 250 and spans the k_bounds range.
        
        First column is wavenumber in units *h*/Mpc and second column is the
        transfer function.
        """
        try:
            return self.__transfer_original
        except AttributeError:
            self.__transfer_original = tools.get_transfer(self._transfer_file, self.cosmo_params,
                                                          self.transfer_fit, self._camb_options,
                                                          self.k_bounds)
            return self.__transfer_original

    @_transfer_original.deleter
    def _transfer_original(self):
        try:
            del self.__transfer_original
            del self._transfer_function_callable
        except:
            pass

    @property
    def _transfer_function_callable(self):
        """
        A callable linear interpolation of the transfer function
        """
        try:
            return self.__transfer_function_callable
        except:
            self.__transfer_function_callable = tools.interpolate_transfer(self._transfer_original[0, :],
                                                                           self._transfer_original[1, :])
            return self.__transfer_function_callable

    @_transfer_function_callable.deleter
    def _transfer_function_callable(self):
        try:
            del self.__transfer_function_callable
            del self.lnk
        except:
            pass

    @property
    def lnk(self):
        """
        The logarithmic wavenumbers for final transfer function [units *h*/Mpc]
        """
        try:
            return self.__lnk
        except:
            self.__lnk, dlnk = np.linspace(np.log(self.k_bounds[0]),
                                           np.log(self.k_bounds[1]),
                                           250, retstep=True)

            # CHECK KR_BOUNDS
            self.max_error, self.min_error = tools.check_kr(self.M[0], self.M[-1], self.cosmo_params.mean_dens,
                                                            np.exp(self.__lnk[0]), np.exp(self.__lnk[-1]))

            if self.max_error:
                print self.max_error
            if self.min_error:
                print self.min_error

            return self.__lnk

    @lnk.deleter
    def lnk(self):
        try:
            del self.__lnk
            del self._unnormalized_power
        except:
            pass

    @property
    def _unnormalized_power(self):
        """
        Un-normalized power spectrum at z=0 for CDM, ``len=len(lnk)`` [units Mpc:math:`^3/h^3`]
        """
        try:
            return self.__unnormalized_power
        except:
            if self.transfer_fit == "CAMB":
                self.__unnormalized_power = self.cosmo_params.n * self.lnk + 2.0 * \
                 self._transfer_function_callable(self.lnk) + np.log(2 * np.pi ** 2)
            else:
                self.__unnormalized_power = self.cosmo_params.n * self.lnk + 2.0 * \
                 self._transfer_original[1, :] + np.log(2 * np.pi ** 2)
            return self.__unnormalized_power

    @_unnormalized_power.deleter
    def _unnormalized_power(self):
        try:
            del self.__unnormalized_power
            del self._power_cdm_0
        except:
            pass

    @property
    def _power_cdm_0(self):
        """
        The normalised power spectrum at z=0 for CDM, ``len=len(lnk)`` [units Mpc:math:`^3/h^3`]
        """
        try:
            return self.__power_cdm_0
        except:
            self.__power_cdm_0, self._normalization = \
                tools.normalize(self.cosmo_params.sigma_8, self._unnormalized_power,
                                self.lnk, self.cosmo_params.mean_dens)
            return self.__power_cdm_0

    @_power_cdm_0.deleter
    def _power_cdm_0(self):
        try:
            del self.__power_cdm_0
            del self._power_0
        except:
            pass

    @property
    def _power_0(self):
        """
        The CDM/WDM power spectrum at z=0, ``len=len(lnk)`` [units Mpc:math:`^3/h^3`]
        """
        try:
            return self.__power_0
        except:
            if self.wdm_mass is not None:
                print "Doing wdm with mass = ", self.wdm_mass
                self.__power_0 = tools.wdm_transfer(self.wdm_mass, self._power_cdm_0, self.lnk,
                                                    self.cosmo_params.h, self.cosmo_params.omegac)
            else:
                self.__power_0 = self._power_cdm_0

            return self.__power_0

    @_power_0.deleter
    def _power_0(self):
        try:
            del self.__power_0
            del self._sigma_0
            del self.power
        except:
            pass

    @property
    def _sigma_0(self):
        """
        The normalised mass variance at z=0 :math:`\sigma`, ``len=len(M)``
        
        Notes
        -----
        
        .. math:: \sigma^2(R) = \frac{1}{2\pi^2}\int_0^\infty{k^2P(k)W^2(kR)dk}
        
        """

        try:
            return self.__sigma_0
        except:
            self.__sigma_0 = tools.mass_variance(self.M, self._power_0, self.lnk,
                                                 self.cosmo_params.mean_dens)
            return self.__sigma_0

    @_sigma_0.deleter
    def _sigma_0(self):
        try:
            del self.__sigma_0
            del self._dlnsdlnm
            del self.sigma
        except:
            pass

    @property
    def _dlnsdlnm(self):
        """
        The value of :math:`\left|\frac{\d \ln \sigma}{\d \ln M}\right|`, ``len=len(M)``
        
        Notes
        -----
        
        .. math:: frac{d\ln\sigma}{d\ln M} = \frac{3}{2\sigma^2\pi^2R^4}\int_0^\infty \frac{dW^2(kR)}{dM}\frac{P(k)}{k^2}dk
        
        """
        try:
            return self.__dlnsdlnm
        except:
            self.__dlnsdlnm = tools.dlnsdlnm(self.M, self._sigma_0, self._power_0,
                                             self.lnk, self.cosmo_params.mean_dens)
            return self.__dlnsdlnm

    @_dlnsdlnm.deleter
    def _dlnsdlnm(self):
        try:
            del self.__dlnsdlnm
            del self.dndm
            del self.n_eff
        except:
            pass

    @property
    def growth(self):
        r"""
        The growth factor :math:`d(z)`
        
        This is calculated (see Lukic 2007) as
        
        .. math:: d(z) = \frac{D^+(z)}{D^+(z=0)}
                
        where
        
        .. math:: D^+(z) = \frac{5\Omega_m}{2}\frac{H(z)}{H_0}\int_z^{\infty}{\frac{(1+z')dz'}{[H(z')/H_0]^3}}
        
        and
        
        .. math:: H(z) = H_0\sqrt{\Omega_m (1+z)^3 + (1-\Omega_m)}
        
        """
        try:
            return self.__growth
        except:
            if self.z > 0:
                self.__growth = tools.growth_factor(self.z, self.cosmo_params)
            else:
                self.__growth = 1
            return self.__growth

    @growth.deleter
    def growth(self):
        try:
            del self.__growth

            del self.power
            del self.sigma
        except:
            pass

    @property
    def power(self):
        """
        The fully realised power spectrum (with all parameters applied),``len=len(lnk)`` [units Mpc:math:`^3/h^3`]
        """
        try:
            return self.__power
        except:
            self.__power = 2 * np.log(self.growth) + self._power_0
            return self.__power
    @power.deleter
    def power(self):
        try:
            del self.__power
        except:
            pass

    @property
    def sigma(self):
        """
        The mass variance at `z`, ``len=len(M)``
        
        Redshift dependece simply enters by multplying `_sigma_0` by `growth`.
        """
        try:
            return self.__sigma
        except:
            self.__sigma = self._sigma_0 * self.growth
            return self.__sigma

    @sigma.deleter
    def sigma(self):
        try:
            del self.__sigma
            del self.fsigma
            del self.lnsigma
        except:
            pass

    @property
    def lnsigma(self):
        """
        Natural log of inverse mass variance, ``len=len(M)``
        """
        try:
            return self.__lnsigma
        except:
            self.__lnsigma = np.log(1 / self.sigma)
            return self.__lnsigma

    @lnsigma.deleter
    def lnsigma(self):
        try:
            del self.__lnsigma
            del self.fsigma
        except:
            pass

    @property
    def n_eff(self):
        """
        Effective spectral index at scale of halo radius, ``len=len(M)``
        """
        try:
            return self.__n_eff
        except:
            self.__n_eff = tools.n_eff(self._dlnsdlnm)
            return self.__n_eff

    @n_eff.deleter
    def n_eff(self):
        try:
            del self.__n_eff
        except:
            pass

    @property
    def fsigma(self):
        """
        The multiplicity function, :math:`f(\sigma)`, for `mf_fit`. ``len=len(M)``
        """
        try:
            return self.__fsigma
        except:
            fits_class = Fits(self, self.cut_fit)
            self.__fsigma = fits_class.nufnu()

            if np.sum(np.isnan(self.__fsigma)) > 0.8 * len(self.__fsigma):
                # the input mass range is almost completely outside the cut
                self.massrange_error = "The specified mass-range was almost entirely outside of the limits from the fit. Ignored fit range..."
                self.cut_fit = False
                fits_class.cut_fit = False
                self.__fsigma = fits_class.nufnu()

            return self.__fsigma
    @fsigma.deleter
    def fsigma(self):
        try:
            del self.__fsigma
            del self.dndm
        except:
            pass

    @property
    def dndm(self):
        """
        The number density of haloes, ``len=len(M)`` [units :math:`h^4 M_\odot^{-1} Mpc^{-3}`]
        """
        try:
            return self.__dndm
        except:
            if self.z2 is None:  # #This is normally the case
                self.__dndm = self.fsigma * self.cosmo_params.mean_dens * np.abs(self._dlnsdlnm) / self.M ** 2
                if self.mf_fit == 'Behroozi':
                    a = 1 / (1 + self.z)
                    theta = 0.144 / (1 + np.exp(14.79 * (a - 0.213))) * (self.M / 10 ** 11.5) ** (0.5 / (1 + np.exp(6.5 * a)))
                    ngtm_tinker = self._ngtm()
                    ngtm_behroozi = 10 ** (theta + np.log10(ngtm_tinker))
                    dthetadM = 0.144 / (1 + np.exp(14.79 * (a - 0.213))) * \
                        (0.5 / (1 + np.exp(6.5 * a))) * (self.M / 10 ** 11.5) ** \
                        (0.5 / (1 + np.exp(6.5 * a)) - 1) / (10 ** 11.5)
                    self.__dndm = self.__dndm * 10 ** theta - ngtm_behroozi * np.log(10) * dthetadM
            else:  # #This is for a survey-volume weighted calculation
                if self.nz is None:
                    self.nz = 10
                zedges = np.linspace(self.z, self.z2, self.nz)
                zcentres = (zedges[:-1] + zedges[1:]) / 2
                dndm = np.zeros_like(zcentres)
                vol = np.zeros_like(zedges)
                vol[0] = cd.comoving_volume(self.z,
                                            **self.cosmo_params.cosmolopy_dict())
                for i, zz in enumerate(zcentres):
                    self.update(z=zz)
                    dndm[i] = self.fsigma * self.cosmo_params.mean_dens * np.abs(self._dlnsdlnm) / self.M ** 2
                    if self.mf_fit == 'Behroozi':
                        a = 1 / (1 + self.z)
                        theta = 0.144 / (1 + np.exp(14.79 * (a - 0.213))) * (self.M / 10 ** 11.5) ** (0.5 / (1 + np.exp(6.5 * a)))
                        ngtm_tinker = self._ngtm()
                        ngtm_behroozi = 10 ** (theta + np.log10(ngtm_tinker))
                        dthetadM = 0.144 / (1 + np.exp(14.79 * (a - 0.213))) * (0.5 / (1 + np.exp(6.5 * a))) * (self.M / 10 ** 11.5) ** (0.5 / (1 + np.exp(6.5 * a)) - 1) / (10 ** 11.5)
                        dndm[i] = dndm[i] * 10 ** theta - ngtm_behroozi * np.log(10) * dthetadM

                    vol[i + 1] = cd.comoving_volume(z=zedges[i + 1],
                                                    **self.cosmo_params.cosmolopy_dict())

                vol = vol[1:] - vol[:-1]  # Volume in shells
                integrand = vol * dndm
                numerator = intg.simps(integrand, x=zcentres)
                denom = intg.simps(vol, zcentres)
                self.__dndm = numerator / denom
            return self.__dndm

    @dndm.deleter
    def dndm(self):
        try:
            del self.__dndm
            del self.dndlnm
            del self.dndlog10m
        except:
            pass


    @property
    def dndlnm(self):
        """
        The differential mass function in terms of natural log of `M`, ``len=len(M)`` [units :math:`h^3 Mpc^{-3}`]
        """
        try:
            return self.__dndlnm
        except:
            self.__dndlnm = self.M * self.dndm
            return self.__dndlnm

    @dndlnm.deleter
    def dndlnm(self):
        try:
            del self.__dndlnm
            del self.ngtm
            del self.nltm
            del self.mgtm
            del self.mltm
            del self.how_big
        except:
            pass

    @property
    def dndlog10m(self):
        """
        The differential mass function in terms of log of `M`, ``len=len(M)`` [units :math:`h^3 Mpc^{-3}`]
        """
        try:
            return self.__dndlog10m
        except:
            self.__dndlog10m = self.M * self.dndm * np.log(10)
            return self.__dndlog10m

    @dndlog10m.deleter
    def dndlog10m(self):
        try:
            del self.__dndlog10m
        except:
            pass

    def _upper_ngtm(self, M, mass_function, cut):
        """Calculate the mass function above given range of `M` in order to integrate"""
        # TODO: a massive tidy-up
        ### WE CALCULATE THE MASS FUNCTION ABOVE THE COMPUTED RANGE ###
        # mass_function is logged already (not log10 though)
        m_upper = np.linspace(np.log(M[-1]), np.log(10 ** 18), 500)
        if cut:  # since its been cut, the best we can do is a power law
            mf_func = spline(np.log(M), mass_function, k=1)
            mf = mf_func(m_upper)
        else:
            # We try to calculate the hmf as far as we can normally
            new_pert = copy.deepcopy(self)
            new_pert.update(M=np.log10(np.exp(m_upper)))
#             sigma_0 = tools.mass_variance(np.exp(m_upper), self._power_0, self.lnk, self.cosmo_params.mean_dens)
#             sigma = sigma_0 * self.growth
#             dlnsdlnm = tools.dlnsdlnm(np.exp(m_upper), sigma_0, self._power_0, self.lnk, self.cosmo_params.mean_dens)
#             n_eff = tools.n_eff(dlnsdlnm)
#             fsigma = fits(m_upper, n_eff, self.mf_fit, sigma, self.cosmo_params.delta_c,
#                           self.z, self.delta_halo, self.cosmo_params, self.user_fit, cut_fit=True).nufnu()()
#             # fsigma = nufnu()
#             dndm = fsigma * self.cosmo_params['mean_dens'] * np.abs(dlnsdlnm) / np.exp(m_upper) ** 2
            mf = np.log(np.exp(m_upper) * new_pert.dndm)

            if np.isnan(mf[-1]):  # Then we couldn't get up all the way, so have to do linear ext.
                if np.isnan(mf[1]):  # Then the whole extension is nan and we have to use the original (start at 1 because 1 val won't work either)
                    mf_func = spline(np.log(M), mass_function, k=1)
                    mf = mf_func(m_upper)
                else:
                    mfslice = mf[np.logical_not(np.isnan(mf))]
                    m_nan = m_upper[np.isnan(mf)]
                    m_true = m_upper[np.logical_not(np.isnan(mf))]
                    mf_func = spline(m_true, mfslice, k=1)
                    mf[len(mfslice):] = mf_func(m_nan)
        return m_upper, mf

    def _lower_ngtm(self, M, mass_function, cut):
        ### WE CALCULATE THE MASS FUNCTION BELOW THE COMPUTED RANGE ###
        # mass_function is logged already (not log10 though)
        m_lower = np.linspace(np.log(10 ** 3), np.log(M[0]), 500)
        if cut:  # since its been cut, the best we can do is a power law
            mf_func = spline(np.log(M), mass_function, k=1)
            mf = mf_func(m_lower)
        else:
            # We try to calculate the hmf as far as we can normally
            new_pert = copy.deepcopy(self)
            new_pert.update(M=np.log10(np.exp(m_lower)))
#             sigma_0 = tools.mass_variance(np.exp(m_lower), self._power_0, self.lnk, self.cosmo_params['mean_dens'])
#             sigma = sigma_0 * self.growth
#             dlnsdlnm = tools.dlnsdlnm(np.exp(m_lower), sigma_0, self._power_0, self.lnk, self.cosmo_params['mean_dens'])
#             n_eff = tools.n_eff(dlnsdlnm)
#             fsigma = fits(m_lower, n_eff, self.mf_fit, sigma, self.cosmo_params['delta_c'],
#                           self.z, self.delta_halo, self.cosmo_params, self.user_fit, cut_fit=True).nufnu()()
#             # fsigma = nufnu()
#             dndm = fsigma * self.cosmo_params['mean_dens'] * np.abs(dlnsdlnm) / np.exp(m_lower) ** 2
            mf = np.log(np.exp(m_lower) * new_pert.dndm)

            if np.isnan(mf[0]):  # Then we couldn't go down all the way, so have to do linear ext.
                mfslice = mf[np.logical_not(np.isnan(mf))]
                m_nan = m_lower[np.isnan(mf)]
                m_true = m_lower[np.logical_not(np.isnan(mf))]
                mf_func = spline(m_true, mfslice, k=1)
                mf[:len(mfslice)] = mf_func(m_nan)
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
        if M[-1] < 10 ** 18:
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

    @property
    def ngtm(self):
        """
        The cumulative mass function above `M`, ``len=len(M)`` [units :math:`h^3 Mpc^{-3}`]
        """
        try:
            return self.__ngtm
        except:
            self.__ngtm = self._ngtm()
            return self.__ngtm

    @ngtm.deleter
    def ngtm(self):
        try:
            del self.__ngtm
            del self.how_big
        except:
            pass

    @property
    def mgtm(self):
        """
        Mass in haloes >`M`, ``len=len(M)`` [units :math:`M_\odot h^2 Mpc^{-3}`]
        """
        try:
            return self.__mgtm
        except:
            M = self.M[np.logical_not(np.isnan(self.dndlnm))]
            mass_function = self.dndlnm[np.logical_not(np.isnan(self.dndlnm))]

            # Calculate the mass function (and its integral) from the highest M up to 10**18
            if M[-1] < 10 ** 18:
                m_upper, mf = self._upper_ngtm(M, np.log(mass_function), M[-1] < self.M[-1])
                int_upper = intg.simps(np.exp(mf + m_upper) , dx=m_upper[2] - m_upper[1], even='first')
            else:
                int_upper = 0

            # Calculate the cumulative integral (backwards) of mass_function (Adding on the upper integral)
            self.__mgtm = np.concatenate((intg.cumtrapz(mass_function[::-1] * M[::-1], dx=np.log(M[1]) - np.log(M[0]))[::-1], np.zeros(1))) + int_upper

            # We need to set ngtm back in the original length vector with nans where they were originally
            if len(self.__mgtm) < len(self.M):
                mgtm_temp = np.zeros_like(self.dndlnm)
                mgtm_temp[:] = np.nan
                mgtm_temp[np.logical_not(np.isnan(self.dndlnm))] = self.__mgtm
                self.__mgtm = mgtm_temp
            return self.__mgtm
    @mgtm.deleter
    def mgtm(self):
        try:
            del self.__mgtm
        except:
            pass

    @property
    def nltm(self):
        """
        Lower cumulative mass function, , ``len=len(M)`` [units :math:`h^3 Mpc^{-3}`]
        """
        try:
            return self.__nltm
        except:
            # set M and mass_function within computed range
            M = self.M[np.logical_not(np.isnan(self.dndlnm))]
            mass_function = self.dndlnm[np.logical_not(np.isnan(self.dndlnm))]

            # Calculate the mass function (and its integral) from 10**3 up to lowest M
            if M[0] > 10 ** 3:
                m_lower, mf = self._lower_ngtm(M, np.log(mass_function), M[0] > self.M[0])

                int_lower = intg.simps(np.exp(mf), dx=m_lower[2] - m_lower[1], even='first')
            else:
                int_lower = 0

            print "INT LOWER = ", int_lower
            # Calculate the cumulative integral of mass_function (Adding on the lower integral)
            self.__nltm = np.concatenate((intg.cumtrapz(mass_function, dx=np.log(M[1]) - np.log(M[0])), np.zeros(1))) + int_lower

            # We need to set ngtm back in the original length vector with nans where they were originally
            if len(self.__nltm) < len(self.M):
                nltm_temp = np.zeros_like(self.dndlnm)
                nltm_temp[:] = np.nan
                nltm_temp[np.logical_not(np.isnan(self.dndlnm))] = self.__nltm
                self.__nltm = nltm_temp

            return self.__nltm
    @nltm.deleter
    def nltm(self):
        try:
            del self.__nltm
        except:
            pass

    @property
    def mltm(self):
        """
        Total mass in haloes <`M`, ``len=len(M)`` [units :math:`M_\odot h^2 Mpc^{-3}`]
        """
        try:
            return self.__mltm
        except:
            # Set M within calculated range
            M = self.M[np.logical_not(np.isnan(self.dndlnm))]
            mass_function = self.dndlnm[np.logical_not(np.isnan(self.dndlnm))]

            # Calculate the mass function (and its integral) from 10**3 up to lowest M
            if M[0] > 10 ** 3:
                m_lower, mf = self._lower_ngtm(M, np.log(mass_function), M[0] > self.M[0])

                int_lower = intg.simps(np.exp(mf + m_lower), dx=m_lower[2] - m_lower[1], even='first')
            else:
                int_lower = 0

            # Calculate the cumulative integral of mass_function (Adding on the upper integral)
            self.__mltm = np.concatenate((intg.cumtrapz(mass_function * M, dx=np.log(M[1]) - np.log(M[0])), np.zeros(1))) + int_lower

            # We need to set ngtm back in the original length vector with nans where they were originally
            if len(self.__mltm) < len(self.M):
                nltm_temp = np.zeros_like(self.dndlnm)
                nltm_temp[:] = np.nan
                nltm_temp[np.logical_not(np.isnan(self.dndlnm))] = self.__mltm
                self.__mltm = nltm_temp

            return self.__mltm

    @property
    def how_big(self):
        """ 
        Size of simulation volume in which to expect one halo of mass M, ``len=len(M)`` [units :math:`Mpch^{-1}`]
        """

        return self.ngtm ** (-1. / 3.)

    @how_big.deleter
    def how_big(self):
        try:
            del self.how_big
        except:
            pass




