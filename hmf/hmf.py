'''
Perturbations.py contains a single class (Perturbations), which contains
methods that act upon a transfer function to gain functions such as the
mass function.
'''

version = '1.1.9'

###############################################################################
# Some Imports
###############################################################################
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import scipy.integrate as intg
import numpy as np
from numpy import sin, cos, tan, abs, arctan, arccos, arcsin, exp

# from scitools.std import sin,cos,tan,abs,arctan,arccos,arcsin #Must be in this form to work for some reason.

import cosmography
import tools
from fitting_functions import fits

#TODO: possibly implement weighting by survey volume: n(M,z1,z2) = \int n(M,z)*V(z) dz / \int V(z) dz
###############################################################################
# The Class
###############################################################################
class Perturbations(object):
    """
    A class which contains the functions necessary to evaluate the HMF and Mass Variance.
    
    The purpose and idea behind the class is to be able to calculate many quantities associated
    with perturbations in the early Universe. The class is initialized to form a cosmology and
    takes in various options as to how to calculate the various quantities. Most non-initial 
    methods of the class need no further input after the cosmology has been initialized.
    
    Contains an update() method which can be passed arguments to update, in the most optimal manner.
    All output quantities are called as properties of the class, and are calculated at need (but
    stored after first calculation for quick access).
    
    Output Quantities:
        sigma:         The mass variance of spheres of the given radius/mass. NOTE: not sigma^2
        lnsigma:       The natural logarithm of the inverse of sigma
        lnk:           The natural log of the wavenumbers in the power spectrum [h/Mpc]
        growth:        The growth factor for the given cosmology and redshift
        power:         The natural log of the normalised power spectrum for the given cosmology (at lnk)
        n_eff:         Effective spectral index at the radius of a halo
        M:             The masses at which analysis is performed. (not log) [M_sun/h]
        fsigma:        The multiplicity function, or fitting function at M
        dndm:          The comoving number density of halos in mass interval M [h**3/Mpc**3]
        dndlnm:        The comoving number density of halos in log mass interval M [h**3/Mpc**3]
        dndlog10m:     The comoving number density of halo in log10 mass interval M [h**3/Mpc**3]
        ngtm:          Comoving number density of halos > M [h**3/Mpc**3]
        nltm:          Comoving number density of halos < M [h**3/Mpc**3]
        mgtm:          Comoving mass density of halos > M [M_sun h**3/Mpc**3]
        mltm:          Comoving mass density of halos < M [M_sun h**3/Mpc**3]
        how_big:       The requisite size of a simulation box, L, to have at least one halo > M [Mpc/h]
        
    Input:
        M:             A vector of floats containing the log10(Solar Masses/h) at which to perform analysis.    
                       Default: M = np.linspace(10,15,501)
                       
        transfer_file: Either a string pointing to a file with a CAMB-produced transfer function,
                       or None. If None, will use CAMB on the fly to produce the function.
                       Default is None.
                       
        z:             a float giving the redshift of the analysis.
                       Default z = 0.0
                   
        wdm_mass:      a float giving warm dark matter particle size in keV, or None for CDM. 
                       Default is wdm_mass = None
                                                       
        k_bounds:      a list/tuple defining two values: the lower and upper limit of k. Used to truncate/extend
                       the power spectrum. 
                       Default k_bounds = [0.0000001,20000.0]
                     
        mf_fit:        A string indicating which fitting function to use for f(sigma)
                       Default: mf_fit = 'ST'
                       
                       Options:                               
                            1. 'PS': Press-Schechter form from 1974
                            2. 'ST': Sheth-Mo-Tormen empirical fit from 2001
                            3. 'Jenkins': Jenkins empirical fit from 2001
                            4. 'Warren': Warren empirical fit from 2006
                            5. 'Reed03': Reed empirical from 2003
                            6. 'Reed07': Reed empirical from 2007
                            7. 'Tinker': Tinker empirical from 2008
                            8. 'Watson': Watson empirical 2012
                            9. 'Watson_FoF': Watson Friend-of-friend fit 2012
                            10. 'Crocce': Crocce 2010
                            11. 'Courtin': Courtin 2011
                            12. 'Angulo': Angulo 2012
                            13. 'Angulo_Bound': Angulo sub-halo function 2012
                            14. "Bhattacharya": Bhattacharya empirical fit 2011
                            15. "Behroozi": Behroozi extension to Tinker for high-z 2013
                            16. 'user_model': A user-input string function
        
        delta_wrt:     Defines what the overdensity of a halo is with respect to, can take 'mean' or 'crit'
                       Default: 'mean' 
                       
        delta_halo:    The overdensity for the halo definition, with respect to delta_wrt
                       Default: delta_halo = 200.0
                       
        user_fit:      A string defining a mathematical function in terms of 'x', used as the fitting function,
                       where x is taken as sigma. Will only be applicable if mf_fit == "user_model".
                       Default: user_fit = ""
                       
        transfer_fit:  A string defining which transfer function fit to use. Current options are 'CAMB' and 'EH' (Eistenstein-Hu)
                       Default: transfer_fit = "CAMB"
                       
        cut_fit:       Whether to forcibly cut the f(sigma) at bounds given by respective papers.
                       If False, will use function to calculate all values specified in M (may give ridiculous results)
                       Default: True
                       
                       
        **kwargs:      There is a placeholder for any additional cosmological parameters, or camb
                       parameters, that one wishes to include. Parameters that aren't used won't
                       break the program, they will just be ignored. Here follows a list of parameters
                       that will be used by various parts of the program, and their respective defaults,
                       note that cosmological parameters follow PLANCK1 results:
                       
                       PARAMETERS USED OUTSIDE OF CAMB ONLY:
                       sigma_8         :: 0.8347
                       n               :: 0.9619
                       delta_c         :: 1.686
                       
                       PARAMETERS USED IN CAMB AND OUTSIDE:
                       omegab          :: 0.049
                       omegac          :: 0.2678
                       omegav          :: 0.6817
                       omegak          :: 1 - omegab - omegac - omegal - omegan
                       
                       PARAMETERS USED ONLY IN CAMB
                       H0              :: 67.04
                       omegan          :: 0.0
                       TCMB            :: 2.725
                       yhe             :: 0.24
                       Num_Nu_massless :: 3.04 
                       Num_Nu_massive  :: 0
                       reion__redshift :: 10.3 
                       reion__optical_depth :: 0.09
                       reion__fraction :: -1
                       reion__delta_redshift :: 1.5
                       Scalar_initial_condition :: 1
                       scalar_amp      :: 1E-9
                       scalar_running  :: 0
                       tensor_index    :: 0
                       tensor_ratio    :: 1
                       lAccuracyBoost :: 1
                       lSampleBoost   :: 1
                       w_lam          :: -1
                       cs2_lam        :: 0
                       AccuracyBoost  :: 1
                       WantScalars     :: True
                       WantTensors     :: False
                       reion__reionization :: True
                       reion__use_optical_depth :: True
                       w_perturb      :: False
                       DoLensing       :: False 
                       ThreadNum       :: 0 
                       transfer__k_per_logint :: 11
                       transfer__kmax :: 0.25
                                

    """


    def __init__(self, M=np.linspace(10, 15, 501),
                 mf_fit="ST",
                 transfer_file=None,
                 z=0.0,
                 wdm_mass=None, k_bounds=[0.0000001, 20000.0],
                 delta_halo=200.0, delta_wrt='mean', user_fit='', transfer_fit='CAMB',
                 cut_fit=True, ** kwargs):
        """
        Initializes the cosmology for which to perform the perturbation analysis.      
        """
        #All cosmological parameters that affect the transfer function (CAMB)
        #Extra derivative parameters are omegak, reion__optical_depth or reion__redshift
        self._transfer_cosmo = {"w_lam"    :-1,
                               "omegab"   : 0.0455,
                               "omegac"   : 0.226,
                               "omegav"   : 0.728,
                               "omegan"   : 0.0,
                               "H0"       : 70.4,
                               'cs2_lam' : 1,
                               'TCMB'     : 2.725,
                               'yhe'      : 0.24,
                               'Num_Nu_massless' : 3.04,
                               'reion__redshift': 10.3,
                               'reion__optical_depth': 0.085
                               }

        #All other cosmological parameters, derivative parameters are mean_dens and omegam
        self._extra_cosmo = {"sigma_8":0.81,
                            "n":0.967,
                            "delta_c":1.686,
                            "crit_dens":27.755e10
                            }

        self._transfer_options = {'Num_Nu_massive'  : 0,
                                 'reion__fraction' :-1,
                                 'reion__delta_redshift' : 1.5,
                                 'Scalar_initial_condition' : 1,
                                 'scalar_amp'      : 1E-9,
                                 'scalar_running'  : 0,
                                 'tensor_index'    : 0,
                                 'tensor_ratio'    : 1,
                                 'lAccuracyBoost' : 1,
                                 'lSampleBoost'   : 1,
                                 'AccuracyBoost'  : 1,
                                 'WantScalars'     : True,
                                 'WantTensors'     : False,
                                 'reion__reionization' : True,
                                 'reion__use_optical_depth' : True,
                                 'w_perturb'      : False,
                                 'DoLensing'       : False,
                                 'transfer__k_per_logint': 11,
                                 'transfer__kmax':0.25,
                                 'ThreadNum':0}


        #A list of available HMF fitting functions and their identifiers
        self.mf_fits = ["PS", "ST", "SMT", "Warren", "Jenkins", "Reed03", "Reed07", "Angulo", "Angulo_Bound", "Tinker",
                        "Watson_FoF", "Watson", "Crocce", "Courtin", "Bhattacharya", "Behroozi", "user_model",
                        "Peacock"]

        self.update(M=M, mf_fit=mf_fit, k_bounds=k_bounds, transfer_file=transfer_file, wdm_mass=wdm_mass,
                     delta_halo=delta_halo, delta_wrt=delta_wrt, user_fit=user_fit, z=z,
                     transfer_fit=transfer_fit, cut_fit=cut_fit, ** kwargs)

    def update(self, **kwargs):
        """
        This is a convenience method to update any variable that is meant to be in the class in an optimal manner
        """

        #First update the basic dictionaries
        for key, val in kwargs.iteritems():
            if key in self._transfer_cosmo:
                self._transfer_cosmo.update({key:val})
                del self.cosmo_params
                del self.camb_params

                if key is 'omegab':
                    del self._power_cdm_0
                    del self._sigma_0
                    del self._dlnsdlnm
                    del self.dndm
                elif key is 'omegac':
                    del self._power_cdm_0
                    del self._power_0
                    del self._sigma_0
                    del self._dlnsdlnm
                    del self.dndm
                elif key is 'H0':
                    del self._power_0

            elif key in self._extra_cosmo:
                self._extra_cosmo.update({key:val})
                del self.cosmo_params
                if key is 'n':
                    del self._unnormalized_power
                elif key is 'sigma_8':
                    del self._power_cdm_0
                elif key is 'crit_dens':
                    del self._power_cdm_0
                    del self._sigma_0
                    del self._dlnsdlnm
                    del self.dndm
                elif key is 'delta_c':
                    del self.fsigma
            elif key in self._transfer_options:
                self._transfer_options.update({key:val})
                del self.camb_params

            elif key is 'M':
                self.M = kwargs['M']
            elif key is "mf_fit":
                self.mf_fit = kwargs["mf_fit"]
            elif key is "k_bounds":
                self.k_bounds = kwargs['k_bounds']
            elif key is "transfer_file":
                self._transfer_file = kwargs['transfer_file']
            elif key is 'z':
                self.z = kwargs['z']
            elif key is "wdm_mass":
                self.wdm_mass = kwargs['wdm_mass']
            elif key is 'delta_halo':
                self._delta_halo_base = kwargs['delta_halo']
            elif key is 'delta_wrt':
                self.delta_wrt = kwargs['delta_wrt']
            elif key is "user_fit":
                self.user_fit = kwargs['user_fit']
            elif key is "transfer_fit":
                self.transfer_fit = kwargs['transfer_fit']
            elif key is "cut_fit":
                self.cut_fit = kwargs['cut_fit']
            elif key is "R":
                self.R = kwargs['R']

            else:
                print "WARNING: ", key, " is not a valid parameter for the Perturbations class"

        #Some extra logic for deletes
        if ('omegab' in kwargs or 'omegac' in kwargs or 'omegav' in kwargs) and self.z > 0:
            del self.growth
        elif 'z' in kwargs:
            if kwargs['z'] == 0:
                del self.growth


    @property
    def cosmo_params(self):
        """All the cosmological parameters"""
        try:
            return self.__cosmo_params
        except:
            #Piece together the cosmo dictionaries
            self.__cosmo_params = dict(self._transfer_cosmo.items() + self._extra_cosmo.items())

            #Set some derivative parameters
            self.__cosmo_params['omegam'] = self.__cosmo_params['omegab'] + self.__cosmo_params['omegac']
            self.__cosmo_params['omegak'] = 1 - self.__cosmo_params['omegam'] - self.__cosmo_params['omegav'] - self.__cosmo_params['omegan']
            self.__cosmo_params['crit_dens'] = 2.7755E7 * self.__cosmo_params["H0"] ** 2
            self.__cosmo_params['mean_dens'] = self.__cosmo_params['omegam'] * self.__cosmo_params['crit_dens']  #FIXME redshift evolution?
            if self._transfer_options['reion__use_optical_depth']:
                del self.__cosmo_params['reion__redshift']
            else:
                del self.__cosmo_params['reion__optical_depth']

            return self.__cosmo_params
    @cosmo_params.deleter
    def cosmo_params(self):
        try:
            del self.__cosmo_params
        except AttributeError:
            pass
    @property
    def camb_params(self):
        """Every parameter that is passed to camb"""

        try:
            return self.__camb_params
        except:
            #Piece together the cosmo dictionaries
            self.__camb_params = dict(self._transfer_cosmo.items() + self._transfer_options.items())

            #Set some derivative parameters
            self.__camb_params['omegak'] = 1 - self.__camb_params['omegab'] - self.__camb_params['omegac'] - \
             self.__camb_params['omegav'] - self.__camb_params['omegan']

            if self.__camb_params['reion__use_optical_depth']:
                del self.__camb_params['reion__redshift']
            else:
                del self.__camb_params['reion__optical_depth']

            if self.__camb_params['transfer__kmax'] < 0.2:
                print "WARNING: transfer__kmax may be too low for accuracy"

            if self.__camb_params['transfer__k_per_logint'] < 11 and self.__camb_params['transfer__k_per_logint'] != 0:
                print "WARNING: transfer__k_per_logint may be too low for accuracy"
            return self.__camb_params

    @camb_params.deleter
    def camb_params(self):
        try:
            del self.__camb_params

            #We also delete stuff that DIRECTLY depends on it
            del self._lnk_original
        except AttributeError:
            pass

    @property
    def M(self):
        """
        Mass in units of M_sun/h
        """
        return self.__M

    @M.setter
    def M(self, val):
        try:
            if len(val) == 1:
                raise ValueError("M must be a sequence of length > 1")
        except TypeError:
            raise TypeError("M must be a sequence of length > 1")

        #Delete stuff dependent on it
        del self._sigma_0
        del self._dlnsdlnm
        del self.dndm
        del self.dndlnm
        del self.dndlog10m

        self.__M = 10 ** val

    @property
    def mf_fit(self):
        """
        The mass function fitting function
        """
        return self.__mf_fit

    @mf_fit.setter
    def mf_fit(self, val):

        try:
            val = str(val)
        except:
            raise ValueError("mf_fit must be a string, got ", val)

        if val not in self.mf_fits:
            raise ValueError("mf_fit is not in the list of available fitting functions: ", val)

        #Also delete stuff dependent on it
        del self.fsigma

        self.__mf_fit = val

    @property
    def k_bounds(self):
        """
        The fourier-space limits of the transfer function in wavenumber k
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

        #We delete stuff directly dependent on it
        del self.lnk
        #Wrap the following in a try: except: because self.transfer_fit may not be set yet (at instantiation)
        try:
            if self.transfer_fit == "EH":
                del self._transfer_original
        except:
            pass


        self.__k_bounds = val

    @property
    def wdm_mass(self):
        """
        The WDM particle mass for a single-species model
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

        #Also delete stuff dependent on it
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

        #Set the value

        self.__delta_halo_base = val

        #Delete stuff dependent on it
        del self.delta_halo

    @property
    def delta_wrt(self):
        """
        Defines what the delta_halo is with respect to
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

        #Delete stuff dependent on it
        del self.growth
        self.__z = val

    @property
    def _transfer_file(self):
        """
        The path to the file where the transfer function is.
        If None, triggers on-the-fly calculation by CAMB
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

        #Here we delete properties that are directly dependent on transfer_file
        del self._transfer_original

        self.__transfer_file = val

    @property
    def user_fit(self):
        """
        A string specifying a mathematical function in terms of x, counted as the mass variance, which defines a fitting function
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
        
        Currenty implemented are 'CAMB' (Code for anisotropies in the Microwave Background) and 'EH' (Eisenstein-Hu fit)
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
        """ Overdensity of a halo wrt mean density"""
        try:
            return self.__delta_halo
        except:
            if self.delta_wrt == 'mean':
                self.__delta_halo = self._delta_halo_base

            elif self.delta_wrt == 'crit':
                self.__delta_halo = self._delta_halo_base / cosmography.omegam_z(self.z, self.cosmo_params['omegam'], self.cosmo_params['omegav'],
                                                                                self.cosmo_params['omegak'])
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
        return self.M[1] - self.M[0]

    @property
    def _transfer_original(self):
        """
        Original values of lnk from the call to the transfer function
        """
        try:
            return self.__transfer_original
        except:
            self.__transfer_original = tools.get_transfer(self._transfer_file, self.camb_params,
                                                          self.transfer_fit, self.k_bounds)
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
        Returns a callable function which is the interpolation of the transfer function
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

            #Also delete things directly dependent on it (or other self variables defined in it)
            del self.lnk
        except:
            pass

    @property
    def lnkh(self):
        """
        The logarithmic bins in k-space - this is k/h (ie h/Mpc)
        """
        try:
            return self.__lnkh
        except:
            if self.transfer_fit == "CAMB":
                self.__lnkh, dlnk = tools.new_k_grid(self._transfer_original[0, :], self.k_bounds)
            elif self.transfer_fit == "EH":
                self.__lnkh = self._transfer_original[0, :]

            # CHECK KR_BOUNDS
            self.max_error, self.min_error = tools.check_kr(self.M[0], self.M[-1], self.cosmo_params['mean_dens'],
                                                            np.exp(self.__lnkh[0]), np.exp(self.__lnkh[-1]))

            if self.max_error:
                print self.max_error
            if self.min_error:
                print self.min_error


            return self.__lnkh

    @lnkh.deleter
    def lnkh(self):
        try:
            del self.__lnkh

            del self._unnormalized_power
            del self._power_cdm_0
            del self._power_0
            del self._sigma_0
            del self._dlnsdlnm
            del self.lnk
        except:
            pass

    @property
    def lnk(self):
        """ Logarithmic k-bins. This is NOT k/h, just k (ie. 1/Mpc)"""
        try:
            return self.__lnk
        except:
            self.__lnk = self.lnkh + np.log(self.cosmo_params['H0'] / 100.0)
            return self.__lnk

    @lnk.deleter
    def lnk(self):
        try:
            del self.__lnk
        except:
            pass

    @property
    def _unnormalized_power(self):
        """
        The unnormalized power spectrum at z=0 for CDM
        """

        try:
            return self.__unnormalized_power
        except:
            if self.transfer_fit == "CAMB":
                self.__unnormalized_power = self.cosmo_params['n'] * self.lnkh + 2.0 * self._transfer_function_callable(self.lnkh) + np.log(2 * np.pi ** 2)
            else:
                self.__unnormalized_power = self.cosmo_params['n'] * self.lnkh + 2.0 * self._transfer_original[1, :] + np.log(2 * np.pi ** 2)
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
        The power spectrum at z=0 for CDM (normalised)
        """
        try:
            return self.__power_cdm_0
        except:
            self.__power_cdm_0, self._normalization = tools.normalize(self.cosmo_params['sigma_8'], self._unnormalized_power,
                                                                self.lnkh, self.cosmo_params['mean_dens'])
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
        The CDM/WDM power spectrum at z=0
        """
        try:
            return self.__power_0
        except:
            if self.wdm_mass is not None:
                print "Doing wdm with mass = ", self.wdm_mass
                self.__power_0 = tools.wdm_transfer(self.wdm_mass, self._power_cdm_0, self.lnkh,
                                                    self.cosmo_params["H0"], self.cosmo_params['omegac'])
            else:
                self.__power_0 = self._power_cdm_0

            return self.__power_0

    @_power_0.deleter
    def _power_0(self):
        try:
            del self.__power_0
            del self._sigma_0
            del self._dlnsdlnm
            del self.power
        except:
            pass

    @property
    def _sigma_0(self):
        """
        The mass variance at z=0, sigma (NOTE: not sigma^2)
        """

        try:
            return self.__sigma_0
        except:
            self.__sigma_0 = tools.mass_variance(self.M, self._power_0, self.lnkh, self.cosmo_params['mean_dens'])
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
        The derivative of the natural log of sigma with respect to the natural log of mass (absolute value)
        """
        try:
            return self.__dlnsdlnm
        except:
            self.__dlnsdlnm = tools.dlnsdlnm(self.M, self._sigma_0, self._power_0, self.lnkh, self.cosmo_params['mean_dens'])
            return self.__dlnsdlnm

    @_dlnsdlnm.deleter
    def _dlnsdlnm(self):
        try:
            del self.__dlnsdlnm
            del self.dndm
        except:
            pass

    @property
    def growth(self):
        """
        The growth factor d(z)
        """
        try:
            return self.__growth
        except:
            if self.z > 0:
                self.__growth = cosmography.growth_factor(self.z, self.cosmo_params['omegam'],
                                                           self.cosmo_params['omegak'], self.cosmo_params['omegav'])
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
        The fully realised power spectrum (with all parameters applied)
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
        The mass variance at arbitrary redshift (specified in instantiated object)
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
        Natural log of inverse mass variance
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
        Effective spectral index at scale of halo radius
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
        The multiplicity function f(sigma) for the fitting function specified as mf_fit
        """
        try:
            return self.__fsigma
        except:
            fits_class = fits(self.M, self.n_eff, self.mf_fit, self.sigma, self.cosmo_params['delta_c'], self.z, self.delta_halo, self.cosmo_params, self.user_fit, self.cut_fit)
            nufnu = fits_class.nufnu()
            self.__fsigma = nufnu()
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
        The derivative of the number density of haloes with respect to mass for the range of masses
        """
        try:
            return self.__dndm
        except:
            self.__dndm = self.fsigma * self.cosmo_params['mean_dens'] * np.abs(self._dlnsdlnm) / self.M ** 2
            if self.mf_fit == 'Behroozi':
                a = 1 / (1 + self.z)
                theta = 0.144 / (1 + np.exp(14.79 * (a - 0.213))) * (self.M / 10 ** 11.5) ** (0.5 / (1 + np.exp(6.5 * a)))
                ngtm_tinker = self._ngtm()
                ngtm_behroozi = 10 ** (theta + np.log10(ngtm_tinker))
                dthetadM = 0.144 / (1 + np.exp(14.79 * (a - 0.213))) * (0.5 / (1 + np.exp(6.5 * a))) * (self.M / 10 ** 11.5) ** (0.5 / (1 + np.exp(6.5 * a)) - 1) / (10 ** 11.5)
                self.__dndm = self.__dndm * 10 ** theta - ngtm_behroozi * np.log(10) * dthetadM

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
        The differential mass function in terms of natural log of M
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
        The differential mass function in terms of natural log of M
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
        ### WE CALCULATE THE MASS FUNCTION ABOVE THE COMPUTED RANGE ###
        # mass_function is logged already (not log10 though)
        m_upper = np.linspace(np.log(M[-1]), np.log(10 ** 18), 500)
        if cut:  #since its been cut, the best we can do is a power law
            mf_func = spline(np.log(M), mass_function, k=1)
            mf = mf_func(m_upper)
        else:
            #We try to calculate the hmf as far as we can normally
            sigma_0 = tools.mass_variance(np.exp(m_upper), self._power_0, self.lnkh, self.cosmo_params['mean_dens'])
            sigma = sigma_0 * self.growth
            dlnsdlnm = tools.dlnsdlnm(np.exp(m_upper), sigma_0, self._power_0, self.lnkh, self.cosmo_params['mean_dens'])
            n_eff = tools.n_eff(dlnsdlnm)
            fsigma = fits(m_upper, n_eff, self.mf_fit, sigma, self.cosmo_params['delta_c'],
                          self.z, self.delta_halo, self.cosmo_params, self.user_fit, cut_fit=True).nufnu()()
            #fsigma = nufnu()
            dndm = fsigma * self.cosmo_params['mean_dens'] * np.abs(dlnsdlnm) / np.exp(m_upper) ** 2
            mf = np.log(np.exp(m_upper) * dndm)

            if np.isnan(mf[-1]):  #Then we couldn't get up all the way, so have to do linear ext.
                if np.isnan(mf[1]):  #Then the whole extension is nan and we have to use the original (start at 1 because 1 val won't work either)
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
        if cut:  #since its been cut, the best we can do is a power law
            mf_func = spline(np.log(M), mass_function, k=1)
            mf = mf_func(m_lower)
        else:
            #We try to calculate the hmf as far as we can normally
            sigma_0 = tools.mass_variance(np.exp(m_lower), self._power_0, self.lnkh, self.cosmo_params['mean_dens'])
            sigma = sigma_0 * self.growth
            dlnsdlnm = tools.dlnsdlnm(np.exp(m_lower), sigma_0, self._power_0, self.lnkh, self.cosmo_params['mean_dens'])
            n_eff = tools.n_eff(dlnsdlnm)
            fsigma = fits(m_lower, n_eff, self.mf_fit, sigma, self.cosmo_params['delta_c'],
                          self.z, self.delta_halo, self.cosmo_params, self.user_fit, cut_fit=True).nufnu()()
            #fsigma = nufnu()
            dndm = fsigma * self.cosmo_params['mean_dens'] * np.abs(dlnsdlnm) / np.exp(m_lower) ** 2
            mf = np.log(np.exp(m_lower) * dndm)

            if np.isnan(mf[0]):  #Then we couldn't go down all the way, so have to do linear ext.
                mfslice = mf[np.logical_not(np.isnan(mf))]
                m_nan = m_lower[np.isnan(mf)]
                m_true = m_lower[np.logical_not(np.isnan(mf))]
                mf_func = spline(m_true, mfslice, k=1)
                mf[:len(mfslice)] = mf_func(m_nan)
        return m_lower, mf

    def _ngtm(self):
        """
        This function is separated from the property because of the Behroozi fit
        """
        # set M and mass_function within computed range
        M = self.M[np.logical_not(np.isnan(self.dndlnm))]
        mass_function = self.dndlnm[np.logical_not(np.isnan(self.dndlnm))]

        #Calculate the mass function (and its integral) from the highest M up to 10**18
        if M[-1] < 10 ** 18:
            m_upper, mf = self._upper_ngtm(M, np.log(mass_function), M[-1] < self.M[-1])

            int_upper = intg.simps(np.exp(mf), dx=m_upper[2] - m_upper[1], even='first')
        else:
            int_upper = 0

        #Calculate the cumulative integral (backwards) of mass_function (Adding on the upper integral)
        ngtm = intg.cumtrapz(mass_function[::-1], dx=np.log(M[1]) - np.log(M[0]), initial=0)[::-1] + int_upper

        #We need to set ngtm back in the original length vector with nans where they were originally
        if len(ngtm) < len(self.M):
            ngtm_temp = np.zeros_like(self.dndlnm)
            ngtm_temp[:] = np.nan
            ngtm_temp[np.logical_not(np.isnan(self.dndlnm))] = ngtm
            ngtm = ngtm_temp

        return ngtm

    @property
    def ngtm(self):
        """
        Integrates the mass function above a certain mass to calculate the number of haloes above a certain mass
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
        Integrates the mass function above a certain mass to calculate the mass in haloes above a certain mass
        """
        try:
            return self.__mgtm
        except:
            M = self.M[np.logical_not(np.isnan(self.dndlnm))]
            mass_function = self.dndlnm[np.logical_not(np.isnan(self.dndlnm))]

            #Calculate the mass function (and its integral) from the highest M up to 10**18
            if M[-1] < 10 ** 18:
                m_upper, mf = self._upper_ngtm(M, np.log(mass_function), M[-1] < self.M[-1])
                int_upper = intg.simps(np.exp(mf + m_upper) , dx=m_upper[2] - m_upper[1], even='first')
            else:
                int_upper = 0

            #Calculate the cumulative integral (backwards) of mass_function (Adding on the upper integral)
            self.__mgtm = intg.cumtrapz(mass_function[::-1] * M[::-1], dx=np.log(M[1]) - np.log(M[0]), initial=0)[::-1] + int_upper

            #We need to set ngtm back in the original length vector with nans where they were originally
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
        Integrates the mass function below a certain mass to calculate the number of haloes below that mass
        """
        try:
            return self.__nltm
        except:
            # set M and mass_function within computed range
            M = self.M[np.logical_not(np.isnan(self.dndlnm))]
            mass_function = self.dndlnm[np.logical_not(np.isnan(self.dndlnm))]

            #Calculate the mass function (and its integral) from 10**3 up to lowest M
            if M[0] > 10 ** 3:
                m_lower, mf = self._lower_ngtm(M, np.log(mass_function), M[0] > self.M[0])

                int_lower = intg.simps(np.exp(mf), dx=m_lower[2] - m_lower[1], even='first')
            else:
                int_lower = 0

            print "INT LOWER = ", int_lower
            #Calculate the cumulative integral of mass_function (Adding on the lower integral)
            self.__nltm = intg.cumtrapz(mass_function, dx=np.log(M[1]) - np.log(M[0]), initial=0) + int_lower

            #We need to set ngtm back in the original length vector with nans where they were originally
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
        Integrates the mass function above a certain mass to calculate the mass in haloes below a certain mass
        """
        try:
            return self.__mltm
        except:
            # Set M within calculated range
            M = self.M[np.logical_not(np.isnan(self.dndlnm))]
            mass_function = self.dndlnm[np.logical_not(np.isnan(self.dndlnm))]

            #Calculate the mass function (and its integral) from 10**3 up to lowest M
            if M[0] > 10 ** 3:
                m_lower, mf = self._lower_ngtm(M, np.log(mass_function), M[0] > self.M[0])

                int_lower = intg.simps(np.exp(mf + m_lower), dx=m_lower[2] - m_lower[1], even='first')
            else:
                int_lower = 0

            #Calculate the cumulative integral of mass_function (Adding on the upper integral)
            self.__mltm = intg.cumtrapz(mass_function * M, dx=np.log(M[1]) - np.log(M[0]), initial=0) + int_lower

            #We need to set ngtm back in the original length vector with nans where they were originally
            if len(self.__nltm) < len(self.M):
                nltm_temp = np.zeros_like(self.dndlnm)
                nltm_temp[:] = np.nan
                nltm_temp[np.logical_not(np.isnan(self.dndlnm))] = self.__mltm
                self.__mltm = nltm_temp

            return self.__mltm

    @property
    def how_big(self):
        """ 
        Calculates how big a box must be to expect one halo in natural log mass interval M, for given mass function
        """

        return self.ngtm ** (-1. / 3.)

    @how_big.deleter
    def how_big(self):
        try:
            del self.how_big
        except:
            pass




