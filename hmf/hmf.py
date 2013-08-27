'''
Perturbations.py contains a single class (Perturbations), which contains
methods that act upon a transfer function to gain functions such as the
mass function.
'''

version = '1.1.4'

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
        M:             The masses at which analysis is performed. (not log) [M/h]
        fsigma:        The multiplicity function, or fitting function at M
        dndm:          The comoving number density of halos in mass interval M [h**3/Mpc**3]
        dndlnm:        The comoving number density of halos in log mass interval M [h**3/Mpc**3]
        dndlog10m:     The comoving number density of halo in log10 mass interval M [h**3/Mpc**3]
        ngtm:          Comoving number density of halos > M [h**3/Mpc**3]
        nltm:          Comoving number density of halos < M [h**3/Mpc**3]
        mgtm:          Comoving mass density of halos > M [h**3/Mpc**3]
        mltm:          Comoving mass density of halos < M [h**3/Mpc**3]
        how_big:       The requisite size of a simulation box, L, to have at least one halo > M [Mpc/h]
        
    Input:
        M:             A vector of floats containing the log10(Solar Masses) at which to perform analysis.    
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
                            1. 'PS': Press-Schechter Approach
                            2. 'ST': Sheth-Tormen
                            3. 'Jenkins': Jenkins empirical fit
                            4. 'Warren': Warren empirical fit
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
                            15. "Behroozi": Behroozi extension to Tinker for high-z
                            16. 'user_model': A user-input string function
        
        delta_vir:     The virial overdensity for the halo definition
                       Default: delta_vir = 200.0
                       
        user_fit:      A string defining a mathematical function in terms of 'x', used as the fitting function,
                       where x is taken as sigma. Will only be applicable if mf_fit == "user_model".
                       Default: user_fit = ""
                       
        transfer_fit:  A string defining which transfer function fit to use. Current options are 'CAMB' and 'EH' (Eistenstein-Hu)
                       Default: transfer_fit = "CAMB"
                       
        cut_fit:       Whether to forcibly cut the f(sigma) at bounds given by respective papers.
                       If False, will use function to calculate all values specified in M (may give ridiculous results)
                       Default: True
                       
        R:             The distances at which the dark matter correlation function is calculated in Mpc/h
                       Default: np.linspace(1, 200, 200)
                       
        **kwargs:      There is a placeholder for any additional cosmological parameters, or camb
                       parameters, that one wishes to include. Parameters that aren't used won't
                       break the program, they will just be ignored. Here follows a list of parameters
                       that will be used by various parts of the program, and their respective defaults:
                       
                       PARAMETERS USED OUTSIDE OF CAMB ONLY:
                       sigma_8         :: 0.812
                       n               :: 1
                       delta_c         :: 1.686
                       
                       PARAMETERS USED IN CAMB AND OUTSIDE:
                       omegab          :: 0.0456
                       omegac          :: 0.2274
                       omegav          :: 0.727
                       omegak          :: 1 - omegab - omegac - omegal - omegan
                       
                       PARAMETERS USED ONLY IN CAMB
                       H0              :: 70.0
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
                       scalar_amp      :: 1
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
                       transfer__k_per_logint :: 0
                       transfer__kmax :: 2
                                

    """


    def __init__(self, M=np.linspace(10, 15, 501),
                 mf_fit="ST",
                 transfer_file=None,
                 z=0.0,
                 wdm_mass=None, k_bounds=[0.0000001, 20000.0],
                 delta_vir=200.0, user_fit='', transfer_fit='CAMB',
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
                               'cs2_lam' : 0,
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
                            "crit_dens":27.755 * 10 ** 10
                            }

        self._transfer_options = {'Num_Nu_massive'  : 0,
                                 'reion__fraction' :-1,
                                 'reion__delta_redshift' : 1.5,
                                 'Scalar_initial_condition' : 1,
                                 'scalar_amp'      : 1,
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
                                 'transfer__k_per_logint': 0,
                                 'transfer__kmax':2}


        #A dictionary of available HMF fitting functions and their identifiers
        self.mf_fits = {
            "PS":self._nufnu_PS,
            "ST":self._nufnu_ST,
            "Warren":self._nufnu_Warren,
            "Jenkins":self._nufnu_Jenkins,
            "Reed03":self._nufnu_Reed03,
            "Reed07":self._nufnu_Reed07,
            "Angulo":self._nufnu_Angulo,
            "Angulo_Bound":self._nufnu_Angulo_Bound,
            "Tinker":self._nufnu_Tinker,
            "Watson_FoF":self._nufnu_Watson_FoF,
            "Watson":self._nufnu_Watson,
            "Crocce":self._nufnu_Crocce,
            "Courtin":self._nufnu_Courtin,
            "Bhattacharya": self._nufnu_Bhattacharya,
            "Behroozi": self._nufnu_Tinker,
            "user_model":self._nufnu_user_model
            }

        self.update(M=M, mf_fit=mf_fit, k_bounds=k_bounds, transfer_file=transfer_file, wdm_mass=wdm_mass,
                     delta_vir=delta_vir, user_fit=user_fit, z=z, transfer_fit=transfer_fit, cut_fit=cut_fit, ** kwargs)

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
            elif key is 'delta_vir':
                self.delta_vir = kwargs['delta_vir']
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
        """
        All the cosmological parameters
        """
        try:
            #If it is defined already, just return it.
            #Whenever a parameter is modified, the dictionary is deleted and so will be remade here
            return self.__cosmo_params
        except:
            #Piece together the cosmo dictionaries
            self.__cosmo_params = dict(self._transfer_cosmo.items() + self._extra_cosmo.items())

            #Set some derivative parameters
            self.__cosmo_params['omegam'] = self.__cosmo_params['omegab'] + self.__cosmo_params['omegac']
            self.__cosmo_params['omegak'] = 1 - self.__cosmo_params['omegam'] - self.__cosmo_params['omegav'] - self.__cosmo_params['omegan']
            self.__cosmo_params['mean_dens'] = self.__cosmo_params['omegam'] * 27.755 * 10 ** 10
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
        """
        Here we track the camb parameters
        """

        try:
            #If it is defined already, just return it.
            #Whenever a parameter is modified, the dictionary is deleted and so will be remade here
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
    def delta_vir(self):
        """
        The virial overdensity with respect to the background
        """
        return self.__delta_vir

    @delta_vir.setter
    def delta_vir(self, val):
        try:
            val = float(val)
        except ValueError:
            raise ValueError("delta_vir must be a number: ", val)

        if val <= 0:
            raise ValueError("delta_vir must be > 0 (", val, ")")

        #Delete stuff dependent on it
        del self.fsigma
        self.__delta_vir = val


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
    def lnk(self):
        """
        The logarithmic bins in k-space
        """
        try:
            return self.__lnk
        except:
            if self.transfer_fit == "CAMB":
                self.__lnk, dlnk = tools.new_k_grid(self._transfer_original[0, :], self.k_bounds)
            elif self.transfer_fit == "EH":
                self.__lnk = self._transfer_original[0, :]

            # CHECK KR_BOUNDS
            self.max_error, self.min_error = tools.check_kr(self.M[0], self.M[-1], self.cosmo_params['mean_dens'],
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
            del self._power_cdm_0
            del self._power_0
            del self._sigma_0
            del self._dlnsdlnm
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
                self.__unnormalized_power = self.cosmo_params['n'] * self.lnk + 2.0 * self._transfer_function_callable(self.lnk)
            else:
                self.__unnormalized_power = self.cosmo_params['n'] * self.lnk + 2.0 * self._transfer_original[1, :]
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
            self.__power_cdm_0 = tools.normalize(self.cosmo_params['sigma_8'], self._unnormalized_power,
                                                                self.lnk, self.cosmo_params['mean_dens'])
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
                self.__power_0 = tools.wdm_transfer(self.wdm_mass, self._power_cdm_0, self.lnk,
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
            self.__sigma_0 = tools.mass_variance(self.M, self._power_0, self.lnk, self.cosmo_params['mean_dens'])
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
            self.__dlnsdlnm = tools.dlnsdlnm(self.M, self._sigma_0, self._power_0, self.lnk, self.cosmo_params['mean_dens'])
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

            nufnu = self.mf_fits[self.mf_fit]
            #This is a little ambiguous, but nufnu could depend on: sigma, overdensity, delta_c, lnsigma
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
            return self.__dndlnm
        except:
            self.__dndlnm = self.M * self.dndm * np.log(10)
            return self.__dndlnm

    @dndlog10m.deleter
    def dndlog10m(self):
        try:
            del self.__dndlnm
        except:
            pass

    def _ngtm(self):
        ngtm = np.zeros_like(self.dndlnm)

        # set M and mass_function within computed range
        M = self.M[np.logical_not(np.isnan(self.dndlnm))]
        mass_function = np.log(self.dndlnm[np.logical_not(np.isnan(self.dndlnm))])

        # Interpolate the mass_function - this is in log-log space.
        mf = spline(np.log(M), mass_function, k=1)

        # Define max_M as either 18 or the maximum set by user
        max_M = np.log(np.max([10 ** 18, M[-1]]))

        for i, m in enumerate(self.M):
            if np.isnan(m):
                ngtm[i] = np.nan
            else:
                # Set up new grid with 4097 steps from m to M=17
                M_new, dlnM = np.linspace(np.log(m), max_M, 4097, retstep=True)
                mf_new = mf(M_new)

                ngtm[i] = intg.romb(np.exp(mf_new), dx=dlnM)

        return ngtm

    @property
    def ngtm(self):
        """
        Integrates the mass function above a certain mass to calculate the number of haloes above a certain mass
        """
        try:
            return self.__ngtm
        except:
            # Initialize the function
#            self.__ngtm = np.zeros_like(self.dndlnm)
#
#            # set M and mass_function within computed range
#            M = self.M[np.logical_not(np.isnan(self.dndlnm))]
#            mass_function = np.log(self.dndlnm[np.logical_not(np.isnan(self.dndlnm))])
#
#            # Interpolate the mass_function - this is in log-log space.
#            mf = spline(np.log(M), mass_function, k=1)
#
#            # Define max_M as either 18 or the maximum set by user
#            max_M = np.log(np.max([10 ** 18, M[-1]]))
#
#            for i, m in enumerate(self.M):
#                if np.isnan(m):
#                    self.__ngtm[i] = np.nan
#                else:
#                    # Set up new grid with 4097 steps from m to M=17
#                    M_new, dlnM = np.linspace(np.log(m), max_M, 4097, retstep=True)
#                    mf_new = mf(M_new)
#
#                    self.__ngtm[i] = intg.romb(np.exp(mf_new), dx=dlnM)
#
#            #Here we add a correction for high-z for Tinker, by Behroozi.
#            if self.mf_fit == 'Behroozi':
#                pivot_factor = 0.144 / (1 + np.exp(14.79 * (1 / (1 + self.z)) - 0.213))
#                self.__ngtm = 10 ** (pivot_factor * (self.M / (10 ** 11.5)) ** (0.5 / (1 + np.exp(6.5 * (1 / (1 + self.z))))) + np.log10(self.__ngtm))
            self.__ngtm = self._ngtm()
            return self.__ngtm

    @ngtm.deleter
    def ngtm(self):
        try:
            del self.__ngtm
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
            # Initialize the function
            self.__mgtm = np.zeros_like(self.dndlnm)

            # set M and mass_function within computed range
            M = self.M[np.logical_not(np.isnan(self.dndlnm))]
            mass_function = np.log(self.dndlnm[np.logical_not(np.isnan(self.dndlnm))])

            # Interpolate the mass_function - this is in log-log space.
            mf = spline(np.log(M), mass_function, k=1)

            # Define max_M as either 18 or the maximum set by user
            max_M = np.log(np.max([10 ** 17, M[-1]]))

            for i, m in enumerate(self.M):
                if np.isnan(m):
                    self.__mgtm[i] = np.nan
                else:
                    # Set up new grid with 4097 steps from m to M=17
                    M_new, dlnM = np.linspace(np.log10(m), max_M, 4097, retstep=True)
                    mf_new = mf(M_new)
                    self.__mgtm[i] = intg.romb(np.exp(mf_new), dx=dlnM)

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
            # Initialize the function
            self.__nltm = np.zeros_like(self.dndlnm)

            # set M and mass_function within computed range
            M = self.M[np.logical_not(np.isnan(self.dndlnm))]
            mass_function = np.log(self.dndlnm[np.logical_not(np.isnan(self.dndlnm))])

            # Interpolate the mass_function - this is in log-log space.
            mf = spline(np.log(M), mass_function, k=3)

            # Define min_M as either 3 or the minimum set by user
            min_M = np.log(np.min([10 ** 3, M[0]]))

            for i, m in enumerate(self.M):
                if np.isnan(m):
                    self.__nltm[i] = np.nan
                else:
                    # Set up new grid with 4097 steps from m to M=17
                    M_new, dlnM = np.linspace(min_M, np.log10(m), 4097, retstep=True)
                    mf_new = mf(M_new)
                    self.__nltm[i] = intg.romb(np.exp(mf_new), dx=dlnM)

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
            # Initialize the function
            self.__mltm = np.zeros_like(self.dndlnm)

            # set M and mass_function within computed range
            M = self.M[np.logical_not(np.isnan(self.dndlnm))]
            mass_function = np.log(self.dndlnm[np.logical_not(np.isnan(self.dndlnm))])


            # Interpolate the mass_function - this is in log-log space.
            mf = spline(np.log(M), mass_function, k=3)

            # Define max_M as either 17 or the maximum set by user
            min_M = np.log(np.min([10 ** 3, M[0]]))

            for i, m in enumerate(self.M):
                if np.isnan(m):
                    self.__mltm[i] = np.nan
                else:
                    # Set up new grid with 4097 steps from m to M=17
                    M_new, dlnM = np.linspace(min_M, np.log10(m), 4097, retstep=True)
                    mf_new = mf(M_new)
                    self.__mltm[i] = intg.romb(np.exp(mf_new), dx=dlnM)

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

    def _nufnu_PS(self):
        """
        Computes the function nu*f(nu) for the Press-Schechter approach at a given radius.
        
        Input R: radius of the top-hat function
        Output: f_of_nu: the function nu*f(nu) for the PS approach.
        """

        vfv = np.sqrt(2.0 / np.pi) * (self.cosmo_params['delta_c'] / self.sigma) * np.exp(-0.5 * (self.cosmo_params['delta_c'] / self.sigma) ** 2)

        return vfv

    def _nufnu_ST(self):
        """
        Finds the Sheth Tormen vf(v) 
        
        Input R: radius of the top-hat function
        Output: vfv: the Sheth-Tormen mass function fit.
        """

        nu = self.cosmo_params['delta_c'] / self.sigma
        a = 0.707

        vfv = 0.3222 * np.sqrt(2.0 * a / np.pi) * nu * np.exp(-(a * nu ** 2) / 2.0) * (1 + (1.0 / (a * nu ** 2)) ** 0.3)

        return vfv

    def _nufnu_Jenkins(self):
        """
        Finds the Jenkins empirical vf(v) 
        
        Output: vfv: the Jenkins mass function fit.
        """

        vfv = 0.315 * np.exp(-np.abs(self.lnsigma + 0.61) ** 3.8)
        # Conditional on sigma range.
        if self.cut_fit:
            vfv[np.logical_or(self.lnsigma < -1.2, self.lnsigma > 1.05)] = np.NaN

        return vfv

    def _nufnu_Warren(self):
        """
        Finds the Warren empirical vf(v) 
        
        Input R: radius of the top-hat function
        Output: vfv: the Warren mass function fit.
        """

        vfv = 0.7234 * ((1.0 / self.sigma) ** 1.625 + 0.2538) * np.exp(-1.1982 / self.sigma ** 2)

        if self.cut_fit:
            vfv[np.logical_or(self.M < 10 ** 10, self.M > 10 ** 15)] = np.NaN
        return vfv

    def _nufnu_Reed03(self):
        """
        Finds the Reed 2003 empirical vf(v) 
        
        Input R: radius of the top-hat function
        Output: vfv: the Reed 2003 mass function fit.
        
        NOTE: Only valid from -1.7 < ln sigma^-1 < 0.9
        """

        ST_Fit = self._nufnu_ST()

        vfv = ST_Fit * np.exp(-0.7 / (self.sigma * np.cosh(2.0 * self.sigma) ** 5))

        if self.cut_fit:
            vfv[np.logical_or(self.lnsigma < -1.7, self.lnsigma > 0.9)] = np.NaN
        return vfv

    def _nufnu_Reed07(self):
        """
        Finds the Reed 2007 empirical vf(v) 
        
        Input R: radius of the top-hat function
        Output: vfv: the Reed 2003 mass function fit.
        
        NOTE: Only valid from -1.7 < ln sigma^-1 < 0.9
        """
        nu = self.cosmo_params['delta_c'] / self.sigma

        G_1 = np.exp(-((1.0 / self.sigma - 0.4) ** 2) / (2 * 0.6 ** 2))
        G_2 = np.exp(-((1.0 / self.sigma - 0.75) ** 2) / (2 * 0.2 ** 2))

        c = 1.08
        a = 0.764 / c
        A = 0.3222
        p = 0.3

        vfv = A * np.sqrt(2.0 * a / np.pi) * (1.0 + (1.0 / (a * nu ** 2)) ** p + 0.6 * G_1 + 0.4 * G_2) * nu * np.exp(-c * a * nu ** 2 / 2.0 - 0.03 * nu ** 0.6 / (self.n_eff + 3) ** 2)

        if self.cut_fit:
            vfv[np.logical_or(self.lnsigma < -0.5, self.lnsigma > 1.2)] = np.NaN

        return vfv


    def _nufnu_Angulo(self):

        vfv = 0.201 * ((2.08 / self.sigma) ** 1.7 + 1) * np.exp(-1.172 / self.sigma ** 2)
        return vfv

    def _nufnu_Angulo_Bound(self):
        vfv = 0.265 * ((1.675 / self.sigma) ** 1.9 + 1) * np.exp(-1.4 / self.sigma ** 2)
        return vfv

    def _nufnu_Tinker(self):

        #The Tinker function is a bit tricky - we use the code from http://cosmo.nyu.edu/~tinker/massfunction/MF_code.tar
        #to aide us.
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

        A_0 = A_func(self.delta_vir)
        a_0 = a_func(self.delta_vir)
        b_0 = b_func(self.delta_vir)
        c_0 = c_func(self.delta_vir)

        A = A_0 * (1 + self.z) ** (-0.14)
        a = a_0 * (1 + self.z) ** (-0.06)
        alpha = np.exp(-(0.75 / np.log(self.delta_vir / 75)) ** 1.2)
        b = b_0 * (1 + self.z) ** (-alpha)
        c = c_0


        vfv = A * ((self.sigma / b) ** (-a) + 1) * np.exp(-c / self.sigma ** 2)

        if self.cut_fit:
            vfv[np.logical_or(self.lnsigma < -0.6 , self.lnsigma > 0.4)] = np.nan

        return vfv

    def _watson_gamma(self):
        C = np.exp(0.023 * (self.delta_vir / 178 - 1))
        d = -0.456 * cosmography.omegam_z(self.z, self.cosmo_params['omegam'], self.cosmo_params['omegav'], self.cosmo_params['omegak']) - 0.139
        p = 0.072
        q = 2.13

        return C * (self.delta_vir / 178) ** d * np.exp(p * (1 - self.delta_vir / 178) / self.sigma ** q)


    def _nufnu_Watson_FoF(self):
        vfv = 0.282 * ((1.406 / self.sigma) ** 2.163 + 1) * np.exp(-1.21 / self.sigma ** 2)
        if self.cut_fit:
            vfv[np.logical_or(self.lnsigma < -0.55 , self.lnsigma > 1.31)] = np.NaN
        return vfv

    def _nufnu_Watson(self):

        if self.z == 0:
            A = 0.194
            alpha = 2.267
            beta = 1.805
            gamma = 1.287
        elif self.z > 6:
            A = 0.563
            alpha = 0.874
            beta = 3.810
            gamma = 1.453
        else:
            A = cosmography.omegam_z(self.z, self.cosmo_params['omegam'], self.cosmo_params['omegav'], self.cosmo_params['omegak']) * (1.097 * (1 + self.z) ** (-3.216) + 0.074)
            alpha = cosmography.omegam_z(self.z, self.cosmo_params['omegam'], self.cosmo_params['omegav'], self.cosmo_params['omegak']) * (3.136 * (1 + self.z) ** (-3.058) + 2.349)
            beta = cosmography.omegam_z(self.z, self.cosmo_params['omegam'], self.cosmo_params['omegav'], self.cosmo_params['omegak']) * (5.907 * (1 + self.z) ** (-3.599) + 2.344)
            gamma = 1.318

        vfv = self._watson_gamma() * A * ((beta / self.sigma) ** alpha + 1) * np.exp(-gamma / self.sigma ** 2)

        if self.cut_fit:
            vfv[np.logical_or(self.lnsigma < -0.55, self.lnsigma > 1.05)] = np.NaN

        return vfv

    def _nufnu_Crocce(self):

        A = 0.58 * (1 + self.z) ** (-0.13)
        a = 1.37 * (1 + self.z) ** (-0.15)
        b = 0.3 * (1 + self.z) ** (-0.084)
        c = 1.036 * (1 + self.z) ** (-0.024)

        vfv = A * (self.sigma ** (-a) + b) * np.exp(-c / self.sigma ** 2)
        return vfv

    def _nufnu_Courtin(self):
        A = 0.348
        a = 0.695
        p = 0.1
        d_c = self.cosmo_params['delta_c']  # Note for WMAP5 they find delta_c = 1.673

        vfv = A * np.sqrt(2 * a / np.pi) * (d_c / self.sigma) * (1 + (d_c / (self.sigma * np.sqrt(a))) ** (-2 * p)) * np.exp(-d_c ** 2 * a / (2 * self.sigma ** 2))
        return vfv

    def _nufnu_Bhattacharya(self):
        A = 0.333 * (1 + self.z) ** -0.11
        a = 0.788 * (1 + self.z) ** -0.01
        p = 0.807
        q = 1.795

        nu = self.cosmo_params['delta_c'] / self.sigma

        vfv = A * np.sqrt(2.0 / np.pi) * np.exp(-(a * nu ** 2) / 2.0) * (1 + (1.0 / (a * nu ** 2)) ** p) * (nu * np.sqrt(a)) ** q
        if self.cut_fit:
            vfv[np.logical_or(self.M < 6 * 10 ** 11, self.M > 3 * 10 ** 15)] = np.NaN

        return vfv

    def _nufnu_user_model(self):
        """
        Calculates vfv based on a user-input model.
        """
        from scitools.StringFunction import StringFunction

        f = StringFunction(self.user_fit, globals=globals())


        return f(self.sigma)



if __name__ == "__main__":
    M = np.arange(10, 15, 0.01)
    pert = Perturbations(M)
    pert.dndlnm
    pert.how_big

    pert.update(z=1, omegac=0.2)
    pert.dndlog10m



