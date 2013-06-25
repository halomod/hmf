'''
Perturbations.py contains a single class (Perturbations), which contains
methods that act upon a transfer function to gain functions such as the
mass function.
'''

version = '1.0.10'

###############################################################################
# Some Imports
###############################################################################
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import scipy.integrate as intg
import numpy as np
from numpy import sin, cos, tan, abs, arctan, arccos, arcsin, exp
import collections
# from scitools.std import sin,cos,tan,abs,arctan,arccos,arcsin #Must be in this form to work for some reason.

from Distances import Distances
from SetupFunctions import Setup, check_kR

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
    
    Required Input:
        M: a float or vector of floats containing the log10(Solar Masses) at which to perform analysis.
        
    Optional Input:
        transfer_file: Either a string pointing to a file with a CAMB-produced transfer function,
                       or None. If None, will use CAMB on the fly to produce the function.
                       Default is None.
                       
        z:             a float giving the redshift of the analysis.
                       Default z = 0.0
                   
        WDM:           a float giving warm dark matter particle size in keV. 
                       Default is None (corresponds to CDM)
                                                       
        k_bounds:      a list/tuple defining two values: the lower and upper limit of k. Used to truncate/extend
                       the power spectrum. 
                       Default [0.0000001,20000.0]
   
        extrapolate:   Whether to use the k_bounds for extrapolation/truncation
                       Default: True
                     
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
                                
    """


    def __init__(self, M,
                 transfer_file=None,
                 z=0.0,
                 WDM=None, k_bounds=[0.0000001, 20000.0],
                 extrapolate=True, **kwargs):
        """
        Initializes the cosmology for which to perform the perturbation analysis.      
        """
        self.transfer_cosmo = {"w_lam"    :-1,
                               "omegab"   : 0.0455,
                               "omegac"   : 0.226,
                               "omegav"   : 0.728,
                               "omegan"   : 0.0,
                               "H0"       : 70.4,
                               'cs2_lam' : 0,
                               #'reion__optical_depth' : 0.0085,
                               'TCMB'     : 2.725,
                               'yhe'      : 0.24,
                               'Num_Nu_massless' : 3.04,
                               }
        self.transfer_cosmo['omegak'] = 1 - self.transfer_cosmo['omegab'] - self.transfer_cosmo['omegav'] - self.transfer_cosmo['omegac'] - self.transfer_cosmo['omegan']

        self.extra_cosmo = {"sigma_8":0.81,
                            "n":0.967,
                            "delta_c":1.686,
                            'A_s':None}

        self.crit_dens = 27.755 * 10 ** 10

        self.transfer_options = {'Num_Nu_massive'  : 0,
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
                                 'DoLensing'       : False }

        if self.transfer_options['reion__use_optical_depth']:
            self.transfer_cosmo['reion__optical_depth'] = 0.085
        else:
            self.transfer_cosmo['reion__redshift'] = 10.3

        # Put together the transfer options and transfer_cosmo dictionaries
        self.camb_dict = dict(self.transfer_cosmo.items() + self.transfer_options.items())

        #======== Save Some Parameters To The Class========
        # Run Parameters
        self.M = 10 ** M
        self.dlogM = M[1] - M[0]

        self.transfer_file = transfer_file
        self.k_bounds = k_bounds
        self.extrapolate = extrapolate
        self.z = z
        self.WDM = WDM


        # Start the cascade of variable setting.
        self.set_transfer_cosmo(**kwargs)

    def set_transfer_cosmo(self, **kwargs):

        #UPDATE ----
        # cosmology and transfer options
        for key, val in kwargs.iteritems():
            if key in self.camb_dict:
                self.camb_dict.update({key:val})

        if 'transfer_file' in kwargs:
            self.transfer_file = kwargs.pop('transfer_file')


        # DO transfer-unique stuff
        self.k_original, self.Transfer = Setup(self.transfer_file, self.camb_dict)

        self.transfer_function = self.Interpolate(self.k_original, self.Transfer)
        self.dist = Distances(**kwargs)
        self.mean_dens = (self.camb_dict['omegac'] + self.camb_dict['omegab']) * self.crit_dens

        # Pass on on the baton
        self.set_kbounds(**kwargs)

    def set_kbounds(self, **kwargs):

        #UPDATE ----
        if 'extrapolate' in kwargs:
            self.extrapolate = kwargs.pop('extrapolate')
        if 'k_bounds' in kwargs:
            self.k_bounds = kwargs.pop('k_bounds')
        if 'n' in kwargs:
            self.extra_cosmo['n'] = kwargs.pop('n')
        if 'sigma_8' in kwargs:
            self.extra_cosmo['sigma_8'] = kwargs.pop('sigma_8')

        # Set up a new grid for the transfer function (for romberg integration)
        self.k, self.dlnk = self.NewKGrid(self.k_original, self.extrapolate, self.k_bounds)

        # We check the bounds and generate a warning if they are not good
        self.max_error, self.min_error = check_kR(self.M[0], self.M[-1], self.mean_dens, np.exp(self.k[0]), np.exp(self.k[-1]))

        if self.max_error:
            print self.max_error
        if self.min_error:
            print self.min_error

        # power_base is the unnormalized power spectrum at z=0 and for CDM
        power_base = self.extra_cosmo['n'] * self.k + 2.0 * self.transfer_function(self.k)

        # Normalize the Power Spectrum (power_base)
        self.power_base , self.normalization = self.Normalize(self.extra_cosmo['sigma_8'], power_base)

        self.set_WDM(**kwargs)

    def set_WDM(self, **kwargs):

        #UPDATE -----
        if 'WDM' in kwargs:
            self.WDM = kwargs.pop('WDM')


        # Apply further WDM filter if applicable
        if self.WDM is not None:
            # power_spectrum_0 is power spectrum at z=0 for arbitrary WDM
            self.power_spectrum_0 = self.WDM_PS(self.WDM, self.power_base)
        else:
            self.power_spectrum_0 = self.power_base

        # Calculate Mass Variance at z=0
        self.sigma_0 = self.MassVariance(self.M, self.power_spectrum_0)

        # calculate dln(sigma)/dln(M)
        self.dlnsdlnM()

        # Set the redshift, and calculate mass variance and other
        # functions used throughout dependent on z
        self.set_z(**kwargs)

    def set_z(self, **kwargs):
        """
        Wrapper to set the redshift and calculate quantities dependent on it
        """

        #UPDATE ----
        if 'z' in kwargs:
            self.z = kwargs.pop('z')

        # Calculate the Linear Growth Factor
        if self.z > 0:
            self.growth = self.dist.GrowthFactor(self.z)
        else:
            self.growth = 1

        # Apply linear growth to power spectrum
        self.power_spectrum = 2 * np.log(self.growth) + self.power_spectrum_0

        # Apply linear growth to Mass Variance
        self.sigma = self.sigma_0 * self.growth

        # Calculate ln(1/sigma)
        self.lnsigma = np.log(1 / self.sigma)

        # Calculate effective spectral slope at scale of radius of halo.
        self.n_eff = self.N_Eff(self.dlnsdlnm)


    def update(self, **kwargs):
        """
        A convenience wrapper to auto-update the cosmology or parameters in an optimized way
        """
        set_transfer = ['transfer_file']
        set_kbounds = ['k_bounds', 'extrapolate']
        set_kbounds_extra_cosmo = ['n', 'sigma_8']
        set_WDM = ['WDM']
        set_z = ['z']

        # We call the top-most function which has a kwarg associated with it (the rest will cascade from there)
        for key, val in kwargs.iteritems():
            if key in set_transfer and val != self.__dict__[key]:
                self.set_transfer_cosmo(**kwargs)
                return
            elif key in self.camb_dict and val != self.camb_dict[key]:
                self.set_transfer_cosmo(**kwargs)
                return

        for key, val in kwargs.iteritems():
            if (key in set_kbounds and val != self.__dict__[key]) or (key in set_kbounds_extra_cosmo and val != self.extra_cosmo[key]):
                self.set_kbounds(**kwargs)
                return

        for key, val in kwargs.iteritems():
            if key in set_WDM and val != self.__dict__[key]:
                self.set_WDM(**kwargs)
                return

        for key, val in kwargs.iteritems():
            if key in set_z and val != self.__dict__[key]:
                self.set_z(**kwargs)
                return

        print "Warning: No variables were updated!"
        for key, val in kwargs.iteritems():
            if key not in set_transfer + set_kbounds + set_kbounds_extra_cosmo + set_WDM + set_z + self.camb_dict.keys() + self.extra_cosmo.keys():
                print "Warning: Variable entered (", key, ") is not a valid keyword"
            if key in self.__dict__:
                if val == self.__dict__[key]:
                    print key, " was already ", val
            if key in self.camb_dict:
                if val == self.camb_dict[key]:
                    print key, " was already ", val
            if key in self.extra_cosmo:
                if val == self.extra_cosmo[key]:
                    print key, " was already ", val
        return

    def NewKGrid(self, k, extrapolate, k_bounds):
        """
        Creates a new grid for the transfer function, for application of Romberg integration.
        
        Note: for Romberg integration, the number of steps must be 2**p+1 where p is an integer, which is why this scaling
                should be performed. We choose 4097 bins for ln(k). This could possibly be optimized or made variable.
        """

        # Determine the true k_bounds.
        if extrapolate:
            min_k = np.log(k_bounds[0])
            max_k = np.log(k_bounds[1])
        else:
            min_k = np.min(k)
            max_k = np.max(k)

        # Setup the grid and fetch the grid-spacing as well
        k, dlnk = np.linspace(min_k, max_k, 4097, retstep=True)

        return k, dlnk

    def Interpolate(self, k, Transfer, tol=0.01):
        """
        Interpolates the given Transfer function and transforms it into a Power Spectrum.
        
        Input: k        : an array of values of lnk
               Transfer : an array of values of lnT
               step_size: the step_size between values of lnk required in interpolation.
                  
        Notes: the power spectrum is calculated using P(k) =  A.k^n.T^2, with A=1 in this method
                (which leaves the power spectrum un-normalized). The interpolation is done using
                cubic splines.
        """

        # Unfortunately it looks like there's a turn-up at low-k for some CAMB
        # transfers which makes the extrapolation silly.
        # If this is the case, we start when it is nice.

        start = 0
        for i in range(len(k) - 1):
            if abs((Transfer[i + 1] - Transfer[i]) / (k[i + 1] - k[i])) < tol:
                start = i
                break
        if start > 0:
            Transfer = Transfer[start:-1]
            k = k[start:-1]
            spline_order = 1
        else:
            spline_order = 1

        transfer_function = spline(k, Transfer, k=spline_order)

        return transfer_function

    def Normalize(self, norm_sigma_8, unn_power):
        """
        Normalize the power spectrum to a given sigma_8
        
        Input: norm_sigma_8: The sigma_8 value to normalize to.
        """
        # Calculate the value of sigma_8 without prior normalization.

        sigma_8 = self.MassVariance(4.*np.pi * 8 ** 3 * self.mean_dens / 3., unn_power)[0]


        # Calculate the normalization factor A.
        normalization = norm_sigma_8 / sigma_8

        # Normalize the previously calculated power spectrum.
        power = 2 * np.log(normalization) + unn_power
        return power, normalization


    def Radius(self, M):
        """
        Calculates radius from mass given mean density.
        """
        return (3.*M / (4.*np.pi * self.mean_dens)) ** (1. / 3.)


    def N_Eff(self, dlnsdlnm):
        """
        Calculates the power spectral slope at the scale of the halo radius, using eq. 42 in Lukic et. al 2007.
        """

        n_eff = -3.0 * (2.0 * dlnsdlnm + 1.0)

        return n_eff


    def WDM_PS(self, WDM, power_CDM):
        """
        Tansforms the CDM Power Spectrum into a WDM power spectrum for a given warm particle mass m_x.
        
        NOTE: formula from Bode et. al. 2001 eq. A9
        """

        h = self.camb_dict['H0'] / 100
        g_x = 1.5
        m_x = WDM
        nu = 1.12

        alpha = 0.049 * (self.camb_dict['omegac'] / 0.25) ** 0.11 * (h / 0.7) ** 1.22 * (1 / m_x) ** 1.11 * (1.5 / g_x) ** 0.29

        Transfer = (1 + (alpha * np.exp(self.k)) ** (2 * nu)) ** -(5.0 / nu)

        return power_CDM + 2 * np.log(Transfer)

    def TopHat_WindowFunction(self, m):
        """
        Constructs the window function squared in Fourier space for given radii
        
        Input: R: The radius of the top-hat function
        Output: W_squared: The square of the top-hat window function in Fourier space.
        """

        # Calculate the factor kR, minding to un-log k before use.
        kR = np.exp(self.k) * self.Radius(m)

        W_squared = (3 * (np.sin(kR) / kR ** 3 - np.cos(kR) / kR ** 2)) ** 2
        #W_squared[kR < 0.01] = 1.0
        #if W_squared[0] < 0.5 or W_squared[0] > 1.5:
        #    print np.sin(kR[0]), kR[0] ** 3, np.cos(kR[0]), kR[0] ** 2, np.sin(kR[0]) / kR[0] ** 3, np.cos(kR[0]) / kR[0] ** 2, W_squared[0]
        return W_squared


    def MassVariance(self, M, power):
        """
        Finds the Mass Variance of M using the top-hat window function.
        
        Input: M: the radius(mass) of the top-hat function (vector).
        Output: sigma: the mass variance.
        """

        # If we input a scalar as M, then just make it a one-element list.
        if not isinstance(M, collections.Iterable):
            M = [M]

        # Calculate the integrand of the function. Note that the power spectrum and k values must be
        # 'un-logged' before use, and we multiply by k because our steps are in logk.
        sigma = np.zeros_like(M)
        rest = np.exp(power + 3 * self.k)
        for i, m in enumerate(M):
            integ = rest * self.TopHat_WindowFunction(m)
            sigma[i] = (0.5 / np.pi ** 2) * intg.romb(integ, dx=self.dlnk)

        return np.sqrt(sigma)


    def dlnsdlnM(self):
        """
        Uses a top-hat window function to calculate |dlnsigma/dlnM| at a given radius.
        
        Input: R: the radius of the top-hat function. 
        Output: integral: the derivatiave of log sigma with respect to log Mass.
        """
        R = self.Radius(self.M)
        self.dlnsdlnm = np.zeros_like(self.M)
        for i, r in enumerate(R):
            g = np.exp(self.k) * r
            w = (np.sin(g) - g * np.cos(g)) * (np.sin(g) * (1 - 3.0 / (g ** 2)) + 3.0 * np.cos(g) / g)  # Derivative of W^2
            integ = w * np.exp(self.power_spectrum_0 - self.k)
            self.dlnsdlnm[i] = (3.0 / (2.0 * self.sigma_0[i] ** 2 * np.pi ** 2 * r ** 4)) * intg.romb(integ, dx=self.dlnk)

    def dndM(self):
        """
        Computes the value of dn/dM for a given radius.
        
        Input: R: radius of the top-hat function.
              Approach: A string indicating which approach to take to the fitting function. 
                        Valid values are:
                        1. PS: Press-Schechter Approach
                        2. ST: Sheth-Tormen
                        3. Jenkins: Jenkins empirical fit
                        4. Warren: Warren empirical fit
                        5. Reed03: Reed empirical from 2003
                        6. Reed07: Reed empirical from 2007
                        
                      All fits are taken from Lukic et. al. 2007
        Output dndm: the number density of objects in a mass range dM.
        """

        nufnu_dict = {
            "PS":self.nufnu_PS,
            "ST":self.nufnu_ST,
            "Warren":self.nufnu_Warren,
            "Jenkins":self.nufnu_Jenkins,
            "Reed03":self.nufnu_Reed03,
            "Reed07":self.nufnu_Reed07,
            "Angulo":self.nufnu_Angulo,
            "Angulo_Bound":self.nufnu_Angulo_Bound,
            "Tinker":self.nufnu_Tinker,
            "Watson_FoF":self.nufnu_Watson_FoF,
            "Watson":self.nufnu_Watson,
            "Crocce":self.nufnu_Crocce,
            "Courtin":self.nufnu_Courtin,
            "Bhattacharya": self.nufnu_Bhattacharya,
            "user_model":self.nufnu_user_model
            }

        nufnu = nufnu_dict[self.fsigma]
        self.vfv = nufnu()

        leftovers = self.mean_dens / self.M ** 2

        dndm = self.vfv * leftovers * np.abs(self.dlnsdlnm)

        return dndm

    def MassFunction(self, fsigma='ST', user_model='', overdensity=178, delta_c=1.686):
        """
        Uses EPS framework to calculate the mass function.
        
        Input:
            fsigma: str, default = 'ST'
                An option defining the fitting function to be used.
                Valid values are:
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
                        14. 'user_model': A user-input string function
                        
            user_model: str, default = ''
                A string defining a mathematical function of x, which will be
                the mass variance (sigma). 
            
            overdensity: float, default=178
                The virial overdensity of the halo definition. The only fitting
                functions that will take any notice of this are Tinker and Watson.
            
            delta_c: float, default = 1.686
                The critical overdensity parameter. Default corresponds to an
                Einstein-deSitter Universe. Many later fitting functions don't
                notice this parameter at all (it is fit to).
                
        Output: 
            mass_function: ndarray shape(len(M))
                The mass function, log10(dn/dlnM).
        """
        if fsigma == 'user_model' and user_model == '':
            print "Warning: Fitting Function is a user model, but no model supplied"
            return

        self.fsigma = fsigma
        self.user_model = user_model
        self.overdensity = overdensity
        self.extra_cosmo['delta_c'] = delta_c

        mass_function = self.M * self.dndM()

        return mass_function

    def NgtM(self, mass_function):
        """
        Integrates the mass function above a certain mass to calculate the number of haloes above a certain mass
        
        INPUT:
        mass_function: an array containing the mass function ( NOT log10)
        
        OUTPUT:
        ngtm: the number of haloes greater than M for all M in self.M
        """
        # Initialize the function
        ngtm = np.zeros_like(mass_function)

        # set M and mass_function within computed range
        M = self.M[np.logical_not(np.isnan(mass_function))]
        mass_function = np.log(mass_function[np.logical_not(np.isnan(mass_function))])

        # Interpolate the mass_function - this is in log-log space.
        mf = spline(np.log(M), mass_function, k=1)

        # Define max_M as either 18 or the maximum set by user
        max_M = np.log(np.max([10 ** 18, M[-1]]))

        #M_new = np.arange(np.log(M[-1]), max_M, M[1] / M[0])
#        if len(M_new) > 0:
#            upper_bit = np.sum(np.exp(mf(M_new)) * M[1] / M[0])
#        else:
#            upper_bit = 0
#        print "upper bit ", upper_bit
#        for i, m in enumerate(self.M):
#            if np.isnan(m):
#                ngtm[i] = np.nan
#            elif i == 0:
#                ngtm[len(mass_function) - 1] = np.exp(mf(np.log(m))) * M[1] / M[0] + upper_bit
#            else:
#                ngtm[len(mass_function) - i - 1] = ngtm[len(mass_function) - i ] + np.exp(mf(np.log(m))) * M[1] / M[0]

        for i, m in enumerate(self.M):
            if np.isnan(m):
                ngtm[i] = np.nan
            else:
                # Set up new grid with 4097 steps from m to M=17
                M_new, dlnM = np.linspace(np.log(m), max_M, 4097, retstep=True)
                mf_new = mf(M_new)

                ngtm[i] = intg.romb(np.exp(mf_new), dx=dlnM)
        return ngtm

    def MgtM(self, mass_function):
        """
        Integrates the mass function above a certain mass to calculate the number of haloes above a certain mass
        
        INPUT:
        mass_function: an array containing the mass function (log10)
        
        OUTPUT:
        mgtm: the mass in haloes greater than M for all M in self.M
        """

        # Initialize the function
        mgtm = np.zeros_like(mass_function)

        # set M and mass_function within computed range
        M = self.M[np.logical_not(np.isnan(mass_function))]
        mass_function = np.log(mass_function[np.logical_not(np.isnan(mass_function))])

        # Interpolate the mass_function - this is in log-log space.
        mf = spline(np.log(M), mass_function, k=1)

        # Define max_M as either 18 or the maximum set by user
        max_M = np.log(np.max([10 ** 17, M[-1]]))

        for i, m in enumerate(self.M):
            if np.isnan(m):
                mgtm[i] = np.nan
            else:
                # Set up new grid with 4097 steps from m to M=17
                M_new, dlnM = np.linspace(np.log10(m), max_M, 4097, retstep=True)
                mf_new = mf(M_new)
                mgtm[i] = intg.romb(np.exp(mf_new), dx=dlnM)

        return mgtm

    def NltM(self, mass_function):
        """
        Integrates the mass function below a certain mass to calculate the number of haloes below that mass
        
        INPUT:
        mass_function: an array containing the mass function (log10)
        
        OUTPUT:
        nltm: the number of haloes less than M for all M in self.M
        """

        # Initialize the function
        nltm = np.zeros_like(mass_function)

        # set M and mass_function within computed range
        M = self.M[np.logical_not(np.isnan(mass_function))]
        mass_function = np.log(mass_function[np.logical_not(np.isnan(mass_function))])

        # Interpolate the mass_function - this is in log-log space.
        mf = spline(np.log(M), mass_function, k=3)

        # Define min_M as either 3 or the minimum set by user
        max_M = np.log(np.min([10 ** 3, M[0]]))

        for i, m in enumerate(self.M):
            if np.isnan(m):
                nltm[i] = np.nan
            else:
                # Set up new grid with 4097 steps from m to M=17
                M_new, dlnM = np.linspace(min_M, np.log10(m), 4097, retstep=True)
                mf_new = mf(M_new)
                nltm[i] = intg.romb(np.exp(mf_new), dx=dlnM)

        return nltm

    def MltM(self, mass_function):
        """
        Integrates the mass function above a certain mass to calculate the number of haloes above a certain mass
        
        INPUT:
        mass_function: an array containing the mass function (log10)
        
        OUTPUT:
        mgtm: the mass in haloes greater than M for all M in self.M
        """

        # Initialize the function
        mltm = np.zeros_like(mass_function)

        # set M and mass_function within computed range
        M = self.M[np.logical_not(np.isnan(mass_function))]
        mass_function = np.log(mass_function[np.logical_not(np.isnan(mass_function))])


        # Interpolate the mass_function - this is in log-log space.
        mf = spline(np.log(M), mass_function, k=3)

        # Define max_M as either 17 or the maximum set by user
        max_M = np.log(np.min([10 ** 3, M[0]]))

        for i, m in enumerate(self.M):
            if np.isnan(m):
                mltm[i] = np.nan
            else:
                # Set up new grid with 4097 steps from m to M=17
                M_new, dlnM = np.linspace(min_M, np.log10(m), 4097, retstep=True)
                mf_new = mf(M_new)
                mltm[i] = intg.romb(np.exp(mf_new), dx=dlnM)

        return mltm

    def how_big(self, mass_function):
        # Calculates how big a box must be to expect
        # one halo for given mass function

        return mass_function ** (-1. / 3.)

    def nufnu_PS(self):
        """
        Computes the function nu*f(nu) for the Press-Schechter approach at a given radius.
        
        Input R: radius of the top-hat function
        Output: f_of_nu: the function nu*f(nu) for the PS approach.
        """

        vfv = np.sqrt(2.0 / np.pi) * (self.extra_cosmo['delta_c'] / self.sigma) * np.exp(-0.5 * (self.extra_cosmo['delta_c'] / self.sigma) ** 2)

        return vfv

    def nufnu_ST(self):
        """
        Finds the Sheth Tormen vf(v) 
        
        Input R: radius of the top-hat function
        Output: vfv: the Sheth-Tormen mass function fit.
        """

        nu = self.extra_cosmo['delta_c'] / self.sigma
        a = 0.707

        vfv = 0.3222 * np.sqrt(2.0 * a / np.pi) * nu * np.exp(-(a * nu ** 2) / 2.0) * (1 + (1.0 / (a * nu ** 2)) ** 0.3)

        return vfv

    def nufnu_Jenkins(self):
        """
        Finds the Jenkins empirical vf(v) 
        
        Output: vfv: the Jenkins mass function fit.
        """

        vfv = 0.315 * np.exp(-np.abs(self.lnsigma + 0.61) ** 3.8)
        # Conditional on sigma range.
        vfv[np.logical_or(self.lnsigma < -1.2, self.lnsigma > 1.05)] = np.NaN
        return vfv

    def nufnu_Warren(self):
        """
        Finds the Warren empirical vf(v) 
        
        Input R: radius of the top-hat function
        Output: vfv: the Warren mass function fit.
        """

        vfv = 0.7234 * ((1.0 / self.sigma) ** 1.625 + 0.2538) * np.exp(-1.1982 / self.sigma ** 2)

        vfv[np.logical_or(self.M < 10 ** 10, self.M > 10 ** 15)] = np.NaN
        return vfv

    def nufnu_Reed03(self):
        """
        Finds the Reed 2003 empirical vf(v) 
        
        Input R: radius of the top-hat function
        Output: vfv: the Reed 2003 mass function fit.
        
        NOTE: Only valid from -1.7 < ln sigma^-1 < 0.9
        """

        ST_Fit = self.nufnu_ST()

        vfv = ST_Fit * np.exp(-0.7 / (self.sigma * np.cosh(2.0 * self.sigma) ** 5))

        vfv[np.logical_or(self.lnsigma < -1.7, self.lnsigma > 0.9)] = np.NaN
        return vfv

    def nufnu_Reed07(self):
        """
        Finds the Reed 2007 empirical vf(v) 
        
        Input R: radius of the top-hat function
        Output: vfv: the Reed 2003 mass function fit.
        
        NOTE: Only valid from -1.7 < ln sigma^-1 < 0.9
        """
        nu = self.extra_cosmo['delta_c'] / self.sigma

        G_1 = np.exp(-((1.0 / self.sigma - 0.4) ** 2) / (2 * 0.6 ** 2))
        G_2 = np.exp(-((1.0 / self.sigma - 0.75) ** 2) / (2 * 0.2 ** 2))

        c = 1.08
        a = 0.764 / c
        A = 0.3222
        p = 0.3

        vfv = A * np.sqrt(2.0 * a / np.pi) * (1.0 + (1.0 / (a * nu ** 2)) ** p + 0.6 * G_1 + 0.4 * G_2) * nu * np.exp(-c * a * nu ** 2 / 2.0 - 0.03 * nu ** 0.6 / (self.n_eff + 3) ** 2)

        vfv[np.logical_or(self.lnsigma < -0.5, self.lnsigma > 1.2)] = np.NaN

        return vfv


    def nufnu_Angulo(self):

        vfv = 0.201 * ((2.08 / self.sigma) ** 1.7 + 1) * np.exp(-1.172 / self.sigma ** 2)
        return vfv

    def nufnu_Angulo_Bound(self):
        vfv = 0.265 * ((1.675 / self.sigma) ** 1.9 + 1) * np.exp(-1.4 / self.sigma ** 2)
        return vfv

    def nufnu_Tinker(self):

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

        A_0 = A_func(self.overdensity)
        a_0 = a_func(self.overdensity)
        b_0 = b_func(self.overdensity)
        c_0 = c_func(self.overdensity)

#        print "A_0: ", A_0
#        print "a_0:", a_0
#        print "b_0:", b_0
#        print "c_0:", c_0

        A = A_0 * (1 + self.z) ** (-0.14)
        a = a_0 * (1 + self.z) ** (-0.06)
        alpha = np.exp(-(0.75 / np.log(self.overdensity / 75)) ** 1.2)
        b = b_0 * (1 + self.z) ** (-alpha)
        c = c_0


        vfv = A * ((self.sigma / b) ** (-a) + 1) * np.exp(-c / self.sigma ** 2)
        vfv[np.logical_or(self.lnsigma < -0.6 , self.lnsigma > 0.4)] = np.nan
        return vfv

    def Watson_Gamma(self):
        C = 0.947 * np.exp(0.023 * (self.overdensity / 178 - 1))
        d = -0.456 * self.dist.Omega_M(self.z) - 0.139
        p = 0.072
        q = 2.13

        return C * (self.overdensity / 178) ** d * np.exp(p * (1 - self.overdensity / 178) / self.sigma ** q)


    def nufnu_Watson_FoF(self):
        vfv = 0.282 * ((2.163 / self.sigma) ** 1.406 + 1) * np.exp(-1.21 / self.sigma ** 2)
        vfv[np.logical_or(self.lnsigma < -0.55 , self.lnsigma > 1.31)] = np.NaN
        return vfv

    def nufnu_Watson(self):

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
            A = self.dist.Omega_M(self.z) * (1.097 * (1 + self.z) ** (-3.216) + 0.074)
            alpha = self.dist.Omega_M(self.z) * (3.136 * (1 + self.z) ** (-3.058) + 2.349)
            beta = self.dist.Omega_M(self.z) * (5.907 * (1 + self.z) ** (-3.599) + 2.344)
            gamma = 1.318

        vfv = self.Watson_Gamma() * A * ((beta / self.sigma) ** alpha + 1) * np.exp(-gamma / self.sigma ** 2)

        vfv[np.logical_or(self.lnsigma < -0.55, self.lnsigma > 1.05)] = np.NaN

        return vfv

    def nufnu_Crocce(self):

        A = 0.58 * (1 + self.z) ** (-0.13)
        a = 1.37 * (1 + self.z) ** (-0.15)
        b = 0.3 * (1 + self.z) ** (-0.084)
        c = 1.036 * (1 + self.z) ** (-0.024)

        vfv = A * (self.sigma ** (-a) + b) * np.exp(-c / self.sigma ** 2)
        return vfv

    def nufnu_Courtin(self):
        A = 0.348
        a = 0.695
        p = 0.1
        d_c = self.extra_cosmo['delta_c']  # Note for WMAP5 they find delta_c = 1.673

        vfv = A * np.sqrt(2 * a / np.pi) * (d_c / self.sigma) * (1 + (d_c / (self.sigma * np.sqrt(a))) ** (-2 * p)) * np.exp(-d_c ** 2 * a / (2 * self.sigma ** 2))
        return vfv

    def nufnu_Bhattacharya(self):
        A = 0.333 * (1 + self.z) ** -0.11
        a = 0.788 * (1 + self.z) ** -0.01
        p = 0.807
        q = 1.795

        nu = self.extra_cosmo['delta_c'] / self.sigma

        vfv = A * np.sqrt(2.0 / np.pi) * np.exp(-(a * nu ** 2) / 2.0) * (1 + (1.0 / (a * nu ** 2)) ** p) * (nu * np.sqrt(a)) ** q
        vfv[np.logical_or(self.M < 6 * 10 ** 11, self.M > 3 * 10 ** 15)] = np.NaN

        return vfv
    def nufnu_user_model(self, user_model):
        """
        Calculates vfv based on a user-input model.
        """
        from scitools.StringFunction import StringFunction

        f = StringFunction(user_model, globals=globals())


        return f(self.sigma)










