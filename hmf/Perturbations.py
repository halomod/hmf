'''
Created on Apr 5, 2012

@author: Steven Murray
@contact: steven.murray@uwa.edu.au

Last Modified: March 13, 2013

Perturbations.py contains a single class (Perturbations), which contains
methods that act upon a transfer function to gain functions such as the
mass function.
'''

history = \
"""
This history is based on the versions of hmf_finder itself, but is specific
to the changes made in this file only.

0.9.7 - March 18, 2013
        Modified set_z() so it only does calculations necessary when z changes
        Made calculation of dlnsdlnM in init since it is same for all z
        Removed mean density redshift dependence
        
0.9.5 - March 10, 2013
        The class has been in the works for almost a year now, but it currently
        will calculate a mass function based on any of several fitting functions.
        
"""

###############################################################################
# Some Imports
###############################################################################
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import scipy.integrate as intg
import numpy as np
from scitools.std import * #Must be in this form to work for some reason.

from Distances import Distances


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
    
    Required Input:
        k: numpy vector of wavenumbers for the Transfer function
        Transfer: numpy vector of values for the Transfer function
        M: a float or vector of floats containing the log10(Solar Masses) at which to perform analyses.
        
    Optional Input:
        z:         a float or vector of floats containing the redshift(s) of the analysis.
                   Default z = 0.0
                   
        cosmology: a dictionary of cosmological parameters. 
                   Default cosmology = {"sigma_8":0.812,"n":1,"mean_dens":0.273*2.7755*10**11,
                                        "crit_dens":1.686,"hubble":0.7,"omega_lambda":0.727,
                                        "omega_baryon":0.0456,"omega_cdm":0.2274}
                                        
        k_bounds:  a list defining two values: the lower and upper limit of ln(k). Used to truncate/extend
                   the power spectrum. 
                   Default is empty, ie. no truncation or extension.
                   
        WDM:       a list of warm dark matter particle sizes in keV. 
                   Default is None
                   
        approach:  A string specifying which fitting function to use.
                   Options: ("PS","ST","Jenkins","Warren","Reed03","Reed07",
                             "Tinker", "Courtin", "Crocce","Watson_FoF","Angulo",
                             "Angulo_Bound","user_model")
                   Default: "ST"
        
        user_model:A string defining an equation for a custom fitting function
                   Default empty string.
                   
        get_ngtm:  A bool which chooses whether to calculated the cumulative MF
                   while calculating the MF
                   Default: False
                   
        extrapolate: Whether to use the k_bounds for extrapolation/truncation
                     Default: False
                
    """
    
    
    def __init__(self,M,k,Transfer,z = 0.0,
                 cosmology = {"sigma_8":0.812,
                              "n":1,
                              "mean_dens":0.273*2.7755*10**11,
                              "crit_dens":1.686,
                              "hubble":0.7,
                              "omega_lambda":0.727,
                              "omega_baryon":0.0456,
                              "omega_cdm":0.2274}, 
                 WDM = None, overdensity=178,k_bounds = [],approach="ST",
                 user_model='',extrapolate=False):
        """
        Initializes the cosmology for which to perform the perturbation analysis.      
        """
        
        #======== Save Some Parameters To The Class======== 
        #Run Parameters
        self.approach = approach
        self.z = z
        self.M = M
        self.WDM = WDM
        self.dlogM = np.log10(self.M[1])-np.log10(self.M[0])
        
        #Cosmological Parameters
        self.sigma_8 = cosmology["sigma_8"]
        self.n = cosmology["n"]
        self.mean_dens = cosmology["mean_dens"]
        self.crit_dens = cosmology["crit_dens"]
        self.hubble = cosmology["hubble"]
        self.omega_lambda = cosmology["omega_lambda"]
        self.omega_baryon = cosmology["omega_baryon"]
        self.omega_cdm = cosmology["omega_cdm"]
        self.overdensity = overdensity
        #Other parameters
        self.user_model= user_model

        # Set up a new grid for the transfer function (for romberg integration) 
        self.NewKGrid(k,extrapolate,k_bounds)
        
        #Set up a cosmographic Distances() class with some cosmo parameters.
        self.dist = Distances(self.omega_lambda,self.omega_cdm+self.omega_baryon,self.mean_dens/(self.omega_baryon+self.omega_cdm))

        # Preliminary operations on the transfer function performed, to transform it
        # into a power spectrum that is extrapolated to limits defined by k_bounds
        self.Interpolate(k,Transfer,extrapolate)

        #Normalize the Power Spectrum
        self.Normalize(self.sigma_8)
        
        # Apply further WDM filter if applicable
        if self.WDM:
            self.WDM_PS(self.WDM)
        
        #Calculate Mass Variance
        self.sigma_0 = self.MassVariance(self.M)
        
        self.power_spectrum_0 = self.power_spectrum
        
        #calculate dln(sigma)/dln(M)
        self.dlnsdlnM() 
        
        #Set the redshift, and calculate mass variance and other
        #functions used throughout dependent on z
        self.set_z(z)
        
    def set_z(self,z):
        """
        Wrapper to set the redshift and calculate quantities dependepent on it
        
        Calculates the mean density, sigma, dlnsdlnm, ln(1/sigma) and n_eff
        """
        #Set redshift
        self.z = z
        #Calculate Mean Density at Redshift
        #self.mean_dens = self.dist.MeanDens(z)
        #Calculate the Linear Growth Factor
        if self.z>0:
            self.growth = self.dist.GrowthFactor(self.z)  
        else:
            self.growth = 1
        
        #Apply linear growth to power spectrum
        self.power_spectrum = 2*np.log(self.growth) + self.power_spectrum_0
        
        #Apply linear growth to Mass Variance
        self.sigma = self.sigma_0*self.growth
        
        #Calculate ln(1/sigma)
        self.lnsigma = np.log(1/self.sigma)
        #Calculate effective spectral slope at scale of radius of halo.
        self.n_eff = self.N_Eff()
        
    def NewKGrid(self,k,extrapolate,k_bounds):
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
            
        #Setup the grid and fetch the grid-spacing as well
        self.k, self.dlnk = np.linspace(min_k, max_k, 4097, retstep=True)
        
        
    def Interpolate(self,k,Transfer,extrapolate):
        """
        Interpolates the given Transfer function and transforms it into a Power Spectrum.
        
        Input: k        : an array of values of lnk
               Transfer : an array of values of lnT
               step_size: the step_size between values of lnk required in interpolation.
                  
        Notes: the power spectrum is calculated using P(k) =  A.k^n.T^2, with A=1 in this method
                (which leaves the power spectrum un-normalized). The interpolation is done using
                cubic splines.
        """
        print "In interpolate()"

        # Use cubic spline interpolation over the range of values in the vector transfer function.
        power = self.n*k+2.0*Transfer
        
        #Unfortunately it looks like there's a turn-up at low-k for some transfers which makes the extrapolation silly. 
        #If this is the case, we start when it is nice.
        if extrapolate:
            for i in range(len(k)):
                if power[i] < power[i+1]:
                    start = i
                    break
            if start > 0:
                power = power[(start+100):-1]
                k = k[(start+100):-1]
                spline_order = 1
            else:
                spline_order = 2
            power_function = spline(k,power,k=spline_order)
        else:
            power_function = spline(k,power,k=3)
            
        # Evaluate the power spectrum (un-normalized) at new grid-spacing defined by NewKGrid() 
        self.power_spectrum = power_function(self.k)
        
        
    def Normalize(self,norm_sigma_8):
        """
        Normalize the power spectrum to a given sigma_8
        
        Input: norm_sigma_8: The sigma_8 value to normalize to.
        """
        # Calculate the value of sigma_8 without prior normalization.
        sigma_8 = self.MassVariance(4.*np.pi*8**3*self.mean_dens/3.)

        #sigma_8 will be a list of one element. Extract this element.
        sigma_8 = sigma_8[0]
        
        # Calculate the normalization factor A.
        self.normalization = norm_sigma_8/sigma_8
    
        # Normalize the previously calculated power spectrum.
        self.power_spectrum = 2*np.log(self.normalization)+self.power_spectrum

     
    def Radius(self,M):
        """
        Calculates radius from mass given mean density.
        """
        return (3.*M/(4.*np.pi*self.mean_dens))**(1./3.)
        
        
    def N_Eff(self):
        """
        Calculates the power spectral slope at the scale of the halo radius, using eq. 42 in Lukic et. al 2007.
        """
        
        n_eff = -3.0 *(2.0*self.dlnsdlnm+1.0)
        
        return n_eff
        

    def WDM_PS(self,WDM):
        """
        Tansforms the CDM Power Spectrum into a WDM power spectrum for a given warm particle mass m_x.
        
        NOTE: formula from Bode et. al. 2001 eq. A9
        """

        h = self.hubble
        g_x = 1.5
        m_x = WDM
        nu = 1.12
            
        alpha = 0.049*(self.omega_cdm/0.25)**0.11*(h/0.7)**1.22*(1/m_x)**1.11*(1.5/g_x)**0.29
            
        Transfer = (1+(alpha*np.exp(self.k))**(2*nu))**-(5.0/nu)

        self.power_spectrum = self.power_spectrum+ 2*np.log(Transfer)
        
    def TopHat_WindowFunction(self,m):
        """
        Constructs the window function squared in Fourier space for given radii
        
        Input: R: The radius of the top-hat function
        Output: W_squared: The square of the top-hat window function in Fourier space.
        """

        # Calculate the factor kR, minding to un-log k before use.
        kR =np.exp(self.k)*self.Radius(m)
            
        W_squared = (3*(np.sin(kR)/kR**3 - np.cos(kR)/kR**2))**2
        W_squared[kR<0.01] = 1.0
        if W_squared[0]<0.5 or W_squared[0]>1.5:
            print np.sin(kR[0]), kR[0]**3,np.cos(kR[0]),kR[0]**2,np.sin(kR[0])/kR[0]**3,np.cos(kR[0])/kR[0]**2,W_squared[0]
        return W_squared
    

    def MassVariance(self,M):
        """
        Finds the Mass Variance of M using the top-hat window function.
        
        Input: M: the radius(mass) of the top-hat function (vector).
        Output: sigma: the mass variance.
        """
        
        #If we input a scalar as M, then just make it a one-element list.
        if type(M) == type(3.0):
            M = [M]
            
        # Calculate the integrand of the function. Note that the power spectrum and k values must be
        #'un-logged' before use, and we multiply by k because our steps are in logk. 
        sigma = np.zeros_like(M)
        for i,m in enumerate(M):
            integ = np.exp(self.power_spectrum+3*self.k)*self.TopHat_WindowFunction(m)
            sigma[i] = (0.5/np.pi**2)*intg.romb(integ, dx = self.dlnk)
                  
        return np.sqrt(sigma)
    
     
    def dlnsdlnM(self):
        """
        Uses a top-hat window function to calculate |dlnsigma/dlnM| at a given radius.
        
        Input: R: the radius of the top-hat function. 
        Output: integral: the derivatiave of log sigma with respect to log Mass.
        """
        R = self.Radius(self.M)
        self.dlnsdlnm = np.zeros_like(self.M)
        for i,r in enumerate(R):
            g = np.exp(self.k)*r
            w = (np.sin(g)-g*np.cos(g))*(np.sin(g)*(1-3.0/(g**2))+3.0*np.cos(g)/g) #Derivative of W^2
            integ =  w*np.exp(self.power_spectrum_0-self.k)
            self.dlnsdlnm[i] = (3.0/(2.0*self.sigma_0[i]**2*np.pi**2*r**4))*intg.romb(integ,dx=self.dlnk)
            
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
            "user_model":self.nufnu_user_model
            }

        nufnu = nufnu_dict[self.approach]
        self.vfv = nufnu()
        
        leftovers = self.mean_dens/self.M**2
        
        dndm = self.vfv *leftovers*np.abs(self.dlnsdlnm)
        
        return dndm
    
    def MassFunction(self):
        """
        Uses the Press Schechter approach with a spherical top-hat to calculate the mass function.
        
        Output: mass_function log10(dn/dlnM).
        """
        mass_function = np.log(10.0)*self.M*self.dndM()
               
        return mass_function
    
    def NgtM(self,mass_function):
        """
        Integrates the mass function above a certain mass to calculate the number of haloes above a certain mass
        
        INPUT:
        mass_function: an array containing the mass function (log10)
        
        OUTPUT:
        ngtm: the number of haloes greater than M for all M in self.M
        """
        #Initialize the function
        ngtm = np.zeros_like(mass_function)
        
        #set M and mass_function within computed range
        M = self.M[np.logical_not(np.isnan(mass_function))]
        mass_function = mass_function[np.logical_not(np.isnan(mass_function))]
        
        #Interpolate the mass_function - this is in log-log space.
        mf = spline(np.log10(M),mass_function,k=3)
        
        #Define max_M as either 17 or the maximum set by user
        max_M = np.max([17,np.log10(M[-1])])
        
        for i,m in enumerate(self.M):
            if np.isnan(m):
                ngtm[i] = np.nan
            else:
                #Set up new grid with 4097 steps from m to M=17
                M_new, dlnM = np.linspace(np.log10(m), max_M, 4097,retstep=True)
                mf_new = mf(M_new)
                
                #Here we multiply by the mass because we are in log steps
                integ = 10**(M_new + mf_new)
                #Divide by ln(10) because we are in log10 steps
                ngtm[i] = np.log10(intg.romb(integ, dx = dlnM)/np.log(10))
        return ngtm
    
    def MgtM(self,mass_function):
        """
        Integrates the mass function above a certain mass to calculate the number of haloes above a certain mass
        
        INPUT:
        mass_function: an array containing the mass function (log10)
        
        OUTPUT:
        mgtm: the mass in haloes greater than M for all M in self.M
        """
        
        #Initialize the function
        mgtm = np.zeros_like(mass_function)
        
        #set M and mass_function within computed range
        M = self.M[np.logical_not(np.isnan(mass_function))]
        mass_function = mass_function[np.logical_not(np.isnan(mass_function))]
        
        #Interpolate the mass_function - this is in log-log space.
        mf = spline(np.log10(M),mass_function,k=3)
        
        #Define max_M as either 17 or the maximum set by user
        max_M = np.max([17,np.log10(M[-1])])
        
        for i,m in enumerate(self.M):
            if np.isnan(m):
                mgtm[i] = np.nan
            else:
                #Set up new grid with 4097 steps from m to M=17
                M_new, dlnM = np.linspace(np.log10(m), max_M, 4097,retstep=True)
                mf_new = mf(M_new)
                integ = 10**(2*M_new + mf_new)
                mgtm[i] = np.log10(intg.romb(integ, dx = dlnM)/np.log(10))
        
        return mgtm
    
    def NltM(self,mass_function):
        """
        Integrates the mass function below a certain mass to calculate the number of haloes below that mass
        
        INPUT:
        mass_function: an array containing the mass function (log10)
        
        OUTPUT:
        nltm: the number of haloes less than M for all M in self.M
        """
        
        #Initialize the function
        nltm = np.zeros_like(mass_function)
        
        #set M and mass_function within computed range
        M = self.M[np.logical_not(np.isnan(mass_function))]
        mass_function = mass_function[np.logical_not(np.isnan(mass_function))]
        
        #Interpolate the mass_function - this is in log-log space.
        mf = spline(np.log10(M),mass_function,k=3)
        
        #Define min_M as either 3 or the minimum set by user
        min_M = np.min([3,np.log10(M[0])])
        
        for i,m in enumerate(self.M):
            if np.isnan(m):
                nltm[i] = np.nan
            else:
                #Set up new grid with 4097 steps from m to M=17
                M_new, dlnM = np.linspace(min_M,np.log10(m), 4097,retstep=True)
                mf_new = mf(M_new)
                integ = 10**(M_new + mf_new)
                nltm[i] = np.log10(intg.romb(integ, dx = dlnM)/np.log(10))
        
        return nltm
    
    def MltM(self,mass_function):
        """
        Integrates the mass function above a certain mass to calculate the number of haloes above a certain mass
        
        INPUT:
        mass_function: an array containing the mass function (log10)
        
        OUTPUT:
        mgtm: the mass in haloes greater than M for all M in self.M
        """
        
        #Initialize the function
        mltm = np.zeros_like(mass_function)
        
        #set M and mass_function within computed range
        M = self.M[np.logical_not(np.isnan(mass_function))]
        mass_function = mass_function[np.logical_not(np.isnan(mass_function))]
        
            
        #Interpolate the mass_function - this is in log-log space.
        mf = spline(np.log10(M),mass_function,k=3)
        
        #Define max_M as either 17 or the maximum set by user
        min_M = np.min([3,np.log10(M[0])])
        
        for i,m in enumerate(self.M):
            if np.isnan(m):
                mltm[i] = np.nan
            else:
                #Set up new grid with 4097 steps from m to M=17
                M_new, dlnM = np.linspace(min_M,np.log10(m), 4097,retstep=True)
                mf_new = mf(M_new)
                integ = 10**(2*M_new + mf_new)
                mltm[i] = np.log10(intg.romb(integ, dx = dlnM)/np.log(10))
        
        return mltm
    
    def how_big(self, mass_function):
        # Calculates how big a box must be to expect 
        # one halo for given mass function

        return 10**(-mass_function/3)
    
    def nufnu_PS(self):
        """
        Computes the function nu*f(nu) for the Press-Schechter approach at a given radius.
        
        Input R: radius of the top-hat function
        Output: f_of_nu: the function nu*f(nu) for the PS approach.
        """
        
        vfv = np.sqrt(2.0/np.pi)*(self.crit_dens/self.sigma)*np.exp(-0.5*(self.crit_dens/self.sigma)**2)   
        
        return vfv 
    
    def nufnu_ST(self):
        """
        Finds the Sheth Tormen vf(v) 
        
        Input R: radius of the top-hat function
        Output: vfv: the Sheth-Tormen mass function fit.
        """

        nu = self.crit_dens/self.sigma
        a = 0.707
        
        vfv = 0.3222*np.sqrt(2.0*a/np.pi)*nu*np.exp(-(a*nu**2)/2.0)*(1+(1.0/(a*nu**2))**0.3)
        
        return vfv
    
    def nufnu_Jenkins(self):
        """
        Finds the Jenkins empirical vf(v) 
        
        Output: vfv: the Jenkins mass function fit.
        """
        
        vfv = 0.315*np.exp(-np.abs(self.lnsigma+0.61)**3.8)
        print self.lnsigma < -1.2
        print self.lnsigma > 1.05
        #Conditional on sigma range.
        vfv[np.logical_or(self.lnsigma<-1.2,self.lnsigma>1.05)] = np.NaN
        return vfv
    
    def nufnu_Warren(self):
        """
        Finds the Warren empirical vf(v) 
        
        Input R: radius of the top-hat function
        Output: vfv: the Warren mass function fit.
        """
        
        vfv = 0.7234*((1.0/self.sigma)**1.625+0.2538)*np.exp(-1.1982/self.sigma**2)
        
        vfv[np.logical_or(self.M/self.hubble < 10**10, self.M/self.hubble > 10**15)] = np.NaN
        return vfv
    
    def nufnu_Reed03(self):
        """
        Finds the Reed 2003 empirical vf(v) 
        
        Input R: radius of the top-hat function
        Output: vfv: the Reed 2003 mass function fit.
        
        NOTE: Only valid from -1.7 < ln sigma^-1 < 0.9
        """
                
        ST_Fit = self.nufnu_ST()
        
        vfv = ST_Fit*np.exp(-0.7/(self.sigma*np.cosh(2.0*self.sigma)**5))
        
        vfv[np.logical_or(self.lnsigma<-1.7, self.lnsigma>0.9)] = np.NaN
        return vfv
    
    def nufnu_Reed07(self):
        """
        Finds the Reed 2007 empirical vf(v) 
        
        Input R: radius of the top-hat function
        Output: vfv: the Reed 2003 mass function fit.
        
        NOTE: Only valid from -1.7 < ln sigma^-1 < 0.9
        """
        nu = self.crit_dens/self.sigma
        
        G_1 = np.exp(-((1.0/self.sigma-0.4)**2)/(2*0.6**2))
        G_2 = np.exp(-((1.0/self.sigma-0.75)**2)/(2*0.2**2))
        
        c = 1.08
        a = 0.764/c
        A = 0.3222
        p = 0.3
        
        vfv = A*np.sqrt(2.0*a/np.pi)*(1.0+(1.0/(a*nu**2))**p+0.6*G_1+0.4*G_2)*nu*np.exp(-c*a*nu**2/2.0-0.03*nu**0.6/(self.n_eff+3)**2)
        
        vfv[np.logical_or(self.lnsigma < -0.5, self.lnsigma > 1.2)] = np.NaN
        
        return vfv


    def nufnu_Angulo(self):

        vfv = 0.201*(2.08/self.sigma)**1.7 * np.exp(-1.172/self.sigma**2)
        return vfv

    def nufnu_Angulo_Bound(self):
        vfv = 0.265*(1.675/self.sigma)**1.9 * np.exp(-1.4/self.sigma**2)
        return vfv

    def nufnu_Tinker(self):

        #if self.overdensity==178:
        A_0 = 0.186
        a_0 = 1.47
        b_0 = 2.57
        c_0 = 1.19
        #else:
        #    a_0 = 1.43 + (np.log10(self.overdensity) - 2.3)**1.5 
        #    b_0 = 1.0 + (np.log10(self.overdensity) - 1.6)**(-1.5)
        #    c_0 = 1.2 + (np.log10(self.overdensity) - 2.35)**1.6
        #    if self.overdensity < 1600:
        #        A_0 = 0.1*np.log10(self.overdensity) - 0.05
        #    else:
        #        A_0 = 0.26#

        #print "A_0: ", A_0
        #print "a_0:", a_0
        #print "b_0:", b_0
        #print "c_0:", c_0
        A = A_0 * (1+self.z)**(-0.14)
        a = a_0*(1+self.z)**(-0.06)
        alpha = np.exp(-(0.75/np.log(self.overdensity/75))**1.2)
        b = b_0 * (1+self.z)**(-alpha)
        c = c_0

        
        vfv = A*((self.sigma/b)**(-a) + 1)*np.exp(-c/self.sigma**2)
        return vfv

    def Watson_Gamma(self):
        C = 0.947*np.exp(0.023*(self.overdensity/178-1))
        d = -0.456*self.dist.Omega_M(self.z) - 0.139
        p = 0.072
        q = 2.13

        return C*(self.overdensity/178)**d * np.exp(p*(1-self.overdensity/178)/self.sigma**q)
        

    def nufnu_Watson_FoF(self):
        vfv = 0.282 * ((2.163/self.sigma)**1.406 + 1)*np.exp(-1.21/self.sigma**2)
        vfv[np.logical_or(self.lnsigma<-0.55 ,self.lnsigma>1.31)] = np.NaN
        return vfv

    def nufnu_Watson(self):
        
        if self.z==0:
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
            A = self.dist.Omega_M(self.z)*(1.097*(1+self.z)**(-3.216)+0.074)
            alpha = self.dist.Omega_M(self.z)*(3.136*(1+self.z)**(-3.058)+2.349)
            beta = self.dist.Omega_M(self.z)*(5.907*(1+self.z)**(-3.599)+2.344)
            gamma = 1.318

        vfv = self.Watson_Gamma()*A*((beta/self.sigma)**alpha + 1)*np.exp(-gamma/self.sigma**2)
        
        #vfv[np.logical_or(self.lnsigma < -0.55, self.lnsigma > 1.05)] = np.NaN
            
        return vfv

    def nufnu_Crocce(self):
        
        A = 0.58*(1+self.z)**(-0.13)
        a = 1.37*(1+self.z)**(-0.15)
        b = 0.3*(1+self.z)**(-0.084)
        c = 1.036*(1+self.z)**(-0.024)

        vfv = A*(self.sigma**(-a) + b)*np.exp(-c/self.sigma**2)
        return vfv

    def nufnu_Courtin(self):
        A = 0.348
        a = 0.695
        p = 0.1
        d_c = 1.673
        
        vfv = A*np.sqrt(2*a/np.pi)*(d_c/self.sigma)*(1+(d_c/(self.sigma*np.sqrt(a)))**(-2*p))*np.exp(-d_c**2*a/(2*self.sigma**2))
        return vfv
    
    def nufnu_user_model(self):
        """
        Calculates vfv based on a user-input model.
        """
        from scitools.StringFunction import StringFunction
       
        f = StringFunction(self.user_model,globals=globals())
        
        return f(self.sigma)
    

        

        
        

        
        
    
