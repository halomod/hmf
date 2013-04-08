'''
Created on Jan 21, 2013

@author: Steven
'''
import numpy as np
import scipy.integrate as intg
#import cosmolopy.density as cd

class Distances(object):
    """
    A Class to calculate distances and other cosmological quantities.
    
    The cosmology is input to the init function (at the moment only a flat cosmology)
    """        
    
    def __init__(self,omega_lambda = 0.7,omega_mass = 0.3,crit_dens = 27.755*10**11):
        self.omega_lambda = omega_lambda
        self.omega_mass = omega_mass
        self.omega_k = 1 - omega_lambda - omega_mass
        self.crit_dens = crit_dens
        
        self.D_h = 9.26*10**25 #h^{-1}m Hogg eq.4
        
    def MeanDens(self,z):
        return self.Omega_M(z)*self.crit_dens
    
    def ScaleFactor(self,z):
        """
        Calculates the scale factor at a given redshift
        
        Input: z: the redshift.
        Output a: the scale factor at the given redshift.
        """
        
        a = 1.0/(1.0+z)

        return a
    
    def zofa(self,a):
        """
        Returns the redshift of a particular scale factor
        """
        
        z = 1.0/a -1.0
        
        return z
    
    def HubbleFunction(self,z):
        """
        Finds the hubble parameter at z normalized by the current hubble constant: E(z).
        
        Source: Hogg 1999, Eq. 14
        
        Input: z: redshift
        Output: hubble: the hubble parameter at z divided by H_0 (ie. we don't need H_0)
        """
        
        hubble = np.sqrt(self.omega_mass*(1+z)**3 + self.omega_k*(1+z)**2 + self.omega_lambda)
        
        return hubble
    

    def Omega_M(self,z):
        """
        Finds Omega_M as a function of redshift
        """
        top = self.omega_mass * (1+z)**3
        bottom = self.omega_mass * (1+z)**3 + self.omega_lambda + self.omega_k*(1+z)**2
        
        return top/bottom
                    
    def ComovingDistance(self,z):
        """
        The comoving distance to an object at redshift z
        """
        
        
    def Dplus(self,redshifts):
        """
        Finds the factor D+(a), from Lukic et. al. 2007, eq. 8.
        
        Uses romberg integration with a suitable step-size. 
        
        Input: z: redshift.
        
        Output: dplus: the factor.
        """
        if type(redshifts) is type (0.4):
            redshifts = [redshifts]
            
        dplus = np.zeros_like(redshifts)
        for i,z in enumerate(redshifts):
            step_size = self.StepSize(0.0000001,self.ScaleFactor(z))
            a_vector = np.arange(0.0000001,self.ScaleFactor(z),step_size)
            integrand = 1.0/(a_vector*self.HubbleFunction(self.zofa(a_vector)))**3
            
            integral = intg.romb(integrand,dx=step_size)
            dplus[i] = 5.0*self.omega_mass*self.HubbleFunction(z)*integral/2.0
        
        return dplus
    
    def GrowthFactor(self,z):
        """
        Finds the factor d(a) = D+(a)/D+(a=1), from Lukic et. al. 2007, eq. 8.
        
        Input: z: redshift.
        
        Output: growth: the growth factor.
        """
        
        growth = self.Dplus(z)/self.Dplus(0.0)

        return growth
    
    def StepSize(self,mini,maxi):
        """
        Calculates a suitable step size for romberg integration given data limits
        """
        
        p = 13
        
        while (maxi-mini)/(2**p+1) < 10**(-5):
            p=p-1
            
        step_size = (maxi-mini)/(2**p+1.0)
        
        return step_size