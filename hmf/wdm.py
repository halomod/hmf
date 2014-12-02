'''
Created on 02/12/2014

@author: Steven

Module containing WDM models
'''
import copy
import sys
import numpy as np

def get_wdm(name, **kwargs):
    """
    Returns the correct subclass of :class:`WDM`.
    
    Parameters
    ----------
    name : str
        The class name of the appropriate model
        
    \*\*kwargs : 
        Any parameters for the instantiated fit (including model parameters)
    """
    try:
        return getattr(sys.modules[__name__], name)(**kwargs)
    except AttributeError:
        raise AttributeError(str(name) + "  is not a valid WDM class")

class WDM(object):
    '''
    Abstract base class for all WDM models
    '''

    _defaults = {}
    def __init__(self, mx, omegac, h, rho_mean, **model_params):
        '''
        Constructor
        '''
        self.mx = mx
        self.omegac = omegac
        self.h = h
        self.rho_mean = rho_mean

        # Check that all parameters passed are valid
        for k in model_params:
            if k not in self._defaults:
                raise ValueError("%s is not a valid argument for the %s WDM model" % (k, self.__class__.__name__))

        # Gather model parameters
        self.params = copy.copy(self._defaults)
        self.params.update(model_params)

    def transfer(self, lnk):
        """
        Transfer function for WDM models
                
        Parameters
        ----------
        lnk : array
            The wavenumbers *k/h* corresponding to  ``power_cdm``.
            
        m_x : float
            The mass of the single-species WDM particle in *keV*
            
        power_cdm : array
            The normalised power spectrum of CDM.
            
        
            
        h : float
            Hubble parameter
            
        omegac : float
            The dark matter density as a ratio of critical density at the current 
            epoch.
        
        Returns
        -------
        power_wdm : array
            The normalised WDM power spectrum at ``lnk``.
            
        """
        pass

class Bode01(WDM):
    _defaults = {"g_x":1.5,
                 "nu":1.12}
    def transfer(self, lnk):
        g_x = self.params['g_x']
        nu = self.params["nu"]

        alpha = 0.049 * (omegac / 0.25) ** 0.11 * (h / 0.7) ** 1.22 * (1 / m_x) ** 1.11 * (1.5 / g_x) ** 0.29

        transfer = (1 + (alpha * np.exp(lnk)) ** (2 * nu)) ** -(5.0 / nu)



class Viel05(WDM):
    _defaults = {"mu":1.12}
    def transfer(self, lnk):
        return (1 + (self.lam_eff_fs * np.exp(lnk)) ** (2 * self.params["mu"])) ** (-5.0 / self.params["mu"])

    @property
    def lam_eff_fs(self):
        return 0.049 * self.mx ** -1.11 * (self.omegac / 0.25) ** 0.11 * (self.h / 0.7) ** 1.22

    @property
    def m_fs(self):
        return (4.0 / 3.0) * np.pi * self.rho_mean * (self.lam_eff_fs / 2) ** 3

    @property
    def lam_hm(self):
        return 2 * np.pi * self.lam_eff_fs * (2 ** (self.params['mu'] / 5) - 1) ** (-0.5 / self.params['mu'])

    @property
    def m_hm(self):
        return (4.0 / 3.0) * np.pi * self.rho_mean * (self.lam_hm / 2) ** 3
