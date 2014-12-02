'''
Created on 02/12/2014

@author: Steven

Module containing WDM models
'''
import copy
import sys

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
    def __init__(self, mx, **model_params):
        '''
        Constructor
        '''
        self.mx = mx

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
