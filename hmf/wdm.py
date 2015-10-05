'''
Created on 02/12/2014

@author: Steven

Module containing WDM models
'''
import numpy as np
from transfer import Transfer
from hmf import MassFunction
from _cache import parameter, cached_property
from numpy import issubclass_
from _framework import Component, get_model
from cosmo import h_unit
import astropy.units as u
#===============================================================================
# Model Components
#===============================================================================
class WDM(Component):
    '''
    Abstract base class for all WDM models
    '''
    def __init__(self, mx, cosmo,z, **model_params):
        '''
        Constructor
        '''
        self.mx = mx
        self.cosmo = cosmo
        self.rho_mean = (1+z)**3 * h_unit ** 2 * (self.cosmo.Om0 * self.cosmo.critical_density0 / self.cosmo.h ** 2).to(u.MsolMass / u.Mpc ** 3) * 1e6
        self.Oc0 = cosmo.Om0 - cosmo.Ob0

        super(WDM, self).__init__(**model_params)

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



class Viel05(WDM):
    """
    Transfer function from Viel 2005 (which is exactly the same as Bode et al.
    2001).

    Formula from Bode et. al. 2001 eq. A9
    """

    _defaults = {"mu":1.12,
                 "g_x":1.5}

    def transfer(self, k):
        return (1 + (self.lam_eff_fs * k) ** (2 * self.params["mu"])) ** (-5.0 / self.params["mu"])

    @property
    def lam_eff_fs(self):
        return 0.049 * self.mx ** -1.11 * (self.Oc0 / 0.25) ** 0.11 * (self.cosmo.h / 0.7) ** 1.22 * (1.5 / self.params['g_x']) ** 0.29 * u.Mpc/h_unit

    @property
    def m_fs(self):
        return (4.0 / 3.0) * np.pi * self.rho_mean * (self.lam_eff_fs / 2) ** 3

    @property
    def lam_hm(self):
        return 2 * np.pi * self.lam_eff_fs * (2 ** (self.params['mu'] / 5) - 1) ** (-0.5 / self.params['mu'])

    @property
    def m_hm(self):
        return (4.0 / 3.0) * np.pi * self.rho_mean * (self.lam_hm / 2) ** 3

class Bode01(Viel05):
    pass



class WDMRecalibrateMF(Model):
    """
    A model that merely transforms the mass function for WDM.
    """
    def __init__(self, M, dndm0, wdm, **model_parameters):
        self.M = M
        self.dndm0 = dndm0
        self.wdm = wdm
        super(WDMRecalibrateMF, self).__init__(**model_parameters)

    def dndm_alter(self):
        pass

class Schneider12_vCDM(WDMRecalibrateMF):
    _defaults = {"beta":1.16}

    def dndm_alter(self):
        return self.dndm0 * (1 + self.wdm.m_hm / self.M) ** (-self.params["beta"])

class Schneider12(WDMRecalibrateMF):
    _defaults = {"alpha":0.6}

    def dndm_alter(self):
        return self.dndm0 * (1 + self.wdm.m_hm / self.M) ** (-self.params["alpha"])

class Lovell14(WDMRecalibrateMF):
    _defaults = {"beta":0.99, "gamma":2.7}

    def dndm_alter(self):
        return self.dndm0 * (1 + self.params["gamma"] * self.wdm.m_hm / self.M) ** (-self.params["beta"])



#===============================================================================
# Frameworks
#===============================================================================
class TransferWDM(Transfer):
    def __init__(self, wdm_mass=3.0, wdm_transfer=Viel05, wdm_params={},
                 **transfer_kwargs):
        # Call standard transfer
        super(TransferWDM, self).__init__(**transfer_kwargs)

        # Set given parameters
        self.wdm_mass = wdm_mass
        self.wdm_transfer = wdm_transfer
        self.wdm_params = wdm_params

    #===========================================================================
    # Parameters
    #===========================================================================
    @parameter
    def wdm_transfer(self, val):
        if not issubclass_(val, WDM) and not isinstance(val, basestring):
            raise ValueError("wdm_transfer must be a WDM subclass or string, got %s" % type(val))
        return val

    @parameter
    def wdm_params(self, val):
        return val

    @parameter
    def wdm_mass(self, val):
        try:
            val = float(val)
        except ValueError:
            raise ValueError("wdm_mass must be a number (", val, ")")

        if val <= 0:
            raise ValueError("wdm_mass must be > 0 (", val, ")")
        return val

    #===========================================================================
    # Derived properties
    #===========================================================================
    @cached_property("cosmo", "wdm_mass", "wdm_transfer", "wdm_params")
    def _wdm(self):
        if issubclass_(self.wdm_transfer, WDM):
            return self.wdm_transfer(self.wdm_mass, self.cosmo,self.z,
                                     **self.wdm_params)
        elif isinstance(self.wdm_transfer, basestring):
            return get_wdm(self.wdm_transfer, mx=self.wdm_mass, cosmo=self.cosmo,
                           z=self.z,**self.wdm_params)

    @cached_property("_wdm")
    def _unnormalised_power(self):
        """
        Normalised log power at :math:`z=0`
        """
        powercdm = super(TransferWDM, self)._unnormalised_power
        return self._wdm.transfer(self.k) ** 2 * powercdm

class MassFunctionWDM(MassFunction, TransferWDM):
    def __init__(self, alter_dndm=None, alter_params=None, **kwargs):
        super(MassFunctionWDM, self).__init__(**kwargs)

        self.alter_dndm = alter_dndm
        self.alter_params = alter_params or {}

    @parameter
    def alter_dndm(self, val):
        if not np.issubclass_(val, WDMRecalibrateMF) and val is not None and not np.issubclass_(val, basestring):

            raise TypeError("alter_dndm must be a WDMRecalibrateMF subclass, string, or None")
        else:
            return val

    @parameter
    def alter_params(self, val):
        return val

    @cached_property("alter_dndm", "M", "_wdm", "alter_params")
    def dndm(self):
        dndm = super(MassFunctionWDM, self).dndm

        if self.alter_dndm is not None:
            if np.issubclass_(self.alter_dndm, WDMRecalibrateMF):
                alter = self.alter_dndm(M=self.M, dndm0=dndm, wdm=self._wdm,
                                        **self.alter_params)
            else:
                alter = get_model(self.alter_dndm, __name__, M=self.M,
                                  dndm0=dndm, wdm=self._wdm, **self.alter_params)
            dndm = alter.dndm_alter()

        return dndm
