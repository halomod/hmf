"""
Module containing Warm Dark Matter models.

This module contains both WDM Components (basic WDM models and also recalibrators for
the HMF) and Frameworks (Transfer and MassFunction). The latter inject WDM modelling
into the standard CDM Frameworks, and provide an example of how one would go about this
for other alternative cosmologies.
"""

import astropy.units as u
import numpy as np

from .._internals._cache import cached_quantity, parameter
from .._internals._framework import Component, get_mdl, pluggable
from ..cosmology.cosmo import Planck15
from ..density_field.transfer import Transfer as _Tr
from ..mass_function.hmf import MassFunction as _MF


# ===============================================================================
# Model Components
# ===============================================================================
@pluggable
class WDM(Component):
    r"""
    Base class for all WDM components.

    Do not use this class directly. The primary purpose of the WDM Component is
    to modify the transfer function. Thus, the only requisite method to define
    in any given subclass is :meth:`transfer`, which calculates this quantity in the
    proposed WDM model.

    Parameters
    ----------
    mx : float
        Mass of the particle in keV

    cosmo : `hmf.cosmo.Cosmology` instance
        A cosmology.

    z : float
        Redshift.

    \*\*model_parameters : unpack-dict
        Parameters specific to a model.
        To see the default values, check the :attr:`_defaults`
        class attribute.
    """

    def __init__(self, mx, cosmo=Planck15, z=0, **model_params):
        self.mx = mx
        self.cosmo = cosmo
        self.rho_mean = (1 + z) ** 3 * (
            self.cosmo.Om0 * self.cosmo.critical_density0 / self.cosmo.h ** 2
        ).to(u.solMass / u.Mpc ** 3).value
        self.Oc0 = cosmo.Om0 - cosmo.Ob0

        super(WDM, self).__init__(**model_params)

    def transfer(self, lnk):
        """Transfer function for WDM models.

        Parameters
        ----------
        lnk : array
            The wavenumbers *k/h* corresponding to  ``power_cdm``.

        Returns
        -------
        transfer : array_like
            The WDM transfer function at `lnk`.

        """
        raise NotImplementedError(
            "You shouldn't call the WDM class, and any subclass should define the "
            "transfer method."
        )


class Viel05(WDM):
    r"""
    Transfer function from Viel 2005 (which is exactly the same as Bode et al. 2001).

    Formula from Bode et. al. 2001 eq. A9.

    Parameters
    ----------
    mx : float
        Mass of the particle in keV
    cosmo : `hmf.cosmo.Cosmology` instance
        A cosmology.
    z : float
        Redshift.
    \*\*model_parameters : unpack-dict
        Parameters specific to a model. Available parameters are as follows.
        To see the default values, check the :attr:`_defaults`
        class attribute.

        :mu:
        :g_x:
    """

    _defaults = {"mu": 1.12, "g_x": 1.5}

    def transfer(self, k):
        """Compute the modified WDM transfer function."""
        return (1 + (self.lam_eff_fs * k) ** (2 * self.params["mu"])) ** (
            -5.0 / self.params["mu"]
        )

    @property
    def lam_eff_fs(self):
        """Effective free-streaming scale.

        From Schneider+2013, Eq. 6
        """
        return (
            0.049
            * self.mx ** -1.11
            * (self.Oc0 / 0.25) ** 0.11
            * (self.cosmo.h / 0.7) ** 1.22
            * (1.5 / self.params["g_x"]) ** 0.29
        )

    @property
    def m_fs(self):
        """
        Free-streaming mass scale.

        From Schneider+2012, Eq. 7
        """
        return (4.0 / 3.0) * np.pi * self.rho_mean * (self.lam_eff_fs / 2) ** 3

    @property
    def lam_hm(self):
        """
        Half-mode scale.

        From Schneider+2012, Eq. 8.
        """
        return (
            2
            * np.pi
            * self.lam_eff_fs
            * (2 ** (self.params["mu"] / 5) - 1) ** (-0.5 / self.params["mu"])
        )

    @property
    def m_hm(self):
        """
        Half-mode mass scale.

        From Schneider+2013, Eq. 8
        """
        return (4.0 / 3.0) * np.pi * self.rho_mean * (self.lam_hm / 2) ** 3


class Bode01(Viel05):
    """The WDM model of Bode et al. (2001)."""

    pass


viel_model = Viel05(mx=1.0)


@pluggable
class WDMRecalibrateMF(Component):
    r"""
    Base class for Components that emulate the effect of WDM on the HMF empirically.

    Required method is :meth:`dndm_alter`.

    Parameters
    ----------
    m : array_like
        Masses at which the HMF is calculated.
    dndm0 : array_like
        The original HMF at `m`.
    wdm : :class:`WDM` subclass instance
        An instance of :class:`WDM` providing a Warm Dark Matter model.
    \*\*model_parameters : unpack-dict
        Parameters specific to a model.
        To see the default values, check the :attr:`_defaults`
        class attribute.
    """

    def __init__(self, m, dndm0, wdm=viel_model, **model_parameters):
        self.m = m
        self.dndm0 = dndm0
        self.wdm = wdm
        super(WDMRecalibrateMF, self).__init__(**model_parameters)

    def dndm_alter(self):
        """Alter the CDM dn/dm to impose WDM modeling."""
        pass


class Schneider12_vCDM(WDMRecalibrateMF):
    r"""
    Schneider+2012 recalibration of the CDM HMF.

    Parameters
    ----------
    m : array_like
        Masses at which the HMF is calculated.
    dndm0 : array_like
        The CDM HMF at `m`.
    wdm : :class:`WDM` subclass instance
        An instance of :class:`WDM` providing a Warm Dark Matter model.
    \*\*model_parameters : unpack-dict
        Parameters specific to this model: **beta**.
        To see the default values, check the :attr:`_defaults`
        class attribute.
    """

    _defaults = {"beta": 1.16}

    def dndm_alter(self):
        return self.dndm0 * (1 + self.wdm.m_hm / self.m) ** (-self.params["beta"])


class Schneider12(WDMRecalibrateMF):
    r"""
    Schneider+2012 recalibration of the WDM HMF.

    Parameters
    ----------
    m : array_like
        Masses at which the HMF is calculated.
    dndm0 : array_like
        The original WDM HMF at `m`.
    wdm : :class:`WDM` subclass instance
        An instance of :class:`WDM` providing a Warm Dark Matter model.
    \*\*model_parameters : unpack-dict
        Parameters specific to this model: **alpha**.
        To see the default values, check the :attr:`_defaults`
        class attribute.
    """

    _defaults = {"alpha": 0.6}

    def dndm_alter(self):
        return self.dndm0 * (1 + self.wdm.m_hm / self.m) ** (-self.params["alpha"])


class Lovell14(WDMRecalibrateMF):
    r"""
    Lovell+2014 recalibration of the WDM HMF.

    Parameters
    ----------
    m : array_like
        Masses at which the HMF is calculated.
    dndm0 : array_like
        The original HMF at `m`.
    wdm : :class:`WDM` subclass instance
        An instance of :class:`WDM` providing a Warm Dark Matter model.
    \*\*model_parameters : unpack-dict
        Parameters specific to this model: **beta**.
        To see the default values, check the :attr:`_defaults`
        class attribute.
    """

    _defaults = {"beta": 0.99, "gamma": 2.7}

    def dndm_alter(self):
        return self.dndm0 * (1 + self.params["gamma"] * self.wdm.m_hm / self.m) ** (
            -self.params["beta"]
        )


# ===============================================================================
# Frameworks
# ===============================================================================
class TransferWDM(_Tr):
    """
    A subclass of :class:`hmf.transfer.Transfer` that mixes in WDM capabilities.

    This replaces the standard CDM quantities with WDM-derived ones, where relevant.

    In addition to the parameters directly passed to this class, others are available
    which are passed on to its superclass. To read a standard documented list of (all)
    parameters, use :meth:`parameter_info`. If you want to just see the plain
    list of available parameters, use :meth`get_all_parameters`. To see the
    actual defaults for each parameter, use :meth:`get_all_parameter_defaults`.
    """

    def __init__(
        self, wdm_mass=3.0, wdm_model=Viel05, wdm_params=None, **transfer_kwargs
    ):
        wdm_params = wdm_params or {}

        # Call standard transfer
        super(TransferWDM, self).__init__(**transfer_kwargs)

        # Set given parameters
        self.wdm_mass = wdm_mass
        self.wdm_model = wdm_model
        self.wdm_params = wdm_params

    @parameter("model")
    def wdm_model(self, val):
        """
        A model for the WDM effect on the transfer function.

        :type: str or :class:`WDM` subclass
        """
        return get_mdl(val, WDM)

    @parameter("param")
    def wdm_params(self, val):
        """
        Parameters of the WDM model.

        :type: dict
        """
        return val

    @parameter("param")
    def wdm_mass(self, val):
        """
        Mass of the WDM particle.

        :type: float
        """
        try:
            val = float(val)
        except ValueError:
            raise ValueError("wdm_mass must be a number (", val, ")")

        if val <= 0:
            raise ValueError("wdm_mass must be > 0 (", val, ")")
        return val

    @cached_quantity
    def wdm(self):
        """
        The instantiated WDM model.

        Contains quantities relevant to WDM.
        """
        return self.wdm_model(
            mx=self.wdm_mass, cosmo=self.cosmo, z=self.z, **self.wdm_params
        )

    @cached_quantity
    def _unnormalised_lnT(self):
        return super(TransferWDM, self)._unnormalised_lnT + np.log(
            self.wdm.transfer(self.k)
        )


class MassFunctionWDM(_MF, TransferWDM):
    """
    A subclass of :class:`hmf.MassFunction` that mixes in WDM capabilities.

    This replaces the standard CDM quantities with WDM-derived ones, where relevant.

    In addition to the parameters directly passed to this class, others are available
    which are passed on to its superclass. To read a standard documented list of (all)
    parameters, use :meth:`parameter_info`. If you want to just see the plain
    list of available parameters, use :meth`get_all_parameters`. To see the
    actual defaults for each parameter, use :meth:`get_all_parameter_defaults`.
    """

    def __init__(self, alter_model=None, alter_params=None, **kwargs):
        super(MassFunctionWDM, self).__init__(**kwargs)

        self.alter_model = alter_model
        self.alter_params = alter_params or {}

    @parameter("switch")
    def alter_model(self, val):
        """
        A model for empirical recalibration of the HMF.

        :type: None, str, or :class`WDMRecalibrateMF` subclass.
        """
        if val is None:
            return None
        return get_mdl(val, WDMRecalibrateMF)

    @parameter("param")
    def alter_params(self, val):
        """Model parameters for `alter_model`."""
        return val

    @cached_quantity
    def dndm(self):
        r"""
        The number density of haloes in WDM, ``len=len(m)``.

        Units of :math:`h^4 M_\odot^{-1} Mpc^{-3}`
        """
        dndm = super(MassFunctionWDM, self).dndm

        if self.alter_model is not None:
            alter = self.alter_model(
                m=self.m, dndm0=dndm, wdm=self.wdm, **self.alter_params
            )
            dndm = alter.dndm_alter()

        return dndm
