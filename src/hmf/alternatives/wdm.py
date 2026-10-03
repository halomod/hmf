"""
Module containing Warm Dark Matter models.

This module contains both WDM Components (basic WDM models and also recalibrators for
the HMF) and Frameworks (Transfer and MassFunction). The latter inject WDM modelling
into the standard CDM Frameworks, and provide an example of how one would go about this
for other alternative cosmologies.
"""

import warnings
from typing import ClassVar, override

import astropy.units as u
import numpy as np

from .._internals import _references as refs
from .._internals._cache import cached_quantity, parameter
from .._internals._framework import Component, get_mdl, pluggable
from ..cosmology.cosmo import Planck15
from ..density_field.transfer import Transfer
from ..mass_function.hmf import MassFunction


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

    z : float, optional
        Deprecated and ignored. The WDM transfer function and its characteristic
        (comoving) scales and masses are independent of redshift.

    \*\*model_parameters : unpack-dict
        Parameters specific to a model.
        To see the default values, check the :attr:`_defaults`
        class attribute.

    Attributes
    ----------
    rho_mean : float
        Comoving mean matter density, in :math:`h^2 M_\odot {\rm Mpc}^{-3}`.
    """

    def __init__(self, mx, cosmo=Planck15, z: float | None = None, **model_params):
        if z is not None:
            warnings.warn(
                "The 'z' argument to WDM models is deprecated and ignored: WDM transfer "
                "functions and their comoving mass scales do not depend on redshift.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.mx = mx
        self.cosmo = cosmo
        self.rho_mean = (
            (self.cosmo.Om0 * self.cosmo.critical_density0 / self.cosmo.h**2)
            .to(u.solMass / u.Mpc**3)
            .value
        )
        self.Oc0 = cosmo.Om0 - cosmo.Ob0

        super().__init__(**model_params)

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
            "You shouldn't call the WDM class, and any subclass should define the transfer method."
        )


class _FittedWDM(WDM, abstract=True):
    r"""
    Shared machinery for WDM models using the transfer function of Bode et al. (2001).

    Both :class:`Bode01` and :class:`Viel05` approximate the WDM/CDM transfer
    function by

    .. math:: T(k) = \left[1 + (\alpha k)^{2\nu}\right]^{-5/\nu}

    (Bode et al. 2001, eq. A8; Viel et al. 2005, eq. 6). They differ only in the
    exponent :math:`\nu` and in the fit for the break scale :math:`\alpha`
    (:attr:`lam_eff_fs`), which subclasses must define.

    The parameter used to be called ``mu``. It is still accepted, with a
    :class:`DeprecationWarning`, and mapped to ``nu``.
    """

    def __init__(self, mx, cosmo=Planck15, z: float | None = None, **model_params):
        if "mu" in model_params:
            if "nu" in model_params:
                raise ValueError(
                    f"{self.__class__.__name__} got both 'mu' and 'nu'. 'mu' is a deprecated "
                    "alias of 'nu'; pass only 'nu'."
                )
            warnings.warn(
                "The WDM parameter 'mu' has been renamed to 'nu', as in Bode et al. (2001) and "
                "Viel et al. (2005). 'mu' is deprecated and will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )
            model_params["nu"] = model_params.pop("mu")
        super().__init__(mx, cosmo=cosmo, z=z, **model_params)

    def transfer(self, k):
        r"""Compute the WDM/CDM transfer function, :math:`[1 + (\alpha k)^{2\nu}]^{-5/\nu}`."""
        nu = self.params["nu"]
        return (1 + (self.lam_eff_fs * k) ** (2 * nu)) ** (-5.0 / nu)

    @property
    def lam_eff_fs(self):
        r"""Effective free-streaming scale :math:`\alpha`, in comoving :math:`h^{-1}{\rm Mpc}`."""
        raise NotImplementedError

    @property
    def m_fs(self):
        r"""
        Free-streaming mass scale, in :math:`h^{-1}M_\odot`.

        :math:`M_{\rm fs} = (4\pi/3)\bar\rho(\lambda^{\rm eff}_{\rm fs}/2)^3`, from
        Schneider et al. (2012), eq. 7.
        """
        return (4.0 / 3.0) * np.pi * self.rho_mean * (self.lam_eff_fs / 2) ** 3

    @property
    def lam_hm(self):
        r"""
        Half-mode length scale, in comoving :math:`h^{-1}{\rm Mpc}`.

        The wavelength :math:`2\pi/k` at which :math:`T(k) = 1/2`:
        :math:`\lambda_{\rm hm} = 2\pi\alpha(2^{\nu/5} - 1)^{-1/(2\nu)}`, from
        Schneider et al. (2012), eq. 8.
        """
        nu = self.params["nu"]
        return 2 * np.pi * self.lam_eff_fs * (2 ** (nu / 5) - 1) ** (-0.5 / nu)

    @property
    def m_hm(self):
        r"""
        Half-mode mass scale, in :math:`h^{-1}M_\odot`.

        :math:`M_{\rm hm} = (4\pi/3)\bar\rho(\lambda_{\rm hm}/2)^3`, from
        Schneider et al. (2012), eq. 9. Since :math:`\lambda_{\rm hm}/2` is the
        comoving half-wavelength of the mode suppressed by a factor of two, this is
        also the mass :math:`(4\pi/3)\bar\rho R_s^3` of Bode et al. (2001), Sect. 7.
        """
        return (4.0 / 3.0) * np.pi * self.rho_mean * (self.lam_hm / 2) ** 3


class Viel05(_FittedWDM):
    r"""
    WDM transfer function of Viel et al. (2005).

    Uses the functional form of Bode et al. (2001) (Viel et al. 2005, eq. 6) with
    Viel et al.'s own Boltzmann-code fit: :math:`\nu = 1.12` and the break scale of
    their eq. 7 (see :attr:`lam_eff_fs`). This differs from :class:`Bode01`, which has
    :math:`\nu = 1.2` and a different fit for the break scale.

    Parameters
    ----------
    mx : float
        Mass of the particle in keV
    cosmo : `hmf.cosmo.Cosmology` instance
        A cosmology.
    z : float, optional
        Deprecated and ignored. See :class:`WDM`.
    \*\*model_parameters : unpack-dict
        Parameters specific to a model. Available parameters are as follows.
        To see the default values, check the :attr:`_defaults`
        class attribute.

        :nu: Exponent of the transfer function (Viel et al. 2005, eqs 6 and 7).
             Formerly called ``mu``, which is still accepted but deprecated.
        :g_x: Number of degrees of freedom of the WDM particle. Not part of Viel
              et al. (2005); see :attr:`lam_eff_fs`.
    """

    references: ClassVar[tuple[str, ...]] = (refs.VIEL05,)

    _defaults: ClassVar[dict[str, float]] = {"nu": 1.12, "g_x": 1.5}

    @property
    def lam_eff_fs(self):
        r"""
        Effective free-streaming scale :math:`\alpha`, in comoving :math:`h^{-1}{\rm Mpc}`.

        Viel et al. (2005), eq. 7 (thermal relics), also Schneider et al. (2012), eq. 5:

        .. math:: \alpha = 0.049 \left(\frac{m_x}{\rm keV}\right)^{-1.11}
                  \left(\frac{\Omega_x}{0.25}\right)^{0.11}
                  \left(\frac{h}{0.7}\right)^{1.22} h^{-1}{\rm Mpc}.

        The extra factor :math:`(1.5/g_x)^{0.29}` is not in Viel et al. (2005); hmf
        borrows it from Bode et al. (2001), eq. A9. It is unity for the default
        :math:`g_x = 1.5`.
        """
        return (
            0.049
            * self.mx**-1.11
            * (self.Oc0 / 0.25) ** 0.11
            * (self.cosmo.h / 0.7) ** 1.22
            * (1.5 / self.params["g_x"]) ** 0.29
        )


class Bode01(_FittedWDM):
    r"""
    WDM transfer function of Bode, Ostriker & Turok (2001).

    Their fit to a full Boltzmann-code calculation (Bode et al. 2001, eqs A8 and A9):

    .. math:: T(k) = \left[1 + (\alpha k)^{2\nu}\right]^{-5/\nu}, \qquad \nu = 1.2,

    with the break scale :math:`\alpha` given in :attr:`lam_eff_fs`. This differs
    from :class:`Viel05` (:math:`\nu = 1.12`, and a break scale that falls as
    :math:`m_x^{-1.11}` rather than :math:`m_x^{-1.15}`).

    Parameters
    ----------
    mx : float
        Mass of the particle in keV
    cosmo : `hmf.cosmo.Cosmology` instance
        A cosmology.
    z : float, optional
        Deprecated and ignored. See :class:`WDM`.
    \*\*model_parameters : unpack-dict
        Parameters specific to a model. Available parameters are as follows.
        To see the default values, check the :attr:`_defaults`
        class attribute.

        :nu: Exponent of the transfer function (Bode et al. 2001, eq. A8). Formerly
             called ``mu``, which is still accepted but deprecated.
        :g_x: Number of degrees of freedom of the WDM particle (Bode et al. 2001,
              eq. A9; 1.5 for a neutrino-like fermion).
    """

    references: ClassVar[tuple[str, ...]] = (refs.BODE01,)

    _defaults: ClassVar[dict[str, float]] = {"nu": 1.2, "g_x": 1.5}

    @property
    def lam_eff_fs(self):
        r"""
        Effective free-streaming scale :math:`\alpha`, in comoving :math:`h^{-1}{\rm Mpc}`.

        Bode et al. (2001), eq. A9:

        .. math:: \alpha = 0.048 \left(\frac{\Omega_X}{0.4}\right)^{0.15}
                  \left(\frac{h}{0.65}\right)^{1.3}
                  \left(\frac{\rm keV}{m_X}\right)^{1.15}
                  \left(\frac{1.5}{g_X}\right)^{0.29} h^{-1}{\rm Mpc},

        where :math:`\Omega_X` is the WDM density, taken here as
        :math:`\Omega_m - \Omega_b`.
        """
        return (
            0.048
            * (self.Oc0 / 0.4) ** 0.15
            * (self.cosmo.h / 0.65) ** 1.3
            * self.mx**-1.15
            * (1.5 / self.params["g_x"]) ** 0.29
        )


viel_model = Viel05(mx=1.0)


@pluggable
class WDMRecalibrateMF(Component):
    r"""
    Base class for Components that emulate the effect of WDM on the HMF empirically.

    Required method is :meth:`dndm_alter`.

    The recalibrations below depend on the WDM model only through its half-mode
    mass, ``wdm.m_hm``, so they work with any :class:`WDM` model that defines it.
    Their parameters were, however, fitted with a particular transfer function
    (noted in each class), and the half-mode mass of another model differs: for
    Planck15, :class:`Bode01` gives an :math:`M_{\rm hm}` 15%, 21% and 40% smaller
    than :class:`Viel05` at :math:`m_x` = 0.5, 1 and 10 keV.

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
        super().__init__(**model_parameters)

    def dndm_alter(self):
        """Alter the CDM dn/dm to impose WDM modeling."""


class Schneider12_vCDM(WDMRecalibrateMF):
    r"""
    Schneider+2012 recalibration of the CDM HMF.

    Schneider et al. (2012) defined :math:`M_{\rm hm}` with the :class:`Viel05`
    transfer function (their eqs 4, 5 and 9), which is the default ``wdm`` here.

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

    references: ClassVar[tuple[str, ...]] = (refs.SCHNEIDER12,)

    _defaults: ClassVar[dict[str, float]] = {"beta": 1.16}

    @override
    def dndm_alter(self):
        return self.dndm0 * (1 + self.wdm.m_hm / self.m) ** (-self.params["beta"])


class Schneider12(WDMRecalibrateMF):
    r"""
    Schneider+2012 recalibration of the WDM HMF.

    Schneider et al. (2012) defined :math:`M_{\rm hm}` with the :class:`Viel05`
    transfer function (their eqs 4, 5 and 9), which is the default ``wdm`` here.

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

    references: ClassVar[tuple[str, ...]] = (refs.SCHNEIDER12,)

    _defaults: ClassVar[dict[str, float]] = {"alpha": 0.6}

    @override
    def dndm_alter(self):
        return self.dndm0 * (1 + self.wdm.m_hm / self.m) ** (-self.params["alpha"])


class Lovell14(WDMRecalibrateMF):
    r"""
    Lovell+2014 recalibration of the WDM HMF.

    Lovell et al. (2014) used the Bode et al. (2001) functional form with
    :math:`\nu = 1` and the break scale of Bode et al.'s eq. A7 (their eqs 2 and 3),
    so their :math:`M_{\rm hm}` matches neither :class:`Viel05` nor :class:`Bode01`
    exactly.

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

    references: ClassVar[tuple[str, ...]] = (refs.LOVELL14,)

    _defaults: ClassVar[dict[str, float]] = {"beta": 0.99, "gamma": 2.7}

    @override
    def dndm_alter(self):
        return self.dndm0 * (1 + self.params["gamma"] * self.wdm.m_hm / self.m) ** (
            -self.params["beta"]
        )


# ===============================================================================
# Frameworks
# ===============================================================================
class TransferWDM(Transfer):
    """
    A subclass of :class:`hmf.transfer.Transfer` that mixes in WDM capabilities.

    This replaces the standard CDM quantities with WDM-derived ones, where relevant.

    In addition to the parameters directly passed to this class, others are available
    which are passed on to its superclass. To read a standard documented list of (all)
    parameters, use :meth:`parameter_info`. If you want to just see the plain
    list of available parameters, use :meth`get_all_parameters`. To see the
    actual defaults for each parameter, use :meth:`get_all_parameter_defaults`.
    """

    def __init__(self, wdm_mass=3.0, wdm_model=Viel05, wdm_params=None, **transfer_kwargs):
        wdm_params = wdm_params or {}

        # Call standard transfer
        super().__init__(**transfer_kwargs)

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
        except ValueError as e:
            raise ValueError("wdm_mass must be a number (", val, ")") from e

        if val <= 0:
            raise ValueError("wdm_mass must be > 0 (", val, ")")
        return val

    @cached_quantity
    def wdm(self):
        """
        The instantiated WDM model.

        Contains quantities relevant to WDM.
        """
        return self.wdm_model(mx=self.wdm_mass, cosmo=self.cosmo, **self.wdm_params)

    @override
    @cached_quantity
    def _unnormalised_lnT(self):
        return super()._unnormalised_lnT + np.log(self.wdm.transfer(self.k))


class MassFunctionWDM(MassFunction, TransferWDM):
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
        super().__init__(**kwargs)

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

    @override
    @cached_quantity
    def dndm(self):
        r"""
        The number density of haloes in WDM, ``len=len(m)``.

        Units of :math:`h^4 M_\odot^{-1} Mpc^{-3}`
        """
        dndm = super().dndm

        if self.alter_model is not None:
            alter = self.alter_model(m=self.m, dndm0=dndm, wdm=self.wdm, **self.alter_params)
            dndm = alter.dndm_alter()

        return dndm
