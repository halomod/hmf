'''
Various models for computing the transfer function.

Note that these are not transfer function "frameworks". The framework is found
in :mod:`hmf.transfer`.
'''
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from _framework import Component
try:
    import pycamb
except ImportError:
    pass


_allfits = ["CAMB", "FromFile", "EH_BAO", "EH_NoBAO", "BBKS", "BondEfs"]


class TransferComponent(Component):
    """
    Base class for transfer models.

    The only necessary function to specify is ``lnt``, which returns the log
    transfer given ``lnk``.

    Parameters
    ----------
    cosmo : :class:`astropy.cosmology.FLRW` instance
        The cosmology used in the calculation

    \*\*model_parameters :
        Any model-specific parameters.
    """
    _defaults = {}
    def __init__(self, cosmo, **model_parameters):
        self.cosmo = cosmo
        super(TransferComponent, self).__init__(**model_parameters)

    def lnt(self, lnk):
        """
        Natural log of the transfer function

        Parameters
        ----------
        lnk : array_like
            Wavenumbers [Mpc/h]

        Returns
        -------
        lnt : array_like
            The log of the transfer function at lnk.
        """
        pass

class CAMB(TransferComponent):
    """
    Transfer function computed by CAMB.

    Parameters
    ----------
    cosmo : :class:`astropy.cosmology.FLRW` instance
        The cosmology used in the calculation

    \*\*model_parameters : unpack-dict
        Parameters specific to this model. In this case, available
        parameters are the following. To see their default values,
        check the :attr:`_defaults` class attribute.

        :Scalar_initial_condition: Initial scalar perturbation mode (adiabatic=1, CDM iso=2,
                                   Baryon iso=3,neutrino density iso =4, neutrino velocity iso = 5)
        :lAccuracyBoost: Larger to keep more terms in the hierarchy evolution
        :AccuracyBoost: Increase accuracy_boost to decrease time steps, use more k
                        values,  etc.Decrease to speed up at cost of worse accuracy.
                        Suggest 0.8 to 3.
        :w_perturb: Whether to perturb the dark energy equation of state.
        :transfer__k_per_logint: Number of wavenumbers estimated per log interval by CAMB
                                 Default of 11 gets best performance for requisite accuracy of mass function.
        :transfer__kmax: Maximum value of the wavenumber.
                         Default of 0.25 is high enough for requisite accuracy of mass function.
        :ThreadNum: Number of threads to use for calculation of transfer
                    function by CAMB. Default 0 automatically determines the number.
        :scalar_amp: Amplitude of the power spectrum. It is not recommended to modify
                     this parameter, as the amplitude is typically set by sigma_8.
    """
    _defaults = {"Scalar_initial_condition":1,
                 "lAccuracyBoost":1,
                 "AccuracyBoost":1,
                 "w_perturb":False,
                 "transfer__k_per_logint":11,
                 "transfer__kmax":5,
                 "ThreadNum":0,
                 "scalar_amp":1e-9}

    def _check_low_k(self, lnk, lnT, lnkmin):
        """
        Check convergence of transfer function at low k.

        Unfortunately, some versions of CAMB produce a transfer which has a
        turn-up at low k, which we cut out here.

        Parameters
        ----------
        lnk : array_like
            Value of log(k)

        lnT : array_like
            Value of log(transfer)
        """

        start = 0
        for i in range(len(lnk) - 1):
            if abs((lnT[i + 1] - lnT[i]) / (lnk[i + 1] - lnk[i])) < 0.0001:
                start = i
                break
        lnT = lnT[start:-1]
        lnk = lnk[start:-1]

        lnk[0] = lnkmin
        return lnk, lnT

    def lnt(self, lnk):
        """
        Natural log of the transfer function

        Parameters
        ----------
        lnk : array_like
            Wavenumbers [Mpc/h]

        Returns
        -------
        lnt : array_like
            The log of the transfer function at lnk.
        """

        pycamb_dict = {"w_lam":self.cosmo.w(0),
                       "TCMB":self.cosmo.Tcmb0.value,
                       "Num_Nu_massless":self.cosmo.Neff,
                       "omegab":self.cosmo.Ob0,
                       "omegac":self.cosmo.Om0 - self.cosmo.Ob0,
                       "H0":self.cosmo.H0.value,
                       "omegav":self.cosmo.Ode0,
                       "omegak":self.cosmo.Ok0,
                       "omegan":self.cosmo.Onu0,
#                        "scalar_index":self.n,
                       }

        cdict = dict(pycamb_dict,
                     **self.params)
        T = pycamb.transfers(**cdict)[1]
        T = np.log(T[[0, 6], :, 0])

        if lnk[0] < T[0, 0]:
            lnkout, lnT = self._check_low_k(T[0, :], T[1, :], lnk[0])
        else:
            lnkout = T[0, :]
            lnT = T[1, :]
        return spline(lnkout, lnT, k=1)(lnk)

class FromFile(CAMB):
    """
    Import a transfer function from file.

    .. note:: The file should be in the same format as output from CAMB,
              or else in two-column ASCII format (k,T).

    Parameters
    ----------
    cosmo : :class:`astropy.cosmology.FLRW` instance
        The cosmology used in the calculation

    \*\*model_parameters : unpack-dict
        Parameters specific to this model. In this case, available
        parameters are the following. To see their default values,
        check the :attr:`_defaults` class attribute.

        :fname: str
            Location of the file to import.
    """
    _defaults = {"fname":""}

    def lnt(self, lnk):
        """
        Natural log of the transfer function

        Parameters
        ----------
        lnk : array_like
            Wavenumbers [Mpc/h]

        Returns
        -------
        lnt : array_like
            The log of the transfer function at lnk.
        """
        try:
            T = np.log(np.genfromtxt(self.params["fname"])[:, [0, 6]].T)
        except IndexError:
            T = np.log(np.genfromtxt(self.params["fname"])[:, [0, 1]].T)

        if lnk[0] < T[0, 0]:
            lnkout, lnT = self._check_low_k(T[0, :], T[1, :], lnk[0])
        else:
            lnkout = T[0, :]
            lnT = T[1, :]
        return spline(lnkout, lnT, k=1)(lnk)


class EH_BAO(TransferComponent):
    """
    Eisenstein & Hu (1998) fitting function with BAO wiggles

    From EH1998, Eqs. 26,28-31. Code adapted from CHOMP.

    Parameters
    ----------
    cosmo : :class:`astropy.cosmology.FLRW` instance
        The cosmology used in the calculation

    \*\*model_parameters : unpack-dict
        Parameters specific to this model. In this case, there
        are no model parameters.
    """

    def lnt(self, lnk):
        """
        Natural log of the transfer function

        Parameters
        ----------
        lnk : array_like
            Wavenumbers [Mpc/h]

        Returns
        -------
        lnt : array_like
            The log of the transfer function at lnk.
        """

        k = np.exp(lnk)
        theta = self.cosmo.Tcmb0.value / 2.7
        Oc0 = self.cosmo.Om0 - self.cosmo.Ob0
        O = self.cosmo.Om0
        Obh2 = self.cosmo.Ob0 * self.cosmo.h ** 2
        Oh2 = self.cosmo.Om0 * self.cosmo.h ** 2
        ObO = self.cosmo.Ob0 / self.cosmo.Om0

        zeq = 2.5e4 * Oh2 * theta ** (-4)
        keq = 7.46e-2 * Oh2 * theta ** (-2)
        b1 = 0.313 * Oh2 ** (-0.419) * (1. + 0.607 * Oh2 ** 0.674)
        b2 = 0.238 * Oh2 ** 0.223
        zd = 1291.*(Oh2 ** 0.251 / (1. + 0.659 * Oh2 ** 0.828)) * (1. + b1 * Obh2 ** b2)

        R = lambda z: 31.5 * Obh2 * theta ** (-4) * (1000. / z)
        Req = R(zeq)
        Rd = R(zd)

        s = (2. / (3.*keq)) * np.sqrt(6. / Req) * np.log(
            (np.sqrt(1. + Rd) + np.sqrt(Rd + Req)) / (1. + np.sqrt(Req)))
        ks = k * self.cosmo.h * s

        kSilk = 1.6 * Obh2 ** 0.52 * Oh2 ** 0.73 * (1. + (10.4 * Oh2) ** (-0.95))
        q = lambda k: k * self.cosmo.h / (13.41 * keq)

        G = lambda y: y * (-6.*np.sqrt(1. + y) + (2 + 3 * y) * np.log(
                           (np.sqrt(1. + y) + 1.) / (np.sqrt(1. + y) - 1.)))
        alpha_b = 2.07 * keq * s * (1. + Rd) ** (-3. / 4.) * G((1. + zeq) / (1. + zd))
        beta_b = 0.5 + (ObO) + (3. - 2.*ObO) * np.sqrt((17.2 * Oh2) ** 2 + 1.)

        C = lambda x, a: (14.2 / a) + 386. / (1. + 69.9 * q(x) ** 1.08)
        T0t = lambda x, a, b: np.log(np.e + 1.8 * b * q(x)) / (
            np.log(np.e + 1.8 * b * q(k)) + C(x, a) * q(x) ** 2)

        a1 = (46.9 * Oh2) ** 0.670 * (1. + (32.1 * Oh2) ** (-0.532))
        a2 = (12.*Oh2) ** 0.424 * (1. + (45.*Oh2) ** (-0.582))
        alpha_c = a1 ** (-ObO) * a2 ** (-ObO ** 3)
        b1 = 0.944 * (1. + (458.*Oh2) ** (-0.708)) ** (-1)
        b2 = (0.395 * Oh2) ** (-0.0266)
        beta_c = 1. / (1. + b1 * ((Oc0 / O) ** b2 - 1))

        f = 1. / (1. + (ks / 5.4) ** 4)
        Tc = f * T0t(k, 1, beta_c) + (1. - f) * T0t(k, alpha_c, beta_c)

        beta_node = 8.41 * (Oh2 ** 0.435)
        stilde = s / (1. + (beta_node / (ks)) ** 3) ** (1. / 3.)

        Tb1 = T0t(k, 1., 1.) / (1. + (ks / 5.2) ** 2)
        Tb2 = (alpha_b / (1. + (beta_b / ks) ** 3)) * np.exp(-(k * self.cosmo.h / kSilk) ** 1.4)
        Tb = np.sinc(k * stilde / np.pi) * (Tb1 + Tb2)
        return np.log(ObO * Tb + (Oc0 / O) * Tc)

class EH_NoBAO(TransferComponent):
    """
    Eisenstein & Hu (1998) fitting function without BAO wiggles

    From EH 1998 Eqs. 26,28-31. Code adapted from CHOMP project.

    Parameters
    ----------
    cosmo : :class:`astropy.cosmology.FLRW` instance
        The cosmology used in the calculation

    \*\*model_parameters : unpack-dict
        Parameters specific to this model. In this case, there are
        no model parameters.
    """
    def lnt(self, lnk):
        """
        Natural log of the transfer function

        Parameters
        ----------
        lnk : array_like
            Wavenumbers [Mpc/h]

        Returns
        -------
        lnt : array_like
            The log of the transfer function at lnk.
        """
        k = np.exp(lnk)
        theta = self.cosmo.Tcmb0.value / 2.7  # Temperature of CMB_2.7
        Omh2 = self.cosmo.Om0 * self.cosmo.h ** 2
        Omb2 = self.cosmo.Ob0 * self.cosmo.h ** 2
        omega_ratio = self.cosmo.Ob0 / self.cosmo.Om0
        s = 44.5 * np.log(9.83 / Omh2) / np.sqrt(1 + 10.0 * (Omb2) ** (3. / 4.))
        alpha = (1 - 0.328 * np.log(431.0 * Omh2) * omega_ratio +
                 0.38 * np.log(22.3 * Omh2) * omega_ratio ** 2)
        Gamma_eff = self.cosmo.Om0 * self.cosmo.h * (
            alpha + (1 - alpha) / (1 + (0.43 * k * s) ** 4))
        q = k * theta / Gamma_eff
        L0 = np.log(2 * np.e + 1.8 * q)
        C0 = 14.2 + 731.0 / (1 + 62.5 * q)
        return np.log(L0 / (L0 + C0 * q * q))

class BBKS(TransferComponent):
    r"""
    BBKS (1986) transfer function.

    Parameters
    ----------
    cosmo : :class:`astropy.cosmology.FLRW` instance
        The cosmology used in the calculation

    \*\*model_parameters : unpack-dict
        Parameters specific to this model. In this case, available
        parameters are the following: **a, b, c, d, e**.
        To see their default values, check the :attr:`_defaults`
        class attribute.

    Notes
    -----
    The fit is given as

    .. math:: T(k) = \frac{\ln(1+aq)}{aq}\left(1 + bq + (cq)^2 + (dq)^3 + (eq)^4\right)^{-1/4},

    where

    .. math:: q = \frac{k}{\Gamma} \exp\left(\Omega_{b,0} + \frac{\sqrt{2h}\Omega_{b,0}}{\Omega_{m,0}}\right)

    and :math:`\Gamma = \Omega_{m,0} h`.
    """
    _defaults = {"a":2.34,"b":3.89,"c":16.1,"d":5.47,"e":6.71}
    def lnt(self, lnk):
        """
        Natural log of the transfer function

        Parameters
        ----------
        lnk : array_like
            Wavenumbers [Mpc/h]

        Returns
        -------
        lnt : array_like
            The log of the transfer function at lnk.
        """
        a = self.params['a']
        b = self.params['b']
        c = self.params['c']
        d = self.params['d']
        e = self.params['e']

        Gamma = self.cosmo.Om0 * self.cosmo.h
        q = np.exp(lnk) / Gamma * np.exp(self.cosmo.Ob0 + np.sqrt(2 * self.cosmo.h) *
                               self.cosmo.Ob0 / self.cosmo.Om0)
        return np.log((np.log(1.0 + a * q) / (a * q) *
                (1 + b * q + (c * q) ** 2 + (d * q) ** 3 +
                 (e * q) ** 4) ** (-0.25)))

class BondEfs(TransferComponent):
    r"""
    Transfer function of Bond and Efstathiou

    Parameters
    ----------
    cosmo : :class:`astropy.cosmology.FLRW` instance
        The cosmology used in the calculation

    \*\*model_parameters : unpack-dict
        Parameters specific to this model. In this case, available
        parameters are the following: **a, b, c, nu**.
        To see their default values, check the :attr:`_defaults`
        class attribute.

    Notes
    -----
    The fit is given as

    .. math:: T(k) = \left[1 + (\tilde{a}k + (\tilde{b}k)^{3/2} + (\tilde{c}k)^2)^\nu\right]^{-1/\nu}

    where :math:`\tilde{x} = x\alpha` and

    .. math:: \alpha = \frac{0.3\times 0.75^2}{\Omega_{m,0} h^2}.
    """
    _defaults = {"a":37.1,"b":21.1,"c":10.8,"nu":1.12}

    def lnt(self, lnk):
        """
        Natural log of the transfer function

        Parameters
        ----------
        lnk : array_like
            Wavenumbers [Mpc/h]

        Returns
        -------
        lnt : array_like
            The log of the transfer function at lnk.
        """

        scale = (0.3*0.75**2) / (self.cosmo.Om0 * self.cosmo.h ** 2)

        a = self.params['a'] * scale
        b = self.params['b'] * scale
        c = self.params['c'] * scale
        nu = self.params['nu']
        k = np.exp(lnk)
        return np.log((1 + (a * k + (b * k) ** 1.5 + (c * k) ** 2) ** nu) ** (-1 / nu))

class EH(EH_BAO):
    "Alias of :class:`EH_BAO`"
    pass
