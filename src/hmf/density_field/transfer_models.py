"""
Various models for computing the transfer function.

Note that these are not transfer function "frameworks". The framework is found
in :mod:`hmf.transfer`.
"""
import numpy as np
import pickle
import warnings
from astropy import cosmology
from copy import deepcopy
from scipy.interpolate import InterpolatedUnivariateSpline as spline

from .._internals._framework import Component, pluggable

try:
    import camb

    HAVE_CAMB = True
except ImportError:  # pragma: no cover
    HAVE_CAMB = False

_allfits = ["CAMB", "FromFile", "EH_BAO", "EH_NoBAO", "BBKS", "BondEfs"]


@pluggable
class TransferComponent(Component):
    r"""
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
        r"""
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


class FromFile(TransferComponent):
    r"""
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

    _defaults = {"fname": ""}

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
        r"""
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


if HAVE_CAMB:

    class CAMB(FromFile):
        r"""
        Transfer function computed by CAMB.

        Parameters
        ----------
        cosmo : :class:`astropy.cosmology.FLRW` instance
            The cosmology used in the calculation
        \*\*model_parameters : unpack-dict
            Parameters specific to this model.

            **camb_params:** An instantiated ``CAMBparams`` object, pre-set with desired
                             accuracy options etc.
            **dark_energy_params:** A dictionary of values passed to CAMB's `
                                    `set_dark_energy`` method. Values include
                                    `sound_speed` and `dark_energy_model`.
            **extrapolate_with_eh:** Whether to extrapolate past the intrinsic CAMB
                                     kmax by using an EH model. Can cause some problems
                                     if kmax is high, since CAMB diverges from the EH
                                     approximation.
        """

        _defaults = {
            "camb_params": None,
            "dark_energy_params": {},
            "extrapolate_with_eh": False,
            "kmax": None,
        }

        def __init__(self, *args, **kwargs):
            super(CAMB, self).__init__(*args, **kwargs)

            if not isinstance(
                self.cosmo, (cosmology.LambdaCDM, cosmology.wCDM, cosmology.w0waCDM)
            ):
                raise ValueError("CAMB will only work with LCDM or wCDM cosmologies")

            # Save the CAMB object properly for use
            # Set the cosmology
            if self.params["camb_params"] is None:
                self.params["camb_params"] = camb.CAMBparams(
                    DoLensing=False,
                    Want_CMB=False,
                    Want_CMB_lensing=False,
                    WantCls=False,
                    WantDerivedParameters=False,
                )

                self.params["camb_params"].Transfer.high_precision = False
                self.params["camb_params"].Transfer.k_per_logint = 0

                # If extrapolating with EH, use a lower value of kmax so that the
                # calculation is faster.
                if self.params["kmax"]:
                    self.params["camb_params"].Transfer.kmax = self.params["kmax"]

            if self.cosmo.Ob0 is None:
                raise ValueError(
                    "To use CAMB, you must set the baryon density in the cosmology "
                    "explicitly."
                )

            if self.cosmo.Tcmb0.value == 0:
                raise ValueError(
                    "If using CAMB, the CMB temperature must be set explicitly in the "
                    "cosmology."
                )

            self.params["camb_params"].set_cosmology(
                H0=self.cosmo.H0.value,
                ombh2=self.cosmo.Ob0 * self.cosmo.h ** 2,
                omch2=(self.cosmo.Om0 - self.cosmo.Ob0 - self.cosmo.Onu0)
                * self.cosmo.h ** 2,
                mnu=sum(self.cosmo.m_nu.value),
                neutrino_hierarchy="degenerate",
                omk=self.cosmo.Ok0,
                nnu=self.cosmo.Neff,
                standard_neutrino_neff=self.cosmo.Neff,
                TCMB=self.cosmo.Tcmb0.value,
            )
            self.params["camb_params"].WantTransfer = True

            # Set the DE equation of state. We only support constant w.
            if isinstance(self.cosmo, cosmology.wCDM):
                self.params["camb_params"].set_dark_energy(w=self.cosmo.w0)
            elif isinstance(self.cosmo, cosmology.w0waCDM):
                self.params["camb_params"].set_dark_energy(
                    w=self.cosmo.w0, wa=self.cosmo.wa
                )

            if self.params["extrapolate_with_eh"]:
                # Create an EH transfer to extrapolate to at high k.
                self._eh = EH(self.cosmo)

        def lnt(self, lnk):
            r"""
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

            camb_transfers = camb.get_transfer_functions(self.params["camb_params"])
            T = camb_transfers.get_matter_transfer_data().transfer_data
            T = np.log(T[[0, 6], :, 0])

            if lnk[0] < T[0, 0]:
                lnkout, lnT = self._check_low_k(T[0, :], T[1, :], lnk[0])
            else:
                lnkout = T[0, :]
                lnT = T[1, :]

            lnT -= lnT[0]

            if not self.params["extrapolate_with_eh"]:
                return spline(lnkout, lnT, k=1)(lnk)

            # Now add a point one e-fold above the max, with an EH-generated transfer
            lnkout = np.concatenate((lnkout, [lnkout[-1] + 1]))
            # normalise EH at the final CAMB point.
            norm = self._eh.lnt(lnkout[-2]) - lnT[-1]
            lnT = np.concatenate((lnT, [self._eh.lnt(lnkout[-1]) - norm]))

            lnkmin = lnkout.min()
            lnkmax = lnkout.max()

            inner_spline = spline(lnkout, lnT, k=3)

            out = np.zeros_like(lnk)
            out[lnk < lnkmin] = 0
            out[(lnkmin <= lnk) & (lnk <= lnkmax)] = inner_spline(
                lnk[(lnkmin <= lnk) & (lnk <= lnkmax)]
            )
            out[lnk >= lnkmax] = self._eh.lnt(lnk[lnk >= lnkmax]) - norm

            return out

        def __getstate__(self):
            # We need to get rid of the CAMBparams() object, as it cannot be pickled.
            p = self.params["camb_params"]

            potential_keys = [  # From https://camb.readthedocs.io/en/latest/model.html
                "WantCls",
                "WantTransfer",
                "WantScalars",
                "WantTensors",
                "WantVectors",
                "WantDerivedParameters",
                "Want_cl_2D_array",
                "Want_CMB",
                "Want_CMB_lensing",
                "DoLensing",
                "NonLinear",
                "Transfer",
                "want_zstar",
                "want_zdrag",
                "min_l",
                "max_l",
                "max_l_tensor",
                "max_eta_k",
                "max_eta_k_tensor",
                "ombh2",
                "omch2",
                "omk",
                "omnuh2",
                "H0",
                "TCMB",
                "YHe",
                "num_nu_massless",
                "num_nu_massive",
                "nu_mass_eigenstates",
                "share_delta_neff",
                "InitPower",
                "Recomb",
                "Reion",
                "DarkEnergy",
                "NonLinearModel",
                "Accuracy",
                "SourceTerms",
                "z_outputs",
                "scalar_initial_condition",
                "InitialConditionVector",
                "OutputNormalization",
                "Alens",
                "MassiveNuMethod",
                "DoLateRadTruncation",
                "Evolve_baryon_cs",
                "Evolve_delta_xe",
                "Evolve_delta_Ts",
                "Do21cm",
                "transfer_21cm_cl",
                "Log_lvalues",
                "use_cl_spline_template",
                "SourceWindows",
            ]

            # Unsaveable parameters:
            # "nu_mass_degeneracies", "nu_mass_fractions", "nu_mass_numbers", "CustomSources"

            dct = {}
            for pk in potential_keys:
                try:
                    pickle.dumps(getattr(p, pk))

                    dct[pk] = getattr(p, pk)
                except AttributeError:
                    warnings.warn(
                        f"CAMB key '{pk}' is not an attribute. If you provided a "
                        f"custom CAMBparams, results may be inconsistent. Available: "
                        f"{dir(p)}"
                    )

                except Exception:
                    warnings.warn(f"CAMB key {pk} is not pickle-able.")

            # Deepcopy self
            this = {}
            for key, val in self.__dict__.items():
                if key != "params":
                    this[key] = deepcopy(val)

            this["params"] = {"camb_params": dct}

            return this

        def __setstate__(self, state):
            self.__dict__ = state

            self.params["camb_params"] = camb.CAMBparams(**self.params["camb_params"])


class FromArray(FromFile):
    r"""
    Use a spline over a given array to define the transfer function

    Parameters
    ----------
    cosmo : :class:`astropy.cosmology.FLRW` instance
        The cosmology used in the calculation
    \*\*model_parameters : unpack-dict
        Parameters specific to this model. In this case, available
        parameters are the following. To see their default values,
        check the :attr:`_defaults` class attribute.

        :k: array
            Wavenumbers, in [h/Mpc]

        :T: array
            Transfer function
    """

    _defaults = {"k": None, "T": None}

    def lnt(self, lnk):
        r"""
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
        k = self.params["k"]
        T = self.params["T"]

        if k is None or T is None:
            raise ValueError(
                "You must supply an array for both k and T for this Transfer Model"
            )
        if len(k) != len(T):
            raise ValueError("k and T must have same length")

        if lnk[0] < np.log(k.min()):
            lnkout, lnT = self._check_low_k(np.log(k), np.log(T), lnk[0])
        else:
            lnkout = np.log(k)
            lnT = np.log(T)
        return spline(lnkout, lnT, k=1)(lnk)


class EH_BAO(TransferComponent):
    r"""
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

    def __init__(self, *args, **kwargs):
        super(EH_BAO, self).__init__(*args, **kwargs)
        self._set_params()

    def _set_params(self):
        """
        Port of ``TFset_parameters`` from original EH code.
        """
        self.Obh2 = self.cosmo.Ob0 * self.cosmo.h ** 2
        self.Omh2 = self.cosmo.Om0 * self.cosmo.h ** 2
        self.f_baryon = self.cosmo.Ob0 / self.cosmo.Om0

        self.theta_cmb = self.cosmo.Tcmb0.value / 2.7

        self.z_eq = 2.5e4 * self.Omh2 * self.theta_cmb ** (-4)  # really 1+z
        self.k_eq = 7.46e-2 * self.Omh2 * self.theta_cmb ** (-2)  # units Mpc^-1 (no h!)

        self.z_drag_b1 = (
            0.313 * self.Omh2 ** -0.419 * (1.0 + 0.607 * self.Omh2 ** 0.674)
        )
        self.z_drag_b2 = 0.238 * self.Omh2 ** 0.223
        self.z_drag = (
            1291.0
            * self.Omh2 ** 0.251
            / (1.0 + 0.659 * self.Omh2 ** 0.828)
            * (1.0 + self.z_drag_b1 * self.Obh2 ** self.z_drag_b2)
        )

        self.r_drag = (
            31.5 * self.Obh2 * self.theta_cmb ** -4 * (1000.0 / (1 + self.z_drag))
        )
        self.r_eq = 31.5 * self.Obh2 * self.theta_cmb ** -4 * (1000.0 / self.z_eq)

        self.sound_horizon = (
            (2.0 / (3.0 * self.k_eq))
            * np.sqrt(6.0 / self.r_eq)
            * np.log(
                (np.sqrt(1.0 + self.r_drag) + np.sqrt(self.r_drag + self.r_eq))
                / (1.0 + np.sqrt(self.r_eq))
            )
        )

        self.k_silk = (
            1.6
            * self.Obh2 ** 0.52
            * self.Omh2 ** 0.73
            * (1.0 + (10.4 * self.Omh2) ** (-0.95))
        )

        alpha_c_a1 = (46.9 * self.Omh2) ** 0.670 * (
            1.0 + (32.1 * self.Omh2) ** (-0.532)
        )
        alpha_c_a2 = (12.0 * self.Omh2) ** 0.424 * (
            1.0 + (45.0 * self.Omh2) ** (-0.582)
        )
        self.alpha_c = alpha_c_a1 ** (-self.f_baryon) * alpha_c_a2 ** (
            -self.f_baryon ** 3
        )

        beta_c_b1 = 0.944 / (1.0 + (458.0 * self.Omh2) ** -0.708)
        beta_c_b2 = (0.395 * self.Omh2) ** -0.0266
        self.beta_c = 1.0 / (1.0 + beta_c_b1 * ((1 - self.f_baryon) ** beta_c_b2 - 1))

        y = self.z_eq / (1 + self.z_drag)
        alpha_b_G = y * (
            -6 * np.sqrt(1 + y)
            + (2 + 3 * y) * np.log((np.sqrt(1 + y) + 1) / (np.sqrt(1 + y) - 1))
        )
        self.alpha_b = (
            2.07
            * self.k_eq
            * self.sound_horizon
            * (1.0 + self.r_drag) ** -0.75
            * alpha_b_G
        )

        self.beta_node = 8.41 * self.Omh2 ** 0.435
        self.beta_b = (
            0.5
            + self.f_baryon
            + (3.0 - 2.0 * self.f_baryon) * np.sqrt((17.2 * self.Omh2) ** 2 + 1.0)
        )

    @property
    def k_peak(self):
        return 2.5 * np.pi * (1 + 0.217 * self.Omh2) / self.sound_horizon

    @property
    def sound_horizon_fit(self):
        """
        Sound horizon in Mpc/h
        """
        return (
            self.cosmo.h
            * 44.5
            * np.log(9.83 / self.Omh2)
            / np.sqrt(1 + 10 * (self.Obh2 ** 0.75))
        )

    def lnt(self, lnk):
        r"""
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
        # Get k in Mpc^-1
        k = np.exp(lnk) * self.cosmo.h

        q = k / (13.41 * self.k_eq)
        ks = k * self.sound_horizon

        T_c_ln_beta = np.log(np.e + 1.8 * self.beta_c * q)
        T_c_ln_nobeta = np.log(np.e + 1.8 * q)
        T_c_C_alpha = (14.2 / self.alpha_c) + 386.0 / (1.0 + 69.9 * q ** 1.08)
        T_c_C_noalpha = 14.2 + 386.0 / (1.0 + 69.9 * q ** 1.08)

        T_c_f = 1.0 / (1.0 + (ks / 5.4) ** 4)

        def term(a, b):
            return a / (a + b * q ** 2)

        T_c = T_c_f * term(T_c_ln_beta, T_c_C_noalpha) + (1 - T_c_f) * term(
            T_c_ln_beta, T_c_C_alpha
        )

        s_tilde = self.sound_horizon / (1.0 + (self.beta_node / ks) ** 3) ** (1.0 / 3.0)
        ks_tilde = k * s_tilde

        T_b_T0 = term(T_c_ln_nobeta, T_c_C_noalpha)
        Tb1 = T_b_T0 / (1.0 + (ks / 5.2) ** 2)
        Tb2 = (self.alpha_b / (1.0 + (self.beta_b / ks) ** 3)) * np.exp(
            -((k / self.k_silk) ** 1.4)
        )
        T_b = np.sin(ks_tilde) / ks_tilde * (Tb1 + Tb2)

        return np.log(self.f_baryon * T_b + (1 - self.f_baryon) * T_c)


class EH_NoBAO(EH_BAO):
    r"""
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

    @property
    def alpha_gamma(self):
        return (
            1
            - 0.328 * np.log(431 * self.Omh2) * self.f_baryon
            + 0.38 * np.log(22.3 * self.Omh2) * self.f_baryon ** 2
        )

    def lnt(self, lnk):
        r"""
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
        k = np.exp(lnk) * self.cosmo.h

        ks = k * self.sound_horizon_fit / self.cosmo.h  # need sound horizon in Mpc here

        gamma_eff = self.Omh2 * (
            self.alpha_gamma + (1 - self.alpha_gamma) / (1 + (0.43 * ks) ** 4)
        )
        q = k / (13.4 * self.k_eq)

        q_eff = q * self.Omh2 / gamma_eff

        L0 = np.log(2 * np.e + 1.8 * q_eff)
        C0 = 14.2 + 731.0 / (1 + 62.5 * q_eff)
        return np.log(L0 / (L0 + C0 * q_eff * q_eff))


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

    .. math:: T(k) = \frac{\ln(1+aq)}{aq}
              \left(1 + bq + (cq)^2 + (dq)^3 + (eq)^4\right)^{-1/4},

    where

    .. math:: q = \frac{k}{\Gamma}

    and :math:`\Gamma = \Omega_{m,0} h`. Note that here *k* is in units of h/Mpc, which
    accounts for the extra *h* in the equations in BBKS.

    These equations are taken from BBKS 1986, Eq. G3.

    Further modifications can be made in the presence of baryons. Sugiyama 1995, Eq. 3.9
    gives

    .. math:: \Gamma \rightarrow \Gamma
              \exp\left(-\Omega_{b,0}(1 + 1/\Omega_{m,0})\right)

    and Liddle and Lythe (2000) Eq. 5.14 give a slight extra:

    .. math:: \Gamma \rightarrow \Gamma
              \exp\left(-\Omega_{b,0}(1 + \sqrt{2h}/\Omega_{m,0})\right).
    """
    _defaults = {
        "a": 2.34,
        "b": 3.89,
        "c": 16.1,
        "d": 5.46,
        "e": 6.71,
        "use_sugiyama_baryons": False,
        "use_liddle_baryons": True,
    }

    def lnt(self, lnk):
        """
        Natural log of the transfer function

        Parameters
        ----------
        lnk : array_like
            Wavenumbers [h/Mpc]

        Returns
        -------
        lnt : array_like
            The log of the transfer function at lnk.
        """
        a = self.params["a"]
        b = self.params["b"]
        c = self.params["c"]
        d = self.params["d"]
        e = self.params["e"]

        Gamma = self.cosmo.Om0 * self.cosmo.h

        if self.params["use_sugiyama_baryons"]:
            Gamma *= np.exp(-self.cosmo.Ob0 * (1 + 1 / self.cosmo.Om0))
        elif self.params["use_liddle_baryons"]:
            Gamma *= np.exp(
                -self.cosmo.Ob0
                * (1 + np.sqrt(self.cosmo.Ob0 * self.cosmo.h) / self.cosmo.Om0)
            )

        q = np.exp(lnk) / Gamma

        return np.log(
            np.log(1.0 + a * q)
            / (a * q)
            * (1 + b * q + (c * q) ** 2 + (d * q) ** 3 + (e * q) ** 4) ** (-0.25)
        )


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

    .. math:: T(k) = \left[1 + (\tilde{a}k + (\tilde{b}k)^{3/2} +
              (\tilde{c}k)^2)^\nu\right]^{-1/\nu}

    where :math:`\tilde{x} = x\alpha` and

    .. math:: \alpha = \frac{0.3\times 0.75^2}{\Omega_{m,0} h^2}.
    """
    _defaults = {"a": 37.1, "b": 21.1, "c": 10.8, "nu": 1.12}

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

        scale = (0.3 * 0.75 ** 2) / (self.cosmo.Om0 * self.cosmo.h)

        a = self.params["a"] * scale
        b = self.params["b"] * scale
        c = self.params["c"] * scale
        nu = self.params["nu"]
        k = np.exp(lnk)
        return np.log((1 + (a * k + (b * k) ** 1.5 + (c * k) ** 2) ** nu) ** (-1 / nu))


class EH(EH_BAO):
    """Alias of :class:`EH_BAO`."""

    pass
