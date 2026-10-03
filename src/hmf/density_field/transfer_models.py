"""
Models for computing the transfer function.

Note that these are not transfer function "frameworks". The framework is found
in :mod:`~hmf.density_field.transfer`.
"""

import pickle
import warnings
from copy import deepcopy
from typing import Any, ClassVar

import numpy as np
from astropy import cosmology
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from .._internals import _references as refs
from .._internals._framework import Component, pluggable
from .._internals._utils import resolve_matter_species, set_camb_cosmology

try:
    import camb

    HAVE_CAMB = True
except ImportError:  # pragma: no cover
    HAVE_CAMB = False

try:
    import classy

    HAVE_CLASS = True
except ImportError:  # pragma: no cover
    HAVE_CLASS = False

_allfits = ["CAMB", "CLASS", "FromFile", "EH_BAO", "EH_NoBAO", "BBKS", "BondEfs"]

# Zero-based columns of a CAMB transfer-function output file for each matter species.
# These are also the rows of CAMB's ``MatterTransferData.transfer_data``: CAMB's
# ``Transfer_*`` constants are 1-based (Fortran), so these are
# ``camb.model.Transfer_tot - 1`` and ``camb.model.Transfer_nonu - 1`` (checked in tests).
_CAMB_FILE_COLUMNS: dict[str, int] = {"tot": 6, "cb": 7}


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

    _defaults: ClassVar[dict[str, Any]] = {}

    def __init__(self, cosmo, **model_parameters):
        self.cosmo = cosmo
        super().__init__(**model_parameters)

    def lnt(self, lnk):
        r"""
        Natural log of the transfer function.

        Parameters
        ----------
        lnk : array_like
            Wavenumbers [Mpc/h]

        Returns
        -------
        lnt : array_like
            The log of the transfer function at lnk.
        """


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
        :matter_species: str or None
            Which matter density field to read from a CAMB-format file: ``"cb"``, the
            CDM+baryon transfer function (column 7, CAMB's ``Transfer_nonu``), or
            ``"tot"``, the total matter transfer function (column 6, CAMB's
            ``Transfer_tot``, including massive neutrinos). See :class:`CAMB` for
            when to use which. The default, ``None``, means ``"cb"`` if the file has
            that column, with a warning if the cosmology has massive neutrinos, and
            ``"tot"`` for older CAMB files without it. A two-column ``(k, T)`` file
            is used as-is, whatever this is set to.
    """

    _defaults: ClassVar[dict[str, str | None]] = {"fname": "", "matter_species": None}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Subclasses that don't read CAMB columns (e.g. FromArray) don't have the option.
        # The default (None) is resolved in lnt, once we know the file's columns.
        if self.params.get("matter_species") is not None:
            resolve_matter_species(self.params["matter_species"], self.cosmo, type(self).__name__)

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
        Natural log of the transfer function.

        Parameters
        ----------
        lnk : array_like
            Wavenumbers [Mpc/h]

        Returns
        -------
        lnt : array_like
            The log of the transfer function at lnk.
        """
        data = np.genfromtxt(self.params["fname"])
        species = self.params["matter_species"]
        if species is None:
            # Only a CAMB file that has the CDM+baryon column is affected by the default.
            species = (
                resolve_matter_species(None, self.cosmo, "FromFile")
                if data.shape[1] > _CAMB_FILE_COLUMNS["cb"]
                else "tot"
            )

        col = _CAMB_FILE_COLUMNS[species]
        if data.shape[1] > col:
            T = np.log(data[:, [0, col]].T)
        elif species == "tot" or data.shape[1] == 2:
            T = np.log(data[:, [0, 1]].T)
        else:
            raise ValueError(
                f"{self.params['fname']} has {data.shape[1]} columns, so it has no "
                f"CDM+baryon column (column {col}) to read for matter_species='cb'."
            )

        if lnk[0] < T[0, 0]:
            lnkout, lnT = self._check_low_k(T[0, :], T[1, :], lnk[0])
        else:
            lnkout = T[0, :]
            lnT = T[1, :]
        return Spline(lnkout, lnT, k=1)(lnk)


def check_boltzmann_cosmology(cosmo: cosmology.FLRW, name: str) -> None:
    """
    Raise an error if a Boltzmann code (CAMB, CLASS) cannot compute ``cosmo``.

    Parameters
    ----------
    cosmo
        The cosmology.
    name
        Name of the model using the Boltzmann code, used in the error messages.

    Raises
    ------
    ValueError
        If ``cosmo`` is not a LambdaCDM, wCDM or w0waCDM cosmology, or does not set
        the baryon density or the CMB temperature.
    """
    if not isinstance(cosmo, (cosmology.LambdaCDM, cosmology.wCDM, cosmology.w0waCDM)):
        # Kept as ValueError (not TypeError): part of the public API contract,
        # asserted verbatim by
        # tests/test_transfer_models.py::test_camb_rejects_non_lcdm_cosmology.
        raise ValueError(f"{name} will only work with LCDM or wCDM cosmologies")  # noqa: TRY004

    if cosmo.Ob0 is None or cosmo.Ob0 == 0.0:
        raise ValueError(
            f"To use {name}, you must set the baryon density in the cosmology explicitly."
        )

    if cosmo.Tcmb0.value == 0:
        raise ValueError(
            f"If using {name}, the CMB temperature must be set explicitly in the cosmology."
        )


class _BoltzmannTransfer(FromFile, abstract=True):
    r"""
    Shared machinery for transfer models computed by a Boltzmann code (CAMB, CLASS).

    Subclasses compute the transfer function on the code's own grid of wavenumbers
    (see :meth:`_compute_transfers`), and this class turns that into ``lnt``: it
    removes spurious low-k features, normalises the transfer function to unity at
    low k and, if ``extrapolate_with_eh`` is set, extrapolates it beyond the code's
    maximum wavenumber with the shape of the :class:`EH` fit. The results of a run
    are cached, keyed on the code's full input, and can be shared between components
    (see :meth:`_share_results`), since one run gives every matter species.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._validate_cosmology()
        self.params["matter_species"] = resolve_matter_species(
            self.params["matter_species"], self.cosmo, type(self).__name__
        )

        # Results of runs of the Boltzmann code, keyed on its full input.
        self._results: dict[str, dict[str, np.ndarray]] = {}

    def _validate_cosmology(self) -> None:
        """Raise an error if the Boltzmann code cannot compute ``cosmo``."""
        check_boltzmann_cosmology(self.cosmo, type(self).__name__)

    def _setup_extrapolation(self) -> None:
        """Create the EH transfer used to extrapolate to high k, if requested."""
        if self.params["extrapolate_with_eh"]:
            self._eh = EH(self.cosmo)

    def _results_key(self) -> str:
        """A string identifying the full input of a run of the Boltzmann code."""
        raise NotImplementedError

    def _compute_transfers(self) -> dict[str, np.ndarray]:
        """
        Run the Boltzmann code.

        Returns
        -------
        dict
            The wavenumbers, ``"kh"`` [h/Mpc], and the (un-normalised) transfer
            function of each species in :data:`~hmf._internals._utils.MATTER_SPECIES`
            on those wavenumbers.
        """
        raise NotImplementedError

    def _transfers(self) -> dict[str, np.ndarray]:
        """The transfer functions of all species, running the code only for new inputs."""
        key = self._results_key()
        if key not in self._results:
            self._results[key] = self._compute_transfers()
        return self._results[key]

    def _share_results(self, other: "_BoltzmannTransfer") -> None:
        """
        Let ``other`` reuse this component's runs of the Boltzmann code (and vice versa).

        Results are keyed on the full input of the code, so sharing is always safe:
        ``other`` only reuses a run made with exactly its own inputs. A single run
        computes every matter species, so this lets e.g. the component normalising
        ``sigma_8`` avoid a second run.

        Parameters
        ----------
        other
            The component to share results with. It must be of the same type.
        """
        if type(other) is not type(self):
            raise TypeError(
                f"Can only share results with another {type(self).__name__}, "
                f"not {type(other).__name__}."
            )
        other._results = self._results

    def lnt(self, lnk):
        r"""
        Natural log of the transfer function.

        Parameters
        ----------
        lnk : array_like
            Wavenumbers [Mpc/h]

        Returns
        -------
        lnt : array_like
            The log of the transfer function at lnk.
        """
        transfers = self._transfers()
        T = np.log([transfers["kh"], transfers[self.params["matter_species"]]])

        if lnk[0] < T[0, 0]:
            lnkout, lnT = self._check_low_k(T[0, :], T[1, :], lnk[0])
        else:
            lnkout = T[0, :]
            lnT = T[1, :]

        lnT -= lnT[0]

        if not self.params["extrapolate_with_eh"]:
            return Spline(lnkout, lnT, k=1)(lnk)

        # Now add a point one e-fold above the max, with an EH-generated transfer
        lnkout = np.concatenate((lnkout, [lnkout[-1] + 1]))
        # normalise EH at the final point computed by the Boltzmann code.
        norm = self._eh.lnt(lnkout[-2]) - lnT[-1]
        lnT = np.concatenate((lnT, [self._eh.lnt(lnkout[-1]) - norm]))

        lnkmin = lnkout.min()
        lnkmax = lnkout.max()

        inner_spline = Spline(lnkout, lnT, k=3)

        out = np.zeros_like(lnk)
        out[lnk < lnkmin] = 0
        out[(lnkmin <= lnk) & (lnk <= lnkmax)] = inner_spline(
            lnk[(lnkmin <= lnk) & (lnk <= lnkmax)]
        )
        out[lnk >= lnkmax] = self._eh.lnt(lnk[lnk >= lnkmax]) - norm

        return out


if HAVE_CAMB:

    class CAMB(_BoltzmannTransfer):
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
            **matter_species:** Which matter density field the transfer function
                                describes: ``"cb"`` or ``"tot"``. ``"cb"`` uses
                                CAMB's ``Transfer_nonu`` (the CDM+baryon
                                perturbation, :math:`\delta_{\rm cb}`), while
                                ``"tot"`` uses CAMB's ``Transfer_tot`` (the total
                                matter perturbation, :math:`\delta_{\rm tot}`,
                                including massive neutrinos). The two are identical
                                for massless neutrinos. Halo mass function fits
                                calibrated on simulations with massive neutrinos
                                should be used with ``"cb"``, since the HMF is found
                                to be universal in terms of the CDM+baryon field
                                (Costanzi et al. 2013; Castorina et al. 2014); it
                                also matches ``mean_density0``, which excludes
                                neutrinos. The default, ``None``, means ``"cb"``,
                                with a warning if the cosmology has massive
                                neutrinos (earlier versions of hmf used ``"tot"``).
                                By default ``sigma_8`` still describes the total
                                matter field; see ``sigma_8_species`` on the
                                :class:`~hmf.density_field.transfer.Transfer`
                                framework and :doc:`/massive_neutrinos`.

        Notes
        -----
        Neutrino masses are passed to CAMB via the ``mnu`` parameter (sum of neutrino
        masses in eV), from which CAMB computes the neutrino physical density
        ``omnuh2``, and the individual masses in ``cosmo.m_nu`` set the number of
        massive species and their mass eigenstates. So ``m_nu=[0.1, 0.1, 0.1]`` eV is
        three massive species of 0.1 eV each (not one of 0.3 eV), which sets the
        free-streaming scale correctly. The cold dark matter density ``omch2`` is set from astropy's
        ``Odm0`` attribute (i.e. ``cosmo.Odm0 * cosmo.h**2``), which represents
        CDM-only density (excluding neutrinos and baryons). In astropy, ``Om0``
        represents the sum of CDM and baryonic matter only (not massive neutrinos),
        so ``Odm0 = Om0 - Ob0``. The neutrino contribution to the matter budget is
        accounted for separately by CAMB from the ``mnu`` parameter, ensuring the
        total matter density (CDM + baryons + neutrinos) is correctly captured.
        """

        references: ClassVar[tuple[str, ...]] = (refs.CAMB,)

        _defaults: ClassVar[dict[str, Any]] = {
            "camb_params": None,
            "dark_energy_params": {},
            "extrapolate_with_eh": None,
            "kmax": None,
            "matter_species": None,
        }

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # Save the CAMB object properly for use
            # Set the cosmology
            if self.params["camb_params"] is not None:
                # Work on our own copy: we set the cosmology on it below, and must not
                # change (or share state with) the caller's object.
                self.params["camb_params"] = self.params["camb_params"].copy()
            else:
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

            self._set_camb_cosmology()

            if self.params["extrapolate_with_eh"] is None:
                warnings.warn(
                    "'extrapolate_with_eh' was not set. Defaulting to True, which is "
                    "different behaviour than versions <=3.4.4. This warning may be "
                    "removed in v4.0. Silence it by setting extrapolate_with_eh explicitly.",
                    stacklevel=2,
                )
                self.params["extrapolate_with_eh"] = True

            self._setup_extrapolation()

        def _set_camb_cosmology(self):
            """Set the cosmology (and transfer output) of the CAMBparams from ``cosmo``."""
            set_camb_cosmology(self.params["camb_params"], self.cosmo)
            self.params["camb_params"].WantTransfer = True

            # Set the DE equation of state. We only support constant w.
            if isinstance(self.cosmo, cosmology.wCDM):
                self.params["camb_params"].set_dark_energy(w=self.cosmo.w0)
            elif isinstance(self.cosmo, cosmology.w0waCDM):
                self.params["camb_params"].set_dark_energy(w=self.cosmo.w0, wa=self.cosmo.wa)

        def _results_key(self) -> str:
            """The full state of the CAMBparams, which identifies a CAMB run."""
            return repr(self.params["camb_params"])

        def _compute_transfers(self) -> dict[str, np.ndarray]:
            """Run CAMB, returning the transfer function of every matter species."""
            camb_transfers = camb.get_transfer_functions(self.params["camb_params"])
            T = camb_transfers.get_matter_transfer_data().transfer_data
            out = {"kh": T[camb.model.Transfer_kh - 1, :, 0]}
            out.update({sp: T[col, :, 0] for sp, col in _CAMB_FILE_COLUMNS.items()})
            return out

        def __getstate__(self):
            """Get the state of the object, including converting the CAMBparams object to a dict."""
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
                        f"{dir(p)}",
                        stacklevel=2,
                    )

                except (pickle.PicklingError, TypeError):
                    warnings.warn(f"CAMB key {pk} is not pickle-able.", stacklevel=2)

            # Deepcopy self
            this = {}
            for key, val in self.__dict__.items():
                if key != "params":
                    this[key] = deepcopy(val)

            this["params"] = {
                key: (dct if key == "camb_params" else deepcopy(val))
                for key, val in self.params.items()
            }

            return this

        def __setstate__(self, state):
            """Set the state of the object, including reconstructing the CAMBparams object."""
            self.__dict__ = state

            self.params["camb_params"] = camb.CAMBparams(**self.params["camb_params"])
            # Not all of the CAMBparams state can be saved (e.g. the neutrino mass
            # fractions), so set the cosmology again to make it consistent.
            self._set_camb_cosmology()
            # States saved before CAMB results were memoised don't have them, and older
            # states memoised CAMB's raw transfer data, under another name.
            self.__dict__.pop("_camb_results", None)
            self.__dict__.setdefault("_results", {})


# CLASS input parameters that describe the background cosmology (including aliases),
# which must come from the astropy cosmology, or that the CLASS model relies on.
_CLASS_FIXED_PARAMS = frozenset(
    {
        "h",
        "H0",
        "100*theta_s",
        "theta_s_100",
        "omega_b",
        "Omega_b",
        "omega_cdm",
        "Omega_cdm",
        "omega_m",
        "Omega_m",
        "Omega_k",
        "T_cmb",
        "Omega_g",
        "omega_g",
        "N_ur",
        "Omega_ur",
        "omega_ur",
        "N_ncdm",
        "m_ncdm",
        "deg_ncdm",
        "T_ncdm",
        "Omega_ncdm",
        "omega_ncdm",
        "Omega_Lambda",
        "Omega_fld",
        "w0_fld",
        "wa_fld",
        "Omega_scf",
        "gauge",
    }
)


def class_cosmology(cosmo: cosmology.FLRW) -> dict[str, Any]:
    r"""
    The CLASS input parameters that describe an astropy cosmology.

    Neutrinos are described as astropy describes them: each of the
    :math:`\lfloor N_{\rm eff} \rfloor` species has the standard temperature,
    :math:`(4/11)^{1/3} T_{\rm CMB}`, and contributes :math:`N_{\rm eff} /
    \lfloor N_{\rm eff} \rfloor` to :math:`N_{\rm eff}`. The massive species (those
    with non-zero ``m_nu``) are CLASS's non-cold dark matter species (``ncdm``), with
    that contribution as their degeneracy, and the rest are ultra-relativistic
    (``N_ur``). The density of CDM is that of ``cosmo.Odm0``, which excludes
    neutrinos.

    Parameters
    ----------
    cosmo
        The cosmology. Must be a :class:`~astropy.cosmology.LambdaCDM`,
        :class:`~astropy.cosmology.wCDM` or :class:`~astropy.cosmology.w0waCDM`
        (or one of their flat versions), with the baryon density and CMB temperature
        set.

    Returns
    -------
    dict
        CLASS input parameters.
    """
    h = cosmo.h
    params = {
        "h": h,
        "omega_b": cosmo.Ob0 * h**2,
        "omega_cdm": cosmo.Odm0 * h**2,
        "Omega_k": cosmo.Ok0,
        "T_cmb": cosmo.Tcmb0.value,
    }

    n_nu = int(np.floor(cosmo.Neff))
    neff_per_nu = cosmo.Neff / n_nu if n_nu else 0.0
    m_nu = np.atleast_1d(cosmo.m_nu.to_value("eV"))
    m_nu = m_nu[m_nu > 0]
    params["N_ur"] = cosmo.Neff - len(m_nu) * neff_per_nu
    if len(m_nu):
        params["N_ncdm"] = len(m_nu)
        params["m_ncdm"] = ",".join(repr(float(m)) for m in m_nu)
        params["deg_ncdm"] = ",".join([repr(neff_per_nu)] * len(m_nu))
        params["T_ncdm"] = ",".join([repr((4 / 11) ** (1 / 3))] * len(m_nu))

    if isinstance(cosmo, (cosmology.wCDM, cosmology.w0waCDM)):
        # Dark energy is a fluid, with density set by closure.
        params["Omega_Lambda"] = 0.0
        params["w0_fld"] = cosmo.w0
        params["wa_fld"] = getattr(cosmo, "wa", 0.0)

    return params


class CLASS(_BoltzmannTransfer):
    r"""
    Transfer function computed by CLASS.

    This needs the optional ``classy`` package (the Python wrapper of CLASS), which
    you can install with ``pip install hmf[class]``.

    It is used in the same way as :class:`CAMB`: the cosmology (including massive
    neutrinos and a constant or :math:`w_0`-:math:`w_a` dark energy equation of
    state) is taken from ``cosmo``, CLASS computes the transfer function up to its
    maximum wavenumber, and beyond that it is (optionally) extrapolated with the
    shape of the :class:`EH` fit.

    Parameters
    ----------
    cosmo : :class:`astropy.cosmology.FLRW` instance
        The cosmology used in the calculation
    \*\*model_parameters : unpack-dict
        Parameters specific to this model.

        **class_params:** A dictionary of further CLASS input parameters (as
                          passed to ``classy.Class.set``), e.g. precision
                          parameters, or ``YHe``. Parameters that describe
                          the cosmology are set from ``cosmo`` (see
                          :func:`class_cosmology`), and it is an error to set
                          them here, or the ``gauge``, which is always
                          synchronous. ``output`` always includes ``mTk``.
        **extrapolate_with_eh:** Whether to extrapolate past the CLASS kmax by
                                 using an EH model. Default ``True``.
        **kmax:** The maximum wavenumber [h/Mpc] that CLASS computes the
                  transfer function to (its ``P_k_max_h/Mpc`` parameter). The
                  default, ``None``, uses CLASS's default (about 1 h/Mpc).
                  Above it, the transfer function is extrapolated (see
                  ``extrapolate_with_eh``).
        **matter_species:** Which matter density field the transfer function
                            describes: ``"cb"``, the CDM+baryon field, or
                            ``"tot"``, the total matter field (including massive
                            neutrinos, CLASS's ``d_m``). See :class:`CAMB` for
                            when to use which. The default, ``None``, means
                            ``"cb"``, with a warning if the cosmology has massive
                            neutrinos.

    Notes
    -----
    The transfer function is :math:`-\delta(k)/k^2` at :math:`z=0`, normalised to
    unity at low :math:`k`, with :math:`\delta` the density contrast output by CLASS
    in the synchronous gauge, as for :class:`CAMB`. Its shape is that of CLASS's
    own linear matter power spectrum (``pk_lin`` for ``"tot"``, ``pk_cb_lin`` for
    ``"cb"``). CLASS outputs no CDM+baryon transfer function, so
    :math:`\delta_{\rm cb}` is the density-weighted mean of :math:`\delta_{\rm cdm}`
    and :math:`\delta_{\rm b}`.

    The CLASS input made from ``cosmo`` describes each massive neutrino in
    ``cosmo.m_nu`` separately, as astropy does.
    """

    references: ClassVar[tuple[str, ...]] = (refs.CLASS_I, refs.CLASS_II)

    _defaults: ClassVar[dict[str, Any]] = {
        "class_params": None,
        "extrapolate_with_eh": True,
        "kmax": None,
        "matter_species": None,
    }

    def __init__(self, *args, **kwargs):
        if not HAVE_CLASS:
            raise ImportError(
                "The CLASS transfer model needs the classy package. Install it with "
                "`pip install hmf[class]` (or `pip install classy`)."
            )
        super().__init__(*args, **kwargs)

        self.params["class_params"] = dict(self.params["class_params"] or {})
        if clash := sorted(set(self.params["class_params"]) & _CLASS_FIXED_PARAMS):
            raise ValueError(
                f"class_params cannot set {clash}. The cosmology is set from the astropy "
                "cosmology (change that instead), and the gauge must be synchronous."
            )

        self._setup_extrapolation()

    def _class_input(self) -> dict[str, Any]:
        """The full input parameters for CLASS."""
        params = dict(self.params["class_params"])
        if self.params["kmax"] is not None:
            params["P_k_max_h/Mpc"] = self.params["kmax"]

        outputs = [o.strip() for o in str(params.get("output", "")).split(",") if o.strip()]
        if "mTk" not in outputs:
            outputs.append("mTk")
        params["output"] = ", ".join(outputs)

        params.update(class_cosmology(self.cosmo))
        # delta_cdm and delta_b (unlike delta_m) are output in the gauge used, and
        # their density-weighted mean is the CDM+baryon field in the synchronous gauge.
        params["gauge"] = "synchronous"
        return params

    def _results_key(self) -> str:
        """The full CLASS input, which identifies a CLASS run."""
        return repr(sorted(self._class_input().items()))

    def _compute_transfers(self) -> dict[str, np.ndarray]:
        """Run CLASS, returning the transfer function of every matter species."""
        cl = classy.Class()
        cl.set(self._class_input())
        try:
            cl.compute()
            transfers = cl.get_transfer(z=0.0)
            omega_cdm, omega_b = cl.Omega0_cdm(), cl.Omega_b()
        finally:
            cl.struct_cleanup()
            cl.empty()

        k = transfers["k (h/Mpc)"]
        delta_cb = (omega_cdm * transfers["d_cdm"] + omega_b * transfers["d_b"]) / (
            omega_cdm + omega_b
        )
        return {"kh": k, "tot": -transfers["d_m"] / k**2, "cb": -delta_cb / k**2}


class FromArray(FromFile):
    r"""
    Use a spline over a given array to define the transfer function.

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

    _defaults: ClassVar[dict[str, Any]] = {"k": None, "T": None}

    def lnt(self, lnk):
        r"""
        Natural log of the transfer function.

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
            raise ValueError("You must supply an array for both k and T for this Transfer Model")
        if len(k) != len(T):
            raise ValueError("k and T must have same length")

        if lnk[0] < np.log(k.min()):
            lnkout, lnT = self._check_low_k(np.log(k), np.log(T), lnk[0])
        else:
            lnkout = np.log(k)
            lnT = np.log(T)
        return Spline(lnkout, lnT, k=1)(lnk)


class EH_BAO(TransferComponent):
    r"""
    Eisenstein & Hu (1998) fitting function with BAO wiggles.

    From EH1998, Eqs. 26,28-31. Code adapted from CHOMP.

    Parameters
    ----------
    cosmo : :class:`astropy.cosmology.FLRW` instance
        The cosmology used in the calculation
    \*\*model_parameters : unpack-dict
        Parameters specific to this model. In this case, there
        are no model parameters.
    """

    references: ClassVar[tuple[str, ...]] = (refs.EH98,)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_params()

    def _set_params(self):
        """Port of ``TFset_parameters`` from original EH code."""
        self.Obh2 = self.cosmo.Ob0 * self.cosmo.h**2
        self.Omh2 = self.cosmo.Om0 * self.cosmo.h**2
        self.f_baryon = self.cosmo.Ob0 / self.cosmo.Om0

        self.theta_cmb = self.cosmo.Tcmb0.value / 2.7

        self.z_eq = 2.5e4 * self.Omh2 * self.theta_cmb ** (-4)  # really 1+z
        self.k_eq = 7.46e-2 * self.Omh2 * self.theta_cmb ** (-2)  # units Mpc^-1 (no h!)

        self.z_drag_b1 = 0.313 * self.Omh2**-0.419 * (1.0 + 0.607 * self.Omh2**0.674)
        self.z_drag_b2 = 0.238 * self.Omh2**0.223
        self.z_drag = (
            1291.0
            * self.Omh2**0.251
            / (1.0 + 0.659 * self.Omh2**0.828)
            * (1.0 + self.z_drag_b1 * self.Obh2**self.z_drag_b2)
        )

        self.r_drag = 31.5 * self.Obh2 * self.theta_cmb**-4 * (1000.0 / (1 + self.z_drag))
        self.r_eq = 31.5 * self.Obh2 * self.theta_cmb**-4 * (1000.0 / self.z_eq)

        self.sound_horizon = (
            (2.0 / (3.0 * self.k_eq))
            * np.sqrt(6.0 / self.r_eq)
            * np.log(
                (np.sqrt(1.0 + self.r_drag) + np.sqrt(self.r_drag + self.r_eq))
                / (1.0 + np.sqrt(self.r_eq))
            )
        )

        self.k_silk = (
            1.6 * self.Obh2**0.52 * self.Omh2**0.73 * (1.0 + (10.4 * self.Omh2) ** (-0.95))
        )

        alpha_c_a1 = (46.9 * self.Omh2) ** 0.670 * (1.0 + (32.1 * self.Omh2) ** (-0.532))
        alpha_c_a2 = (12.0 * self.Omh2) ** 0.424 * (1.0 + (45.0 * self.Omh2) ** (-0.582))
        self.alpha_c = alpha_c_a1 ** (-self.f_baryon) * alpha_c_a2 ** (-(self.f_baryon**3))

        beta_c_b1 = 0.944 / (1.0 + (458.0 * self.Omh2) ** -0.708)
        beta_c_b2 = (0.395 * self.Omh2) ** -0.0266
        self.beta_c = 1.0 / (1.0 + beta_c_b1 * ((1 - self.f_baryon) ** beta_c_b2 - 1))

        y = self.z_eq / (1 + self.z_drag)
        alpha_b_G = y * (
            -6 * np.sqrt(1 + y) + (2 + 3 * y) * np.log((np.sqrt(1 + y) + 1) / (np.sqrt(1 + y) - 1))
        )
        self.alpha_b = (
            2.07 * self.k_eq * self.sound_horizon * (1.0 + self.r_drag) ** -0.75 * alpha_b_G
        )

        self.beta_node = 8.41 * self.Omh2**0.435
        self.beta_b = (
            0.5
            + self.f_baryon
            + (3.0 - 2.0 * self.f_baryon) * np.sqrt((17.2 * self.Omh2) ** 2 + 1.0)
        )

    @property
    def k_peak(self):
        """The k_peak parameter from EH1998, Eq. 29, in units of Mpc^-1 (no h)."""
        return 2.5 * np.pi * (1 + 0.217 * self.Omh2) / self.sound_horizon

    @property
    def sound_horizon_fit(self):
        """Sound horizon in Mpc/h."""
        return self.cosmo.h * 44.5 * np.log(9.83 / self.Omh2) / np.sqrt(1 + 10 * (self.Obh2**0.75))

    def lnt(self, lnk):
        r"""
        Natural log of the transfer function.

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
        T_c_C_alpha = (14.2 / self.alpha_c) + 386.0 / (1.0 + 69.9 * q**1.08)
        T_c_C_noalpha = 14.2 + 386.0 / (1.0 + 69.9 * q**1.08)

        T_c_f = 1.0 / (1.0 + (ks / 5.4) ** 4)

        def term(a, b):
            return a / (a + b * q**2)

        T_c = T_c_f * term(T_c_ln_beta, T_c_C_noalpha) + (1 - T_c_f) * term(
            T_c_ln_beta, T_c_C_alpha
        )

        s_tilde = self.sound_horizon / (1.0 + (self.beta_node / ks) ** 3) ** (1.0 / 3.0)
        ks_tilde = k * s_tilde

        T_b_T0 = term(T_c_ln_nobeta, T_c_C_noalpha)
        Tb1 = T_b_T0 / (1.0 + (ks / 5.2) ** 2)
        Tb2 = (self.alpha_b / (1.0 + (self.beta_b / ks) ** 3)) * np.exp(-((k / self.k_silk) ** 1.4))
        T_b = np.sin(ks_tilde) / ks_tilde * (Tb1 + Tb2)

        return np.log(self.f_baryon * T_b + (1 - self.f_baryon) * T_c)


class EH_NoBAO(EH_BAO):
    r"""
    Eisenstein & Hu (1998) fitting function without BAO wiggles.

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
        """The alpha_gamma parameter from EH1998, Eq. 31."""
        return (
            1
            - 0.328 * np.log(431 * self.Omh2) * self.f_baryon
            + 0.38 * np.log(22.3 * self.Omh2) * self.f_baryon**2
        )

    def lnt(self, lnk):
        r"""
        Natural log of the transfer function.

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

        gamma_eff = self.Omh2 * (self.alpha_gamma + (1 - self.alpha_gamma) / (1 + (0.43 * ks) ** 4))
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

    Further modifications can be made in the presence of baryons. With
    ``use_sugiyama_baryons``, the form of the Sugiyama (1995) preprint
    (astro-ph/9412025, Eq. 3.9; also quoted by Liddle et al. 1996, Eq. 6) is used:

    .. math:: \Gamma \rightarrow \Gamma
              \exp\left(-\Omega_{b,0}(1 + 1/\Omega_{m,0})\right).

    With ``use_liddle_baryons`` (the default), the published form of Sugiyama
    (1995, ApJS 100, 281), as quoted by Meiksin, White & Peacock (1999, Eq. 4) and
    Liddle & Lyth (2000, Eq. 5.14), is used:

    .. math:: \Gamma \rightarrow \Gamma
              \exp\left(-\Omega_{b,0}(1 + \sqrt{2h}/\Omega_{m,0})\right).

    The two agree exactly at :math:`h = 0.5`. If both are set,
    ``use_sugiyama_baryons`` takes precedence.
    """

    references: ClassVar[tuple[str, ...]] = (refs.BBKS86,)

    _defaults: ClassVar[dict[str, Any]] = {
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
        Natural log of the transfer function.

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
            Gamma *= np.exp(-self.cosmo.Ob0 * (1 + np.sqrt(2 * self.cosmo.h) / self.cosmo.Om0))

        q = np.exp(lnk) / Gamma

        return np.log(
            np.log(1.0 + a * q)
            / (a * q)
            * (1 + b * q + (c * q) ** 2 + (d * q) ** 3 + (e * q) ** 4) ** (-0.25)
        )


class BondEfs(TransferComponent):
    r"""
    Transfer function of Bond and Efstathiou, in the shape-parameter form of EBW92.

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
    The fit is given by Efstathiou, Bond & White (1992, EBW92), eq. 7:

    .. math:: T(k) = \left[1 + \left(aq + (bq)^{3/2} +
              (cq)^2\right)^\nu\right]^{-1/\nu},
              \qquad q = k/\Gamma, \qquad \Gamma = \Omega_{m,0} h,

    with :math:`k` in :math:`h\,{\rm Mpc}^{-1}` and the defaults
    :math:`a = 6.4`, :math:`b = 3.0`, :math:`c = 1.7` :math:`h^{-1}{\rm Mpc}` and
    :math:`\nu = 1.13`. The transfer function depends on cosmology only through
    :math:`\Gamma`. These are the coefficients that the GIF and Virgo simulations
    used for their initial conditions (Jenkins et al. 1998, eq. 4; Jenkins et al.
    2001, eq. 1), which cite BE84 for them.

    The functional form is BE84's eq. 6. BE84 fit it separately to each model in
    their Table 1 (coefficients in Mpc, :math:`k` in :math:`{\rm Mpc}^{-1}`) and note
    that :math:`a, b, c` should scale as :math:`(\Omega h^2)^{-1}`. In units of
    :math:`h^{-1}{\rm Mpc}` that scaling is :math:`1/\Gamma`. EBW92's coefficients
    are BE84's :math:`\Omega = 1`, :math:`h = 0.75` row (11.3, 5.29, 3.10 Mpc;
    :math:`\nu = 1.13`) times :math:`\Omega h^2`, rounded to two figures.

    :math:`\Gamma = \Omega_{m,0} h` is the exact shape parameter only for a
    vanishing baryon fraction. The BE84 models have :math:`\Omega_B = 0.03`, so this
    form describes a nearly baryon-free universe. Baryon suppression of small-scale
    power is not included; use :class:`EH_NoBAO` for that.

    Earlier versions of hmf used BE84's :math:`\Omega = 0.3`, :math:`h = 0.75`
    row (37.1, 21.1, 10.8 Mpc, :math:`\nu = 1.12`) as defaults, scaled by
    :math:`0.3 \times 0.75^2/(\Omega_{m,0} h)`. That row has a 10% baryon
    fraction, which the scaling then applied to every cosmology. The parameters
    :math:`a, b, c` are now in units of :math:`h^{-1}{\rm Mpc}` at
    :math:`\Gamma = 1`. To recover the old result, pass ``a=6.260625``,
    ``b=3.560625``, ``c=1.8225`` and ``nu=1.12``.
    """

    references: ClassVar[tuple[str, ...]] = (refs.BE84, refs.EBW92)

    _defaults: ClassVar[dict[str, float]] = {"a": 6.4, "b": 3.0, "c": 1.7, "nu": 1.13}

    def lnt(self, lnk):
        """
        Natural log of the transfer function.

        Parameters
        ----------
        lnk : array_like
            Natural log of wavenumbers [h/Mpc]

        Returns
        -------
        lnt : array_like
            The log of the transfer function at lnk.
        """
        q = np.exp(lnk) / (self.cosmo.Om0 * self.cosmo.h)

        a = self.params["a"]
        b = self.params["b"]
        c = self.params["c"]
        nu = self.params["nu"]
        return -np.log1p((a * q + (b * q) ** 1.5 + (c * q) ** 2) ** nu) / nu


class EH(EH_BAO):
    """Alias of :class:`EH_BAO`."""
