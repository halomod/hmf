"""Transfer-function models: the :class:`TransferModel` kind and its models.

A transfer model turns a cosmology into the transfer function T(k) at z = 0 of every
matter species, normalised to T -> 1 as k -> 0. It has no cosmology of its own: its
:meth:`~TransferModel.solve` takes the cosmology and the wavenumber accuracy, and
returns a :class:`TransferSolution`, which evaluates ln T(k) on plain arrays (k in
h/Mpc). The :class:`~hmf.core.transfer.Transfer` stage wraps that in a unit-checked
public API.

The matter species are ``"cb"`` (CDM + baryons) and ``"tot"`` (total matter,
including massive neutrinos). The Boltzmann codes (:class:`CAMB`, :class:`CLASS`)
compute both in one run. The fitting formulae have no massive neutrinos, so they give
the same transfer function for both.

Models computed by a Boltzmann code, or given as a table (:class:`FromArray`,
:class:`FromFile`), are interpolated and extrapolated by
:class:`~hmf.core._kernels.transfer.TabulatedTransfer`: smoothly in value and slope
beyond the table's largest wavenumber, following the shape of the EH98 fit.

CAMB and classy are only imported when a model that needs them is solved, so this
module imports without them.
"""

from __future__ import annotations

import abc
import functools
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Literal

import astropy.units as u
import attrs
import numpy as np
import numpy.typing as npt
from astropy.cosmology import FLRW

from . import _references as refs
from ._boltzmann import (
    MATTER_SPECIES,
    BoltzmannRun,
    camb_cosmology_input,
    check_boltzmann_cosmology,
    class_cosmology_input,
    get_run,
)
from ._fields import field
from ._kernels import transfer as kt
from ._validators import positive
from .accuracy import KAccuracy
from .cache import DiskCache
from .domain import Domain
from .model import Model
from .units import h_Mpc, quantity_field

__all__ = [
    "BBKS",
    "CAMB",
    "CLASS",
    "EH",
    "EH_BAO",
    "MATTER_SPECIES",
    "BondEfs",
    "EH_NoBAO",
    "FromArray",
    "FromFile",
    "Species",
    "TransferModel",
    "TransferSolution",
]

Array = npt.NDArray[np.float64]

#: A matter species: ``"cb"`` (CDM + baryons) or ``"tot"`` (total matter).
Species = Literal["cb", "tot"]

#: The wavenumber, in 1/Mpc (no h), at which the Boltzmann codes evaluate the growth
#: factor of each species (see :mod:`hmf.core.growth_models`).
BOLTZMANN_GROWTH_K_REF = 0.01


def check_species(species: str) -> str:
    """Raise ``ValueError`` unless ``species`` is one of :data:`MATTER_SPECIES`."""
    if species not in MATTER_SPECIES:
        raise ValueError(f"species must be one of {MATTER_SPECIES}, got {species!r}.")
    return species


@attrs.frozen(eq=False)
class TransferSolution:
    """The transfer function of every matter species, from one evaluation of a model.

    This is the kernel-level result of :meth:`TransferModel.solve`: its methods take
    and return plain arrays, with k in h/Mpc. They are pure.
    """

    #: ln T(k) of each species, as a function of k in h/Mpc.
    ln_t_functions: Mapping[str, Callable[[Array], Array]]
    #: The Boltzmann run behind it, if any.
    run: BoltzmannRun | None = None
    #: The largest wavenumber (h/Mpc) of the table, above which T(k) is extrapolated;
    #: ``None`` for a fitting formula.
    k_max_table: float | None = None

    @property
    def species(self) -> tuple[str, ...]:
        """The species this solution has."""
        return tuple(self.ln_t_functions)

    def ln_transfer(self, k: Array, species: str = "cb") -> Array:
        """Ln T(k) of ``species``, with k in h/Mpc."""
        return self.ln_t_functions[check_species(species)](np.asarray(k, dtype=float))

    def transfer(self, k: Array, species: str = "cb") -> Array:
        """T(k) of ``species``, with k in h/Mpc."""
        out: Array = np.exp(self.ln_transfer(k, species))
        return out


def _all_species(fn: Callable[[Array], Array]) -> Mapping[str, Callable[[Array], Array]]:
    """The same function for every species (for models without massive neutrinos)."""
    return MappingProxyType(dict.fromkeys(MATTER_SPECIES, fn))


def _positive_quantity(instance: Any, attribute: attrs.Attribute[Any], value: u.Quantity) -> None:
    """Validate that a scalar Quantity field is finite and > 0 (in its stored unit)."""
    positive(instance, attribute, float(value.value))


def _floats(value: Any) -> tuple[float, ...]:
    """Convert an array of dimensionless numbers to a tuple of floats."""
    if isinstance(value, u.Quantity):
        value = value.to_value(u.dimensionless_unscaled)
    return tuple(float(x) for x in np.atleast_1d(np.asarray(value, dtype=float)))


def _pairs(
    value: Mapping[str, Any] | tuple[tuple[str, Any], ...] | None,
) -> tuple[tuple[str, Any], ...]:
    """Convert a mapping of settings to a sorted tuple of pairs (hashable)."""
    if value is None:
        return ()
    items = value.items() if isinstance(value, Mapping) else value
    out = tuple(sorted((str(k), v) for k, v in items))
    for k, v in out:
        if not isinstance(v, (bool, int, float, str)):
            raise TypeError(
                f"Setting {k!r} must be a bool, int, float or str, not {type(v).__name__}."
            )
    return out


@attrs.frozen(kw_only=True)
class TransferModel(Model, kind=True):
    """The kind of transfer-function models.

    Subclasses implement :meth:`solve`.
    """

    #: The Boltzmann code that computes the model (``"camb"`` or ``"class"``), if any.
    backend: ClassVar[str | None] = None

    #: Where the model can be evaluated: any k > 0.
    valid_domain: ClassVar[Domain] = Domain({"k": (0 * h_Mpc, None)})

    #: Where the model was calibrated, if that is stated by its source.
    calibration_domain: ClassVar[Domain | None] = None

    def check_cosmology(self, cosmology: FLRW) -> None:
        """Raise if the model does not apply to ``cosmology`` (by default it does)."""

    @abc.abstractmethod
    def solve(
        self, cosmology: FLRW, accuracy: KAccuracy, *, disk_cache: DiskCache | None = None
    ) -> TransferSolution:
        """Compute the transfer function of every species for a cosmology.

        Parameters
        ----------
        cosmology
            The cosmology.
        accuracy
            The wavenumber accuracy; it sets the sampling of the Boltzmann codes.
        disk_cache
            Where to cache Boltzmann-code output on disk, if anywhere.

        Returns
        -------
        TransferSolution
        """


# ---------------------------------------------------------------------------------
# Fitting formulae
# ---------------------------------------------------------------------------------


def _eh98_scales(cosmology: FLRW) -> kt.EH98Scales:
    return kt.eh98_scales(
        h=cosmology.h,
        omega_m=cosmology.Om0,
        omega_b=cosmology.Ob0 or 0.0,
        t_cmb=cosmology.Tcmb0.to_value(u.K) or 2.7255,
    )


@attrs.frozen(kw_only=True)
class EH_BAO(TransferModel, alias="EH_BAO"):
    """The Eisenstein & Hu (1998) fit, with baryon acoustic oscillations.

    EH98 Eqs. 16-24, for the CDM + baryon field. The cosmology must have baryons.
    A cosmology without a CMB temperature (``Tcmb0 = 0``) is taken to have 2.7255 K.
    """

    references: ClassVar[tuple[str, ...]] = (refs.EH98,)
    parameter_source: ClassVar[str] = "Eisenstein & Hu 1998, ApJ 496, 605, Eqs. 2-24"

    def check_cosmology(self, cosmology: FLRW) -> None:
        """Raise unless the cosmology has baryons."""
        if not cosmology.Ob0:
            raise ValueError(f"{type(self).__name__} needs a cosmology with baryons (Ob0 > 0).")

    def solve(
        self, cosmology: FLRW, accuracy: KAccuracy, *, disk_cache: DiskCache | None = None
    ) -> TransferSolution:
        """Compute the transfer function (see :meth:`TransferModel.solve`)."""
        self.check_cosmology(cosmology)
        return TransferSolution(
            _all_species(functools.partial(kt.ln_t_eh98, scales=_eh98_scales(cosmology)))
        )


@attrs.frozen(kw_only=True)
class EH(EH_BAO, alias="EH"):
    """The Eisenstein & Hu (1998) fit with BAO: another name for :class:`EH_BAO`."""


@attrs.frozen(kw_only=True)
class EH_NoBAO(TransferModel, alias="EH_NoBAO"):
    """The Eisenstein & Hu (1998) "no-wiggle" fit, without BAO.

    EH98 Eqs. 26-31. The shape parameter q is that of their Eq. 28 (v3 used
    ``k / (13.4 k_eq)``, which differs from it by 4e-4).
    """

    references: ClassVar[tuple[str, ...]] = (refs.EH98,)
    parameter_source: ClassVar[str] = "Eisenstein & Hu 1998, ApJ 496, 605, Eqs. 26-31"

    def solve(
        self, cosmology: FLRW, accuracy: KAccuracy, *, disk_cache: DiskCache | None = None
    ) -> TransferSolution:
        """Compute the transfer function (see :meth:`TransferModel.solve`)."""
        return TransferSolution(
            _all_species(functools.partial(kt.ln_t_eh98_no_wiggle, scales=_eh98_scales(cosmology)))
        )


@attrs.frozen(kw_only=True)
class BBKS(TransferModel, alias="BBKS"):
    r"""The Bardeen et al. (1986) fit, with a baryon correction to the shape parameter.

    .. math:: T(k) = \frac{\ln(1+aq)}{aq}
              \left(1 + bq + (cq)^2 + (dq)^3 + (eq)^4\right)^{-1/4},
              \qquad q = k / \Gamma,

    with k in h/Mpc and :math:`\Gamma = \Omega_{m,0} h` (BBKS Eq. G3). The baryon
    correction multiplies :math:`\Gamma` by

    * ``"sugiyama95"`` (the default): :math:`\exp(-\Omega_b(1 + \sqrt{2h}/\Omega_m))`,
      the published form of Sugiyama (1995), as quoted by Meiksin, White & Peacock
      (1999, Eq. 4) and Liddle & Lyth (2000, Eq. 5.14);
    * ``"sugiyama95_preprint"``: :math:`\exp(-\Omega_b(1 + 1/\Omega_m))`, the form of
      the preprint (astro-ph/9412025, Eq. 3.9);
    * ``"none"``: 1.

    The two Sugiyama forms agree at h = 0.5. (In hmf v3 these were the
    ``use_liddle_baryons`` and ``use_sugiyama_baryons`` flags.)
    """

    references: ClassVar[tuple[str, ...]] = (refs.BBKS86, refs.SUGIYAMA95)
    parameter_source: ClassVar[str] = "Bardeen et al. 1986, ApJ 304, 15, Eq. G3"

    a: float = field(default=2.34, converter=float, doc="Coefficient a of Eq. G3.")
    b: float = field(default=3.89, converter=float, doc="Coefficient b of Eq. G3.")
    c: float = field(default=16.1, converter=float, doc="Coefficient c of Eq. G3.")
    d: float = field(default=5.46, converter=float, doc="Coefficient d of Eq. G3.")
    e: float = field(default=6.71, converter=float, doc="Coefficient e of Eq. G3.")
    baryons: Literal["sugiyama95", "sugiyama95_preprint", "none"] = field(
        default="sugiyama95",
        validator=attrs.validators.in_(("sugiyama95", "sugiyama95_preprint", "none")),
        doc="The baryon correction to the shape parameter (see above).",
    )

    def shape_parameter(self, cosmology: FLRW) -> float:
        """The shape parameter Gamma, in h/Mpc, including the baryon correction."""
        om, ob, h = cosmology.Om0, cosmology.Ob0 or 0.0, cosmology.h
        gamma = om * h
        if self.baryons == "sugiyama95":
            gamma *= math.exp(-ob * (1 + math.sqrt(2 * h) / om))
        elif self.baryons == "sugiyama95_preprint":
            gamma *= math.exp(-ob * (1 + 1 / om))
        return float(gamma)

    def solve(
        self, cosmology: FLRW, accuracy: KAccuracy, *, disk_cache: DiskCache | None = None
    ) -> TransferSolution:
        """Compute the transfer function (see :meth:`TransferModel.solve`)."""
        fn = functools.partial(
            kt.ln_t_bbks,
            gamma=self.shape_parameter(cosmology),
            a=self.a,
            b=self.b,
            c=self.c,
            d=self.d,
            e=self.e,
        )
        return TransferSolution(_all_species(fn))


@attrs.frozen(kw_only=True)
class BondEfs(TransferModel, alias="BondEfs"):
    r"""The Bond & Efstathiou (1984) fit, in the shape-parameter form of EBW92.

    .. math:: T(k) = \left[1 + \left(aq + (bq)^{3/2} + (cq)^2\right)^\nu\right]^{-1/\nu},
              \qquad q = k/\Gamma, \qquad \Gamma = \Omega_{m,0} h,

    with k in h/Mpc (Efstathiou, Bond & White 1992, Eq. 7). The defaults are EBW92's,
    which the GIF and Virgo simulations used. The form has no baryon suppression; use
    :class:`EH_NoBAO` for that.
    """

    references: ClassVar[tuple[str, ...]] = (refs.BE84, refs.EBW92)
    parameter_source: ClassVar[str] = "Efstathiou, Bond & White 1992, MNRAS 258, 1P, Eq. 7"

    a: float = field(default=6.4, converter=float, doc="Coefficient a, in Mpc/h at Gamma = 1.")
    b: float = field(default=3.0, converter=float, doc="Coefficient b, in Mpc/h at Gamma = 1.")
    c: float = field(default=1.7, converter=float, doc="Coefficient c, in Mpc/h at Gamma = 1.")
    nu: float = field(default=1.13, converter=float, doc="The exponent nu.")

    def solve(
        self, cosmology: FLRW, accuracy: KAccuracy, *, disk_cache: DiskCache | None = None
    ) -> TransferSolution:
        """Compute the transfer function (see :meth:`TransferModel.solve`)."""
        fn = functools.partial(
            kt.ln_t_bond_efs,
            gamma=float(cosmology.Om0 * cosmology.h),
            a=self.a,
            b=self.b,
            c=self.c,
            nu=self.nu,
        )
        return TransferSolution(_all_species(fn))


# ---------------------------------------------------------------------------------
# Tabulated models
# ---------------------------------------------------------------------------------


def _trim_low_k(k: Array, t: Array) -> tuple[Array, Array]:
    """Drop the nodes below a spurious low-k turn-up in a table, if there is one.

    Some versions of CAMB produce a transfer function that turns up at low k. The
    table is cut at the first node where ``|d ln T / d ln k| < 1e-4`` (the plateau),
    if any node is.
    """
    slope = np.abs(np.diff(np.log(t)) / np.diff(np.log(k)))
    flat = np.flatnonzero(slope < 1e-4)
    start = int(flat[0]) if flat.size else 0
    return k[start:], t[start:]


@attrs.frozen(kw_only=True)
class _Tabulated(TransferModel, abstract=True):
    """A transfer model given by a table, interpolated by TabulatedTransfer."""

    tail_decay_ln_k: float = field(
        default=1.0,
        converter=float,
        validator=positive,
        doc=(
            "Above the table's largest wavenumber, T(k) follows the EH98 no-wiggle shape, "
            "rescaled to match the table's value and logarithmic slope at the join. The "
            "tilt correction decays over this many e-folds in k."
        ),
    )

    @abc.abstractmethod
    def _table(
        self, cosmology: FLRW, accuracy: KAccuracy, disk_cache: DiskCache | None
    ) -> tuple[Array, Mapping[str, Array], BoltzmannRun | None]:
        """The table: k (h/Mpc), T of each species, and the run it came from."""

    def solve(
        self, cosmology: FLRW, accuracy: KAccuracy, *, disk_cache: DiskCache | None = None
    ) -> TransferSolution:
        """Compute the transfer function (see :meth:`TransferModel.solve`)."""
        k, transfers, run = self._table(cosmology, accuracy, disk_cache)
        scales = _eh98_scales(cosmology)
        tables = {}
        for species, t in transfers.items():
            kk, tt = _trim_low_k(np.asarray(k, dtype=float), np.abs(np.asarray(t, dtype=float)))
            tables[species] = kt.tabulate_transfer(kk, tt, scales, decay_ln_k=self.tail_decay_ln_k)
        return TransferSolution(
            MappingProxyType({s: tab.ln_t for s, tab in tables.items()}),
            run=run,
            k_max_table=min(tab.k_max for tab in tables.values()),
        )


@attrs.frozen(kw_only=True)
class FromArray(_Tabulated, alias="FromArray"):
    """A transfer function given as arrays.

    ``t`` is the transfer function of the CDM + baryon field; ``t_tot``, if given,
    that of total matter (else the same as ``t``). Neither needs to be normalised.
    """

    k: u.Quantity = quantity_field(
        h_Mpc,
        ndim=1,
        doc=(
            "Wavenumbers, increasing: a Quantity in h-units (e.g. k * hmf.core.units.h_Mpc), "
            "stored in h/Mpc."
        ),
    )
    t: tuple[float, ...] = field(converter=_floats, doc="The CDM + baryon transfer function at k.")
    t_tot: tuple[float, ...] | None = field(
        default=None,
        converter=attrs.converters.optional(_floats),
        doc="The total-matter transfer function at k; None means the same as t.",
    )

    @functools.cached_property
    def _k(self) -> Array:
        """The wavenumbers, as plain floats in h/Mpc."""
        return np.asarray(self.k.value)

    @t.validator
    def _check_t(
        self, attribute: attrs.Attribute[tuple[float, ...]], value: tuple[float, ...]
    ) -> None:
        if len(value) != len(self.k):
            raise ValueError(
                f"FromArray: k and t must have the same length ({len(self.k)}, {len(value)})."
            )

    @t_tot.validator
    def _check_t_tot(
        self, attribute: attrs.Attribute[Any], value: tuple[float, ...] | None
    ) -> None:
        if value is not None and len(value) != len(self.k):
            raise ValueError(
                f"FromArray: k and t_tot must have the same length ({len(self.k)}, {len(value)})."
            )

    def _table(
        self, cosmology: FLRW, accuracy: KAccuracy, disk_cache: DiskCache | None
    ) -> tuple[Array, Mapping[str, Array], BoltzmannRun | None]:
        t = np.array(self.t)
        return (
            np.array(self._k),
            {"cb": t, "tot": t if self.t_tot is None else np.array(self.t_tot)},
            None,
        )


#: Zero-based columns of a CAMB transfer-function file for each species.
_CAMB_FILE_COLUMNS: Mapping[str, int] = MappingProxyType({"tot": 6, "cb": 7})


@attrs.frozen(kw_only=True)
class FromFile(_Tabulated, alias="FromFile"):
    """A transfer function read from a text file.

    The file has either two columns, k (in h/Mpc) and T, used for both species; or
    the columns of a CAMB transfer-function file, of which column 0 is k (h/Mpc),
    column 6 total matter and column 7 (if present) CDM + baryons. The file is read
    when the model is solved; its content is not part of the model's value.
    """

    fname: Path = field(converter=Path, doc="The file to read.")

    def _table(
        self, cosmology: FLRW, accuracy: KAccuracy, disk_cache: DiskCache | None
    ) -> tuple[Array, Mapping[str, Array], BoltzmannRun | None]:
        data = np.atleast_2d(np.genfromtxt(self.fname))
        k = data[:, 0]
        if data.shape[1] == 2:
            return k, {"cb": data[:, 1], "tot": data[:, 1]}, None
        tot_col, cb_col = _CAMB_FILE_COLUMNS["tot"], _CAMB_FILE_COLUMNS["cb"]
        if data.shape[1] <= tot_col:
            raise ValueError(
                f"{self.fname} has {data.shape[1]} columns: expected 2 (k, T) or a CAMB "
                "transfer-function file."
            )
        cb = data[:, cb_col] if data.shape[1] > cb_col else data[:, tot_col]
        return k, {"cb": cb, "tot": data[:, tot_col]}, None


@attrs.frozen(kw_only=True)
class _Boltzmann(_Tabulated, abstract=True):
    """A transfer model computed by a Boltzmann code."""

    k_max: u.Quantity = quantity_field(
        h_Mpc,
        default=20.0 * h_Mpc,
        ndim=0,
        validator=_positive_quantity,
        doc=(
            "The largest wavenumber the code computes, a Quantity in h-units (stored in "
            "h/Mpc). Above it (strictly: above the code's last output node) T(k) is "
            "extrapolated smoothly (see tail_decay_ln_k)."
        ),
    )
    z_max_growth: float = field(
        default=20.0,
        converter=float,
        validator=positive,
        doc=(
            "The largest redshift at which the run also computes the growth factor, for "
            "a growth stage that shares the run."
        ),
    )

    @functools.cached_property
    def _k_max(self) -> float:
        """``k_max`` as a plain float in h/Mpc."""
        return float(self.k_max.value)

    @abc.abstractmethod
    def run_input(self, cosmology: FLRW, accuracy: KAccuracy) -> dict[str, Any]:
        """The input of the Boltzmann code: plain values that fully define its run.

        Parameters
        ----------
        cosmology
            The cosmology.
        accuracy
            The wavenumber accuracy.

        Returns
        -------
        dict
        """

    def check_cosmology(self, cosmology: FLRW) -> None:
        """Raise if the Boltzmann code can't compute ``cosmology``."""
        check_boltzmann_cosmology(cosmology, type(self).__name__)

    def run(
        self, cosmology: FLRW, accuracy: KAccuracy, *, disk_cache: DiskCache | None = None
    ) -> BoltzmannRun:
        """The output of the Boltzmann code for a cosmology: every species, one run.

        Runs are memoised in this process by the content hash of
        :meth:`run_input`, and cached on disk if ``disk_cache`` is given.
        """
        self.check_cosmology(cosmology)
        assert self.backend is not None
        return get_run(self.backend, self.run_input(cosmology, accuracy), disk_cache=disk_cache)

    def _table(
        self, cosmology: FLRW, accuracy: KAccuracy, disk_cache: DiskCache | None
    ) -> tuple[Array, Mapping[str, Array], BoltzmannRun | None]:
        run = self.run(cosmology, accuracy, disk_cache=disk_cache)
        return run.k, run.transfer, run


#: How many redshifts the CAMB run computes the growth factor at.
_CAMB_GROWTH_N_Z = 201


def _camb_k_per_logint(accuracy: KAccuracy) -> int:
    """CAMB's ``Transfer.k_per_logint`` for a wavenumber accuracy.

    0 (CAMB's own choice, about 0.05 in ln k at high k and coarser at low k) at the
    default and fast accuracies; otherwise a fifth of 1 / dln_k per e-fold, since T(k)
    is interpolated relative to EH98 (see :class:`TabulatedTransfer`) and needs far
    fewer nodes than the grid of the mass variance.
    """
    if accuracy.dln_k >= KAccuracy().dln_k:
        return 0
    return math.ceil(0.2 / accuracy.dln_k)


@attrs.frozen(kw_only=True)
class CAMB(_Boltzmann, alias="CAMB"):
    """The transfer function computed by CAMB.

    The cosmology must be a LambdaCDM, wCDM or w0waCDM (or a flat one) with the
    baryon density and the CMB temperature set. Massive neutrinos are passed species
    by species (see :func:`~hmf.core._boltzmann.camb_cosmology_input`). Both species
    come from one run: ``"cb"`` is CAMB's ``Transfer_nonu`` and ``"tot"`` its
    ``Transfer_tot``. The run also computes the growth factor of both species at
    k = 0.01/Mpc, for a :class:`~hmf.core.growth.Growth` stage that shares it.
    """

    backend: ClassVar[str | None] = "camb"
    references: ClassVar[tuple[str, ...]] = (refs.CAMB,)

    dark_energy_model: Literal["fluid", "ppf"] = field(
        default="fluid",
        validator=attrs.validators.in_(("fluid", "ppf")),
        doc="CAMB's dark-energy model ('ppf' can cross w = -1).",
    )
    settings: tuple[tuple[str, Any], ...] = field(
        factory=tuple,
        converter=_pairs,
        doc=(
            "Further CAMBparams settings, as a mapping from attribute path to value, e.g. "
            "{'Accuracy.AccuracyBoost': 2}. They are applied last."
        ),
    )

    def run_input(self, cosmology: FLRW, accuracy: KAccuracy) -> dict[str, Any]:
        """The CAMB input (see :meth:`_Boltzmann.run_input`)."""
        return {
            "cosmology": camb_cosmology_input(cosmology),
            "dark_energy_model": self.dark_energy_model,
            "transfer": {
                "kmax_mpc": self._k_max * float(cosmology.h),
                "k_per_logint": _camb_k_per_logint(accuracy),
                "high_precision": accuracy.dln_k < KAccuracy().dln_k,
            },
            "growth": {
                "z_max": self.z_max_growth,
                "n_z": _CAMB_GROWTH_N_Z,
                "k_ref_mpc": BOLTZMANN_GROWTH_K_REF,
            },
            "settings": dict(self.settings),
        }


#: CLASS input parameters set from the cosmology or by the model.
_CLASS_FIXED_PARAMS = frozenset(
    {
        "h", "H0", "100*theta_s", "theta_s_100", "omega_b", "Omega_b", "omega_cdm",
        "Omega_cdm", "omega_m", "Omega_m", "Omega_k", "T_cmb", "Omega_g", "omega_g",
        "N_ur", "Omega_ur", "omega_ur", "N_ncdm", "m_ncdm", "deg_ncdm", "T_ncdm",
        "Omega_ncdm", "omega_ncdm", "Omega_Lambda", "Omega_fld", "w0_fld", "wa_fld",
        "Omega_scf", "gauge", "output", "P_k_max_h/Mpc", "P_k_max_1/Mpc", "z_max_pk",
    }
)  # fmt: skip


def _class_k_per_decade(accuracy: KAccuracy) -> int | None:
    """CLASS's ``k_per_decade_for_pk`` for a wavenumber accuracy (None: CLASS's default)."""
    if accuracy.dln_k >= KAccuracy().dln_k:
        return None
    return math.ceil(0.2 * math.log(10) / accuracy.dln_k)


@attrs.frozen(kw_only=True)
class CLASS(_Boltzmann, alias="CLASS"):
    r"""The transfer function computed by CLASS.

    Needs the optional ``classy`` package (``pip install hmf[class]``). The cosmology
    is converted as described in
    :func:`~hmf.core._boltzmann.class_cosmology_input`. The transfer function is
    :math:`-\delta(k)/k^2` at z = 0 in the synchronous gauge: ``"tot"`` from CLASS's
    ``d_m``, and ``"cb"`` the density-weighted mean of ``d_cdm`` and ``d_b``. The run
    also computes the growth factor of both species at k = 0.01/Mpc (from CLASS's
    P(k, z)), for a :class:`~hmf.core.growth.Growth` stage that shares it.
    """

    backend: ClassVar[str | None] = "class"
    references: ClassVar[tuple[str, ...]] = (refs.CLASS_I, refs.CLASS_II)

    class_params: tuple[tuple[str, Any], ...] = field(
        factory=tuple,
        converter=_pairs,
        doc=(
            "Further CLASS input parameters (e.g. precision parameters, or YHe), as a "
            "mapping. Those that describe the cosmology, the gauge, the output and the "
            "k and z range are set by the model and can't be given here."
        ),
    )

    @class_params.validator
    def _check_class_params(
        self, attribute: attrs.Attribute[Any], value: tuple[tuple[str, Any], ...]
    ) -> None:
        clash = sorted({k for k, _ in value} & _CLASS_FIXED_PARAMS)
        if clash:
            raise ValueError(
                f"class_params can't set {clash}: they are set from the cosmology and the "
                "model's fields."
            )

    def run_input(self, cosmology: FLRW, accuracy: KAccuracy) -> dict[str, Any]:
        """The CLASS input (see :meth:`_Boltzmann.run_input`)."""
        params: dict[str, Any] = dict(self.class_params)
        k_per_decade = _class_k_per_decade(accuracy)
        if k_per_decade is not None and "k_per_decade_for_pk" not in params:
            params["k_per_decade_for_pk"] = k_per_decade
        params.update(class_cosmology_input(cosmology))
        params.update(
            {
                "output": "mTk,mPk",
                "P_k_max_h/Mpc": self._k_max,
                "z_max_pk": self.z_max_growth,
                "gauge": "synchronous",
            }
        )
        return {"class": params, "k_ref_mpc": BOLTZMANN_GROWTH_K_REF}
