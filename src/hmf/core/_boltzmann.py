"""Runs of the Boltzmann codes (CAMB and CLASS) behind the transfer and growth models.

A run is a pure function of its *input*: a mapping of plain values (numbers,
strings, lists, dicts) built by a model from the cosmology and its own fields. The
runners here build the Boltzmann code's parameters from that input alone, so the
input fully identifies the run's output. That makes it safe to reuse runs:

* in this process, through a small memo keyed by the content hash of the input, so
  that e.g. a transfer stage that only changed ``n_s``, or a growth stage that shares
  the transfer's model, does not run the code again;
* across processes, through the opt-in :class:`~hmf.core.cache.DiskCache`, whose key
  also includes the versions of hmf and of the code.

Each run computes every matter species at once (``"cb"``, CDM + baryons, and
``"tot"``, total matter including massive neutrinos), and also the growth factor of
each at a reference wavenumber, so one run serves both the transfer and the growth.

The memo holds read-only results only; it is not state that objects share and
mutate. :func:`run_counts` counts the actual runs, for tests and benchmarks.

CAMB and classy are imported here only when a run is needed.
"""

from __future__ import annotations

import importlib.metadata
import threading
from collections import Counter, OrderedDict
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

import attrs
import numpy as np
import numpy.typing as npt
from astropy import cosmology as ac
from scipy.interpolate import CubicSpline

from ._arrays import read_only
from ._serialise import content_hash
from ._species import CAMB_COLUMNS, MATTER_SPECIES
from .cache import DiskCache

__all__ = [
    "MATTER_SPECIES",
    "BoltzmannRun",
    "camb_cosmology_input",
    "check_boltzmann_cosmology",
    "class_cosmology_input",
    "clear_memo",
    "get_run",
    "run_counts",
    "run_key",
]

Array = npt.NDArray[np.float64]

#: CAMB's evolution variables for each species.
CAMB_GROWTH_VARIABLES: Mapping[str, str] = MappingProxyType(
    {"tot": "delta_tot", "cb": "delta_nonu"}
)

#: How many runs the memo keeps.
MEMO_SIZE = 32


@attrs.frozen(eq=False)
class BoltzmannRun:
    """The output of one run of a Boltzmann code, for every matter species.

    All arrays are read-only.
    """

    #: ``"camb"`` or ``"class"``.
    backend: str
    #: Wavenumbers of the transfer functions, in h/Mpc (empty if not computed).
    k: Array
    #: The transfer function of each species at z = 0 on ``k``, unnormalised.
    transfer: Mapping[str, Array]
    #: Increasing redshifts of the growth factors, from 0.
    growth_z: Array
    #: The growth factor of each species at the reference wavenumber, D(z)/D(0).
    growth: Mapping[str, Array]

    def to_arrays(self) -> dict[str, Array]:
        """The run as a flat mapping of arrays, for the disk cache."""
        out = {"k": self.k, "growth_z": self.growth_z}
        out.update({f"transfer_{s}": t for s, t in self.transfer.items()})
        out.update({f"growth_{s}": d for s, d in self.growth.items()})
        return out

    @classmethod
    def from_arrays(cls, backend: str, arrays: Mapping[str, npt.NDArray[Any]]) -> BoltzmannRun:
        """Rebuild a run from :meth:`to_arrays`."""
        return make_run(
            backend,
            k=arrays["k"],
            transfer={
                s: arrays[f"transfer_{s}"] for s in MATTER_SPECIES if f"transfer_{s}" in arrays
            },
            growth_z=arrays["growth_z"],
            growth={s: arrays[f"growth_{s}"] for s in MATTER_SPECIES if f"growth_{s}" in arrays},
        )


def make_run(
    backend: str,
    *,
    k: npt.ArrayLike,
    transfer: Mapping[str, npt.ArrayLike],
    growth_z: npt.ArrayLike,
    growth: Mapping[str, npt.ArrayLike],
) -> BoltzmannRun:
    """Build a :class:`BoltzmannRun` with read-only copies of the arrays."""
    return BoltzmannRun(
        backend=backend,
        k=read_only(k),
        transfer=MappingProxyType({s: read_only(t) for s, t in transfer.items()}),
        growth_z=read_only(growth_z),
        growth=MappingProxyType({s: read_only(d) for s, d in growth.items()}),
    )


# ---------------------------------------------------------------------------------
# Cosmology -> input
# ---------------------------------------------------------------------------------


def check_boltzmann_cosmology(cosmo: ac.FLRW, name: str) -> None:
    """Raise if a Boltzmann code can't compute ``cosmo``.

    Parameters
    ----------
    cosmo
        The cosmology.
    name
        The model's name, for the error messages.

    Raises
    ------
    ValueError
        If ``cosmo`` is not a LambdaCDM, wCDM or w0waCDM (or a flat variant), or does
        not set the baryon density or the CMB temperature: the code does not apply to
        it.
    """
    if not isinstance(cosmo, (ac.LambdaCDM, ac.wCDM, ac.w0waCDM)):
        raise ValueError(  # noqa: TRY004 (a configuration error: see hmf.core.domain)
            f"{name} needs a LambdaCDM, wCDM or w0waCDM cosmology (or a flat one), not "
            f"{type(cosmo).__name__}."
        )
    if not cosmo.Ob0:
        raise ValueError(f"To use {name}, set the baryon density (Ob0) of the cosmology.")
    if cosmo.Tcmb0.value == 0:
        raise ValueError(f"To use {name}, set the CMB temperature (Tcmb0) of the cosmology.")


def _dark_energy(cosmo: ac.FLRW) -> tuple[float, float]:
    """(w0, wa) of a LambdaCDM, wCDM or w0waCDM cosmology."""
    return float(getattr(cosmo, "w0", -1.0)), float(getattr(cosmo, "wa", 0.0))


def camb_cosmology_input(cosmo: ac.FLRW) -> dict[str, Any]:
    """The CAMB input describing an astropy cosmology, as plain values.

    The CDM density is ``Om0 - Ob0`` (astropy's ``Om0`` excludes massive neutrinos,
    whose density CAMB computes from their masses). Each distinct non-zero mass in
    ``cosmo.m_nu`` becomes a CAMB mass eigenstate, with as many degenerate species
    as share it, so ``m_nu = [0.1, 0.1, 0.1]`` eV is three species of 0.1 eV, not one
    of 0.3 eV.

    Parameters
    ----------
    cosmo
        A cosmology that passes :func:`check_boltzmann_cosmology`.

    Returns
    -------
    dict
    """
    h = cosmo.h
    m_nu = np.zeros(0) if cosmo.m_nu is None else np.atleast_1d(cosmo.m_nu.to_value("eV"))
    w0, wa = _dark_energy(cosmo)
    return {
        "H0": float(cosmo.H0.to_value("km / (s Mpc)")),
        "ombh2": float(cosmo.Ob0 * h**2),
        "omch2": float((cosmo.Om0 - cosmo.Ob0) * h**2),
        "omk": float(cosmo.Ok0),
        "nnu": float(cosmo.Neff),
        "TCMB": float(cosmo.Tcmb0.to_value("K")),
        "m_nu": sorted(float(m) for m in m_nu if m > 0),
        "w": w0,
        "wa": wa,
    }


def class_cosmology_input(cosmo: ac.FLRW) -> dict[str, Any]:
    r"""The CLASS input describing an astropy cosmology.

    Neutrinos are described as astropy describes them: each of the
    :math:`\lfloor N_{\rm eff} \rfloor` species has the standard temperature and
    contributes :math:`N_{\rm eff} / \lfloor N_{\rm eff} \rfloor` to
    :math:`N_{\rm eff}`. The massive ones (non-zero ``m_nu``) are CLASS's ``ncdm``
    species, with that contribution as their degeneracy; the rest are ``N_ur``. The
    CDM density is ``Om0 - Ob0``. Dark energy other than Lambda is a fluid.

    Parameters
    ----------
    cosmo
        A cosmology that passes :func:`check_boltzmann_cosmology`.

    Returns
    -------
    dict
        CLASS input parameters.
    """
    h = cosmo.h
    params: dict[str, Any] = {
        "h": float(h),
        "omega_b": float(cosmo.Ob0 * h**2),
        "omega_cdm": float((cosmo.Om0 - cosmo.Ob0) * h**2),
        "Omega_k": float(cosmo.Ok0),
        "T_cmb": float(cosmo.Tcmb0.to_value("K")),
    }
    n_nu = int(np.floor(cosmo.Neff))
    neff_per_nu = cosmo.Neff / n_nu if n_nu else 0.0
    m_nu = np.zeros(0) if cosmo.m_nu is None else np.atleast_1d(cosmo.m_nu.to_value("eV"))
    m_nu = m_nu[m_nu > 0]
    params["N_ur"] = float(cosmo.Neff - len(m_nu) * neff_per_nu)
    if len(m_nu):
        params["N_ncdm"] = len(m_nu)
        params["m_ncdm"] = ",".join(repr(float(m)) for m in m_nu)
        params["deg_ncdm"] = ",".join([repr(float(neff_per_nu))] * len(m_nu))
        params["T_ncdm"] = ",".join([repr((4 / 11) ** (1 / 3))] * len(m_nu))
    w0, wa = _dark_energy(cosmo)
    if isinstance(cosmo, (ac.wCDM, ac.w0waCDM)):
        params["Omega_Lambda"] = 0.0
        params["w0_fld"] = w0
        params["wa_fld"] = wa
    return params


# ---------------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------------


def _import_camb() -> Any:
    try:
        import camb
    except ImportError as e:  # pragma: no cover - camb is a dependency for now
        raise ImportError("This model needs CAMB: `pip install camb`.") from e
    return camb


def _import_classy() -> Any:
    try:
        import classy
    except ImportError as e:
        raise ImportError(
            "This model needs CLASS: `pip install hmf[class]` (or `pip install classy`)."
        ) from e
    return classy


def _set_attribute(obj: Any, dotted: str, value: Any) -> None:
    """``setattr`` through a dotted path, e.g. ``"Accuracy.AccuracyBoost"``."""
    *parents, name = dotted.split(".")
    for p in parents:
        obj = getattr(obj, p)
    if not hasattr(obj, name):
        raise AttributeError(f"CAMBparams has no setting {dotted!r}.")
    setattr(obj, name, value)


def _camb_params(inputs: Mapping[str, Any]) -> Any:
    """Build a ``camb.CAMBparams`` from a CAMB run input."""
    camb = _import_camb()
    cosmo = inputs["cosmology"]
    p = camb.CAMBparams(
        DoLensing=False,
        Want_CMB=False,
        Want_CMB_lensing=False,
        WantCls=False,
        WantDerivedParameters=False,
    )
    m_nu = np.asarray(cosmo["m_nu"], dtype=float)
    masses, counts = np.unique(m_nu, return_counts=True)
    num_massive = int(counts.sum())
    p.set_cosmology(
        H0=cosmo["H0"],
        ombh2=cosmo["ombh2"],
        omch2=cosmo["omch2"],
        mnu=float(m_nu.sum()),
        neutrino_hierarchy="degenerate",
        num_massive_neutrinos=num_massive,
        omk=cosmo["omk"],
        nnu=cosmo["nnu"],
        standard_neutrino_neff=cosmo["nnu"],
        TCMB=cosmo["TCMB"],
    )
    if len(masses) > 1:
        # CAMB put all massive species in one eigenstate: split them, one eigenstate
        # per distinct mass, with the same degeneracy per species (so the same Neff).
        degeneracy = p.nu_mass_degeneracies[0] / num_massive
        p.nu_mass_eigenstates = len(masses)
        p.nu_mass_numbers = [int(c) for c in counts]
        p.nu_mass_degeneracies = [c * degeneracy for c in counts]
        p.nu_mass_fractions = [c * m / m_nu.sum() for c, m in zip(counts, masses, strict=True)]
    p.set_dark_energy(w=cosmo["w"], wa=cosmo["wa"], dark_energy_model=inputs["dark_energy_model"])

    transfer = inputs["transfer"]
    p.WantTransfer = True
    p.Transfer.kmax = transfer["kmax_mpc"]
    p.Transfer.k_per_logint = transfer["k_per_logint"]
    p.Transfer.high_precision = transfer["high_precision"]
    p.Transfer.PK_redshifts = [0.0]
    p.Transfer.PK_num_redshifts = 1
    for name, value in inputs["settings"].items():
        _set_attribute(p, name, value)
    return p


def _growth_redshifts(z_max: float, n: int) -> Array:
    """Redshifts from 0 to ``z_max``, evenly spaced in ln a."""
    return np.expm1(np.linspace(0.0, np.log1p(z_max), n))


def run_camb(inputs: Mapping[str, Any]) -> BoltzmannRun:
    """Run CAMB on a run input (see :class:`~hmf.core.transfer_models.CAMB`)."""
    camb = _import_camb()
    p = _camb_params(inputs)
    results = camb.get_transfer_functions(p)
    data = results.get_matter_transfer_data().transfer_data
    k = data[camb.model.Transfer_kh - 1, :, 0]
    transfer = {s: data[row, :, 0] for s, row in CAMB_COLUMNS.items()}

    growth = inputs["growth"]
    z = _growth_redshifts(growth["z_max"], growth["n_z"])
    names = [CAMB_GROWTH_VARIABLES[s] for s in MATTER_SPECIES]
    evolution = np.asarray(results.get_redshift_evolution(growth["k_ref_mpc"], z, names))
    evolution = evolution.reshape(len(z), len(names))
    d = {s: evolution[:, i] / evolution[0, i] for i, s in enumerate(MATTER_SPECIES)}
    return make_run("camb", k=k, transfer=transfer, growth_z=z, growth=d)


def run_class(inputs: Mapping[str, Any]) -> BoltzmannRun:
    """Run CLASS on a run input (see :class:`~hmf.core.transfer_models.CLASS`)."""
    classy = _import_classy()
    params = dict(inputs["class"])
    outputs = [o.strip() for o in str(params["output"]).split(",")]
    cl = classy.Class()
    cl.set(params)
    try:
        cl.compute()
        k: Array = np.zeros(0)
        transfer: dict[str, Array] = {}
        if "mTk" in outputs:
            tk = cl.get_transfer(z=0.0)
            omega_cdm, omega_b = cl.Omega0_cdm(), cl.Omega_b()
            k = np.asarray(tk["k (h/Mpc)"])
            # delta_cdm and delta_b are in the synchronous gauge; their density-weighted
            # mean is the CDM + baryon field. T = -delta / k^2 at z = 0.
            delta_cb = (omega_cdm * tk["d_cdm"] + omega_b * tk["d_b"]) / (omega_cdm + omega_b)
            transfer = {"tot": -tk["d_m"] / k**2, "cb": -delta_cb / k**2}
        ln_k_ref = np.log(inputs["k_ref_mpc"])
        growth: dict[str, Array] = {}
        z: Array = np.zeros(0)
        for s in MATTER_SPECIES:
            pk, kk, zz = cl.get_pk_and_k_and_z(nonlinear=False, only_clustering_species=s == "cb")
            order = np.argsort(zz)
            z = np.asarray(zz)[order]
            # P(k, z) is smooth in ln k around k_ref: a cubic spline in ln k.
            ln_pk = CubicSpline(np.log(kk), np.log(pk[:, order]), axis=0)(ln_k_ref)
            growth[s] = np.exp(0.5 * (ln_pk - ln_pk[0]))
    finally:
        cl.struct_cleanup()
        cl.empty()
    return make_run("class", k=k, transfer=transfer, growth_z=z, growth=growth)


_RUNNERS: Mapping[str, Callable[[Mapping[str, Any]], BoltzmannRun]] = MappingProxyType(
    {"camb": run_camb, "class": run_class}
)

_MODULES: Mapping[str, str] = MappingProxyType({"camb": "camb", "class": "classy"})


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def run_key(backend: str, inputs: Mapping[str, Any]) -> str:
    """The content hash identifying a run: of its input and the code versions.

    Parameters
    ----------
    backend
        ``"camb"`` or ``"class"``.
    inputs
        The run input.

    Returns
    -------
    str
        A SHA-256 hex digest.
    """
    module = _MODULES[backend]
    if backend == "class":
        _import_classy()  # a clear error if it's missing
    return content_hash(
        {
            "hmf": _version("hmf"),
            "backend": backend,
            "backend_version": _version(module),
            "input": inputs,
        }
    )


_memo: OrderedDict[str, BoltzmannRun] = OrderedDict()
_counts: Counter[str] = Counter()
_lock = threading.Lock()


def run_counts() -> Mapping[str, int]:
    """How many times each Boltzmann code has actually run in this process."""
    with _lock:
        return MappingProxyType(dict(_counts))


def clear_memo() -> None:
    """Forget the runs memoised in this process (not the disk cache)."""
    with _lock:
        _memo.clear()


def get_run(
    backend: str, inputs: Mapping[str, Any], *, disk_cache: DiskCache | None = None
) -> BoltzmannRun:
    """Return the output of a run, running the code only if no cache has it.

    The in-process memo is tried first, then the disk cache (if given); if neither
    has the run, the code runs and the result is stored in both.

    Parameters
    ----------
    backend
        ``"camb"`` or ``"class"``.
    inputs
        The run input.
    disk_cache
        The disk cache, or ``None`` for none.

    Returns
    -------
    BoltzmannRun
    """
    if backend not in _RUNNERS:
        raise ValueError(f"Unknown Boltzmann code {backend!r}; use one of {tuple(_RUNNERS)}.")
    key = run_key(backend, inputs)
    with _lock:
        if key in _memo:
            _memo.move_to_end(key)
            return _memo[key]

    run = None
    if disk_cache is not None:
        arrays = disk_cache.load(key)
        if arrays is not None:
            try:
                run = BoltzmannRun.from_arrays(backend, arrays)
            except KeyError:
                run = None
    if run is None:
        run = _RUNNERS[backend](inputs)
        with _lock:
            _counts[backend] += 1
        if disk_cache is not None:
            disk_cache.store(key, run.to_arrays())

    with _lock:
        _memo[key] = run
        _memo.move_to_end(key)
        while len(_memo) > MEMO_SIZE:
            _memo.popitem(last=False)
    return run
