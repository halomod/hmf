"""The comparison harness for the v3.7.2 regression reference (issue #394).

The reference (``data/reference_v3.7.2.npz`` and ``.json``) holds v3.7.2's results,
computed at high resolution, for a set of *cases*: a quantity (``"sigma"``,
``"dndm"``, ...) for one combination of cosmology, transfer model, filter, fit and
redshifts. This module

* loads it (:func:`load_reference`, :class:`Reference`, :class:`Case`);
* compares a result with it, with per-quantity tolerances read from
  ``data/tolerances.json`` (:func:`compare`);
* keeps the registry of *v4 providers*: one function per quantity that computes a
  case with ``hmf.core`` (:func:`register_provider`). ``test_regression_v4.py`` runs
  every case of every quantity that has a provider, and skips the others.

It imports only numpy at import time (astropy only when asked to convert a Quantity or
build a cosmology), so that ``generate_reference.py`` can use it from its isolated
v3.7.2 environment.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DATA = Path(__file__).parent / "data"
REFERENCE_NAME = "reference_v3.7.2"

#: The quantities in the reference, with the step-2 PR expected to provide each.
QUANTITIES: Mapping[str, str] = {
    "transfer": "Transfer function T(k), normalised to 1 at low k (step 2a)",
    "power": "Linear power spectrum P(k) at z = 0, normalised to sigma_8 (step 2a/2b)",
    "growth": "Linear growth factor D(z), D(0) = 1 (step 2a)",
    "sigma": "Mass variance sigma(M, z) (step 2b)",
    "dlnsdlnm": "dln sigma / dln M (step 2b)",
    "fsigma": "Multiplicity function f(sigma) of each fit (step 2c)",
    "dndm": "Mass function dn/dM (steps 2b + 2c)",
    "ngtm": "Cumulative mass function n(>M) (steps 2b + 2c)",
}


# ---------------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Tolerance:
    """The tolerance of one quantity.

    Attributes
    ----------
    rtol
        The relative tolerance.
    amplify
        ``None``, or how the sigma tolerance is propagated into this quantity:
        ``"fsigma_slope"`` adds ``|dln f/dln sigma| * rtol(sigma)`` point by point, and
        ``"ngtm_weighted"`` adds the dn/dln M-weighted average of that over M' > M.
    sigma_rtol
        The sigma tolerance used by ``amplify``.
    justification
        Why this tolerance.
    floor
        Reference values smaller than this in magnitude are not compared (at least
        the global floor of ``tolerances.json``).
    """

    rtol: float
    amplify: str | None = None
    sigma_rtol: float = 0.0
    justification: str = ""
    floor: float = 0.0


@dataclass(frozen=True)
class Tolerances:
    """All tolerances, as read from ``tolerances.json``."""

    quantities: Mapping[str, Tolerance]
    overrides: tuple[Mapping[str, Any], ...]
    floor: float

    def get(self, quantity: str, case: Case | None = None) -> Tolerance:
        """The tolerance of ``quantity``, with any override matching ``case``.

        An override is a dict with a ``"quantity"``, a ``"rtol"``, a
        ``"justification"``, and any of the :class:`Case` fields (``"filter"``,
        ``"transfer"``, ...) to match. The last matching override wins.
        """
        tol = self.quantities[quantity]
        if case is None:
            return tol
        for o in self.overrides:
            fields = {k: v for k, v in o.items() if k not in ("quantity", "rtol", "justification")}
            if o["quantity"] == quantity and all(getattr(case, k) == v for k, v in fields.items()):
                tol = Tolerance(
                    rtol=o["rtol"],
                    amplify=tol.amplify,
                    sigma_rtol=tol.sigma_rtol,
                    justification=o["justification"],
                    floor=tol.floor,
                )
        return tol


def load_tolerances(path: Path | None = None) -> Tolerances:
    """Read the per-quantity tolerances.

    Parameters
    ----------
    path
        The JSON file; ``data/tolerances.json`` by default.

    Returns
    -------
    Tolerances
        The tolerances.
    """
    raw = json.loads((path or DATA / "tolerances.json").read_text())
    sigma_rtol = raw["quantities"]["sigma"]["rtol"]
    floor = float(raw["floor"])
    quantities = {
        name: Tolerance(
            rtol=q["rtol"],
            amplify=q.get("amplify"),
            sigma_rtol=sigma_rtol if q.get("amplify") else 0.0,
            justification=q["justification"],
            floor=max(floor, q.get("floor", 0.0)),
        )
        for name, q in raw["quantities"].items()
    }
    return Tolerances(quantities, tuple(raw.get("overrides", ())), floor)


# ---------------------------------------------------------------------------------
# The reference
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One comparable case of the reference.

    Attributes
    ----------
    quantity
        One of :data:`QUANTITIES`.
    key
        The array's key in the npz file.
    cosmology
        The name of the cosmology (see :meth:`Reference.cosmology`), or None.
    transfer
        The v3 transfer model (``"CAMB"`` or ``"EH"``), or None.
    species
        The matter species of a CAMB transfer function (``"cb"`` or ``"tot"``), or
        None.
    filter
        The v3 filter (``"TopHat"``, ``"SharpK"`` or ``"SmoothK"``), or None.
    fit
        The v3 fitting-function name (e.g. ``"Tinker08"``), or None.
    z
        The redshifts of the first axis, if the case has a redshift axis.
    axes
        The names of the array's axes: ``"z"``, ``"log10m"`` (the masses
        :attr:`Reference.m`), ``"lnk"`` (:attr:`Reference.lnk`) or ``"z_growth"``
        (:attr:`Reference.z_growth`).
    """

    quantity: str
    key: str
    cosmology: str | None = None
    transfer: str | None = None
    species: str | None = None
    filter: str | None = None
    fit: str | None = None
    z: tuple[float, ...] | None = None
    axes: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        """A short readable id: the npz key."""
        return self.key


class Reference:
    """The v3.7.2 regression reference: its arrays and metadata.

    Parameters
    ----------
    arrays
        The arrays, by npz key.
    metadata
        The parsed JSON metadata.
    """

    def __init__(self, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]):
        self.arrays = arrays
        self.metadata = metadata

    @property
    def lnk(self) -> np.ndarray:
        """ln(k / (h/Mpc)) of the ``"lnk"`` axis."""
        return np.asarray(self.arrays["grid/lnk"])

    @property
    def m(self) -> np.ndarray:
        """The masses [Msun/h] of the ``"log10m"`` axis (exactly those v3 used)."""
        return np.asarray(self.arrays["grid/m"])

    @property
    def log10m(self) -> np.ndarray:
        """log10(m / (Msun/h)) of the ``"log10m"`` axis."""
        return np.asarray(self.arrays["grid/log10m"])

    @property
    def z_growth(self) -> np.ndarray:
        """The redshifts of the ``"z_growth"`` axis."""
        return np.asarray(self.metadata["grids"]["z_growth"])

    @property
    def settings(self) -> Mapping[str, Any]:
        """The v3 settings shared by every case (sigma_8, n, delta_c, ...)."""
        return self.metadata["settings"]

    def cases(self, quantity: str | None = None) -> Iterator[Case]:
        """The comparable cases, optionally of one quantity only."""
        for c in self.metadata["cases"]:
            if quantity is None or c["quantity"] == quantity:
                yield Case(
                    **{
                        **c,
                        "z": tuple(c["z"]) if c["z"] is not None else None,
                        "axes": tuple(c["axes"]),
                    }
                )

    def values(self, case: Case) -> np.ndarray:
        """The reference values of ``case``."""
        return np.asarray(self.arrays[case.key])

    def find(self, quantity: str, **fields: Any) -> Case:
        """The one case of ``quantity`` whose fields match ``fields``."""
        found = [
            c for c in self.cases(quantity) if all(getattr(c, k) == v for k, v in fields.items())
        ]
        if len(found) != 1:
            raise KeyError(f"{len(found)} cases of {quantity!r} match {fields}")
        return found[0]

    def cosmology(self, name: str) -> Any:
        """The astropy cosmology called ``name``, rebuilt from the metadata."""
        import astropy.cosmology
        import astropy.units as u

        meta = self.metadata["cosmologies"][name]
        params = {
            k: (v["value"] * u.Unit(v["unit"]) if isinstance(v, dict) else v)
            for k, v in meta["parameters"].items()
        }
        return getattr(astropy.cosmology, meta["class"])(**params, name=meta["name"])

    def sigma_at(self, case: Case) -> np.ndarray:
        """sigma(M) of a fit's case, at the case's redshifts (shape ``(n_z, n_m)``)."""
        sig = self.find(
            "sigma", cosmology=case.cosmology, transfer=case.transfer, filter=case.filter
        )
        sigma = self.values(sig)
        rows = []
        for z in case.z or ():
            if sig.z is not None and z in sig.z:
                rows.append(sigma[sig.z.index(z)])
            else:
                # sigma(M, z) = D(z) sigma(M, 0), with v3's D.
                growth = self.values(self.find("growth", cosmology=case.cosmology))
                d = growth[int(np.argmin(np.abs(self.z_growth - z)))]
                rows.append(sigma[sig.z.index(0.0)] * d)
        return np.array(rows)

    def context(self, case: Case) -> dict[str, np.ndarray]:
        """What :func:`compare` needs, besides the values, to compare ``case``.

        For the quantities of a fit (``dndm``, ``fsigma``, ``ngtm``): the slope
        ``dln f/dln sigma`` along the mass axis (``"fsigma_slope"``) and, for
        ``ngtm``, its dn/dln M-weighted average above each mass
        (``"ngtm_weighted_slope"``). Empty for the other quantities.
        """
        if case.fit is None:
            return {}
        fields = {
            "cosmology": case.cosmology,
            "transfer": case.transfer,
            "filter": case.filter,
            "fit": case.fit,
        }
        f = self.values(self.find("fsigma", **fields))
        sigma = self.sigma_at(case)
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.abs(np.gradient(np.log(f), axis=-1) / np.gradient(np.log(sigma), axis=-1))
        out = {"fsigma_slope": slope}
        if case.quantity == "ngtm":
            dndm = self.values(self.find("dndm", **fields))
            ngtm = self.values(case)
            out["ngtm_weighted_slope"] = _ngtm_weighted(slope, dndm * self.m, ngtm, np.log(self.m))
        return out


def _ngtm_weighted(slope, dndlnm, ngtm, lnm) -> np.ndarray:
    """The dn/dln M-weighted average of ``slope`` over M' >= M, along the last axis.

    The part above the top mass, n(>M_top), gets the slope at the top mass.
    """
    with np.errstate(invalid="ignore"):
        w = np.where(np.isfinite(slope * dndlnm), slope * dndlnm, 0.0)
        d = np.where(np.isfinite(dndlnm), dndlnm, 0.0)
        top = np.nan_to_num(slope[..., -1:] * ngtm[..., -1:])

        def above(y):
            seg = 0.5 * (y[..., 1:] + y[..., :-1]) * np.diff(lnm)
            cum = np.cumsum(seg[..., ::-1], axis=-1)[..., ::-1]
            return np.concatenate([cum, np.zeros_like(cum[..., :1])], axis=-1)

        num = above(w) + top
        den = above(d) + np.nan_to_num(ngtm[..., -1:])
        return num / den


@functools.cache
def load_reference(name: str = REFERENCE_NAME) -> Reference:
    """Load the reference from ``data/`` (cached).

    Parameters
    ----------
    name
        The base name of the npz and JSON files.

    Returns
    -------
    Reference
        The reference.
    """
    with np.load(DATA / f"{name}.npz") as f:
        arrays = {k: f[k] for k in f.files}
    metadata = json.loads((DATA / f"{name}.json").read_text())
    return Reference(arrays, metadata)


# ---------------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------------


class RegressionMismatchError(AssertionError):
    """A result differs from the reference by more than its tolerance."""


@dataclass(frozen=True)
class ComparisonResult:
    """The outcome of :func:`compare`.

    Attributes
    ----------
    quantity
        The quantity compared.
    passed
        Whether every compared point is within tolerance.
    max_rel_diff
        The largest relative difference over the compared points.
    max_ratio
        The largest relative difference divided by the point's tolerance (< 1 passes).
    worst_index
        The index of the point with the largest ratio (None if nothing was compared).
    n_compared
        The number of points compared.
    n_skipped
        The number of points not compared (reference below the floor or not finite).
    """

    quantity: str
    passed: bool
    max_rel_diff: float
    max_ratio: float
    worst_index: tuple[int, ...] | None
    n_compared: int
    n_skipped: int

    def __str__(self) -> str:
        """A one-line summary."""
        status = "ok" if self.passed else "FAIL"
        return (
            f"{self.quantity}: {status}, max |rel diff| {self.max_rel_diff:.3g}, "
            f"max diff/tolerance {self.max_ratio:.3g} at {self.worst_index} "
            f"({self.n_compared} points, {self.n_skipped} skipped)"
        )


def _to_reference_units(quantity: str, actual: Any) -> np.ndarray:
    """``actual`` as a plain array in the reference's unit of ``quantity``."""
    if not hasattr(actual, "unit"):
        return np.asarray(actual, dtype=float)
    import astropy.cosmology.units as cu
    import astropy.units as u

    unit = load_reference().metadata["units"][quantity]
    with u.add_enabled_units(cu):
        target = u.Unit(unit)
    return np.asarray(actual.to_value(target), dtype=float)


def compare(
    quantity: str,
    actual: Any,
    reference: Any,
    *,
    case: Case | None = None,
    context: Mapping[str, np.ndarray] | None = None,
    tolerances: Tolerances | None = None,
    raise_on_failure: bool = True,
) -> ComparisonResult:
    """Compare ``actual`` with ``reference`` using the tolerance of ``quantity``.

    Parameters
    ----------
    quantity
        One of :data:`QUANTITIES`; selects the tolerance.
    actual
        The result to check: an array in the reference's units, or an astropy
        Quantity (converted to them; see the ``"units"`` of the metadata).
    reference
        The reference values (same shape as ``actual``).
    case
        The case, to apply any per-case override of the tolerance.
    context
        :meth:`Reference.context` of the case. Required for the quantities whose
        tolerance propagates the sigma tolerance (``dndm``, ``fsigma``, ``ngtm``).
    tolerances
        The tolerances; ``data/tolerances.json`` by default.
    raise_on_failure
        Raise :class:`RegressionMismatchError` if any point is out of tolerance.

    Returns
    -------
    ComparisonResult
        The largest differences, and whether they pass.

    Notes
    -----
    Points where the reference is not finite, or below the quantity's floor in
    magnitude, are skipped. Everywhere else the relative difference
    ``|actual / reference - 1|`` must be at most the tolerance, which is ``rtol``, plus,
    for quantities with ``amplify``, the propagated sigma tolerance (see
    ``tolerances.json``).
    """
    tolerances = tolerances or load_tolerances()
    tol = tolerances.get(quantity, case)
    act = _to_reference_units(quantity, actual)
    ref = np.asarray(reference, dtype=float)
    if act.shape != ref.shape:
        raise ValueError(f"{quantity}: shape {act.shape} differs from the reference's {ref.shape}")

    bound = np.full(ref.shape, tol.rtol)
    if tol.amplify is not None:
        name = {"fsigma_slope": "fsigma_slope", "ngtm_weighted": "ngtm_weighted_slope"}[tol.amplify]
        if context is None or name not in context:
            raise ValueError(f"{quantity}: comparing needs context[{name!r}] (Reference.context)")
        # A non-finite slope (f underflowed) leaves the point to the floor test.
        slope = np.nan_to_num(np.asarray(context[name]), nan=np.inf, posinf=np.inf)
        bound = bound + slope * tol.sigma_rtol

    floor = max(tolerances.floor, tol.floor)
    mask = np.isfinite(ref) & (np.abs(ref) > floor) & np.isfinite(bound)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(act / ref - 1)
        rel = np.where(np.isfinite(rel), rel, np.inf)
        ratio = np.where(mask, rel / bound, 0.0)

    n = int(mask.sum())
    worst = tuple(int(i) for i in np.unravel_index(np.argmax(ratio), ratio.shape)) if n else None
    result = ComparisonResult(
        quantity=quantity,
        passed=bool(n == 0 or ratio.max() <= 1),
        max_rel_diff=float(rel[mask].max()) if n else 0.0,
        max_ratio=float(ratio.max()) if n else 0.0,
        worst_index=worst,
        n_compared=n,
        n_skipped=int(ref.size - n),
    )
    if raise_on_failure and not result.passed:
        where = f" ({case.key})" if case is not None else ""
        raise RegressionMismatchError(f"{result}{where}")
    return result


# ---------------------------------------------------------------------------------
# v4 providers
# ---------------------------------------------------------------------------------

#: A v4 provider: computes ``case`` with hmf.core and returns it with the shape of
#: ``reference.values(case)`` (a Quantity, or an array in the reference's units), or
#: ``None`` if it does not support the case (e.g. a filter v4 does not have yet).
Provider = Callable[[Case, Reference], Any]

_PROVIDERS: dict[str, Provider] = {}


def register_provider(quantity: str) -> Callable[[Provider], Provider]:
    """Register the v4 provider of ``quantity`` (a decorator).

    Parameters
    ----------
    quantity
        One of :data:`QUANTITIES`. Each quantity has at most one provider.

    Returns
    -------
    callable
        The decorator, which returns the provider unchanged.

    Examples
    --------
    In ``v4_providers.py``::

        @register_provider("growth")
        def growth(case, reference):
            cosmo = reference.cosmology(case.cosmology)
            return GrowthStage(cosmo=cosmo).growth_factor(reference.z_growth)
    """
    if quantity not in QUANTITIES:
        raise KeyError(f"unknown quantity {quantity!r}; expected one of {list(QUANTITIES)}")

    def decorator(fn: Provider) -> Provider:
        if quantity in _PROVIDERS and _PROVIDERS[quantity] is not fn:
            raise ValueError(f"a v4 provider of {quantity!r} is already registered")
        _PROVIDERS[quantity] = fn
        return fn

    return decorator


def get_provider(quantity: str) -> Provider | None:
    """The registered v4 provider of ``quantity``, or None."""
    return _PROVIDERS.get(quantity)
