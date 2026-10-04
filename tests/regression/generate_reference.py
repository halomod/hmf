# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     "hmf==3.7.2",
#     "camb==1.6.6",
#     "numpy==2.4.3",
#     "scipy==1.17.1",
#     "astropy==7.2.0",
# ]
# ///
"""Generate the v3.7.2 regression reference for hmf v4 (issue #394).

The reference is computed with **hmf 3.7.2 exactly**, at high resolution, in an
isolated environment whose numerical dependencies are pinned to the versions in
``uv.lock`` (so that CI, which installs ``uv.lock``, runs the same CAMB). Run it from
the repository root with::

    uv run --no-project --script tests/regression/generate_reference.py

``uv`` builds the environment from the inline metadata above, so the working tree's
own ``hmf`` is never imported (the script checks this). It writes, into
``tests/regression/data/``:

``reference_v3.7.2.npz``
    The reference arrays (compressed), keyed ``quantity/cosmology/transfer/...``.
``reference_v3.7.2.json``
    The metadata: package versions, settings, grids, cosmologies, the list of cases,
    and the convergence of every quantity (see below).

The per-quantity tolerances are read from ``tolerances.json`` in the same directory,
which is written by hand, not by this script.

Resolution and convergence
--------------------------
Everything is computed with ``dlnk = 0.005`` and ``dlog10m = 0.005`` on
``-18 <= ln k <= 12`` and ``6 <= log10 m <= 16``, then stored on a coarser output grid
(every 4th wavenumber, i.e. 0.02 in ln k, and every 10th mass, i.e. 0.05 dex).

n(>M) is the exception: v3 integrates it with a cumulative trapezoid, which is not
converged in the steep high-mass tail at this step (up to 1%), so it is computed again
with dlog10m halved and Richardson-extrapolated to dlog10m -> 0 (``richardson_ngtm``).

The whole calculation is then repeated (a) with both steps halved, and (b) on the
wider range ``-20 <= ln k <= 14``, and the largest change of each quantity, in units
of its tolerance, is recorded under ``"convergence"`` in the metadata. A test checks
that it is below 1, which is what justifies the tolerances.

``--no-convergence`` skips those runs (for development only: the metadata then has
no convergence record, and the tests fail).
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import astropy
import astropy.units as u
import numpy as np
import scipy
from astropy.cosmology import FlatLambdaCDM, Flatw0waCDM, FlatwCDM, Planck18

import hmf
from hmf import MassFunction
from hmf.density_field.transfer_models import TransferComponent
from hmf.mass_function.fitting_functions import BaseFittingFunction

HERE = Path(__file__).parent
DATA = HERE / "data"
HMF_VERSION = "3.7.2"
NAME = f"reference_v{HMF_VERSION}"

# ---------------------------------------------------------------------------------
# What the reference covers
# ---------------------------------------------------------------------------------

#: Resolution of the calculation: (dlnk, dlog10m), and the ln k range.
DLNK = 0.005
DLOG10M = 0.005
LNK_RANGE = (-18.0, 12.0)
LNK_RANGE_WIDE = (-20.0, 14.0)
LOG10M_RANGE = (6.0, 16.0)

#: Output grid: every K_STRIDE-th wavenumber (0.02 in ln k) and M_STRIDE-th mass
#: (0.05 dex) of the base-resolution grid.
K_STRIDE = 4
M_STRIDE = 10

#: Redshifts of every case, except fits calibrated only at high z.
Z = (0.0, 0.5, 1.0, 2.0, 4.0)
#: Redshifts for fits whose calibration domain excludes Z.
Z_FIT = {"Yung24": (6.0, 8.0, 10.0)}
#: The growth factor is stored on this grid (it contains every z above).
Z_GROWTH = tuple(float(z) for z in np.round(np.arange(0, 10.0001, 0.1), 10))

_BASE = {
    "H0": Planck18.H0,
    "Om0": Planck18.Om0,
    "Ob0": Planck18.Ob0,
    "Tcmb0": Planck18.Tcmb0,
    "Neff": Planck18.Neff,
}

#: The cosmologies. ``planck18`` is astropy's (and hmf's default): flat LCDM with one
#: massive neutrino of 0.06 eV.
COSMOLOGIES = {
    "planck18": Planck18,
    "lcdm_massless_nu": FlatLambdaCDM(**_BASE, m_nu=0 * u.eV, name="lcdm_massless_nu"),
    "lcdm_mnu0p3": FlatLambdaCDM(**_BASE, m_nu=[0.1, 0.1, 0.1] * u.eV, name="lcdm_mnu0p3"),
    "wcdm": FlatwCDM(**_BASE, m_nu=Planck18.m_nu, w0=-0.9, name="wcdm"),
    "w0wacdm": Flatw0waCDM(**_BASE, m_nu=Planck18.m_nu, w0=-0.9, wa=0.2, name="w0wacdm"),
}

TRANSFERS = ("CAMB", "EH")
FILTERS = ("TopHat", "SharpK", "SmoothK")
#: The transfer parameters of each transfer model (v3.7.2's defaults, made explicit).
TRANSFER_PARAMS: dict[str, dict[str, Any]] = {"CAMB": {"extrapolate_with_eh": True}, "EH": {}}
#: Matter species stored for each transfer model (EH does not distinguish them).
SPECIES = {"CAMB": ("cb", "tot"), "EH": (None,)}

#: Every built-in fitting function of hmf 3.7.2.
FITS = tuple(
    sorted(
        name
        for name, cls in BaseFittingFunction.get_models().items()
        if cls.__module__.startswith("hmf.")
    )
)
#: Fits evaluated for SharpK and SmoothK (for TopHat, every fit is).
FITS_K_FILTERS = ("PS", "SMT", "Tinker08")


def wanted_fits(cosmology: str, transfer: str, filt: str) -> tuple[str, ...]:
    """The fits whose mass function is stored for one (cosmology, transfer, filter)."""
    if filt == "TopHat" and (transfer == "CAMB" or cosmology == "planck18"):
        return FITS
    if transfer == "CAMB" and cosmology == "planck18":
        return FITS_K_FILTERS
    return ()


#: Settings shared by every MassFunction, besides the resolution.
COMMON: dict[str, Any] = {
    "sigma_8": 0.8102,
    "n": 0.9665,
    "delta_c": 1.686,
    "disable_mass_conversion": True,
    # v3.7.2's default growth model, GrowthFactor, switches between Eisenstein97 (no
    # radiation) and the ODE solution (with radiation) at a radiation fraction of 5e-4,
    # but normalises both by the Eisenstein97 D(0): D(z) jumps by ~1e-4 at the switch,
    # and an array of z gives a different D than each z alone. The ODE solution alone
    # is smooth, includes radiation and dark-energy evolution, and agrees with the
    # exact radiation-free flat LCDM growth to 1e-6 (test_reference_physics.py).
    "growth_model": "ODEGrowthFactor",
}

#: Normalised fits whose f(sigma) is stored on a wide sigma grid, to check that they
#: integrate to 1. The ln(1/sigma) grid and redshifts:
LN_INV_SIGMA = (-40.0, 4.0, 0.005)
Z_FSIGMA = (0.0, 1.0)

#: The power-law Press-Schechter case: P(k) = A k^n with unit transfer function. n = -2
#: keeps v3's fixed-step integrals accurate at both ends of the k range: for n = -1 the
#: oscillating TopHat tail of the dln sigma/dln M integrand decays only as 1/(kR), and
#: once kR > pi/dlnk Simpson's rule aliases it into ~4e-4 noise; for n = -2.5 the part
#: of sigma below k_min is ~1e-3.
POWERLAW_N = -2.0


class PowerLawUnitTransfer(TransferComponent):
    """T(k) = 1, so that P(k) is a pure power law k^n (for the analytic PS check)."""

    def lnt(self, lnk):
        """Return ln T = 0."""
        return np.zeros_like(lnk)


# ---------------------------------------------------------------------------------
# Calculation
# ---------------------------------------------------------------------------------


def key(quantity: str, *parts: str | None) -> str:
    """The npz key of a case: parts joined by '/', with '-' for an absent part."""
    return "/".join([quantity, *(p if p is not None else "-" for p in parts)])


def mass_function(cosmology, transfer, filt, dlnk, dlog10m, lnk_range) -> MassFunction:
    """A v3.7.2 MassFunction at the given resolution."""
    return MassFunction(
        cosmo_model=cosmology,
        transfer_model=transfer,
        transfer_params=dict(TRANSFER_PARAMS.get(transfer, {})),
        filter_model=filt,
        hmf_model="PS",
        z=0.0,
        Mmin=LOG10M_RANGE[0],
        # Half a step above the top, so that the top mass is on the grid.
        Mmax=LOG10M_RANGE[1] + dlog10m / 2,
        dlog10m=dlog10m,
        lnk_min=lnk_range[0],
        lnk_max=lnk_range[1] + dlnk / 2,
        dlnk=dlnk,
        **COMMON,
    )


def strides(dlnk: float, dlog10m: float, lnk_range) -> tuple[slice, slice]:
    """The slices that pick the output grid out of a calculation's k and m grids."""
    k_stride = round(K_STRIDE * DLNK / dlnk)
    m_stride = round(M_STRIDE * DLOG10M / dlog10m)
    k_start = round((LNK_RANGE[0] - lnk_range[0]) / dlnk)
    n_k = round((LNK_RANGE[1] - LNK_RANGE[0]) / (K_STRIDE * DLNK)) + 1
    n_m = round((LOG10M_RANGE[1] - LOG10M_RANGE[0]) / (M_STRIDE * DLOG10M)) + 1
    return (
        slice(k_start, k_start + k_stride * (n_k - 1) + 1, k_stride),
        slice(0, m_stride * (n_m - 1) + 1, m_stride),
    )


def camb_lnk_valid_min(mf: MassFunction) -> float:
    """The lowest ln k at which v3's CAMB transfer function is CAMB's, not an artefact.

    Below the lowest k of CAMB's table (about 1e-4 h/Mpc), v3.7.2 moves that table's
    first point down to ``lnk_min`` and joins it to the next with a cubic spline
    (``_BoltzmannTransfer._check_low_k``, after dropping any low-k turn-up), so T(k)
    there depends on ``lnk_min`` (by up to 3% between lnk_min = -18 and -20) and dips
    below 1. That region contributes nothing to sigma (below 1e-9), so it is masked
    (NaN) in the reference: this returns the second knot, above which the spline is
    CAMB's own.
    """
    transfers = mf.transfer._transfers()
    lnk = np.log(transfers["kh"])
    lowest = -np.inf
    for species in SPECIES["CAMB"]:
        lnt = np.log(transfers[species])
        start = 0
        for i in range(len(lnk) - 1):
            if abs((lnt[i + 1] - lnt[i]) / (lnk[i + 1] - lnk[i])) < 0.0001:
                start = i
                break
        lowest = max(lowest, float(lnk[start + 1]))
    return lowest


def compute(
    dlnk: float, dlog10m: float, lnk_range=LNK_RANGE, log=print
) -> tuple[dict[str, np.ndarray], list[dict], dict[str, Any]]:
    """Compute every case at one resolution, on the output grid.

    Returns
    -------
    arrays
        The reference arrays, by npz key.
    cases
        One dict per comparable case (the keys of ``arrays`` that v4 is compared on).
    info
        Per-cosmology information (mean density, growth model) and any failures.
    """
    ks, ms = strides(dlnk, dlog10m, lnk_range)
    arrays: dict[str, np.ndarray] = {}
    cases: list[dict] = []
    info: dict[str, Any] = {"cosmologies": {}, "failures": [], "camb_lnk_valid_min": {}}

    def add(
        quantity,
        value,
        *,
        axes,
        cosmology=None,
        transfer=None,
        species=None,
        filt=None,
        fit=None,
        z=None,
    ):
        k = key(quantity, cosmology, transfer, species or filt, fit)
        arrays[k] = np.asarray(value, dtype=float)
        cases.append(
            {
                "quantity": quantity,
                "key": k,
                "cosmology": cosmology,
                "transfer": transfer,
                "species": species,
                "filter": filt,
                "fit": fit,
                "z": list(z) if z is not None else None,
                "axes": list(axes),
            }
        )

    grids_done = False
    for cname, cosmo in COSMOLOGIES.items():
        for transfer in TRANSFERS:
            for filt in FILTERS:
                t0 = time.time()
                mf = mass_function(cosmo, transfer, filt, dlnk, dlog10m, lnk_range)
                if not grids_done:
                    arrays["grid/lnk"] = np.log(mf.k[ks])
                    arrays["grid/log10m"] = np.log10(mf.m[ms])
                    arrays["grid/m"] = mf.m[ms]
                    grids_done = True
                else:
                    # Every case is on the same grid.
                    assert np.allclose(np.log(mf.k[ks]), arrays["grid/lnk"], rtol=0, atol=1e-9)
                    assert np.allclose(mf.m[ms], arrays["grid/m"], rtol=1e-12, atol=0)

                if filt == "TopHat":
                    # Filter-independent quantities, once per (cosmology, transfer).
                    low_k = np.zeros(len(arrays["grid/lnk"]), dtype=bool)
                    if transfer == "CAMB":
                        lnk_valid = camb_lnk_valid_min(mf)
                        low_k = arrays["grid/lnk"] < lnk_valid
                        info["camb_lnk_valid_min"][cname] = lnk_valid
                    for species in SPECIES[transfer]:
                        if species is None:
                            lnt = mf._unnormalised_lnT
                        else:
                            tr = mf.transfer_model(
                                mf.cosmo, **{**mf.transfer.params, "matter_species": species}
                            )
                            mf.transfer._share_results(tr)
                            lnt = tr.lnt(np.log(mf.k))
                        add(
                            "transfer",
                            np.where(low_k, np.nan, np.exp(lnt[ks])),
                            axes=["lnk"],
                            cosmology=cname,
                            transfer=transfer,
                            species=species,
                        )
                    add(
                        "power",
                        np.where(low_k, np.nan, mf._power0[ks]),
                        axes=["lnk"],
                        cosmology=cname,
                        transfer=transfer,
                    )
                    if transfer == "CAMB":
                        growth = mf.growth.growth_factor(np.array(Z_GROWTH))
                        add("growth", growth, axes=["z_growth"], cosmology=cname)
                        for z in sorted({*Z, *(z for zs in Z_FIT.values() for z in zs)}):
                            mf.update(z=z)
                            # The growth stored is the one sigma(z) uses.
                            i = Z_GROWTH.index(z)
                            assert abs(mf.growth_factor / growth[i] - 1) < 1e-12
                        mf.update(z=0.0)
                        info["cosmologies"][cname] = {
                            "mean_density0": float(mf.mean_density0),
                            "growth_model": mf.growth_model.__name__,
                        }

                sigma = []
                for z in Z:
                    mf.update(z=z)
                    sigma.append(mf.sigma[ms])
                add(
                    "sigma",
                    sigma,
                    axes=["z", "log10m"],
                    cosmology=cname,
                    transfer=transfer,
                    filt=filt,
                    z=Z,
                )
                add(
                    "dlnsdlnm",
                    mf._dlnsdlnm[ms],
                    axes=["log10m"],
                    cosmology=cname,
                    transfer=transfer,
                    filt=filt,
                )

                for fit in wanted_fits(cname, transfer, filt):
                    zs = Z_FIT.get(fit, Z)
                    shape = (len(zs), len(arrays["grid/m"]))
                    out = {q: np.full(shape, np.nan) for q in ("dndm", "ngtm", "fsigma")}
                    for i, z in enumerate(zs):
                        try:
                            mf.update(hmf_model=fit, z=z)
                            for q, v in out.items():
                                v[i] = getattr(mf, q)[ms]
                        except Exception as e:  # noqa: BLE001
                            info["failures"].append(
                                {
                                    "case": key("dndm", cname, transfer, filt, fit),
                                    "z": z,
                                    "error": f"{type(e).__name__}: {e}",
                                }
                            )
                    for q, v in out.items():
                        add(
                            q,
                            v,
                            axes=["z", "log10m"],
                            cosmology=cname,
                            transfer=transfer,
                            filt=filt,
                            fit=fit,
                            z=zs,
                        )
                    mf.update(hmf_model="PS", z=0.0)
                log(f"  {cname:18s} {transfer:5s} {filt:8s} {time.time() - t0:6.1f} s")
    return arrays, cases, info


def richardson_ngtm(coarse: dict[str, np.ndarray], fine: dict[str, np.ndarray]) -> dict:
    """``coarse`` with every n(>M) replaced by its Richardson extrapolation to dlog10m -> 0.

    v3.7.2 integrates dn/dln M with a cumulative trapezoid, whose error is a series in
    even powers of the step, (s h)^2 / 12 to leading order for an integrand falling as
    exp(-s ln M). In the steep high-mass tail s reaches ~30, which makes it ~1e-2 at
    dlog10m = 0.005. ``fine`` is the same calculation with dlog10m halved, so
    (4 fine - coarse) / 3 cancels the h^2 term.
    """
    out = dict(coarse)
    for k, v in coarse.items():
        if k.startswith("ngtm/") or k == "analytic/powerlaw_ps/ngtm":
            out[k] = (4 * fine[k] - v) / 3
    return out


#: n(>M) is masked where v3's power-law tail above the top mass is more than this
#: fraction of it (see mask_extrapolated_ngtm).
NGTM_TAIL_FRACTION = 1e-5


def mask_extrapolated_ngtm(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Mask (NaN) n(>M) of Behroozi where v3's extrapolated tail matters.

    For every other fit v3 computes dn/dM up to 1e18 Msun/h before integrating. For
    Behroozi it cannot (the fit's correction needs the cumulative function itself),
    so ``hmf_integral_gtm`` adds the integral above the top mass of a power law
    through the last two grid points. That tail is an extrapolation, whose slope
    changes with the step at first order, so it is neither physical nor removed by
    the Richardson extrapolation. It equals n(>M_top), so the fraction of n(>M) it
    makes up is n(>M_top) / n(>M).
    """
    out = dict(arrays)
    for k, v in arrays.items():
        if k.startswith("ngtm/") and k.endswith("/Behroozi"):
            with np.errstate(divide="ignore", invalid="ignore"):
                tail = v[:, -1:] / v
            out[k] = np.where(tail > NGTM_TAIL_FRACTION, np.nan, v)
    return out


def analytic_cases(dlog10m: float = DLOG10M, log=print) -> dict[str, np.ndarray]:
    """The cases with a closed-form answer, computed with v3.7.2."""
    arrays: dict[str, np.ndarray] = {}
    _, ms = strides(DLNK, dlog10m, LNK_RANGE)

    # Power-law Press-Schechter: P(k) = A k^n, TopHat filter, z = 0.
    mf = mass_function(
        COSMOLOGIES["lcdm_massless_nu"], PowerLawUnitTransfer, "TopHat", DLNK, dlog10m, LNK_RANGE
    )
    mf.update(n=POWERLAW_N)
    arrays["analytic/powerlaw_ps/m"] = mf.m[ms]
    arrays["analytic/powerlaw_ps/sigma"] = mf.sigma[ms]
    arrays["analytic/powerlaw_ps/dlnsdlnm"] = mf._dlnsdlnm[ms]
    arrays["analytic/powerlaw_ps/dndm"] = mf.dndm[ms]
    arrays["analytic/powerlaw_ps/ngtm"] = mf.ngtm[ms]
    arrays["analytic/powerlaw_ps/radii"] = mf.radii[ms]
    arrays["analytic/powerlaw_ps/mean_density0"] = np.array(mf.mean_density0)
    if dlog10m != DLOG10M:
        return arrays

    # Einstein-de Sitter growth, with v3's default growth model.
    # Einstein-de Sitter, and radiation-free flat LCDM, growth with the reference's
    # growth model.
    for name, cosmo in {
        "eds": FlatLambdaCDM(H0=70, Om0=1.0, Tcmb0=0, name="eds"),
        "lcdm_no_radiation": FlatLambdaCDM(
            H0=Planck18.H0, Om0=Planck18.Om0, Ob0=Planck18.Ob0, Tcmb0=0, name="lcdm_no_rad"
        ),
    }.items():
        mf = MassFunction(
            cosmo_model=cosmo, transfer_model="EH", growth_model=COMMON["growth_model"]
        )
        arrays[f"analytic/{name}/growth"] = mf.growth.growth_factor(np.array(Z_GROWTH))
        arrays[f"analytic/{name}/Om0"] = np.array(cosmo.Om0)
    arrays["analytic/z_growth"] = np.array(Z_GROWTH)

    # f(sigma) of the normalised fits on a wide sigma grid.
    ln_inv_sigma = np.arange(*LN_INV_SIGMA)
    nu2 = (COMMON["delta_c"] * np.exp(ln_inv_sigma)) ** 2
    arrays["analytic/fsigma_wide/ln_inv_sigma"] = ln_inv_sigma
    mf = MassFunction(cosmo_model=Planck18, transfer_model="EH", **COMMON)
    for fit in FITS:
        if not BaseFittingFunction.get_models()[fit].normalized:
            continue
        for z in Z_FSIGMA:
            mf.update(hmf_model=fit, z=z)
            f = mf.hmf_model(
                m=np.full_like(nu2, 1e12),
                nu2=nu2,
                z=z,
                mass_definition=mf.mdef,
                cosmo=mf.cosmo,
                delta_c=mf.delta_c,
                n_eff=np.full_like(nu2, -2.0),
                **mf.hmf_params,
            ).fsigma
            arrays[f"analytic/fsigma_wide/{fit}/z={z:g}"] = f
    log("  analytic cases done")
    return arrays


# ---------------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------------


def _load_harness():
    """Import the comparison harness (numpy-only) from this directory."""
    sys.path.insert(0, str(HERE))
    import regression_harness

    return regression_harness


def convergence(
    base: dict[str, np.ndarray], other: dict[str, np.ndarray], cases: list[dict], harness
) -> dict[str, Any]:
    """The largest change of each quantity between two runs, in units of its tolerance."""
    tolerances = harness.load_tolerances()
    reference = harness.Reference(base, {"cases": cases, "grids": {"z_growth": list(Z_GROWTH)}})
    per_quantity: dict[str, dict[str, Any]] = {}
    for case in reference.cases():
        res = harness.compare(
            case.quantity,
            other[case.key],
            base[case.key],
            context=reference.context(case),
            tolerances=tolerances,
            raise_on_failure=False,
        )
        entry = per_quantity.setdefault(
            case.quantity,
            {"max_ratio_to_tolerance": 0.0, "max_rel_diff": 0.0, "worst_case": None},
        )
        entry["max_rel_diff"] = max(entry["max_rel_diff"], res.max_rel_diff)
        if res.max_ratio > entry["max_ratio_to_tolerance"]:
            entry["max_ratio_to_tolerance"] = res.max_ratio
            entry["worst_case"] = case.key
    return per_quantity


# ---------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------


def _cosmology_metadata(cosmo) -> dict[str, Any]:
    """The class and parameters of an astropy cosmology, as JSON."""
    params = {}
    for name, value in cosmo.parameters.items():
        if isinstance(value, u.Quantity):
            params[name] = {"value": np.atleast_1d(value.value).tolist(), "unit": str(value.unit)}
            if value.isscalar:
                params[name]["value"] = float(value.value)
        else:
            params[name] = value
    return {"class": type(cosmo).__name__, "name": cosmo.name, "parameters": params}


def main() -> None:
    """Generate the reference."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=DATA)
    parser.add_argument("--no-convergence", action="store_true")
    args = parser.parse_args()

    if hmf.__version__ != HMF_VERSION:
        raise SystemExit(f"Need hmf {HMF_VERSION}, have {hmf.__version__}.")
    if Path(hmf.__file__).resolve().is_relative_to(HERE.parent.parent.resolve()):
        raise SystemExit(f"hmf was imported from the working tree ({hmf.__file__}).")

    warnings.simplefilter("ignore")
    t0 = time.time()
    print(f"Base resolution: dlnk={DLNK}, dlog10m={DLOG10M}")
    base, cases, info = compute(DLNK, DLOG10M)
    base.update(analytic_cases())
    print("Mass step halved, for n(>M):")
    fine, _, _ = compute(DLNK, DLOG10M / 2)
    fine.update(analytic_cases(DLOG10M / 2))
    arrays = mask_extrapolated_ngtm(richardson_ngtm(base, fine))

    conv = None
    if not args.no_convergence:
        harness = _load_harness()
        print("Half steps:")
        half, _, _ = compute(DLNK / 2, DLOG10M / 2)
        # The mass step halved again, for n(>M). dn/dM converges in k to ~1e-7, so
        # this keeps the base dlnk: (dlnk/2, dlog10m/4) needs too much memory.
        print("Mass step quartered, for n(>M):")
        half_fine, _, _ = compute(DLNK, DLOG10M / 4)
        print("Wider k range:")
        wide, _, _ = compute(DLNK, DLOG10M, LNK_RANGE_WIDE)
        print("Wider k range, mass step halved, for n(>M):")
        wide_fine, _, _ = compute(DLNK, DLOG10M / 2, LNK_RANGE_WIDE)
        conv = {
            "half_steps": {
                "dlnk": DLNK / 2,
                "dlog10m": DLOG10M / 2,
                "note": (
                    "n(>M) is extrapolated from (dlnk/2, dlog10m/2) and (dlnk, dlog10m/4) here."
                ),
                "quantities": convergence(arrays, richardson_ngtm(half, half_fine), cases, harness),
            },
            "wider_k_range": {
                "lnk_range": list(LNK_RANGE_WIDE),
                "quantities": convergence(arrays, richardson_ngtm(wide, wide_fine), cases, harness),
            },
        }

    versions = {
        p: importlib.metadata.version(p) for p in ("hmf", "camb", "numpy", "scipy", "astropy")
    }
    assert versions["scipy"] == scipy.__version__
    assert versions["astropy"] == astropy.__version__
    metadata = {
        "description": (
            "Regression reference for hmf v4, computed with hmf v3.7.2 at high resolution "
            "(issue #394). Generated by tests/regression/generate_reference.py."
        ),
        "versions": {**versions, "python": platform.python_version()},
        "settings": {
            **COMMON,
            "dlnk": DLNK,
            "dlog10m": DLOG10M,
            "lnk_range": list(LNK_RANGE),
            "log10m_range": list(LOG10M_RANGE),
            "output_dlnk": DLNK * K_STRIDE,
            "output_dlog10m": DLOG10M * M_STRIDE,
            "transfer_params": TRANSFER_PARAMS,
            "mdef": "each fit's own measured mass definition (disable_mass_conversion=True)",
            "ngtm": (
                "Richardson-extrapolated to dlog10m -> 0 from dlog10m and dlog10m/2: "
                "(4 n(dlog10m/2) - n(dlog10m)) / 3, removing the h^2 error of v3's "
                "trapezoid integration (see richardson_ngtm in generate_reference.py)"
            ),
        },
        "grids": {
            "z": list(Z),
            "z_fit": {k: list(v) for k, v in Z_FIT.items()},
            "z_growth": list(Z_GROWTH),
        },
        # astropy unit strings (parse with astropy.cosmology.units enabled); lnk is
        # ln(k / (littleh / Mpc)).
        "units": {
            "k": "littleh / Mpc",
            "m": "solMass / littleh",
            "transfer": "",
            "power": "Mpc3 / littleh3",
            "growth": "",
            "sigma": "",
            "dlnsdlnm": "",
            "dndm": "littleh4 / (solMass Mpc3)",
            "ngtm": "littleh3 / Mpc3",
            "fsigma": "",
        },
        "cosmologies": {
            name: {**_cosmology_metadata(c), **info["cosmologies"][name]}
            for name, c in COSMOLOGIES.items()
        },
        "fits": list(FITS),
        "ngtm_behroozi_tail_fraction": NGTM_TAIL_FRACTION,
        # T(k) and P(k) of the CAMB cases are NaN below this ln k (see
        # camb_lnk_valid_min in generate_reference.py).
        "camb_lnk_valid_min": info["camb_lnk_valid_min"],
        "analytic": {
            "powerlaw_ps": {
                "cosmology": "lcdm_massless_nu",
                "n": POWERLAW_N,
                "z": 0.0,
                "filter": "TopHat",
                "fit": "PS",
            },
            "eds": {"H0": 70.0, "Om0": 1.0, "Tcmb0": 0.0},
            "lcdm_no_radiation": {"Om0": float(Planck18.Om0), "Tcmb0": 0.0, "flat": True},
            "fsigma_wide": {"ln_inv_sigma": list(LN_INV_SIGMA), "z": list(Z_FSIGMA)},
        },
        "failures": info["failures"],
        "convergence": conv,
        "cases": cases,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / f"{NAME}.npz", **arrays)
    (args.out_dir / f"{NAME}.json").write_text(json.dumps(metadata, indent=1) + "\n")
    size = (args.out_dir / f"{NAME}.npz").stat().st_size
    print(f"Wrote {len(arrays)} arrays ({size / 1e6:.2f} MB) in {time.time() - t0:.0f} s.")
    if conv:
        for which, c in conv.items():
            for q, e in c["quantities"].items():
                print(
                    f"  {which:14s} {q:9s} max ratio {e['max_ratio_to_tolerance']:.3g} "
                    f"(rel {e['max_rel_diff']:.2e}) at {e['worst_case']}"
                )


if __name__ == "__main__":
    main()
