"""Integrity of the v3.7.2 regression reference (issue #394).

Checks that the committed reference is what ``generate_reference.py`` documents:
made with hmf 3.7.2 and the locked numerical dependencies, at the stated resolution,
covering what issue #394 asks for, small enough to keep in the repository, and
converged to within its tolerances (which is what justifies them).
"""

import tomllib
from pathlib import Path

import numpy as np
import pytest
import regression_harness as rh

ROOT = Path(__file__).parents[2]


def test_made_with_hmf_3_7_2(reference):
    assert reference.metadata["versions"]["hmf"] == "3.7.2"


@pytest.mark.parametrize("package", ["camb", "numpy", "scipy", "astropy"])
def test_dependencies_match_lockfile(reference, package):
    """The reference was made with the versions CI installs (``uv.lock``).

    CAMB matters most: the CAMB cases are only compared under the same CAMB, so a
    lockfile bump of CAMB needs the reference regenerated.
    """
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    locked = {p["name"]: p.get("version") for p in lock["package"]}
    assert reference.metadata["versions"][package] == locked[package]


def test_resolution(reference):
    """Computed at dlnk, dlog10m <= 0.005, stored at 0.02 in ln k and 0.05 dex."""
    s = reference.settings
    assert s["dlnk"] <= 0.005
    assert s["dlog10m"] <= 0.005
    np.testing.assert_allclose(np.diff(reference.lnk), s["output_dlnk"], rtol=1e-9)
    np.testing.assert_allclose(np.diff(reference.log10m), s["output_dlog10m"], rtol=1e-9)
    np.testing.assert_allclose(10**reference.log10m, reference.m, rtol=1e-12)


def test_kmax_covers_lowest_mass(reference):
    """k_max R(M_min) >= 1000, v3's threshold for a converged sigma.

    The lowest mass has the smallest Lagrangian radius, R = (3M / 4 pi rho)^(1/3).
    """
    k_max = np.exp(reference.settings["lnk_range"][1])
    for name, c in reference.metadata["cosmologies"].items():
        r_min = (3 * reference.m[0] / (4 * np.pi * c["mean_density0"])) ** (1 / 3)
        assert k_max * r_min >= 1e3, name


def test_size_within_budget():
    """The reference stays under 10 MB, so it can live in the repository."""
    size = sum((rh.DATA / f"{rh.REFERENCE_NAME}.{ext}").stat().st_size for ext in ("npz", "json"))
    assert size <= 10e6


def test_coverage(reference):
    """Every fit, the five cosmologies, z up to 4, both transfers, three filters."""
    meta = reference.metadata
    assert len(meta["fits"]) >= 28
    assert set(meta["cosmologies"]) == {
        "planck18",
        "lcdm_massless_nu",
        "lcdm_mnu0p3",
        "wcdm",
        "w0wacdm",
    }
    assert meta["cosmologies"]["lcdm_mnu0p3"]["parameters"]["m_nu"]["value"] == [0.1] * 3
    assert {0.0, 0.5, 1.0, 2.0, 4.0} <= set(meta["grids"]["z"])

    tophat_camb = {
        (c.cosmology, c.fit)
        for c in reference.cases("dndm")
        if c.filter == "TopHat" and c.transfer == "CAMB"
    }
    assert tophat_camb == {(c, f) for c in meta["cosmologies"] for f in meta["fits"]}
    assert {c.transfer for c in reference.cases("dndm")} == {"CAMB", "EH"}
    assert {c.filter for c in reference.cases("sigma")} == {"TopHat", "SharpK", "SmoothK"}
    assert {c.species for c in reference.cases("transfer")} == {"cb", "tot", None}
    for quantity in rh.QUANTITIES:
        assert any(True for _ in reference.cases(quantity)), quantity


def test_every_case_is_stored_and_finite(reference):
    """Every case is finite, except where masked as a v3 artefact.

    That is CAMB's T(k) and P(k) below CAMB's own k range, and Behroozi's n(>M) where
    v3's power-law tail above the top mass contributes (see generate_reference.py).
    """
    assert reference.metadata["failures"] == []
    valid_min = reference.metadata["camb_lnk_valid_min"]
    for case in reference.cases():
        v = reference.values(case)
        assert v.ndim == len(case.axes), case.key
        if case.quantity in ("transfer", "power") and case.transfer == "CAMB":
            masked = reference.lnk < valid_min[case.cosmology]
            assert np.all(np.isnan(v[masked])), case.key
            assert np.all(np.isfinite(v[~masked])), case.key
            assert np.exp(valid_min[case.cosmology]) < 2e-4, case.key  # ~CAMB's k_min
        elif case.quantity == "ngtm" and case.fit == "Behroozi":
            # Masked where v3's extrapolated tail above the top mass exceeds 1e-5;
            # that region only grows towards high mass.
            finite = np.isfinite(v)
            assert np.all(np.diff(finite.astype(int), axis=1) <= 0), case.key
            assert np.all(finite[:, 0]), case.key
        else:
            assert np.all(np.isfinite(v)), case.key


def test_tolerances_are_justified(tolerances):
    assert set(tolerances.quantities) == set(rh.QUANTITIES)
    for name, tol in tolerances.quantities.items():
        assert 0 < tol.rtol <= 1e-4, name
        assert len(tol.justification) > 40, name
    for override in tolerances.overrides:
        assert len(override["justification"]) > 40, override


@pytest.mark.parametrize("study", ["half_steps", "wider_k_range"])
def test_reference_is_converged(reference, study):
    """Halving both steps, or widening the k range, changes nothing by its tolerance.

    So the reference is accurate to within the tolerances it is compared with.
    """
    conv = reference.metadata["convergence"]
    assert conv is not None, "the reference was generated without --convergence"
    quantities = conv[study]["quantities"]
    assert set(quantities) == set(rh.QUANTITIES)
    for name, q in quantities.items():
        assert q["max_ratio_to_tolerance"] < 1, (name, q)
