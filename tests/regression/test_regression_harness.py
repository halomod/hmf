"""Tests of the regression comparison harness itself (``regression_harness.py``)."""

import astropy.units as u
import numpy as np
import pytest
import regression_harness as rh

from hmf.core.units import dndm_unit, number_density_unit


def test_compare_within_and_outside_tolerance(tolerances):
    ref = np.linspace(1, 2, 10)
    ok = rh.compare("sigma", ref * (1 + 0.9e-5), ref, tolerances=tolerances)
    assert ok.passed
    assert ok.max_rel_diff == pytest.approx(0.9e-5, rel=1e-6)
    assert ok.max_ratio == pytest.approx(0.9, rel=1e-6)

    bad = ref.copy()
    bad[3] *= 1 + 2e-5
    with pytest.raises(rh.RegressionMismatchError, match="sigma: FAIL"):
        rh.compare("sigma", bad, ref, tolerances=tolerances)
    res = rh.compare("sigma", bad, ref, tolerances=tolerances, raise_on_failure=False)
    assert not res.passed
    assert res.worst_index == (3,)


def test_compare_skips_floor_and_nonfinite(tolerances):
    ref = np.array([1.0, 0.0, 1e-300, np.nan, 2.0])
    act = np.array([1.0, 5.0, 3e-300, 7.0, 2.0])
    res = rh.compare("dlnsdlnm", act, ref, tolerances=tolerances)
    assert res.passed
    assert (res.n_compared, res.n_skipped) == (2, 3)


def test_compare_nan_result_fails(tolerances):
    ref = np.ones(3)
    with pytest.raises(rh.RegressionMismatchError):
        rh.compare("sigma", np.array([1.0, np.nan, 1.0]), ref, tolerances=tolerances)


def test_compare_shape_mismatch(tolerances):
    with pytest.raises(ValueError, match="shape"):
        rh.compare("sigma", np.ones(3), np.ones(4), tolerances=tolerances)


def test_amplified_tolerance_needs_context(tolerances):
    with pytest.raises(ValueError, match="fsigma_slope"):
        rh.compare("dndm", np.ones(3), np.ones(3), tolerances=tolerances)


def test_amplified_tolerance_adds_propagated_sigma_error(tolerances):
    """Dndm tolerance = rtol + |dln f/dln sigma| * rtol(sigma)."""
    ref = np.ones(3)
    slope = np.array([0.0, 10.0, 100.0])
    # Allowed: 1e-4, 2e-4, 1.1e-3.
    act = ref * (1 + np.array([0.95e-4, 1.9e-4, 1.05e-3]))
    res = rh.compare("dndm", act, ref, context={"fsigma_slope": slope}, tolerances=tolerances)
    assert res.passed
    act[0] = 1 + 1.05e-4
    assert not rh.compare(
        "dndm",
        act,
        ref,
        context={"fsigma_slope": slope},
        tolerances=tolerances,
        raise_on_failure=False,
    ).passed


def _toy_reference(sigma, fsigma, dndm, ngtm, m):
    """A one-fit reference with the given arrays (one redshift)."""
    common = {"cosmology": "c", "transfer": "EH", "species": None, "filter": "TopHat"}
    cases = [
        {
            **common,
            "quantity": "sigma",
            "key": "sigma",
            "fit": None,
            "z": [0.0],
            "axes": ["z", "log10m"],
        },
        {
            **common,
            "quantity": "fsigma",
            "key": "fsigma",
            "fit": "F",
            "z": [0.0],
            "axes": ["z", "log10m"],
        },
        {
            **common,
            "quantity": "dndm",
            "key": "dndm",
            "fit": "F",
            "z": [0.0],
            "axes": ["z", "log10m"],
        },
        {
            **common,
            "quantity": "ngtm",
            "key": "ngtm",
            "fit": "F",
            "z": [0.0],
            "axes": ["z", "log10m"],
        },
    ]
    arrays = {
        "sigma": sigma[None],
        "fsigma": fsigma[None],
        "dndm": dndm[None],
        "ngtm": ngtm[None],
        "grid/m": m,
    }
    return rh.Reference(arrays, {"cases": cases, "grids": {"z_growth": [0.0]}})


def test_context_slope_is_dlnf_dlnsigma():
    """For f = exp(-c / sigma^2), the context's slope is |dln f/dln sigma| = 2c / sigma^2.

    It is a finite difference on the stored grid, so it matches to that accuracy.
    """
    m = np.logspace(10, 15, 501)
    sigma = 3 * (m / 1e10) ** -0.2
    f = np.exp(-1.5 / sigma**2)
    ref = _toy_reference(sigma, f, f, f, m)
    slope = ref.context(ref.find("dndm"))["fsigma_slope"][0]
    np.testing.assert_allclose(slope[1:-1], 3 / sigma[1:-1] ** 2, rtol=1e-4)


def test_context_ngtm_weight_of_constant_slope_is_that_slope():
    """A dn/dln M-weighted average of a constant is the constant."""
    m = np.logspace(10, 15, 201)
    sigma = 3 * (m / 1e10) ** -0.2
    f = sigma**-2.5  # |dln f/dln sigma| = 2.5 everywhere
    dndm = m**-2.0
    ngtm = 1 / m
    ref = _toy_reference(sigma, f, dndm, ngtm, m)
    w = ref.context(ref.find("ngtm"))["ngtm_weighted_slope"][0]
    np.testing.assert_allclose(w, 2.5, rtol=1e-6)


def test_compare_converts_quantities(reference, tolerances):
    case = next(reference.cases("ngtm"))
    ref = reference.values(case)
    ctx = reference.context(case)
    res = rh.compare("ngtm", ref * number_density_unit, ref, context=ctx, tolerances=tolerances)
    assert res.max_rel_diff == 0
    # Physical units are not converted silently: the h's must match.
    with pytest.raises(u.UnitConversionError):
        rh.compare("ngtm", ref * u.Mpc**-3, ref, context=ctx, tolerances=tolerances)
    with pytest.raises(u.UnitConversionError):
        rh.compare("ngtm", ref * dndm_unit, ref, context=ctx, tolerances=tolerances)


def test_tolerance_override(tolerances):
    tols = rh.Tolerances(
        tolerances.quantities,
        ({"quantity": "sigma", "filter": "SharpK", "rtol": 1e-3, "justification": "x"},),
        tolerances.floor,
    )
    sharpk = rh.Case("sigma", "k", filter="SharpK")
    tophat = rh.Case("sigma", "k", filter="TopHat")
    assert tols.get("sigma", sharpk).rtol == 1e-3
    assert tols.get("sigma", tophat).rtol == tolerances.quantities["sigma"].rtol


def test_tolerance_override_on_a_range_of_lnk(tolerances):
    """An override with an lnk_range applies to that part of the k axis only."""
    override = {
        "quantity": "transfer",
        "transfer": "CAMB",
        "lnk_range": [0.0, None],
        "rtol": 1e-3,
        "justification": "x",
    }
    tols = rh.Tolerances(tolerances.quantities, (override,), tolerances.floor)
    camb = rh.Case("transfer", "k", transfer="CAMB", axes=("lnk",))
    eh = rh.Case("transfer", "k", transfer="EH", axes=("lnk",))
    assert tols.get("transfer", camb).rtol == tolerances.quantities["transfer"].rtol
    assert tols.lnk_ranges("transfer", camb) == [(0.0, np.inf, 1e-3)]
    assert tols.lnk_ranges("transfer", eh) == []

    lnk = np.array([-2.0, -1.0, 0.0, 1.0])
    ref = np.ones(4)
    off = ref * np.array([1, 1, 1 + 5e-4, 1 + 5e-4])
    ok = rh.compare("transfer", off, ref, case=camb, context={"lnk": lnk}, tolerances=tols)
    assert ok.passed
    with pytest.raises(rh.RegressionMismatchError):
        rh.compare("transfer", off, ref, case=eh, context={"lnk": lnk}, tolerances=tols)
    low = ref * np.array([1 + 5e-4, 1, 1, 1])
    with pytest.raises(rh.RegressionMismatchError):
        rh.compare("transfer", low, ref, case=camb, context={"lnk": lnk}, tolerances=tols)
    with pytest.raises(ValueError, match="lnk_range"):
        rh.compare("transfer", off, ref, case=camb, tolerances=tols)


def test_reference_cosmology_round_trips(reference):
    """The cosmologies rebuilt from the metadata are the ones v3 used."""
    from astropy.cosmology import Planck18

    p18 = reference.cosmology("planck18")
    assert p18.H0 == Planck18.H0
    assert p18.Om0 == Planck18.Om0
    assert np.all(p18.m_nu == Planck18.m_nu)
    assert reference.cosmology("w0wacdm").wa == 0.2


def test_sigma_at_uses_growth_off_the_sigma_grid(reference):
    """For redshifts the sigma table lacks (Yung24's), sigma = D(z) sigma(z=0)."""
    case = reference.find(
        "dndm", cosmology="planck18", transfer="CAMB", filter="TopHat", fit="Yung24"
    )
    sig = reference.sigma_at(case)
    s0 = reference.values(
        reference.find("sigma", cosmology="planck18", transfer="CAMB", filter="TopHat")
    )[0]
    growth = reference.values(reference.find("growth", cosmology="planck18"))
    i = list(reference.z_growth).index(case.z[0])
    np.testing.assert_allclose(sig[0], s0 * growth[i], rtol=1e-14)


def test_provider_registry(monkeypatch):
    monkeypatch.setattr(rh, "_PROVIDERS", {})
    assert rh.get_provider("sigma") is None

    @rh.register_provider("sigma")
    def provider(case, reference):
        return None

    assert rh.get_provider("sigma") is provider
    rh.register_provider("sigma")(provider)  # re-registering the same function is fine
    with pytest.raises(ValueError, match="already registered"):
        rh.register_provider("sigma")(lambda case, reference: None)
    with pytest.raises(KeyError, match="unknown quantity"):
        rh.register_provider("nope")


def test_quantity_floor(tolerances):
    """The n(>M) floor: below 1e-30 h^3/Mpc^3 it is not compared, above it it is.

    That is fewer than 1e-17 haloes in the observable universe.
    """
    ref = np.array([1e-3, 1e-29, 1e-31, 1e-230])
    act = ref * np.array([1.0, 1.0, 2.0, 1.5])
    res = rh.compare(
        "ngtm", act, ref, context={"ngtm_weighted_slope": np.zeros(4)}, tolerances=tolerances
    )
    assert (res.n_compared, res.n_skipped) == (2, 2)
    assert tolerances.get("ngtm").floor == 1e-30
