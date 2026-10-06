"""Compare hmf v4 with the v3.7.2 regression reference (issue #394).

One test per quantity runs every case of that quantity through its v4 provider (see
``v4_providers.py``) and compares the result with the reference, with the tolerance
of ``data/tolerances.json``. A quantity without a provider is skipped: the v4
stages do not exist yet, so for now every test here is skipped.
"""

import numpy as np
import pytest
import regression_harness as rh
import v4_providers  # registers the providers


def _installed_camb() -> str | None:
    try:
        import camb
    except ImportError:  # pragma: no cover
        return None
    return camb.__version__


@pytest.mark.parametrize("quantity", list(rh.QUANTITIES))
def test_v4_matches_v3_reference(quantity, reference, tolerances):
    provider = rh.get_provider(quantity)
    if provider is None:
        pytest.skip(f"no v4 provider of {quantity!r} is registered yet (see v4_providers.py)")

    # CAMB's own output is not what is regressed: compare CAMB cases only with the CAMB
    # the reference was made with.
    same_camb = _installed_camb() == reference.metadata["versions"]["camb"]
    failures, n_compared, n_skipped = [], 0, 0
    for case in reference.cases(quantity):
        if case.transfer == "CAMB" and not same_camb:
            n_skipped += 1
            continue
        actual = provider(case, reference)
        if actual is None:
            n_skipped += 1
            continue
        result = rh.compare(
            quantity,
            actual,
            reference.values(case),
            case=case,
            context=reference.context(case),
            tolerances=tolerances,
            raise_on_failure=False,
        )
        n_compared += 1
        if not result.passed:
            failures.append(f"{case.key}: {result}")

    if n_compared == 0:
        pytest.skip(f"the v4 provider of {quantity!r} supports none of its {n_skipped} cases")
    assert not failures, f"{len(failures)} of {n_compared} cases differ:\n" + "\n".join(failures)


def _wide_fits(reference):
    keys = [k for k in reference.arrays if k.startswith("analytic/fsigma_wide/") and "=" in k]
    return sorted(k.split("/", 2)[2] for k in keys)


@pytest.mark.parametrize("fit_z", _wide_fits(rh.load_reference()))
def test_v4_normalised_fits_on_wide_sigma_range(reference, fit_z):
    """The v4 normalised fits against v3.7.2 over ln(1/sigma) in [-40, 4].

    The reference's f(sigma) of each normalised fit on that wide grid (Planck18, the
    inputs of generate_reference.py: n_eff = -2, m = 1e12 Msun/h) checks the fits far
    beyond the masses of the other cases, where their normalisation is decided.
    Tolerance 1e-10: the same formulas, up to rounding at extreme sigma.
    """
    fit, z = fit_z.split("/z=")
    x = reference.arrays["analytic/fsigma_wide/ln_inv_sigma"]
    expected = reference.arrays[f"analytic/fsigma_wide/{fit_z}"]
    actual = v4_providers.v4_fsigma(
        fit,
        np.exp(-x),
        z=float(z),
        cosmo=reference.cosmology("planck18"),
        delta_c=reference.settings["delta_c"],
        n_eff=np.full_like(x, -2.0),
        m=np.full_like(x, 1e12),
    )
    ok = expected > 1e-290
    assert ok.sum() > 0.5 * x.size
    np.testing.assert_allclose(actual[ok], expected[ok], rtol=1e-10)
    assert np.all(actual[~ok] < 1e-280)
