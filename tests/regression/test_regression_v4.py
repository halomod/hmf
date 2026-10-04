"""Compare hmf v4 with the v3.7.2 regression reference (issue #394).

One test per quantity runs every case of that quantity through its v4 provider (see
``v4_providers.py``) and compares the result with the reference, with the tolerance
of ``data/tolerances.json``. A quantity without a provider is skipped: the v4
stages do not exist yet, so for now every test here is skipped.
"""

import pytest
import regression_harness as rh
import v4_providers  # noqa: F401  (registers the providers)


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
