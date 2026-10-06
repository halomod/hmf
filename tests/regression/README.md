# The v3.7.2 regression reference for hmf v4

Issue #394: hmf v4 is checked against a **high-resolution v3.7.2** reference, not
against default-resolution v3 (whose own discretisation error is ~7e-4 in `dndm`).

| File | What |
|---|---|
| `generate_reference.py` | Makes the reference, with hmf **3.7.2** in an isolated, pinned environment |
| `data/reference_v3.7.2.npz` | The reference arrays (compressed, ~5.6 MB) |
| `data/reference_v3.7.2.json` | Metadata: versions, settings, cosmologies, cases, convergence |
| `data/tolerances.json` | Per-quantity tolerances, each with its justification (hand-written) |
| `regression_harness.py` | Loader, `compare()`, and the v4-provider registry |
| `v4_providers.py` | Where the step-2 PRs register their v4 providers |
| `test_reference_data.py` | The reference is what it claims: versions, resolution, coverage, size, convergence |
| `test_reference_physics.py` | Physical checks of the reference itself (closed forms, normalisations, bounds) |
| `test_regression_harness.py` | Tests of the harness |
| `test_regression_v4.py` | v4 against the reference: one test per quantity, skipped until it has a provider; and the v4 normalised fits on the wide σ grid |

## Regenerating

```bash
uv run --no-project --script tests/regression/generate_reference.py
```

`uv` builds the environment from the script's inline metadata: Python 3.13,
`hmf==3.7.2`, and camb, numpy, scipy and astropy pinned to the versions in `uv.lock`
(so CI runs the same CAMB the reference was made with). The script refuses to run
against the working tree's `hmf`. It takes about 7 minutes, of which 6 are the
convergence runs. Regenerate when `uv.lock` changes CAMB (a test checks that the
versions match); do not regenerate to make a v4 comparison pass.

## What is in it

Computed with `dlnk = 0.005` on `-18 ≤ ln k ≤ 12` and `dlog10m = 0.005` on
`6 ≤ log10 M ≤ 16` (k_max R(M_min) > 2000), and stored every 0.02 in ln k and every
0.05 dex in M:

- **cosmologies:** `planck18` (astropy's, Σm_ν = 0.06 eV), `lcdm_massless_nu`,
  `lcdm_mnu0p3` (3 × 0.1 eV), `wcdm` (w = −0.9), `w0wacdm` (w0 = −0.9, wa = 0.2), all
  with Planck18's other parameters; σ8 = 0.8102, n = 0.9665, δc = 1.686;
- **transfer** (CAMB `cb` and `tot`, and EH) and **power** on the k grid; **growth**
  on 0 ≤ z ≤ 10;
- **sigma** (z = 0, 0.5, 1, 2, 4) and **dlnsdlnm** for every cosmology, transfer and
  filter (TopHat, SharpK, SmoothK);
- **dndm**, **ngtm** and **fsigma** for every one of the 28 fits × every cosmology ×
  z (CAMB, TopHat); every fit for EH (Planck18); PS, SMT and Tinker08 for SharpK and
  SmoothK (CAMB, Planck18). Yung24 is calibrated only for z ≥ 6, so it has z = 6, 8, 10;
- **analytic cases** (not compared with v4): power-law (P ∝ k⁻²) Press–Schechter, Einstein–de
  Sitter and radiation-free ΛCDM growth, and f(σ) of the normalised fits on a wide σ
  grid.

Choices that differ from v3.7.2's defaults, and why:

- **Each fit in its own measured mass definition** (`disable_mass_conversion=True`):
  the reference regresses the fits, not v3's NFW mass conversion.
- **`ODEGrowthFactor` for every cosmology.** The default `GrowthFactor` switches
  between Eisenstein97 (no radiation) and the ODE (with radiation) at a radiation
  fraction of 5e-4, normalising both by the Eisenstein97 D(0): D(z) jumps by ~1e-4 at
  the switch, and an array of z gives different values from each z alone.
- **n(>M) is Richardson-extrapolated to dlog10m → 0.** v3 integrates dn/dln M with a
  cumulative trapezoid, whose error (s Δln M)²/12, for a tail falling as e^(−s ln M),
  reaches ~1% at the top masses at dlog10m = 0.005 (s ≈ 30 for SharpK at z = 2). The
  reference stores (4 n(Δ/2) − n(Δ))/3, which cancels it.
- **Behroozi's n(>M) is masked where v3 extrapolates.** For Behroozi alone v3 does
  not compute dn/dM above the top mass, but adds a power law through the last two
  grid points; where that tail is over 1e-5 of n(>M) (near the top mass), the value
  is an extrapolation artefact, so it is NaN.
- **CAMB's T(k) and P(k) are masked below CAMB's lowest k** (~1e-4 h/Mpc), where v3
  joins its table to `lnk_min` with a cubic spline, so T depends on `lnk_min` (by 3%)
  and dips below 1. That region changes σ by < 1e-9.

## Tolerances and convergence

`data/tolerances.json` gives each quantity's relative tolerance and why: σ 1e-5 and
`dndm` 1e-4 are v4's accuracy targets (#384); T 1e-5, P 2e-5, D 2e-6 and dlnσ/dlnM
1e-5 follow from the σ budget. For `dndm`, `fsigma` and `ngtm` the σ tolerance is
propagated: their tolerance is `1e-4 + |dln f/dln σ| × 1e-5`, since in the exponential
tail a relative error ε in σ changes f by |dln f/dln σ| ε ~ ν² ε. Values below a
physical floor (fewer than 1e-17 haloes in the observable universe) are not compared.

The generator recomputes everything with both steps halved, and on a wider k range
(`-20 ≤ ln k ≤ 14`), and records the largest change of each quantity in units of its
tolerance; `test_reference_is_converged` requires it to be below 1. That is what makes
the tolerances meaningful: the reference itself is accurate to well within them.

## v4 providers

| Quantity | Provider | Status |
|---|---|---|
| `fsigma` | `v4_providers.fsigma`: each `hmf.core.fits` fit on the reference's own σ(M, z) and n_eff, so the fit alone is compared | all 174 cases agree with v3.7.2 to ~1e-14 |
| `transfer`, `power`, `growth`, `sigma`, `dlnsdlnm` | — | the stages exist; providers to come |
| `dndm`, `ngtm` | — | need the mass-function stage |

The `fsigma` provider gives each fit the inputs v3.7.2 gave it, and undoes v4's
intentional changes of a default: Manera's p (0.248 in v4, 0.289 in v3), Watson's
mass definition (SO-mean(178) in v4, virial in v3), and Bocquet 200c/500c's mass
ratio M_Δ/M200m, which v4 keeps out of f(σ) (`mass_ratio_to_200m`). These are listed
in `V3_PARAMETERS` and `V3_MASS_DEFINITIONS` in `v4_providers.py`.

## Adding a v4 provider

In `v4_providers.py`:

```python
from regression_harness import register_provider


@register_provider("sigma")
def sigma(case, reference):
    if case.filter not in ("TopHat",):
        return None  # not supported yet: skipped
    cosmo = reference.cosmology(case.cosmology)
    ...  # build the v4 stages for case.transfer, case.filter
    return np.array([variance.sigma(reference.m * Msun_h, z=z) for z in case.z])
```

`test_regression_v4.py` then runs every `sigma` case through it. A provider returns
the case's values with the shape of `reference.values(case)`, as a Quantity (in
h-units) or as an array in the units of `reference.metadata["units"]`, or `None` to
skip a case. CAMB cases are skipped when the installed CAMB differs from the
reference's.
