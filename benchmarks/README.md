# hmf benchmarks

A benchmark suite recording hmf v3's performance, so that later performance PRs
and the v4 rewrite can be measured against it. The workloads are the common ways
hmf is used (building a mass function, scanning redshift, cosmology and fitting
function, derived quantities) plus the paths known to be slow in v3.

The suite lives outside `tests/` and is not collected by `uv run pytest`
(`testpaths = ["tests"]`), so it does not slow the test suite down.

## Running

```bash
uv sync --locked --all-extras --dev

# Timings plus the call-count assertions (about 35 s):
uv run pytest benchmarks --no-cov -p no:cacheprovider --benchmark-json=results.json

# Only the call-count assertions, each workload run once (about 10 s):
uv run pytest benchmarks --no-cov -p no:cacheprovider --benchmark-disable
```

Pass `--no-cov`: the project's pytest options turn coverage on, and tracing adds
its own overhead to every timing.

## Comparing

```bash
uv run python benchmarks/compare.py benchmarks/baseline.json results.json
```

prints a Markdown table of median times against the baseline, with the time per
item for the loops and the ratio. It never fails on a slowdown. pytest-benchmark's
own comparison also works:

```bash
uv run pytest-benchmark compare benchmarks/baseline.json results.json --columns=median,iqr,rounds
```

**Compare runs from the same machine.** The baseline was recorded on the machine
below; a CI runner or your laptop will differ from it by more than most real
changes. To measure a change, run the suite on `main` and on your branch on the
same machine, and compare those two files.

## In CI

`.github/workflows/benchmarks.yaml` runs the suite on every PR into `main` and on
every push to `main`. It

- **never fails on timings**: shared runners are too noisy to gate on. The
  comparison with `baseline.json` goes to the job summary and the results file is
  uploaded as the `benchmark-results` artifact;
- **fails on the hard assertions**, which count calls and do not time anything:
  see below.

## What is asserted

The `calls` fixture (`conftest.py`) counts, per timed round:

- `camb`: calls to `camb.get_transfer_functions`, i.e. CAMB runs;
- `sigma_grid`: evaluations of σ(R) on an array of radii (calls of the σ
  integral shared by `BaseFilter.sigma` and the fused
  `BaseFilter.sigma_and_dlnss_dlnr`, with more than one radius).

Setup runs before the counters are reset, so only the timed region is counted.
The assertions are:

| Workload | Assertion |
|---|---|
| Default `MassFunction()` (+ `dndm`) | exactly 1 CAMB run (#363; the double run must not come back) |
| First / cached `dndm` after construction | 0 CAMB runs; cached access recomputes no σ(R) |
| `get_hmf("dndm", z=[20 values])` | exactly 1 CAMB run |
| z, σ8, fitting-function and WDM z scans | 0 CAMB runs and 0 σ(R) evaluations |
| `n` scan, `ngtm` z-loop, mass conversion, halofit | 0 CAMB runs |
| `ngtm` z-loop | 0 σ(R) evaluations per z (a ratchet) |
| Introspection classmethods | 0 CAMB runs (a ratchet) |

The ratchets are upper bounds on known waste in v3: a PR
that removes it should lower them.

## Workloads

| # | Benchmark | What one round times |
|---|---|---|
| 1 | `test_default_construct_and_dndm` | `MassFunction().dndm` with CAMB |
| 1 | `test_default_construct` | `MassFunction()` alone |
| 1 | `test_default_first_dndm` | first `dndm` of a freshly built object |
| 1 | `test_default_cached_dndm` | `dndm` again |
| 2 | `test_eh_construct_and_dndm` | `MassFunction(transfer_model="EH").dndm` |
| 3 | `test_z_loop_dndm` | 20 × `update(z=...)` + `dndm` |
| 4 | `test_z_loop_ngtm` | 20 × `update(z=...)` + `ngtm` |
| 5 | `test_sigma8_scan` | 20 × `update(sigma_8=...)` + `dndm` |
| 6 | `test_n_scan` | 5 × `update(n=...)` + `dndm` |
| 7 | `test_fit_scan` | `update(hmf_model=...)` + `dndm` for each of the 28 built-in fits (`Yung24` at z=6, the rest at z=0) |
| 8 | `test_get_hmf_z` | `get_hmf("dndm", z=[20 values])` |
| 9 | `test_mass_conversion_first` | first `dndm` converting SMT → SOMean(200), σ already computed |
| 9 | `test_mass_conversion_z_loop` | 5 × `update(z=...)` + `dndm` with that conversion |
| 10 | `test_wdm_z_loop` | `MassFunctionWDM`: 20 × `update(z=...)` + `dndm` |
| 11 | `test_halofit_first` | first `nonlinear_power` of a fresh object |
| 11 | `test_halofit_after_z_change` | `nonlinear_power` after a z change |
| 12 | `test_introspection[...]` | `get_all_parameter_names`, `quantities_available`, `parameter_info`, `get_all_parameter_defaults` |
| 13 | `test_import[import_hmf]` | `python -c "import hmf"` in a subprocess |
| 13 | `test_import[python]` | `python -c "pass"`, the interpreter start-up to subtract |
| 14 | `test_unit_boundary_overhead[undecorated]` | 10,000 calls of an identity method on 500 masses |
| 14 | `test_unit_boundary_overhead[canonical]` | the same method behind `hmf.core.units.unit_boundary`, masses in `Msun_h` |
| 14 | `test_unit_boundary_overhead[physical]` | the same, masses in `u.Msun` (cached H0 conversion, plus one array multiply) |

Workload 14 measures the fixed cost of the `hmf.core` units boundary: the
*per item* time of `[canonical]` minus that of `[undecorated]` is the overhead per
call, whose budget is **2 µs** (issue #389). `baseline.json` predates it, so it has
no baseline entry. `test_unit_boundary_overhead_sanity` asserts only a generous
20 µs, so that it never fails on a noisy runner.

Inputs are fixed, everything a benchmark uses is imported before timing starts
(halomod included, for the mass conversion), and warnings are silenced so that
formatting them is not timed. The loops start each round from a state where the
quantity is already computed at the start of the scan.

Note that in v3 `MassFunction()` already runs CAMB and computes σ: `validate()`
evaluates `_sigma_k_truncation_error` at construction. So the first `dndm` after
construction is cheap, and construction is where the time goes.

## The baseline

`baseline.json` was recorded on v3 (`main` at `57b3ae5`, with this suite applied)
with

```bash
uv run pytest benchmarks --no-cov -p no:cacheprovider --benchmark-json=benchmarks/baseline.json
```

on:

- Intel Xeon @ 2.10 GHz, 4 vCPUs, 15 GB RAM (a cloud container, not a dedicated
  machine);
- Linux 6.18, x86_64;
- Python 3.13.14, with the dependencies pinned in `uv.lock` (camb 1.6.6,
  numpy 2.4.3, scipy 1.17.1, astropy 7.2.0).

Full details are in its `machine_info` and `commit_info` fields.

### Refreshing it

Refresh the baseline when a PR deliberately changes performance (and say so in
the PR), or when the workloads change. Run the command above on a quiet machine,
check the call-count assertions pass, and update the machine description here.
Don't refresh it to hide a slowdown.
