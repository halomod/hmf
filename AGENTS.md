# AGENTS.md

Instructions for AI coding agents working in this repository.

## Do

- Use `uv run` for all project tooling commands (pytest, prek, etc.).
- Set up the environment with `uv sync`; prefer `uv sync --locked --all-extras --dev`
  for CI parity.
- Add type hints for all new parameters.
- Write or update tests for every behavior change.
- Add numpydoc-style docstrings for new public modules, classes, and functions.
- Update docstrings whenever parameters are added or changed.
- Prefer minimal, file-scoped checks before broad checks.

## Do not

- Do not edit `CHANGELOG.rst` unless explicitly asked. The changelog is generated
  from GitHub Release notes, so hand-edits to that file go stale and confuse
  readers.
- Do not modify regression artifacts or data snapshots unless explicitly asked.
- Do not run full test suites or expensive notebook/doc executions without approval.
- Do not introduce API-breaking renames or default changes without explicit
  confirmation.

## Commands

File-scoped quality checks (preferred). Run lint and format through the
pre-commit hooks, not by calling `ruff` directly:

```bash
uv run prek run --files path/to/file.py [path/to/other.py ...]
```

The hooks pin their own ruff version in `.pre-commit-config.yaml`, which is
what pre-commit.ci enforces. The ruff in the uv environment can lag behind it,
and a different version can disagree with the hooks (e.g. strip a `# noqa`
the pinned version needs). Do not call `uv run ruff ...` for checks or fixes.

Targeted tests (preferred):

```bash
uv run pytest tests/path/to/test_file.py
uv run pytest tests/path/to/test_file.py -k keyword
```

Wider checks (ask first):

```bash
uv run prek run -a
uv run pytest
```

## Safety and permissions

Allowed without prompt:

- read, list, and search files
- edit source, tests, and docs text files relevant to the task
- run file-scoped prek checks (`uv run prek run --files ...`)
- run one or a few targeted pytest files

Ask first:

- package or environment installs/upgrades
- full-suite pytest runs
- notebook execution across docs/examples
- regenerating regression data
- deleting files, chmod, git push

## Testing expectations

- Bug fixes should include regression tests.
- Use explicit numeric tolerances for floating-point comparisons.
- Handle warnings intentionally (fix or filter in tests).
- Physical features and physical bug fixes must be backed by *physical* tests,
  not just tests that exercise the code for coverage. Acceptable physical tests
  include:
  - comparison against an analytic solution;
  - a special case or limit in which the result is known (e.g. a parameter
    choice that reduces the model to a simpler one);
  - checking that values lie within physically reasonable bounds.

  A test whose reference values come from an explicit rewrite of the function
  under test is **not** a physical test. Re-implementing the same formula in
  the test only checks that the code agrees with itself.

## hmf.core (v4) conventions

`src/hmf/core` is the experimental v4 preview (see `docs/core.rst`). Don't import it
from v3 code: `import hmf` must not import `hmf.core`. In it:

- **Units:** Quantities at every public boundary; plain arrays in canonical units
  (`hmf.core.units.CANONICAL_UNITS`) inside. Decorate public methods with
  `units.unit_boundary`; attach units only with the shared constants (`Msun_h`, …),
  never a freshly built `u.Msun / cu.littleh`. Bare floats for dimensional inputs
  raise `UnitBoundaryError`. Never enable `cu.with_H0` globally. Budget: ≤ 2 µs/call.
- **Kernels** (`hmf.core._kernels`): pure, unit-free, vectorised, no input mutation,
  batch-size-independent. Library code calls kernels, not decorated methods.
- **Models:** `@attrs.frozen(kw_only=True)` subclasses of a kind
  (`class X(Model, kind=True)`); register with `alias=...`; look up via `Kind.get()`.
- **Stages:** `@attrs.frozen(kw_only=True)` `Stage`s; change via `evolve()`; cache
  with `functools.cached_property`; document fields with `hmf.core.field(doc=...)`.
- **Domains/accuracy:** use `domain.Domain`/`apply_domain_policy` and the
  `accuracy` classes; don't invent new sentinels or grid settings.
- Name logs `log10_…` or `ln_…`, never bare `log`. `uv run mypy` (strict) must pass.

## API docs

- Docs are built with sphinx and numpydoc.
- Keep API docs indexed in `docs/api.rst` when adding new public objects.

## Process note

- Branch/release docs and workflows may differ historically; do not assume a
  release branching strategy unless the user specifies one.

## PR checklist

- Formatting and lint checks pass.
- Unit tests pass.
- Tests added for new or changed behavior.
- The PR is labeled appropriately. Use only the official labels defined in
  `.github/labels.yml`. Release notes (and hence the changelog) are grouped by
  these labels via `.github/release-drafter.yml`, so an unlabeled or mislabeled
  PR ends up in the wrong section. If no existing label fits, add the new label
  to `labels.yml` (and to `release-drafter.yml` if it should get its own
  release-notes section) rather than creating it ad hoc on GitHub.

## When stuck

- Ask a clarifying question.
- Propose a short plan with assumptions.
- Provide a minimal draft patch with explicit unknowns.

## Test-first mode

- For new features, write or update tests first, then code to green.
