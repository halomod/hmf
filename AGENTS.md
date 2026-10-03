# AGENTS.md

Instructions for AI coding agents working in this repository.

## Do

- Use `uv run` for all project tooling commands (pytest, ruff, etc.).
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

File-scoped quality checks (preferred):

```bash
uv run ruff format path/to/file.py
uv run ruff check --fix path/to/file.py
```

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
- run file-scoped ruff checks
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
