"""Compare a pytest-benchmark JSON run against the committed baseline.

Prints a Markdown table of median times (and the time per item for loops), with the
ratio to the baseline. It is informational only: it never fails on a slowdown,
because timings on shared CI runners are too noisy to gate on. It exits non-zero
only if a file cannot be read.

Usage::

    python benchmarks/compare.py benchmarks/baseline.json results.json
"""

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict[str, dict]:
    """Map each benchmark's name to its entry in a pytest-benchmark JSON file."""
    data = json.loads(path.read_text())
    return {b["name"]: b for b in data["benchmarks"]}


def _machine(path: Path) -> str:
    """A one-line description of the machine a JSON file was recorded on."""
    data = json.loads(path.read_text())
    info = data.get("machine_info", {})
    cpu = info.get("cpu", {}).get("brand_raw", "unknown CPU")
    return f"{cpu}, Python {info.get('python_version', '?')}, {info.get('system', '?')}"


def _fmt(seconds: float) -> str:
    """Format a time in s, ms or µs."""
    if seconds >= 1:
        return f"{seconds:.2f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds * 1e6:.1f} µs"


def compare(baseline: Path, current: Path) -> str:
    """Return the Markdown comparison of ``current`` against ``baseline``.

    Parameters
    ----------
    baseline
        The pytest-benchmark JSON file to compare against.
    current
        The pytest-benchmark JSON file of the new run.

    Returns
    -------
    str
        A Markdown report.
    """
    base = _load(baseline)
    new = _load(current)

    lines = [
        "## hmf benchmarks",
        "",
        f"- Baseline: {_machine(baseline)}",
        f"- This run: {_machine(current)}",
        "",
        (
            "Median wall time per round; *per item* divides by the number of redshifts, "
            "parameter values or models a round loops over. Ratio is this run over the "
            "baseline (> 1 is slower). Differences between machines dominate a "
            "cross-machine ratio: compare runs from the same machine."
        ),
        "",
        "| Benchmark | Baseline | This run | Per item (baseline → this run) | Ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in sorted(base.keys() | new.keys()):
        b, n = base.get(name), new.get(name)
        b_med = b["stats"]["median"] if b else None
        n_med = n["stats"]["median"] if n else None
        items = (n or b).get("extra_info", {}).get("n_items", 1)
        per_item = ""
        if items > 1:
            per_item = " → ".join(_fmt(m / items) if m is not None else "—" for m in (b_med, n_med))
        ratio = f"{n_med / b_med:.2f}" if b_med and n_med else "—"
        lines.append(
            f"| `{name}` | {_fmt(b_med) if b_med else '—'} | {_fmt(n_med) if n_med else '—'} "
            f"| {per_item} | {ratio} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("baseline", type=Path, help="baseline pytest-benchmark JSON")
    parser.add_argument("current", type=Path, help="pytest-benchmark JSON of the new run")
    args = parser.parse_args()
    print(compare(args.baseline, args.current))  # noqa: T201


if __name__ == "__main__":
    main()
