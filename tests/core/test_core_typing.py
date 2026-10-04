"""hmf.core is type-checked with mypy in strict mode (configured in pyproject.toml)."""

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mypy")

ROOT = Path(__file__).parents[2]


def test_core_passes_mypy_strict():
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--config-file", "pyproject.toml"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Success" in result.stdout
