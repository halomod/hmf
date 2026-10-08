"""``import hmf.core``: the experimental warning, and isolation from ``import hmf``."""

import subprocess
import sys


def _run(code: str) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter, with every warning shown."""
    return subprocess.run(
        [sys.executable, "-W", "default", "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )


def test_import_hmf_does_not_import_core():
    out = _run("import sys, hmf; print('hmf.core' in sys.modules)")
    assert out.stdout.strip() == "False"
    assert "HMFCoreExperimentalWarning" not in out.stderr


def test_import_core_warns_once():
    code = """
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    import hmf.core
    import hmf.core
    from hmf.core import units
from hmf.exceptions import HMFCoreExperimentalWarning
caught = [x for x in w if issubclass(x.category, HMFCoreExperimentalWarning)]
print(len(caught), issubclass(HMFCoreExperimentalWarning, FutureWarning))
"""
    assert _run(code).stdout.split() == ["1", "True"]


def test_core_warning_is_filterable_before_import():
    code = """
import warnings
from hmf.exceptions import HMFCoreExperimentalWarning
warnings.simplefilter("error")
warnings.simplefilter("ignore", HMFCoreExperimentalWarning)
import hmf.core
print("ok")
"""
    assert _run(code).stdout.strip() == "ok"


def test_core_warning_reexported():
    import hmf.core
    from hmf.exceptions import HMFCoreExperimentalWarning

    assert hmf.core.HMFCoreExperimentalWarning is HMFCoreExperimentalWarning


def test_every_public_module_is_imported_and_listed():
    """``import hmf.core`` alone gives every public module, e.g. ``hmf.core.transfer``.

    In a fresh interpreter, so that no other test has imported the submodules.
    """
    code = """
import pkgutil, sys
import hmf.core
names = sorted(m.name for m in pkgutil.iter_modules(hmf.core.__path__) if m.name[0] != "_")
print(" ".join(names))
print(all(n in hmf.core.__all__ and sys.modules["hmf.core." + n] is getattr(hmf.core, n)
          for n in names))
"""
    names, ok = _run(code).stdout.splitlines()
    assert {"transfer", "growth", "transfer_models", "growth_models", "species"} <= set(
        names.split()
    )
    assert ok == "True"
