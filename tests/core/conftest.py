"""Shared setup of the hmf.core (v4 preview) tests.

``import hmf.core`` emits an :class:`~hmf.exceptions.HMFCoreExperimentalWarning`.
The ``filterwarnings`` entry in ``pyproject.toml`` ignores it, but the repo's
``-Werror`` CI job passes ``-W error`` on the command line, which takes precedence
over ini filters. So import it here, once, with the warning silenced: afterwards
``import hmf.core`` in a test module is a cached no-op and warns nothing.
``test_core_init.py`` checks the warning itself, in a subprocess.
"""

import warnings

from hmf.exceptions import HMFCoreExperimentalWarning

with warnings.catch_warnings():
    warnings.simplefilter("ignore", HMFCoreExperimentalWarning)
    import hmf.core  # noqa: F401
