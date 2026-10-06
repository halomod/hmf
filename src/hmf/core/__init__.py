"""The hmf v4 core: an experimental preview.

.. warning::

   :mod:`hmf.core` is an **experimental preview** of the hmf 4.0 API. Its API may
   change, without deprecation, before 4.0. The v3 API (everything else in
   :mod:`hmf`) is unaffected by it, and ``import hmf`` does not import it.

Importing it emits a :class:`~hmf.exceptions.HMFCoreExperimentalWarning` (a
:class:`FutureWarning`). To silence it::

    import warnings
    from hmf.exceptions import HMFCoreExperimentalWarning

    warnings.simplefilter("ignore", HMFCoreExperimentalWarning)
    import hmf.core

Its modules are listed, grouped by role, on the "hmf.core API" page of the
documentation; the "hmf.core (experimental)" page describes the conventions they
follow.
"""

import warnings

from ..exceptions import HMFCoreExperimentalWarning

warnings.warn(
    "hmf.core is an experimental preview of the hmf 4.0 API, and may change before "
    "4.0. Silence this warning with "
    "warnings.simplefilter('ignore', hmf.exceptions.HMFCoreExperimentalWarning).",
    HMFCoreExperimentalWarning,
    stacklevel=2,
)

from . import (  # noqa: E402
    accuracy,
    domain,
    filters,
    fits,
    mass_variance,
    model,
    power_source,
    stage,
    units,
)
from ._fields import FieldInfo, field  # noqa: E402

__all__ = [
    "FieldInfo",
    "HMFCoreExperimentalWarning",
    "accuracy",
    "domain",
    "field",
    "filters",
    "fits",
    "mass_variance",
    "model",
    "power_source",
    "stage",
    "units",
]
