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

This package currently holds the conventions the rest of the v4 core builds on, and
the first ported models:

* :mod:`hmf.core.units`: the units boundary between public methods and kernels;
* :mod:`hmf.core.model`: the :class:`~hmf.core.model.Model` base class and the
  per-kind model registry;
* :mod:`hmf.core.stage`: the :class:`~hmf.core.stage.Stage` base class;
* :mod:`hmf.core.accuracy`: typed accuracy settings for the internal grids;
* :mod:`hmf.core.domain`: model domains and the domain policy;
* :mod:`hmf.core.fits`: the halo mass function fitting functions, as models;
* :mod:`hmf.core._kernels`: the conventions for unit-free numerical kernels;
* :func:`field`: an ``attrs`` field with documentation, for models and stages.

See the "hmf.core (experimental)" page of the documentation for an overview.
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

from . import accuracy, domain, fits, model, stage, units  # noqa: E402
from ._fields import FieldInfo, field  # noqa: E402

__all__ = [
    "FieldInfo",
    "HMFCoreExperimentalWarning",
    "accuracy",
    "domain",
    "field",
    "fits",
    "model",
    "stage",
    "units",
]
