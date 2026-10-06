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

This package holds the conventions the rest of the v4 core builds on, and the first
stages and models:

* :mod:`hmf.core.units`: the units boundary between public methods and kernels;
* :mod:`hmf.core.model`: the :class:`~hmf.core.model.Model` base class and the
  per-kind model registry;
* :mod:`hmf.core.stage`: the :class:`~hmf.core.stage.Stage` base class;
* :mod:`hmf.core.accuracy`: typed accuracy settings for the internal grids;
* :mod:`hmf.core.domain`: model domains and the domain policy;
* :mod:`hmf.core.fits`: the halo mass function fitting functions, as models;
* :mod:`hmf.core._kernels`: the conventions for unit-free numerical kernels;
* :mod:`hmf.core.filters`: the smoothing filters (the :class:`~hmf.core.filters.Filter` kind);
* :mod:`hmf.core.power_source`: sources of the linear power spectrum at z = 0;
* :mod:`hmf.core.mass_variance`: the :class:`~hmf.core.mass_variance.MassVariance`
  stage: sigma(m) on a deterministic mass lattice;
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

from . import (  # noqa: E402  # noqa: E402
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
