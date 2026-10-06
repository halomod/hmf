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
stages and models.

**Conventions:**

* :mod:`hmf.core.units`: the units boundary between public methods and kernels;
* :mod:`hmf.core.model`: the :class:`~hmf.core.model.Model` base class and the
  per-kind model registry;
* :mod:`hmf.core.stage`: the :class:`~hmf.core.stage.Stage` base class;
* :mod:`hmf.core.accuracy`: typed accuracy settings for the internal grids;
* :mod:`hmf.core.domain`: model domains and the domain policy;
* :func:`field`: an ``attrs`` field with documentation, for models and stages.

**Stages:**

* :mod:`hmf.core.transfer`: the :class:`~hmf.core.transfer.Transfer` stage, T(k) for
  every matter species from one run of the transfer model;
* :mod:`hmf.core.growth`: the :class:`~hmf.core.growth.Growth` stage, D(z);
* :mod:`hmf.core.power_source`: sources of the linear power spectrum at z = 0;
* :mod:`hmf.core.mass_variance`: the :class:`~hmf.core.mass_variance.MassVariance`
  stage, sigma(m) on a deterministic mass lattice.

**Models:**

* :mod:`hmf.core.transfer_models`: transfer functions (the
  :class:`~hmf.core.transfer_models.TransferModel` kind);
* :mod:`hmf.core.growth_models`: growth factors (the
  :class:`~hmf.core.growth_models.GrowthModel` kind);
* :mod:`hmf.core.filters`: smoothing filters (the :class:`~hmf.core.filters.Filter`
  kind);
* :mod:`hmf.core.fits`: halo mass function fitting functions.

**Utilities:** :mod:`hmf.core.cache` (the on-disk cache of Boltzmann-code output),
and :mod:`hmf.core._kernels` (the conventions for the unit-free numerical kernels).

See the "hmf.core (experimental)" page of the documentation for an overview, and the
"hmf.core API" page for the full reference.
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
