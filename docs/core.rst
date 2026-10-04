hmf.core (experimental)
=======================

.. warning::

   :mod:`hmf.core` is an **experimental preview** of the hmf 4.0 API. It may change,
   without deprecation, before 4.0. Importing it emits a
   :class:`~hmf.exceptions.HMFCoreExperimentalWarning` (a :class:`FutureWarning`),
   which you can silence with::

       import warnings
       from hmf.exceptions import HMFCoreExperimentalWarning

       warnings.simplefilter("ignore", HMFCoreExperimentalWarning)

   The v3 API is unaffected, and ``import hmf`` does not import :mod:`hmf.core`.

The v4 core is developed alongside the v3 API, as the subpackage :mod:`hmf.core`
(see `#396 <https://github.com/halomod/hmf/issues/396>`_). At 4.0 it becomes
:mod:`hmf`. So far it contains only the conventions the rest of the core will be
built on; no physics has been ported to it yet.

Conventions
-----------

Units (:mod:`hmf.core.units`)
    Every public input and output that has a dimension is an
    :class:`astropy.units.Quantity`; a bare number for a dimensional input raises
    :class:`~hmf.core.units.UnitBoundaryError`. Outputs are in h-units, written
    explicitly with :data:`astropy.cosmology.units.littleh` (e.g.
    :data:`~hmf.core.units.Msun_h`). Inputs may be in h-units or in physical units,
    which are converted with the object's own H0. Dimensionless inputs (z, sigma,
    peak height, ...) are plain numbers. The
    :func:`~hmf.core.units.unit_boundary` decorator implements this at a fixed cost
    of under 2 µs per call.

Kernels (:mod:`hmf.core._kernels`)
    The numerics are pure functions on plain arrays, in the canonical unit of each
    physical kind (:data:`~hmf.core.units.CANONICAL_UNITS`). They take no
    Quantities, do not loop over array elements in Python, never modify their
    inputs, and give results that do not depend on the batch size.

Models (:mod:`hmf.core.model`)
    A model (a fitting function, a transfer function, ...) is a frozen,
    keyword-only ``attrs`` class whose fields are its parameters. Each belongs to a
    *kind* with its own registry, and is registered under its qualified name
    ``package.module:Class`` and an optional unique alias. Lookup is per kind
    (``Kind.get(name)``), and also finds models by import path and through the
    ``hmf.models`` entry-point group.

Stages (:mod:`hmf.core.stage`)
    A stage is one immutable step of a calculation. Parameters change through
    ``evolve()``, which returns a new stage; expensive results are cached with
    :func:`functools.cached_property`. ``fields_info()`` describes the parameters
    without creating an instance.

Accuracy (:mod:`hmf.core.accuracy`)
    The internal mass and wavenumber grids are set by
    :class:`~hmf.core.accuracy.MassAccuracy` and
    :class:`~hmf.core.accuracy.KAccuracy`, each with ``fast()`` and ``high()``
    presets.

Domains (:mod:`hmf.core.domain`)
    Models will declare a *valid* domain (outside which they always raise) and a
    *calibration* domain (outside which a user-chosen
    :data:`~hmf.core.domain.DomainPolicy` applies: ``"ignore"``, ``"warn"``,
    ``"mask"`` or ``"raise"``).

Logarithms always name their base: ``log10_...`` or ``ln_...``, never ``log``.

Mass variance
-------------

:class:`~hmf.core.mass_variance.MassVariance` is the first physics stage: the mass
variance σ(M) of the *unnormalised* linear power at z = 0, and its slope
dlnσ/dlnM, for a smoothing filter (:mod:`hmf.core.filters`: ``TopHat``, ``SharpK``,
``SmoothK``). The power comes from any :class:`~hmf.core.power_source.PowerSource`,
e.g. a :class:`~hmf.core.power_source.TabulatedPower`::

    from hmf.core.mass_variance import MassVariance
    from hmf.core.power_source import TabulatedPower
    from hmf.core.units import Msun_h, h_Mpc, power_unit, rho_unit

    source = TabulatedPower(k=k * h_Mpc, pk=pk * power_unit, mean_density=rho * rho_unit)
    mv = MassVariance(power=source, filter="SharpK")
    mv.sigma(m * Msun_h), mv.dlnsigma_dlnm(m * Msun_h), mv.m_from_sigma(0.5)

Both are interpolated on a lattice of nodes at :math:`\log_{10} m = j\Delta`, built
lazily: results do not depend, bit for bit, on the order or batching of requests.
σ uses a quintic Hermite interpolant in ln–ln; dlnσ/dlnM is interpolated
separately, never by differentiating σ. The k grid is a lattice in ln k, fixed by
the accuracy settings. Masses whose integrals the grid can't resolve (truncated at
either end, or aliased) raise rather than return a wrong value.

API
---
The modules are listed in the :doc:`API reference <api>`, under "hmf.core
(experimental)".
