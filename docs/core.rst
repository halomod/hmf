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
:mod:`hmf`. It contains the conventions the rest of the core is built on, and the
stages and models ported so far (below).

Conventions
-----------

Units (:mod:`hmf.core.units`)
    Every public input and output that has a dimension is an
    :class:`astropy.units.Quantity`; a bare number for a dimensional input raises
    :class:`~hmf.core.units.UnitBoundaryError`. Outputs are in h-units, written
    explicitly with :data:`astropy.cosmology.units.littleh` (e.g.
    :data:`~hmf.core.units.Msun_h`). Inputs may be in h-units or in physical units,
    which are converted with the object's own H0. Dimensionless inputs and outputs
    (z, sigma, peak height, T(k), ...) are plain numbers and arrays. The
    :func:`~hmf.core.units.unit_boundary` decorator implements this at a fixed cost
    of under 2 µs per call, and every conversion, inside or outside it, goes through
    :func:`~hmf.core.units.to_canonical`, so the errors are the same everywhere.

    * **H0.** An object with a cosmology (a :class:`~hmf.core.transfer.Transfer`, a
      :class:`~hmf.core.power_source.TabulatedPower` given an ``H0``, ...) converts
      physical units with its H0. Models have no cosmology, so they accept h-units
      only, except in methods that take an ``H0`` argument (e.g.
      :meth:`FittingFunction.modify_dndm <hmf.core.fits.FittingFunction.modify_dndm>`).
      An H0 is always a scalar Quantity in km/s/Mpc
      (e.g. ``70 * hmf.core.units.H0_unit``).
    * **Model fields with units** (e.g. ``CAMB.k_max``) take a Quantity, and are
      stored and read back as a Quantity in the canonical unit
      (:func:`~hmf.core.units.quantity_field`), so ``attrs.evolve`` round-trips
      them; they compare and hash by that value. Numeric parameters are stored as
      floats, so ``SMT(a=1)`` and ``SMT(a=1.0)`` are the same model, with the same
      content hash.
    * **The scalar rule.** A scalar input gives a scalar output: a 0-d Quantity if
      the output has a dimension, a :class:`numpy.float64` (which is also a
      :class:`float`) if it is dimensionless. Arrays keep their shape.

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

Fitting functions (:mod:`hmf.core.fits`)
    Every hmf 3.x halo mass function fit is a
    :class:`~hmf.core.fits.FittingFunction` model, registered under its 3.x name
    (``FittingFunction.get("Tinker08")``), with the 3.x parameter names and
    defaults. :meth:`~hmf.core.fits.FittingFunction.fsigma` takes already-resolved
    inputs (sigma, z, Omega_m(z), the overdensity relative to the mean, delta_c,
    n_eff, and the mass m as a Quantity), never a cosmology or a mass-definition
    object; each fit lists the inputs it needs. Code that already holds plain
    arrays in canonical units, such as a :class:`~hmf.core.stage.Stage`, passes them
    in a :class:`~hmf.core.fits.FitInputs` instead. Each declares a valid domain (where ``fsigma`` always
    raises outside) and a calibration domain taken from its paper, with the source
    cited; :func:`~hmf.core.fits.evaluate_fsigma` applies a domain policy to the
    latter. Each also records, as metadata, the mass definition it was measured in and
    the simulations it was calibrated on
    (:class:`~hmf.core.fits.SimulationDetails`).

Logarithms always name their base: ``log10_...`` or ``ln_...``, never ``log``.

Transfer and growth
-------------------
:class:`~hmf.core.transfer.Transfer` (a function of k) gives the transfer function
T(k) and the unnormalised power spectrum :math:`k^{n_s} T^2` at z = 0 (a plain,
dimensionless array with an arbitrary amplitude, which a later stage normalises), and
:class:`~hmf.core.growth.Growth` (a function of z) the growth factor and growth rate,
for each matter species: ``"cb"`` (CDM + baryons) and ``"tot"`` (total matter,
including massive neutrinos). Their models are in
:mod:`~hmf.core.transfer_models` (CAMB, CLASS, EH, BBKS, BondEfs, tables) and
:mod:`~hmf.core.growth_models` (the growth ODE, the integral form and its closed
forms, GenMF, Carroll et al., CAMB, CLASS)::

    from hmf.core.growth import Growth
    from hmf.core.transfer import Transfer
    from hmf.core.units import h_Mpc

    transfer = Transfer(model="CAMB")  # Planck18 by default
    transfer.transfer_function([0.1, 1.0] * h_Mpc, species="tot")
    growth = Growth.from_transfer(transfer, model="CAMB")
    growth.growth_factor([0.0, 1.0, 2.0])

* **One Boltzmann run.** A CAMB or CLASS run computes every species, and the growth
  factor at k = 0.01/Mpc, at once. Runs are memoised by a content hash of their
  input, so choosing different species for P(k) and for sigma_8, changing ``n_s``,
  or a growth stage built from the transfer stage, never runs the code again.
* **Smooth extrapolation.** Above the code's largest wavenumber (``k_max``, 20 h/Mpc
  by default), T(k) follows the EH98 shape, matched to the table in value *and*
  logarithmic slope, so d ln T / d ln k is continuous at the join.
* **Disk cache.** With ``disk_cache=True`` (or a
  :class:`~hmf.core.cache.DiskCache`), runs are also kept on disk (by default in
  ``$HMF_CACHE_DIR``, or the platform's user cache directory, e.g. ``~/.cache/hmf``),
  keyed by the run's input and the versions of hmf and of the code.

Mass variance
-------------

:class:`~hmf.core.mass_variance.MassVariance` gives the mass
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
The modules are listed in the :doc:`hmf.core API reference <api_core>`.
