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

    Stages and models expose them to library code (a stage built on other stages)
    as *kernel-level entry points*: public methods or functions whose names end in
    ``_kernel``, which take and return plain arrays in canonical units, state the
    unit of each argument in their docstrings, and follow the same rules. Library
    code calls these, never the unit-checked public methods.

    Every kernel-level entry point checks its input against its owner's domain and
    raises a :class:`~hmf.core.domain.DomainError` outside it (NaN and infinities
    included), so its caller never gets a silent extrapolation. The check is cheap:
    it compares only the smallest and largest value with the bounds
    (:func:`~hmf.core.domain.check_extent`). The pure functions inside
    :mod:`hmf.core._kernels` do not check: the entry points that call them do.

    Read-only data holders (``Transfer.solution``, ``Growth.solution``, a power
    source's ``rho_mean0``) keep plain names and do not check their input:
    ``Growth.solution`` is NaN beyond its table. Library code goes through the
    entry points (e.g. ``Growth.growth_factor_kernel``).

Models (:mod:`hmf.core.model`)
    A model (a fitting function, a transfer function, ...) is a frozen,
    keyword-only ``attrs`` class whose fields are its parameters. Each belongs to a
    *kind* with its own registry, and is registered under its qualified name
    ``package.module:Class`` and an optional unique alias. Lookup is per kind
    (``Kind.get(name)``), and also finds models by import path and through the
    ``hmf.models`` entry-point group. ``Kind.coerce(value)`` turns an instance, a
    name or a class into an instance of the kind: it is the converter of every
    stage's model fields.

Stages (:mod:`hmf.core.stage`)
    A stage is one immutable step of a calculation. Parameters change through
    ``evolve()``, which returns a new stage; expensive results are cached with
    :func:`functools.cached_property`. ``fields_info()`` describes the parameters
    without creating an instance. A stage computed from a cosmology
    (:class:`~hmf.core.transfer.Transfer`, :class:`~hmf.core.growth.Growth`)
    subclasses :class:`~hmf.core.stage.CosmologyStage`, which holds the
    ``cosmology`` field and converts physical units with its H0.

Accuracy (:mod:`hmf.core.accuracy`)
    The internal mass and wavenumber grids are set by
    :class:`~hmf.core.accuracy.MassAccuracy` and
    :class:`~hmf.core.accuracy.KAccuracy`, each with ``fast()`` and ``high()``
    presets. A stage's accuracy fields are named for their grid:
    ``Transfer.k_accuracy``, ``MassVariance.mass_accuracy`` and
    ``MassVariance.k_accuracy``.

Domains (:mod:`hmf.core.domain`)
    Every model declares a *valid* domain (outside which it always raises a
    :class:`~hmf.core.domain.DomainError`) and a *calibration* domain (outside which
    a user-chosen :data:`~hmf.core.domain.DomainPolicy` applies: ``"ignore"``,
    ``"warn"``, ``"mask"`` or ``"raise"``), or ``None`` for no calibration domain. A
    :class:`~hmf.core.domain.Domain` bounds each variable by an interval whose ends
    are closed (``>=``, ``<=``) or open (``>``, ``<``): ``(0, None, "(]")`` is
    ``> 0``. :meth:`~hmf.core.domain.Domain.describe` writes it as inequalities, and
    :meth:`~hmf.core.domain.Domain.check` raises a
    :class:`~hmf.core.domain.DomainError` that quotes them. Stages check their inputs
    against their model's valid domain (that of the model's class, so a model can
    narrow it).

Out-of-range switches
    Two settings say what to do with values out of range, and they are distinct:

    * :data:`~hmf.core.accuracy.Extension` (``"auto"`` or ``"raise"``) is about the
      *numerics*: what to do with a request outside an internal grid or a table.
      ``MassAccuracy.extension="auto"`` extends the mass lattice lazily;
      ``TabulatedPower.extension="auto"`` extrapolates the table as a power law. With
      ``"raise"`` both raise a :class:`~hmf.core.domain.DomainError`.
    * :data:`~hmf.core.domain.DomainPolicy` is about the *physics*: what to do with
      a result outside a model's calibration domain, where it is an extrapolation of
      the fit.

Errors
    Every error about an input is one of three types:

    * :class:`~hmf.core.domain.DomainError` (a :class:`ValueError`): an input
      *value* is outside what can be evaluated. Non-positive, NaN or infinite k, m or
      sigma; a mass outside the lattice with ``extension="raise"``; k outside a table
      that is not extrapolated; z beyond a growth table; anything outside a model's
      valid domain. Infinite values are outside every domain, so an infinite z or
      ``delta_halo`` raises a :class:`~hmf.core.domain.DomainError` too.
    * :class:`ValueError`: a bad option, configuration or combination, or a missing
      input. A model that does not apply to the cosmology it is given (e.g.
      ``Eisenstein97Growth`` with a non-flat cosmology) is a configuration error.
    * :class:`TypeError`: an object of the wrong kind.

    So ``except ValueError`` catches every bad input, and ``except DomainError`` only
    values out of range.

Extrapolation warnings
    Extrapolating beyond a table *you* supplied (a
    :class:`~hmf.core.power_source.TabulatedPower`, or a ``FromArray`` or
    ``FromFile`` transfer model) emits an
    :class:`~hmf.exceptions.HMFExtrapolationWarning`, once per owner and end of the
    table (:func:`~hmf.core.domain.warn_once`). Every path warns through one
    mechanism, :meth:`TableRange.warn_outside
    <hmf.core.power_source.TableRange.warn_outside>`: the public methods of a
    :class:`~hmf.core.transfer.Transfer` stage or a
    :class:`~hmf.core.power_source.TabulatedPower` warn for k outside the table, and
    a :class:`~hmf.core.mass_variance.MassVariance` warns once if its k grid
    reaches beyond its power source's table (its ``table_range``). The ends of the k
    grid are rounded outwards onto the lattice, so a table that reaches the
    unrounded ends (e.g. from exactly 10⁻⁸ h/Mpc) does not warn. A power source's
    ``ln_power_kernel`` itself does not warn. Extrapolation by design is silent: the
    tail of T(k) beyond a Boltzmann code's ``k_max``, or the lazy extension of the
    mass lattice. So the default configurations emit no warnings. Growth tables are
    never extrapolated: z beyond them raises.

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
for each matter species (:mod:`~hmf.core.species`): ``"cb"`` (CDM + baryons) and
``"tot"`` (total matter, including massive neutrinos). Their models are in
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
``SmoothK``). The power comes from any :class:`~hmf.core.power_source.PowerSource`:
the power of one species of a :class:`~hmf.core.transfer.Transfer` stage
(:meth:`~hmf.core.transfer.Transfer.power_source`, which carries the mean density
of the same species), or a :class:`~hmf.core.power_source.TabulatedPower`::

    from hmf.core.mass_variance import MassVariance
    from hmf.core.power_source import TabulatedPower
    from hmf.core.transfer import Transfer
    from hmf.core.units import Msun_h, h_Mpc, power_unit, rho_unit

    mv = MassVariance(power=Transfer(model="EH").power_source("cb"), filter="SharpK")
    mv.sigma(m * Msun_h), mv.dlnsigma_dlnm(m * Msun_h), mv.m_from_sigma(0.5)

    source = TabulatedPower(k=k * h_Mpc, pk=pk * power_unit, mean_density=rho * rho_unit)
    MassVariance(power=source)

Both are interpolated on a lattice of nodes at :math:`\log_{10} m = j\Delta`, built
lazily: results do not depend, bit for bit, on the order or batching of requests.
σ uses a quintic Hermite interpolant in ln–ln; dlnσ/dlnM is interpolated
separately, never by differentiating σ. The k grid is a lattice in ln k, fixed by
``k_accuracy`` (and ``mass_accuracy.log10_m_min``). Masses whose integrals the grid can't resolve (truncated at
either end, or aliased) raise rather than return a wrong value.

Composing stages
----------------
The mass function will be built from these stages and the fits, through their
kernel-level entry points only. Each one checks its input and raises a
:class:`~hmf.core.domain.DomainError` rather than extrapolate silently.

* σ(m, z) = D(z) · √A · σ_raw(m), where σ_raw is the σ of
  :class:`~hmf.core.mass_variance.MassVariance`, the mass variance of the
  *unnormalised* power, and A is the amplitude that fixes the power to σ8. A
  ``LinearPower`` stage will fix A; it does not exist yet. MassVariance keeps the
  unnormalised power, so changing σ8 changes only the scalar A, and recomputes no
  lattice nodes;
* ln σ_raw(m) and dlnσ/dlnm (which A and D do not change) from
  :meth:`MassVariance.ln_sigma_and_slope_kernel
  <hmf.core.mass_variance.MassVariance.ln_sigma_and_slope_kernel>`, and ``n_eff``
  from them with :func:`~hmf.core.mass_variance.n_eff_kernel`;
* the growth factor D(z) from :meth:`Growth.growth_factor_kernel
  <hmf.core.growth.Growth.growth_factor_kernel>` (and the growth rate from
  :meth:`Growth.growth_rate_kernel <hmf.core.growth.Growth.growth_rate_kernel>`),
  which raise beyond the growth model's table, not ``Growth.solution``, which is NaN
  there;
* the mean density from the power source's ``rho_mean0``, and Ω_m(z) of CDM +
  baryons from :func:`hmf.core.species.omega_m` (``"cb"``);
* the overdensity of a fit's mass definition from
  :meth:`MeasuredMassDefinition.delta_halo_mean_kernel
  <hmf.core.fits.MeasuredMassDefinition.delta_halo_mean_kernel>`;
* f(σ) from :meth:`FittingFunction.fsigma_kernel
  <hmf.core.fits.FittingFunction.fsigma_kernel>` (or
  :func:`~hmf.core.fits.evaluate_fsigma`, with a domain policy, whose ``"warn"``
  policy warns once per ``owner``: the mass-function stage), and the fit's mass
  function from :meth:`FittingFunction.modify_dndm_kernel
  <hmf.core.fits.FittingFunction.modify_dndm_kernel>`, which every fit has.

The critical overdensity δc is an open input: it will be a field of the
mass-function stage.

API
---
The modules are listed in the :doc:`hmf.core API reference <api_core>`.
