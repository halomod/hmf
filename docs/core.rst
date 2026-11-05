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
    :func:`functools.cached_property`. ``evolve()`` and ``from_flat()`` take the
    parameters of the whole stage tree by name, through a routing table built from
    the classes (:mod:`hmf.core.routing`, see `Changing parameters`_), and
    ``fields_info()``, ``parameter_info()``, ``quantities_available()`` and
    ``invalidated_by()`` describe a stage tree without creating an instance. A stage computed from a cosmology
    (:class:`~hmf.core.transfer.Transfer`, :class:`~hmf.core.growth.Growth`)
    subclasses :class:`~hmf.core.stage.CosmologyStage`, which holds the
    ``cosmology`` field and converts physical units with its H0.

Accuracy (:mod:`hmf.core.accuracy`)
    The internal mass and wavenumber grids are set by
    :class:`~hmf.core.accuracy.MassAccuracy` and
    :class:`~hmf.core.accuracy.KAccuracy`, each with ``fast()`` and ``high()``
    presets. A stage's accuracy fields are named for their grid:
    ``Transfer.k_accuracy``, ``Growth.k_accuracy``, ``MassVariance.mass_accuracy``
    and ``MassVariance.k_accuracy``.

    **One KAccuracy drives the k-space calculation.** Each stage reads only some of
    its settings:

    * :class:`~hmf.core.transfer.Transfer`: ``dln_k``, which sets the precision and
      k sampling of a CAMB or CLASS run (finer than the code's own below
      ``dln_k = 0.02``, so ``KAccuracy.high()`` runs the codes at high precision);
    * :class:`~hmf.core.growth.Growth`: ``dln_k``, in the same way, for a run its
      CAMB or CLASS growth model makes itself;
    * :class:`~hmf.core.mass_variance.MassVariance`: ``dln_k``, ``ln_k_min`` and
      ``k_max_r_min``, the k grid of the σ integrals.

    A stage composed of these passes one ``KAccuracy`` to all of them
    (:class:`~hmf.core.linear_power.LinearPower` owns it), and checks it with
    :func:`~hmf.core.accuracy.check_consistent`.
    :meth:`Growth.from_transfer <hmf.core.growth.Growth.from_transfer>` takes the
    transfer stage's, and a growth stage shares its transfer stage's Boltzmann run
    only when their accuracies are equal.

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
    narrow it). A stage whose domain depends on its own data has a ``valid_domain``
    of its own, on the instance: :class:`~hmf.core.mass_variance.MassVariance` (masses,
    filter radii and σ; masses are bounded by the lattice with
    ``extension="raise"``) and :class:`~hmf.core.power_source.TabulatedPower` (k; the
    table's range with ``extension="raise"``). So every stage's domain can be
    inspected the same way, and every public method checks its input with
    :meth:`~hmf.core.domain.Domain.check`.

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
    * :class:`TypeError`: an object of the wrong kind, or a parameter name that is
      not one: unknown (with the closest names), ambiguous (with the dotted paths
      to choose from), derived, given twice, or a required one that is missing.

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
    in a :class:`~hmf.core.fits.FitInputs` instead, built by
    :meth:`FittingFunction.inputs_from_arrays
    <hmf.core.fits.FittingFunction.inputs_from_arrays>`. Each declares a valid domain (where ``fsigma`` always
    raises outside) and a calibration domain taken from its paper, with the source
    cited; :func:`~hmf.core.fits.evaluate_fsigma` applies a domain policy to the
    latter. Each also records, as metadata, the mass definition it was measured in and
    the simulations it was calibrated on
    (:class:`~hmf.core.fits.SimulationDetails`). See `Mass definitions and conversion`_.

Logarithms always name their base: ``log10_...`` or ``ln_...``, never ``log``.

Transfer and growth
-------------------
:class:`~hmf.core.transfer.Transfer` (a function of k) gives the transfer function
T(k) and the unnormalised power spectrum :math:`k^{n_s} T^2` at z = 0 (a plain,
dimensionless array with an arbitrary amplitude, which
:class:`~hmf.core.linear_power.LinearPower` normalises), and
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
(:meth:`~hmf.core.transfer.Transfer.power_source`), or a
:class:`~hmf.core.power_source.TabulatedPower`. A power source also carries the mean
density of CDM + baryons, :math:`\bar\rho_{\rm cb}`, whatever its species: it sets
the mass of a filter radius, :math:`M = \tfrac{4\pi}{3}\bar\rho_{\rm cb}(cR)^3`,
as in hmf 3.x and the calibrations of the fits, so the total-matter and the
CDM + baryon power give the same R(M)::

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
either end, or aliased) raise rather than return a wrong value. σ at any filter
radius can also be evaluated directly, without the lattice
(:meth:`~hmf.core.mass_variance.MassVariance.ln_sigma_at_radius_kernel`, e.g. σ8),
and agrees with the lattice to
:data:`~hmf.core.mass_variance.INTERPOLATION_RTOL` at the default settings.

To use your own power spectrum, implement the
:class:`~hmf.core.power_source.PowerSource` protocol: ``ln_power_kernel``,
``rho_mean0``, ``table_range``, and ``_unit_context``, the
:class:`~hmf.core.units.UnitContext` (with its H0) of the stages built on the
source. ``_unit_context`` comes from :class:`~hmf.core.units.HasUnitContext` and is
required, despite its leading underscore: every
:func:`~hmf.core.units.unit_boundary` method reads it by that name on its fast path,
and the underscore keeps it out of a stage's public outputs, since it is a detail of
the units machinery rather than a result. Provide it as a
:func:`functools.cached_property`::

    @functools.cached_property
    def _unit_context(self) -> UnitContext:
        return UnitContext(self.H0)

Composing stages
----------------
:class:`~hmf.core.mass_function.MassFunction` combines the stages above and a fitting
function into the halo mass function, as a function of mass and redshift. Each stage
holds the stages it is computed from as fields, so together they form a *stage tree*.
A stage computed from the fields above it, rather than given, is a *derived* stage
(:func:`~hmf.core.stage.derived_property`)::

    MassFunction
    ├── linear_power: LinearPower    σ8 normalisation and P(k, z)
    │   ├── transfer: Transfer       T(k), from one Boltzmann-code run
    │   └── growth (derived): Growth the growth factor D(z), of growth_model
    ├── variance (derived): MassVariance  σ(m) of the power before normalisation
    └── fit: FittingFunction         f(σ), e.g. Tinker08

plus the variance's ``filter``, ``mass_accuracy`` and ``truncation_rtol``, the
critical overdensity ``delta_c`` and the ``domain_policy`` as fields of
``MassFunction``. :meth:`MassFunction.build <hmf.core.mass_function.MassFunction.build>`
builds the whole tree, with hmf 3.x-like defaults (Planck18, CAMB, the growth ODE, σ8
and n_s from the cosmology, Tinker08, a top-hat filter, and the CDM + baryon power
normalised by the σ8 of total matter). Every method takes its arguments by keyword::

    import numpy as np
    from hmf.core.mass_function import MassFunction
    from hmf.core.units import Msun_h

    mf = MassFunction.build(fit="Tinker08")
    m = np.logspace(10, 15, 51) * Msun_h
    z = np.array([0.0, 0.5, 1.0])
    dndm = mf.dndm(m=m[None, :], z=z[:, None])  # shape (3, 51), in h^4 / (Msun Mpc^3)
    ngtm = mf.ngtm(m=m, z=0.0)
    m_star = mf.m_from_peak_height(peak_height=1.0, z=0.0)

    view = mf.at(z=1.0, m=m)  # the quantities at z = 1, as arrays: view.dndm, view.ngtm

Changing a parameter gives a new stage, with ``evolve``; the old one is unchanged::

    lower = mf.evolve(sigma_8=0.75)
    lower.variance is mf.variance  # True: the mass variance is reused, not recomputed

What each part does:

* :class:`~hmf.core.linear_power.LinearPower` normalises the power spectrum to σ8:
  P(k, z) = A D²(z) P_raw(k), where P_raw is the transfer stage's k^n_s T²(k) and the
  amplitude A = (σ8/σ8,raw)² is a single number (σ8,raw is the rms of P_raw in a
  sphere of 8 Mpc/h). Its growth stage is built from its transfer stage, so the
  cosmology, the one :class:`~hmf.core.accuracy.KAccuracy` of the tree and the disk
  cache are the transfer stage's, held in one place.
* The :class:`~hmf.core.mass_variance.MassVariance` is computed from P_raw, the power
  *before* it is normalised and scaled to a redshift: σ8 and z only multiply it, by
  √A D(z). So changing σ8, z, the fit, δc or the domain policy reuses the mass
  variance already computed (the expensive part of the calculation) and runs no
  Boltzmann code again.
* The fit is evaluated in its own mass definition (e.g. SO-mean(200) for Tinker08,
  FoF for Jenkins), unless ``mass_definition`` asks for another; see
  `Mass definitions and conversion`_.
* n(>m) and ρ(>m) are integrated from 10^17.5 M☉/h down, over a fixed grid of masses,
  so the value at a mass does not depend on which other masses are asked for with
  it.
* Outside the fit's valid domain every method raises; outside the range it was
  calibrated on, ``domain_policy`` (``"ignore"``, ``"warn"``, ``"mask"`` or ``"raise"``)
  decides what happens at each mass and redshift asked for.

Mass definitions and conversion
-------------------------------
A halo's mass depends on where its edge is put. :mod:`hmf.core.mass_definitions`
gives the definitions as frozen ``attrs`` values, which compare (symmetrically) and
hash by class and value: ``SOMean(overdensity=200)``, ``SOCritical(overdensity=500)``,
``SOVirial()`` (Bryan & Norman 1998), ``FOF(linking_length=0.2)``, ``SelfBound()``,
and ``SOGeneric(preferred=...)``, the definition of a fit that takes the overdensity
as an input. Each has a short name (``"200m"``, ``"500c"``, ``"vir"``, ``"fof"``)
that is accepted wherever a definition is. Overdensities are relative to the mean
density of CDM + baryons, and radii are comoving.

Every fit records the definition it was measured in
(``FittingFunction.measured_mass_definition``), and
``measured.accepts(definition)`` says whether it can be evaluated in another directly:
every definition accepts itself, and an ``SOGeneric`` (Tinker08, Tinker10, Watson)
accepts every SO definition. A :class:`~hmf.core.mass_function.MassFunction` takes
the definition its masses are in as ``mass_definition`` (``None``, the default, is the
fit's own). When the fit does not accept it, the masses are converted by
``mass_conversion``, which must be given: the conversion is opt-in::

    from hmf.core.mass_conversion import ProfileConversion
    from hmf.core.concentration import Duffy08

    mf = MassFunction.build(fit="SMT")         # in SMT's own, virial, definition
    mf.evolve(mass_definition="200c")          # ValueError: needs a mass_conversion
    mf200c = mf.evolve(mass_definition="200c", mass_conversion=ProfileConversion())
    mf.evolve(fit="Tinker08", mass_definition="200c")   # accepted: no conversion
    mf200c.evolve(**{"mass_conversion.concentration": Duffy08(sample="relaxed")})

:class:`~hmf.core.mass_conversion.ProfileConversion` takes haloes of a
:class:`~hmf.core.profiles.DensityProfile` (:class:`~hmf.core.profiles.NFW` by
default; another profile, e.g. Einasto, is another ``DensityProfile``) with the
concentrations of a :class:`~hmf.core.concentration.ConcentrationModel` (a model kind:
:class:`~hmf.core.concentration.Duffy08` by default, with every parameter set of their
Table 1), converted to the definition of the masses if it was calibrated in another.
The profile solves for the new concentration of every mass at once (for NFW, Newton's
method from the approximation of Hu & Kravtsov 2003, to round-off), and gives the Jacobian dM'/dM
analytically. The mass function then conserves halo number:
dn/dM(M) = dn'/dM'(M'(M)) dM'/dM and n(>M) = n'(>M'(M)), with the fit evaluated at
the masses M' in its own definition. Converting to the same definition is the
identity, and A → B → A returns the masses to round-off. The domain policy also
applies to the concentration model's calibration domain. A profile gives only what
the conversion needs, its enclosed mass :math:`\mu(x)`; halomod's profiles, with
their Fourier transforms, build on these.

Changing parameters
-------------------
A stage tree's parameters are routed by a table that :mod:`hmf.core.routing` builds
once per stage class, from the ``attrs`` fields of the classes: it creates no stage
and computes nothing. :meth:`~hmf.core.stage.Stage.evolve`,
:meth:`~hmf.core.stage.Stage.from_flat` and
:meth:`MassFunction.build <hmf.core.mass_function.MassFunction.build>` take the same
names, of three kinds:

* **Flat names**, for every field whose name appears once in the tree:
  ``cosmology``, ``n_s``, ``k_accuracy``, ``disk_cache``, ``growth_model``,
  ``sigma_8``, ``species``, ``sigma_8_species``, ``filter``, ``mass_accuracy``,
  ``truncation_rtol``, ``fit``, ``delta_c``, ``domain_policy``, ``mass_definition``,
  ``mass_conversion``. More generally, any
  dotted suffix of a field's path that ends no other field's path names it:
  ``"transfer.model"`` is the transfer model however many other stages have a
  ``model``. A stage lists the fields whose names are too generic to stand alone
  in ``qualified_parameters``; :meth:`~hmf.core.stage.Stage.parameter_names` gives
  them with the stage's name, so the transfer model is listed as
  ``"transfer.model"`` (a bare ``model`` still works while it is unique).
* **Shared parameters**: a name that a stage lists in ``shared_parameters`` is one
  parameter in its tree, and a change to it changes every field of that name below
  the stage, so that they stay equal. A :class:`~hmf.core.growth.Growth` stage with a
  transfer stage shares ``cosmology``, ``k_accuracy`` and ``disk_cache`` with it. In a
  mass function each of them is held once, by the transfer stage.
* **Dotted paths**, always: ``"linear_power.sigma_8"``,
  ``"linear_power.transfer.model"``; a suffix that ends several paths is ambiguous,
  and its error lists the names to choose from. A stage's path replaces the whole stage
  (``mf.evolve(linear_power=lp)``), and a path into a model field changes that field
  of the model (``"fit.a"``, ``"transfer.model.z_max"``).

So the flat names of a mass function are exactly :meth:`build
<hmf.core.mass_function.MassFunction.build>`'s keywords::

    mf = MassFunction.build(**{"transfer.model": "EH"}, fit="ST", sigma_8=0.8)
    mf.evolve(sigma_8=0.85, n_s=0.95)
    mf.evolve(cosmology=Planck15)              # the cosmology of the whole tree
    mf.evolve(**{"fit.a": 0.75})               # a parameter of the fit
    mf.evolve(**{"linear_power.transfer.n_s": 0.95})
    MassFunction.from_flat({"transfer.model": "EH", "linear_power": {"sigma_8": 0.8}})

A name that is not a parameter raises a :class:`TypeError` at once, before anything
is built, with the closest names (``mf.evolve(sigma8=0.8)``: "Did you mean
'sigma_8'?"); an ambiguous one lists the dotted paths to choose from.

``evolve`` rebuilds only the stages on the path from each changed field up to the
root, with each stage's own ``evolve_own``; every other stage is *shared*, the same
object, with every result it has cached. A derived stage is computed from fields it
declares (the variance from the linear power's transfer stage and species and the
variance settings; the growth from the transfer stage and ``growth_model``), and
``evolve_own`` carries it, once computed, to the new stage when none of them
changes. So ``mf.evolve(sigma_8=0.85)`` rebuilds the linear power and the mass
function, and keeps the transfer, growth and variance stages;
``mf.evolve(filter="SharpK")`` keeps the linear power; ``mf.evolve(fit="ST")`` keeps
both; ``mf.evolve(**{"transfer.model": "BBKS"})`` gives a new variance, of the new power.
Each new stage is validated by its constructor; if any fails, ``evolve`` raises and
the original tree is unchanged.

Every question about the parameters is answered from the classes, without building
a stage or running a Boltzmann code::

    MassFunction.parameter_names()        # ('cosmology', 'transfer.model', 'n_s', ...)
    MassFunction.parameter_info()["sigma_8"].path    # 'linear_power.sigma_8'
    MassFunction.parameter_defaults()["sigma_8"]     # 0.8102, from Planck18
    MassFunction.quantities_available()[""]          # ('dlnsigma_dlnm', 'dndlnm', ...)
    MassFunction.invalidated_by("sigma_8")           # ('', 'linear_power')
    mf.to_flat()                          # every parameter's value, for from_flat

Extension stages
    A stage that holds other stages (as fields whose type is a
    :class:`~hmf.core.stage.Stage`) gets a routing table of its own, which includes
    the tables of the stages it holds: a stage holding a ``MassFunction`` takes
    ``evolve(sigma_8=...)`` with no further code. It declares

    * the stages it computes from its fields with
      :func:`@derived_property(*inputs) <hmf.core.stage.derived_property>`, naming
      the fields (dotted paths below the stage) each is computed from, so that
      ``evolve`` carries it when none of them changes;
    * the names of its tree that are one parameter in the class attribute
      ``shared_parameters``: the stage that composes the others decides, whatever
      the stages it holds declare;
    * the fields whose names are too generic to stand alone in
      ``qualified_parameters``, listed with the stage's name;
    * and defaults that depend on other parameters (as ``build`` takes σ8 and n_s from
      the cosmology) in the class method ``computed_defaults``.

    A derived property's input that is not a field of the tree raises a
    :class:`TypeError` when the table is first built.
    :func:`~hmf.core.stage.undeclared_inputs` checks, in a test, that each derived
    property declares every field it reads (and every stage it holds): run on a stage
    of the extension, it returns the fields each one reads but does not declare.
    Declaring an input that is read only on some branches is safe.

API
---
The modules are listed in the :doc:`hmf.core API reference <api_core>`.
