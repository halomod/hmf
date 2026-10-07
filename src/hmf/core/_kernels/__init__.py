"""Unit-free numerical kernels of the v4 core.

This subpackage holds the numerics behind every public method of :mod:`hmf.core`,
one module per topic (e.g. :mod:`~hmf.core._kernels.transfer`,
:mod:`~hmf.core._kernels.growth`). This docstring sets the rules every kernel
follows.

Conventions
-----------
1. **Pure functions on plain arrays.** A kernel takes floats and
   :class:`numpy.ndarray` and returns them. It has no side effects and reads no
   global state.
2. **Canonical units.** Every dimensional argument and result is in the canonical
   unit of its kind (the table below), and each kernel's docstring states the unit
   of each argument and result.
3. **No Quantities.** Kernels never receive, create or return an
   :class:`astropy.units.Quantity`. Units are attached and removed only at the public
   boundary, by :func:`hmf.core.units.unit_boundary`. Library code that already works
   in canonical units calls kernels directly, not the decorated public methods.
4. **No Python loops over array elements.** Vectorise with numpy (loops over a few
   grid levels or ODE steps are fine).
5. **No in-place mutation of inputs.** Kernels may be called with views of cached
   arrays.
6. **Batch-size-independent results.** A kernel evaluated on an array gives exactly
   the same values, bit for bit, as on any subset of it. In particular, quadratures
   must not use BLAS matrix products, whose summation order depends on the batch
   size; :func:`scipy.integrate.simpson` along an axis is fine.
7. **Explicit log bases.** Names say ``log10_`` or ``ln_``, never a bare ``log``.

Together these make the kernels the seam for a future JAX backend.

Kernel-level entry points of stages and models
----------------------------------------------
Library code, such as a stage built on other stages, never calls the unit-checked
public methods. It calls their *kernel-level entry points*: public methods (or
functions) whose names end in ``_kernel``, e.g.
:meth:`MassVariance.ln_sigma_and_slope_kernel
<hmf.core.mass_variance.MassVariance.ln_sigma_and_slope_kernel>`,
:meth:`Growth.growth_factor_kernel <hmf.core.growth.Growth.growth_factor_kernel>`,
:meth:`FittingFunction.fsigma_kernel <hmf.core.fits.FittingFunction.fsigma_kernel>`,
:meth:`FittingFunction.modify_dndm_kernel
<hmf.core.fits.FittingFunction.modify_dndm_kernel>` and the ``ln_power_kernel`` of a
:class:`~hmf.core.power_source.PowerSource`. Each one

* takes and returns plain floats and arrays in the canonical units below, and states
  the unit of each argument and result in its docstring;
* follows rules 1-7 above: pure, vectorised, batch-size independent. A stage may
  memoise what it computes (e.g. the lattice of a
  :class:`~hmf.core.mass_variance.MassVariance`), as long as no result depends on
  it;
* **checks its input against its owner's domain**, and raises a
  :class:`~hmf.core.domain.DomainError` outside it (NaN and infinities included), so
  that its caller never gets a silent extrapolation. The check is cheap: it compares
  the smallest and largest value with the domain's bounds
  (:func:`~hmf.core.domain.check_extent`), with no other pass over the array. Where
  an owner extrapolates a table the user supplied by design (e.g. a
  :class:`~hmf.core.power_source.TabulatedPower`), the kernel does not warn: the
  owner exposes the table's range (a ``PowerSource``'s ``table_range``), and the
  stage that evaluates it warns once (see :mod:`hmf.core.power_source`).

The pure functions in this subpackage do not check their inputs: their callers, the
kernel-level entry points, do.

Read-only data holders keep plain names but follow rules 1-7, and do not check
their inputs: e.g. :attr:`Transfer.solution <hmf.core.transfer.Transfer.solution>`,
:attr:`Growth.solution <hmf.core.growth.Growth.solution>` (whose methods take and
return plain arrays, and are NaN beyond its table) and a ``PowerSource``'s
``rho_mean0`` (a float in M☉ h² / Mpc³). Library code evaluates them through the
kernel-level entry points (e.g. :meth:`Growth.growth_factor_kernel
<hmf.core.growth.Growth.growth_factor_kernel>`), not directly.

Canonical units
---------------
This table mirrors :data:`hmf.core.units.CANONICAL_UNITS`; a test checks that the two
agree.

==============  ====================  ===============================
Kind            Unit                  Constant in :mod:`hmf.core.units`
==============  ====================  ===============================
mass            Msun / h              ``Msun_h``
length          Mpc / h               ``Mpc_h``
wavenumber      h / Mpc               ``h_Mpc``
number_density  h^3 / Mpc^3           ``number_density_unit``
dndm            h^4 / (Msun Mpc^3)    ``dndm_unit``
power           Mpc^3 / h^3           ``power_unit``
density         Msun h^2 / Mpc^3      ``rho_unit``
hubble          km / (s Mpc)          ``H0_unit``
==============  ====================  ===============================

Dimensionless quantities (z, sigma, peak height nu, delta_c, Omega, transfer
function, growth factor) are plain floats.
"""
