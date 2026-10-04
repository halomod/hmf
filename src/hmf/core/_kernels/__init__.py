"""Unit-free numerical kernels of the v4 core.

This subpackage will hold the numerics behind every public method of
:mod:`hmf.core`. It is empty for now; this docstring sets the rules every kernel
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
