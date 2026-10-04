
hmf
===

Frameworks
----------

.. autosummary::
   :caption: Frameworks
   :toctree: _autosummary
   :template: modules.rst

   hmf.cosmology.cosmo
   hmf.density_field.transfer
   hmf.mass_function.hmf

Model Components
----------------

.. autosummary::
   :caption: Model Components
   :toctree: _autosummary
   :template: component-module.rst

   hmf.cosmology.growth_factor
   hmf.density_field.transfer_models
   hmf.density_field.filters
   hmf.halos.mass_definitions
   hmf.mass_function.fitting_functions

Alternative Cosmologies
-----------------------
Alternative cosmology modules contain both new model components and patched Frameworks
to enable consistent modeling of alternative cosmological scenarios.

.. autosummary::
   :caption: Alternative Cosmologies
   :toctree: _autosummary
   :template: modules.rst

   hmf.alternatives.wdm

Other Calculations and Utilities
--------------------------------

.. autosummary::
   :caption: Functions & Utilities
   :toctree: _autosummary
   :template: modules.rst

   hmf.density_field.halofit
   hmf.mass_function.integrate_hmf
   hmf.helpers.sample
   hmf.helpers.functional
   hmf.exceptions

hmf.core (experimental)
-----------------------
The v4 core preview: see :doc:`core`. Its API may change before 4.0.

.. autosummary::
   :caption: hmf.core (experimental)
   :toctree: _autosummary
   :template: core-module.rst

   hmf.core.units
   hmf.core.model
   hmf.core.stage
   hmf.core.accuracy
   hmf.core.domain
   hmf.core._kernels
   hmf.core._kernels.fits

Models (their parameters are listed in each class's "Parameters" section):

.. autosummary::
   :toctree: _autosummary
   :template: core-model-module.rst

   hmf.core.fits
