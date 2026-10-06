
hmf API (v3)
============
The stable v3 API. The experimental v4 core, :mod:`hmf.core`, is documented
separately in :doc:`api_core`.

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
