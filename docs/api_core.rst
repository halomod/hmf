hmf.core API (experimental)
===========================

.. automodule:: hmf.core

.. The summaries below are relative to hmf.core, the current module.

For the conventions these modules follow, see :doc:`core`.

Conventions
-----------
The building blocks that every stage and model uses, and the matter species they
describe.

.. autosummary::
   :toctree: _autosummary
   :template: core-module.rst

   units
   model
   stage
   accuracy
   domain
   species

Stages
------
The steps of a calculation, each a function of its natural variables.

.. autosummary::
   :toctree: _autosummary
   :template: core-module.rst

   transfer
   growth
   power_source
   mass_variance
   linear_power
   mass_function

Models
------
The interchangeable physical models that the stages are built from.

.. autosummary::
   :toctree: _autosummary
   :template: core-module.rst

   transfer_models
   growth_models
   filters

Fitting functions (their parameters are listed in each class's "Parameters" section):

.. autosummary::
   :toctree: _autosummary
   :template: core-model-module.rst

   fits

Utilities
---------

.. autosummary::
   :toctree: _autosummary
   :template: core-module.rst

   cache

Kernels (internal)
------------------
The unit-free numerics behind the public methods. They are documented for
contributors; they are not part of the public API.

.. autosummary::
   :toctree: _autosummary
   :template: core-module.rst

   _kernels
   _kernels.transfer
   _kernels.growth
   _kernels.filters
   _kernels.mass_variance
   _kernels.mass_function
   _kernels.interpolation
   _kernels.lattice
   _kernels.arrays
   _kernels.fits
