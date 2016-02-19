.. hmf documentation master file, created by
   sphinx-quickstart on Mon Dec  2 10:40:08 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. include:: ../README.rst

Features
--------
* Calculate mass functions and related quantities extremely easily.
* Very simple to start using, but wide-ranging flexibility.
* Caching system for optimal parameter updates, for efficient iteration over parameter space.
* Support for all LambdaCDM cosmologies.
* Focus on flexibility in models. Each "Component", such as fitting functions, filter functions,
  growth factor models and transfer function fits are implemented as generic classes that
  can easily be altered by the user without touching the source code.
* Focus on simplicity in frameworks. Each "Framework" mixes available "Components" to derive
  useful quantities -- all given as attributes of the Framework.
* Comprehensive in terms of output quantities: access differential and cumulative mass functions,
  mass variance, effective spectral index, growth rate, cosmographic functions and more.
* Comprehensive in terms of implemented Component models
    * 5+ models of transfer functions including directly from CAMB
    * 4 filter functions
    * 20 hmf fitting functions
* Includes models for Warm Dark Matter
* Nonlinear power spectra via HALOFIT
* Functions for sampling the mass function.
* CLI scripts both for producing any quantity included, or fitting any quantity.

Installation
------------
`hmf` is built on several other packages, most of which will be familiar to the
scientific python programmer. All of these dependencies *should* be automatically
installed when installing `hmf`, except for one. Explicitly, the dependencies are
numpy, scipy, scitools, cosmolopy and emcee. 

You will only need `emcee` if you are going to be using the fitting capabilities
of `hmf`.

The final, optional, library is pycamb. It is a bit trickier to install.
Please follow the guidelines on its `readme page <https://github.com/steven-murray/pycamb.git>`_.

.. note:: At present, versions of CAMB post March 2013 are not working with
		  `pycamb`. Please use earlier versions until further notice.

Finally the `hmf` package needs to be installed: ``pip install hmf``. If you want
to install the latest build (not necessarily stable), grab it `here 
<https://github.com/steven-murray/hmf.git>`_.

To go really bleeding edge, install the develop branch using
``pip install git+git://github.com/steven-murray/hmf.git@develop``.

Quickstart
----------
Once you have `hmf` installed, you can quickly generate a mass function
by opening an interpreter (e.g. IPython) and doing:

>>> from hmf import MassFunction
>>> hmf = MassFunction()
>>> mass_func = hmf.dndlnm

Note that all parameters have (what I consider reasonable) defaults. In particular,
this will return a Sheth-Mo-Tormen (2001) mass function between
:math:`10^{10}-10^{15} M_\odot`, at :math:`z=0` for the default PLANCK15 cosmology.
Nevertheless, there are several parameters which can be input, either cosmological
or otherwise. The best way to see these is to do

>>> MassFunction.parameter_info()

We can also check which parameters have been set in our "default" instance:

>>> hmf.parameter_values

To change the parameters (cosmological or otherwise), one should use the 
`update()` method, if a MassFunction() object already exists. For example

>>> hmf = MassFunction()
>>> hmf.update(Ob0 = 0.05, z=10) #update baryon density and redshift
>>> cumulative_mass_func = hmf.ngtm

For a more involved introduction to `hmf`, check out the `tutorials <tutorials.html>`_,
which are currently under construction, or the `API docs <api.html>`_.


Contents
--------
.. toctree::
   :maxdepth: 2

   tutorials
   license
   api
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

