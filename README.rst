===
hmf
===

**The halo mass function calculator.**

.. image:: https://github.com/halomod/hmf/workflows/Tests/badge.svg
    :target: https://github.com/halomod/hmf
.. image:: https://badge.fury.io/py/hmf.svg
    :target: https://badge.fury.io/py/hmf
.. image:: https://codecov.io/gh/halomod/hmf/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/halomod/hmf
.. image:: https://img.shields.io/pypi/pyversions/hmf.svg
    :target: https://pypi.org/project/hmf/
.. image:: https://readthedocs.org/projects/hmf/badge/?version=latest
    :target: https://hmf.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation
.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/halomod/hmf

``hmf`` is a python application that provides a flexible and simple way to calculate the
Halo Mass Function for a range of varying parameters. It is also the backend to
`HMFcalc <http://thehalomod.app>`_, the online HMF calculator.

To get started, see the
`quickstart guide <https://hmf.readthedocs.io/en/latest/quickstart.html>`_ and the
`API docs <https://hmf.readthedocs.io/en/latest/autoapi/hmf/index.html>`_.

.. important:: Pleae remember to cite ``hmf`` if you use it in your work. The citation information can be found
   in the `attribution page <https://hmf.readthedocs.io/en/latest/attribution.html>`_.

Features
--------
* Calculate mass functions and related quantities extremely easily.
* Very simple to start using, but wide-ranging flexibility.
* Caching system for optimal parameter updates, for efficient iteration over parameter space.
* Support for all LambdaCDM cosmologies, as well as WDM.
* Focus on flexibility in models. Each "Component", such as fitting functions, filter functions,
  growth factor models and transfer function fits are implemented as generic classes that
  can easily be altered by the user without touching the source code.
* Focus on simplicity in frameworks. Each "Framework" mixes available "Components" to derive
  useful quantities -- all given as attributes of the Framework.
* Comprehensive in terms of output quantities: access differential and cumulative mass functions,
  mass variance, effective spectral index, growth rate, cosmographic functions and more.
* Comprehensive in terms of implemented Component models:

  * 5+ models of transfer functions including directly from CAMB
  * 4 filter functions
  * 20 hmf fitting functions

* Nonlinear power spectra via HALOFIT
* Functions for sampling the mass function.
* CLI commands for producing any quantity included.
