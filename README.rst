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
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

``hmf`` is a python application that provides a flexible and simple way to calculate the
Halo Mass Function for a range of varying parameters. It is also the backend to
`HMFcalc <http://hmf.icrar.org>`_, the online HMF calculator.

Full Documentation
------------------
`Read the docs. <http://hmf.readthedocs.org>`_

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
* Comprehensive in terms of implemented Component models:

  * 5+ models of transfer functions including directly from CAMB
  * 4 filter functions
  * 20 hmf fitting functions

* Includes models for Warm Dark Matter
* Nonlinear power spectra via HALOFIT
* Functions for sampling the mass function.
* CLI scripts for producing any quantity included.
* Python 2 and 3 compatible

Note
~~~~
From v3.1, ``hmf`` supports Python 3.6+, and has dropped support for Python 2.


Quickstart
----------
Once you have ``hmf`` installed, you can quickly generate a mass function
by opening an interpreter (e.g. IPython/Jupyter) and doing::

    >>> from hmf import MassFunction
    >>> hmf = MassFunction()
    >>> mass_func = hmf.dndlnm

Note that all parameters have (what I consider reasonable) defaults. In particular,
this will return a Tinker (2008) mass function between
10^10 and 10^15 solar masses, at z=0 for the default PLANCK15 cosmology.
Nevertheless, there are several parameters which can be input, either cosmological
or otherwise. The best way to see these is to do::

    >>> MassFunction.parameter_info()

We can also check which parameters have been set in our "default" instance::

    >>> hmf.parameter_values

To change the parameters (cosmological or otherwise), one should use the
``update()`` method, if a MassFunction() object already exists. For example::

    >>> hmf = MassFunction()
    >>> hmf.update(cosmo_params={"Ob0": 0.05}, z=10) #update baryon density and redshift
    >>> cumulative_mass_func = hmf.ngtm

For a more involved introduction to ``hmf``, check out the `tutorials <tutorials.html>`_,
or the `API docs <api.html>`_.

Using the CLI
~~~~~~~~~~~~~
You can also run ``hmf`` from the command-line. For basic usage, do::

    hmf run --help

Configuration for the run can be specified on the CLI or via a TOML file (recommended).
An example TOML file can be found in `examples/example_run_config.toml <examples/example_run_config>`_.
Any parameter specifiable in the TOML file can alternatively be specified on the commmand
line after an isolated double-dash, eg.::

    hmf run -- z=1.0 hmf_model='SMT01'

Versioning
----------
From v3.1.0, ``hmf`` will be using strict semantic versioning, such that increases in
the **major** version have potential API breaking changes, **minor** versions introduce
new features, and **patch** versions fix bugs and other non-breaking internal changes.

If your package depends on ``hmf``, set the dependent version like this::

    hmf>=3.1<4.0

Attribution
-----------
Please cite `Murray, Power and Robotham (2013) <https://arxiv.org/abs/1306.6721>`_,
`Murray, Diemer, Chen, et al. (2021) <https://arxiv.org/abs/2009.14066>`_
and/or https://ascl.net/1412.006 (whichever is more appropriate) if you find this
code useful in your research. Please also consider starring the GitHub repository.
