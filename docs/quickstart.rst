Quickstart
==========

Installation
------------
Please see the `installation instructions <installation.html>`_ for details on how to
install ``hmf`` and its dependencies. For most users, the following will be sufficient::

    pip install hmf[extra]

Using the Library
-----------------

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
-------------
You can also run ``hmf`` from the command-line. For basic usage, do::

    hmf run --help

Configuration for the run can be specified on the CLI or via a TOML file (recommended).
An example TOML file can be found in
`examples/example_run_config.toml <https://github.com/halomod/hmf/tree/master/examples/example_run_config.toml>`_.
Any parameter specifiable in the TOML file can alternatively be specified on the commmand
line after an isolated double-dash, eg.::

    hmf run -- z=1.0 hmf_model='SMT01'
