.. hmf documentation master file, created by
   sphinx-quickstart on Mon Dec  2 10:40:08 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. include:: ../README.rst

Installation
------------
`hmf` is built on several other packages, most of which will be familiar to the
scientific python programmer. All of these dependencies *should* be automatically
installed when installing `hmf`, except for one. Explicitly, the dependencies are
numpy, scipy, scitools, cosmolopy and emcee. 

You will only need `emcee` if you are going to be using the fitting capabilities
of `hmf`. The final, optional, library is pycamb. It is a bit trickier to install.
Please follow the guidelines on its `readme page <https://github.com/steven-murray/pycamb.git>`_.

.. note:: At present, versions of CAMB post March 2013 are not working with
		  `pycamb`. Please use earlier versions until further notice.

Finally the `hmf` package needs to be installed: ``pip install hmf``. If you want
to install the latest build (not necessarily stable), grab it `here 
<https://github.com/steven-murray/hmf.git>`_.

To go really bleeding edge, install the develop branch using
``pip install git+git://github.com/steven-murray/hmf.git@develop``.

Basic Usage
-----------
`hmf` can be used interactively (for instance in ipython) or in a script and is
called like this:

>>> from hmf import MassFunction
>>> hmf = MassFunction()
>>> mass_func = hmf.dndlnm
>>> mass_variance = hmf.sigma
>>> ...

This will return a Sheth-Mo-Tormen (2001) mass function between 
:math:`10^{10}-10^{15} M_\odot`, at :math:`z=0` for the default PLANCK cosmology. 
Cosmological parameters may be passed to the initialiser, ``MassFunction()``

To change the parameters (cosmological or otherwise), one should use the 
`update()` method, if a MassFunction() object already exists. For example

>>> hmf = MassFunction()
>>> hmf.update(omegab = 0.05,z=10) #update baryon density and redshift
>>> cumulative_mass_func = hmf.ngtm

Please check the more in-depth user-guide for more details, or even the API
documentation.


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

