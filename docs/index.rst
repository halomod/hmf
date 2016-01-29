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
of `hmf`. The final, optional, library is pycamb, which can not be installed 
using pip currently. To install pycamb::

	cd <Directory that pycamb source will live in>
	git clone https://github.com/steven-murray/pycamb
	cd pycamb
	[sudo] python setup.py install [--get=www.address-where-camb-code-lives.org]
	
The final command gives the option of automatically downloading and 
compiling CAMB while installing pycamb. It cannot be done more automatically
at this point due to licensing. Alternatively, if one does not know the 
location of the camb downloads, go to camb.info and follow the instructions.
Download the source directory to your pycamb folder, and untar it there.
Then use ``python setup.py install`` and it should work.

.. note :: At present, versions of CAMB post March 2013 are not working with 
		`pycamb`. Please use earlier versions until further notice.

Finally the `hmf` package needs to be installed: ``pip install hmf``. If you want
to install the latest build (not necessarily stable), grab it `here 
<https://github.com/steven-murray/hmf.git>`_.

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

