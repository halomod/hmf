.. hmf documentation master file, created by
   sphinx-quickstart on Mon Dec  2 10:40:08 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

hmf
===
`hmf` is a python application that provides a flexible and simple way to calculate the 
Halo Mass Function for any input cosmology, redshift, dark matter model, virial
overdensity or several other variables. Addition of further variables should be simple. 

It is also the backend to `HMFcalc <http://hmf.icrar.org>`_, the online HMF calculator.

Installation
------------
There are several requirements to install to use `hmf`, as we have operated on
the principle of not re-inventing the wheel. 

The dependencies will not automatically install when installing the `hmf` 
package, because this can create other problems, so you'll need to do it yourself.
Don't worry, it's mostly pretty easy. Just use `pip`, skipping anything you 
alread have:

1. ``pip install numpy``
2. ``pip install scitools``
3. ``pip install scipy``
4. ``pip install cosmolopy``
5. ``pip install emcee``

You will only need `emcee` if you are going to be using the fitting capabilities
of `hmf`. Once these are installed, there is a final dependency, pycamb, which is
a little more involved. It is not mandatory to install, but does provide some good
features. To install pycamb::

	cd <Directory that pycamb source will live in>
	git clone https://github.com/steven-murray/pycamb
	cd pycamb
	[sudo] python setup.py install [--get=www.address-where-camb-code-lives.org]
	
The final command gives the option of automatically downloading and 
compiling CAMB while installing pycamb. It cannot be done more automatically
at this point due to licensing. Alternatively, if one does not know the 
location of the camb downloads, go to camb.info and follow the instructions.
Download the source directory to your pycamb folder, and untar it there.
Then use ``python setup.py install" and it should work.

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

.. note :: Older versions of `hmf` used the class called `Perturbations()` 
		rather than `MassFunction()`.
		
Please check the more in-depth user-guide for more details, or even the API
documentation.

User Guide
----------
Look here for more details concerning the usage in general.
 
API Documentation
-----------------
.. toctree::
   :maxdepth: 2
   
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

