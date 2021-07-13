Releases
========

dev-version
----------------------

v3.4.2 [13 Jul 2021]
----------------------

**Bugfixes**

- Watson high-z parameters had a typo which is now fixed (#141)

v3.4.1 [07 Jul 2021]
----------------------

**Features**

- Now the user can use non-matched mass definitions for the HMF without translating
  the definition, by setting ``MassFunction.ERROR_ON_BAD_MDEF=False``.

v3.4.0 [07 Jul 2021]
----------------------

**Features**

- Added ``normalized`` attribute of all ``FittingFunction`` models, to tell whether the
  model integrates to the mean density over all mass.
- New ``Bocquet`` fitting functions (for Hydro and DMOnly at all overdensities).

**Changes**

- Internal working of ``GrowthFactor`` has been updated to use splines. It is now faster
  and can natively compute multiple redshifts at once (which was already true of the
  other growth functions).

**Bugfixes**

- Fixes to the BBKS transfer function
- Fixes to allow w0waCDM models (#113)
- Fixes to mass conversion for HMF (according to Bocquet), and raising of error when the
  wrong mass definition is used without allowing mass conversion.


v3.3.4 [08 Jan 2021]
----------------------

**Bugfixes**

- Added ``validate()`` method that is automatically called after ``__init__()`` and
  ``update()`` and also (optionally) on every ``parameter`` update. This can cross-validate different
  inputs. **NOTE**: if you currently have code that sets parameters directly, eg ``mf.z=2.0``,
  you should update to using the ``update()`` method, as this allows multiple parameters
  to be set, and then a call to ``validate()``.

v3.3.3 [21 Dec 2020]
----------------------
**Bugfixes**

- Redefined omch2 to account for massive neutrinos `PR #118 <https://github.com/steven-murray/hmf/pull/118>`_


v3.3.2 [11th Dec 2020]
----------------------
**Changes**

- ``camb`` is no longer installed by default. If you want to install it along with ``hmf``
  (as well as other useful utilities) install with ``pip install hmf[all]``.


v3.3.1 [30th Nov 2020]
----------------------

**Bugfixes**

- Faster and more reliable halofit! `PR #106 <https://github.com/steven-murray/hmf/pull/106>`_

v3.3.0  [24th Nov 2020]
-----------------------
**Features**

- Option to extrapolate high-k for CAMB transfer using EH. This gives a fast and
  reasonably consistent high-k transfer function. If not extrapolating with EH (which
  is still the default) you can now set ``transfer_params={'kmax':1e3}`` to get better
  low-mass/small-scale accuracy. By default, kmax is the same as the default from
  CAMB, which is about 1.4 h/Mpc. Above this value, the transfer function is by default
  linearly extrapolated in log-log space.
- Components now automatically keep track of models, and the base component class
  tracks Component types. This also means that user-defined models can be input via
  strings into a framework (useful for running from CLI).

3.2.0
-----
**Features**

- Updated (working) CLI: use ``hmf run`` to produce mass functions on the command-line!

**Internals**

- Deprecated astropy_to_colossus function.


3.1.1
-----
**Bugfixes**

- Consistently allow string "None" for some parameters.

3.1.0
-----
**Notes**

New versions of CAMB (1.0.0+) seem to run significantly slower by default than previous
versions. This could just be a setting in CAMB that has changed its default. Be aware
of this.

**Features**

- Added new ``CambGrowth`` growth factor model, which computes the growth using CAMB. This is useful especially when
  using w > -1, for which the other growth factor models are inadequate. Solves issue #19 raised by @tijmen.
- Added new module ``mass_definitions`` which more robustly deals with various halo mass definitions, and also includes
  ability to convert mass functions between different definitions.
- New set of tutorials: quickstart/first plot and a broad overview.
- Added shortcuts to ``halo_overdensity_mean`` and ``halo_overdensity_crit``
- Added ``normalized_filter`` to ``MassFunction`` class to make it easier to use a filter
  with a pre-normalized power spectrum.
- Added ``.clone()`` method to all ``Framework`` classes, which gives ability to clone
  the current instance (while also changing some parameters).

**Bugfixes**

- When using camb for the transfer function, some cosmologies would lead to a segfault (i.e. when Ob0 or Tcmb0 are not
  set explicitly). This now raises a helpful error.
- Fixed bug in ordering of iterator in ``get_hmf``.

**Internals**

- Removed logging, which was redundant.
- Moved from nose to pytest
- Significant overhaul of package structure to more modularised form.
- Added flake8 checks and black formatting
- Moved away from git-flow as it is not easy to use with GitHub.


v3.0.8 [26th March 2020]
------------------------
**Bugfixes**

- Some documentation and CI fixes.

**Enhancement**

- Added ``get_dependencies`` method to _Framework, to enable finding all parameters that
  a quantity depends on.
- Growth factor speed-up (thanks to @jlashner)
- Added use_splined_growth argument (thanks to @jlashner)


v3.0.7 [29th April 2019]
------------------------
**Bugfixes**

- Fixed bugs in WDM models that remained from previous versions.
- Fixed bug in SharpK Ellipsoid filter following previous API change.

v3.0.5 [23rd April 2019]
------------------------
**Bugfixes**

- Fixed naming of WDM ``alter_dndm`` to ``alter_model`` to be consistent.
- Fixed bug which made CAMB transfer function not pickleable or copyable.

  .. note:: the fix is not perfect -- if you provide a custom CAMBparams class
            to ``transfer_params``, it is not certain that results will be consistent,
            as a few of the attributes of this object are not pickleable, and thus
            are ignored.

v3.0.3 [1st Dec 2017]
---------------------
**Bugfixes**

- Fixed usage of deprecated MsolMass in wdm
- Fixed Bhattachrya fitting function (thanks to Benedikt Diemer!)
- Fixed typo in Watson fitting function (thanks to Benedikt Diemer!)
- Update cosmo test to use new Astropy constants.
- Fixed issue with sampling function where zeros in ngtm would yield an error.

v3.0.2 [3rd Nov 2017]
---------------------
**Bugfixes**

- Changed parameter checks on instantiation to post-conversion.

v3.0.1 [31st Oct 2017]
----------------------
**Enhancement**

- Normalised all <>_model properties to be actual classes, rather than either class or string.
- Added consistent checking of dictionaries for <>_params parameters.

v3.0.0 [7th June 2017]
----------------------
**Features**

- Now provides compatibility with Python 3.x. Support for 2.x will be removed in hmf v3.1 (whenever that comes).
- Complete overhaul of the caching system. Should be invisible to the user, but streamlines writing of framework
  code considerably. Removes required manual specification of dependencies between quantities, and adds ability
  to specify parameter kinds (model, param, res, option, switch).

**Bugfixes**

- Fixed bug in Caroll1992 GrowthFactor class which affected high-redshift growth factors (thanks to Olmo Piana).
- Fixed astropy dependency to be >= 1.1
- Fixed bug where Takahashi parameters were always passed through regardess of ``takahashi`` setting.
- Fixed small bug where the functional.get_label method returned differently ordered parameters because of dicts.
- Note that the fitting subpackage is temporarily unsupported and I discourage its use for the time being.

**Enhancement**

- Completely removes dependence on archaic pycamb package. Now supports natively supplied python interface to CAMB.
  Install camb with ``pip install --egg camb``. This means that much more modern versions of CAMB can be used.
- Many new tests, to bring total coverage up to >80%, and continuous testing on Python 2.7, 3.5 and 3.6


v2.0.5 [12th January 2017]
--------------------------
**Bugfixes**

- Fixed bug in GrowthFactor which gave ripples in functions of z when a coarse grid was used. Thanks to @mirochaj and
  @thomasguillet!

**Enhancments**

- Streamlined the caching framework a bit (removing cruft)
- Totally removed dependency on the Cache (super)class -- caching indexes now inherent to the called class.
- More robust parameter information based on introspection.


v2.0.4 [11th November, 2016]
----------------------------

**Bugfixes**

- **IMPORTANT**: Fixed a bug in which updating the cosmology after creation did not update the transfer function.

v2.0.3 [22nd September, 2016]
-----------------------------
**Bugfixes**

- SharpK filter integrated over incorrect range of k, now fixed.

**Enhancments**

- WDM models now more consistent with MassFunction API.
- Better warning in HALOFIT module when derivatives don't work first time.


v2.0.2 [2nd August, 2016]
-------------------------

**Features**

- Added a bunch of information to each hmf_model, indicating simulation parameters from which the fit was derived.
- Added ``FromArray`` transfer model, allowing an array to be passed programmatically for ``k`` and ``T``.
- Added ``Carroll1992`` growth factor approximation model.

**Enhancments**

- how_big now gives the boxsize required to expect at least one halo above m in 95% of boxes.

**Bugfixes**

- Removed unnecessary multiplication by 1e6 in cosmo.py (thanks @iw381)
- **IMPORTANT**: normalisation now calculated using convergent limits on k, rather than
  user-supplied values.
- **IMPORTANT**: fixed bug in Bhattacharya fit, which was multiplying by an extra delta_c/sigma.
- fixed issue with ``nonlinear_mass`` raising exception when mass outside given mass range.

v2.0.1 [2nd May, 2016]
----------------------

**Bugfixes**

- Corrects the default sigma_8 and n (spectral index) parameters to be from Planck15 (previously
  from Planck13), which corresponds to the default cosmology. **NOTE:** this will change user-code
  output silently unless sigma_8 and n are explicitly set.


v2.0.0
------
v2.0.0 is a (long overdue) major release with several backward-incompatible changes.
There are several major features still to
come in v2.1.0, which may again be backward incompatible. Though this is not ideal (ideally
backwards-incompatible changes will be restricted to increase in the major version number),
this has been driven by time constraints.

Known issues with this version, to be addressed by the next, are that both scripts (hmf and hmf-fit)
are buggy and untested. Don't use these until the next version unless you're crazy.

**Features**

- New methods on all frameworks to list all parameters, defaults and current values.
- New general structure for Frameworks and Components makes for simpler delineation and extensibility
- New growth_factor module which adds extensibility to the growth factor calculation
- New transfer_models module which separates the transfer models from the general framework
- New Component which can alter dn/dm in WDM via ad-hoc adjustment
- Added a Prior() abstract base class to the fitting routines
- Added a guess() method to fitting routines
- Added ll() method to Prior classes for future extendibility
- New fit from Ishiyama+2015, Manera+2010 and Pillepich+2009

**Enhancments**

- Removed nz and z2 from MassFunction. They will return in a later version but better.
- Improved structure for FittingFunction Component, with ``cutmask`` property defining valid mass range. NOTE: the default
  MassFunction is no longer to mask values outside the valid range. In fact, the parameter ``cut_fit`` no longer exists.
  One can achieve the effect by accessing a relevant array as dndm[MassFunction.hmf.cutmask]
- Renamed some parameters/quantities for more consistency (esp. M --> m)
- No longer dependent on cosmolopy, but rather uses Astropy (v1.0+)
- ``mean_dens`` now ``mean_density0``, as per Astropy
- Added exception to catch when dndm has many NaN values in it.
- Many more tests
- Made the ``cosmo`` class pickleable by cutting out a method and using it as a function instead.
- Added normalise() to Transfer class.
- Updated fit.py extensively, and provided new example config files
- Send arbitrary kwargs to downhill solver
- New internal _utils module provides inheritable docstrings

**Bugfixes**

- fixed problem with _gtm method returning nans.
- fixed simple bugs in BBKS and BondEfs transfer models.
- fixes in _cache module
- simple bug when updating sigma_8 fixed.
- Made the EnsembleSampler object pickleable by setting ``__getstate__``
- Major bug fix for EH transfer function without BAO wiggles
- @parameter properties now return docstrings

----------------------


v1.8.0 [February 2, 2015]
-------------------------

**Features**

- Better WDM models
- Added SharpK and SharpKEllipsoid filters and overhauled filter system.


**Enhancments**

- Separated WDM models from main class for extendibility
- Enhanced caching to deal with subclassing

**Bugfixes**

- Minor bugfixes

----------------------

1.7.1 [January 28, 2015]
------------------------

**Enhancments**

- Added warning to docstring of _dlnsdlnm and n_eff for non-physical
  oscillations.

----------------------

1.7.0 [October 28, 2014]
------------------------

**Features**

- Very much updated fitting routines, in class structure
- Made fitting_functions more flexible and model-like.

**Enhancments**

- Modified get_hmf to be more general
- Moved get_hmf and related functions to "functional.py"

----------------------


1.6.2 [September 16, 2014]
--------------------------

**Features**

- New HALOFIT option for original co-oefficients from Smith+03

**Enhancments**

- Better Singleton labelling in get_hmf
- Much cleaning of mass function integrations. New separate module for it.
- **IMPORTANT**: Removal of nltm routine altogether, as it is inconsistent.
- **IMPORTANT**: mltm now called rho_ltm, and mgtm called rho_gtm
- **IMPORTANT**: Definition of rho_ltm now assumes all mass is in halos.
- Behroozi-specific modifications moved to Behroozi class
- New property hmf which is the actual class for mf_fit

**Bugfixes**

- Fixed bug in Behroozi fit which caused an infinite recursion
- Tests fixed for new cumulants.


----------------------

1.6.1 [September 8, 2014]
-------------------------

**Enhancments**

- Better get_hmf function

**Bugfixes**

- Fixed "transfer" property
- Updates fixed for transfer_fit
- Updates fixed for nu
- Fixed cache bug where unexecuted branches caused some properties to be misinterpreted
- Fixed bug in CAMB transfer options, where defaults would overwrite user-given values (introduced in 1.6.0)
- Fixed dependence on transfer_options
- Fixed typo in Tinker10 fit at z = 0

----------------------

1.6.0 [August 19, 2014]
-----------------------

**Features**

- New Tinker10 fit (Tinker renamed Tinker08, but Tinker still available)

**Enhancments**

- Completely re-worked caching module to be easier to code and faster.
- Better Cosmology class -- more input combinations available.

**Bugfixes**

- Fixed all tests.


----------------------

1.5.0 [May 08, 2014]
--------------------

**Features**

- Introduced _cache module: Extracts all caching logic to a
  separate module which defines decorators -- much simpler coding!

----------------------

1.4.5 [January 24, 2014]
------------------------

**Features**

- Added get_hmf function to tools.py -- easy iteration over models!
- Added hmf script which provides cmd-line access to most functionality.

**Enhancments**

- Added Behroozi alias to fits
- Changed kmax and k_per_logint back to have ``transfer__`` prefix.

**Bugfixes**

- Fixed a bug on updating delta_c
- Changed default kmax and k_per_logint values a little higher for accuracy.


----------------------


1.4.4 [January 23, 2014]
------------------------

**Features**

- Added ability to change the default cosmology parameters

**Enhancments**

- Made updating Cosmology simpler.

**Bugfixes**

- Fixed a bug in the Tinker function (log was meant to be log10):
  - thanks to Sebastian Bocquet for pointing this out!
- Fixed a bug in updating n and sigma_8 on their own (introduced in 1.4.0)
- Fixed a bug when using a file for the transfer function.

----------------------

1.4.3 [January 10, 2014]
------------------------

**Bugfixes**

- Changed license in setup

----------------------

1.4.2 [January 10, 2014]
------------------------

**Enhancments**

- Mocked imports of cosmolopy for setup
- Cleaner imports of cosmolopy

----------------------

1.4.1 [January 10,2014]
-----------------------

**Enhancments**

- Updated setup requirements and fixed a few tests

----------------------

1.4.0 [January 10, 2014]
------------------------

**Enhancments**

- Upgraded API once more:
  - Now Perturbations --> MassFunction
- Added transfer.py which handles all k-based quantities
- Upgraded docs significantly.

----------------------

1.3.1 [January 06, 2014]
------------------------

**Bugfixes**

- Fixed bug in transfer read-in introduced in 1.3.0

----------------------

1.3.0 [January 03, 2014]
------------------------

**Enhancments**

- A few more documentation updates (especially tools.py)
- Removed new_k_bounds function from tools.py
- Added ``w`` parameter to cosmolopy dictionary in ``cosmo.py``
- Changed cosmography significantly to use cosmolopy in general
- Generally tidied up some of the update mechanisms.
- **API CHANGE**: cosmography.py no longer exists -- I've chosen to utilise
  cosmolopy more heavily here.
- Added Travis CI usage

**Bugfixes**

- Fixed a pretty bad bug where updating h/H0 would crash the program if
  only one of omegab/omegac was updated alongside it
- Fixed a compatibility issue with older versions of numpy in cumulative
  functions

----------------------

1.2.2 [December 10, 2013]
-------------------------

**Bugfixes**

- Bug in "EH" transfer function call

----------------------

1.2.1 [December 6, 2013]
------------------------

**Bugfixes**

- Small bugfixes to update() method

----------------------

1.2.0 [December 5, 2013]
------------------------

**Features**

- Addition of cosmo module, which deals with the cosmological parameters in a cleaner way

**Enhancments**

- Major documentation overhaul -- most docstrings are now in Sphinx/numpydoc format
- Some tidying up of several functions.

----------------------

1.1.10 [October 29, 2013]
-------------------------

**Enhancements**
- Better updating -- checks if update value is actually different.
- Now performs a check to see if mass range is inside fit range.

**Bugfixes**

- Fixed bug in mltm property

----------------------

1.1.9 [October 4, 2013]
-----------------------

**Bugfixes**

- Fixed some issues with n(<m) and M(<m) causing them to give NaN's

----------------------

1.1.85 [October 2, 2013]
------------------------

**Enhancments**

- The normalization of the power spectrum now saved as an attribute

----------------------

1.1.8 [September 19, 2013]
--------------------------

**Bugfixes**

- Fixed small bug in SMT function which made it crash

----------------------

1.1.7 [September 19, 2013]
--------------------------

**Enhancments**

- Updated "ST" fit to "SMT" fit to avoid confusion. "ST" is still available for now.
- Now uses trapezoid rule for integration as it is faster.

----------------------

1.1.6 [September 05, 2013]
--------------------------

**Enhancments**

- Included an option to use delta_halo as compared to critical rather than mean density (thanks to A. Vikhlinin and anonymous referree)

**Bugfixes**

- Couple of bugfixes for fitting_functions.py
- Fixed mass range of Tinker (thanks to J. Tinker and anonymous referee for this)

----------------------

1.1.5 [September 03, 2013]
--------------------------

**Enhancments**

-Added a whole suite of tests against genmf that actually work

**Bugfixes**

- Fixed bug in mgtm (thanks to J. Mirocha)
- Fixed an embarrassing error in Reed07 fitting function
- Fixed a bug in which dndlnm and its dependents (ngtm, etc..) were calculated wrong
  if dndlog10m was called first.
- Fixed error in which for some choices of M, the whole extension in ngtm would be NAN and give error

----------------------

1.1.4 [August 27, 2013]
-----------------------

**Features**

- Added ability to change resolution in CAMB from hmf interface
  (This requires a re-install of pycamb to the newest version on the fork)

----------------------

1.1.3 [August 7, 2013]
----------------------

**Features**

- Added Behroozi Fit (thanks to P. Behroozi)

----------------------

1.1.2 [July 02, 2013]
---------------------

**Features**

- Ability to calculate fitting functions to whatever mass you want (BEWARE!!)

----------------------

1.1.1 [July 02, 2013]
---------------------

**Features**

- Added Eisenstein-Hu fit to the transfer function

**Enhancments**

- Improved docstring for Perturbations class

**Bugfixes**

- Corrections to Watson fitting function from latest update on arXiv (thanks to W. Watson)
- **IMPORTANT**:  Fixed units for k and transfer function (Thanks to A. Knebe)

----------------------

1.1.0 [June 27, 2013]
---------------------

**Enhancments**

- Massive overhaul of structure: Now dependencies are tracked throughout the program, making updates even faster

----------------------

1.0.10 [June 24, 2013]
----------------------

**Enhancments**

- Added dependence on Delta_vir to Tinker

----------------------

1.0.9 [June 19, 2013]
---------------------

**Bugfixes**

- Fixed an error with an extra ln(10) in the mass function (quoted as dn/dlnM but actually outputting dn/dlog10M)

----------------------

1.0.8 [June 19, 2013]
---------------------

**Enhancments**

- Took out log10 from cumulative mass functions
- Better cumulative mass function logic

----------------------

1.0.6 [June 19, 2013]
---------------------

**Bugfixes**

- Fixed cumulative mass functions (extra factor of M was in there)

----------------------

1.0.4 [June 6, 2013]
--------------------

**Features**

- Added Bhattacharya fitting function

**Bugfixes**

- Fixed concatenation of list and dict issue

----------------------

1.0.2 [May 21, 2013]
--------------------

**Bugfixes**

- Fixed some warnings for non-updated variables passed to update()

----------------------

1.0.1 [May 20, 2013]
--------------------

**Enhancments**

- Added better warnings for non-updated variables passed to update()
- Made default cosmology WMAP7

----------------------

0.9.99 [May 10, 2013]
---------------------

**Enhancments**

- Added warning for k*R limits

**Bugfixes**

- Couple of minor bugfixes
- **Important** Angulo fitting function corrected (arXiv version had a typo).

----------------------

0.9.97 [April 15, 2013]
-----------------------

**Bugfixes**

- Urgent Bugfix for updating cosmology (for transfer functions)

----------------------

0.9.96 [April 11, 2013]
-----------------------

**Bugfixes**

- Few bugfixes

----------------------

0.9.95 [April 09, 2013]
-----------------------

**Features**

- Added cascading variable changes for optimization
- Added the README
- Added update() function to simply change parameters using cascading approach

----------------------

0.9.9 [April 08, 2013]
----------------------

**Features**

- First version in its own package
- Added pycamb integration

**Enhancments**

- Removed fitting function from being a class variable
- Removed overdensity form being a class variable

----------------------

0.9.7 [March 18, 2013]
----------------------

**Enhancments**

- Modified set_z() so it only does calculations necessary when z changes
- Made calculation of dlnsdlnM in init since it is same for all z
- Removed mean density redshift dependence

----------------------

0.9.5 [March 10, 2013]
----------------------

**Features**

- The class has been in the works for almost a year now, but it currently
  will calculate a mass function based on any of several fitting functions.
