FAQ
---

What are the units of hmf's outputs?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Firstly, the units of all quantities should be specified in their docstring (please
`make an issue <https://github.com/steven-murray/hmf/issues/new>`_ if this isn't the case!).
This can also be seen in the `API docs <https://hmf.readthedocs.io/en/latest/_autosummary/hmf/hmf.mass_function.hmf.MassFunction.html#hmf.mass_function.hmf.MassFunction>`_.

Nevertheless, it should be said that *all* quantities in ``hmf`` include little-*h*. So,
for example, masses are in units of Msun/h, and distances in Mpc/h. This is consistent
for every quantity in ``hmf``. Furthermore, little-h is defined as the value of the
Hubble constant divided by 100 km/s/Mpc (i.e. it is h_100, not h_70).


How can I find out all components (i.e. kinds of models) defined?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Simply do::

    >>> from hmf import get_base_components
    >>> get_base_components()

How can I determine each kind of model defined for a particular component?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Simply do::

    >>> from hmf import get_base_component
    >>> bias = get_base_component("Bias")
    >>> bias.get_models()

This returns a dictionary of ``name: class`` for every model defined for the ``Bias``
component -- even user-defined models (as long as they've been imported)!

To get a particular model just from its string name::

    >>> from hmf import get_mdl
    >>> mdl = get_mdl("SMT")

Now ``mdl`` will be the ``SMT`` fitting-function class. If there are name conflicts
between models for different components, the last-defined model will be returned. You
can specify more accurately what kind of model you want::

    >>> mdl = get_mdl("SMT", kind='FittingFunction')

Here the kind is the *name* of the class defining this component (see above section
for how to determine what kinds are available).

My mass function looks wrong at small masses, why?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
One common reason for this is that the mass function is being calculated at masses
that correspond to scales at higher *k* than the transfer function was calculated at.
You can set your *kmax* to be higher to fix this.

However, note that setting a higher *kmax* for the CAMB transfer function will not
automatically fix this, since the underlying CAMB code only calculates the transfer
function up to an internally-specific *kmax*, and this is extrapolated by HMF.
This can be easily fixed by setting the CAMB-specific parameter
``transfer_params = {"kmax": some_high_number}``. However, note that this will slow down
the calculation of the transfer function quite significantly.

A cheaper way to fix the problem is to leave the default *kmax* for CAMB, and instead
use ``extrapolate_with_eh=True``, which will use Eisenstein & Hu's fitting formula to
extrapolate the transfer function to higher *k* values. This is not as accurate as
calculating the transfer function with CAMB out to  higher *k* values, but is much faster.

.. note:: From v3.5.0, the default behaviour is to extrapolate with Eisenstein & Hu
          fitting formula. This is because the default *kmax* for CAMB is too low for
          the default *kmax* for the mass function, and this was causing problems for
          users. If you want to use the old behaviour, you can set
          ``extrapolate_with_eh=False``.
