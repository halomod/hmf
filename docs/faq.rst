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
