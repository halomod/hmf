Attribution
-----------
Please cite `Murray, Power and Robotham (2013) <https://arxiv.org/abs/1306.6721>`_,
`Murray, Diemer, Chen, et al. (2021) <https://arxiv.org/abs/2009.14066>`_
and/or https://ascl.net/1412.006 (whichever is more appropriate) if you find this
code useful in your research. Please also consider starring the GitHub repository.

Citing the models you use
^^^^^^^^^^^^^^^^^^^^^^^^^
Most models in ``hmf`` (fitting functions, transfer functions, growth factors,
filters, etc.) come from published papers, which should also be cited. Every
framework can list the references for its current setup, grouped by the
parameter that selects each model::

    >>> from hmf import MassFunction
    >>> mf = MassFunction(hmf_model="Tinker08", transfer_model="EH")
    >>> for source, refs in mf.get_acknowledgments().items():
    ...     print(source, refs)

The first entry, ``"hmf"``, is the ``hmf`` paper itself. Each ``*_model``
parameter (``"hmf_model"``, ``"transfer_model"``, ``"growth_model"``, etc.)
then gives the references of the model currently set on it; ``"cosmo_model"``
gives the source of the cosmological parameters for astropy's built-in
cosmologies. A model with nothing to cite gives an empty tuple. The references
are read from the model classes, so this does not compute anything, and
changing a model (e.g. ``mf.update(hmf_model="ST")``) changes the result.

For a bibliography, ask for a single list with duplicates removed::

    >>> bibliography = mf.get_acknowledgments(flat=True)

The keys say which model each citation belongs to, so you can drop those for
models that played no part in your results. The result covers the
chosen models, not the methods behind every quantity a framework can compute:
HALOFIT, used only for ``nonlinear_power``, is not included, and the docstring
of a quantity like that says what to cite.

When you write your own component, set its ``references`` class attribute to a
tuple of citation strings so that it is included::

    class MyFit(hmf.mass_function.fitting_functions.PS):
        references = ("Me, A., 2026. Journal 1, 1.",)

Not every model has references yet, so check the list against the models you
actually use.
