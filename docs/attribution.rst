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
framework can list the references for its current setup::

    >>> from hmf import MassFunction
    >>> mf = MassFunction(hmf_model="Tinker08", transfer_model="EH")
    >>> for ref in mf.get_acknowledgments():
    ...     print(ref)

The list starts with the ``hmf`` paper, followed by the references of the
framework itself (e.g. the source of the cosmological parameters), and then
those of each chosen component model. It is gathered from the model classes
alone, so it does not compute anything. Changing a model (e.g.
``mf.update(hmf_model="ST")``) changes the list.

The list covers the chosen models, not the methods behind every quantity a
framework can compute. For example, HALOFIT (used only for
``nonlinear_power``) is not included; if you use a quantity like that, its
docstring says what to cite.

When you write your own component, set its ``references`` class attribute to a
tuple of citation strings so that it is included::

    class MyFit(hmf.mass_function.fitting_functions.PS):
        references = ("Me, A., 2026. Journal 1, 1.",)

Not every model has references yet, so check the list against the models you
actually use.
