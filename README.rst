===
hmf
===

**The halo mass function calculator.**

.. image:: https://github.com/steven-murray/hmf/workflows/Tests/badge.svg
    :target: https://github.com/steven-murray/hmf
.. image:: https://badge.fury.io/py/hmf.svg
    :target: https://badge.fury.io/py/hmf
.. image:: https://codecov.io/gh/steven-murray/hmf/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/steven-murray/hmf
.. image:: https://img.shields.io/pypi/pyversions/hmf.svg
    :target: https://pypi.org/project/hmf/
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

`hmf` is a python application that provides a flexible and simple way to calculate the
Halo Mass Function for a range of varying parameters. It is also the backend to
`HMFcalc <http://hmf.icrar.org>`_, the online HMF calculator.

Note
~~~~
From v3.1, ``hmf`` supports Python 3.6+, and has dropped support for Python 2.

Documentation
-------------
`Read the docs. <http://hmf.readthedocs.org>`_

Versioning
----------
From v3.1.0, ``hmf`` will be using strict semantic versioning, such that increases in
the **major** version have potential API breaking changes, **minor** versions introduce
new features, and **patch** versions fix bugs and other non-breaking internal changes.

If your package depends on ``hmf``, set the dependent version like this::

    hmf>=3.1<4.0

Attribution
-----------
Please cite `Murray, Power and Robotham (2013)
<https://arxiv.org/abs/1306.6721>`_ and/or https://ascl.net/1412.006 (whichever is more appropriate) if you find this
code useful in your research. Please also consider starring the GitHub repository.
