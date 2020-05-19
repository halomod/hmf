Installation
============

This page will guide you through installing ``hmf`` -- either as purely a user, or
as a potential developer.

Dependencies
------------
``hmf`` has a number of dependencies, all of which should be automatically installed
as you install the package itself. You therefore do not need to worry about installing
them yourself, except in some circumstances.

If you are using ``conda`` to create your Python environments, you may wish to manually
install the dependencies using ``conda`` (as they will be installed automatically with
``pip`` otherwise). For a user install, this can be done with the following::

    conda install -c conda-forge numpy scipy astropy camb

If you do not wish to install ``camb`` using ``conda``, you will need to ensure you
have a fortran compiler on your system path. If using ``gcc``, the version needs to be
greater than v6.


User Install
------------
You may install the latest release of ``hmf`` using ``pip``::

    pip install hmf

This will install all uninstalled dependencies (see previous section).
Alternatively, for the very bleeding edge, install from the master branch of the repo::

    pip install hmf @ git+git://github.com/steven-murray/hmf.git

Developer Install
-----------------
If you intend to develop ``hmf``, clone the repository (or your fork of it)::

    git clone https://github.com/<your-username>/hmf.git

Move to the directory and install with::

    pip install -e ".[dev]"

This will install all dependencies -- both for using and developing the package (testing,
creating docs, etc.). Again, see above about dependencies with ``conda`` if you are
using a ``conda`` environment (which is recommended).

.. note:: Once the package is installed, you will need to locally run ``pre-commit install``,
          to have constant checks on your code formatting before commits are accepted.
