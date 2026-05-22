Colossus Comparison Notes
=========================

This note documents the known causes of differences between ``hmf`` and
`Colossus <https://bdiemer.bitbucket.io/colossus/>`_ for the native
``Tinker08`` halo mass function.
In versions pre-3.6.1, a major difference was the growth factor computation, which was
definitively less accurate in ``hmf`` than in Colossus. However, after
fixing the growth factor implementation in
`this PR <https://github.com/halomod/hmf/pull/270>`_ for v3.6.0, and then tightening
the growth-selector threshold in v3.6.1, some small residual differences remain,
particularly at high redshift.

The goal of this note is not to argue that either code is definitively "more
correct". Instead, it records the main implementation choices that explain the
observed residual differences so that users and developers understand where
agreement is expected and where small systematic offsets are normal.

Setup of the comparison
-----------------------

The comparisons discussed here were made with:

- the native ``200m`` form of ``Tinker08``,
- matched flat cosmologies with ``H0 = 67.74``, ``Om0 = 0.3089``,
  ``Ob0 = 0.0486``, ``sigma8 = 0.8159``, and ``ns = 0.9667``,
- the ``EH`` transfer model in ``hmf``, and
- direct comparisons of ``dndlnm`` at representative masses
  :math:`10^{11}`, :math:`10^{12}`, and :math:`10^{13}\,M_\odot/h`.

With this setup, the mismatch is small at low redshift and grows
toward high redshift, particularly in the rare-halo tail.

What is *not* driving the mismatch
----------------------------------

Several obvious suspects were checked and found not to be the dominant cause:

- **Mass-definition conversion:** this comparison uses native ``200m``
  ``Tinker08`` predictions, so it does not rely on the mass-definition
  conversion path.
- **Transfer-function normalization at** :math:`z=0`: ``hmf`` and Colossus
  agree on :math:`\sigma(M,0)` at about the :math:`10^{-4}` level for the
  matched setup.
- **Slope term:** the logarithmic slope entering ``dndlnm``,
  :math:`d \ln \sigma / d \ln R`, agrees at the sub-:math:`0.1\%` level.

The main causes of the residual difference
------------------------------------------

Two effects dominate the remaining mismatch.

High-redshift growth implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After the selector update, ``hmf`` uses the full ODE growth solution whenever
the radiation fraction exceeds the calibrated threshold (essentially for z>1.5).
Colossus uses a different hybrid approach for LCDM cosmologies:

- an analytic matter-radiation approximation at high redshift, and
- an integral solution at low redshift,

with a transition regime between them.

These are both reasonable algorithmic choices, but they do not produce exactly
the same high-redshift growth history. In the matched comparison used here,
``hmf`` ends up with a slightly larger :math:`\sigma(M, z)` than Colossus at
high redshift, typically by about :math:`0.26`--:math:`0.34\%` over
``z = 6``--``10`` for the masses tested.

That difference is tiny in :math:`\sigma` itself, but it is evaluated in the
exponential tail of the halo mass function, where very small shifts in
:math:`\sigma` can produce multi-percent shifts in abundance.

Tinker08 coefficient precision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two codes also do not use numerically identical ``Tinker08`` coefficients.

``hmf`` stores the more precise coefficient values, for example:

- ``A_200 = 0.1858659``
- ``a_200 = 1.466904``
- ``b_200 = 2.571104``
- ``c_200 = 1.193958``

Colossus uses the rounded table values:

- ``A_200 = 0.186``
- ``a_200 = 1.47``
- ``b_200 = 2.57``
- ``c_200 = 1.19``

At fixed :math:`\sigma`, this coefficient rounding changes :math:`f(\sigma)` by
only a little at low redshift, but by several percent in the high-redshift
tail. In the matched tests performed here, the coefficient choice alone
accounts for approximately:

- :math:`2`--:math:`6\%` at ``z = 6``,
- :math:`3`--:math:`10\%` at ``z = 8``, and
- :math:`4`--:math:`15\%` at ``z = 10``,

depending on mass.

How the effects combine
-----------------------

The two dominant effects push in opposite directions:

- the slightly larger high-redshift :math:`\sigma(M, z)` in ``hmf`` tends to
  increase the abundance relative to Colossus, while
- the more precise ``Tinker08`` coefficients in ``hmf`` tend to decrease the
  abundance relative to Colossus' rounded table.

Because these effects partially cancel, the final mismatch is significantly
smaller than either ingredient by itself. For the matched setup used in the
external regression test, the resulting ``hmf`` versus Colossus difference is
typically of order:

- ``z = 6``: :math:`\sim 0.1\%` to :math:`2.7\%`,
- ``z = 8``: :math:`\sim 1\%` to :math:`7\%`,
- ``z = 10``: :math:`\sim 2\%` to :math:`14\%`,

over the range :math:`10^{11}`--:math:`10^{13}\,M_\odot/h`.
