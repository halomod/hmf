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

This investigation was itself prompted by Aaron Yung and Miguel Vargas, who
independently noticed disagreement between ``hmf`` and Colossus predictions for
the Tinker08 mass function while preparing [Yung24]_ and [Yung25]_. Their reports
are what led to the growth-factor fix linked above, and we are grateful to them
for flagging the discrepancy and giving permission to reference their work here.

Setup of the comparison
-----------------------

The comparisons discussed here were made with:

- the native ``200m`` form of ``Tinker08``,
- matched flat cosmologies with ``H0 = 67.74``, ``Om0 = 0.3089``,
  ``Ob0 = 0.0486``, ``sigma8 = 0.8159``, and ``ns = 0.9667``, and massless
  neutrinos (``m_nu = 0`` in ``hmf``, whose default cosmology has 0.06 eV,
  because Colossus has no massive-neutrino background),
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

Three effects matter. Two are input mismatches that the regression test
removes, and one is a genuine algorithmic difference.

Massive-neutrino background
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default ``hmf`` cosmology carries a 0.06 eV neutrino. Colossus has no
massive-neutrino background. At fixed ``Om0`` (which in astropy excludes
neutrinos) the extra non-relativistic density changes :math:`E(z)` and so the
growth history. With :math:`\sigma_8` fixed at :math:`z=0`, ``hmf`` then has a
larger :math:`\sigma(M, z)` at high redshift. This was the main source of the
"slightly larger high-redshift :math:`\sigma`" previously attributed to the
growth algorithm. At fixed Tinker08 coefficients it raises the ``hmf``
abundance by about 6--47% at ``z = 8``--``10`` over
:math:`10^{11}`--:math:`10^{13}\,M_\odot/h`. Set ``m_nu = 0`` to compare
like with like.

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
tail. The more precise ``hmf`` coefficients lower the abundance relative to
Colossus by up to about 6% at ``z = 6`` and 14% at ``z = 10``, depending on mass.
The regression test passes the rounded values through ``hmf_params``.

High-redshift growth implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``hmf`` uses the full ODE growth solution whenever the radiation fraction
exceeds the calibrated threshold (essentially for z>1.5). Colossus uses a
hybrid approach for LCDM cosmologies:

- an analytic matter-radiation approximation at high redshift, and
- an integral solution at low redshift,

with a transition regime between them.

With cosmology and coefficients matched, this is the only remaining
difference. Against CAMB's CDM+baryon growth at :math:`k/h = 5`, the Colossus
growth factor is 5--7 :math:`\times 10^{-4}` low at ``z = 4``--``10``, while
``hmf`` is within :math:`2\times 10^{-4}`. The halo abundance is very
sensitive to :math:`\sigma` in the exponential tail, so the residual is
:math:`\approx (d\ln f/d\ln\sigma)\,\delta\ln D`. That predicts 0.8%, 1.9%,
2.7% and 3.7% at :math:`10^{13}\,M_\odot/h` for ``z = 4, 6, 8, 10``, against
measured values of 0.8%, 1.9%, 2.9% and 4.0%.

Resulting agreement
-------------------

With the cosmology and the Tinker08 coefficients matched, the ``hmf`` versus
Colossus difference over :math:`10^{11}`--:math:`10^{13}\,M_\odot/h` is:

- ``z = 0``--``2``: below :math:`0.1\%`,
- ``z = 4``: up to :math:`0.8\%`,
- ``z = 6``: up to :math:`1.9\%`,
- ``z = 8``: up to :math:`2.9\%`,
- ``z = 10``: up to :math:`4.0\%`,

all at the high-mass end, with ``hmf`` above Colossus.

References
----------
.. [Yung24] Yung, L. Y. Aaron, Rachel S. Somerville, Tri Nguyen, Peter Behroozi, Chirag Modi, and Jonathan P. Gardner. 'Characterizing Ultra-High-Redshift Dark Matter Halo Demographics and Assembly Histories with the GUREFT Simulations'. Monthly Notices of the Royal Astronomical Society 530, no. 4 (2024): 4868-86. https://doi.org/10.1093/mnras/stae1188.
.. [Yung25] Yung, L. Y. Aaron, Rachel S. Somerville, and Kartheik G. Iyer. 'ΛCDM Is Still Not Broken: Empirical Constraints on the Star Formation Efficiency at z ~ 12-30'. Monthly Notices of the Royal Astronomical Society 543, no. 4 (2025): 3802-13. https://doi.org/10.1093/mnras/staf1699.
