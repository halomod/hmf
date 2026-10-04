"""The v4 providers of the regression reference: one per quantity.

A provider computes one :class:`~regression_harness.Case` of the v3.7.2 reference
with :mod:`hmf.core`, and returns it with the shape of ``reference.values(case)``,
either as a Quantity or as an array in the reference's units (``reference.metadata
["units"]``). It returns ``None`` for a case it does not support yet (e.g. a filter
that v4 does not have), which is then skipped. ``test_regression_v4.py`` runs every
case of every quantity that has a provider, and skips the quantities that have none.

The cases tell the provider what to compute: ``case.cosmology`` (build it with
``reference.cosmology(case.cosmology)``), ``case.transfer`` (``"CAMB"`` or ``"EH"``),
``case.species``, ``case.filter``, ``case.fit`` (v3 names) and ``case.z``; the grids
are ``reference.m``, ``reference.lnk`` and ``reference.z_growth``, and the shared v3
settings (sigma_8, n, delta_c, ...) are ``reference.settings``. Note that the
reference evaluates every fit in its own measured mass definition (no mass
conversion), and uses the ODE growth factor for every cosmology.

Each step-2 PR fills in its slots below, e.g.::

    @register_provider("growth")
    def growth(case, reference):
        ...

Slots:

========== =========== ================================================================
Quantity   Step        Compared on
========== =========== ================================================================
transfer   2a          T(k) on ``reference.lnk``, per cosmology, transfer and species
power      2a/2b       P(k, z=0) on ``reference.lnk``, normalised to sigma_8
growth     2a          D(z) on ``reference.z_growth``
sigma      2b          sigma(M, z) on ``reference.m`` x ``case.z``, per filter
dlnsdlnm   2b          dln sigma/dln M on ``reference.m``
fsigma     2c          f(sigma(M, z)) of each fit, on ``reference.m`` x ``case.z``
dndm       2b + 2c     dn/dM on ``reference.m`` x ``case.z``
ngtm       2b + 2c     n(>M) on ``reference.m`` x ``case.z``
========== =========== ================================================================
"""

from regression_harness import register_provider  # noqa: F401

# No v4 stage exists yet: the step-2 PRs register their providers here.
