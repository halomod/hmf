Massive neutrinos: which matter field to use
============================================

If your cosmology has massive neutrinos (the default ``Planck18`` cosmology does, with
:math:`\sum m_\nu = 0.06` eV), there are two different "matter" density fields you could
mean, and ``hmf`` lets you choose between them. This page explains the difference,
which one to use for what, and how to keep all the related settings consistent.

The two fields
--------------

``"cb"``
    Cold dark matter plus baryons, :math:`\delta_{\rm cb}`.

``"tot"``
    All non-relativistic matter, i.e. CDM, baryons *and* massive neutrinos:

    .. math:: \delta_{\rm tot} = (1 - f_\nu)\,\delta_{\rm cb} + f_\nu\,\delta_\nu,
              \qquad f_\nu = \Omega_\nu / \Omega_m.

The only extra mass in ``"tot"`` is the massive neutrinos. Their share is small:
:math:`\Omega_\nu h^2 \approx \sum m_\nu / 93.14\,{\rm eV}`, so :math:`f_\nu \approx 0.5\%`
for :math:`\sum m_\nu = 0.06` eV and about 2% for 0.3 eV.

Neutrinos are hot. On scales larger than their free-streaming length they fall into
potential wells like everything else, so :math:`\delta_\nu = \delta_{\rm cb}` and the two
fields are identical. On smaller scales the neutrinos stay smooth
(:math:`\delta_\nu \to 0`), so

.. math:: P_{\rm tot}(k) \to (1 - f_\nu)^2\, P_{\rm cb}(k).

The transition happens between the scale on which the neutrinos became
non-relativistic, :math:`k_{\rm nr} \approx 0.018\,\Omega_m^{1/2}\,(m_\nu /
1\,{\rm eV})^{1/2}\,h/{\rm Mpc}`, and today's free-streaming scale,
:math:`k_{\rm fs} \approx 0.82\,(m_\nu / 1\,{\rm eV})\,h/{\rm Mpc}` (for each neutrino
species of mass :math:`m_\nu`; Lesgourgues & Pastor 2006). For realistic masses the
scales that form halos (:math:`k \gtrsim 0.1\,h/{\rm Mpc}`) are well inside this
transition or beyond it. With massless neutrinos the two fields are the same and none of
this matters.

Note that this is a different effect from the overall slow-down of structure growth
caused by massive neutrinos (a massive-neutrino cosmology has less small-scale power
than a massless one, even in :math:`P_{\rm cb}`). CAMB includes that effect in both
fields. The choice here only decides whether the smooth neutrino component is counted
in the field you look at.

Which field should I use?
-------------------------

It depends on what responds to the field.

Halo mass function and halo bias: ``"cb"``
    Halos are made of CDM and baryons; neutrinos are mostly too fast to be captured.
    Simulations with massive neutrinos show that the halo mass function is universal
    (the same fit works across cosmologies) only when :math:`\sigma(M)` is computed
    from :math:`P_{\rm cb}` and masses are related to radii with
    :math:`\bar\rho_{\rm cb}` (Costanzi et al. 2013; Castorina et al. 2014). The same
    is true of halo bias, which is scale-independent only with respect to
    :math:`\delta_{\rm cb}`. Using :math:`P_{\rm tot}` underestimates :math:`\sigma(M)`
    by about :math:`f_\nu`, which the exponential tail of the mass function amplifies
    into a larger error in the abundance of massive clusters.

Gravitational lensing: ``"tot"``
    Light is deflected by all of the mass, neutrinos included.

Galaxy clustering: ``"cb"``
    Galaxies live in halos, so they trace :math:`\delta_{\rm cb}`:
    :math:`P_{gg} \approx b^2 P_{\rm cb}`.

Since ``hmf`` is a halo mass function code, it uses ``"cb"`` by default.

.. note::

    Earlier versions of ``hmf`` used ``"tot"``. If you don't set ``matter_species``
    and your cosmology has massive neutrinos, ``hmf`` warns you that the default has
    changed. Set the species explicitly to silence the warning.

    For the default ``Planck18`` cosmology (:math:`\sum m_\nu = 0.06` eV) and
    ``sigma_8``, the new defaults raise :math:`\sigma(M)` by 0.45% (that is,
    :math:`1/(1 - f_\nu)`). This changes the mass function by about :math:`-0.2\%` below
    :math:`10^{12}\,M_\odot/h`, :math:`+0.7\%` at :math:`10^{14}\,M_\odot/h` and
    :math:`+2.9\%` at :math:`10^{15}\,M_\odot/h`, with larger changes for heavier
    neutrinos.

Keeping everything consistent
-----------------------------

Four settings are involved.

The transfer function: ``transfer_params={"matter_species": ...}``
    Used by the ``CAMB`` and ``FromFile`` transfer models. ``FromFile`` reads column 7
    (``"cb"``) or column 6 (``"tot"``) of a CAMB transfer file. A two-column
    ``(k, T)`` file is used as-is, so make sure it holds the field you want.

The growth function: ``growth_params={"matter_species": ...}``
    Only used by :class:`~hmf.cosmology.growth_factor.CambGrowth`; set it to the same
    value as the transfer function. The other growth models are scale-independent and
    use ``astropy``'s :math:`\Omega_m(z)`, which excludes massive neutrinos. That is the
    growth of :math:`\delta_{\rm cb}` below the free-streaming scale, i.e. on the
    scales that matter for halos.

The mean density: nothing to set
    ``mean_density0`` is computed from ``astropy``'s ``Om0``, which is CDM plus baryons
    only (massive neutrinos are in ``Onu0``). It is therefore always
    :math:`\bar\rho_{\rm cb}`, which is what ``"cb"`` needs.

The normalisation: ``sigma_8_species``
    Measured values of :math:`\sigma_8`, such as Planck's (and ``hmf``'s default), are
    for the *total* matter field, so by default ``sigma_8_species="tot"``: with
    ``"cb"``, ``hmf`` normalises :math:`P_{\rm cb}` so that the total matter field from
    the same transfer model has the given :math:`\sigma_8`. The resulting
    :math:`\sigma_{8,\rm cb}` is slightly larger than :math:`\sigma_8`, by at most a
    factor :math:`1/(1 - f_\nu)`. This needs a second transfer calculation (with CAMB,
    a second CAMB run), unless the neutrinos are massless. If your value of
    :math:`\sigma_8` is for the CDM+baryon field (e.g. from a simulation), set
    ``sigma_8_species="cb"``.

Recipes
-------

Halo mass function with a measured (total-matter) :math:`\sigma_8`. This is the
fully consistent choice for most uses, and the default; setting ``matter_species``
explicitly just silences the warning::

    from hmf import MassFunction

    mf = MassFunction(sigma_8=0.81, transfer_params={"matter_species": "cb"})

Halo mass function with :math:`\sigma_8` of the CDM+baryon field, e.g. to match a
simulation that quotes it::

    mf = MassFunction(
        sigma_8=0.82, sigma_8_species="cb", transfer_params={"matter_species": "cb"}
    )

The same, if you use :class:`~hmf.cosmology.growth_factor.CambGrowth` (e.g. for a
wCDM cosmology)::

    mf = MassFunction(
        sigma_8=0.81,
        transfer_params={"matter_species": "cb"},
        growth_model="CambGrowth",
        growth_params={"matter_species": "cb"},
    )

Reproducing results from earlier versions of ``hmf`` (if you use ``CambGrowth``, also
pass ``growth_params={"matter_species": "tot"}``; the other growth models don't take
this parameter)::

    mf = MassFunction(transfer_params={"matter_species": "tot"})

The total-matter power spectrum, e.g. for lensing::

    from hmf import Transfer

    t = Transfer(sigma_8=0.81, transfer_params={"matter_species": "tot"})
    t.power

References
----------

- Castorina, E. et al. 2014, *Cosmology with massive neutrinos II: on the universality
  of the halo mass function and bias*, `arXiv:1311.1212 <https://arxiv.org/abs/1311.1212>`_
- Costanzi, M. et al. 2013, *Cosmology with massive neutrinos III: the halo mass
  function and an application to galaxy clusters*,
  `arXiv:1311.1514 <https://arxiv.org/abs/1311.1514>`_
- Lesgourgues, J. & Pastor, S. 2006, *Massive neutrinos and cosmology*, Physics
  Reports 429, 307
