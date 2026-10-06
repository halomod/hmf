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

Each stage fills in its slots below, e.g.::

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
fsigma     2c (done)   f(sigma(M, z)) of each fit, on ``reference.m`` x ``case.z``
dndm       2b + 2c     dn/dM on ``reference.m`` x ``case.z``
ngtm       2b + 2c     n(>M) on ``reference.m`` x ``case.z``
========== =========== ================================================================
"""

import numpy as np
from regression_harness import Case, Reference, register_provider

from hmf.core.fits import FittingFunction, MeasuredMassDefinition
from hmf.core.units import Msun_h

# ---------------------------------------------------------------------------------
# fsigma: the fitting functions (step 2c)
# ---------------------------------------------------------------------------------

#: Parameters that undo v4's intentional changes to a v3.7.2 default, so that the
#: fit is compared like for like (see tests/core/test_core_fits_regression.py):
#: v4's Manera uses p = 0.248 (the l_link = 0.2 row of Manera et al. 2010), v3 0.289.
V3_PARAMETERS = {"Manera": {"p": 0.289}}

#: Mass definitions in which v3.7.2 evaluated a fit, where v4's differs: v4 evaluates
#: Watson (an SO-any fit) by default in SO-mean(178), where its Gamma = 1, as the paper
#: measured it; v3 preferred the virial definition.
V3_MASS_DEFINITIONS = {"Watson": MeasuredMassDefinition(kind="so_virial")}


def delta_halo_mean(mdef: MeasuredMassDefinition, cosmo, z: float) -> float | None:
    """The halo overdensity relative to the mean density of a measured mass definition.

    As v3.7.2 computes it (``halo_overdensity_mean``): Delta for SO-mean, Delta /
    Omega_m(z) for SO-critical, Bryan & Norman (1998) for SO-virial, and that of the
    preferred definition for SO-any. None for definitions with no overdensity (FoF,
    self-bound), whose fits do not take one.
    """
    if mdef.kind == "so_any":
        assert mdef.preferred is not None
        return delta_halo_mean(mdef.preferred, cosmo, z)
    om = float(cosmo.Om(z))
    if mdef.kind == "so_mean":
        return mdef.overdensity
    if mdef.kind == "so_critical":
        return mdef.overdensity / om
    if mdef.kind == "so_virial":
        x = om - 1
        return (18 * np.pi**2 + 82 * x - 39 * x**2) / om
    return None


def v4_fsigma(
    name: str,
    sigma: np.ndarray,
    *,
    z: float,
    cosmo,
    delta_c: float,
    n_eff: np.ndarray,
    m: np.ndarray,
) -> np.ndarray:
    """f(sigma) of the v4 fit ``name``, with the inputs v3.7.2 gave it.

    Each fit is evaluated in its v3 mass definition, with Omega_m(z) of ``cosmo``.
    v4's f(sigma) of the Bocquet 200c/500c fits leaves out the mass ratio M_Delta /
    M200m that v3 folded in, so it is applied here; and v4's intentional changes of a
    default (:data:`V3_PARAMETERS`, :data:`V3_MASS_DEFINITIONS`) are undone.

    Parameters
    ----------
    name
        The fit's (v3 and v4) name.
    sigma, n_eff, m
        sigma, the effective spectral index, and the mass in Msun/h (plain arrays).
    z, cosmo, delta_c
        The redshift, the astropy cosmology and the critical overdensity.

    Returns
    -------
    numpy.ndarray
        f(sigma).
    """
    fit = FittingFunction.get(name)(**V3_PARAMETERS.get(name, {}))
    mdef = V3_MASS_DEFINITIONS.get(name, fit.measured_mass_definition)
    f = fit.fsigma(
        sigma,
        z=z,
        omega_m_z=float(cosmo.Om(z)),
        delta_halo=delta_halo_mean(mdef, cosmo, z),
        delta_c=delta_c,
        n_eff=n_eff,
        m=m * Msun_h,
    )
    if hasattr(fit, "mass_ratio_to_200m"):
        f = f * fit.mass_ratio_to_200m(m * Msun_h, z=z, omega_m0=cosmo.Om0, h=cosmo.h)
    return np.asarray(f)


@register_provider("fsigma")
def fsigma(case: Case, reference: Reference) -> np.ndarray:
    """f(sigma) of the v4 fit, on the reference's own sigma(M, z).

    Feeding v4 the reference's sigma (and the n_eff from its dln sigma/dln M)
    isolates the fit: this compares :mod:`hmf.core.fits` alone with v3.7.2, with the
    inputs v3 gave it (see :func:`v4_fsigma`).
    """
    cosmo = reference.cosmology(case.cosmology)
    sigma = reference.sigma_at(case)
    dlnsdlnm = reference.values(
        reference.find(
            "dlnsdlnm", cosmology=case.cosmology, transfer=case.transfer, filter=case.filter
        )
    )
    n_eff = -3 * (2 * dlnsdlnm + 1)
    return np.array(
        [
            v4_fsigma(
                case.fit,
                sigma[i],
                z=z,
                cosmo=cosmo,
                n_eff=n_eff,
                m=reference.m,
                delta_c=reference.settings["delta_c"],
            )
            for i, z in enumerate(case.z or ())
        ]
    )
