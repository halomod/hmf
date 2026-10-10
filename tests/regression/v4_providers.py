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
transfer   2a (done)   T(k) on ``reference.lnk``, per cosmology, transfer and species
power      3 (done)    P(k, z=0) on ``reference.lnk``, normalised to sigma_8
growth     2a (done)   D(z) on ``reference.z_growth``
sigma      3 (done)    sigma(M, z) on ``reference.m`` x ``case.z``, per filter
dlnsdlnm   3 (done)    dln sigma/dln M on ``reference.m``
fsigma     2c (done)   f(sigma(M, z)) of each fit, on ``reference.m`` x ``case.z``
dndm       3 (done)    dn/dM on ``reference.m`` x ``case.z``
ngtm       3 (done)    n(>M) on ``reference.m`` x ``case.z``
========== =========== ================================================================
"""

import functools
import warnings
from typing import ClassVar

import attrs
import numpy as np
from regression_harness import Case, Reference, register_provider

from hmf.core.fits import FittingFunction, MeasuredMassDefinition, Watson
from hmf.core.growth import Growth
from hmf.core.linear_power import LinearPower
from hmf.core.mass_function import MassFunction
from hmf.core.mass_variance import MassVariance, n_eff_kernel
from hmf.core.species import omega_m, omega_m0
from hmf.core.transfer import Transfer
from hmf.core.transfer_models import CAMB, FromArray
from hmf.core.units import Msun_h, h_Mpc
from hmf.exceptions import HMFExtrapolationWarning

# ---------------------------------------------------------------------------------
# transfer and growth (step 2a)
# ---------------------------------------------------------------------------------

#: v3.7.2 ran CAMB with CAMBparams' default ``Transfer.kmax`` (0.9 / Mpc), and
#: extrapolated T(k) beyond it with EH98's shape (``extrapolate_with_eh``). The
#: provider runs v4's CAMB to the same k_max, so that the two compare the same CAMB
#: output (v4's default is 20 h/Mpc).
V3_CAMB_KMAX_MPC = 0.9


@register_provider("transfer")
def transfer(case: Case, reference: Reference) -> np.ndarray:
    """T(k) of the v4 Transfer stage, with v3.7.2's CAMB settings and normalisation.

    v4 normalises T to 1 as k -> 0; v3.7.2 to 1 at CAMB's smallest k (7.4e-5 h/Mpc),
    where T = 1 - 5.6e-5. So a CAMB case is divided by its value there (EH, a fit,
    is 1 at k -> 0 in both).
    """
    cosmo = reference.cosmology(case.cosmology)
    species = case.species or "cb"
    if case.transfer == "CAMB":
        model = CAMB(k_max=V3_CAMB_KMAX_MPC / float(cosmo.h) * h_Mpc)
    else:
        model = case.transfer
    stage = Transfer(cosmology=cosmo, model=model)
    t = stage.transfer_function(np.exp(reference.lnk) * h_Mpc, species)
    k_min = stage.solution.k_min_table
    if k_min is not None:
        t = t / stage.transfer_function(k_min * h_Mpc, species)
    return np.asarray(t)


@register_provider("growth")
def growth(case: Case, reference: Reference) -> np.ndarray:
    """D(z) of the v4 Growth stage with its default model, the growth ODE.

    The reference is v3.7.2's ODEGrowthFactor, for every cosmology.
    """
    stage = Growth(cosmology=reference.cosmology(case.cosmology), model="ODE")
    return np.asarray(stage.growth_factor(reference.z_growth))


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


#: Fits whose v4 modify_dndm_kernel v3.7.2 applied to dn/dm only, so that its f(sigma)
#: does not include it. v3 folded the other fits' modification (Bocquet's mass ratio
#: M_Delta/M200m) into f(sigma).
V3_DNDM_ONLY = frozenset({"Behroozi"})


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

    Each fit is evaluated in its v3 mass definition, with Omega_m(z) of CDM +
    baryons of ``cosmo`` and the overdensity of that definition. v4 applies the mass
    ratio M_Delta / M200m of the Bocquet 200c/500c fits to dn/dm, through
    ``modify_dndm_kernel``, but v3 folded it into f(sigma); dn/dm is proportional to
    f(sigma) at fixed m and the ratio is a factor, so it is applied to f here (except
    for :data:`V3_DNDM_ONLY`). And v4's intentional changes of a default
    (:data:`V3_PARAMETERS`, :data:`V3_MASS_DEFINITIONS`) are undone.

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
    om = omega_m(cosmo, z, "cb")
    needs_delta = "delta_halo" in fit.domain_inputs()
    f = fit.fsigma(
        sigma,
        z=z,
        omega_m_z=om,
        delta_halo=mdef.delta_halo_mean_kernel(om) if needs_delta else None,
        delta_c=delta_c,
        n_eff=n_eff,
        m=m * Msun_h,
    )
    if fit.modifies_dndm and name not in V3_DNDM_ONLY:
        # n(>m) is not needed by these; NaN makes any use of it fail the comparison.
        f = fit.modify_dndm_kernel(
            m,
            f,
            z=z,
            ngtm=np.full_like(f, np.nan),
            h=float(cosmo.h),
            omega_m0=omega_m0(cosmo, "cb"),
        )
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
    n_eff = n_eff_kernel(dlnsdlnm)
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


# ---------------------------------------------------------------------------------
# power, sigma, dlnsdlnm, dndm and ngtm: the LinearPower and MassFunction stages (step 3)
# ---------------------------------------------------------------------------------


@attrs.frozen(kw_only=True)
class WatsonVirial(Watson, abstract=True):
    """Watson (2013), measured in the virial definition, as v3.7.2 evaluated it.

    v4's Watson prefers SO-mean(178) (see :data:`V3_MASS_DEFINITIONS`); a mass-function
    stage evaluates a fit in its own measured definition, so the v3 one is a subclass.
    It is not registered (``abstract=True``): the stage takes it as an instance.
    """

    measured_mass_definition: ClassVar[MeasuredMassDefinition] = V3_MASS_DEFINITIONS["Watson"]


def v3_fit(name: str) -> FittingFunction:
    """The v4 fit ``name`` with v3.7.2's parameters and mass definition."""
    if name == "Watson":
        return WatsonVirial()
    return FittingFunction.get(name)(**V3_PARAMETERS.get(name, {}))


def _reference_transfer(reference: Reference, cosmology: str) -> FromArray:
    """v3.7.2's own CAMB T(k) of both species, as a v4 FromArray model.

    The stages built on the transfer function are compared on v3's T(k), so that the
    comparison is of them alone: CAMB's T(k) itself is compared by the ``transfer``
    provider, and above the k_max of v3's CAMB run v3 and v4 extrapolate T differently
    (by up to 5.7e-3, see tolerances.json), which changes sigma at 1e6 Msun/h by
    1.4e-3. The table starts at CAMB's lowest k (below it v3's T is an artefact, NaN in
    the reference); v4 extrapolates it below, and above k = e^12 h/Mpc, which changes
    sigma by < 1e-9 (the extrapolation warnings are expected, and silenced).
    """
    t_cb, t_tot = (
        reference.values(
            reference.find("transfer", cosmology=cosmology, transfer="CAMB", species=s)
        )
        for s in ("cb", "tot")
    )
    ok = np.isfinite(t_cb) & np.isfinite(t_tot)
    return FromArray(k=np.exp(reference.lnk[ok]) * h_Mpc, t=t_cb[ok], t_tot=t_tot[ok])


@functools.cache
def _linear_power(reference: Reference, cosmology: str, transfer_model: str) -> LinearPower:
    """The v4 LinearPower of v3.7.2's settings: P_cb normalised by sigma_8 of total matter.

    For CAMB, on v3's own T(k) (see :func:`_reference_transfer`). The ODE growth (the
    reference's ODEGrowthFactor), and sigma_8 and n_s of the reference's settings.
    """
    settings = reference.settings
    model = _reference_transfer(reference, cosmology) if transfer_model == "CAMB" else "EH"
    stage = Transfer(cosmology=reference.cosmology(cosmology), model=model, n_s=settings["n"])
    return LinearPower(
        transfer=stage,
        growth=Growth.from_transfer(stage, model="ODE"),
        sigma_8=settings["sigma_8"],
        species="cb",
        sigma_8_species="tot",
    )


@functools.cache
def _mass_function(reference: Reference, cosmology: str, transfer: str, filt: str) -> MassFunction:
    """The v4 MassFunction of a case (its fit is set per case with ``evolve``)."""
    lp = _linear_power(reference, cosmology, transfer)
    variance = MassVariance(power=lp.power_source, filter=filt, k_accuracy=lp.k_accuracy)
    return MassFunction(linear_power=lp, variance=variance, delta_c=reference.settings["delta_c"])


def _quiet_extrapolation(fn):
    """Silence the expected warnings of extrapolating v3's T(k) table (see above)."""

    @functools.wraps(fn)
    def wrapper(case: Case, reference: Reference) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", HMFExtrapolationWarning)
            return fn(case, reference)

    return wrapper


@register_provider("power")
@_quiet_extrapolation
def power(case: Case, reference: Reference) -> np.ndarray:
    """P(k, z = 0) of the v4 LinearPower stage, on the reference's k grid."""
    lp = _linear_power(reference, case.cosmology, case.transfer)
    return np.asarray(lp.power_kernel(k=np.exp(reference.lnk), z=0.0))


@register_provider("sigma")
@_quiet_extrapolation
def sigma(case: Case, reference: Reference) -> np.ndarray:
    """sigma(M, z) of the v4 MassFunction stage."""
    mf = _mass_function(reference, case.cosmology, case.transfer, case.filter)
    z = np.asarray(case.z or (), dtype=float)
    return np.exp(mf.ln_sigma_and_slope_kernel(m=reference.m[None, :], z=z[:, None])[0])


@register_provider("dlnsdlnm")
@_quiet_extrapolation
def dlnsdlnm(case: Case, reference: Reference) -> np.ndarray:
    """Dln sigma/dln M of the v4 MassFunction stage (the same at every z)."""
    mf = _mass_function(reference, case.cosmology, case.transfer, case.filter)
    return np.asarray(mf.ln_sigma_and_slope_kernel(m=reference.m, z=0.0)[1])


def _fit_case(case: Case, reference: Reference) -> tuple[MassFunction, np.ndarray]:
    mf = _mass_function(reference, case.cosmology, case.transfer, case.filter)
    return mf.evolve(fit=v3_fit(case.fit)), np.asarray(case.z or (), dtype=float)


@register_provider("dndm")
@_quiet_extrapolation
def dndm(case: Case, reference: Reference) -> np.ndarray:
    """dn/dM of the v4 MassFunction stage, each fit in its v3 mass definition."""
    mf, z = _fit_case(case, reference)
    return mf.dndm_kernel(m=reference.m[None, :], z=z[:, None])


@register_provider("ngtm")
@_quiet_extrapolation
def ngtm(case: Case, reference: Reference) -> np.ndarray:
    """n(>M) of the v4 MassFunction stage, integrated on its mass lattice to 10^17.5."""
    mf, z = _fit_case(case, reference)
    return mf.ngtm_kernel(m=reference.m[None, :], z=z[:, None])
