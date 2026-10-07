r"""Halo mass function fitting functions, as :mod:`hmf.core` models.

A fitting function gives the multiplicity function :math:`f(\sigma) = \nu f(\nu)`, with
:math:`\nu = \delta_c/\sigma` the peak height, from which the mass function follows as

.. math:: \frac{dn}{dm} = f(\sigma)\frac{\bar\rho_0}{m^2}\left|\frac{d\ln\sigma}{d\ln m}\right|.

Every fit is a :class:`FittingFunction` model: a frozen ``attrs`` class whose fields
are its parameters (with hmf 3.x's names and defaults), registered under its hmf 3.x
name as an alias, so ``FittingFunction.get("Tinker08")`` finds it.

Inputs
------
:meth:`FittingFunction.fsigma` takes no cosmology and no mass-definition object, only
already-resolved physical inputs. Computing them (sigma from the power spectrum,
the overdensity from the mass definition, ...) is the job of the code that calls the
fit, normally a mass-function :class:`~hmf.core.stage.Stage`.

================  ===================================================================
``sigma``         :math:`\sigma(m, z)`; always required.
``z``             redshift.
``omega_m_z``     the density parameter of CDM + baryons at ``z``, :math:`\Omega_m(z)`.
``delta_halo``    the halo overdensity relative to the **mean** density, :math:`\Delta_m`.
``delta_c``       the critical overdensity for collapse, :math:`\delta_c`.
``n_eff``         the effective spectral index at ``m``.
``m``             the halo mass.
================  ===================================================================

Each fit lists the ones it needs in :attr:`FittingFunction.requires`. All of them
broadcast against each other.

:math:`\Omega_m` (``omega_m_z``, and ``omega_m0`` of :meth:`FittingFunction.modify_dndm`)
is that of CDM + baryons, the matter species ``"cb"``
(:func:`hmf.core._species.omega_m`), as is the mean density of the mass function:
the fits are universal in the CDM + baryon field when neutrinos are massive (Costanzi
et al. 2013; Castorina et al. 2014), and none was calibrated with massive neutrinos.
The overdensity of a fit's own mass definition follows from it with
:meth:`MeasuredMassDefinition.delta_halo_mean_kernel`, and ``n_eff`` from the slope
of sigma with :func:`hmf.core.mass_variance.n_eff_kernel`.

Evaluating a fit
----------------
There are two ways to call a fit, which differ only in how the mass is given:

With Quantities
    :meth:`FittingFunction.fsigma` (and the fit's other public methods) take the
    mass as an :class:`~astropy.units.Quantity`, e.g. ``m=1e12 * Msun_h``, as every
    public method of :mod:`hmf.core` does (see :mod:`hmf.core.units`). The other
    inputs are dimensionless numbers or arrays. This is the way to use a fit directly.
With plain arrays in canonical units
    Code that already holds every input as a plain array in the canonical units of
    :data:`hmf.core.units.CANONICAL_UNITS` (mass in Msun/h), such as a
    :class:`~hmf.core.stage.Stage`, puts them in a :class:`FitInputs` and calls
    :func:`evaluate_fsigma` (or :meth:`FittingFunction.fsigma_kernel`). This
    skips the unit checks and conversions, which matters inside loops, and is also
    where the domain policy (below) is applied.

Both give the same numbers. The kernel-level methods, which end in ``_kernel``,
follow the conventions of :mod:`hmf.core._kernels`.

Post-processing the mass function
---------------------------------
:math:`dn/dm` from :math:`f(\sigma)` (the equation above) is not the end for every
fit: :meth:`FittingFunction.modify_dndm` (and its kernel,
:meth:`FittingFunction.modify_dndm_kernel`) maps it to the fit's mass function. Every
fit has it, with the same arguments; it is the identity unless the class sets
:attr:`FittingFunction.modifies_dndm`:

* :class:`Behroozi` applies its high-redshift correction, which needs n(>m);
* the Bocquet 200c and 500c fits multiply by the ratio of masses
  :math:`M_\Delta/M_{200m}` of their eq. 5, which needs :math:`\Omega_{m,0}` and h.

So a mass function calls ``modify_dndm_kernel`` for every fit, and can skip it, and
computing n(>m) for it, when ``modifies_dndm`` is False.

Domains
-------
Each fit has two :class:`~hmf.core.domain.Domain`\ s (issue #390):

:attr:`FittingFunction.valid_domain`
    where its formula is defined, or sensible. :meth:`FittingFunction.fsigma`
    **always raises** :class:`~hmf.core.domain.DomainError` outside it.
:attr:`FittingFunction.calibration_domain`
    where it was calibrated against simulations, with its source (``None`` for a fit
    that was not, such as :class:`PS`). Outside it, the result is an extrapolation,
    handled by a :data:`~hmf.core.domain.DomainPolicy` with :func:`evaluate_fsigma`.

The bounds of both are in these variables: the inputs above, and the derived
``ln_sigma_inv`` (:math:`\ln\sigma^{-1}`), ``log10_sigma_inv``
(:math:`\log_{10}\sigma^{-1}`) and ``peak_height`` (:math:`\nu`).

Mass definitions
----------------
Each fit records the mass definition it was measured in, as a
:class:`MeasuredMassDefinition`, which also gives its overdensity relative to the
mean density (:meth:`MeasuredMassDefinition.delta_halo_mean_kernel`). Converting
between mass definitions is not part of this module (issue #392).
"""

from __future__ import annotations

import abc
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, ClassVar, Literal

import astropy.units as u
import attrs
import numpy as np
import numpy.typing as npt

from . import _references as refs
from ._arrays import float_array, optional_float_array
from ._fields import field
from ._kernels import fits as _k
from ._validators import check_finite_positive, less_than, positive
from .domain import Domain, DomainError, DomainPolicy, apply_domain_policy
from .model import Model
from .units import (
    RHO_CRIT0_H2,
    H0_unit,
    Mpc_h,
    Msun_h,
    UnitBoundaryError,
    UnitContext,
    dndm_unit,
    kpc_h,
    number_density_unit,
    quantity_field,
    rho_unit,
    to_canonical,
    unit_boundary,
)

__all__ = [
    "INPUTS",
    "PS",
    "SMT",
    "ST",
    "Angulo",
    "AnguloBound",
    "Behroozi",
    "Bhattacharya",
    "Bocquet200cDMOnly",
    "Bocquet200cHydro",
    "Bocquet200mDMOnly",
    "Bocquet200mHydro",
    "Bocquet500cDMOnly",
    "Bocquet500cHydro",
    "Courtin",
    "Crocce",
    "FSigmaResult",
    "FitInputs",
    "FittingFunction",
    "Ishiyama",
    "Jenkins",
    "Manera",
    "MeasuredMassDefinition",
    "Peacock",
    "Pillepich",
    "Reed03",
    "Reed07",
    "SimulationDetails",
    "Tinker08",
    "Tinker10",
    "Warren",
    "Watson",
    "Watson_FoF",
    "Yung24",
    "evaluate_fsigma",
]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

#: The physical inputs a fit may require, besides ``sigma`` (which all require).
INPUTS: tuple[str, ...] = ("z", "omega_m_z", "delta_halo", "delta_c", "n_eff", "m")

#: The bounds of a variable that must be > 0 (e.g. sigma, in every fit's valid domain).
_POSITIVE = (0, None, "(]")


# ---------------------------------------------------------------------------------
# Metadata: measured mass definitions
# ---------------------------------------------------------------------------------
MassDefinitionKind = Literal["fof", "so_mean", "so_critical", "so_virial", "so_any", "self_bound"]


@attrs.frozen(kw_only=True)
class MeasuredMassDefinition:
    """The halo mass definition a fit was measured in.

    This is a simple, hashable record, to be mapped onto the mass-definition types
    of issue #392. It does no conversion, but gives the overdensity relative to the
    mean density (:meth:`delta_halo_mean_kernel`), the ``delta_halo`` input of the
    fits.

    Parameters
    ----------
    kind
        ``"fof"`` (friends-of-friends), ``"so_mean"`` / ``"so_critical"``
        (spherical overdensity relative to the mean / critical density),
        ``"so_virial"`` (the Bryan & Norman 1998 virial overdensity), or ``"so_any"``
        (a fit parameterised by the SO overdensity, which accepts any SO definition),
        or ``"self_bound"`` (the bound mass of SUBFIND (sub)haloes).
    linking_length
        The FoF linking length b, in units of the mean interparticle separation.
    overdensity
        The SO overdensity, relative to the density named by ``kind``.
    preferred
        For ``"so_any"``: the definition used when none is given.
    note
        Anything else worth knowing (halo finder, unbinding, ...).
    """

    kind: MassDefinitionKind
    linking_length: float | None = None
    overdensity: float | None = None
    preferred: MeasuredMassDefinition | None = None
    note: str = ""

    def __attrs_post_init__(self) -> None:
        """Check that the fields given match ``kind``."""
        if (self.kind == "fof") != (self.linking_length is not None):
            raise ValueError("A linking length is given for, and only for, kind='fof'.")
        if (self.kind in ("so_mean", "so_critical")) != (self.overdensity is not None):
            raise ValueError("An overdensity is given for, and only for, SO-mean/critical.")
        if (self.kind == "so_any") != (self.preferred is not None):
            raise ValueError("A preferred definition is given for, and only for, 'so_any'.")

    def delta_halo_mean_kernel(self, omega_m_z: npt.ArrayLike) -> FloatArray:
        r"""The halo overdensity relative to the mean density, :math:`\Delta_m`.

        A pure, elementwise kernel of :math:`\Omega_m(z)` (of CDM + baryons, see
        the :mod:`module documentation <hmf.core.fits>`), on plain arrays:

        ``"so_mean"``
            the overdensity itself;
        ``"so_critical"``
            :math:`\Delta_c/\Omega_m(z)`;
        ``"so_virial"``
            Bryan & Norman (1998): :math:`(18\pi^2 + 82x - 39x^2)/\Omega_m(z)`, with
            :math:`x = \Omega_m(z) - 1` (so :math:`18\pi^2` in Einstein-de Sitter);
        ``"fof"``
            the overdensity of the isodensity surface the linking length b selects,
            for a singular isothermal sphere: :math:`9/(2\pi b^3)`, independent of
            :math:`\Omega_m(z)` (White, Hernquist & Springel 2001; about 179 for
            b = 0.2), as hmf 3.x's ``FOF``;
        ``"so_any"``
            that of the ``preferred`` definition.

        Parameters
        ----------
        omega_m_z
            :math:`\Omega_m(z)`, dimensionless.

        Returns
        -------
        numpy.ndarray
            :math:`\Delta_m`, dimensionless, with the shape of ``omega_m_z``.

        Raises
        ------
        ValueError
            For ``"self_bound"``, which has no overdensity.
        """
        if self.kind == "so_any":
            assert self.preferred is not None
            return self.preferred.delta_halo_mean_kernel(omega_m_z)
        om = np.asarray(omega_m_z, dtype=np.float64)
        if self.kind == "so_mean":
            assert self.overdensity is not None
            return np.full_like(om, self.overdensity)
        out: FloatArray
        if self.kind == "so_critical":
            assert self.overdensity is not None
            out = self.overdensity / om
            return out
        if self.kind == "so_virial":
            x = om - 1
            out = (18 * np.pi**2 + 82 * x - 39 * x**2) / om
            return out
        if self.kind == "fof":
            assert self.linking_length is not None
            return np.full_like(om, 9 / (2 * np.pi * self.linking_length**3))
        raise ValueError(f"The mass definition {self} has no overdensity.")

    def __str__(self) -> str:
        """A short label, e.g. ``FoF(b=0.2)`` or ``SO-crit(500)``."""
        if self.kind == "fof":
            return f"FoF(b={self.linking_length:g})"
        if self.kind == "so_mean":
            return f"SO-mean({self.overdensity:g})"
        if self.kind == "so_critical":
            return f"SO-crit({self.overdensity:g})"
        if self.kind == "so_virial":
            return "SO-virial"
        if self.kind == "self_bound":
            return "self-bound"
        return f"SO-any (preferred {self.preferred})"


def _fof(b: float = 0.2, note: str = "") -> MeasuredMassDefinition:
    return MeasuredMassDefinition(kind="fof", linking_length=b, note=note)


def _so_mean(overdensity: float, note: str = "") -> MeasuredMassDefinition:
    return MeasuredMassDefinition(kind="so_mean", overdensity=overdensity, note=note)


def _so_crit(overdensity: float, note: str = "") -> MeasuredMassDefinition:
    return MeasuredMassDefinition(kind="so_critical", overdensity=overdensity, note=note)


def _so_virial(note: str = "") -> MeasuredMassDefinition:
    return MeasuredMassDefinition(kind="so_virial", note=note)


def _so_any(preferred: MeasuredMassDefinition, note: str = "") -> MeasuredMassDefinition:
    return MeasuredMassDefinition(kind="so_any", preferred=preferred, note=note)


# ---------------------------------------------------------------------------------
# Metadata: the simulations a fit was calibrated on
# ---------------------------------------------------------------------------------

PerSimulation = tuple[Any, ...]


def _tuple_or_none(x: Any) -> PerSimulation | None:
    """Convert a scalar or sequence to a tuple (``None`` stays ``None``)."""
    if x is None:
        return None
    if isinstance(x, str) or np.ndim(x) == 0:
        return (x,)
    return tuple(x)


# eq=False: these are metadata about a fit (a class variable), never part of a model's
# value, so they are not compared or hashed by value, nor serialised.
@attrs.frozen(kw_only=True, eq=False)
class SimulationDetails:
    """The suite of simulations a fit was calibrated on (metadata only).

    This ports hmf 3.x's ``SimDetails``, corrected against the papers where it was
    wrong. It describes the simulations used to *define* the fit, not every one the
    paper compared it with. Per-simulation values are tuples (or Quantities) with
    one entry per simulation; a single value applies to all of them, and ``None``
    means the paper does not state it. Being metadata, it compares by identity, not
    by value.

    Parameters
    ----------
    box_size
        The comoving box sizes, a Quantity in h-units (stored in Mpc/h).
    n_particles
        The number of dark-matter particles in each simulation (hydrodynamical runs
        have as many gas particles again; see ``notes``).
    omega_m, omega_b, sigma_8, h, n_s
        The cosmological parameters of each simulation.
    softening
        The gravitational softening length (or, for AMR codes, the finest cell
        size), a Quantity in h-units (stored in Mpc/h).
    transfer
        The transfer function used for the initial conditions.
    z_start
        The starting redshift.
    initial_conditions
        How the initial conditions were made: ``"ZA"`` (the Zel'dovich
        approximation) or ``"2LPT"``.
    halo_finder
        The halo-finding code.
    n_min
        The minimum number of particles per halo used in the fit.
    z_range
        The redshift range over which the fit was measured, ``(min, max)``.
    names
        The simulations' names in the paper.
    notes
        Anything else worth knowing (corrections applied, other simulations, ...).
    source
        Where the details come from (paper, section, table).
    """

    box_size: u.Quantity = quantity_field(Mpc_h, ndim=1, doc="The comoving box sizes.")
    n_particles: PerSimulation = attrs.field(converter=_tuple_or_none)
    omega_m: PerSimulation | None = attrs.field(default=None, converter=_tuple_or_none)
    omega_b: PerSimulation | None = attrs.field(default=None, converter=_tuple_or_none)
    sigma_8: PerSimulation | None = attrs.field(default=None, converter=_tuple_or_none)
    h: PerSimulation | None = attrs.field(default=None, converter=_tuple_or_none)
    n_s: PerSimulation | None = attrs.field(default=None, converter=_tuple_or_none)
    softening: u.Quantity | None = quantity_field(
        Mpc_h, ndim=1, optional=True, default=None, doc="The gravitational softening lengths."
    )
    transfer: PerSimulation | None = attrs.field(default=None, converter=_tuple_or_none)
    z_start: PerSimulation | None = attrs.field(default=None, converter=_tuple_or_none)
    initial_conditions: PerSimulation | None = attrs.field(default=None, converter=_tuple_or_none)
    halo_finder: str | None = None
    n_min: int | None = None
    z_range: tuple[float, float] | None = None
    names: PerSimulation | None = attrs.field(default=None, converter=_tuple_or_none)
    notes: str = ""
    source: str = ""

    def __attrs_post_init__(self) -> None:
        """Broadcast single values to every simulation, and check the lengths."""
        n = len(self.box_size)
        for a in attrs.fields(type(self)):
            value = getattr(self, a.name)
            if value is None or a.name == "z_range" or not isinstance(value, (tuple, u.Quantity)):
                continue
            if len(value) == 1 and n > 1:
                value = value * n if isinstance(value, tuple) else np.repeat(value, n)
                object.__setattr__(self, a.name, value)
            if len(value) != n:
                raise ValueError(
                    f"SimulationDetails.{a.name} has {len(value)} entries for {n} simulations."
                )

    @property
    def n_simulations(self) -> int:
        """The number of simulations."""
        return len(self.box_size)

    @property
    def particle_mass(self) -> Any:
        """The dark-matter particle mass of each simulation, a Quantity in Msun/h.

        Computed as Omega_m rho_crit0 L^3 / N, so for hydrodynamical runs (where the
        dark matter has only Omega_m - Omega_b) it is the mass of a particle pair.
        ``None`` if Omega_m is not known for every simulation.
        """
        if self.omega_m is None or any(om is None for om in self.omega_m):
            return None
        volume = self.box_size.to_value(Mpc_h) ** 3
        mass = np.array(self.omega_m, dtype=float) * RHO_CRIT0_H2.to_value(rho_unit) * volume
        return (mass / np.array(self.n_particles, dtype=float)) << Msun_h


def _derived(
    base: SimulationDetails | None, *, prepend_note: str = "", **changes: Any
) -> SimulationDetails:
    """Another fit's simulation details, with some fields changed."""
    if base is None:
        raise TypeError("Can't derive simulation details from None.")
    if prepend_note:
        changes["notes"] = prepend_note + " " + base.notes
    return attrs.evolve(base, **changes)


# ---------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------
@attrs.frozen(kw_only=True)
class FitInputs:
    r"""The resolved physical inputs of a fit, as float arrays.

    Reading an input that was not given raises a :class:`ValueError` (rather than
    returning a sentinel). See the module documentation for each input's meaning.

    Parameters
    ----------
    sigma
        :math:`\sigma(m, z)`.
    z, omega_m_z, delta_halo, delta_c, n_eff, m
        The optional inputs (``m`` in canonical Msun/h); ``None`` if not given.
    """

    sigma: FloatArray = attrs.field(converter=float_array)
    _z: FloatArray | None = attrs.field(default=None, converter=optional_float_array, alias="z")
    _omega_m_z: FloatArray | None = attrs.field(
        default=None, converter=optional_float_array, alias="omega_m_z"
    )
    _delta_halo: FloatArray | None = attrs.field(
        default=None, converter=optional_float_array, alias="delta_halo"
    )
    _delta_c: FloatArray | None = attrs.field(
        default=None, converter=optional_float_array, alias="delta_c"
    )
    _n_eff: FloatArray | None = attrs.field(
        default=None, converter=optional_float_array, alias="n_eff"
    )
    _m: FloatArray | None = attrs.field(default=None, converter=optional_float_array, alias="m")

    def given(self, name: str) -> bool:
        """Whether the input ``name`` was given."""
        return name == "sigma" or getattr(self, f"_{name}") is not None

    def _get(self, name: str) -> FloatArray:
        value: FloatArray | None = getattr(self, f"_{name}")
        if value is None:
            raise ValueError(f"The input {name!r} is needed, but was not given.")
        return value

    @property
    def z(self) -> FloatArray:
        """Redshift."""
        return self._get("z")

    @property
    def omega_m_z(self) -> FloatArray:
        r""":math:`\Omega_m(z)`."""
        return self._get("omega_m_z")

    @property
    def delta_halo(self) -> FloatArray:
        r"""Halo overdensity relative to the mean density, :math:`\Delta_m`."""
        return self._get("delta_halo")

    @property
    def delta_c(self) -> FloatArray:
        r"""Critical overdensity for collapse, :math:`\delta_c`."""
        return self._get("delta_c")

    @property
    def n_eff(self) -> FloatArray:
        """Effective spectral index."""
        return self._get("n_eff")

    @property
    def m(self) -> FloatArray:
        """Halo mass, in canonical Msun/h."""
        return self._get("m")

    @property
    def nu(self) -> FloatArray:
        r"""The peak height, :math:`\nu = \delta_c/\sigma`."""
        return _k.peak_height(self.sigma, self.delta_c)

    @property
    def ln_sigma_inv(self) -> FloatArray:
        r""":math:`\ln\sigma^{-1}`."""
        return _k.ln_sigma_inv(self.sigma)

    def domain_value(self, name: str) -> Any:
        """The value of a domain variable (an input, or one derived from them).

        Masses are returned as Quantities in Msun/h, as :class:`Domain` expects.
        """
        derived: Mapping[str, Callable[[FitInputs], Any]] = _DERIVED
        if name in derived:
            return derived[name](self)
        if name == "m":
            return self.m << Msun_h
        if name == "sigma" or name in INPUTS:
            return self.sigma if name == "sigma" else self._get(name)
        raise ValueError(f"Unknown domain variable {name!r}.")


_DERIVED: Mapping[str, Callable[[FitInputs], Any]] = MappingProxyType(
    {
        "ln_sigma_inv": lambda x: x.ln_sigma_inv,
        "log10_sigma_inv": lambda x: -np.log10(x.sigma),
        "peak_height": lambda x: x.nu,
    }
)

#: The inputs each derived domain variable needs.
_DERIVED_NEEDS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {"ln_sigma_inv": (), "log10_sigma_inv": (), "peak_height": ("delta_c",)}
)


def _domain_inputs(domain: Domain | None) -> frozenset[str]:
    """The inputs (besides sigma) needed to check a domain (none for ``None``)."""
    needs: set[str] = set()
    for name in () if domain is None else domain.variables:
        if name in _DERIVED_NEEDS:
            needs.update(_DERIVED_NEEDS[name])
        elif name != "sigma":
            needs.add(name)
    return frozenset(needs)


def _contains(domain: Domain | None, x: FitInputs) -> bool | BoolArray:
    """Whether the inputs ``x`` are inside ``domain`` (everywhere, for ``None``)."""
    if domain is None:
        return True
    return domain.contains(**{name: x.domain_value(name) for name in domain.variables})


# ---------------------------------------------------------------------------------
# The kind
# ---------------------------------------------------------------------------------
@attrs.frozen(kw_only=True)
class FittingFunction(Model, kind=True):
    r"""A halo mass function fit: the multiplicity function :math:`f(\sigma)`.

    Subclasses implement :meth:`_fsigma`, and declare, as class variables, the
    inputs they need (:attr:`requires`), both domains, the mass definition they were
    measured in, ``references`` and ``parameter_source``.
    """

    #: The inputs (from :data:`INPUTS`) that :meth:`fsigma` needs, besides sigma.
    requires: ClassVar[frozenset[str]] = frozenset()

    #: Where the formula is defined or sensible: :meth:`fsigma` raises outside it.
    valid_domain: ClassVar[Domain]

    #: Where the fit was calibrated against simulations, with its source; ``None`` for
    #: a fit that was not calibrated (e.g. an analytic one).
    calibration_domain: ClassVar[Domain | None]

    #: The halo mass definition the fit was measured in.
    measured_mass_definition: ClassVar[MeasuredMassDefinition]

    #: The simulations the fit was calibrated on (None for a fit to no simulations).
    simulations: ClassVar[SimulationDetails | None]

    #: Whether :meth:`modify_dndm_kernel` changes the mass function (see
    #: "Post-processing the mass function" in the :mod:`module documentation
    #: <hmf.core.fits>`); if False, it is the identity.
    modifies_dndm: ClassVar[bool] = False

    #: Whether the fit is normalised by construction (see :attr:`normalized`).
    _normalized: ClassVar[bool] = False

    @property
    def normalized(self) -> bool:
        r"""Whether :math:`f` integrates to 1 over :math:`\ln\sigma^{-1}` (all mass in haloes)."""
        return self._normalized

    @classmethod
    def domain_inputs(cls) -> frozenset[str]:
        """The inputs (besides sigma) needed to evaluate the fit and check both domains."""
        return (
            cls.requires | _domain_inputs(cls.valid_domain) | _domain_inputs(cls.calibration_domain)
        )

    @unit_boundary(m=Msun_h)
    def inputs(
        self,
        sigma: npt.ArrayLike,
        *,
        z: npt.ArrayLike | None = None,
        omega_m_z: npt.ArrayLike | None = None,
        delta_halo: npt.ArrayLike | None = None,
        delta_c: npt.ArrayLike | None = None,
        n_eff: npt.ArrayLike | None = None,
        m: npt.ArrayLike | None = None,
    ) -> FitInputs:
        """Collect the inputs, checking that the required ones are given.

        Parameters are as for :meth:`fsigma` (``m`` is a Quantity). Code that already
        has plain arrays in canonical units builds :class:`FitInputs` directly (see
        "Evaluating a fit" in the :mod:`module documentation <hmf.core.fits>`).

        Returns
        -------
        FitInputs
            The inputs as plain arrays, in canonical units.

        Raises
        ------
        ValueError
            If an input in :attr:`requires` (or needed to check :attr:`valid_domain`)
            is missing.
        """
        x = FitInputs(
            sigma=sigma,
            z=z,
            omega_m_z=omega_m_z,
            delta_halo=delta_halo,
            delta_c=delta_c,
            n_eff=n_eff,
            m=m,
        )
        self.check_required(x)
        return x

    def check_required(self, x: FitInputs) -> None:
        """Raise if an input needed to evaluate the fit is missing from ``x``.

        Raises
        ------
        ValueError
            If an input in :attr:`requires` (or needed to check :attr:`valid_domain`)
            is missing.
        """
        needed = self.requires | _domain_inputs(self.valid_domain)
        missing = sorted(name for name in needed if not x.given(name))
        if missing:
            raise ValueError(
                f"{type(self).__name__}.fsigma() needs the input(s) {missing} "
                f"(it requires {sorted(needed)})."
            )

    def check_valid(self, x: FitInputs) -> None:
        """Raise if any input is outside :attr:`valid_domain`.

        Raises
        ------
        DomainError
            Always, if any value is outside the valid domain: this does not depend on
            any policy.
        """
        domain = self.valid_domain
        domain.check(
            {name: x.domain_value(name) for name in domain.variables},
            where=f"{type(self).__name__}'s valid domain",
        )

    def in_calibration_domain(self, x: FitInputs) -> bool | BoolArray:
        """Whether each input is inside :attr:`calibration_domain`.

        Parameters
        ----------
        x
            The inputs, from :meth:`inputs`.

        Returns
        -------
        bool or numpy.ndarray of bool
            A bool if every input is a scalar, else the broadcast array. ``True`` if
            the fit has no calibration domain (it is ``None``).

        Raises
        ------
        ValueError
            If an input needed to check the domain was not given.
        """
        return _contains(self.calibration_domain, x)

    @unit_boundary(m=Msun_h)
    def fsigma(
        self,
        sigma: npt.ArrayLike,
        *,
        z: npt.ArrayLike | None = None,
        omega_m_z: npt.ArrayLike | None = None,
        delta_halo: npt.ArrayLike | None = None,
        delta_c: npt.ArrayLike | None = None,
        n_eff: npt.ArrayLike | None = None,
        m: npt.ArrayLike | None = None,
    ) -> FloatArray:
        r"""The multiplicity function, :math:`f(\sigma) = \nu f(\nu)`.

        ``m`` must be a Quantity (see :mod:`hmf.core.units`); every other input is
        dimensionless. Code that already has plain arrays in canonical units calls
        :meth:`fsigma_kernel` or :func:`evaluate_fsigma` instead (see
        "Evaluating a fit" in the :mod:`module documentation <hmf.core.fits>`). It
        applies no calibration policy (see :func:`evaluate_fsigma` for that), but
        always checks :attr:`valid_domain`.

        Parameters
        ----------
        sigma
            :math:`\sigma(m, z)`.
        z
            Redshift.
        omega_m_z
            :math:`\Omega_m(z)`.
        delta_halo
            The halo overdensity relative to the mean density.
        delta_c
            The critical overdensity for collapse.
        n_eff
            The effective spectral index.
        m
            The halo mass, a Quantity (e.g. in :data:`~hmf.core.units.Msun_h`).

        Returns
        -------
        numpy.ndarray
            :math:`f(\sigma)` (dimensionless), broadcast over the inputs.

        Raises
        ------
        ValueError
            If an input the fit requires is not given.
        UnitBoundaryError
            If ``m`` is not a Quantity.
        DomainError
            If any input is outside :attr:`valid_domain`, or makes the fit's
            parameters unphysical.
        """
        x = FitInputs(
            sigma=sigma,
            z=z,
            omega_m_z=omega_m_z,
            delta_halo=delta_halo,
            delta_c=delta_c,
            n_eff=n_eff,
            m=m,
        )
        return self.fsigma_kernel(x)

    def fsigma_kernel(self, x: FitInputs) -> FloatArray:
        """:meth:`fsigma` at kernel level, from inputs that are plain arrays.

        The inputs are in canonical units (the mass in Msun/h); see
        :mod:`hmf.core._kernels`. It checks the valid domain, as :meth:`fsigma`
        does, and applies no calibration policy (see :func:`evaluate_fsigma`).

        Parameters
        ----------
        x
            The inputs.

        Returns
        -------
        numpy.ndarray

        Raises
        ------
        ValueError
            If an input the fit requires is not in ``x``.
        DomainError
            As for :meth:`fsigma`.
        """
        self.check_required(x)
        self.check_valid(x)
        return np.asarray(self._fsigma(x), dtype=np.float64)

    @abc.abstractmethod
    def _fsigma(self, x: FitInputs) -> FloatArray:
        """Compute f(sigma) from inputs inside the valid domain."""

    @unit_boundary(returns=dndm_unit)
    def modify_dndm(
        self,
        m: Any,
        dndm: Any,
        *,
        z: npt.ArrayLike,
        ngtm: Any,
        H0: Any,
        omega_m0: npt.ArrayLike,
    ) -> Any:
        """Map the mass function computed from :meth:`fsigma` to the fit's.

        The identity unless :attr:`modifies_dndm` (see "Post-processing the mass
        function" in the :mod:`module documentation <hmf.core.fits>`). Every fit
        takes the same arguments, whether it uses them or not. This method converts
        them and calls :meth:`modify_dndm_kernel`, which a mass-function
        :class:`~hmf.core.stage.Stage` calls directly.

        Parameters
        ----------
        m
            Halo masses, a Quantity: in h-units, or physical units (converted with
            ``H0``).
        dndm
            The mass function from :meth:`fsigma`, a Quantity (number density per mass).
        z
            Redshift.
        ngtm
            The cumulative mass function n(>m) from :meth:`fsigma`, a Quantity.
        H0
            The Hubble constant, a scalar Quantity (e.g. ``70 * H0_unit``). It converts
            physical units, and gives h to fits whose formula is in physical units.
        omega_m0
            The density parameter of CDM + baryons today.

        Returns
        -------
        Quantity
            The fit's mass function, in :data:`~hmf.core.units.dndm_unit`.
        """
        where = f"{type(self).__name__}.modify_dndm()"
        context, h = _hubble(H0, where)
        return self.modify_dndm_kernel(
            to_canonical(m, Msun_h, context=context, where=where, name="m"),
            to_canonical(dndm, dndm_unit, context=context, where=where, name="dndm"),
            z=z,
            ngtm=to_canonical(ngtm, number_density_unit, context=context, where=where, name="ngtm"),
            h=h,
            omega_m0=omega_m0,
        )

    def modify_dndm_kernel(
        self,
        m: npt.ArrayLike,
        dndm: npt.ArrayLike,
        *,
        z: npt.ArrayLike,
        ngtm: npt.ArrayLike,
        h: npt.ArrayLike,
        omega_m0: npt.ArrayLike,
    ) -> FloatArray:
        """:meth:`modify_dndm` at kernel level, on plain arrays (the identity here).

        A pure kernel in canonical units (see :mod:`hmf.core._kernels`), which a fit
        that sets :attr:`modifies_dndm` overrides. All arguments broadcast.

        Parameters
        ----------
        m
            Halo masses, in Msun/h.
        dndm
            The mass function from :meth:`fsigma`, in h^4 / (Msun Mpc^3).
        z
            Redshift.
        ngtm
            The cumulative mass function n(>m) from :meth:`fsigma`, in h^3 / Mpc^3.
        h
            The dimensionless Hubble parameter, H0 / (100 km/s/Mpc).
        omega_m0
            The density parameter of CDM + baryons today.

        Returns
        -------
        numpy.ndarray
            The fit's mass function, in h^4 / (Msun Mpc^3).
        """
        return np.asarray(dndm, dtype=np.float64)


def _hubble(H0: Any, where: str) -> tuple[UnitContext, float]:
    """The units context of a fit method's ``H0`` argument, and h = H0 / (100 km/s/Mpc).

    Fits have no cosmology, so a method that needs h, or physical units converted,
    takes the H0 as an argument and builds its context per call.
    """
    if H0 is None:
        raise UnitBoundaryError(
            f"{where}: H0 must be a Quantity in km/s/Mpc, e.g. 70 * hmf.core.units.H0_unit, "
            "not None."
        )
    context = UnitContext(H0, where=where)
    return context, float(H0.to_value(H0_unit)) / 100


# ---------------------------------------------------------------------------------
# Evaluation with a domain policy
# ---------------------------------------------------------------------------------
@attrs.frozen
class FSigmaResult:
    r"""The result of :func:`evaluate_fsigma`.

    Parameters
    ----------
    fsigma
        :math:`f(\sigma)`; NaN outside the calibration domain with the ``"mask"``
        policy, and nowhere else.
    in_calibration_domain
        Whether each value is inside the fit's calibration domain (the mask), with
        the shape of ``fsigma``.
    """

    fsigma: FloatArray
    in_calibration_domain: BoolArray


def evaluate_fsigma(
    model: FittingFunction, inputs: FitInputs, *, policy: DomainPolicy = "ignore"
) -> FSigmaResult:
    r"""Evaluate a fit, applying a domain policy to its calibration domain.

    The inputs are a :class:`FitInputs`: plain arrays in canonical units, built
    directly (by code such as a mass-function :class:`~hmf.core.stage.Stage`, which
    already has them) or by :meth:`FittingFunction.inputs` from Quantities. See
    "Evaluating a fit" in the :mod:`module documentation <hmf.core.fits>`.

    The valid domain always raises (see :meth:`FittingFunction.fsigma`); the policy
    applies to the calibration domain only:

    ``"ignore"``
        return :math:`f(\sigma)` everywhere;
    ``"warn"``
        also emit an :class:`~hmf.exceptions.HMFExtrapolationWarning` if any value is
        outside. It is emitted on each call: a :class:`~hmf.core.stage.Stage` that
        wants it once per instance calls this once and caches the result;
    ``"mask"``
        return NaN outside;
    ``"raise"``
        raise :class:`~hmf.core.domain.DomainError` if any value is outside.

    Parameters
    ----------
    model
        The fit.
    inputs
        The inputs. Those needed to check the calibration domain (see
        :meth:`FittingFunction.domain_inputs`) must be given.
    policy
        What to do outside the calibration domain.

    Returns
    -------
    FSigmaResult
        :math:`f(\sigma)` and the calibration mask.

    Raises
    ------
    ValueError
        If an input needed by the fit or by either domain is missing.
    DomainError
        If any input is outside the valid domain (whatever the policy), or outside
        the calibration domain with ``policy="raise"``.
    """
    x = inputs
    model.check_required(x)
    missing = sorted(name for name in _domain_inputs(model.calibration_domain) if not x.given(name))
    if missing:
        raise ValueError(
            f"Checking {type(model).__name__}'s calibration domain needs the input(s) {missing}."
        )
    f = model.fsigma_kernel(x)
    inside = np.broadcast_to(model.in_calibration_domain(x), f.shape)
    calibration = model.calibration_domain
    # Without a calibration domain everything is inside: this only checks the policy.
    out = apply_domain_policy(
        f,
        inside,
        policy,
        description=f"{type(model).__name__}'s calibration domain "
        f"({'none' if calibration is None else calibration.describe()})",
    )
    return FSigmaResult(np.asarray(out, dtype=np.float64), np.array(inside, dtype=bool))


def _p(default: float | None, doc: str, **kwargs: Any) -> Any:
    """A fit parameter: a float field with documentation.

    The value is converted to a float (so ``SMT(a=1)`` and ``SMT(a=1.0)`` are the same
    model, with the same content hash). A parameter whose default is ``None`` may also
    be ``None``.
    """
    converter = attrs.converters.optional(float) if default is None else float
    return field(default=default, doc=doc, converter=converter, **kwargs)


def _check_physical(name: str, **params: FloatArray) -> None:
    """Raise a DomainError if any of ``params`` is not finite and > 0."""
    for key, value in params.items():
        check_finite_positive(f"parameter {key}", value, where=name, error=DomainError)


# ---------------------------------------------------------------------------------
# Press-Schechter and the Sheth-Tormen family
# ---------------------------------------------------------------------------------
@attrs.frozen(kw_only=True)
class PS(FittingFunction, alias="PS"):
    r"""The Press & Schechter (1974) mass function.

    .. math:: f(\sigma) = \sqrt{2/\pi}\,\nu\exp(-\nu^2/2).

    It follows from spherical collapse with the "fudge factor" of 2, which makes it
    normalised: all mass is in haloes. It has no free parameters.
    """

    references: ClassVar[tuple[str, ...]] = (refs.PS74,)
    parameter_source: ClassVar[str] = "Press & Schechter 1974: analytic, no free parameters."
    requires: ClassVar[frozenset[str]] = frozenset({"delta_c"})
    valid_domain: ClassVar[Domain] = Domain(
        {"sigma": _POSITIVE, "delta_c": _POSITIVE},
        source="sigma > 0 and delta_c > 0.",
    )
    #: Analytic (spherical collapse): not calibrated on simulations.
    calibration_domain: ClassVar[Domain | None] = None
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = _so_virial(
        note="Analytic: spherical collapse, i.e. virialised haloes."
    )
    simulations: ClassVar[SimulationDetails | None] = None
    _normalized: ClassVar[bool] = True

    def _fsigma(self, x: FitInputs) -> FloatArray:
        return _k.press_schechter(x.nu)


#: p < 1/2, so that the Sheth-Tormen form can be normalised.
_P_BELOW_HALF = less_than(0.5)

_NU_VALID = Domain({"sigma": _POSITIVE, "delta_c": _POSITIVE}, source="sigma > 0 and delta_c > 0.")


@attrs.frozen(kw_only=True)
class SMT(FittingFunction, alias="SMT"):
    r"""The Sheth-Tormen (Sheth, Mo & Tormen 2001) mass function.

    .. math::

        f(\sigma) = A\sqrt{\frac{2a}{\pi}}\,\nu\exp\left(-\frac{a\nu^2}{2}\right)
            \left[1 + (a\nu^2)^{-p}\right].

    With ``A=None`` (the default), A is the value that normalises the fit,
    :math:`A = [1 + 2^{-p}\Gamma(1/2 - p)/\Gamma(1/2)]^{-1}` (0.3222 for p = 0.3).
    It reduces to :class:`PS` for :math:`A = 1/2`, :math:`a = 1`, :math:`p = 0`.
    """

    references: ClassVar[tuple[str, ...]] = (refs.SMT01, refs.ST99)
    parameter_source: ClassVar[str] = (
        "Sheth & Tormen 1999, eq. 10 (a = 0.707, p = 0.3); A from the normalisation "
        "(Sheth, Mo & Tormen 2001, eq. 5: A = 0.3222)."
    )
    requires: ClassVar[frozenset[str]] = frozenset({"delta_c"})
    valid_domain: ClassVar[Domain] = _NU_VALID
    calibration_domain: ClassVar[Domain | None] = Domain(
        {"z": (0, 4)},
        source=(
            "Sheth & Tormen 1999, Fig. 2 caption: GIF simulations at z = 0, 0.5, 1, 2, 4. "
            "sigma / mass range: not stated (the Fig. 2 data span roughly "
            "-1.1 < ln(1/sigma) < 0.85). Cosmologies: SCDM, OCDM and LCDM GIF runs "
            "(Sec. 3)."
        ),
    )
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = _so_virial(
        note=(
            "Sheth & Tormen 1999, Sec. 3: a spherical-overdensity group finder (Tormen "
            "1998). Sheth, Mo & Tormen 2001, Sec. 4.1, instead associate a = 0.707 with "
            "FoF(b=0.2)."
        )
    )
    simulations: ClassVar[SimulationDetails | None] = SimulationDetails(
        box_size=[84.5, 141.3, 141.3] * Mpc_h,
        n_particles=256**3,
        omega_m=(1.0, 0.3, 0.3),
        h=(0.5, 0.7, 0.7),
        z_range=(0.0, 4.0),
        halo_finder="SO (Tormen 1998)",
        names=("SCDM", "OCDM", "LCDM"),
        notes=(
            "The GIF simulations; OCDM has Omega_Lambda = 0, LCDM 0.7. sigma_8, "
            "softening, transfer function and starting redshift: not stated (hmf 3.x "
            "took them from Jenkins et al. 2001, Table 1)."
        ),
        source="Sheth & Tormen 1999, Sec. 3 and Fig. 2.",
    )

    a: float = _p(0.707, "The parameter a (rescales nu^2).", validator=positive)
    p: float = _p(0.3, "The low-mass slope parameter p (< 0.5).", validator=_P_BELOW_HALF)
    A: float | None = _p(
        None, "The amplitude; None for the value that normalises the fit to unit mass."
    )

    @property
    def normalized(self) -> bool:
        """Whether the amplitude is the normalising one (``A is None``)."""
        return self.A is None

    @property
    def amplitude(self) -> float:
        """The amplitude A used: ``A``, or the normalising value if it is None."""
        return float(_k.sheth_tormen_norm(self.p)) if self.A is None else self.A

    def _fsigma(self, x: FitInputs) -> FloatArray:
        return _k.sheth_tormen(x.nu, self.amplitude, self.a, self.p)


@attrs.frozen(kw_only=True)
class ST(SMT, alias="ST"):
    """The Sheth-Tormen mass function: the same model as :class:`SMT`, under its other name."""

    calibration_domain: ClassVar[Domain | None] = SMT.calibration_domain
    valid_domain: ClassVar[Domain] = SMT.valid_domain
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = SMT.measured_mass_definition
    simulations: ClassVar[SimulationDetails | None] = SMT.simulations


@attrs.frozen(kw_only=True)
class Reed03(SMT, alias="Reed03"):
    r"""The Reed et al. (2003) mass function: Sheth-Tormen with a high-z correction.

    .. math:: f(\sigma) = f_{\rm ST}(\sigma)\exp\left[-\frac{c}{\sigma\cosh^5(2\sigma)}\right].
    """

    references: ClassVar[tuple[str, ...]] = (refs.REED03,)
    parameter_source: ClassVar[str] = (
        "Reed et al. 2003 (arXiv v2), eq. 5 (A, a, p of Sheth-Tormen) and eq. 9 (c = 0.7)."
    )
    valid_domain: ClassVar[Domain] = _NU_VALID
    calibration_domain: ClassVar[Domain | None] = Domain(
        {"ln_sigma_inv": (-1.7, 0.9), "z": (0, 14.5)},
        source=(
            "Reed et al. 2003: 'valid over the range -1.7 <= ln(1/sigma) <= 0.9' (Sec. 4, "
            "after eq. 9); outputs from z = 0 to 14.5 (Fig. 4 caption). One cosmology "
            "(Omega_m = 0.3, sigma_8 = 1.0, Sec. 2). Accuracy 10% for ln(1/sigma) <= 0.5, "
            "~20% above (Sec. 4)."
        ),
    )
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = _fof(
        note="Reed et al. 2003, Sec. 2.1."
    )
    simulations: ClassVar[SimulationDetails | None] = SimulationDetails(
        box_size=[50.0, 50.0] * Mpc_h,
        n_particles=432**3,
        omega_m=0.3,
        sigma_8=1.0,
        softening=5.0 * kpc_h,
        transfer="BBKS",
        z_start=(69, 139),
        initial_conditions="ZA",
        n_min=64,
        z_range=(0.0, 14.5),
        notes=(
            "One box run from two starting redshifts: z_start = 69 for the outputs at "
            "z < 7, 139 for z >= 7 (Sec. 4.2). Omega_Lambda = 0.7; h, n_s and Omega_b: "
            "not stated."
        ),
        source="Reed et al. 2003, Secs. 2 and 4.2, Fig. 4.",
    )
    A: float | None = _p(0.3222, "The Sheth-Tormen amplitude; None to normalise it.")
    c: float = _p(0.7, "The strength c of the high-z suppression.")

    @property
    def normalized(self) -> bool:
        """Reed03 is not normalised (its correction removes mass)."""
        return False

    def _fsigma(self, x: FitInputs) -> FloatArray:
        return super()._fsigma(x) * _k.reed03_factor(x.sigma, self.c)


@attrs.frozen(kw_only=True)
class Courtin(SMT, alias="Courtin"):
    r"""The Courtin et al. (2011) mass function: Sheth-Tormen with refitted parameters."""

    references: ClassVar[tuple[str, ...]] = (refs.COURTIN11,)
    parameter_source: ClassVar[str] = (
        "Courtin et al. 2011 (arXiv v2), eq. 22, which was fitted with delta_c fixed "
        "to 1.673 (pass that delta_c to reproduce the paper)."
    )
    valid_domain: ClassVar[Domain] = _NU_VALID
    calibration_domain: ClassVar[Domain | None] = Domain(
        {"ln_sigma_inv": (-0.8, 0.7), "z": (0, 0)},
        source=(
            "Courtin et al. 2011: the LCDM-WMAP5 runs 'at z = 0 in the range "
            "-0.8 < ln(1/sigma) < 0.7' (Sec. 5.1, Fig. 8; 1e12-1e15 Msun/h). Omega_m = "
            "0.26, sigma_8 = 0.79 (Table 1). Accuracy 5-10% (Sec. 5.1)."
        ),
    )
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = _fof(
        note="Courtin et al. 2011, Sec. 3.2."
    )
    simulations: ClassVar[SimulationDetails | None] = SimulationDetails(
        box_size=[162, 648, 1296] * Mpc_h,
        n_particles=512**3,
        omega_m=0.26,
        omega_b=0.044,
        sigma_8=0.79,
        h=0.72,
        n_s=0.963,
        softening=[2.47, 19.78, 39.55] * kpc_h,
        z_start=(93, 56, 41),
        initial_conditions="ZA",
        n_min=350,
        z_range=(0.0, 0.0),
        names=("LCDM-W5 162", "LCDM-W5 648", "LCDM-W5 1296"),
        notes=(
            "RAMSES (AMR): the softening is the finest cell size. Haloes with at least "
            "350 particles and Poisson noise below 10%."
        ),
        source="Courtin et al. 2011 (arXiv v2), Secs. 3.1-3.2, Tables 1 and 4.",
    )
    A: float | None = _p(0.348, "The amplitude; None to normalise the fit.")
    a: float = _p(0.695, "The parameter a (rescales nu^2).", validator=positive)
    p: float = _p(0.1, "The low-mass slope parameter p (< 0.5).", validator=_P_BELOW_HALF)


@attrs.frozen(kw_only=True)
class Manera(SMT, alias="Manera"):
    r"""The Manera, Sheth & Scoccimarro (2010) mass function: Sheth-Tormen, refitted."""

    references: ClassVar[tuple[str, ...]] = (refs.MANERA10,)
    parameter_source: ClassVar[str] = (
        "Manera et al. 2010 (arXiv v2), Table 2: z = 0, 'New ML', linking length 0.2 "
        "(q = 0.709, p = 0.248). hmf 3.x used p = 0.289, from the l_link = 0.15 row."
    )
    valid_domain: ClassVar[Domain] = _NU_VALID
    calibration_domain: ClassVar[Domain | None] = Domain(
        {"m": (6.3e13 * Msun_h, None), "z": (0, 0)},
        source=(
            "Manera et al. 2010: haloes with more than 105 particles, M >~ 6.3e13 Msun/h "
            "(Sec. 3.3); upper mass: not stated. These parameters are the z = 0 fit "
            "(Table 2; z = 0.5 has its own). One cosmology (Omega_m = 0.27, "
            "sigma_8 = 0.9; Sec. 3.1). sigma range: not stated."
        ),
    )
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = _fof(
        note="Manera et al. 2010, Sec. 3.2, with the Warren et al. mass correction."
    )
    simulations: ClassVar[SimulationDetails | None] = SimulationDetails(
        box_size=[1280.0] * Mpc_h,
        n_particles=640**3,
        omega_m=0.27,
        omega_b=0.046,
        sigma_8=0.9,
        h=0.72,
        n_s=1.0,
        softening=20.0 * kpc_h,
        transfer="CMBFAST",
        z_start=50,
        initial_conditions="2LPT",
        n_min=105,
        z_range=(0.0, 0.5),
        notes=(
            "49 realisations of this box. The Warren et al. FoF mass correction is "
            "applied; the default parameters are the z = 0 fit."
        ),
        source="Manera et al. 2010 (arXiv v2), Secs. 3.1-3.3.",
    )
    a: float = _p(0.709, "The parameter a (rescales nu^2).", validator=positive)
    p: float = _p(0.248, "The low-mass slope parameter p (< 0.5).", validator=_P_BELOW_HALF)


# ---------------------------------------------------------------------------------
# Jenkins, Warren and the Warren form
# ---------------------------------------------------------------------------------
@attrs.frozen(kw_only=True)
class Jenkins(FittingFunction, alias="Jenkins"):
    r"""The Jenkins et al. (2001) mass function.

    .. math:: f(\sigma) = A\exp\left(-\left|\ln\sigma^{-1} + b\right|^c\right).
    """

    references: ClassVar[tuple[str, ...]] = (refs.JENKINS01,)
    parameter_source: ClassVar[str] = "Jenkins et al. 2001 (arXiv v2, as accepted), eq. 9."
    valid_domain: ClassVar[Domain] = Domain({"sigma": _POSITIVE}, source="sigma > 0.")
    calibration_domain: ClassVar[Domain | None] = Domain(
        {"ln_sigma_inv": (-1.2, 1.05), "z": (0, 5)},
        source=(
            "Jenkins et al. 2001: 'valid over the range -1.2 <= ln(1/sigma) <= 1.05' "
            "(Sec. 5.2, after eq. 9; and Sec. 6); outputs from z = 0 to 5.03 (Table 2). "
            "0.3 <= Omega_m <= 1, open and flat (Table 2). Accuracy ~20% (Sec. 5.2)."
        ),
    )
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = _fof(
        note="Jenkins et al. 2001, Sec. 5.2."
    )
    simulations: ClassVar[SimulationDetails | None] = SimulationDetails(
        box_size=[84.5, 141.3, 479.0, 3000.0] * Mpc_h,
        n_particles=(256**3, 256**3, 512**3, 1000**3),
        omega_m=(1.0, 0.3, 0.3, 0.3),
        sigma_8=(0.6, 0.9, 0.9, 0.9),
        h=(None, 0.7, 0.7, 0.7),
        n_s=1.0,
        softening=[30.0, 25.0, 30.0, 100.0] * kpc_h,
        transfer=("BondEfs", "BondEfs", "CMBFAST", "CMBFAST"),
        n_min=20,
        z_range=(0.0, 5.0),
        names=("tauCDM-gif", "LCDM-gif", "LCDM-512", "LCDM-Hubble volume"),
        notes=(
            "Representative runs: eq. 9 is fitted to about ten simulations in five "
            "cosmologies (0.3 <= Omega_m <= 1, sigma_8 = 0.51-1.0; Table 2). The data "
            "were smoothed and deconvolved before fitting (Sec. 5.2)."
        ),
        source="Jenkins et al. 2001, Sec. 2.1 and Tables 1-2.",
    )
    A: float = _p(0.315, "The amplitude A.")
    b: float = _p(0.61, "The offset b of ln(1/sigma).")
    c: float = _p(3.8, "The exponent c.")

    def _fsigma(self, x: FitInputs) -> FloatArray:
        return _k.jenkins(x.sigma, self.A, self.b, self.c)


_SIGMA_VALID = Domain({"sigma": _POSITIVE}, source="sigma > 0.")


@attrs.frozen(kw_only=True)
class Warren(FittingFunction, alias="Warren"):
    r"""The Warren et al. (2006) mass function.

    .. math:: f(\sigma) = A\left[\left(\frac{e}{\sigma}\right)^b + c\right]
        \exp\left(-\frac{d}{\sigma^2}\right).

    The paper's (A, a, b, c) are hmf's (A, b, c, d), with e = 1.
    """

    references: ClassVar[tuple[str, ...]] = (refs.WARREN06,)
    parameter_source: ClassVar[str] = "Warren et al. 2006 (arXiv v1), eq. 8."
    valid_domain: ClassVar[Domain] = _SIGMA_VALID
    calibration_domain: ClassVar[Domain | None] = Domain(
        {"m": [1e10, 1e15] * Msun_h, "z": (0, 0)},
        source=(
            "Warren et al. 2006 state only 'over five orders of magnitude' in mass (Sec. "
            "2); the range 1e10-1e15 Msun/h is Bhattacharya et al. 2011, Table 3. z: not "
            "stated, a single epoch, presumably z = 0. One cosmology: (Omega_m, Omega_b, "
            "n, h, sigma_8) = (0.3, 0.04, 1, 0.7, 0.9) (eq. 1)."
        ),
    )
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = _fof(
        note="Warren et al. 2006, Sec. 2, with the N(1 - N^-0.6) correction (eq. 3)."
    )
    simulations: ClassVar[SimulationDetails | None] = SimulationDetails(
        box_size=[96, 135, 192, 272, 384, 543, 768, 1086, 1536, 2172, 2583, 3072] * Mpc_h,
        n_particles=1024**3,
        omega_m=0.3,
        omega_b=0.04,
        sigma_8=0.9,
        h=0.7,
        n_s=1.0,
        transfer="CMBFAST",
        n_min=400,
        notes=(
            "16 runs in all (three 384, two 272 and two 3072 Mpc/h boxes). Softening: "
            "2.1 kpc/h (physical) in the highest-resolution run to 98 kpc/h (comoving) "
            "in the largest; not stated for the others. Redshift: not stated (a single "
            "epoch, presumably z = 0). FoF masses corrected by N(1 - N^-0.6) (eq. 3); "
            "fit by maximum likelihood on Poisson counts."
        ),
        source="Warren et al. 2006 (arXiv v1), Sec. 2 and eq. 1.",
    )
    A: float = _p(0.7234, "The amplitude A.")
    b: float = _p(1.625, "The exponent b of e/sigma.")
    c: float = _p(0.2538, "The constant c added to (e/sigma)^b.")
    d: float = _p(1.1982, "The exponential cut-off d.")
    e: float = _p(1.0, "The scale e of sigma.")

    def _fsigma(self, x: FitInputs) -> FloatArray:
        return _k.warren(x.sigma, self.A, self.b, self.c, self.d, self.e)


@attrs.frozen(kw_only=True)
class Reed07(FittingFunction, alias="Reed07"):
    r"""The Reed et al. (2007) mass function, which depends on the spectral slope.

    .. math::

        f(\sigma) = A\sqrt{\frac{2a}{\pi}}\left[1 + (a\nu^2)^{-p} + 0.6G_1 + 0.4G_2\right]
            \nu\exp\left[-\frac{ca\nu^2}{2} - \frac{0.03\nu^{0.6}}{(n_{\rm eff}+3)^2}\right],

    with :math:`G_1, G_2` Gaussians in :math:`\ln\sigma^{-1}`. As in hmf 3.x, the
    parameter ``a`` is the paper's product :math:`ca` (0.764), so the a of the formula
    is ``a / c``.
    """

    references: ClassVar[tuple[str, ...]] = (refs.REED07,)
    parameter_source: ClassVar[str] = (
        "Reed et al. 2007 (arXiv v4), eqs. 11-12: c = 1.08, ca = 0.764, p = 0.3, "
        "A = 0.3222 (A' = 0.310 = A / sqrt(c) in eq. 12)."
    )
    requires: ClassVar[frozenset[str]] = frozenset({"delta_c", "n_eff"})
    valid_domain: ClassVar[Domain] = Domain(
        {"sigma": _POSITIVE, "delta_c": _POSITIVE, "n_eff": (-3, None, "(]")},
        source="sigma > 0, delta_c > 0, and n_eff > -3 (the n_eff term diverges at -3).",
    )
    calibration_domain: ClassVar[Domain | None] = Domain(
        {"z": (0, 30), "ln_sigma_inv": (-0.4, 1.2)},
        source=(
            "Reed et al. 2007: data at z = 0, 1, 4, 10, 20, 30 (Figs. 4, 6). ln(1/sigma) "
            "range: not stated; -0.4 to 1.2 is inferred from the data in Fig. 6. "
            "Accuracy 4% rms (Sec. 4.2). Omega_m = 0.25, sigma_8 = 0.9 (Sec. 1)."
        ),
    )
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = _fof(
        note="Reed et al. 2007, Sec. 2."
    )
    simulations: ClassVar[SimulationDetails | None] = SimulationDetails(
        box_size=[1.0, 2.5, 2.5, 2.5, 2.5, 4.64, 11.6, 20, 50, 100, 500, 1340, 3000] * Mpc_h,
        n_particles=(
            400**3,
            1000**3,
            1000**3,
            500**3,
            200**3,
            400**3,
            1000**3,
            400**3,
            1000**3,
            900**3,
            2160**3,
            1448**3,
            1000**3,
        ),
        omega_m=(0.25,) * 12 + (0.3,),
        omega_b=(0.045,) * 12 + (None,),
        sigma_8=0.9,
        h=(0.73,) * 12 + (0.7,),
        n_s=1.0,
        softening=[0.125, 0.125, 0.125, 0.25, 0.625, 0.58, 0.58, 2.5, 2.4, 2.4, 5, 20, 100] * kpc_h,
        transfer=("CMBFAST",) * 9 + ("Millennium", "CMBFAST", "Millennium", "BondEfs"),
        z_start=(299, 299, 299, 299, 299, 249, 249, 249, 299, 149, 127, 63, 35),
        initial_conditions="ZA",
        n_min=100,
        z_range=(0.0, 30.0),
        notes=(
            "The 3000 Mpc/h box is the Hubble Volume (Omega_m = 0.3). Data at z = 0, 1, "
            "4, 10, 20 and 30 (Figs. 4, 6); finite-volume corrections applied."
        ),
        source="Reed et al. 2007 (arXiv v4), Sec. 1, Sec. 3, Table 1 and App. A2.",
    )
    A: float = _p(0.3222, "The amplitude A.")
    p: float = _p(0.3, "The low-mass slope p.")
    c: float = _p(1.08, "The exponential cut-off factor c.")
    a: float = _p(0.764, "The product ca of the paper (the a of the formula is a / c).")

    def _fsigma(self, x: FitInputs) -> FloatArray:
        return _k.reed07(x.sigma, x.nu, x.n_eff, self.A, self.a / self.c, self.c, self.p)


@attrs.frozen(kw_only=True)
class Peacock(FittingFunction, alias="Peacock"):
    r"""The Peacock (2007) mass function, a normalised fit to :class:`Warren`.

    The collapsed fraction is :math:`F(\nu) = e^{-c\nu^2}/(1 + a\nu^b)`, and
    :math:`f = -dF/d\ln\nu`. Since :math:`F(0) = 1`, it is normalised.
    """

    references: ClassVar[tuple[str, ...]] = (refs.PEACOCK07,)
    parameter_source: ClassVar[str] = "Peacock 2007 (arXiv v2), eq. 9."
    requires: ClassVar[frozenset[str]] = frozenset({"delta_c"})
    valid_domain: ClassVar[Domain] = _NU_VALID
    calibration_domain: ClassVar[Domain | None] = Domain(
        {"m": [1e10, 1e15] * Msun_h, "z": (0, 0)},
        source=(
            "Peacock 2007 fits the Warren et al. 2006 formula, not simulations, 'to a "
            "maximum error of about 1% over the whole range where data exist' (Sec. 2.3); "
            "so this is Warren's domain (see Warren)."
        ),
    )
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = _fof(
        note="Inherited from Warren et al. 2006."
    )
    simulations: ClassVar[SimulationDetails | None] = None
    _normalized: ClassVar[bool] = True
    a: float = _p(1.529, "The parameter a.")
    b: float = _p(0.704, "The exponent b.")
    c: float = _p(0.412, "The exponential cut-off c.")

    def _fsigma(self, x: FitInputs) -> FloatArray:
        return _k.peacock(x.nu, self.a, self.b, self.c)


# ---------------------------------------------------------------------------------
# Angulo, Watson, Crocce, Bhattacharya
# ---------------------------------------------------------------------------------


@attrs.frozen(kw_only=True)
class Angulo(FittingFunction, alias="Angulo"):
    r"""The Angulo et al. (2012) mass function (Millennium-XXL, FoF masses).

    .. math:: f(\sigma) = A\left[\left(\frac{d}{\sigma}\right)^b + 1\right]
        \exp\left(-\frac{c}{\sigma^2}\right).
    """

    references: ClassVar[tuple[str, ...]] = (refs.ANGULO12,)
    parameter_source: ClassVar[str] = (
        "Angulo et al. 2012 (arXiv v2), eq. 2. The paper prints A[d/sigma + 1]^b; the "
        "form used here, A[(d/sigma)^b + 1], is the one that matches its data (and "
        "Warren et al. 2006 to 1-3%)."
    )
    valid_domain: ClassVar[Domain] = _SIGMA_VALID
    calibration_domain: ClassVar[Domain | None] = Domain(
        {"m": [7.3e7, 7.3e15] * Msun_h, "z": (0, 0)},
        source=(
            "Angulo et al. 2012: MXXL + MS + MS-II at z = 0, 'about 8 decades in halo "
            "mass', 1e8-1e16 Msun (Fig. 2; Sec. 2.2), i.e. 7.3e7-7.3e15 Msun/h for "
            "their h = 0.73. Omega_m = 0.25, sigma_8 = 0.9 (Sec. 2.1). Accuracy better "
            "than 5% over most of the range (Sec. 2.2)."
        ),
    )
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = _fof(
        note="Angulo et al. 2012, Sec. 2.2."
    )
    simulations: ClassVar[SimulationDetails | None] = SimulationDetails(
        box_size=[3000.0] * Mpc_h,
        n_particles=6720**3,
        omega_m=0.25,
        omega_b=0.045,
        sigma_8=0.9,
        h=0.73,
        softening=10.0 * kpc_h,
        z_start=63,
        initial_conditions="2LPT",
        n_min=20,
        z_range=(0.0, 0.0),
        names="MXXL",
        notes=(
            "The fit combines MXXL with the Millennium and Millennium-II simulations "
            "(same cosmology), whose details are not given in this paper. n_s: not "
            "stated. Softening 13.7 kpc."
        ),
        source="Angulo et al. 2012, Secs. 2.1-2.2.",
    )
    A: float = _p(0.201, "The amplitude A.")
    b: float = _p(1.7, "The exponent b.")
    c: float = _p(1.172, "The exponential cut-off c.")
    d: float = _p(2.08, "The scale d of sigma.")

    def _fsigma(self, x: FitInputs) -> FloatArray:
        return _k.angulo(x.sigma, self.A, self.b, self.c, self.d)


@attrs.frozen(kw_only=True)
class AnguloBound(Angulo, alias="AnguloBound"):
    r"""The Angulo et al. (2012) mass function of self-bound (SUBFIND) subhaloes.

    The same form as :class:`Angulo` (eq. 3 of the paper), fitted to all SUBFIND
    self-bound subhaloes, satellites included, rather than FoF groups. The paper
    notes that the abundance of :math:`M \sim 10^{15}\,M_\odot` objects changes by
    a factor of ~2 between the two definitions (Sec. 2.2).
    """

    parameter_source: ClassVar[str] = "Angulo et al. 2012 (arXiv v2), eq. 3."
    valid_domain: ClassVar[Domain] = _SIGMA_VALID
    calibration_domain: ClassVar[Domain | None] = Angulo.calibration_domain
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = MeasuredMassDefinition(
        kind="self_bound",
        note="Angulo et al. 2012, Sec. 2.2: all SUBFIND self-bound subhaloes.",
    )
    simulations: ClassVar[SimulationDetails | None] = Angulo.simulations
    A: float = _p(0.265, "The amplitude A.")
    b: float = _p(1.9, "The exponent b.")
    c: float = _p(1.4, "The exponential cut-off c.")
    d: float = _p(1.675, "The scale d of sigma.")


@attrs.frozen(kw_only=True)
class Watson_FoF(Warren, alias="Watson_FoF"):
    r"""The Watson et al. (2013) friends-of-friends mass function.

    The :class:`Warren` form, with Watson's parameters.
    """

    references: ClassVar[tuple[str, ...]] = (refs.WATSON13,)
    parameter_source: ClassVar[str] = "Watson et al. 2013 (arXiv v4 = published), eq. 12, Table 2."
    valid_domain: ClassVar[Domain] = _SIGMA_VALID
    calibration_domain: ClassVar[Domain | None] = Domain(
        {"ln_sigma_inv": (-0.55, 1.31), "z": (0, 26)},
        source=(
            "Watson et al. 2013: 'valid in the range -0.55 <= ln(1/sigma) < 1.31' "
            "(Sec. 4.4; 1.8e12-7.0e15 Msun/h at z = 0), and 'close to universal ... from "
            "z = 26 to the present' (Sec. 5.3), though ~20% high around ln(1/sigma) ~ 0.5 "
            "at z >~ 8. WMAP5 (Omega_m = 0.27, sigma_8 = 0.8; Sec. 2.1). Accuracy ~10% "
            "(Sec. 4.4)."
        ),
    )
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = _fof(
        note="Watson et al. 2013, Sec. 3 (Gadget-3 FoF)."
    )
    simulations: ClassVar[SimulationDetails | None] = SimulationDetails(
        box_size=[11.4, 20, 114, 425, 1000, 3200, 6000] * Mpc_h,
        n_particles=(3072**3, 5488**3, 3072**3, 5488**3, 3456**3, 4000**3, 6000**3),
        omega_m=(0.27, 0.27, 0.27, 0.27, 0.279, 0.279, 0.27),
        omega_b=(0.044, 0.044, 0.044, 0.044, 0.046, 0.046, 0.044),
        sigma_8=(0.8, 0.8, 0.8, 0.8, 0.817, 0.817, 0.8),
        h=(0.7, 0.7, 0.7, 0.7, 0.701, 0.701, 0.7),
        n_s=0.96,
        softening=[0.18, 0.18, 1.86, 3.87, 14.47, 40.0, 50.0] * kpc_h,
        transfer="CAMB",
        z_start=(300, 300, 300, 300, 150, 120, 100),
        initial_conditions="ZA",
        halo_finder="GADGET-3 FoF",
        n_min=1000,
        z_range=(0.0, 30.0),
        notes=(
            "The 1 and 3.2 Gpc/h boxes use the 'alternative' WMAP5 parameters. The "
            "Warren et al. FoF correction and a finite-box correction are applied."
        ),
        source="Watson et al. 2013 (arXiv v4), Secs. 2.1-2.2, 4.3 and Table 1.",
    )
    A: float = _p(0.282, "The amplitude A.")
    b: float = _p(2.163, "The exponent b of e/sigma.")
    c: float = _p(1.0, "The constant c added to (e/sigma)^b.")
    d: float = _p(1.21, "The exponential cut-off d.")
    e: float = _p(1.406, "The scale e of sigma.")


@attrs.frozen(kw_only=True)
class Watson(FittingFunction, alias="Watson"):
    r"""The Watson et al. (2013) spherical-overdensity (AHF) mass function.

    .. math:: f(\sigma) = \Gamma(\Delta, \sigma, z)\, A\left[\left(\frac{\beta}{\sigma}
        \right)^\alpha + 1\right]\exp\left(-\frac{\gamma}{\sigma^2}\right).

    Three fits are used, by redshift (published v4 of the paper): at z = 0 the
    present-day fit (``*_0``); at :math:`z \ge` ``z_hi`` the high-redshift fit
    (``*_hi``); in between the redshift-dependent fit of eqs. 14-16,
    :math:`X(z) = \Omega_m(z)[X_a(1+z)^{-X_b} + X_c]` for :math:`X = A, \alpha, \beta`,
    and :math:`\gamma =` ``gamma_z``. As in the paper, the fits are not joined
    smoothly. :math:`\Gamma` (eq. 12) corrects for the halo overdensity
    :math:`\Delta` (relative to the mean), and is 1 at :math:`\Delta = 178`.
    """

    references: ClassVar[tuple[str, ...]] = (refs.WATSON13,)
    parameter_source: ClassVar[str] = (
        "Watson et al. 2013 (arXiv v4 = published): Table 2 (z = 0 'AHF' and z >= 6 "
        "fits; the Sec. 4.5.2 text swaps alpha_0 and beta_0, Table 2 is followed), "
        "eqs. 14-16 (0 < z < 6) and eqs. 17-19 (Gamma)."
    )
    requires: ClassVar[frozenset[str]] = frozenset({"z", "omega_m_z", "delta_halo"})
    valid_domain: ClassVar[Domain] = Domain(
        {
            "sigma": _POSITIVE,
            "z": (0, None),
            "omega_m_z": _POSITIVE,
            "delta_halo": (1.0, None),
        },
        source=(
            "sigma > 0, z >= 0, Omega_m(z) > 0, and Delta >= 1 (a halo is overdense; "
            "Gamma diverges as Delta -> 0)."
        ),
    )
    calibration_domain: ClassVar[Domain | None] = Domain(
        {"ln_sigma_inv": (-0.55, 1.05), "z": (0, 26), "delta_halo": (100, 1600)},
        source=(
            "Watson et al. 2013, AHF haloes: the z = 0 fit is 'valid in the range "
            "-0.55 <= ln(1/sigma) < 1.05' (Sec. 4.5.2); the z-dependent fit is accurate "
            "to ~10% for z < 15 (Sec. 4.5.2, Fig. 9; no sigma range stated); the z >= 6 "
            "fit to 'all our data from z = 6 upwards' (Fig. 10, z = 6-26), valid for "
            "-0.06 <= ln(1/sigma) < 1.24. Gamma is calibrated for 100 <= Delta <= 1600 "
            "at z = 0, 1 and 3 (Sec. 4.6). WMAP5 (Omega_m = 0.27, sigma_8 = 0.8)."
        ),
    )
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = _so_any(
        _so_mean(178),
        note=(
            "Watson et al. 2013, Sec. 3.3: SO with Delta = 178 times the mean density "
            "(where Gamma = 1); AHF host haloes. hmf 3.x preferred the virial definition."
        ),
    )
    simulations: ClassVar[SimulationDetails | None] = _derived(
        Watson_FoF.simulations,
        halo_finder="AHF",
        notes=(
            "As for Watson_FoF; haloes found with AHF (host haloes only). The redshift "
            "evolution was calibrated on the CPMSO haloes."
        ),
    )

    C_a: float = _p(0.023, "Gamma: the coefficient of the C(Delta) exponent.")
    d_a: float = _p(0.456, "Gamma: d = -d_a Omega_m(z) - d_b.")
    d_b: float = _p(0.139, "Gamma: d = -d_a Omega_m(z) - d_b.")
    p: float = _p(0.072, "Gamma: the amplitude p of the sigma-dependent term.")
    q: float = _p(2.13, "Gamma: the exponent q of sigma.")
    A_0: float = _p(0.194, "A of the z = 0 fit.")
    alpha_0: float = _p(1.805, "alpha of the z = 0 fit.")
    beta_0: float = _p(2.267, "beta of the z = 0 fit.")
    gamma_0: float = _p(1.287, "gamma of the z = 0 fit.")
    z_hi: float = _p(6.0, "The redshift from which the high-redshift fit is used.")
    A_hi: float = _p(0.563, "A of the high-redshift fit.")
    alpha_hi: float = _p(3.810, "alpha of the high-redshift fit.")
    beta_hi: float = _p(0.874, "beta of the high-redshift fit.")
    gamma_hi: float = _p(1.453, "gamma of the high-redshift fit.")
    A_a: float = _p(1.097, "A(z) = Omega_m(z) [A_a (1+z)^-A_b + A_c].")
    A_b: float = _p(3.216, "A(z) = Omega_m(z) [A_a (1+z)^-A_b + A_c].")
    A_c: float = _p(0.074, "A(z) = Omega_m(z) [A_a (1+z)^-A_b + A_c].")
    alpha_a: float = _p(5.907, "alpha(z), as for A(z).")
    alpha_b: float = _p(3.599, "alpha(z), as for A(z).")
    alpha_c: float = _p(2.344, "alpha(z), as for A(z).")
    beta_a: float = _p(3.136, "beta(z), as for A(z).")
    beta_b: float = _p(3.058, "beta(z), as for A(z).")
    beta_c: float = _p(2.349, "beta(z), as for A(z).")
    gamma_z: float = _p(1.318, "gamma of the redshift-dependent fit.")

    @unit_boundary(returns=(None, None, None, None))
    def parameters(
        self, z: npt.ArrayLike, omega_m_z: npt.ArrayLike
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        r"""The parameters :math:`(A, \alpha, \beta, \gamma)` at redshift ``z``.

        Parameters
        ----------
        z
            Redshift.
        omega_m_z
            :math:`\Omega_m(z)`.

        Returns
        -------
        tuple of numpy.ndarray
            :math:`A, \alpha, \beta, \gamma`, broadcast over the inputs.
        """
        return self._parameters(z, omega_m_z)

    def _parameters(
        self, z: npt.ArrayLike, omega_m_z: npt.ArrayLike
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """:meth:`parameters`, for library code (no units boundary)."""
        z = np.asarray(z, dtype=np.float64)
        om = np.asarray(omega_m_z, dtype=np.float64)
        zp1 = 1.0 + z
        mid = (
            om * (self.A_a * zp1**-self.A_b + self.A_c),
            om * (self.alpha_a * zp1**-self.alpha_b + self.alpha_c),
            om * (self.beta_a * zp1**-self.beta_b + self.beta_c),
            np.full(np.broadcast(z, om).shape, self.gamma_z),
        )
        low = (self.A_0, self.alpha_0, self.beta_0, self.gamma_0)
        high = (self.A_hi, self.alpha_hi, self.beta_hi, self.gamma_hi)
        out = tuple(
            np.where(z == 0, lo, np.where(z >= self.z_hi, hi, m))
            for lo, hi, m in zip(low, high, mid, strict=True)
        )
        return out[0], out[1], out[2], out[3]

    def _fsigma(self, x: FitInputs) -> FloatArray:
        A, alpha, beta, gamma = self._parameters(x.z, x.omega_m_z)
        gamma_correction = _k.watson_gamma(
            x.sigma, x.delta_halo, x.omega_m_z, self.C_a, self.d_a, self.d_b, self.p, self.q
        )
        return gamma_correction * _k.angulo(x.sigma, A, alpha, gamma, beta)


_Z_VALID = Domain({"sigma": _POSITIVE, "z": (0, None)}, source="sigma > 0 and z >= 0.")


@attrs.frozen(kw_only=True)
class Crocce(FittingFunction, alias="Crocce"):
    r"""The Crocce et al. (2010) mass function (MICE).

    The :class:`Warren` form, with each parameter :math:`X(z) = X_a(1+z)^{-X_b}`, and
    e = 1.
    """

    references: ClassVar[tuple[str, ...]] = (refs.CROCCE10,)
    parameter_source: ClassVar[str] = "Crocce et al. 2010 (arXiv v2), eqs. 20 and 22, Table 2."
    requires: ClassVar[frozenset[str]] = frozenset({"z"})
    valid_domain: ClassVar[Domain] = _Z_VALID
    calibration_domain: ClassVar[Domain | None] = Domain(
        {"m": [2e10, 2.5e15] * Msun_h, "z": (0, 1)},
        source=(
            "Crocce et al. 2010: 2% accuracy for 2e10 < M < 2.5e15 Msun/h at z = 0 "
            "(Fig. 8 caption; the abstract says 1e10-1e15); fitted at z = 0 and 0.5 "
            "(Table 2) and accurate 'up to z = 1' (abstract), where the range ends at "
            "3.2e14 Msun/h (Sec. 8). Omega_m = 0.25, sigma_8 = 0.8 (Sec. 2)."
        ),
    )
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = _fof(
        note="Crocce et al. 2010, Sec. 3, with the Warren et al. mass correction."
    )
    simulations: ClassVar[SimulationDetails | None] = SimulationDetails(
        box_size=[7680, 3072, 4500, 768, 384, 179] * Mpc_h,
        n_particles=(2048**3, 2048**3, 1200**3, 1024**3, 1024**3, 1024**3),
        omega_m=0.25,
        omega_b=0.044,
        sigma_8=0.8,
        h=0.7,
        n_s=0.95,
        softening=[50, 50, 100, 50, 50, 50] * kpc_h,
        transfer="CAMB",
        z_start=(150, 50, 50, 50, 50, 50),
        initial_conditions=("ZA", "ZA", "2LPT", "2LPT", "2LPT", "2LPT"),
        n_min=200,
        z_range=(0.0, 1.0),
        names=("MICE7680", "MICE3072", "MICE4500", "MICE768", "MICE384", "MICE179"),
        notes=(
            "Haloes with at least 200 particles (50 in MICE179). Fitted at z = 0 and "
            "0.5, checked at z = 1. The Warren et al. FoF correction is applied."
        ),
        source="Crocce et al. 2010 (arXiv v2), Secs. 2, 3 and 7, Table 1.",
    )
    A_a: float = _p(0.58, "A(z) = A_a (1+z)^-A_b.")
    A_b: float = _p(0.13, "A(z) = A_a (1+z)^-A_b.")
    b_a: float = _p(1.37, "The exponent b(z) = b_a (1+z)^-b_b of e/sigma.")
    b_b: float = _p(0.15, "The exponent b(z) = b_a (1+z)^-b_b of e/sigma.")
    c_a: float = _p(0.3, "The constant c(z) = c_a (1+z)^-c_b.")
    c_b: float = _p(0.084, "The constant c(z) = c_a (1+z)^-c_b.")
    d_a: float = _p(1.036, "The cut-off d(z) = d_a (1+z)^-d_b.")
    d_b: float = _p(0.024, "The cut-off d(z) = d_a (1+z)^-d_b.")
    e: float = _p(1.0, "The scale e of sigma.")

    def _fsigma(self, x: FitInputs) -> FloatArray:
        zp1 = 1.0 + x.z
        return _k.warren(
            x.sigma,
            A=self.A_a * zp1**-self.A_b,
            b=self.b_a * zp1**-self.b_b,
            c=self.c_a * zp1**-self.c_b,
            d=self.d_a * zp1**-self.d_b,
            e=self.e,
        )


@attrs.frozen(kw_only=True)
class Bhattacharya(FittingFunction, alias="Bhattacharya"):
    r"""The Bhattacharya et al. (2011) mass function.

    .. math:: f(\sigma) = f_{\rm ST}(\sigma; A, a, p)\,(\sqrt{a}\,\nu)^{q-1},

    with :math:`A = A_a(1+z)^{-A_b}` and :math:`a = a_a(1+z)^{-a_b}`. With
    ``normed=True``, A is instead the value that normalises the fit to unit mass
    (which needs :math:`q > 0` and :math:`2p < q`). For q = 1 it is Sheth-Tormen.
    """

    references: ClassVar[tuple[str, ...]] = (refs.BHATTACHARYA11,)
    parameter_source: ClassVar[str] = "Bhattacharya et al. 2011 (arXiv v6), eq. 12 and Table 4."
    requires: ClassVar[frozenset[str]] = frozenset({"z", "delta_c"})
    valid_domain: ClassVar[Domain] = Domain(
        {"sigma": _POSITIVE, "z": (0, None), "delta_c": _POSITIVE},
        source="sigma > 0, z >= 0 and delta_c > 0.",
    )
    calibration_domain: ClassVar[Domain | None] = Domain(
        {"m": [4.3e11, 2.2e15] * Msun_h, "z": (0, 2)},
        source=(
            "Bhattacharya et al. 2011: 6e11-3e15 Msun (Table 4 title; abstract), i.e. "
            "4.3e11-2.2e15 Msun/h for their h = 0.72, and z = 0-2 (Table 4 title); "
            "redshift evolution better than 3% for 0.6 <= 1/sigma <= 2.4 (Sec. 4.2). "
            "Omega_m = 0.25, sigma_8 = 0.8 (Table 1)."
        ),
    )
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = _fof(
        note="Bhattacharya et al. 2011, Sec. 3, with FoF and finite-volume corrections."
    )
    simulations: ClassVar[SimulationDetails | None] = SimulationDetails(
        box_size=[1000 * 0.72, 1736 * 0.72, 2778 * 0.72, 178 * 0.72, 1300 * 0.72] * Mpc_h,
        n_particles=(1500**3, 1200**3, 1024**3, 512**3, 1024**3),
        omega_m=0.25,
        omega_b=0.0432,
        sigma_8=0.8,
        h=0.72,
        n_s=0.97,
        softening=[24 * 0.72, 51 * 0.72, 97 * 0.72, 14 * 0.72, 50 * 0.72] * kpc_h,
        transfer="CAMB",
        z_start=(75, 100, 100, 211, 211),
        initial_conditions=("2LPT", "2LPT", "2LPT", "ZA", "ZA"),
        n_min=400,
        z_range=(0.0, 2.0),
        notes=(
            "Model 0 (omega_m = 0.1296, omega_b = 0.0224, h = 0.72; Table 1). Box sizes "
            "and softenings are given in Mpc and kpc (converted here). Force-resolution, "
            "FoF and finite-volume corrections are applied."
        ),
        source="Bhattacharya et al. 2011 (arXiv v6), Secs. 2-3, Tables 1-2.",
    )
    A_a: float = _p(0.333, "A(z) = A_a (1+z)^-A_b (unless normed).")
    A_b: float = _p(0.11, "A(z) = A_a (1+z)^-A_b (unless normed).")
    a_a: float = _p(0.788, "a(z) = a_a (1+z)^-a_b.")
    a_b: float = _p(0.01, "a(z) = a_a (1+z)^-a_b.")
    p: float = _p(0.807, "The low-mass slope parameter p.")
    q: float = _p(1.795, "The exponent q (q = 1 gives Sheth-Tormen).", validator=positive)
    normed: bool = field(
        default=False,
        validator=attrs.validators.instance_of(bool),
        doc="Whether to normalise A so that all mass is in haloes.",
    )

    def __attrs_post_init__(self) -> None:
        """Check that the fit can be normalised: 2p < q."""
        if not 2 * self.p < self.q:
            raise ValueError(f"Bhattacharya: 2p must be < q, got p={self.p}, q={self.q}.")

    @property
    def normalized(self) -> bool:
        """Whether A normalises the fit (``normed``)."""
        return self.normed

    def _fsigma(self, x: FitInputs) -> FloatArray:
        zp1 = 1.0 + x.z
        A = _k.bhattacharya_norm(self.p, self.q) if self.normed else self.A_a * zp1**-self.A_b
        a = self.a_a * zp1**-self.a_b
        return _k.bhattacharya(x.nu, A, a, self.p, self.q)


# ---------------------------------------------------------------------------------
# Tinker 2008 and 2010, and Behroozi
# ---------------------------------------------------------------------------------
#: The overdensities (relative to the mean) of the Tinker 2008 and 2010 tables.
_TINKER_DELTAS = (200, 300, 400, 600, 800, 1200, 1600, 2400, 3200)


def _delta_table(fit: Tinker08 | Tinker10, name: str) -> FloatArray:
    """A parameter of a Tinker fit at each tabulated overdensity (its ``{name}_{delta}`` fields)."""
    return np.array([getattr(fit, f"{name}_{d}") for d in fit.delta_tab], dtype=np.float64)


@attrs.frozen(kw_only=True)
class Tinker08(FittingFunction, alias="Tinker08"):
    r"""The Tinker et al. (2008) spherical-overdensity mass function.

    .. math:: f(\sigma) = A\left[\left(\frac{\sigma}{b}\right)^{-a} + 1\right]
        \exp\left(-\frac{c}{\sigma^2}\right)

    (eq. 3). The parameters are tabulated at nine overdensities :math:`\Delta`
    (relative to the mean density; Table 2), and interpolated between them with a
    natural cubic spline in :math:`\log_{10}\Delta` (App. B). They evolve as
    :math:`A_0(1+z)^{-0.14}`, :math:`a_0(1+z)^{-0.06}` and :math:`b_0(1+z)^{-\alpha}`,
    with :math:`\log_{10}\alpha = -[0.75/\log_{10}(\Delta/75)]^{1.2}` (eqs. 5-8),
    which needs :math:`\Delta > 75`.

    The defaults are those of hmf 3.x, which round to Table 2 (they carry digits
    beyond the paper's three).
    """

    references: ClassVar[tuple[str, ...]] = (refs.TINKER08,)
    parameter_source: ClassVar[str] = (
        "Tinker et al. 2008 (arXiv v1 = published), Table 2 and eqs. 5-8; the "
        "extra digits reproduce Table B3's spline second derivatives."
    )
    requires: ClassVar[frozenset[str]] = frozenset({"z", "delta_halo"})
    valid_domain: ClassVar[Domain] = Domain(
        {
            "sigma": _POSITIVE,
            "z": (0, None),
            "delta_halo": (75.0, 3.0e4, "(]"),
        },
        source=(
            "sigma > 0, z >= 0, and 75 < Delta <= 3e4: eq. 8 needs Delta > 75, and the "
            "spline of the default parameters keeps all of them > 0 only for "
            "42.7 < Delta < 3.07e4. Parameters that are not > 0 raise in any case."
        ),
    )
    calibration_domain: ClassVar[Domain | None] = Domain(
        {"sigma": (0.4, 4.0), "z": (0, 2.5), "delta_halo": (200, 3200)},
        source=(
            "Tinker et al. 2008: 'calibrated over the range 0.25 <~ 1/sigma <~ 2.5' "
            "(Sec. 4; roughly 10^10.5-10^15.5 Msun/h at z = 0); z = [0, 2.5] (Sec. 4; "
            "outputs at z = 0, 0.5, 1.25, 2.5, Table 1), noisier and over a smaller "
            "sigma range at z = 2.5 (Sec. 3.3); 200 <= Delta <= 3200 (Table 2). WMAP1 "
            "and WMAP3-like cosmologies (Table 1). Accuracy <~5% at z = 0 (abstract)."
        ),
    )
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = _so_any(
        _so_mean(200), note="Tinker et al. 2008, Sec. 2.2: SO about density peaks."
    )
    simulations: ClassVar[SimulationDetails | None] = SimulationDetails(
        box_size=[
            768,
            384,
            271,
            192,
            96,
            1280,
            500,
            250,
            120,
            80,
            1000,
            500,
            500,
            500,
            384,
            384,
            120,
            80,
        ]
        * Mpc_h,
        n_particles=(1024**3,) * 5
        + (640**3, 1024**3, 512**3, 512**3, 512**3, 1024**3)
        + (512**3,) * 3
        + (1024**3, 1024**3, 1024**3, 512**3),
        omega_m=(0.3,) * 5
        + (0.27, 0.3, 0.3, 0.3, 0.3, 0.27, 0.24, 0.24, 0.24, 0.26, 0.2, 0.27, 0.23),
        omega_b=(0.04,) * 6
        + (0.045, 0.04, 0.04, 0.04, 0.044, 0.042, 0.042, 0.042, 0.044, 0.04, 0.044, 0.04),
        sigma_8=(0.9,) * 10 + (0.79, 0.75, 0.75, 0.8, 0.75, 0.9, 0.79, 0.75),
        h=(0.7,) * 11 + (0.73, 0.73, 0.73, 0.71, 0.7, 0.7, 0.73),
        n_s=(1.0,) * 10 + (0.95, 0.95, 0.95, 0.95, 0.94, 1.0, 0.95, 0.95),
        softening=[25, 14, 10, 4.9, 1.4, 120, 15, 7.6, 1.8, 1.2, 30, 15, 15, 15, 14, 14, 0.9, 1.2]
        * kpc_h,
        z_start=(40, 48, 51, 54, 65, 49, 40, 49, 49, 49, 60, 40, 40, 40, 35, 42, 100, 49),
        initial_conditions=("ZA",) * 5 + ("2LPT",) + ("ZA",) * 12,
        halo_finder="SO (own, about density peaks)",
        z_range=(0.0, 2.5),
        names=(
            "H768",
            "H384",
            "H271",
            "H192",
            "H96",
            "L1280",
            "L500",
            "L250",
            "L120",
            "L80",
            "L1000W",
            "L500Wa",
            "L500Wb",
            "L500Wc",
            "H384W",
            "H384Om",
            "L120W",
            "L80W",
        ),
        notes=(
            "Outputs at z = 0, 0.5, 1.25 and 2.5 (not all for every run). The L500 runs "
            "have as many SPH gas particles as dark matter (no cooling). The minimum "
            "particle number depends on Delta: 400-1600 (Table 2). Table 1's particle "
            "mass for H192 (5.89e8) does not follow from its box size (5.49e8). hmf "
            "3.x's Omega_b list was shifted by one from H384W on, and had all runs "
            "starting from ZA."
        ),
        source="Tinker et al. 2008, Sec. 2.1 and Table 1.",
    )
    delta_tab: ClassVar[tuple[int, ...]] = _TINKER_DELTAS

    A_200: float = _p(0.1858659, "The amplitude A at Delta = 200.")
    A_300: float = _p(0.1995973, "The amplitude A at Delta = 300.")
    A_400: float = _p(0.2115659, "The amplitude A at Delta = 400.")
    A_600: float = _p(0.2184113, "The amplitude A at Delta = 600.")
    A_800: float = _p(0.2480968, "The amplitude A at Delta = 800.")
    A_1200: float = _p(0.2546053, "The amplitude A at Delta = 1200.")
    A_1600: float = _p(0.26, "The amplitude A at Delta = 1600.")
    A_2400: float = _p(0.26, "The amplitude A at Delta = 2400.")
    A_3200: float = _p(0.26, "The amplitude A at Delta = 3200.")
    a_200: float = _p(1.466904, "The slope a at Delta = 200.")
    a_300: float = _p(1.521782, "The slope a at Delta = 300.")
    a_400: float = _p(1.559186, "The slope a at Delta = 400.")
    a_600: float = _p(1.614585, "The slope a at Delta = 600.")
    a_800: float = _p(1.869936, "The slope a at Delta = 800.")
    a_1200: float = _p(2.128056, "The slope a at Delta = 1200.")
    a_1600: float = _p(2.301275, "The slope a at Delta = 1600.")
    a_2400: float = _p(2.529241, "The slope a at Delta = 2400.")
    a_3200: float = _p(2.661983, "The slope a at Delta = 3200.")
    b_200: float = _p(2.571104, "The scale b at Delta = 200.")
    b_300: float = _p(2.254217, "The scale b at Delta = 300.")
    b_400: float = _p(2.048674, "The scale b at Delta = 400.")
    b_600: float = _p(1.869559, "The scale b at Delta = 600.")
    b_800: float = _p(1.588649, "The scale b at Delta = 800.")
    b_1200: float = _p(1.507134, "The scale b at Delta = 1200.")
    b_1600: float = _p(1.464374, "The scale b at Delta = 1600.")
    b_2400: float = _p(1.436827, "The scale b at Delta = 2400.")
    b_3200: float = _p(1.40521, "The scale b at Delta = 3200.")
    c_200: float = _p(1.193958, "The cut-off c at Delta = 200.")
    c_300: float = _p(1.270316, "The cut-off c at Delta = 300.")
    c_400: float = _p(1.335191, "The cut-off c at Delta = 400.")
    c_600: float = _p(1.446266, "The cut-off c at Delta = 600.")
    c_800: float = _p(1.581345, "The cut-off c at Delta = 800.")
    c_1200: float = _p(1.79505, "The cut-off c at Delta = 1200.")
    c_1600: float = _p(1.965613, "The cut-off c at Delta = 1600.")
    c_2400: float = _p(2.237466, "The cut-off c at Delta = 2400.")
    c_3200: float = _p(2.439729, "The cut-off c at Delta = 3200.")
    A_exp: float = _p(0.14, "A(z) = A (1+z)^-A_exp.")
    a_exp: float = _p(0.06, "a(z) = a (1+z)^-a_exp.")

    @unit_boundary(returns=(None, None, None, None))
    def parameters(
        self, delta_halo: npt.ArrayLike, z: npt.ArrayLike
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        r"""The parameters :math:`(A, a, b, c)` at overdensity ``delta_halo`` and ``z``.

        Parameters
        ----------
        delta_halo
            The halo overdensity relative to the mean density (> 75).
        z
            Redshift.

        Returns
        -------
        tuple of numpy.ndarray
            A, a, b, c, broadcast over the inputs.

        Raises
        ------
        DomainError
            If any parameter is not finite and > 0 (the spline extrapolates them far
            outside the tabulated overdensities).
        """
        return self._parameters(delta_halo, z)

    def _parameters(
        self, delta_halo: npt.ArrayLike, z: npt.ArrayLike
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """:meth:`parameters`, for library code (no units boundary)."""
        delta = np.asarray(delta_halo, dtype=np.float64)
        zp1 = 1.0 + np.asarray(z, dtype=np.float64)
        A0, a0, b0, c0 = (
            _k.log10_delta_spline(self.delta_tab, _delta_table(self, n), delta) for n in "Aabc"
        )
        A = A0 * zp1**-self.A_exp
        a = a0 * zp1**-self.a_exp
        b = b0 * zp1 ** -_k.tinker08_b_exponent(delta)
        c = c0 * np.ones_like(zp1)
        _check_physical(type(self).__name__, A=A, a=a, b=b, c=c)
        return A, a, b, c

    def _fsigma(self, x: FitInputs) -> FloatArray:
        A, a, b, c = self._parameters(x.delta_halo, x.z)
        return _k.tinker08(x.sigma, A, a, b, c)


@attrs.frozen(kw_only=True)
class Behroozi(Tinker08, alias="Behroozi"):
    r"""The Behroozi, Wechsler & Conroy (2013) mass function.

    :math:`f(\sigma)` is that of :class:`Tinker08` (evaluated at the virial
    overdensity), and :meth:`modify_dndm_kernel` applies the empirical high-redshift
    correction of App. G to the mass function:

    .. math::

        n(>M) = \theta(M, z)\, n_{\rm T08}(>M), \qquad
        \theta = 10^{\alpha(z)(M/M_\star)^{\gamma(z)}},

    with :math:`M_\star = 10^{11.5}\,M_\odot` (eqs. G2-G3).
    """

    references: ClassVar[tuple[str, ...]] = (refs.BEHROOZI13, refs.TINKER08)
    parameter_source: ClassVar[str] = (
        "Behroozi et al. 2013 (arXiv v2), App. G, eqs. G2-G3 (the correction); "
        "f(sigma) parameters from Tinker et al. 2008, Table 2."
    )
    valid_domain: ClassVar[Domain] = Tinker08.valid_domain
    calibration_domain: ClassVar[Domain | None] = Domain(
        {"z": (0, 9)},
        source=(
            "Behroozi et al. 2013, App. G: the correction is fit to the Consuelo "
            "simulation (Omega_m = 0.25, sigma_8 = 0.8, n_s = 1; Sec. 4) over "
            "a = 0.1-1, i.e. z ~ 0-9 (Fig. 23), with n(>M) at M = 10^11.5-10^13 Msun "
            "(Fig. 23, right). The range of sigma, and of masses for dn/dM: not stated. "
            "Tinker08 is applied at the virial overdensity, outside its calibrated "
            "Delta range at z >~ 1, deliberately."
        ),
    )
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = _so_virial(
        note="Behroozi et al. 2013, Sec. 4: Bryan & Norman (1998) virial SO, ROCKSTAR."
    )
    simulations: ClassVar[SimulationDetails | None] = SimulationDetails(
        box_size=[420.0] * Mpc_h,
        n_particles=1400**3,
        omega_m=0.25,
        sigma_8=0.8,
        h=0.7,
        n_s=1.0,
        softening=8.0 * kpc_h,
        initial_conditions="2LPT",
        halo_finder="ROCKSTAR",
        z_range=(0.0, 9.0),
        names="Consuelo",
        notes=(
            "The correction is fitted to Consuelo only, after an incompleteness "
            "correction (App. G1). Bolshoi (250 Mpc/h, 2048^3, Omega_m = 0.27, "
            "sigma_8 = 0.82) is shown for comparison. Softening: 'eight times worse' "
            "than Bolshoi's 1 kpc/h."
        ),
        source="Behroozi et al. 2013 (arXiv v2), Sec. 4 and App. G.",
    )
    modifies_dndm: ClassVar[bool] = True

    def modify_dndm_kernel(
        self,
        m: npt.ArrayLike,
        dndm: npt.ArrayLike,
        *,
        z: npt.ArrayLike,
        ngtm: npt.ArrayLike,
        h: npt.ArrayLike,
        omega_m0: npt.ArrayLike,
    ) -> FloatArray:
        """Apply the App. G correction to the Tinker (2008) mass function (kernel level).

        Parameters
        ----------
        m
            Halo masses, in Msun/h.
        dndm
            The Tinker (2008) mass function, in h^4 / (Msun Mpc^3).
        z
            Redshift.
        ngtm
            The cumulative Tinker (2008) mass function n(>m), in h^3 / Mpc^3.
        h
            The dimensionless Hubble parameter (the pivot mass is in Msun).
        omega_m0
            Not used.

        Returns
        -------
        numpy.ndarray
            The corrected mass function, in h^4 / (Msun Mpc^3).
        """
        h = np.asarray(h, dtype=np.float64)
        return _k.behroozi_modify_dndm(np.asarray(m, dtype=np.float64) / h, dndm, z, ngtm, h)


@attrs.frozen(kw_only=True)
class Tinker10(FittingFunction, alias="Tinker10"):
    r"""The Tinker et al. (2010) mass function.

    .. math:: f(\sigma) = \alpha\left[1 + (\beta\nu)^{-2\phi}\right]\nu^{2\eta + 1}
        \exp(-\gamma\nu^2/2)

    (eq. 8 times :math:`\nu`). The parameters are tabulated at nine overdensities
    (relative to the mean; Table 4) and splined in :math:`\log_{10}\Delta`. They
    evolve as :math:`\beta_0(1+z)^{0.2}`, :math:`\phi_0(1+z)^{-0.08}`,
    :math:`\eta_0(1+z)^{0.27}` and :math:`\gamma_0(1+z)^{-0.01}` (eqs. 9-12), frozen
    at ``max_z`` = 3, as the paper recommends. :math:`\alpha` is the tabulated value
    at z = 0 and a tabulated :math:`\Delta`, and otherwise the value that normalises
    the fit (which needs :math:`\beta, \gamma > 0`, :math:`\eta > -1/2` and
    :math:`\eta - \phi > -1/2`; otherwise it raises).
    """

    references: ClassVar[tuple[str, ...]] = (refs.TINKER10,)
    parameter_source: ClassVar[str] = (
        "Tinker et al. 2010 (arXiv v2), Table 4 and eqs. 9-12 (and the z = 3 cap, Sec. 4)."
    )
    requires: ClassVar[frozenset[str]] = frozenset({"z", "delta_halo", "delta_c"})
    valid_domain: ClassVar[Domain] = Domain(
        {
            "sigma": _POSITIVE,
            "z": (0, None),
            "delta_c": _POSITIVE,
            "delta_halo": (70.0, 3600.0),
        },
        source=(
            "sigma > 0, z >= 0, delta_c > 0, and 70 <= Delta <= 3600: there the spline "
            "of the default Table 4 parameters can be normalised (beta, gamma > 0, "
            "eta > -1/2, eta - phi > -1/2) at every z (it fails below 67.7 and above "
            "3662). Parameters that can not be normalised raise in any case."
        ),
    )
    calibration_domain: ClassVar[Domain | None] = Domain(
        {"sigma": (0.4, 4.0), "z": (0, 2.5), "delta_halo": (200, 3200)},
        source=(
            "Tinker et al. 2010 fit the z = 0 Tinker et al. 2008 data for "
            "200 <= Delta <= 3200 (Table 4; Sec. 4), so the sigma range is Tinker08's "
            "0.25 <~ 1/sigma <~ 2.5 (T08 Sec. 4; T10 states none). The redshift "
            "evolution (eqs. 9-12) is calibrated for Delta = 200 only (Sec. 4), on the "
            "T08 outputs (z <= 2.5); at other Delta only z = 0 is calibrated. Accuracy "
            "~5% for nu > 0.6 relative to T08 (Sec. 4)."
        ),
    )
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = _so_any(
        _so_mean(200), note="Tinker et al. 2010, Sec. 1: SO relative to the background."
    )
    simulations: ClassVar[SimulationDetails | None] = _derived(
        Tinker08.simulations,
        prepend_note=(
            "Tinker et al. 2010 fit the z = 0 mass functions of Tinker et al. 2008, so "
            "these are Tinker08's simulations (T10's own Table 1 lists 15 of them, for "
            "the bias)."
        ),
    )
    _normalized: ClassVar[bool] = True
    delta_tab: ClassVar[tuple[int, ...]] = _TINKER_DELTAS

    alpha_200: float = _p(0.368, "The normalisation alpha (used at z = 0 only) at Delta = 200.")
    alpha_300: float = _p(0.363, "The normalisation alpha (used at z = 0 only) at Delta = 300.")
    alpha_400: float = _p(0.385, "The normalisation alpha (used at z = 0 only) at Delta = 400.")
    alpha_600: float = _p(0.389, "The normalisation alpha (used at z = 0 only) at Delta = 600.")
    alpha_800: float = _p(0.393, "The normalisation alpha (used at z = 0 only) at Delta = 800.")
    alpha_1200: float = _p(0.365, "The normalisation alpha (used at z = 0 only) at Delta = 1200.")
    alpha_1600: float = _p(0.379, "The normalisation alpha (used at z = 0 only) at Delta = 1600.")
    alpha_2400: float = _p(0.355, "The normalisation alpha (used at z = 0 only) at Delta = 2400.")
    alpha_3200: float = _p(0.327, "The normalisation alpha (used at z = 0 only) at Delta = 3200.")
    beta_200: float = _p(0.589, "The parameter beta at Delta = 200.")
    beta_300: float = _p(0.585, "The parameter beta at Delta = 300.")
    beta_400: float = _p(0.544, "The parameter beta at Delta = 400.")
    beta_600: float = _p(0.543, "The parameter beta at Delta = 600.")
    beta_800: float = _p(0.564, "The parameter beta at Delta = 800.")
    beta_1200: float = _p(0.623, "The parameter beta at Delta = 1200.")
    beta_1600: float = _p(0.637, "The parameter beta at Delta = 1600.")
    beta_2400: float = _p(0.673, "The parameter beta at Delta = 2400.")
    beta_3200: float = _p(0.702, "The parameter beta at Delta = 3200.")
    gamma_200: float = _p(0.864, "The parameter gamma at Delta = 200.")
    gamma_300: float = _p(0.922, "The parameter gamma at Delta = 300.")
    gamma_400: float = _p(0.987, "The parameter gamma at Delta = 400.")
    gamma_600: float = _p(1.09, "The parameter gamma at Delta = 600.")
    gamma_800: float = _p(1.2, "The parameter gamma at Delta = 800.")
    gamma_1200: float = _p(1.34, "The parameter gamma at Delta = 1200.")
    gamma_1600: float = _p(1.5, "The parameter gamma at Delta = 1600.")
    gamma_2400: float = _p(1.68, "The parameter gamma at Delta = 2400.")
    gamma_3200: float = _p(1.81, "The parameter gamma at Delta = 3200.")
    phi_200: float = _p(-0.729, "The parameter phi at Delta = 200.")
    phi_300: float = _p(-0.789, "The parameter phi at Delta = 300.")
    phi_400: float = _p(-0.91, "The parameter phi at Delta = 400.")
    phi_600: float = _p(-1.05, "The parameter phi at Delta = 600.")
    phi_800: float = _p(-1.2, "The parameter phi at Delta = 800.")
    phi_1200: float = _p(-1.26, "The parameter phi at Delta = 1200.")
    phi_1600: float = _p(-1.45, "The parameter phi at Delta = 1600.")
    phi_2400: float = _p(-1.5, "The parameter phi at Delta = 2400.")
    phi_3200: float = _p(-1.49, "The parameter phi at Delta = 3200.")
    eta_200: float = _p(-0.243, "The parameter eta at Delta = 200.")
    eta_300: float = _p(-0.261, "The parameter eta at Delta = 300.")
    eta_400: float = _p(-0.261, "The parameter eta at Delta = 400.")
    eta_600: float = _p(-0.273, "The parameter eta at Delta = 600.")
    eta_800: float = _p(-0.278, "The parameter eta at Delta = 800.")
    eta_1200: float = _p(-0.301, "The parameter eta at Delta = 1200.")
    eta_1600: float = _p(-0.301, "The parameter eta at Delta = 1600.")
    eta_2400: float = _p(-0.319, "The parameter eta at Delta = 2400.")
    eta_3200: float = _p(-0.336, "The parameter eta at Delta = 3200.")
    beta_exp: float = _p(0.2, "beta(z) = beta (1+z)^beta_exp.")
    phi_exp: float = _p(-0.08, "phi(z) = phi (1+z)^phi_exp.")
    eta_exp: float = _p(0.27, "eta(z) = eta (1+z)^eta_exp.")
    gamma_exp: float = _p(-0.01, "gamma(z) = gamma (1+z)^gamma_exp.")
    max_z: float = _p(3.0, "The redshift above which the parameters stop evolving.")

    @unit_boundary(returns=(None, None, None, None, None))
    def parameters(
        self, delta_halo: npt.ArrayLike, z: npt.ArrayLike
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        r"""The parameters :math:`(\alpha, \beta, \gamma, \phi, \eta)` at ``delta_halo``, ``z``.

        Parameters
        ----------
        delta_halo
            The halo overdensity relative to the mean density.
        z
            Redshift.

        Returns
        -------
        tuple of numpy.ndarray
            alpha, beta, gamma, phi, eta, broadcast over the inputs.

        Raises
        ------
        DomainError
            If the parameters can not be normalised.
        """
        return self._parameters(delta_halo, z)

    def _parameters(
        self, delta_halo: npt.ArrayLike, z: npt.ArrayLike
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        """:meth:`parameters`, for library code (no units boundary)."""
        delta = np.asarray(delta_halo, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        zp1 = 1.0 + np.minimum(z, self.max_z)
        beta0, gamma0, phi0, eta0 = (
            _k.log10_delta_spline(self.delta_tab, _delta_table(self, n), delta)
            for n in ("beta", "gamma", "phi", "eta")
        )
        beta = beta0 * zp1**self.beta_exp
        gamma = gamma0 * zp1**self.gamma_exp
        phi = phi0 * zp1**self.phi_exp
        eta = eta0 * zp1**self.eta_exp
        _check_physical(
            type(self).__name__,
            beta=beta,
            gamma=gamma,
            eta_plus_half=eta + 0.5,
            eta_minus_phi_plus_half=eta - phi + 0.5,
        )
        tabulated = (z == 0) & np.isin(delta, self.delta_tab)
        # np.interp is exact at the nodes, where it is used.
        alpha_tab = np.interp(delta, self.delta_tab, _delta_table(self, "alpha"))
        alpha = np.where(tabulated, alpha_tab, _k.tinker10_norm(beta, gamma, phi, eta))
        return alpha, beta, gamma, phi, eta

    def _fsigma(self, x: FitInputs) -> FloatArray:
        return _k.tinker10(x.nu, *self._parameters(x.delta_halo, x.z))


# ---------------------------------------------------------------------------------
# Pillepich, Ishiyama, Bocquet and Yung
# ---------------------------------------------------------------------------------
@attrs.frozen(kw_only=True)
class Pillepich(Warren, alias="Pillepich"):
    r"""The Pillepich, Porciani & Hahn (2010) mass function: the :class:`Warren` form."""

    references: ClassVar[tuple[str, ...]] = (refs.PILLEPICH10,)
    parameter_source: ClassVar[str] = (
        "Pillepich et al. 2010 (arXiv v3), Sec. 3.1: the Gaussian (f_NL = 0) runs, in "
        "Warren notation (A, a, b, c) = (0.6853, 1.868, 0.3324, 1.2266)."
    )
    valid_domain: ClassVar[Domain] = _SIGMA_VALID
    calibration_domain: ClassVar[Domain | None] = Domain(
        {"ln_sigma_inv": (-1.2, 1.1), "z": (0, 1.6)},
        source=(
            "Pillepich et al. 2010, Sec. 3.1: '-1.2 < ln(1/sigma) < 1.1, which roughly "
            "corresponds to ... 2e10 < M < 5e15 Msun/h at z = 0', combining snapshots at "
            "z < 1.6. WMAP5 and WMAP3 (Table 2). Accuracy a few per cent (Sec. 3.1)."
        ),
    )
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = _fof(
        note="Pillepich et al. 2010, Sec. 2.1."
    )
    simulations: ClassVar[SimulationDetails | None] = SimulationDetails(
        box_size=[1200, 1200, 150] * Mpc_h,
        n_particles=1024**3,
        omega_m=(0.279, 0.24, 0.279),
        omega_b=(0.0462, 0.042, 0.0462),
        sigma_8=(0.817, 0.76, 0.817),
        h=(0.701, 0.73, 0.701),
        n_s=(0.96, 0.95, 0.96),
        softening=[20, 20, 3] * kpc_h,
        transfer="LINGER",
        z_start=(50, 50, 70),
        initial_conditions="ZA",
        n_min=100,
        z_range=(0.0, 1.6),
        names=("Run 1.0", "Run 2.0", "Run 3.0"),
        notes="The Gaussian (f_NL = 0) runs: WMAP5 (1.0, 3.0) and WMAP3 (2.0).",
        source="Pillepich et al. 2010 (arXiv v3), Secs. 2.1 and 3.1, Tables 1-2.",
    )
    A: float = _p(0.6853, "The amplitude A.")
    b: float = _p(1.868, "The exponent b of e/sigma.")
    c: float = _p(0.3324, "The constant c added to (e/sigma)^b.")
    d: float = _p(1.2266, "The exponential cut-off d.")
    e: float = _p(1.0, "The scale e of sigma.")


@attrs.frozen(kw_only=True)
class Ishiyama(Warren, alias="Ishiyama"):
    r"""The Ishiyama et al. (2015) mass function (nu^2 GC): the :class:`Warren` form, c = 1.

    The paper's :math:`A[(\sigma/B)^{-C} + 1]\exp(-D/\sigma^2)` (eq. 2) has
    :math:`(A, B, C, D)` = hmf's ``(A, e, b, d)``.
    """

    references: ClassVar[tuple[str, ...]] = (refs.ISHIYAMA15,)
    parameter_source: ClassVar[str] = "Ishiyama et al. 2015 (arXiv v3), eq. 2 and Table 3 (z = 0)."
    valid_domain: ClassVar[Domain] = _SIGMA_VALID
    calibration_domain: ClassVar[Domain | None] = Domain(
        {"log10_sigma_inv": (-0.7, 0.3), "z": (0, 10)},
        source=(
            "Ishiyama et al. 2015, Sec. 3.1: 'calibrated in the mass range of "
            "5e8 ~ 3e15 h^-1 Msun at z = 0, corresponding to -0.7 <= log(1/sigma) <= 0.3'; "
            "fit at z = 0, and within 10% over that sigma range at z = 0-10 (Figs. 7-8). "
            "Planck cosmology (Omega_m = 0.31, sigma_8 = 0.83; Sec. 2)."
        ),
    )
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = _fof(
        note="Ishiyama et al. 2015, Sec. 2."
    )
    simulations: ClassVar[SimulationDetails | None] = SimulationDetails(
        box_size=[1120, 560, 280, 140, 70] * Mpc_h,
        n_particles=(8192**3, 4096**3, 2048**3, 2048**3, 2048**3),
        omega_m=0.31,
        omega_b=0.048,
        sigma_8=0.83,
        h=0.68,
        n_s=0.96,
        softening=[4.27, 4.27, 4.27, 2.14, 1.07] * kpc_h,
        transfer="CAMB",
        z_start=127,
        initial_conditions="2LPT",
        n_min=40,
        z_range=(0.0, 0.0),
        names=("L", "M", "S", "H1", "H2"),
        notes="Fitted at z = 0; checked at z = 3, 7 and 10 (Figs. 7-8).",
        source="Ishiyama et al. 2015 (arXiv v3), Sec. 2 and Table 1.",
    )
    A: float = _p(0.193, "The amplitude A.")
    b: float = _p(1.550, "The exponent b of e/sigma (the paper's C).")
    c: float = _p(1.0, "The constant c added to (e/sigma)^b.")
    d: float = _p(1.186, "The exponential cut-off d (the paper's D).")
    e: float = _p(2.184, "The scale e of sigma (the paper's B).")


_BOCQUET_DOMAIN = Domain(
    {"m": [4.4e11, 7.0e15] * Msun_h, "z": (0, 2)},
    source=(
        "Bocquet et al. 2016 (arXiv v3): halo masses from 6.2e11 Msun (Table 1, Box4) to "
        "1e16 Msun (Sec. 2.2), i.e. 4.4e11-7.0e15 Msun/h for their h = 0.704, in the "
        "fit's own mass definition; z = 0-2 (Fig. 2 caption; Sec. 4.1). One cosmology "
        "(WMAP7: Omega_m = 0.272, sigma_8 = 0.809; Sec. 2.1). sigma range: not stated."
    ),
)


@attrs.frozen(kw_only=True)
class Bocquet200mDMOnly(FittingFunction, alias="Bocquet200mDMOnly"):
    r"""The Bocquet et al. (2016) mass function for M200m, dark matter only.

    .. math:: f(\sigma) = A(z)\left[\left(\frac{e(z)}{\sigma}\right)^{b(z)} + 1\right]
        \exp\left(-\frac{d(z)}{\sigma^2}\right),

    with :math:`X(z) = X(1+z)^{X_z}` (eqs. 3-4). The paper's (a, b, c) are hmf's
    (b, e, d).
    """

    references: ClassVar[tuple[str, ...]] = (refs.BOCQUET16,)
    parameter_source: ClassVar[str] = (
        "Bocquet et al. 2016 (arXiv v3 = published), Table 2; paper (a, b, c) -> hmf (b, e, d)."
    )
    requires: ClassVar[frozenset[str]] = frozenset({"z"})
    valid_domain: ClassVar[Domain] = _Z_VALID
    calibration_domain: ClassVar[Domain | None] = _BOCQUET_DOMAIN
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = _so_mean(
        200, note="Bocquet et al. 2016, Sec. 2.2: SO masses about SUBFIND potential minima."
    )
    simulations: ClassVar[SimulationDetails | None] = SimulationDetails(
        box_size=[68.1 * 0.704, 182 * 0.704, 1274 * 0.704] * Mpc_h,
        n_particles=(576**3, 576**3, 1526**3),
        omega_m=0.272,
        omega_b=0.0456,
        sigma_8=0.809,
        h=0.704,
        n_s=0.963,
        softening=[1.4 * 0.704, 3.75 * 0.704, 10 * 0.704] * kpc_h,
        z_start=60,
        initial_conditions="ZA",
        halo_finder="SUBFIND",
        n_min=10_000,
        z_range=(0.0, 2.0),
        names=("Box4/uhr", "Box3/hr", "Box1/mr"),
        notes=(
            "Magneticum, dark-matter-only runs. Box sizes and softenings are given in Mpc "
            "and kpc (converted here). Haloes have more than 1e4 particles within "
            "r_Delta; Poisson likelihood with a finite-volume correction."
        ),
        source="Bocquet et al. 2016 (arXiv v3), Secs. 2.1-2.2 and Tables 1-2.",
    )
    A: float = _p(0.175, "The amplitude A at z = 0.")
    b: float = _p(1.53, "The exponent b of e/sigma at z = 0 (the paper's a).")
    d: float = _p(1.19, "The exponential cut-off d at z = 0 (the paper's c).")
    e: float = _p(2.55, "The scale e of sigma at z = 0 (the paper's b).")
    A_z: float = _p(-0.012, "The redshift exponent of A.")
    b_z: float = _p(-0.04, "The redshift exponent of b.")
    d_z: float = _p(-0.021, "The redshift exponent of d.")
    e_z: float = _p(-0.194, "The redshift exponent of e.")

    @unit_boundary(returns=(None, None, None, None))
    def parameters(self, z: npt.ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """The parameters (A, b, d, e) at redshift ``z``."""
        return self._parameters(z)

    def _parameters(
        self, z: npt.ArrayLike
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """:meth:`parameters`, for library code (no units boundary)."""
        zp1 = 1.0 + np.asarray(z, dtype=np.float64)
        return (
            self.A * zp1**self.A_z,
            self.b * zp1**self.b_z,
            self.d * zp1**self.d_z,
            self.e * zp1**self.e_z,
        )

    @unit_boundary()
    def mass_ratio_to_200m(
        self, m: Any, *, z: npt.ArrayLike, omega_m0: npt.ArrayLike, H0: Any
    ) -> FloatArray:
        r"""The ratio :math:`M_\Delta/M_{200m}` of the fit's mass to M200m.

        Bocquet et al. build their 200c and 500c mass functions from
        :math:`dn/dM_\Delta = f(\sigma)(\bar\rho_m/M_\Delta)(d\ln\sigma^{-1}/dM_\Delta)
        (M_\Delta/M_{200m})` (eq. 5), with this ratio: 1 for the 200m fits, eq. A2
        for 200c and eq. 6 for 500c. :meth:`modify_dndm_kernel` multiplies the mass
        function by it (with :meth:`mass_ratio_to_200m_kernel`, the same calculation
        on plain arrays in canonical units). Evaluating sigma at M200m is left to the
        mass-function :class:`~hmf.core.stage.Stage` (issue #392).

        Parameters
        ----------
        m
            Halo mass in the fit's definition, a Quantity: in h-units, or physical
            units (converted with ``H0``).
        z
            Redshift.
        omega_m0
            The matter density parameter today.
        H0
            The Hubble constant, a scalar Quantity (e.g. ``70 * H0_unit``). It converts
            physical masses, and gives h to the fits, which take ln(M / Msun).

        Returns
        -------
        numpy.float64 or numpy.ndarray
            The (dimensionless) ratio, broadcast over ``m`` and ``z``.
        """
        where = f"{type(self).__name__}.mass_ratio_to_200m()"
        context, h = _hubble(H0, where)
        m = to_canonical(m, Msun_h, context=context, where=where, name="m")
        return self.mass_ratio_to_200m_kernel(m, z=z, omega_m0=omega_m0, h=h)

    def mass_ratio_to_200m_kernel(
        self, m: npt.ArrayLike, *, z: npt.ArrayLike, omega_m0: npt.ArrayLike, h: npt.ArrayLike
    ) -> FloatArray:
        """:meth:`mass_ratio_to_200m` at kernel level: ``m`` a plain array in Msun/h.

        1 for the 200m fits. A pure kernel (see :mod:`hmf.core._kernels`): ``z``,
        ``omega_m0`` and ``h`` (dimensionless) broadcast with ``m``.
        """
        return np.ones(np.broadcast(np.asarray(m), np.asarray(z)).shape)

    def modify_dndm_kernel(
        self,
        m: npt.ArrayLike,
        dndm: npt.ArrayLike,
        *,
        z: npt.ArrayLike,
        ngtm: npt.ArrayLike,
        h: npt.ArrayLike,
        omega_m0: npt.ArrayLike,
    ) -> FloatArray:
        """The mass function times :meth:`mass_ratio_to_200m_kernel` (eq. 5; kernel level).

        Parameters
        ----------
        m
            Halo masses in the fit's definition, in Msun/h.
        dndm
            The mass function from :meth:`fsigma`, in h^4 / (Msun Mpc^3).
        z
            Redshift.
        ngtm
            Not used.
        h
            The dimensionless Hubble parameter (the ratios take ln(M / Msun)).
        omega_m0
            The density parameter of CDM + baryons today.

        Returns
        -------
        numpy.ndarray
            The mass function in the fit's definition, in h^4 / (Msun Mpc^3).
        """
        ratio = self.mass_ratio_to_200m_kernel(m, z=z, omega_m0=omega_m0, h=h)
        out: FloatArray = np.asarray(dndm, dtype=np.float64) * ratio
        return out

    def _fsigma(self, x: FitInputs) -> FloatArray:
        A, b, d, e = self._parameters(x.z)
        return _k.warren(x.sigma, A=A, b=b, c=1.0, d=d, e=e)


@attrs.frozen(kw_only=True)
class Bocquet200mHydro(Bocquet200mDMOnly, alias="Bocquet200mHydro"):
    r"""The Bocquet et al. (2016) mass function for M200m, with hydrodynamics."""

    valid_domain: ClassVar[Domain] = _Z_VALID
    calibration_domain: ClassVar[Domain | None] = _BOCQUET_DOMAIN
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = (
        Bocquet200mDMOnly.measured_mass_definition
    )
    simulations: ClassVar[SimulationDetails | None] = SimulationDetails(
        box_size=[68.1 * 0.704, 182 * 0.704, 1274 * 0.704, 3818 * 0.704] * Mpc_h,
        n_particles=(576**3, 576**3, 1526**3, 4563**3),
        omega_m=0.272,
        omega_b=0.0456,
        sigma_8=0.809,
        h=0.704,
        n_s=0.963,
        softening=[1.4 * 0.704, 3.75 * 0.704, 10 * 0.704, 10 * 0.704] * kpc_h,
        z_start=60,
        initial_conditions="ZA",
        halo_finder="SUBFIND",
        n_min=10_000,
        z_range=(0.0, 2.0),
        names=("Box4/uhr", "Box3/hr", "Box1/mr", "Box0/mr"),
        notes=(
            "Magneticum, hydrodynamical runs (as many gas particles as dark matter; "
            "Box4/uhr runs only to z = 0.13). Box sizes and softenings are given in Mpc "
            "and kpc (converted here)."
        ),
        source="Bocquet et al. 2016 (arXiv v3), Secs. 2.1-2.2 and Tables 1-2.",
    )
    A: float = _p(0.228, "The amplitude A at z = 0.")
    b: float = _p(2.15, "The exponent b of e/sigma at z = 0 (the paper's a).")
    d: float = _p(1.3, "The exponential cut-off d at z = 0 (the paper's c).")
    e: float = _p(1.69, "The scale e of sigma at z = 0 (the paper's b).")
    A_z: float = _p(0.285, "The redshift exponent of A.")
    b_z: float = _p(-0.058, "The redshift exponent of b.")
    d_z: float = _p(-0.045, "The redshift exponent of d.")
    e_z: float = _p(-0.366, "The redshift exponent of e.")


_BOCQUET_200C_NOTE = (
    "Bocquet et al. 2016, Sec. 2.2. f(sigma) is meant to be used with sigma at "
    "M200m (issue #392), and the mass function with the ratio mass_ratio_to_200m "
    "(eq. 5; App. A), which modify_dndm applies."
)


@attrs.frozen(kw_only=True)
class Bocquet200cDMOnly(Bocquet200mDMOnly, alias="Bocquet200cDMOnly"):
    r"""The Bocquet et al. (2016) mass function for M200c, dark matter only.

    :meth:`fsigma` is the eq. 3 form with the 200c parameters only.
    :meth:`mass_ratio_to_200m` gives :math:`M_{200c}/M_{200m}` (eq. A2), which the
    mass function needs as well (eq. 5): :meth:`modify_dndm_kernel` multiplies it by
    the ratio (hmf 3.x multiplied :math:`f(\sigma)` by it, which is the same).
    """

    valid_domain: ClassVar[Domain] = _Z_VALID
    calibration_domain: ClassVar[Domain | None] = _BOCQUET_DOMAIN
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = _so_crit(
        200, note=_BOCQUET_200C_NOTE
    )
    simulations: ClassVar[SimulationDetails | None] = Bocquet200mDMOnly.simulations
    A: float = _p(0.222, "The amplitude A at z = 0.")
    b: float = _p(1.71, "The exponent b of e/sigma at z = 0 (the paper's a).")
    d: float = _p(1.46, "The exponential cut-off d at z = 0 (the paper's c).")
    e: float = _p(2.24, "The scale e of sigma at z = 0 (the paper's b).")
    A_z: float = _p(0.269, "The redshift exponent of A.")
    b_z: float = _p(0.321, "The redshift exponent of b.")
    d_z: float = _p(-0.153, "The redshift exponent of d.")
    e_z: float = _p(-0.621, "The redshift exponent of e.")

    modifies_dndm: ClassVar[bool] = True

    def mass_ratio_to_200m_kernel(
        self, m: npt.ArrayLike, *, z: npt.ArrayLike, omega_m0: npt.ArrayLike, h: npt.ArrayLike
    ) -> FloatArray:
        r""":math:`M_{200c}/M_{200m}`, Bocquet et al. (2016) eq. A2.

        Calibrated for 0 < z < 2, 1e13 < M200c / Msun < 2e16 and
        0.15 < Omega_m < 0.5 (App. A). ``m`` is a plain array in Msun/h.
        """
        m_msun = np.asarray(m, dtype=np.float64) / np.asarray(h, dtype=np.float64)
        return _k.bocquet16_mass_ratio_200c(m_msun, z, omega_m0)


@attrs.frozen(kw_only=True)
class Bocquet200cHydro(Bocquet200cDMOnly, alias="Bocquet200cHydro"):
    r"""The Bocquet et al. (2016) mass function for M200c, with hydrodynamics.

    See :class:`Bocquet200cDMOnly` for the mass conversion.
    """

    valid_domain: ClassVar[Domain] = _Z_VALID
    calibration_domain: ClassVar[Domain | None] = _BOCQUET_DOMAIN
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = (
        Bocquet200cDMOnly.measured_mass_definition
    )
    simulations: ClassVar[SimulationDetails | None] = Bocquet200mHydro.simulations
    A: float = _p(0.202, "The amplitude A at z = 0.")
    b: float = _p(2.21, "The exponent b of e/sigma at z = 0 (the paper's a).")
    d: float = _p(1.57, "The exponential cut-off d at z = 0 (the paper's c).")
    e: float = _p(2.0, "The scale e of sigma at z = 0 (the paper's b).")
    A_z: float = _p(1.147, "The redshift exponent of A.")
    b_z: float = _p(0.375, "The redshift exponent of b.")
    d_z: float = _p(-0.196, "The redshift exponent of d.")
    e_z: float = _p(-1.074, "The redshift exponent of e.")


@attrs.frozen(kw_only=True)
class Bocquet500cDMOnly(Bocquet200mDMOnly, alias="Bocquet500cDMOnly"):
    r"""The Bocquet et al. (2016) mass function for M500c, dark matter only.

    :meth:`fsigma` is the eq. 3 form with the 500c parameters only.
    :meth:`mass_ratio_to_200m` gives :math:`M_{500c}/M_{200m}` (eq. 6), which the
    mass function needs as well (eq. 5): :meth:`modify_dndm_kernel` multiplies it by
    the ratio (hmf 3.x multiplied :math:`f(\sigma)` by it, which is the same).
    """

    valid_domain: ClassVar[Domain] = _Z_VALID
    calibration_domain: ClassVar[Domain | None] = _BOCQUET_DOMAIN
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = _so_crit(
        500, note=_BOCQUET_200C_NOTE.replace("App. A", "eq. 6")
    )
    simulations: ClassVar[SimulationDetails | None] = Bocquet200mDMOnly.simulations
    A: float = _p(0.241, "The amplitude A at z = 0.")
    b: float = _p(2.18, "The exponent b of e/sigma at z = 0 (the paper's a).")
    d: float = _p(2.02, "The exponential cut-off d at z = 0 (the paper's c).")
    e: float = _p(2.35, "The scale e of sigma at z = 0 (the paper's b).")
    A_z: float = _p(0.37, "The redshift exponent of A.")
    b_z: float = _p(0.251, "The redshift exponent of b.")
    d_z: float = _p(-0.31, "The redshift exponent of d.")
    e_z: float = _p(-0.698, "The redshift exponent of e.")

    modifies_dndm: ClassVar[bool] = True

    def mass_ratio_to_200m_kernel(
        self, m: npt.ArrayLike, *, z: npt.ArrayLike, omega_m0: npt.ArrayLike, h: npt.ArrayLike
    ) -> FloatArray:
        r""":math:`M_{500c}/M_{200m}`, Bocquet et al. (2016) eq. 6.

        Calibrated for 0 < z < 2, 1e13 < M500c / Msun < 1e16 and 0.1 < Omega_m < 0.5
        (Sec. 3.2.2). ``m`` is a plain array in Msun/h.
        """
        m_msun = np.asarray(m, dtype=np.float64) / np.asarray(h, dtype=np.float64)
        return _k.bocquet16_mass_ratio_500c(m_msun, z, omega_m0)


@attrs.frozen(kw_only=True)
class Bocquet500cHydro(Bocquet500cDMOnly, alias="Bocquet500cHydro"):
    r"""The Bocquet et al. (2016) mass function for M500c, with hydrodynamics.

    See :class:`Bocquet500cDMOnly` for the mass conversion.
    """

    valid_domain: ClassVar[Domain] = _Z_VALID
    calibration_domain: ClassVar[Domain | None] = _BOCQUET_DOMAIN
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = (
        Bocquet500cDMOnly.measured_mass_definition
    )
    simulations: ClassVar[SimulationDetails | None] = Bocquet200mHydro.simulations
    A: float = _p(0.18, "The amplitude A at z = 0.")
    b: float = _p(2.29, "The exponent b of e/sigma at z = 0 (the paper's a).")
    d: float = _p(1.97, "The exponential cut-off d at z = 0 (the paper's c).")
    e: float = _p(2.44, "The scale e of sigma at z = 0 (the paper's b).")
    A_z: float = _p(1.088, "The redshift exponent of A.")
    b_z: float = _p(0.15, "The redshift exponent of b.")
    d_z: float = _p(-0.322, "The redshift exponent of d.")
    e_z: float = _p(-1.008, "The redshift exponent of e.")


#: Yung et al. (2024) Table A1 (masses in Msun/h) and Table A2 (masses in Msun).
_YUNG24_TABLES: Mapping[str, Mapping[str, float]] = MappingProxyType(
    {
        "h": MappingProxyType(
            {
                "A_0": 0.11416632,
                "A_1": -0.01486746,
                "A_2": 0.00137191,
                "a_0": 1.05274399,
                "a_1": 0.02803087,
                "a_2": -0.00306126,
                "b_0": 8.62813020,
                "b_1": 0.00384969,
                "b_2": -0.02349983,
                "c_0": 1.13138924,
                "c_1": 0.01713172,
                "c_2": -0.00113630,
            }
        ),
        "physical": MappingProxyType(
            {
                "A_0": 0.13765772,
                "A_1": -0.01003821,
                "A_2": 0.00102964,
                "a_0": 1.06641384,
                "a_1": 0.02475576,
                "a_2": -0.00283342,
                "b_0": 4.86693806,
                "b_1": 0.09212356,
                "b_2": -0.01426283,
                "c_0": 1.19837952,
                "c_1": -0.00142967,
                "c_2": -0.00033074,
            }
        ),
    }
)


def _yung24_default(name: str) -> Any:
    """The default of a Yung24 coefficient: from the table chosen by ``mass_units``."""
    # An invalid `mass_units` falls back to "h" here, and is then rejected by its
    # validator.
    return attrs.Factory(
        lambda self: _YUNG24_TABLES.get(self.mass_units, _YUNG24_TABLES["h"])[name],
        takes_self=True,
    )


def _yung24_coefficient(name: str) -> Any:
    chi, power = name.split("_")
    return field(
        default=_yung24_default(name),
        converter=float,
        doc=f"The z^{power} coefficient of {chi}(z) (default: from the table for `mass_units`).",
    )


@attrs.frozen(kw_only=True)
class Yung24(FittingFunction, alias="Yung24"):
    r"""The Yung et al. (2024) mass function (GUREFT), for :math:`6 \le z \le 19`.

    .. math:: f(\sigma) = A(z)\left[\left(\frac{\sigma}{b(z)}\right)^{-a(z)} + 1\right]
        \exp\left(-\frac{c(z)}{\sigma^2}\right),

    with :math:`\chi(z) = \chi_0 + \chi_1 z + \chi_2 z^2` for
    :math:`\chi \in \{A, a, b, c\}` (eq. A2). ``mass_units="h"`` takes the coefficients
    of Table A1 (masses in Msun/h, hmf's convention), ``mass_units="physical"`` those
    of Table A2 (masses in Msun); coefficients given explicitly override the table.
    The paper's quadratics are fitted for z = 6-19 only, and are not used outside it.
    """

    references: ClassVar[tuple[str, ...]] = (refs.YUNG24,)
    parameter_source: ClassVar[str] = (
        "Yung et al. 2024 (arXiv v3), App. A: Table A1 (mass_units='h') and Table A2 "
        "(mass_units='physical')."
    )
    requires: ClassVar[frozenset[str]] = frozenset({"z"})
    valid_domain: ClassVar[Domain] = Domain(
        {"sigma": _POSITIVE, "z": (6, 19)},
        source=(
            "sigma > 0 and 6 <= z <= 19: the redshift quadratics of App. A are fitted "
            "over z = 6-19 only (hmf 3.x raised outside it too)."
        ),
    )
    calibration_domain: ClassVar[Domain | None] = Domain(
        {"m": [1e6, 1e13] * Msun_h, "z": (6, 19)},
        source=(
            "Yung et al. 2024, App. A: 'fitted to gureft+MultiDark HMFs between z = 6 "
            "to 19', over 6 < log10(M_vir / (Msun/h)) < 13 (for mass_units='h'; for "
            "mass_units='physical' Fig. A1 shows 10^6-10^13 Msun, i.e. ~10^5.83-10^12.83 "
            "Msun/h for their h = 0.678, which this domain does not adjust for). "
            "Planck cosmology (Omega_m = 0.307, sigma_8 = 0.829; Sec. 2)."
        ),
    )
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = _so_virial(
        note="Yung et al. 2024, Sec. 2: ROCKSTAR, Bryan & Norman virial; includes subhaloes."
    )
    simulations: ClassVar[SimulationDetails | None] = SimulationDetails(
        box_size=[5, 15, 35, 90, 250, 160] * Mpc_h,
        n_particles=(1024**3, 1024**3, 1024**3, 1024**3, 2048**3, 3840**3),
        omega_m=0.307,
        sigma_8=0.829,
        h=0.678,
        n_s=0.96,
        z_start=(200, 200, 200, 200, None, None),
        initial_conditions=("2LPT", "2LPT", "2LPT", "2LPT", None, None),
        halo_finder="ROCKSTAR",
        n_min=100,
        z_range=(6.0, 19.0),
        names=("gureft-05", "gureft-15", "gureft-35", "gureft-90", "Bolshoi-Planck", "VSMDPL"),
        notes=(
            "Haloes and subhaloes. Bolshoi-Planck is used for z <~ 10 and VSMDPL for "
            "z > 10 (Fig. 3)."
        ),
        source="Yung et al. 2024 (arXiv v3), Secs. 2-3, Table 1 and Fig. 3.",
    )

    mass_units: Literal["h", "physical"] = field(
        default="h",
        validator=attrs.validators.in_(("h", "physical")),
        doc=(
            "The mass units the table of coefficients was fitted in: 'h' (Msun/h, "
            "Table A1) or 'physical' (Msun, Table A2)."
        ),
    )
    A_0: float = _yung24_coefficient("A_0")
    A_1: float = _yung24_coefficient("A_1")
    A_2: float = _yung24_coefficient("A_2")
    a_0: float = _yung24_coefficient("a_0")
    a_1: float = _yung24_coefficient("a_1")
    a_2: float = _yung24_coefficient("a_2")
    b_0: float = _yung24_coefficient("b_0")
    b_1: float = _yung24_coefficient("b_1")
    b_2: float = _yung24_coefficient("b_2")
    c_0: float = _yung24_coefficient("c_0")
    c_1: float = _yung24_coefficient("c_1")
    c_2: float = _yung24_coefficient("c_2")

    @unit_boundary(returns=(None, None, None, None))
    def parameters(self, z: npt.ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """The parameters (A, a, b, c) at redshift ``z``."""
        return self._parameters(z)

    def _parameters(
        self, z: npt.ArrayLike
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """:meth:`parameters`, for library code (no units boundary)."""
        z = np.asarray(z, dtype=np.float64)
        return (
            self.A_0 + self.A_1 * z + self.A_2 * z**2,
            self.a_0 + self.a_1 * z + self.a_2 * z**2,
            self.b_0 + self.b_1 * z + self.b_2 * z**2,
            self.c_0 + self.c_1 * z + self.c_2 * z**2,
        )

    def _fsigma(self, x: FitInputs) -> FloatArray:
        A, a, b, c = self._parameters(x.z)
        return _k.tinker08(x.sigma, A, a, b, c)
