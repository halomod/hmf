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
already-resolved physical inputs (the :class:`~hmf.core.stage.Stage` that computes the
mass function resolves them). It is a units boundary (:mod:`hmf.core.units`): the
mass is a Quantity, and every other input is dimensionless.

================  ===================================================================
``sigma``         :math:`\sigma(m, z)`; always required.
``z``             redshift.
``omega_m_z``     the matter density parameter at ``z``, :math:`\Omega_m(z)`.
``delta_halo``    the halo overdensity relative to the **mean** density, :math:`\Delta_m`.
``delta_c``       the critical overdensity for collapse, :math:`\delta_c`.
``n_eff``         the effective spectral index at ``m``.
``m``             the halo mass (a Quantity at the public methods).
================  ===================================================================

Each fit lists the ones it needs in :attr:`FittingFunction.requires`. All of them
broadcast against each other.

Library code (the stages) works on plain arrays in canonical units instead: it puts
them in a :class:`FitInputs` and calls :func:`evaluate_fsigma` (or
:meth:`FittingFunction.fsigma_from_inputs`), so it pays no boundary cost.

Domains
-------
Each fit has two :class:`~hmf.core.domain.Domain`\ s (issue #390):

:attr:`FittingFunction.valid_domain`
    where its formula is defined, or sensible. :meth:`FittingFunction.fsigma`
    **always raises** :class:`~hmf.core.domain.DomainError` outside it.
:attr:`FittingFunction.calibration_domain`
    where it was calibrated against simulations, with its source. Outside it, the
    result is an extrapolation, handled by a :data:`~hmf.core.domain.DomainPolicy`
    with :func:`evaluate_fsigma`.

The bounds of both are in these variables: the inputs above, and the derived
``ln_sigma_inv`` (:math:`\ln\sigma^{-1}`), ``log10_sigma_inv``
(:math:`\log_{10}\sigma^{-1}`) and ``peak_height`` (:math:`\nu`).

Mass definitions
----------------
Each fit records the mass definition it was measured in, as
:class:`MeasuredMassDefinition` metadata. Converting between mass definitions is not
part of this module (issue #392).
"""

from __future__ import annotations

import abc
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, ClassVar, Literal

import attrs
import numpy as np
import numpy.typing as npt

from ._fields import field
from ._kernels import fits as _k
from .domain import Domain, DomainError, DomainPolicy, apply_domain_policy
from .model import Model
from .units import Msun_h, dndm_unit, number_density_unit, unit_boundary

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

#: The smallest float above zero: the lower bound of a variable that must be > 0
#: (:class:`~hmf.core.domain.Interval` bounds are inclusive).
_TINY = float(np.nextafter(0.0, 1.0))

#: sigma > 0, the valid-domain bound of every fit.
_SIGMA_POSITIVE = (_TINY, None)


# ---------------------------------------------------------------------------------
# Metadata: measured mass definitions
# ---------------------------------------------------------------------------------
MassDefinitionKind = Literal["fof", "so_mean", "so_critical", "so_virial", "so_any", "self_bound"]


@attrs.frozen(kw_only=True)
class MeasuredMassDefinition:
    """The halo mass definition a fit was measured in (metadata only).

    This is a simple, hashable record, to be mapped onto the mass-definition types
    of issue #392. It does no conversion.

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
# Inputs
# ---------------------------------------------------------------------------------
def _as_float_array(x: npt.ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


def _as_array(x: npt.ArrayLike | None) -> FloatArray | None:
    return None if x is None else np.asarray(x, dtype=np.float64)


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

    sigma: FloatArray = attrs.field(converter=_as_float_array)
    _z: FloatArray | None = attrs.field(default=None, converter=_as_array, alias="z")
    _omega_m_z: FloatArray | None = attrs.field(
        default=None, converter=_as_array, alias="omega_m_z"
    )
    _delta_halo: FloatArray | None = attrs.field(
        default=None, converter=_as_array, alias="delta_halo"
    )
    _delta_c: FloatArray | None = attrs.field(default=None, converter=_as_array, alias="delta_c")
    _n_eff: FloatArray | None = attrs.field(default=None, converter=_as_array, alias="n_eff")
    _m: FloatArray | None = attrs.field(default=None, converter=_as_array, alias="m")

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


def _domain_inputs(domain: Domain) -> frozenset[str]:
    """The inputs (besides sigma) needed to check a domain."""
    needs: set[str] = set()
    for name in domain.variables:
        if name in _DERIVED_NEEDS:
            needs.update(_DERIVED_NEEDS[name])
        elif name != "sigma":
            needs.add(name)
    return frozenset(needs)


def _contains(domain: Domain, x: FitInputs) -> bool | BoolArray:
    """Whether the inputs ``x`` are inside ``domain``."""
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

    #: Where the fit was calibrated against simulations, with its source.
    calibration_domain: ClassVar[Domain]

    #: The halo mass definition the fit was measured in.
    measured_mass_definition: ClassVar[MeasuredMassDefinition]

    #: Whether :meth:`modify_dndm` changes the mass function (see :class:`Behroozi`).
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

        Parameters are as for :meth:`fsigma` (``m`` is a Quantity). Library code that
        already has plain arrays in canonical units builds :class:`FitInputs` directly.

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
        apply_domain_policy(
            None,
            _contains(self.valid_domain, x),
            "raise",
            description=f"{type(self).__name__}'s valid domain ({_describe(self.valid_domain)})",
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
            A bool if every input is a scalar, else the broadcast array.

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

        This is a units boundary (see :mod:`hmf.core.units`): ``m`` must be a
        Quantity; every other input is dimensionless. Library code with plain arrays
        in canonical units calls :meth:`fsigma_from_inputs` or :func:`evaluate_fsigma`
        instead. It applies no calibration policy (see :func:`evaluate_fsigma` for
        that), but always checks :attr:`valid_domain`.

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
        return self.fsigma_from_inputs(x)

    def fsigma_from_inputs(self, x: FitInputs) -> FloatArray:
        """:meth:`fsigma`, from plain inputs in canonical units (the library path).

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

    @unit_boundary(m=Msun_h, dndm=dndm_unit, ngtm=number_density_unit, returns=dndm_unit)
    def modify_dndm(
        self, m: Any, dndm: Any, *, z: npt.ArrayLike, ngtm: Any, h: npt.ArrayLike
    ) -> Any:
        """Modify the mass function computed from :meth:`fsigma` (a no-op by default).

        A fit that is not a pure function of sigma (e.g. :class:`Behroozi`) overrides
        :meth:`_modify_dndm` (the unit-free implementation, which library code calls),
        and sets :attr:`modifies_dndm`.

        Parameters
        ----------
        m
            Halo masses, a Quantity.
        dndm
            The mass function from :meth:`fsigma`, a Quantity (number density per mass).
        z
            Redshift.
        ngtm
            The cumulative mass function n(>m) from :meth:`fsigma`, a Quantity.
        h
            The dimensionless Hubble parameter.

        Returns
        -------
        Quantity
            The modified mass function, in :data:`~hmf.core.units.dndm_unit`.
        """
        return self._modify_dndm(m, dndm, z=z, ngtm=ngtm, h=h)

    def _modify_dndm(
        self,
        m: npt.ArrayLike,
        dndm: npt.ArrayLike,
        *,
        z: npt.ArrayLike,
        ngtm: npt.ArrayLike,
        h: npt.ArrayLike,
    ) -> FloatArray:
        """:meth:`modify_dndm` on plain arrays in canonical units (identity here).

        ``m`` in Msun/h, ``dndm`` in h^4 / (Msun Mpc^3) and ``ngtm`` in h^3 / Mpc^3.
        """
        return np.asarray(dndm, dtype=np.float64)


def _describe(domain: Domain) -> str:
    """A short description of a domain's bounds."""
    parts = []
    for name, interval in domain.bounds:
        unit = f" {interval.unit}" if interval.unit is not None else ""
        lower = (
            ""
            if interval.lower == -np.inf
            else f"{name} > 0"
            if interval.lower == _TINY
            else f"{name} >= {interval.lower:g}"
        )
        upper = "" if interval.upper == np.inf else f"{name} <= {interval.upper:g}"
        text = " and ".join(p for p in (lower, upper) if p)
        parts.append(text + unit if text else f"any {name}")
    return ", ".join(parts) or "unbounded"


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

    This is the library path, for the stages: the inputs are plain arrays in
    canonical units (a :class:`FitInputs`, built directly or by
    :meth:`FittingFunction.inputs` from Quantities).

    The valid domain always raises (see :meth:`FittingFunction.fsigma`); the policy
    applies to the calibration domain only:

    ``"ignore"``
        return :math:`f(\sigma)` everywhere;
    ``"warn"``
        also emit an :class:`~hmf.exceptions.HMFExtrapolationWarning` if any value is
        outside. It is emitted on each call: a stage that wants it once per instance
        calls this once and caches the result;
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
    f = model.fsigma_from_inputs(x)
    inside = np.broadcast_to(model.in_calibration_domain(x), f.shape)
    name = type(model).__name__
    out = apply_domain_policy(
        f,
        inside,
        policy,
        description=f"{name}'s calibration domain ({_describe(model.calibration_domain)})",
    )
    return FSigmaResult(np.asarray(out, dtype=np.float64), np.array(inside, dtype=bool))


def _p(default: Any, doc: str, **kwargs: Any) -> Any:
    """A fit parameter: a float field with documentation."""
    return field(default=default, doc=doc, **kwargs)


def _check_physical(name: str, **params: FloatArray) -> None:
    """Raise a DomainError if any of ``params`` is not finite and > 0."""
    for key, value in params.items():
        bad = ~(np.isfinite(value) & (value > 0))
        if np.any(bad):
            raise DomainError(
                f"{name}: parameter {key} is unphysical (must be finite and > 0) for "
                f"{int(np.count_nonzero(bad))} of {np.size(bad)} input(s)."
            )


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

    references: ClassVar[tuple[str, ...]] = (
        (
            "Press, W. H., Schechter, P., 1974. ApJ 187, 425. "
            "https://ui.adsabs.harvard.edu/abs/1974ApJ...187..425P"
        ),
    )
    parameter_source: ClassVar[str] = "Press & Schechter 1974: analytic, no free parameters."
    requires: ClassVar[frozenset[str]] = frozenset({"delta_c"})
    valid_domain: ClassVar[Domain] = Domain(
        {"sigma": _SIGMA_POSITIVE, "delta_c": (_TINY, None)},
        source="sigma > 0 and delta_c > 0.",
    )
    calibration_domain: ClassVar[Domain] = Domain(
        {}, source="Analytic (spherical collapse); not calibrated on simulations."
    )
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = _so_virial(
        note="Analytic: spherical collapse, i.e. virialised haloes."
    )
    _normalized: ClassVar[bool] = True

    def _fsigma(self, x: FitInputs) -> FloatArray:
        return _k.press_schechter(x.nu)


_SMT_REF = (
    "Sheth, R. K., Mo, H. J., Tormen, G., 2001. MNRAS 323, 1. "
    "https://doi.org/10.1046/j.1365-8711.2001.04006.x"
)
_ST99_REF = (
    "Sheth, R. K., Tormen, G., 1999. MNRAS 308, 119. "
    "https://doi.org/10.1046/j.1365-8711.1999.02692.x"
)

_NU_VALID = Domain(
    {"sigma": _SIGMA_POSITIVE, "delta_c": (_TINY, None)}, source="sigma > 0 and delta_c > 0."
)


def _positive(name: str) -> Callable[[Any, attrs.Attribute[Any], Any], None]:
    """An attrs validator: the value must be finite and > 0."""

    def check(instance: Any, attribute: attrs.Attribute[Any], value: Any) -> None:
        if not (np.isfinite(value) and value > 0):
            raise ValueError(f"{type(instance).__name__}.{name} must be > 0, got {value}.")

    return check


def _below_half(instance: Any, attribute: attrs.Attribute[Any], value: float) -> None:
    """An attrs validator: p < 1/2, so that the Sheth-Tormen form can be normalised."""
    if not value < 0.5:
        raise ValueError(f"{type(instance).__name__}.p must be < 0.5, got {value}.")


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

    references: ClassVar[tuple[str, ...]] = (_SMT_REF, _ST99_REF)
    parameter_source: ClassVar[str] = (
        "Sheth & Tormen 1999, eq. 10 (a = 0.707, p = 0.3); A from the normalisation "
        "(Sheth, Mo & Tormen 2001, eq. 5: A = 0.3222)."
    )
    requires: ClassVar[frozenset[str]] = frozenset({"delta_c"})
    valid_domain: ClassVar[Domain] = _NU_VALID
    calibration_domain: ClassVar[Domain] = Domain(
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

    a: float = _p(0.707, "The parameter a (rescales nu^2).", validator=_positive("a"))
    p: float = _p(0.3, "The low-mass slope parameter p (< 0.5).", validator=_below_half)
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

    calibration_domain: ClassVar[Domain] = SMT.calibration_domain
    valid_domain: ClassVar[Domain] = SMT.valid_domain
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = SMT.measured_mass_definition


@attrs.frozen(kw_only=True)
class Reed03(SMT, alias="Reed03"):
    r"""The Reed et al. (2003) mass function: Sheth-Tormen with a high-z correction.

    .. math:: f(\sigma) = f_{\rm ST}(\sigma)\exp\left[-\frac{c}{\sigma\cosh^5(2\sigma)}\right].
    """

    references: ClassVar[tuple[str, ...]] = (
        "Reed, D., et al., 2003. MNRAS 346, 565. https://doi.org/10.1046/j.1365-2966.2003.07113.x",
    )
    parameter_source: ClassVar[str] = (
        "Reed et al. 2003 (arXiv v2), eq. 5 (A, a, p of Sheth-Tormen) and eq. 9 (c = 0.7)."
    )
    valid_domain: ClassVar[Domain] = _NU_VALID
    calibration_domain: ClassVar[Domain] = Domain(
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

    references: ClassVar[tuple[str, ...]] = (
        (
            "Courtin, J., et al., 2011. MNRAS 410, 1911. "
            "https://doi.org/10.1111/j.1365-2966.2010.17573.x"
        ),
    )
    parameter_source: ClassVar[str] = (
        "Courtin et al. 2011 (arXiv v2), eq. 22, which was fitted with delta_c fixed "
        "to 1.673 (pass that delta_c to reproduce the paper)."
    )
    valid_domain: ClassVar[Domain] = _NU_VALID
    calibration_domain: ClassVar[Domain] = Domain(
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
    A: float | None = _p(0.348, "The amplitude; None to normalise the fit.")
    a: float = _p(0.695, "The parameter a (rescales nu^2).", validator=_positive("a"))
    p: float = _p(0.1, "The low-mass slope parameter p (< 0.5).", validator=_below_half)


@attrs.frozen(kw_only=True)
class Manera(SMT, alias="Manera"):
    r"""The Manera, Sheth & Scoccimarro (2010) mass function: Sheth-Tormen, refitted."""

    references: ClassVar[tuple[str, ...]] = (
        (
            "Manera, M., Sheth, R. K., Scoccimarro, R., 2010. MNRAS 402, 589. "
            "https://doi.org/10.1111/j.1365-2966.2009.15921.x"
        ),
    )
    parameter_source: ClassVar[str] = (
        "Manera et al. 2010 (arXiv v2), Table 2: z = 0, 'New ML', linking length 0.2 "
        "(q = 0.709, p = 0.248). hmf 3.x used p = 0.289, from the l_link = 0.15 row."
    )
    valid_domain: ClassVar[Domain] = _NU_VALID
    calibration_domain: ClassVar[Domain] = Domain(
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
    a: float = _p(0.709, "The parameter a (rescales nu^2).", validator=_positive("a"))
    p: float = _p(0.248, "The low-mass slope parameter p (< 0.5).", validator=_below_half)


# ---------------------------------------------------------------------------------
# Jenkins, Warren and the Warren form
# ---------------------------------------------------------------------------------
@attrs.frozen(kw_only=True)
class Jenkins(FittingFunction, alias="Jenkins"):
    r"""The Jenkins et al. (2001) mass function.

    .. math:: f(\sigma) = A\exp\left(-\left|\ln\sigma^{-1} + b\right|^c\right).
    """

    references: ClassVar[tuple[str, ...]] = (
        (
            "Jenkins, A., et al., 2001. MNRAS 321, 372. "
            "https://doi.org/10.1046/j.1365-8711.2001.04029.x"
        ),
    )
    parameter_source: ClassVar[str] = "Jenkins et al. 2001 (arXiv v2, as accepted), eq. 9."
    valid_domain: ClassVar[Domain] = Domain({"sigma": _SIGMA_POSITIVE}, source="sigma > 0.")
    calibration_domain: ClassVar[Domain] = Domain(
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
    A: float = _p(0.315, "The amplitude A.")
    b: float = _p(0.61, "The offset b of ln(1/sigma).")
    c: float = _p(3.8, "The exponent c.")

    def _fsigma(self, x: FitInputs) -> FloatArray:
        return _k.jenkins(x.sigma, self.A, self.b, self.c)


_SIGMA_VALID = Domain({"sigma": _SIGMA_POSITIVE}, source="sigma > 0.")


@attrs.frozen(kw_only=True)
class Warren(FittingFunction, alias="Warren"):
    r"""The Warren et al. (2006) mass function.

    .. math:: f(\sigma) = A\left[\left(\frac{e}{\sigma}\right)^b + c\right]
        \exp\left(-\frac{d}{\sigma^2}\right).

    The paper's (A, a, b, c) are hmf's (A, b, c, d), with e = 1.
    """

    references: ClassVar[tuple[str, ...]] = (
        "Warren, M. S., et al., 2006. ApJ 646, 881. https://doi.org/10.1086/504962",
    )
    parameter_source: ClassVar[str] = "Warren et al. 2006 (arXiv v1), eq. 8."
    valid_domain: ClassVar[Domain] = _SIGMA_VALID
    calibration_domain: ClassVar[Domain] = Domain(
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

    references: ClassVar[tuple[str, ...]] = (
        "Reed, D. S., et al., 2007. MNRAS 374, 2. https://doi.org/10.1111/j.1365-2966.2006.11204.x",
    )
    parameter_source: ClassVar[str] = (
        "Reed et al. 2007 (arXiv v4), eqs. 11-12: c = 1.08, ca = 0.764, p = 0.3, "
        "A = 0.3222 (A' = 0.310 = A / sqrt(c) in eq. 12)."
    )
    requires: ClassVar[frozenset[str]] = frozenset({"delta_c", "n_eff"})
    valid_domain: ClassVar[Domain] = Domain(
        {"sigma": _SIGMA_POSITIVE, "delta_c": (_TINY, None), "n_eff": (np.nextafter(-3, 0), None)},
        source="sigma > 0, delta_c > 0, and n_eff > -3 (the n_eff term diverges at -3).",
    )
    calibration_domain: ClassVar[Domain] = Domain(
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

    references: ClassVar[tuple[str, ...]] = (
        "Peacock, J. A., 2007. MNRAS 379, 1067. https://doi.org/10.1111/j.1365-2966.2007.11979.x",
    )
    parameter_source: ClassVar[str] = "Peacock 2007 (arXiv v2), eq. 9."
    requires: ClassVar[frozenset[str]] = frozenset({"delta_c"})
    valid_domain: ClassVar[Domain] = _NU_VALID
    calibration_domain: ClassVar[Domain] = Domain(
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
    _normalized: ClassVar[bool] = True
    a: float = _p(1.529, "The parameter a.")
    b: float = _p(0.704, "The exponent b.")
    c: float = _p(0.412, "The exponential cut-off c.")

    def _fsigma(self, x: FitInputs) -> FloatArray:
        return _k.peacock(x.nu, self.a, self.b, self.c)


# ---------------------------------------------------------------------------------
# Angulo, Watson, Crocce, Bhattacharya
# ---------------------------------------------------------------------------------
_ANGULO_REF = (
    "Angulo, R. E., et al., 2012. MNRAS 426, 2046. https://doi.org/10.1111/j.1365-2966.2012.21830.x"
)


@attrs.frozen(kw_only=True)
class Angulo(FittingFunction, alias="Angulo"):
    r"""The Angulo et al. (2012) mass function (Millennium-XXL, FoF masses).

    .. math:: f(\sigma) = A\left[\left(\frac{d}{\sigma}\right)^b + 1\right]
        \exp\left(-\frac{c}{\sigma^2}\right).
    """

    references: ClassVar[tuple[str, ...]] = (_ANGULO_REF,)
    parameter_source: ClassVar[str] = (
        "Angulo et al. 2012 (arXiv v2), eq. 2. The paper prints A[d/sigma + 1]^b; the "
        "form used here, A[(d/sigma)^b + 1], is the one that matches its data (and "
        "Warren et al. 2006 to 1-3%)."
    )
    valid_domain: ClassVar[Domain] = _SIGMA_VALID
    calibration_domain: ClassVar[Domain] = Domain(
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
    calibration_domain: ClassVar[Domain] = Angulo.calibration_domain
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = MeasuredMassDefinition(
        kind="self_bound",
        note="Angulo et al. 2012, Sec. 2.2: all SUBFIND self-bound subhaloes.",
    )
    A: float = _p(0.265, "The amplitude A.")
    b: float = _p(1.9, "The exponent b.")
    c: float = _p(1.4, "The exponential cut-off c.")
    d: float = _p(1.675, "The scale d of sigma.")


_WATSON_REF = "Watson, W. A., et al., 2013. MNRAS 433, 1230. https://doi.org/10.1093/mnras/stt791"


@attrs.frozen(kw_only=True)
class Watson_FoF(Warren, alias="Watson_FoF"):
    r"""The Watson et al. (2013) friends-of-friends mass function.

    The :class:`Warren` form, with Watson's parameters.
    """

    references: ClassVar[tuple[str, ...]] = (_WATSON_REF,)
    parameter_source: ClassVar[str] = "Watson et al. 2013 (arXiv v4 = published), eq. 12, Table 2."
    valid_domain: ClassVar[Domain] = _SIGMA_VALID
    calibration_domain: ClassVar[Domain] = Domain(
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

    references: ClassVar[tuple[str, ...]] = (_WATSON_REF,)
    parameter_source: ClassVar[str] = (
        "Watson et al. 2013 (arXiv v4 = published): Table 2 (z = 0 'AHF' and z >= 6 "
        "fits; the Sec. 4.5.2 text swaps alpha_0 and beta_0, Table 2 is followed), "
        "eqs. 14-16 (0 < z < 6) and eqs. 17-19 (Gamma)."
    )
    requires: ClassVar[frozenset[str]] = frozenset({"z", "omega_m_z", "delta_halo"})
    valid_domain: ClassVar[Domain] = Domain(
        {
            "sigma": _SIGMA_POSITIVE,
            "z": (0, None),
            "omega_m_z": (_TINY, None),
            "delta_halo": (1.0, None),
        },
        source=(
            "sigma > 0, z >= 0, Omega_m(z) > 0, and Delta >= 1 (a halo is overdense; "
            "Gamma diverges as Delta -> 0)."
        ),
    )
    calibration_domain: ClassVar[Domain] = Domain(
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
        A, alpha, beta, gamma = self.parameters(x.z, x.omega_m_z)
        gamma_correction = _k.watson_gamma(
            x.sigma, x.delta_halo, x.omega_m_z, self.C_a, self.d_a, self.d_b, self.p, self.q
        )
        return gamma_correction * _k.angulo(x.sigma, A, alpha, gamma, beta)


_Z_VALID = Domain({"sigma": _SIGMA_POSITIVE, "z": (0, None)}, source="sigma > 0 and z >= 0.")


@attrs.frozen(kw_only=True)
class Crocce(FittingFunction, alias="Crocce"):
    r"""The Crocce et al. (2010) mass function (MICE).

    The :class:`Warren` form, with each parameter :math:`X(z) = X_a(1+z)^{-X_b}`, and
    e = 1.
    """

    references: ClassVar[tuple[str, ...]] = (
        (
            "Crocce, M., et al., 2010. MNRAS 403, 1353. "
            "https://doi.org/10.1111/j.1365-2966.2009.16194.x"
        ),
    )
    parameter_source: ClassVar[str] = "Crocce et al. 2010 (arXiv v2), eqs. 20 and 22, Table 2."
    requires: ClassVar[frozenset[str]] = frozenset({"z"})
    valid_domain: ClassVar[Domain] = _Z_VALID
    calibration_domain: ClassVar[Domain] = Domain(
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

    references: ClassVar[tuple[str, ...]] = (
        "Bhattacharya, S., et al., 2011. ApJ 732, 122. https://doi.org/10.1088/0004-637X/732/2/122",
    )
    parameter_source: ClassVar[str] = "Bhattacharya et al. 2011 (arXiv v6), eq. 12 and Table 4."
    requires: ClassVar[frozenset[str]] = frozenset({"z", "delta_c"})
    valid_domain: ClassVar[Domain] = Domain(
        {"sigma": _SIGMA_POSITIVE, "z": (0, None), "delta_c": (_TINY, None)},
        source="sigma > 0, z >= 0 and delta_c > 0.",
    )
    calibration_domain: ClassVar[Domain] = Domain(
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
    A_a: float = _p(0.333, "A(z) = A_a (1+z)^-A_b (unless normed).")
    A_b: float = _p(0.11, "A(z) = A_a (1+z)^-A_b (unless normed).")
    a_a: float = _p(0.788, "a(z) = a_a (1+z)^-a_b.")
    a_b: float = _p(0.01, "a(z) = a_a (1+z)^-a_b.")
    p: float = _p(0.807, "The low-mass slope parameter p.")
    q: float = _p(1.795, "The exponent q (q = 1 gives Sheth-Tormen).", validator=_positive("q"))
    normed: bool = _p(False, "Whether to normalise A so that all mass is in haloes.")

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

_TINKER08_REF = "Tinker, J., et al., 2008. ApJ 688, 709. https://doi.org/10.1086/591439"


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

    references: ClassVar[tuple[str, ...]] = (_TINKER08_REF,)
    parameter_source: ClassVar[str] = (
        "Tinker et al. 2008 (arXiv v1 = published), Table 2 and eqs. 5-8; the "
        "extra digits reproduce Table B3's spline second derivatives."
    )
    requires: ClassVar[frozenset[str]] = frozenset({"z", "delta_halo"})
    valid_domain: ClassVar[Domain] = Domain(
        {
            "sigma": _SIGMA_POSITIVE,
            "z": (0, None),
            "delta_halo": (np.nextafter(75.0, np.inf), 3.0e4),
        },
        source=(
            "sigma > 0, z >= 0, and 75 < Delta <= 3e4: eq. 8 needs Delta > 75, and the "
            "spline of the default parameters keeps all of them > 0 only for "
            "42.7 < Delta < 3.07e4. Parameters that are not > 0 raise in any case."
        ),
    )
    calibration_domain: ClassVar[Domain] = Domain(
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

    def _table(self, name: str) -> FloatArray:
        return np.array([getattr(self, f"{name}_{d}") for d in self.delta_tab], dtype=np.float64)

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
        delta = np.asarray(delta_halo, dtype=np.float64)
        zp1 = 1.0 + np.asarray(z, dtype=np.float64)
        A0, a0, b0, c0 = (
            _k.log10_delta_spline(self.delta_tab, self._table(n), delta) for n in "Aabc"
        )
        A = A0 * zp1**-self.A_exp
        a = a0 * zp1**-self.a_exp
        b = b0 * zp1 ** -_k.tinker08_b_exponent(delta)
        c = c0 * np.ones_like(zp1)
        _check_physical(type(self).__name__, A=A, a=a, b=b, c=c)
        return A, a, b, c

    def _fsigma(self, x: FitInputs) -> FloatArray:
        A, a, b, c = self.parameters(x.delta_halo, x.z)
        return _k.tinker08(x.sigma, A, a, b, c)


@attrs.frozen(kw_only=True)
class Behroozi(Tinker08, alias="Behroozi"):
    r"""The Behroozi, Wechsler & Conroy (2013) mass function.

    :math:`f(\sigma)` is that of :class:`Tinker08` (evaluated at the virial
    overdensity), and :meth:`modify_dndm` applies the empirical high-redshift
    correction of App. G to the mass function:

    .. math::

        n(>M) = \theta(M, z)\, n_{\rm T08}(>M), \qquad
        \theta = 10^{\alpha(z)(M/M_\star)^{\gamma(z)}},

    with :math:`M_\star = 10^{11.5}\,M_\odot` (eqs. G2-G3).
    """

    references: ClassVar[tuple[str, ...]] = (
        (
            "Behroozi, P. S., Wechsler, R. H., Conroy, C., 2013. ApJ 770, 57. "
            "https://doi.org/10.1088/0004-637X/770/1/57"
        ),
        _TINKER08_REF,
    )
    parameter_source: ClassVar[str] = (
        "Behroozi et al. 2013 (arXiv v2), App. G, eqs. G2-G3 (the correction); "
        "f(sigma) parameters from Tinker et al. 2008, Table 2."
    )
    valid_domain: ClassVar[Domain] = Tinker08.valid_domain
    calibration_domain: ClassVar[Domain] = Domain(
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
    modifies_dndm: ClassVar[bool] = True

    def _modify_dndm(
        self,
        m: npt.ArrayLike,
        dndm: npt.ArrayLike,
        *,
        z: npt.ArrayLike,
        ngtm: npt.ArrayLike,
        h: npt.ArrayLike,
    ) -> FloatArray:
        """Apply the App. G correction to the Tinker (2008) mass function (unit-free).

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

    references: ClassVar[tuple[str, ...]] = (
        "Tinker, J., et al., 2010. ApJ 724, 878. https://doi.org/10.1088/0004-637X/724/2/878",
    )
    parameter_source: ClassVar[str] = (
        "Tinker et al. 2010 (arXiv v2), Table 4 and eqs. 9-12 (and the z = 3 cap, Sec. 4)."
    )
    requires: ClassVar[frozenset[str]] = frozenset({"z", "delta_halo", "delta_c"})
    valid_domain: ClassVar[Domain] = Domain(
        {
            "sigma": _SIGMA_POSITIVE,
            "z": (0, None),
            "delta_c": (_TINY, None),
            "delta_halo": (70.0, 3600.0),
        },
        source=(
            "sigma > 0, z >= 0, delta_c > 0, and 70 <= Delta <= 3600: there the spline "
            "of the default Table 4 parameters can be normalised (beta, gamma > 0, "
            "eta > -1/2, eta - phi > -1/2) at every z (it fails below 67.7 and above "
            "3662). Parameters that can not be normalised raise in any case."
        ),
    )
    calibration_domain: ClassVar[Domain] = Domain(
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

    def _table(self, name: str) -> FloatArray:
        return np.array([getattr(self, f"{name}_{d}") for d in self.delta_tab], dtype=np.float64)

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
        delta = np.asarray(delta_halo, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        zp1 = 1.0 + np.minimum(z, self.max_z)
        beta0, gamma0, phi0, eta0 = (
            _k.log10_delta_spline(self.delta_tab, self._table(n), delta)
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
        alpha_tab = np.interp(delta, self.delta_tab, self._table("alpha"))
        alpha = np.where(tabulated, alpha_tab, _k.tinker10_norm(beta, gamma, phi, eta))
        return alpha, beta, gamma, phi, eta

    def _fsigma(self, x: FitInputs) -> FloatArray:
        return _k.tinker10(x.nu, *self.parameters(x.delta_halo, x.z))


# ---------------------------------------------------------------------------------
# Pillepich, Ishiyama, Bocquet and Yung
# ---------------------------------------------------------------------------------
@attrs.frozen(kw_only=True)
class Pillepich(Warren, alias="Pillepich"):
    r"""The Pillepich, Porciani & Hahn (2010) mass function: the :class:`Warren` form."""

    references: ClassVar[tuple[str, ...]] = (
        (
            "Pillepich, A., Porciani, C., Hahn, O., 2010. MNRAS 402, 191. "
            "https://doi.org/10.1111/j.1365-2966.2009.15914.x"
        ),
    )
    parameter_source: ClassVar[str] = (
        "Pillepich et al. 2010 (arXiv v3), Sec. 3.1: the Gaussian (f_NL = 0) runs, in "
        "Warren notation (A, a, b, c) = (0.6853, 1.868, 0.3324, 1.2266)."
    )
    valid_domain: ClassVar[Domain] = _SIGMA_VALID
    calibration_domain: ClassVar[Domain] = Domain(
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

    references: ClassVar[tuple[str, ...]] = (
        "Ishiyama, T., et al., 2015. PASJ 67, 61. https://doi.org/10.1093/pasj/psv021",
    )
    parameter_source: ClassVar[str] = "Ishiyama et al. 2015 (arXiv v3), eq. 2 and Table 3 (z = 0)."
    valid_domain: ClassVar[Domain] = _SIGMA_VALID
    calibration_domain: ClassVar[Domain] = Domain(
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
    A: float = _p(0.193, "The amplitude A.")
    b: float = _p(1.550, "The exponent b of e/sigma (the paper's C).")
    c: float = _p(1.0, "The constant c added to (e/sigma)^b.")
    d: float = _p(1.186, "The exponential cut-off d (the paper's D).")
    e: float = _p(2.184, "The scale e of sigma (the paper's B).")


_BOCQUET_REF = "Bocquet, S., et al., 2016. MNRAS 456, 2361. https://doi.org/10.1093/mnras/stv2657"
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

    references: ClassVar[tuple[str, ...]] = (_BOCQUET_REF,)
    parameter_source: ClassVar[str] = (
        "Bocquet et al. 2016 (arXiv v3 = published), Table 2; paper (a, b, c) -> hmf (b, e, d)."
    )
    requires: ClassVar[frozenset[str]] = frozenset({"z"})
    valid_domain: ClassVar[Domain] = _Z_VALID
    calibration_domain: ClassVar[Domain] = _BOCQUET_DOMAIN
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = _so_mean(
        200, note="Bocquet et al. 2016, Sec. 2.2: SO masses about SUBFIND potential minima."
    )
    A: float = _p(0.175, "The amplitude A at z = 0.")
    b: float = _p(1.53, "The exponent b of e/sigma at z = 0 (the paper's a).")
    d: float = _p(1.19, "The exponential cut-off d at z = 0 (the paper's c).")
    e: float = _p(2.55, "The scale e of sigma at z = 0 (the paper's b).")
    A_z: float = _p(-0.012, "The redshift exponent of A.")
    b_z: float = _p(-0.04, "The redshift exponent of b.")
    d_z: float = _p(-0.021, "The redshift exponent of d.")
    e_z: float = _p(-0.194, "The redshift exponent of e.")

    def parameters(self, z: npt.ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """The parameters (A, b, d, e) at redshift ``z``."""
        zp1 = 1.0 + np.asarray(z, dtype=np.float64)
        return (
            self.A * zp1**self.A_z,
            self.b * zp1**self.b_z,
            self.d * zp1**self.d_z,
            self.e * zp1**self.e_z,
        )

    @unit_boundary(m=Msun_h)
    def mass_ratio_to_200m(
        self, m: Any, *, z: npt.ArrayLike, omega_m0: npt.ArrayLike, h: npt.ArrayLike
    ) -> FloatArray:
        r"""The ratio :math:`M_\Delta/M_{200m}` of the fit's mass to M200m.

        Bocquet et al. build their 200c and 500c mass functions from
        :math:`dn/dM_\Delta = f(\sigma)(\bar\rho_m/M_\Delta)(d\ln\sigma^{-1}/dM_\Delta)
        (M_\Delta/M_{200m})` (eq. 5), with this ratio: 1 for the 200m fits, eq. A2
        for 200c and eq. 6 for 500c. Applying it (and evaluating sigma at M200m) is
        left to the mass-function stage (issue #392), which calls the unit-free
        :meth:`_mass_ratio_to_200m`.

        Parameters
        ----------
        m
            Halo mass in the fit's definition, a Quantity.
        z
            Redshift.
        omega_m0
            The matter density parameter today.
        h
            The dimensionless Hubble parameter (the fits take ln(M / Msun)).

        Returns
        -------
        numpy.ndarray
            The (dimensionless) ratio.
        """
        return self._mass_ratio_to_200m(m, z=z, omega_m0=omega_m0, h=h)

    def _mass_ratio_to_200m(
        self, m: npt.ArrayLike, *, z: npt.ArrayLike, omega_m0: npt.ArrayLike, h: npt.ArrayLike
    ) -> FloatArray:
        """:meth:`mass_ratio_to_200m`, with ``m`` a plain array in Msun/h (1 here)."""
        return np.ones(np.broadcast(np.asarray(m), np.asarray(z)).shape)

    def _fsigma(self, x: FitInputs) -> FloatArray:
        A, b, d, e = self.parameters(x.z)
        return _k.warren(x.sigma, A=A, b=b, c=1.0, d=d, e=e)


@attrs.frozen(kw_only=True)
class Bocquet200mHydro(Bocquet200mDMOnly, alias="Bocquet200mHydro"):
    r"""The Bocquet et al. (2016) mass function for M200m, with hydrodynamics."""

    valid_domain: ClassVar[Domain] = _Z_VALID
    calibration_domain: ClassVar[Domain] = _BOCQUET_DOMAIN
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = (
        Bocquet200mDMOnly.measured_mass_definition
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
    "M200m and the ratio mass_ratio_to_200m (eq. 5; App. A), which this model does "
    "not apply itself (issue #392)."
)


@attrs.frozen(kw_only=True)
class Bocquet200cDMOnly(Bocquet200mDMOnly, alias="Bocquet200cDMOnly"):
    r"""The Bocquet et al. (2016) mass function for M200c, dark matter only.

    :meth:`fsigma` is the eq. 3 form with the 200c parameters only.
    :meth:`mass_ratio_to_200m` gives :math:`M_{200c}/M_{200m}` (eq. A2), which the
    mass function needs as well (eq. 5); hmf 3.x multiplied :math:`f(\sigma)` by it.
    """

    valid_domain: ClassVar[Domain] = _Z_VALID
    calibration_domain: ClassVar[Domain] = _BOCQUET_DOMAIN
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = _so_crit(
        200, note=_BOCQUET_200C_NOTE
    )
    A: float = _p(0.222, "The amplitude A at z = 0.")
    b: float = _p(1.71, "The exponent b of e/sigma at z = 0 (the paper's a).")
    d: float = _p(1.46, "The exponential cut-off d at z = 0 (the paper's c).")
    e: float = _p(2.24, "The scale e of sigma at z = 0 (the paper's b).")
    A_z: float = _p(0.269, "The redshift exponent of A.")
    b_z: float = _p(0.321, "The redshift exponent of b.")
    d_z: float = _p(-0.153, "The redshift exponent of d.")
    e_z: float = _p(-0.621, "The redshift exponent of e.")

    def _mass_ratio_to_200m(
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
    calibration_domain: ClassVar[Domain] = _BOCQUET_DOMAIN
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = (
        Bocquet200cDMOnly.measured_mass_definition
    )
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
    mass function needs as well (eq. 5); hmf 3.x multiplied :math:`f(\sigma)` by it.
    """

    valid_domain: ClassVar[Domain] = _Z_VALID
    calibration_domain: ClassVar[Domain] = _BOCQUET_DOMAIN
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = _so_crit(
        500, note=_BOCQUET_200C_NOTE.replace("App. A", "eq. 6")
    )
    A: float = _p(0.241, "The amplitude A at z = 0.")
    b: float = _p(2.18, "The exponent b of e/sigma at z = 0 (the paper's a).")
    d: float = _p(2.02, "The exponential cut-off d at z = 0 (the paper's c).")
    e: float = _p(2.35, "The scale e of sigma at z = 0 (the paper's b).")
    A_z: float = _p(0.37, "The redshift exponent of A.")
    b_z: float = _p(0.251, "The redshift exponent of b.")
    d_z: float = _p(-0.31, "The redshift exponent of d.")
    e_z: float = _p(-0.698, "The redshift exponent of e.")

    def _mass_ratio_to_200m(
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
    calibration_domain: ClassVar[Domain] = _BOCQUET_DOMAIN
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = (
        Bocquet500cDMOnly.measured_mass_definition
    )
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
    """The default of a Yung24 coefficient: from the table chosen by ``units``."""
    # An invalid `units` falls back to "h" here, and is then rejected by its validator.
    return attrs.Factory(
        lambda self: _YUNG24_TABLES.get(self.units, _YUNG24_TABLES["h"])[name], takes_self=True
    )


def _yung24_coefficient(name: str) -> Any:
    chi, power = name.split("_")
    return field(
        default=_yung24_default(name),
        doc=f"The z^{power} coefficient of {chi}(z) (default: from the table for `units`).",
    )


@attrs.frozen(kw_only=True)
class Yung24(FittingFunction, alias="Yung24"):
    r"""The Yung et al. (2024) mass function (GUREFT), for :math:`6 \le z \le 19`.

    .. math:: f(\sigma) = A(z)\left[\left(\frac{\sigma}{b(z)}\right)^{-a(z)} + 1\right]
        \exp\left(-\frac{c(z)}{\sigma^2}\right),

    with :math:`\chi(z) = \chi_0 + \chi_1 z + \chi_2 z^2` for
    :math:`\chi \in \{A, a, b, c\}` (eq. A2). ``units="h"`` takes the coefficients of
    Table A1 (masses in Msun/h, hmf's convention), ``units="physical"`` those of
    Table A2 (masses in Msun); coefficients given explicitly override the table.
    The paper's quadratics are fitted for z = 6-19 only, and are not used outside it.
    """

    references: ClassVar[tuple[str, ...]] = (
        "Yung, L. Y. A., et al., 2024. MNRAS 530, 4868. https://doi.org/10.1093/mnras/stae1188",
    )
    parameter_source: ClassVar[str] = (
        "Yung et al. 2024 (arXiv v3), App. A: Table A1 (units='h') and Table A2 (units='physical')."
    )
    requires: ClassVar[frozenset[str]] = frozenset({"z"})
    valid_domain: ClassVar[Domain] = Domain(
        {"sigma": _SIGMA_POSITIVE, "z": (6, 19)},
        source=(
            "sigma > 0 and 6 <= z <= 19: the redshift quadratics of App. A are fitted "
            "over z = 6-19 only (hmf 3.x raised outside it too)."
        ),
    )
    calibration_domain: ClassVar[Domain] = Domain(
        {"m": [1e6, 1e13] * Msun_h, "z": (6, 19)},
        source=(
            "Yung et al. 2024, App. A: 'fitted to gureft+MultiDark HMFs between z = 6 "
            "to 19', over 6 < log10(M_vir / (Msun/h)) < 13 (for units='h'; for "
            "units='physical' Fig. A1 shows 10^6-10^13 Msun, i.e. ~10^5.83-10^12.83 "
            "Msun/h for their h = 0.678, which this domain does not adjust for). "
            "Planck cosmology (Omega_m = 0.307, sigma_8 = 0.829; Sec. 2)."
        ),
    )
    measured_mass_definition: ClassVar[MeasuredMassDefinition] = _so_virial(
        note="Yung et al. 2024, Sec. 2: ROCKSTAR, Bryan & Norman virial; includes subhaloes."
    )

    units: Literal["h", "physical"] = field(
        default="h",
        validator=attrs.validators.in_(("h", "physical")),
        doc="Which table of coefficients to use: 'h' (Table A1) or 'physical' (Table A2).",
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

    def parameters(self, z: npt.ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """The parameters (A, a, b, c) at redshift ``z``."""
        z = np.asarray(z, dtype=np.float64)
        return (
            self.A_0 + self.A_1 * z + self.A_2 * z**2,
            self.a_0 + self.a_1 * z + self.a_2 * z**2,
            self.b_0 + self.b_1 * z + self.b_2 * z**2,
            self.c_0 + self.c_1 * z + self.c_2 * z**2,
        )

    def _fsigma(self, x: FitInputs) -> FloatArray:
        A, a, b, c = self.parameters(x.z)
        return _k.tinker08(x.sigma, A, a, b, c)
