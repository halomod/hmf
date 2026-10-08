r"""Growth-factor models: the :class:`GrowthModel` kind and its models.

A growth model turns a cosmology into the linear growth factor D(z) of every matter
species, normalised to D(0) = 1, and the growth rate
:math:`f = d\ln D / d\ln a`. Its :meth:`~GrowthModel.solve` returns a
:class:`GrowthSolution`, which evaluates both on plain arrays of redshift. The
:class:`~hmf.core.growth.Growth` stage wraps that in the public API.

Most models tabulate D on a grid in ln a (from ``a_min`` to 1, with spacing
``dln_a``) and interpolate it with cubic splines:

* :class:`ODEGrowth` (the default) solves the growth equation and is exact for any
  FLRW background: curvature, radiation, massive neutrinos (as a smooth background)
  and any dark-energy equation of state w(a);
* :class:`IntegralGrowth` (Heath 1977), :class:`Eisenstein97Growth` and
  :class:`Heath77Growth` are exact for a cosmological constant and negligible
  radiation (in any, flat, and Lambda-free universes respectively);
* :class:`GenMFGrowth` and :class:`Carroll92Growth` are approximations of the same
  family;
* :class:`CambGrowth` and :class:`ClassGrowth` take the growth of each species at
  k = 0.01/Mpc from a Boltzmann run, which a :class:`~hmf.core.growth.Growth` stage
  shares with its transfer stage when both use the same code.

The models without massive neutrinos give the same growth for both species: on the
scales below the free-streaming length that matter for haloes, the neutrinos are a
smooth background, and :math:`\delta_{\rm tot} \propto \delta_{\rm cb}`.
"""

from __future__ import annotations

import abc
import math
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar

import attrs
import numpy as np
import numpy.typing as npt
from astropy import cosmology as ac
from astropy.cosmology import FLRW

from . import _references as refs
from ._arrays import dimensionless_floats
from ._boltzmann import BoltzmannRun
from ._cosmology_models import _BoltzmannBacked, _CosmologyModel
from ._fields import field
from ._kernels import growth as kg
from ._species import MATTER_SPECIES, check_species, omega_m
from ._validators import check_finite_positive, check_in_range, check_table, less_than, positive
from .accuracy import KAccuracy
from .cache import DiskCache
from .domain import Domain
from .model import Model
from .transfer_models import CAMB, CLASS

__all__ = [
    "CambGrowth",
    "Carroll92Growth",
    "ClassGrowth",
    "Eisenstein97Growth",
    "FromArray",
    "FromFile",
    "GenMFGrowth",
    "GrowthModel",
    "GrowthSolution",
    "Heath77Growth",
    "IntegralGrowth",
    "ODEGrowth",
    "background",
]

Array = npt.NDArray[np.float64]

#: The wavenumber accuracy of a model's own Boltzmann run, by default.
_DEFAULT_K_ACCURACY = KAccuracy()


@attrs.frozen(eq=False)
class GrowthSolution:
    """The growth factor and rate of every matter species, from one model evaluation.

    The result of :meth:`GrowthModel.solve`, a read-only data holder: plain arrays of
    redshift in, plain arrays out. It is valid for ``0 <= z <= z_max`` and does not
    check its input: outside that range its methods return NaN. Library code calls
    the checked kernel-level entry points,
    :meth:`Growth.growth_factor_kernel <hmf.core.growth.Growth.growth_factor_kernel>`
    and :meth:`Growth.growth_rate_kernel <hmf.core.growth.Growth.growth_rate_kernel>`.
    """

    #: The tabulated growth of each species.
    tables: Mapping[str, kg.GrowthTable]
    #: The largest redshift it covers.
    z_max: float
    #: The Boltzmann run it came from, if any.
    run: BoltzmannRun | None = None

    def growth_factor(self, z: Array, species: str = "cb") -> Array:
        """D(z)/D(0) of ``species`` (NaN outside ``0 <= z <= z_max``)."""
        return self.tables[check_species(species)].growth_factor(z)

    def growth_rate(self, z: Array, species: str = "cb") -> Array:
        """The growth rate d ln D / d ln a of ``species`` (NaN outside the table)."""
        return self.tables[check_species(species)].growth_rate(z)


def _same_for_all(table: kg.GrowthTable, z_max: float) -> GrowthSolution:
    return GrowthSolution(MappingProxyType(dict.fromkeys(MATTER_SPECIES, table)), z_max=z_max)


def background(cosmology: FLRW, ln_a: Array) -> tuple[Array, Array, Array]:
    r"""The background quantities of the growth equation, from an astropy cosmology.

    Parameters
    ----------
    cosmology
        The cosmology.
    ln_a
        ln a, at which to evaluate them.

    Returns
    -------
    efunc : numpy.ndarray
        :math:`E = H/H_0` (astropy's ``efunc``).
    dln_e_dln_a : numpy.ndarray
        :math:`d\ln E/d\ln a`, written down from astropy's definition of E. The dark
        energy density scales as :math:`\exp(-3\int(1+w)\,d\ln a)`, so its
        logarithmic derivative is :math:`-3(1+w(a))`. With massive neutrinos the
        radiation term is :math:`\Omega_\gamma(1 + N(z))a^{-4}` with astropy's
        ``nu_relative_density`` N, whose derivative is taken by central finite
        difference in ln(1+z).
    omega_m_a : numpy.ndarray
        :math:`\Omega_m(a) = \Omega_{m,0} a^{-3}/E^2` of the matter that clusters,
        CDM + baryons (:func:`~hmf.core._species.omega_m` of species ``"cb"``).
    """
    a = np.exp(ln_a)
    z = np.expm1(-ln_a)
    c = cosmology
    inv_e2 = c.inv_efunc(z) ** 2
    if c.has_massive_nu:
        eps = 1e-4
        ln_zp1 = -ln_a
        dn_dln_a = -(
            c.nu_relative_density(np.expm1(ln_zp1 + eps))
            - c.nu_relative_density(np.expm1(ln_zp1 - eps))
        ) / (2 * eps)
        rad = c.Ogamma0 * (-4 * (1 + c.nu_relative_density(z)) + dn_dln_a)
    else:
        rad = -4 * (c.Ogamma0 + c.Onu0)
    d_e2 = (
        rad * a**-4
        - 3 * c.Om0 * a**-3
        - 2 * c.Ok0 * a**-2
        - 3 * (1 + c.w(z)) * c.Ode0 * c.de_density_scale(z)
    )
    efunc = np.sqrt(1 / inv_e2)
    return efunc, 0.5 * d_e2 * inv_e2, omega_m(c, z, "cb")


@attrs.frozen(kw_only=True)
class GrowthModel(_CosmologyModel, Model, kind=True):
    """The kind of growth-factor models.

    Subclasses implement :meth:`solve`, and may check that they apply to a cosmology
    in :meth:`check_cosmology`.
    """

    #: Where the model can be evaluated: z >= 0. The :class:`~hmf.core.growth.Growth`
    #: stage checks z against the model's class's domain (so a model can narrow it),
    #: and also against the redshifts its solution covers (up to
    #: :attr:`GrowthSolution.z_max`, e.g. a table's largest redshift): growth is
    #: never extrapolated.
    valid_domain: ClassVar[Domain] = Domain({"z": (0, None)}, source="z >= 0.")

    @abc.abstractmethod
    def solve(
        self,
        cosmology: FLRW,
        *,
        run: BoltzmannRun | None = None,
        disk_cache: DiskCache | None = None,
        k_accuracy: KAccuracy = _DEFAULT_K_ACCURACY,
    ) -> GrowthSolution:
        """Compute the growth of every species for a cosmology.

        Parameters
        ----------
        cosmology
            The cosmology.
        run
            A run of this model's Boltzmann code (from a transfer stage) to take the
            growth from, if it covers the model's redshifts. Ignored by models without
            a backend.
        disk_cache
            Where to cache Boltzmann-code output on disk, if anywhere.
        k_accuracy
            The wavenumber accuracy of a Boltzmann run the model makes itself (when
            ``run`` is not given or does not cover its redshifts): it sets the code's
            precision, as for :class:`~hmf.core.transfer.Transfer`. Ignored by models
            without a backend.

        Returns
        -------
        GrowthSolution
        """


@attrs.frozen(kw_only=True)
class _Gridded(GrowthModel, abstract=True):
    """A growth model tabulated on a grid in ln a."""

    a_min: float = field(
        default=1e-8,
        converter=float,
        validator=[positive, less_than(1.0)],
        doc="The smallest scale factor of the grid; z_max = 1/a_min - 1.",
    )
    dln_a: float = field(
        default=0.01,
        converter=float,
        validator=positive,
        doc="The spacing of the grid in ln a.",
    )

    @abc.abstractmethod
    def _growth(self, cosmology: FLRW, ln_a: Array) -> tuple[Array, Array | None]:
        """D (unnormalised) and f on the grid (f None: from the spline of ln D)."""

    def solve(
        self,
        cosmology: FLRW,
        *,
        run: BoltzmannRun | None = None,
        disk_cache: DiskCache | None = None,
        k_accuracy: KAccuracy = _DEFAULT_K_ACCURACY,
    ) -> GrowthSolution:
        """Compute the growth (see :meth:`GrowthModel.solve`)."""
        self.check_cosmology(cosmology)
        ln_a = kg.ln_a_grid(self.a_min, self.dln_a)
        d, f = self._growth(cosmology, ln_a)
        table = kg.tabulate_growth(ln_a, d, f)
        return _same_for_all(table, table.z_max)


def _radiation_a4(cosmology: FLRW, z: float) -> float:
    """The radiation density times a^4 at z, in units of today's critical density."""
    c = cosmology
    if c.has_massive_nu:
        return float(c.Ogamma0 * (1 + c.nu_relative_density(z)))
    return float(c.Ogamma0 + c.Onu0)


@attrs.frozen(kw_only=True)
class ODEGrowth(_Gridded, alias="ODE"):
    r"""The growth factor from the linear growth equation, for any FLRW cosmology.

    Solves, in ln a,

    .. math:: D'' + \left(2 + \frac{d\ln E}{d\ln a}\right) D'
              - \frac{3}{2}\Omega_m(a) D = 0

    (e.g. Peebles 1980), with E from astropy (so curvature, radiation, massive
    neutrinos and w(a) are all included; see :func:`background`), by fourth-order
    Runge-Kutta with step ``dln_a``. It starts at ``a_min`` on the growing mode of a
    matter + radiation universe (Meszaros), :math:`D \propto 1 + 3a/(2a_{\rm eq})`.
    The growth rate comes from the solution's own :math:`D'/D`.
    """

    references: ClassVar[tuple[str, ...]] = (refs.PEEBLES80,)

    def _growth(self, cosmology: FLRW, ln_a: Array) -> tuple[Array, Array | None]:
        # The RK4 midpoints: a grid of half the spacing.
        half = np.empty(2 * ln_a.size - 1)
        half[::2] = ln_a
        half[1::2] = 0.5 * (ln_a[1:] + ln_a[:-1])
        _, dln_e, omega_m_a = background(cosmology, half)
        a0 = math.exp(ln_a[0])
        a_eq = _radiation_a4(cosmology, 1 / a0 - 1) / cosmology.Om0
        y = a0 / a_eq if a_eq > 0 else math.inf
        f0 = 1.0 if math.isinf(y) else 1.5 * y / (1 + 1.5 * y)
        d0 = a0 if math.isinf(y) else (1 + 1.5 * y) * a_eq / 1.5
        return kg.solve_growth_ode(half, dln_e, omega_m_a, d0, f0)


def _check_lambda(cosmology: FLRW, name: str) -> None:
    # A model that does not apply to the cosmology is a configuration error (see
    # "Errors" in hmf.core.domain), so a ValueError despite the isinstance check.
    if not isinstance(cosmology, ac.LambdaCDM):
        raise ValueError(  # noqa: TRY004
            f"{name} is exact only for a cosmological constant (w = -1), not "
            f"{type(cosmology).__name__}. Use ODEGrowth (or a Boltzmann code)."
        )


@attrs.frozen(kw_only=True)
class IntegralGrowth(_Gridded, alias="Integral"):
    r"""The integral form of the growth factor (Heath 1977).

    .. math:: D^+(a) = \frac52 \Omega_{m,0} E(a) \int_0^a \frac{da'}{(a'E(a'))^3},

    exact for a cosmological constant (any curvature) and negligible radiation. E is
    astropy's, including any radiation, so the result departs from the exact growth
    at the level of the radiation density (about 3e-4 at z = 1 for Planck18; use
    :class:`ODEGrowth` there). Only for LambdaCDM cosmologies. The growth rate is the
    exact derivative of the formula (see
    :func:`~hmf.core._kernels.growth.integral_growth`).
    """

    references: ClassVar[tuple[str, ...]] = (refs.HEATH77,)

    def check_cosmology(self, cosmology: FLRW) -> None:
        """Raise unless the cosmology has a cosmological constant."""
        _check_lambda(cosmology, type(self).__name__)

    def _growth(self, cosmology: FLRW, ln_a: Array) -> tuple[Array, Array | None]:
        efunc, dln_e, _ = background(cosmology, ln_a)
        return kg.integral_growth(ln_a, efunc, dln_e, cosmology.Om0)


@attrs.frozen(kw_only=True)
class Eisenstein97Growth(_Gridded, alias="Eisenstein97"):
    r"""The closed form of the integral growth for a flat universe (Eisenstein 1997).

    Eisenstein (1997), Eqs. 8-10: flat, with a cosmological constant
    :math:`1 - \Omega_{m,0}` and no radiation (the cosmology's radiation is ignored).
    """

    references: ClassVar[tuple[str, ...]] = (refs.EH97,)

    def check_cosmology(self, cosmology: FLRW) -> None:
        """Raise unless the cosmology is flat with a cosmological constant."""
        _check_lambda(cosmology, type(self).__name__)
        if not cosmology.is_flat:
            raise ValueError("Eisenstein97Growth only applies to flat cosmologies.")

    def _growth(self, cosmology: FLRW, ln_a: Array) -> tuple[Array, Array | None]:
        return kg.eisenstein97_growth(ln_a, float(cosmology.Om0))


@attrs.frozen(kw_only=True)
class Heath77Growth(_Gridded, alias="Heath77"):
    r"""The closed form of the integral growth without Lambda (Heath 1977, Eq. 13).

    For an open or closed universe with no dark energy and no radiation (the
    cosmology's radiation is ignored; the curvature is :math:`1 - \Omega_{m,0}`).
    """

    references: ClassVar[tuple[str, ...]] = (refs.HEATH77,)

    def check_cosmology(self, cosmology: FLRW) -> None:
        """Raise unless the cosmology has no dark energy."""
        if cosmology.Ode0 != 0:
            raise ValueError("Heath77Growth only applies to cosmologies without dark energy.")

    def _growth(self, cosmology: FLRW, ln_a: Array) -> tuple[Array, Array | None]:
        return kg.heath77_growth(ln_a, float(cosmology.Om0))


@attrs.frozen(kw_only=True)
class GenMFGrowth(_Gridded, alias="GenMF"):
    """The growth factor of the ``genmf`` code (Reed et al. 2007).

    Exact for a flat universe with a cosmological constant, or an open universe
    without one, and no radiation. For other LambdaCDM cosmologies with Lambda and
    non-negative curvature it uses the flat formula, as an approximation. The growth
    rate is the derivative of the spline of ln D.
    """

    references: ClassVar[tuple[str, ...]] = (refs.REED07,)

    def check_cosmology(self, cosmology: FLRW) -> None:
        """Raise unless the cosmology is a flat or open LambdaCDM."""
        _check_lambda(cosmology, type(self).__name__)
        if cosmology.Ok0 < -1e-12:
            raise ValueError("GenMFGrowth only applies to flat or open cosmologies.")

    def _growth(self, cosmology: FLRW, ln_a: Array) -> tuple[Array, Array | None]:
        return kg.genmf_growth(ln_a, float(cosmology.Om0), float(cosmology.Ode0)), None


@attrs.frozen(kw_only=True)
class Carroll92Growth(_Gridded, alias="Carroll92"):
    """The approximation of Carroll, Press & Turner (1992), after Lahav et al. (1991).

    Uses astropy's density parameters at each redshift. Accurate to about 1% for
    0.1 < Omega_m < 1 with a cosmological constant.
    """

    references: ClassVar[tuple[str, ...]] = (refs.CARROLL92, refs.LAHAV91)

    def _growth(self, cosmology: FLRW, ln_a: Array) -> tuple[Array, Array | None]:
        z = np.expm1(-ln_a)
        return kg.carroll92_growth(np.exp(ln_a), cosmology.Om(z), cosmology.Ode(z))


@attrs.frozen(kw_only=True)
class FromArray(GrowthModel, alias="FromArray"):
    """A growth factor given as arrays of (z, D), the same for both species.

    D is interpolated with a cubic spline of ln D in ln a and normalised to 1 at
    z = 0, which must be in the table. The growth rate is the spline's derivative.
    """

    z: tuple[float, ...] = field(
        converter=dimensionless_floats,
        doc="Redshifts (at least 4), including 0.",
    )
    d: tuple[float, ...] = field(
        converter=dimensionless_floats,
        doc="The growth factor at z, in any normalisation.",
    )

    @d.validator
    def _check(
        self, attribute: attrs.Attribute[tuple[float, ...]], value: tuple[float, ...]
    ) -> None:
        z, d = check_table({"z": self.z, "d": value}, where="FromArray", min_size=4)
        check_in_range("z", z, where="FromArray", low=0.0)
        if 0.0 not in z:
            raise ValueError("FromArray: the redshifts must include 0.")
        check_finite_positive("d", d, where="FromArray")

    def _arrays(self) -> tuple[Array, Array]:
        return np.array(self.z), np.array(self.d)

    def solve(
        self,
        cosmology: FLRW,
        *,
        run: BoltzmannRun | None = None,
        disk_cache: DiskCache | None = None,
        k_accuracy: KAccuracy = _DEFAULT_K_ACCURACY,
    ) -> GrowthSolution:
        """Compute the growth (see :meth:`GrowthModel.solve`)."""
        z, d = self._arrays()
        order = np.argsort(z)[::-1]
        table = kg.tabulate_growth(-np.log1p(z[order]), d[order])
        return _same_for_all(table, float(z.max()))


@attrs.frozen(kw_only=True)
class FromFile(GrowthModel, alias="FromFile"):
    """A growth factor read from a two-column text file of (z, D).

    As :class:`FromArray`. The file is read when the model is solved; its content is
    not part of the model's value.
    """

    fname: Path = field(converter=Path, doc="The file to read.")

    def solve(
        self,
        cosmology: FLRW,
        *,
        run: BoltzmannRun | None = None,
        disk_cache: DiskCache | None = None,
        k_accuracy: KAccuracy = _DEFAULT_K_ACCURACY,
    ) -> GrowthSolution:
        """Compute the growth (see :meth:`GrowthModel.solve`)."""
        data = np.atleast_2d(np.genfromtxt(self.fname))
        z, d = dimensionless_floats(data[:, 0]), dimensionless_floats(data[:, 1])
        return FromArray(z=z, d=d).solve(cosmology)


@attrs.frozen(kw_only=True)
class _BoltzmannGrowth(_BoltzmannBacked, GrowthModel, abstract=True):
    """The growth of each species at k = 0.01/Mpc, from a Boltzmann run."""

    z_max: float = field(
        default=20.0,
        converter=float,
        validator=positive,
        doc="The largest redshift of the growth factor.",
    )

    @abc.abstractmethod
    def _own_run(
        self, cosmology: FLRW, disk_cache: DiskCache | None, k_accuracy: KAccuracy
    ) -> BoltzmannRun:
        """The run made when no transfer stage shares one, at ``k_accuracy``."""

    def solve(
        self,
        cosmology: FLRW,
        *,
        run: BoltzmannRun | None = None,
        disk_cache: DiskCache | None = None,
        k_accuracy: KAccuracy = _DEFAULT_K_ACCURACY,
    ) -> GrowthSolution:
        """Compute the growth (see :meth:`GrowthModel.solve`)."""
        self.check_cosmology(cosmology)
        usable = (
            run is not None
            and run.backend == self.backend
            and run.growth
            and float(run.growth_z[-1]) >= self.z_max * (1 - 1e-12)
        )
        if not usable:
            run = self._own_run(cosmology, disk_cache, k_accuracy)
        assert run is not None
        ln_a = -np.log1p(run.growth_z[::-1])
        tables = {s: kg.tabulate_growth(ln_a, run.growth[s][::-1]) for s in MATTER_SPECIES}
        return GrowthSolution(MappingProxyType(tables), z_max=self.z_max, run=run)


@attrs.frozen(kw_only=True)
class CambGrowth(_BoltzmannGrowth, alias="CAMB"):
    """The growth of each species from CAMB, at k = 0.01/Mpc.

    ``"cb"`` is the growth of CAMB's ``delta_nonu`` and ``"tot"`` of ``delta_tot``.
    With massive neutrinos the growth is scale-dependent: at this scale the
    neutrinos partly cluster, so it differs from :class:`ODEGrowth`, which is the
    small-scale limit (by about 2% at z = 10 for a 0.3 eV neutrino).

    With a :class:`~hmf.core.growth.Growth` stage whose transfer model is
    :class:`~hmf.core.transfer_models.CAMB`, the growth comes from the transfer's
    run. Otherwise it makes the run of ``CAMB(z_max=z_max)`` at the growth stage's
    ``k_accuracy`` (which a default CAMB transfer stage with the same accuracy
    shares).
    """

    backend: ClassVar[str | None] = "camb"
    references: ClassVar[tuple[str, ...]] = (refs.CAMB,)

    def _own_run(
        self, cosmology: FLRW, disk_cache: DiskCache | None, k_accuracy: KAccuracy
    ) -> BoltzmannRun:
        return CAMB(z_max=self.z_max).run(cosmology, k_accuracy, disk_cache=disk_cache)


@attrs.frozen(kw_only=True)
class ClassGrowth(_BoltzmannGrowth, alias="CLASS"):
    r"""The growth of each species from CLASS, at k = 0.01/Mpc.

    Needs ``classy``. :math:`D(z) = \sqrt{P(k_{\rm ref}, z) / P(k_{\rm ref}, 0)}`
    with CLASS's linear power spectrum (of CDM + baryons for ``"cb"``, of total
    matter for ``"tot"``), which is how CLASS defines its scale-dependent growth
    factor. The growth rate is the derivative of the spline of ln D in ln a through
    CLASS's redshifts. See :class:`CambGrowth` for massive neutrinos and sharing.
    """

    backend: ClassVar[str | None] = "class"
    references: ClassVar[tuple[str, ...]] = (refs.CLASS_I, refs.CLASS_II)

    def _own_run(
        self, cosmology: FLRW, disk_cache: DiskCache | None, k_accuracy: KAccuracy
    ) -> BoltzmannRun:
        return CLASS(z_max=self.z_max).run(cosmology, k_accuracy, disk_cache=disk_cache)
