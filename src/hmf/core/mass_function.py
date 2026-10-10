r"""The :class:`MassFunction` stage: the halo mass function as a function of (m, z).

The stage tree
--------------
A mass function is built from three stages and a fit, each stage holding the stages
it is computed from as fields (the *stage tree*)::

    MassFunction
    ├── linear_power: LinearPower       (sigma_8, the amplitude A, D(z))
    │   ├── transfer: Transfer          (the shape of the power, one Boltzmann run)
    │   └── growth: Growth              (D(z))
    ├── variance: MassVariance          (sigma(m) of the power before normalisation)
    └── fit: FittingFunction            (f(sigma))

with the critical overdensity ``delta_c`` and the calibration-domain policy
``domain_policy`` as fields. :meth:`MassFunction.build` builds the whole tree from
keyword arguments, with hmf 3.x-like defaults.

The variance is a field of its own, not built inside the linear-power stage, because
it does not depend on sigma_8 or z: its power is :attr:`LinearPower.power_source
<hmf.core.linear_power.LinearPower.power_source>`, the power before it is normalised,
and sigma_8 and z only multiply sigma(m). So
``mf.evolve(linear_power=mf.linear_power.evolve(sigma_8=0.75))`` keeps ``mf.variance``,
the same object, with every value of sigma(m) it has already computed. Changing z, the
fit, ``delta_c`` or the policy keeps the variance and the linear-power stage too:
sigma(m) is not computed again, and no Boltzmann code runs again. A stage is immutable,
and the constructor validates every combination of fields, so a cached result can
never be stale (see :mod:`hmf.core.stage`).

Quantities
----------
Every quantity is a function of the mass m and the redshift z, which broadcast like
numpy: ``mf.dndm(m=m[None, :], z=z[:, None])`` has the shape ``(nz, nm)``. They take
their arguments by keyword only, so a call always says which is the mass and which the
redshift. With
:math:`\sigma_{\rm raw}(m)` the mass variance of the unnormalised power
(:class:`~hmf.core.mass_variance.MassVariance`):

* :math:`\sigma(m, z) = \sqrt{A}\,D(z)\,\sigma_{\rm raw}(m)`, and its slope
  :math:`d\ln\sigma/d\ln m`, which does not depend on z;
* the peak height :math:`\nu = \delta_c/\sigma`;
* :math:`f(\sigma)`, from the fit, with the inputs of the fits
  (:mod:`hmf.core.fits`): z, :math:`\Omega_m(z)` of CDM + baryons, the overdensity of
  the fit's mass definition relative to the mean density, :math:`\delta_c`,
  :math:`n_{\rm eff}` and m;
* the mass function

  .. math:: \frac{dn}{dm} = f(\sigma)\frac{\bar\rho_{\rm cb}}{m^2}
            \left|\frac{d\ln\sigma}{d\ln m}\right|,

  mapped by the fit's :meth:`~hmf.core.fits.FittingFunction.modify_dndm_kernel`
  (with n(>m) of the unmodified mass function, computed only for a fit whose
  ``modify_dndm_needs_ngtm`` is set: :class:`~hmf.core.fits.Behroozi`), and
  :math:`dn/d\ln m` and :math:`dn/d\log_{10} m`;
* the cumulative mass function n(>m), and the mass in haloes above m, rho(>m).

The converters :meth:`~MassFunction.m_from_sigma`,
:meth:`~MassFunction.m_from_peak_height` and :meth:`~MassFunction.m_from_radius`
invert sigma(m, z), the peak height and the filter radius. :meth:`MassFunction.at`
gives the quantities at one z and fixed masses, as cached arrays.

The mass definition
-------------------
The mass function is that of the fit's own mass definition
(:attr:`FittingFunction.measured_mass_definition
<hmf.core.fits.FittingFunction.measured_mass_definition>`), whose overdensity it
passes to the fit: a Tinker08 mass function is in SO-mean(200), a Jenkins one in
FoF(b = 0.2). Masses are not converted between definitions.

The cumulative mass function
----------------------------
n(>m) and rho(>m) are integrated on the mass lattice of the variance,
:math:`\log_{10} m = j\Delta`, from its top node,
:math:`M_{\rm top} = 10^{j_{\rm top}\Delta}` with :math:`j_{\rm top}\Delta` the last
node at or below ``mass_accuracy.log10_m_max`` (:math:`10^{17.5}` M☉/h by default),
down to m. Each panel between nodes is integrated by 4-point Gauss-Legendre quadrature
in ln m, and the panels are summed from the top down; between nodes, the rest of the
panel is integrated by the same rule (see :mod:`hmf.core._kernels.mass_function`). So
n(>m) is bit-for-bit the same whether it is asked for one mass or many, in any order,
and however far the lattice has been extended.

The integrals stop at :math:`M_{\rm top}`: masses above it raise a
:class:`~hmf.core.domain.DomainError`, and the haloes above it are not counted. For
ΛCDM they are utterly negligible: at z = 0, :math:`dn/d\ln m` at :math:`10^{17.5}`
M☉/h is :math:`\sim 10^{-50}` of its value at :math:`10^{16}` M☉/h. A spectrum with
much more power on large scales (a larger sigma_8, or a power law) may need a larger
``log10_m_max``.

The integrand is the mass function itself, without the calibration policy: the
policy applies to the masses and redshifts asked for, not to the lattice.

Domains
-------
The fit's valid domain always raises. Outside its calibration domain,
``domain_policy`` applies to each mass and redshift asked for (#390): ``"ignore"``
(the default), ``"warn"`` (once per stage), ``"mask"`` (NaN) or ``"raise"``.
:meth:`MassFunction.in_calibration_domain` gives the mask. Masses are bounded by the
variance's ``valid_domain``, and by :math:`M_{\rm top}` for n(>m) and rho(>m); redshifts
by the growth model's domain and table.
"""

from __future__ import annotations

import math
from functools import cached_property
from typing import Any

import astropy.units as u
import attrs
import numpy as np
import numpy.typing as npt
from astropy.cosmology import FLRW, Planck18

# For the fully qualified annotations below.
import hmf.core.linear_power

from ._fields import field
from ._kernels import fits as fit_kernels
from ._kernels import mass_function as mf_kernels
from ._kernels.lattice import LATTICE_RTOL, lattice_index
from ._validators import positive
from .accuracy import KAccuracy, MassAccuracy, check_consistent
from .cache import DiskCache, to_disk_cache
from .domain import (
    DOMAIN_POLICIES,
    Domain,
    DomainPolicy,
    Interval,
    check_extent,
)
from .filters import Filter
from .fits import FitInputs, FittingFunction, Tinker08, evaluate_fsigma
from .growth import Growth
from .growth_models import GrowthModel
from .linear_power import LinearPower
from .mass_variance import MassVariance, n_eff_kernel
from .species import omega_m, omega_m0
from .stage import Stage
from .transfer import Transfer, UnnormalisedPower
from .transfer_models import TransferModel
from .units import (
    Mpc_h,
    Msun_h,
    UnitContext,
    dndm_unit,
    number_density_unit,
    rho_unit,
    to_canonical,
    unit_boundary,
)

__all__ = ["MassFunction", "MassFunctionView"]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

_LN10 = math.log(10)

#: Redshifts: finite and >= 0 (the growth stage also bounds them by its table).
_Z_DOMAIN = Domain({"z": (0.0, None)})

#: Values that must be finite and > 0: sigma, the peak height and filter radii.
_POSITIVE = Interval(0, None, lower_open=True)
_POSITIVE_DOMAIN = Domain(
    {
        "sigma": _POSITIVE,
        "peak_height": _POSITIVE,
        "r": Interval(0, None, Mpc_h, lower_open=True),
    }
)


def _default_variance(stage: MassFunction) -> MassVariance:
    """The default variance: a TopHat MassVariance of the linear power's power source."""
    lp = stage.linear_power
    return MassVariance(power=lp.power_source, k_accuracy=lp.k_accuracy)


def _cosmology_parameter(cosmology: FLRW, key: str, name: str) -> float:
    """A parameter from an astropy cosmology's ``meta`` (e.g. Planck18's ``sigma8``)."""
    value = cosmology.meta.get(key)
    if value is None:
        raise ValueError(
            f"MassFunction.build: the cosmology {cosmology.name!r} does not give {name} "
            f"(its meta has no {key!r}, as astropy's realisations do): pass {name}=..."
        )
    return float(value)


@attrs.frozen(kw_only=True)
class MassFunction(Stage):
    """The halo mass function and the quantities it is built from, as functions of (m, z).

    See the module documentation. :meth:`build` builds one with hmf 3.x-like defaults.

    Examples
    --------
    >>> import numpy as np
    >>> from hmf.core.units import Msun_h
    >>> mf = MassFunction.build(transfer_model="EH")
    >>> m = np.logspace(10, 15, 6) * Msun_h
    >>> mf.dndm(m=m[None, :], z=np.array([0.0, 1.0])[:, None]).shape
    (2, 6)
    >>> mf2 = mf.evolve(linear_power=mf.linear_power.evolve(sigma_8=0.75))
    >>> mf2.variance is mf.variance
    True
    """

    # Fully qualified, so that the docs can tell it from v3's classes.
    linear_power: hmf.core.linear_power.LinearPower = field(
        validator=attrs.validators.instance_of(LinearPower),
        doc="The linear power: sigma_8 normalisation and growth.",
    )
    variance: MassVariance = field(
        default=attrs.Factory(_default_variance, takes_self=True),
        validator=attrs.validators.instance_of(MassVariance),
        doc=(
            "The mass variance of the unnormalised power: its power must be "
            "linear_power.power_source, and its k_accuracy linear_power's. It defaults to "
            "a TopHat MassVariance with the default mass accuracy."
        ),
    )
    fit: FittingFunction = field(
        factory=Tinker08,
        converter=FittingFunction.coerce,
        doc="The fitting function: an instance, a class or a registered name (e.g. 'ST').",
    )
    delta_c: float = field(
        default=1.686,
        converter=float,
        validator=positive,
        doc="The critical overdensity for collapse, linearly extrapolated to z = 0.",
    )
    domain_policy: DomainPolicy = field(
        default="ignore",
        validator=attrs.validators.in_(DOMAIN_POLICIES),
        doc=(
            "What to do with masses and redshifts outside the fit's calibration domain: "
            "'ignore', 'warn' (once per stage), 'mask' (NaN) or 'raise'."
        ),
    )

    def __attrs_post_init__(self) -> None:
        """Check that the variance is of the linear power's power, at the same accuracy."""
        power, lp = self.variance.power, self.linear_power
        same = (
            isinstance(power, UnnormalisedPower)
            and power.species == lp.species
            and (power.transfer is lp.transfer or power.transfer == lp.transfer)
        )
        if not same:
            raise ValueError(
                "MassFunction: the variance's power must be linear_power.power_source (the "
                "unnormalised power of the linear power's species, from its transfer "
                "stage). Build it as MassVariance(power=linear_power.power_source, ...)."
            )
        check_consistent(self.linear_power, self.variance)

    # ---------------------------------------------------------------------------------
    # Construction
    # ---------------------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        cosmology: FLRW = Planck18,
        transfer_model: TransferModel | str = "CAMB",
        growth_model: GrowthModel | str = "ODE",
        n_s: float | None = None,
        sigma_8: float | None = None,
        species: str = "cb",
        sigma_8_species: str = "tot",
        fit: FittingFunction | str = "Tinker08",
        filter: Filter | str = "TopHat",  # noqa: A002
        delta_c: float = 1.686,
        domain_policy: DomainPolicy = "ignore",
        k_accuracy: KAccuracy | None = None,
        mass_accuracy: MassAccuracy | None = None,
        disk_cache: DiskCache | bool | str | None = None,
    ) -> MassFunction:
        """Build the whole stage tree, from explicit keyword arguments.

        Models may be given as instances, classes or registered names. One
        ``k_accuracy`` is given to the transfer, growth, linear-power and variance
        stages, so a CAMB or CLASS growth model shares the transfer's Boltzmann run.

        Parameters
        ----------
        cosmology
            The astropy cosmology (default Planck18).
        transfer_model
            The transfer model (default CAMB).
        growth_model
            The growth model (default the growth ODE).
        n_s
            The spectral index; by default the cosmology's (``cosmology.meta["n"]``,
            which astropy's realisations give: 0.9665 for Planck18).
        sigma_8
            The rms of the ``sigma_8_species`` field at 8 Mpc/h today; by default the
            cosmology's (``cosmology.meta["sigma8"]``: 0.8102 for Planck18).
        species
            The matter species of the power: ``"cb"`` (CDM + baryons, default) or
            ``"tot"``.
        sigma_8_species
            The matter species sigma_8 is the rms of: ``"tot"`` (default, as hmf 3.x)
            or ``"cb"``.
        fit
            The fitting function (default Tinker08).
        filter
            The filter of the mass variance (default TopHat).
        delta_c
            The critical overdensity (default 1.686).
        domain_policy
            The calibration-domain policy (default ``"ignore"``).
        k_accuracy
            The wavenumber accuracy (default ``KAccuracy()``).
        mass_accuracy
            The mass lattice's accuracy (default ``MassAccuracy()``).
        disk_cache
            Where to cache Boltzmann-code output on disk (see
            :attr:`Transfer.disk_cache <hmf.core.transfer.Transfer.disk_cache>`).

        Returns
        -------
        MassFunction

        Raises
        ------
        ValueError
            If ``n_s`` or ``sigma_8`` is not given and the cosmology's ``meta`` does
            not have it, or for any invalid argument.
        """
        k_accuracy = KAccuracy() if k_accuracy is None else k_accuracy
        mass_accuracy = MassAccuracy() if mass_accuracy is None else mass_accuracy
        if n_s is None:
            n_s = _cosmology_parameter(cosmology, "n", "n_s")
        if sigma_8 is None:
            sigma_8 = _cosmology_parameter(cosmology, "sigma8", "sigma_8")
        transfer = Transfer(
            cosmology=cosmology,
            model=TransferModel.coerce(transfer_model),
            n_s=n_s,
            k_accuracy=k_accuracy,
            disk_cache=to_disk_cache(disk_cache),
        )
        growth = Growth.from_transfer(transfer, model=growth_model)
        linear_power = LinearPower(
            transfer=transfer,
            growth=growth,
            sigma_8=sigma_8,
            species=species,
            sigma_8_species=sigma_8_species,
        )
        variance = MassVariance(
            power=linear_power.power_source,
            filter=Filter.coerce(filter),
            mass_accuracy=mass_accuracy,
            k_accuracy=k_accuracy,
        )
        return cls(
            linear_power=linear_power,
            variance=variance,
            fit=FittingFunction.coerce(fit),
            delta_c=delta_c,
            domain_policy=domain_policy,
        )

    # ---------------------------------------------------------------------------------
    # Setup
    # ---------------------------------------------------------------------------------

    @property
    def cosmology(self) -> FLRW:
        """The cosmology (the linear power's)."""
        return self.linear_power.cosmology

    @cached_property
    def _unit_context(self) -> UnitContext:
        """The units context, with the cosmology's H0."""
        return UnitContext(self.cosmology.H0)

    @cached_property
    def rho_mean0(self) -> float:
        """The mean comoving density of CDM + baryons today, in Msun h^2 / Mpc^3.

        The variance's power source's ``rho_mean0``: the density in dn/dm.
        """
        return float(self.variance.power.rho_mean0)

    @cached_property
    def _h(self) -> float:
        """The dimensionless Hubble parameter, for the fits' modify_dndm_kernel."""
        return float(self.cosmology.h)

    @cached_property
    def _omega_m0(self) -> float:
        """Omega_m0 of CDM + baryons, for the fits' modify_dndm_kernel."""
        return omega_m0(self.cosmology, "cb")

    @cached_property
    def _needs_delta_halo(self) -> bool:
        """Whether the fit, or one of its domains, needs the halo overdensity."""
        return "delta_halo" in self.fit.domain_inputs()

    @cached_property
    def _ln_m_step(self) -> float:
        """The spacing of the mass lattice in ln m."""
        return self.variance.mass_accuracy.dlog10_m * _LN10

    @cached_property
    def _j_top(self) -> int:
        """The index of the top node of the integrals: the last at or below log10_m_max."""
        acc = self.variance.mass_accuracy
        return lattice_index(acc.log10_m_max, acc.dlog10_m, up=False)

    @cached_property
    def m_top(self) -> float:
        r"""The upper limit of the integrals n(>m) and rho(>m), in Msun/h (a plain float).

        The top node of the mass lattice, :math:`10^{j\Delta}` with :math:`j\Delta`
        the last node at or below ``variance.mass_accuracy.log10_m_max``.
        """
        return math.exp(self._j_top * self._ln_m_step)

    @cached_property
    def valid_domain(self) -> Domain:
        """Where the stage's methods can be evaluated.

        ``m`` is the variance's (``> 0``, or the default lattice range with
        ``mass_accuracy.extension="raise"``); ``m_integral``, the masses of n(>m) and
        rho(>m), is that and at most :attr:`m_top`; ``z`` is ``>= 0`` (the growth
        stage also raises above its table); ``sigma``, ``peak_height`` and ``r`` are
        ``> 0``. Bounds are in canonical units (Msun/h, Mpc/h).
        """
        m = self.variance.valid_domain["m"]
        upper = min(m.upper, self.m_top * (1 + LATTICE_RTOL * self._ln_m_step))
        m_integral = Interval(m.lower, upper, Msun_h, lower_open=m.lower_open)
        return Domain(
            {
                "m": m,
                "m_integral": m_integral,
                "z": _Z_DOMAIN["z"],
                "sigma": _POSITIVE_DOMAIN["sigma"],
                "peak_height": _POSITIVE_DOMAIN["peak_height"],
                "r": _POSITIVE_DOMAIN["r"],
            },
            source="MassFunction: the variance's masses (and m_top for the integrals), z >= 0.",
        )

    def _check(self, method: str, **values: Any) -> None:
        """Check the arguments of a public method against :attr:`valid_domain`."""
        domain = self.valid_domain
        where = f"MassFunction.{method}"
        for name, value in values.items():
            if name in ("m", "m_integral"):
                value = value << Msun_h
            elif name == "r":
                value = value << Mpc_h
            domain.check({name: value}, where=where)

    # ---------------------------------------------------------------------------------
    # Kernel-level entry points
    # ---------------------------------------------------------------------------------

    def ln_sigma_and_slope_kernel(
        self, *, m: npt.ArrayLike, z: npt.ArrayLike
    ) -> tuple[FloatArray, FloatArray]:
        """Ln sigma(m, z) and dln(sigma)/dln(m), at kernel level.

        For library code working in plain arrays (see :mod:`hmf.core._kernels`). It
        checks m as :meth:`MassVariance.ln_sigma_and_slope_kernel
        <hmf.core.mass_variance.MassVariance.ln_sigma_and_slope_kernel>` and z as
        :meth:`LinearPower.sigma_scale_kernel
        <hmf.core.linear_power.LinearPower.sigma_scale_kernel>` do.

        Parameters
        ----------
        m
            Masses in Msun/h: a plain array.
        z
            Redshifts (dimensionless), broadcast with ``m`` like numpy.

        Returns
        -------
        ln_sigma, dlnsigma_dlnm : numpy.ndarray
            Dimensionless, with the broadcast shape of ``m`` and ``z``.

        Raises
        ------
        DomainError
            If a mass or redshift is outside its domain (from their extent), or a mass
            can't be resolved by the variance's k grid.
        """
        ln_raw, slope = self.variance.ln_sigma_and_slope_kernel(m)
        ln_sigma = ln_raw + np.log(self.linear_power.sigma_scale_kernel(z))
        return ln_sigma, np.array(np.broadcast_to(slope, ln_sigma.shape))

    def _fit_inputs(
        self, m: FloatArray, z: npt.ArrayLike, sigma: FloatArray, slope: FloatArray
    ) -> FitInputs:
        """The inputs of the fit at (m, z), from sigma and its slope there."""
        omega_m_z = omega_m(self.cosmology, z, "cb")
        delta_halo = (
            self.fit.measured_mass_definition.delta_halo_mean_kernel(omega_m_z)
            if self._needs_delta_halo
            else None
        )
        return FitInputs(
            sigma=sigma,
            z=z,
            omega_m_z=omega_m_z,
            delta_halo=delta_halo,
            delta_c=self.delta_c,
            n_eff=n_eff_kernel(slope),
            m=m,
        )

    def _inputs_at(self, m: npt.ArrayLike, z: npt.ArrayLike) -> tuple[FitInputs, FloatArray]:
        """The fit's inputs at (m, z), and dln(sigma)/dln(m) there (shape of m)."""
        m = np.asarray(m, dtype=float)
        ln_raw, slope = self.variance.ln_sigma_and_slope_kernel(m)
        sigma = np.exp(ln_raw) * self.linear_power.sigma_scale_kernel(z)
        return self._fit_inputs(m, z, sigma, slope), slope

    def _dndm(
        self, m: npt.ArrayLike, z: npt.ArrayLike, *, policy: bool, raw: bool = False
    ) -> FloatArray:
        """dn/dm at (m, z), in h^4 / (Msun Mpc^3).

        With ``policy``, the domain policy applies to f(sigma); ``raw`` skips the fit's
        modify_dndm_kernel (the mass function from f(sigma) alone).
        """
        x, slope = self._inputs_at(m, z)
        if policy and self.domain_policy != "ignore":
            f = evaluate_fsigma(self.fit, x, policy=self.domain_policy, owner=self).fsigma
        else:
            f = self.fit.fsigma_kernel(x)
        m = x.m
        dndm: FloatArray = f * (self.rho_mean0 * np.abs(slope) / m**2)
        if raw or not self.fit.modifies_dndm:
            return dndm
        if self.fit.modify_dndm_needs_ngtm:
            ngtm_raw = self._integral(m, z, raw=True, moment=0)
        else:
            # Not used: NaN would make any use of it show.
            ngtm_raw = np.full(dndm.shape, np.nan)
        return self.fit.modify_dndm_kernel(
            m, dndm, z=z, ngtm=ngtm_raw, h=self._h, omega_m0=self._omega_m0
        )

    def _stencil_start(self, j: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        """The first of the four nodes interpolating the panel [j, j + 1].

        It is j - 1, shifted inwards in the panels at the ends of the default lattice
        (at ``10**log10_m_min`` and :attr:`m_top`), whose outer neighbour the k grid
        may not resolve. Below the default lattice (with ``extension="auto"``) it is
        j - 1 again. It depends on j alone, so every panel always has the same stencil.
        """
        bottom = self._j_bottom
        start = np.where(j >= bottom, np.maximum(j - 1, bottom), j - 1)
        out: npt.NDArray[np.int64] = np.minimum(start, self._j_top - 3)
        return out

    @cached_property
    def _j_bottom(self) -> int:
        """The index of the lowest node of the default lattice (at or above log10_m_min)."""
        acc = self.variance.mass_accuracy
        return lattice_index(acc.log10_m_min, acc.dlog10_m, up=True)

    @cached_property
    def _j_floor(self) -> int:
        """The lowest panel the integrals may use: the bottom with extension='raise'."""
        if self.variance.mass_accuracy.extension == "raise":
            return self._j_bottom
        return int(np.iinfo(np.int64).min // 2)

    def _node_values(self, n_lo: int, z: FloatArray, *, raw: bool, moment: int) -> FloatArray:
        """m^moment dn/dln m at the nodes n_lo to the top, for each z (no domain policy).

        Shape ``(z.size, top - n_lo + 1)``. Without ``raw``, the fit's modify_dndm_kernel
        is applied, with n(>m) at the nodes from :meth:`_lattice_integrals`.
        """
        m = np.exp(np.arange(n_lo, self._j_top + 1, dtype=np.int64) * np.float64(self._ln_m_step))
        zz = z[:, None]
        dndm = self._dndm(m, zz, policy=False, raw=True)
        if not raw and self.fit.modifies_dndm:
            if self.fit.modify_dndm_needs_ngtm:
                ngtm_raw = self._lattice_integrals(n_lo, z, raw=True, moment=0)[2]
            else:
                ngtm_raw = np.full(dndm.shape, np.nan)
            dndm = self.fit.modify_dndm_kernel(
                m, dndm, z=zz, ngtm=ngtm_raw, h=self._h, omega_m0=self._omega_m0
            )
        out: FloatArray = dndm * m ** (1 + moment)
        return out

    def _lattice_integrals(
        self, j_lo: int, z: FloatArray, *, raw: bool, moment: int
    ) -> tuple[int, FloatArray, FloatArray]:
        """The node values, and the integral from each node j_lo to the top, for each z.

        Returns the first node of the values, ``n_lo`` (that of the stencil of the panel
        above ``j_lo``), the values at the nodes ``n_lo`` to the top (from
        :meth:`_node_values`), and the integrals at the nodes ``j_lo`` to the top
        (shape ``(z.size, top - j_lo + 1)``), summed from the top down over the panels,
        each integrated from the node values (see :mod:`hmf.core._kernels.mass_function`).
        """
        panels = np.arange(j_lo, self._j_top, dtype=np.int64)
        start = self._stencil_start(panels)
        n_lo = int(start[0])
        values = self._node_values(n_lo, z, raw=raw, moment=moment)
        stencils = values[:, (start - n_lo)[:, None] + np.arange(4)]
        integrals = mf_kernels.log_lagrange_integrals(
            stencils, start - panels, 0.0, 1.0, self._ln_m_step
        )
        return n_lo, values, mf_kernels.cumulative_from_top(integrals)

    def _integral(
        self, m: npt.ArrayLike, z: npt.ArrayLike, *, raw: bool, moment: int
    ) -> FloatArray:
        """The integral of m^moment dn/dln m from m to :attr:`m_top`, at (m, z).

        On the mass lattice: see "The cumulative mass function" in the module
        documentation.
        """
        m = check_extent(
            "m_integral",
            m,
            self.valid_domain,
            where="MassFunction (the integrals n(>m) and rho(>m) stop at m_top)",
        )
        z = np.asarray(z, dtype=float)
        m_b, z_b = np.broadcast_arrays(m, z)
        shape = m_b.shape
        m_flat, z_flat = m_b.ravel(), z_b.ravel()
        if m_flat.size == 0:
            return np.zeros(shape)
        step = self._ln_m_step
        t = np.log(m_flat) / step
        # The panel [j, j + 1] each mass is in (a mass at the top node is in the last),
        # and its position u in it.
        j = np.clip(np.floor(t).astype(np.int64), self._j_floor, self._j_top - 1)
        u = t - j
        j_lo = int(j.min())
        z_unique, z_index = np.unique(z_flat, return_inverse=True)
        # The integral from each node to the top, for each redshift.
        n_lo, values, cumulative = self._lattice_integrals(j_lo, z_unique, raw=raw, moment=moment)
        # The rest of each mass's panel, above it, from the same interpolant.
        start = self._stencil_start(j)
        stencils = values[z_index[:, None], (start - n_lo)[:, None] + np.arange(4)]
        partial = mf_kernels.log_lagrange_integrals(stencils, start - j, u, 1.0, step)
        out: FloatArray = cumulative[z_index, j + 1 - j_lo] + partial
        return out.reshape(shape)

    def _apply_policy(self, values: FloatArray, m: FloatArray, z: npt.ArrayLike) -> FloatArray:
        """Apply the domain policy at (m, z) to values computed without it (e.g. n(>m))."""
        if self.domain_policy == "ignore":
            return values
        x, _ = self._inputs_at(m, z)
        result = evaluate_fsigma(self.fit, x, policy=self.domain_policy, owner=self)
        inside = np.broadcast_to(result.in_calibration_domain, values.shape)
        if self.domain_policy == "mask":
            return np.where(inside, values, np.nan)
        return values

    def fsigma_kernel(self, *, m: npt.ArrayLike, z: npt.ArrayLike) -> FloatArray:
        """f(sigma) at (m, z), with the domain policy, at kernel level: see :meth:`fsigma`.

        Parameters
        ----------
        m
            Masses in Msun/h: a plain array.
        z
            Redshifts, broadcast with ``m``.

        Returns
        -------
        numpy.ndarray
            f(sigma), dimensionless, with the broadcast shape (NaN outside the
            calibration domain with ``domain_policy="mask"``).

        Raises
        ------
        DomainError
            As :meth:`ln_sigma_and_slope_kernel`, or outside the fit's valid domain, or
            its calibration domain with ``domain_policy="raise"``.
        """
        x, _ = self._inputs_at(m, z)
        if self.domain_policy == "ignore":
            return self.fit.fsigma_kernel(x)
        out: FloatArray = evaluate_fsigma(self.fit, x, policy=self.domain_policy, owner=self).fsigma
        return out

    def dndm_kernel(self, *, m: npt.ArrayLike, z: npt.ArrayLike) -> FloatArray:
        """The mass function dn/dm at (m, z), at kernel level: see :meth:`dndm`.

        Parameters
        ----------
        m
            Masses in Msun/h: a plain array.
        z
            Redshifts, broadcast with ``m``.

        Returns
        -------
        numpy.ndarray
            dn/dm in h^4 / (Msun Mpc^3), with the broadcast shape.

        Raises
        ------
        DomainError
            As :meth:`fsigma_kernel` (and, for a fit that uses n(>m), as
            :meth:`ngtm_kernel`).
        """
        return self._dndm(m, z, policy=True)

    def ngtm_kernel(self, *, m: npt.ArrayLike, z: npt.ArrayLike) -> FloatArray:
        """The cumulative mass function n(>m) at (m, z), at kernel level: see :meth:`ngtm`.

        Parameters
        ----------
        m
            Masses in Msun/h: a plain array (at most :attr:`m_top`).
        z
            Redshifts, broadcast with ``m``.

        Returns
        -------
        numpy.ndarray
            n(>m) in h^3 / Mpc^3, with the broadcast shape.

        Raises
        ------
        DomainError
            If a mass is above :attr:`m_top`, or as :meth:`fsigma_kernel`.
        """
        out = self._integral(m, z, raw=False, moment=0)
        return self._apply_policy(out, np.asarray(m, dtype=float), z)

    def rho_gtm_kernel(self, *, m: npt.ArrayLike, z: npt.ArrayLike) -> FloatArray:
        """The mass density in haloes above m at z, at kernel level: see :meth:`rho_gtm`.

        Parameters
        ----------
        m
            Masses in Msun/h: a plain array (at most :attr:`m_top`).
        z
            Redshifts, broadcast with ``m``.

        Returns
        -------
        numpy.ndarray
            rho(>m) in Msun h^2 / Mpc^3, with the broadcast shape.

        Raises
        ------
        DomainError
            As :meth:`ngtm_kernel`.
        """
        out = self._integral(m, z, raw=False, moment=1)
        return self._apply_policy(out, np.asarray(m, dtype=float), z)

    def in_calibration_domain_kernel(self, *, m: npt.ArrayLike, z: npt.ArrayLike) -> BoolArray:
        """Whether (m, z) is in the fit's calibration domain, at kernel level.

        Parameters
        ----------
        m
            Masses in Msun/h: a plain array.
        z
            Redshifts, broadcast with ``m``.

        Returns
        -------
        numpy.ndarray of bool
            With the broadcast shape; all True if the fit has no calibration domain.

        Raises
        ------
        DomainError
            As :meth:`ln_sigma_and_slope_kernel`.
        """
        x, _ = self._inputs_at(m, z)
        shape = np.broadcast_shapes(np.shape(x.sigma), np.shape(m), np.shape(z))
        return np.array(np.broadcast_to(self.fit.in_calibration_domain(x), shape))

    def m_from_sigma_kernel(self, *, sigma: npt.ArrayLike, z: npt.ArrayLike) -> FloatArray:
        """The mass at which sigma(m, z) takes the given values, at kernel level.

        Parameters
        ----------
        sigma
            Values of sigma(m, z), dimensionless: a plain array.
        z
            Redshifts, broadcast with ``sigma``.

        Returns
        -------
        numpy.ndarray
            Masses in Msun/h, with the broadcast shape.

        Raises
        ------
        DomainError
            As :meth:`MassVariance.m_from_sigma_kernel
            <hmf.core.mass_variance.MassVariance.m_from_sigma_kernel>`, for
            sigma(m, z) / (sqrt(A) D(z)), or if a z is outside its domain.
        """
        sigma = check_extent(
            "sigma", sigma, _POSITIVE_DOMAIN, where="MassFunction.m_from_sigma_kernel"
        )
        return self.variance.m_from_sigma_kernel(sigma / self.linear_power.sigma_scale_kernel(z))

    # ---------------------------------------------------------------------------------
    # Public methods
    # ---------------------------------------------------------------------------------

    @unit_boundary(m=Msun_h)
    def sigma(self, *, m: Any, z: Any) -> FloatArray:
        """The mass variance sigma(m, z) of the linear power, normalised to sigma_8.

        Parameters
        ----------
        m : Quantity
            Masses, in Msun/h (or Msun, converted with the cosmology's H0).
        z : float or array_like
            Redshifts, dimensionless, broadcast with ``m`` like numpy.

        Returns
        -------
        numpy.float64 or numpy.ndarray
            Dimensionless, with the broadcast shape (a scalar for scalar inputs).

        Raises
        ------
        DomainError
            If a mass or redshift is outside :attr:`valid_domain` (or the growth's
            table), or a mass can't be resolved by the variance's k grid.
        """
        self._check("sigma", m=m, z=z)
        return np.exp(self.ln_sigma_and_slope_kernel(m=m, z=z)[0])

    @unit_boundary(m=Msun_h)
    def dlnsigma_dlnm(self, *, m: Any, z: Any) -> FloatArray:
        """The slope dln(sigma)/dln(m), the same at every z.

        Parameters
        ----------
        m : Quantity
            Masses, in Msun/h (or Msun).
        z : float or array_like
            Redshifts, broadcast with ``m`` (they set the shape only; they are checked).

        Returns
        -------
        numpy.float64 or numpy.ndarray
            Dimensionless, with the broadcast shape.

        Raises
        ------
        DomainError
            As :meth:`sigma`.
        """
        self._check("dlnsigma_dlnm", m=m, z=z)
        return self.ln_sigma_and_slope_kernel(m=m, z=z)[1]

    @unit_boundary(m=Msun_h)
    def peak_height(self, *, m: Any, z: Any) -> FloatArray:
        """The peak height, nu = delta_c / sigma(m, z).

        Parameters
        ----------
        m : Quantity
            Masses, in Msun/h (or Msun).
        z : float or array_like
            Redshifts, broadcast with ``m``.

        Returns
        -------
        numpy.float64 or numpy.ndarray
            Dimensionless, with the broadcast shape.

        Raises
        ------
        DomainError
            As :meth:`sigma`.
        """
        self._check("peak_height", m=m, z=z)
        sigma = np.exp(self.ln_sigma_and_slope_kernel(m=m, z=z)[0])
        return fit_kernels.peak_height(sigma, self.delta_c)

    @unit_boundary(m=Msun_h)
    def fsigma(self, *, m: Any, z: Any) -> FloatArray:
        """The fit's multiplicity function f(sigma) at (m, z).

        Parameters
        ----------
        m : Quantity
            Masses, in Msun/h (or Msun).
        z : float or array_like
            Redshifts, broadcast with ``m``.

        Returns
        -------
        numpy.float64 or numpy.ndarray
            Dimensionless, with the broadcast shape; NaN outside the calibration
            domain with ``domain_policy="mask"``.

        Raises
        ------
        DomainError
            As :meth:`sigma`, outside the fit's valid domain, or outside its
            calibration domain with ``domain_policy="raise"``.

        Warns
        -----
        HMFExtrapolationWarning
            Outside the calibration domain with ``domain_policy="warn"`` (once per
            stage).
        """
        self._check("fsigma", m=m, z=z)
        return self.fsigma_kernel(m=m, z=z)

    @unit_boundary(m=Msun_h, returns=dndm_unit)
    def dndm(self, *, m: Any, z: Any) -> FloatArray:
        """The mass function dn/dm at (m, z).

        Parameters
        ----------
        m : Quantity
            Masses, in Msun/h (or Msun), in the fit's mass definition.
        z : float or array_like
            Redshifts, broadcast with ``m``.

        Returns
        -------
        Quantity
            In h^4 / (Msun Mpc^3), with the broadcast shape.

        Raises
        ------
        DomainError
            As :meth:`fsigma`.

        Warns
        -----
        HMFExtrapolationWarning
            As :meth:`fsigma`.
        """
        self._check("dndm", m=m, z=z)
        return self.dndm_kernel(m=m, z=z)

    @unit_boundary(m=Msun_h, returns=number_density_unit)
    def dndlnm(self, *, m: Any, z: Any) -> FloatArray:
        """The mass function per unit ln m, m dn/dm.

        Parameters
        ----------
        m : Quantity
            Masses, in Msun/h (or Msun).
        z : float or array_like
            Redshifts, broadcast with ``m``.

        Returns
        -------
        Quantity
            In h^3 / Mpc^3, with the broadcast shape.

        Raises
        ------
        DomainError
            As :meth:`fsigma`.

        Warns
        -----
        HMFExtrapolationWarning
            As :meth:`fsigma`.
        """
        self._check("dndlnm", m=m, z=z)
        out: FloatArray = m * self.dndm_kernel(m=m, z=z)
        return out

    @unit_boundary(m=Msun_h, returns=number_density_unit)
    def dndlog10m(self, *, m: Any, z: Any) -> FloatArray:
        """The mass function per unit log10 m, ln(10) m dn/dm.

        Parameters
        ----------
        m : Quantity
            Masses, in Msun/h (or Msun).
        z : float or array_like
            Redshifts, broadcast with ``m``.

        Returns
        -------
        Quantity
            In h^3 / Mpc^3, with the broadcast shape.

        Raises
        ------
        DomainError
            As :meth:`fsigma`.

        Warns
        -----
        HMFExtrapolationWarning
            As :meth:`fsigma`.
        """
        self._check("dndlog10m", m=m, z=z)
        out: FloatArray = _LN10 * m * self.dndm_kernel(m=m, z=z)
        return out

    @unit_boundary(m=Msun_h, returns=number_density_unit)
    def ngtm(self, *, m: Any, z: Any) -> FloatArray:
        """The cumulative mass function: the number density of haloes above m.

        Integrated on the mass lattice up to :attr:`m_top` (see "The cumulative mass
        function" in the module documentation): bit-for-bit the same for a mass
        whatever else is asked for with it.

        Parameters
        ----------
        m : Quantity
            Masses, in Msun/h (or Msun), at most :attr:`m_top`.
        z : float or array_like
            Redshifts, broadcast with ``m``.

        Returns
        -------
        Quantity
            In h^3 / Mpc^3, with the broadcast shape; NaN at (m, z) outside the
            calibration domain with ``domain_policy="mask"``.

        Raises
        ------
        DomainError
            If a mass is above :attr:`m_top`, or as :meth:`fsigma` (the policy applies
            at (m, z)).

        Warns
        -----
        HMFExtrapolationWarning
            As :meth:`fsigma`.
        """
        self._check("ngtm", m_integral=m, z=z)
        return self.ngtm_kernel(m=m, z=z)

    @unit_boundary(m=Msun_h, returns=rho_unit)
    def rho_gtm(self, *, m: Any, z: Any) -> FloatArray:
        """The mass density in haloes above m (comoving).

        Integrated as :meth:`ngtm`.

        Parameters
        ----------
        m : Quantity
            Masses, in Msun/h (or Msun), at most :attr:`m_top`.
        z : float or array_like
            Redshifts, broadcast with ``m``.

        Returns
        -------
        Quantity
            In Msun h^2 / Mpc^3, with the broadcast shape.

        Raises
        ------
        DomainError
            As :meth:`ngtm`.

        Warns
        -----
        HMFExtrapolationWarning
            As :meth:`fsigma`.
        """
        self._check("rho_gtm", m_integral=m, z=z)
        return self.rho_gtm_kernel(m=m, z=z)

    @unit_boundary(m=Msun_h)
    def in_calibration_domain(self, *, m: Any, z: Any) -> BoolArray:
        """Whether each (m, z) is inside the fit's calibration domain.

        Parameters
        ----------
        m : Quantity
            Masses, in Msun/h (or Msun).
        z : float or array_like
            Redshifts, broadcast with ``m``.

        Returns
        -------
        bool or numpy.ndarray of bool
            With the broadcast shape (a numpy bool for scalar inputs); True
            everywhere for a fit without a calibration domain.

        Raises
        ------
        DomainError
            As :meth:`sigma`.
        """
        self._check("in_calibration_domain", m=m, z=z)
        return self.in_calibration_domain_kernel(m=m, z=z)

    @unit_boundary(returns=Msun_h)
    def m_from_sigma(self, *, sigma: Any, z: Any) -> FloatArray:
        """The mass at which sigma(m, z) takes the given values: the inverse of :meth:`sigma`.

        sigma(m, z) / (sqrt(A) D(z)) is inverted on the variance's lattice (see
        :meth:`MassVariance.m_from_sigma
        <hmf.core.mass_variance.MassVariance.m_from_sigma>`), so
        ``m_from_sigma(sigma=sigma(m=m, z=z), z=z)`` returns m to about 1e-12.

        Parameters
        ----------
        sigma : float or array_like
            Values of sigma, dimensionless.
        z : float or array_like
            Redshifts, broadcast with ``sigma``.

        Returns
        -------
        Quantity
            Masses, in Msun/h, with the broadcast shape.

        Raises
        ------
        DomainError
            If a value is not finite and > 0, is outside the range of sigma(m, z) on
            the variance's default lattice range, or sigma is not monotonic there.
        """
        self._check("m_from_sigma", sigma=sigma, z=z)
        return self.m_from_sigma_kernel(sigma=sigma, z=z)

    @unit_boundary(returns=Msun_h)
    def m_from_peak_height(self, *, peak_height: Any, z: Any) -> FloatArray:
        """The mass of a peak height nu at z: the inverse of :meth:`peak_height`.

        Parameters
        ----------
        peak_height : float or array_like
            Values of nu = delta_c / sigma, dimensionless.
        z : float or array_like
            Redshifts, broadcast with ``peak_height``.

        Returns
        -------
        Quantity
            Masses, in Msun/h, with the broadcast shape.

        Raises
        ------
        DomainError
            If a value is not finite and > 0, or as :meth:`m_from_sigma` for
            sigma = delta_c / nu.
        """
        self._check("m_from_peak_height", peak_height=peak_height, z=z)
        sigma = fit_kernels.peak_height(peak_height, self.delta_c)  # delta_c / nu
        return self.m_from_sigma_kernel(sigma=sigma, z=z)

    @unit_boundary(r=Mpc_h, returns=Msun_h)
    def m_from_radius(self, *, r: Any) -> FloatArray:
        """The mass of a filter radius, m = (4π/3) rho_cb (cR)³, with the filter's c.

        See :meth:`MassVariance.m_from_radius
        <hmf.core.mass_variance.MassVariance.m_from_radius>`; it does not depend on z.

        Parameters
        ----------
        r : Quantity
            Radii, in Mpc/h (or Mpc).

        Returns
        -------
        Quantity
            Masses, in Msun/h.

        Raises
        ------
        DomainError
            If a radius is not finite and > 0.
        """
        self._check("m_from_radius", r=r)
        return self.variance.m_from_radius_kernel(r)

    def at(self, *, z: float, m: u.Quantity | None = None) -> MassFunctionView:
        """The quantities of the mass function at one redshift and fixed masses.

        Parameters
        ----------
        z
            The redshift, a scalar.
        m
            The masses, a 1D Quantity in Msun/h (or Msun). By default, the nodes of
            the variance's lattice from ``10**log10_m_min`` to :attr:`m_top`.

        Returns
        -------
        MassFunctionView
            Its attributes are cached arrays at ``z`` and ``m``.

        Raises
        ------
        DomainError
            If ``z`` or a mass is outside :attr:`valid_domain`.
        ValueError
            If ``z`` is not a scalar, or ``m`` is not 1D.
        """
        if np.ndim(z) != 0:
            raise ValueError(f"MassFunction.at: z must be a scalar, got shape {np.shape(z)}.")
        z = float(z)
        if m is None:
            acc = self.variance.mass_accuracy
            j_lo = lattice_index(acc.log10_m_min, acc.dlog10_m, up=True)
            masses = np.exp(np.arange(j_lo, self._j_top + 1) * np.float64(self._ln_m_step))
        else:
            masses = np.asarray(
                to_canonical(
                    m, Msun_h, context=self._unit_context, where="MassFunction.at", name="m"
                ),
                dtype=float,
            )
            if masses.ndim != 1:
                raise ValueError(f"MassFunction.at: m must be 1D, got shape {masses.shape}.")
        self._check("at", m=masses, z=z)
        return MassFunctionView(mass_function=self, z=z, masses=masses)


@attrs.frozen(kw_only=True, eq=False)
class MassFunctionView:
    """The quantities of a :class:`MassFunction` at one redshift and fixed masses.

    Made by :meth:`MassFunction.at`. Each attribute is computed once, through the
    stage's kernel-level entry points, and cached. Dimensional ones are Quantities in
    h-units.

    Parameters
    ----------
    mass_function
        The stage.
    z
        The redshift.
    masses
        The masses, a plain array in Msun/h.
    """

    mass_function: MassFunction
    z: float
    masses: FloatArray

    @cached_property
    def m(self) -> u.Quantity:
        """The masses, in Msun/h."""
        return self.masses << Msun_h

    @cached_property
    def _ln_sigma_and_slope(self) -> tuple[FloatArray, FloatArray]:
        return self.mass_function.ln_sigma_and_slope_kernel(m=self.masses, z=self.z)

    @cached_property
    def sigma(self) -> FloatArray:
        """sigma(m, z)."""
        return np.exp(self._ln_sigma_and_slope[0])

    @cached_property
    def dlnsigma_dlnm(self) -> FloatArray:
        """dln(sigma)/dln(m)."""
        return self._ln_sigma_and_slope[1]

    @cached_property
    def peak_height(self) -> FloatArray:
        """The peak height delta_c / sigma."""
        return fit_kernels.peak_height(self.sigma, self.mass_function.delta_c)

    @cached_property
    def fsigma(self) -> FloatArray:
        """f(sigma), with the stage's domain policy."""
        return self.mass_function.fsigma_kernel(m=self.masses, z=self.z)

    @cached_property
    def dndm(self) -> u.Quantity:
        """dn/dm, in h^4 / (Msun Mpc^3)."""
        return self.mass_function.dndm_kernel(m=self.masses, z=self.z) << dndm_unit

    @cached_property
    def dndlnm(self) -> u.Quantity:
        """dn/dln m, in h^3 / Mpc^3."""
        return (self.masses * self.dndm.value) << number_density_unit

    @cached_property
    def dndlog10m(self) -> u.Quantity:
        """dn/dlog10 m, in h^3 / Mpc^3."""
        return (_LN10 * self.masses * self.dndm.value) << number_density_unit

    @cached_property
    def ngtm(self) -> u.Quantity:
        """n(>m), in h^3 / Mpc^3."""
        return self.mass_function.ngtm_kernel(m=self.masses, z=self.z) << number_density_unit

    @cached_property
    def rho_gtm(self) -> u.Quantity:
        """rho(>m), in Msun h^2 / Mpc^3."""
        return self.mass_function.rho_gtm_kernel(m=self.masses, z=self.z) << rho_unit

    @cached_property
    def in_calibration_domain(self) -> BoolArray:
        """Whether each mass is in the fit's calibration domain at z."""
        return self.mass_function.in_calibration_domain_kernel(m=self.masses, z=self.z)
