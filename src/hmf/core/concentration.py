r"""Concentration-mass relations of dark-matter haloes, as :mod:`hmf.core` models.

A concentration-mass relation gives the median concentration :math:`c(M, z)` of
haloes of mass M at redshift z, with :math:`c = r_\Delta/r_s` the ratio of the halo
radius to the scale radius of its density profile. A relation is calibrated in one
mass definition (:attr:`ConcentrationModel.mass_definition`): its masses and
concentrations are in that definition. A
:class:`~hmf.core.mass_conversion.MassConversion` converts them to others.

Every relation is a :class:`ConcentrationModel`, a model kind with its own registry
(``ConcentrationModel.get("Duffy08")``), so other packages (e.g. halomod) add theirs
by subclassing it. Each declares a :attr:`~ConcentrationModel.valid_domain` (outside
which it raises) and a calibration domain, in its own mass definition: an instance's
is :attr:`~ConcentrationModel.calibration` (it can depend on the instance's
parameters; see :class:`~hmf.core.domain.Calibrated`).

Implemented:

* :class:`Duffy08` (alias ``"Duffy08"``): the power laws of Duffy et al. (2008),
  for NFW profiles, with every parameter set of their Table 1.
"""

from __future__ import annotations

import abc
import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, ClassVar, Literal, get_args

import attrs
import numpy as np
import numpy.typing as npt

from . import _references as refs
from ._fields import field
from ._validators import one_of
from .domain import Calibrated, Domain, check_extent
from .mass_definitions import MassDefinition, SOCritical, SOMean, SOVirial
from .model import Model
from .units import Msun_h, unit_boundary

__all__ = ["DUFFY08_PARAMETERS", "ConcentrationModel", "Duffy08"]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@attrs.frozen(kw_only=True)
class ConcentrationModel(Calibrated, Model, kind=True):
    r"""The kind of concentration-mass relations: :math:`c(M, z)`.

    Subclasses implement :meth:`_ln_concentration_and_slope` (on checked plain arrays)
    and :attr:`mass_definition`, and declare both domains. The domains bound ``m``
    (in Msun/h, in the relation's own definition) and ``z``.
    """

    #: Where the relation can be evaluated: :meth:`ln_concentration_and_slope_kernel`
    #: raises a :class:`~hmf.core.domain.DomainError` outside it.
    valid_domain: ClassVar[Domain] = Domain(
        {"m": (0 * Msun_h, None, "(]"), "z": (0, None)}, source="m > 0, z >= 0."
    )

    @property
    @abc.abstractmethod
    def mass_definition(self) -> MassDefinition:
        """The mass definition the relation's masses and concentrations are in."""

    @abc.abstractmethod
    def _ln_concentration_and_slope(
        self, m: FloatArray, z: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Ln c and dln c / dln m, on checked plain arrays (m in Msun/h)."""

    def ln_concentration_and_slope_kernel(
        self, *, m: npt.ArrayLike, z: npt.ArrayLike
    ) -> tuple[FloatArray, FloatArray]:
        r"""Ln c(M, z) and its slope :math:`d\ln c/d\ln M`, at kernel level.

        Parameters
        ----------
        m
            Masses in Msun/h, in :attr:`mass_definition`: a plain array.
        z
            Redshifts, broadcast with ``m``.

        Returns
        -------
        ln_c, dln_c_dln_m : numpy.ndarray
            Dimensionless, with the broadcast shape.

        Raises
        ------
        DomainError
            If a mass or redshift is outside :attr:`valid_domain` (from their extent).
        """
        where = f"{type(self).__name__}.ln_concentration_and_slope_kernel"
        domain = type(self).valid_domain
        m = check_extent("m", m, domain, where=where)
        z = check_extent("z", z, domain, where=where)
        ln_c, slope = self._ln_concentration_and_slope(m, z)
        shape = np.broadcast_shapes(m.shape, z.shape)
        return (
            np.array(np.broadcast_to(ln_c, shape)),
            np.array(np.broadcast_to(slope, shape)),
        )

    def in_calibration_domain_kernel(self, *, m: npt.ArrayLike, z: npt.ArrayLike) -> BoolArray:
        """Whether (m, z) is inside :attr:`calibration`, at kernel level.

        Parameters
        ----------
        m
            Masses in Msun/h, in :attr:`mass_definition`.
        z
            Redshifts, broadcast with ``m``.

        Returns
        -------
        numpy.ndarray of bool
            With the broadcast shape; True everywhere without a calibration domain.

        Raises
        ------
        DomainError
            If a mass or redshift is outside :attr:`valid_domain` (from their extent).
        """
        where = f"{type(self).__name__}.in_calibration_domain_kernel"
        domain = type(self).valid_domain
        m = check_extent("m", m, domain, where=where)
        z = check_extent("z", z, domain, where=where)
        shape = np.broadcast_shapes(m.shape, z.shape)
        calibration = self.calibration
        if calibration is None:
            return np.ones(shape, dtype=bool)
        return np.array(
            np.broadcast_to(calibration.contains(m=m << Msun_h, z=z), shape), dtype=bool
        )

    @unit_boundary(m=Msun_h)
    def concentration(self, *, m: Any, z: npt.ArrayLike) -> FloatArray:
        """The median concentration c(M, z), in :attr:`mass_definition`.

        Parameters
        ----------
        m : Quantity
            Masses in Msun/h, in :attr:`mass_definition`.
        z : float or array_like
            Redshifts, broadcast with ``m``.

        Returns
        -------
        numpy.float64 or numpy.ndarray
            Dimensionless, with the broadcast shape.

        Raises
        ------
        DomainError
            Outside :attr:`valid_domain`.
        """
        return np.exp(self.ln_concentration_and_slope_kernel(m=m, z=z)[0])


# ---------------------------------------------------------------------------------
# Duffy et al. (2008)
# ---------------------------------------------------------------------------------

Duffy08Sample = Literal["full", "relaxed"]
Duffy08Definition = Literal["200c", "vir", "200m"]
Duffy08Redshifts = Literal["0-2", "0"]

#: The NFW parameters (A, B, C) of Duffy et al. (2008), Table 1 (arXiv:0804.2486 v4,
#: which corrects the published version), by (definition, sample, redshift range), for
#: :math:`c = A(M/M_{\rm pivot})^B(1+z)^C` with :math:`M_{\rm pivot} = 2\times10^{12}`
#: Msun/h. "200c" is their :math:`c_{200}` (relative to the critical density), "vir"
#: their :math:`c_{\rm vir}` (Bryan & Norman 1998) and "200m" their
#: :math:`c_{\rm mean}`; "full" is their default sample (F), "relaxed" the relaxed one
#: (R); the "0" sets are fitted at z = 0 alone, with C = 0.
DUFFY08_PARAMETERS: Mapping[tuple[str, str, str], tuple[float, float, float]] = MappingProxyType(
    {
        ("200c", "full", "0"): (5.74, -0.097, 0.0),
        ("200c", "full", "0-2"): (5.71, -0.084, -0.47),
        ("200c", "relaxed", "0"): (6.67, -0.092, 0.0),
        ("200c", "relaxed", "0-2"): (6.71, -0.091, -0.44),
        ("vir", "full", "0"): (7.96, -0.091, 0.0),
        ("vir", "full", "0-2"): (7.85, -0.081, -0.71),
        ("vir", "relaxed", "0"): (9.23, -0.089, 0.0),
        ("vir", "relaxed", "0-2"): (9.23, -0.090, -0.69),
        ("200m", "full", "0"): (10.39, -0.089, 0.0),
        ("200m", "full", "0-2"): (10.14, -0.081, -1.01),
        ("200m", "relaxed", "0"): (12.00, -0.087, 0.0),
        ("200m", "relaxed", "0-2"): (11.93, -0.090, -0.99),
    }
)

_DUFFY08_DEFINITIONS: Mapping[str, MassDefinition] = MappingProxyType(
    {"200c": SOCritical(overdensity=200), "vir": SOVirial(), "200m": SOMean(overdensity=200)}
)

_DUFFY08_SOURCE = (
    "Duffy et al. 2008 (arXiv:0804.2486 v4): haloes of 1e11-1e15 Msun/h (abstract; "
    "Sec. 3), in the relation's own mass definition; z = {z} (Table 1). One cosmology "
    "(WMAP5: Omega_m = 0.258, sigma_8 = 0.796; Sec. 2)."
)

#: The calibration domains of the Duffy08 sets, by redshift range.
_DUFFY08_DOMAINS: Mapping[str, Domain] = MappingProxyType(
    {
        "0-2": Domain(
            {"m": [1e11, 1e15] * Msun_h, "z": (0, 2)}, source=_DUFFY08_SOURCE.format(z="0-2")
        ),
        "0": Domain(
            {"m": [1e11, 1e15] * Msun_h, "z": (0, 0)}, source=_DUFFY08_SOURCE.format(z="0")
        ),
    }
)


@attrs.frozen(kw_only=True)
class Duffy08(ConcentrationModel, alias="Duffy08"):
    r"""The NFW concentration-mass relation of Duffy et al. (2008).

    .. math:: c = A\left(\frac{M}{2\times10^{12}\,h^{-1}M_\odot}\right)^B (1+z)^C,

    with (A, B, C) from their Table 1 (:data:`DUFFY08_PARAMETERS`), for the chosen
    mass definition, sample and redshift range. M and c are in ``definition``. It was
    fitted to haloes of :math:`10^{11}`-:math:`10^{15}` M☉/h, at z = 0-2 (or z = 0),
    in the WMAP5 cosmology. The full sample, the default, is the median of all
    haloes, which is what a mass function counts; the relaxed sample's
    concentrations are about 10% higher.
    """

    references: ClassVar[tuple[str, ...]] = (refs.DUFFY08,)
    parameter_source: ClassVar[str] = "Duffy et al. 2008 (arXiv:0804.2486 v4), Table 1, NFW."

    definition: Duffy08Definition = field(
        default="200c",
        validator=one_of(get_args(Duffy08Definition)),
        doc=(
            "The mass definition of the parameter set: '200c' (200 times the critical "
            "density), 'vir' (Bryan & Norman 1998) or '200m' (200 times the mean)."
        ),
    )
    sample: Duffy08Sample = field(
        default="full",
        validator=one_of(get_args(Duffy08Sample)),
        doc="The halo sample: 'full' (all haloes) or 'relaxed'.",
    )
    redshifts: Duffy08Redshifts = field(
        default="0-2",
        validator=one_of(get_args(Duffy08Redshifts)),
        doc="The redshift range the set was fitted to: '0-2', or '0' (with C = 0).",
    )

    #: The calibration domain of the default set (z = 0-2): ask an instance for
    #: :attr:`calibration`, that of its set.
    calibration_domain: ClassVar[Domain | None] = _DUFFY08_DOMAINS["0-2"]

    #: The pivot mass, in Msun/h.
    m_pivot: ClassVar[float] = 2e12

    @property
    def parameters(self) -> tuple[float, float, float]:
        """The parameters (A, B, C) of the chosen set."""
        return DUFFY08_PARAMETERS[self.definition, self.sample, self.redshifts]

    @property
    def mass_definition(self) -> MassDefinition:
        """SOCritical(200), SOVirial() or SOMean(200), from ``definition``."""
        return _DUFFY08_DEFINITIONS[self.definition]

    @property
    def calibration(self) -> Domain:
        """1e11-1e15 Msun/h (in ``definition``), and z = 0-2 (or z = 0)."""
        return _DUFFY08_DOMAINS[self.redshifts]

    def _ln_concentration_and_slope(
        self, m: FloatArray, z: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        a, b, c = self.parameters
        ln_c: FloatArray = math.log(a) + b * np.log(m / self.m_pivot) + c * np.log1p(z)
        return ln_c, np.full_like(ln_c, b)
