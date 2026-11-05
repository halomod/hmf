r"""Converting halo masses between mass definitions, as :mod:`hmf.core` models.

A :class:`MassConversion` maps the masses M of haloes in one definition (the
*source*, :mod:`hmf.core.mass_definitions`) to their masses M' in another (the
*target*), and gives the Jacobian dM'/dM, which a mass function needs to conserve the
number of haloes:

.. math:: \frac{dn}{dM}(M) = \frac{dn'}{dM'}\big(M'(M)\big)\frac{dM'}{dM}.

It is a model kind (``MassConversion.get("NFW")``), so other packages can add their
own (e.g. for an Einasto profile) by subclassing it and implementing
:meth:`MassConversion.convert_kernel`.

The NFW conversion
------------------
:class:`NFWConversion` assumes NFW haloes (Navarro, Frenk & White 1997) with the
median concentration of a :class:`~hmf.core.concentration.ConcentrationModel`
(default :class:`~hmf.core.concentration.Duffy08`). For masses M in the source
definition A at redshift z:

1. **The concentration in A**, :math:`c_A(M)`. If the concentration model is
   calibrated in A, it is the model's. Otherwise (it is calibrated in C), each halo
   is the one whose mass :math:`M_C` in C, with the concentration :math:`c_C(M_C)`,
   converts to M in A: :math:`M_C` is found by Newton's method in
   :math:`\ln M_C` (:data:`OUTER_NEWTON_STEPS` steps, from :math:`M_C = M`), and
   :math:`c_A` follows from :math:`c_C`, as in step 2.
2. **The concentration in B**, :math:`c_B`: the root of
   :math:`F(c_B) = (\rho_B/\rho_A)F(c_A)`, where :math:`F(x) = \mu(x)/x^3` is the
   mean enclosed density in units of :math:`3\rho_s`,
   :math:`\mu(x) = \ln(1+x) - x/(1+x)`, and :math:`\rho_A, \rho_B` the halo
   densities of the two definitions at z (their ratio is that of their
   overdensities relative to the mean, :math:`\Delta_{m,B}/\Delta_{m,A}`). It is
   solved for every mass at once, by Newton's method from the approximation of
   Hu & Kravtsov (2003, eq. C11), to round-off (about :math:`10^{-13}`; see
   :mod:`hmf.core._kernels.mass_conversion`).
3. **The mass in B**, :math:`M' = M\mu(c_B)/\mu(c_A)`, and the Jacobian, by implicit
   differentiation of the root through :math:`c_A(M)`, with
   :math:`s_A = d\ln c_A/d\ln M`, :math:`\psi = d\ln\mu/d\ln x` and
   :math:`\phi = \psi - 3`:

   .. math:: \frac{d\ln M'}{d\ln M} = 1 + s_A\left[\psi(c_B)\frac{\phi(c_A)}{\phi(c_B)}
             - \psi(c_A)\right], \qquad \frac{dM'}{dM} = \frac{M'}{M}\frac{d\ln M'}{d\ln M}.

   In step 1, :math:`s_A` follows from the model's slope :math:`s_C` the same way.

Every step is elementwise, so the result does not depend on the batch size. Because
every mass is that of one halo (one :math:`\rho_s, r_s`), converting A → B → A
returns the masses to round-off, and a definition with the same overdensity (e.g.
:class:`~hmf.core.mass_definitions.SOVirial` and ``SOMean(18 pi^2)`` in
Einstein-de Sitter) converts to itself exactly.

Definitions with an overdensity convert: the SO definitions and
:class:`~hmf.core.mass_definitions.FOF`, through its approximate overdensity (an
FoF group is not an NFW sphere). :class:`~hmf.core.mass_definitions.SelfBound` has
none, and an :class:`~hmf.core.mass_definitions.SOGeneric` is not a definition masses
are in: both raise ``ValueError``.

Domains
-------
The conversion raises a :class:`~hmf.core.domain.DomainError` for masses, redshifts
or :math:`\Omega_m(z)` that are not finite and positive, and outside the
concentration model's valid domain (at the masses it is evaluated at, in its own
definition). The concentration model's calibration domain is reported, not enforced:
:attr:`ConvertedMasses.in_calibration_domain`, which a
:class:`~hmf.core.mass_function.MassFunction` applies its ``domain_policy`` to.
"""

from __future__ import annotations

import abc
from typing import Any, ClassVar

import attrs
import numpy as np
import numpy.typing as npt

from . import _references as refs
from ._fields import field
from ._kernels import mass_conversion as _k
from .concentration import ConcentrationModel, Duffy08
from .domain import Interval, check_extent
from .mass_definitions import MassDefinition, SelfBound, SOGeneric
from .model import Model
from .units import Msun_h, unit_boundary

__all__ = ["OUTER_NEWTON_STEPS", "ConvertedMasses", "MassConversion", "NFWConversion"]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

#: The Newton steps in ln M_C that find the mass in the concentration model's own
#: definition (step 1 of the NFW conversion). With concentration slopes |s| ~ 0.1
#: each step multiplies the error by about its square, so three reach round-off
#: from M_C = M; the fourth is a margin.
OUTER_NEWTON_STEPS = 4

_POSITIVE = Interval(0, None, lower_open=True)
_Z = Interval(0, None)


@attrs.frozen(kw_only=True, eq=False)
class ConvertedMasses:
    """The result of a mass conversion (plain arrays; see :meth:`MassConversion.convert_kernel`).

    Parameters
    ----------
    m
        The masses in the target definition, in Msun/h.
    dm_dm
        The Jacobian dM'/dM, dimensionless.
    in_calibration_domain
        Whether each mass is inside the calibration domain of the conversion's
        ingredients (e.g. its concentration model).
    calibration_description
        Names that domain, for messages (empty if there is none).
    """

    m: FloatArray
    dm_dm: FloatArray
    in_calibration_domain: BoolArray
    calibration_description: str = ""


@attrs.frozen(kw_only=True)
class MassConversion(Model, kind=True):
    """The kind of conversions of halo masses between mass definitions.

    Subclasses implement :meth:`check_definitions` and :meth:`convert_kernel`.
    """

    @abc.abstractmethod
    def check_definitions(self, source: MassDefinition, target: MassDefinition) -> None:
        """Raise ``ValueError`` if masses can't be converted from ``source`` to ``target``.

        Parameters
        ----------
        source, target
            The mass definitions.

        Raises
        ------
        ValueError
            If the conversion is not possible.
        """

    @abc.abstractmethod
    def convert_kernel(
        self,
        m: npt.ArrayLike,
        *,
        z: npt.ArrayLike,
        omega_m_z: npt.ArrayLike,
        source: MassDefinition,
        target: MassDefinition,
    ) -> ConvertedMasses:
        r"""Convert masses from ``source`` to ``target`` at z, at kernel level.

        Parameters
        ----------
        m
            Masses in Msun/h, in ``source``: a plain array.
        z
            Redshifts, broadcast with ``m``.
        omega_m_z
            :math:`\Omega_m(z)` of CDM + baryons at ``z``, broadcast with ``m``.
        source, target
            The mass definitions.

        Returns
        -------
        ConvertedMasses
            The masses in ``target`` and dM'/dM, with the broadcast shape.

        Raises
        ------
        DomainError
            If an input is outside the conversion's valid domain.
        ValueError
            As :meth:`check_definitions`.
        """

    @unit_boundary(m=Msun_h, returns=(Msun_h, None))
    def convert(
        self,
        m: Any,
        *,
        z: npt.ArrayLike,
        omega_m_z: npt.ArrayLike,
        source: MassDefinition,
        target: MassDefinition,
    ) -> tuple[FloatArray, FloatArray]:
        r"""Convert masses from ``source`` to ``target`` (see :meth:`convert_kernel`).

        Parameters
        ----------
        m : Quantity
            Masses in Msun/h, in ``source``.
        z : float or array_like
            Redshifts, broadcast with ``m``.
        omega_m_z : float or array_like
            :math:`\Omega_m(z)` of CDM + baryons (e.g.
            :func:`hmf.core.species.omega_m`).
        source, target : MassDefinition
            The mass definitions.

        Returns
        -------
        m_new : Quantity
            The masses in ``target``, in Msun/h.
        dm_new_dm : numpy.ndarray
            The Jacobian dM'/dM.
        """
        out = self.convert_kernel(m, z=z, omega_m_z=omega_m_z, source=source, target=target)
        return out.m, out.dm_dm


def _check_convertible(definition: MassDefinition, role: str) -> None:
    if isinstance(definition, SOGeneric):
        raise ValueError(  # noqa: TRY004 (a configuration error: see hmf.core.domain)
            f"Can't convert masses {role} {definition}: an SOGeneric is not a definition "
            "masses are in; use its preferred one, or another SO definition."
        )
    if isinstance(definition, SelfBound):
        raise ValueError(  # noqa: TRY004 (a configuration error: see hmf.core.domain)
            f"Can't convert masses {role} {definition}: self-bound masses have no "
            "overdensity, so no NFW radius."
        )


@attrs.frozen(kw_only=True)
class NFWConversion(MassConversion, alias="NFW"):
    """Mass conversion for NFW haloes with a concentration-mass relation.

    See "The NFW conversion" in the module documentation. A profile other than NFW
    (e.g. Einasto) is another :class:`MassConversion`.
    """

    references: ClassVar[tuple[str, ...]] = (refs.NFW97, refs.HU03)

    concentration: ConcentrationModel = field(
        factory=Duffy08,
        converter=ConcentrationModel.coerce,
        doc=(
            "The concentration-mass relation: an instance, a class or a registered name "
            "(default Duffy08())."
        ),
    )

    def check_definitions(self, source: MassDefinition, target: MassDefinition) -> None:
        """Raise ``ValueError`` for an SOGeneric or SelfBound definition.

        Parameters
        ----------
        source, target
            The mass definitions.

        Raises
        ------
        ValueError
            If either has no overdensity.
        """
        _check_convertible(source, "from")
        _check_convertible(target, "to")
        _check_convertible(self.concentration.mass_definition, "with a concentration in")

    def _concentration_in(
        self, m: FloatArray, z: FloatArray, omega_m_z: FloatArray, definition: MassDefinition
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """c(M) and dln c/dln M in ``definition``, and the masses c was evaluated at.

        Step 1 of the NFW conversion.
        """
        model = self.concentration
        own = model.mass_definition
        if own == definition:
            ln_c, slope = model.ln_concentration_and_slope_kernel(m, z=z)
            return np.exp(ln_c), slope, np.array(np.broadcast_to(m, ln_c.shape))
        ln_q = np.log(definition.delta_halo_mean_kernel(omega_m_z)) - np.log(
            own.delta_halo_mean_kernel(omega_m_z)
        )
        ln_m = np.log(m)
        ln_m_own = np.array(np.broadcast_to(ln_m, np.broadcast_shapes(ln_m.shape, ln_q.shape)))
        for _ in range(OUTER_NEWTON_STEPS):
            ln_c, slope = model.ln_concentration_and_slope_kernel(np.exp(ln_m_own), z=z)
            ln_ratio, _, dln_m, _ = _k.nfw_convert(np.exp(ln_c), slope, ln_q)
            ln_m_own = ln_m_own - (ln_m_own + ln_ratio - ln_m) / dln_m
        m_own = np.exp(ln_m_own)
        ln_c, slope = model.ln_concentration_and_slope_kernel(m_own, z=z)
        _, c_new, dln_m, dln_c = _k.nfw_convert(np.exp(ln_c), slope, ln_q)
        return c_new, dln_c / dln_m, m_own

    def convert_kernel(
        self,
        m: npt.ArrayLike,
        *,
        z: npt.ArrayLike,
        omega_m_z: npt.ArrayLike,
        source: MassDefinition,
        target: MassDefinition,
    ) -> ConvertedMasses:
        r"""Convert masses from ``source`` to ``target`` at z, at kernel level.

        See "The NFW conversion" in the module documentation.

        Parameters
        ----------
        m
            Masses in Msun/h, in ``source``: a plain array.
        z
            Redshifts, broadcast with ``m``.
        omega_m_z
            :math:`\Omega_m(z)` of CDM + baryons at ``z``, broadcast with ``m``.
        source, target
            The mass definitions.

        Returns
        -------
        ConvertedMasses
            The masses in ``target`` and dM'/dM, with the broadcast shape, and whether
            the concentration model was evaluated inside its calibration domain.

        Raises
        ------
        DomainError
            If a mass, redshift or :math:`\Omega_m(z)` is not finite and positive (z
            may be 0), or outside the concentration model's valid domain.
        ValueError
            As :meth:`check_definitions`.
        """
        where = f"NFWConversion.convert_kernel ({source} -> {target})"
        m = check_extent("m", m, _POSITIVE, where=where)
        z = check_extent("z", z, _Z, where=where)
        omega_m_z = check_extent("omega_m_z", omega_m_z, _POSITIVE, where=where)
        shape = np.broadcast_shapes(m.shape, z.shape, omega_m_z.shape)
        if source == target:
            return ConvertedMasses(
                m=np.array(np.broadcast_to(m, shape)),
                dm_dm=np.ones(shape),
                in_calibration_domain=np.ones(shape, dtype=bool),
            )
        self.check_definitions(source, target)
        c_source, slope, m_concentration = self._concentration_in(m, z, omega_m_z, source)
        ln_q = np.log(target.delta_halo_mean_kernel(omega_m_z)) - np.log(
            source.delta_halo_mean_kernel(omega_m_z)
        )
        ln_ratio, _, dln_m, _ = _k.nfw_convert(c_source, slope, ln_q)
        ratio = np.exp(ln_ratio)
        model = self.concentration
        calibration = model.calibration
        return ConvertedMasses(
            m=np.array(np.broadcast_to(m * ratio, shape)),
            dm_dm=np.array(np.broadcast_to(ratio * dln_m, shape)),
            in_calibration_domain=np.array(
                np.broadcast_to(model.in_calibration_domain_kernel(m_concentration, z=z), shape)
            ),
            calibration_description=(
                ""
                if calibration is None
                else f"{type(model).__name__}'s calibration domain ({calibration.describe()}, "
                f"in {model.mass_definition})"
            ),
        )
