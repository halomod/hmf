r"""Halo density profiles, as :mod:`hmf.core` models: what a mass conversion needs.

A spherical density profile :math:`\rho(r) = \rho_s f(r/r_s)`, with scale radius
:math:`r_s`, encloses within :math:`r = x r_s` the mass
:math:`4\pi\rho_s r_s^3\,\mu(x)`, with :math:`\mu(x) = \int_0^x f(y)y^2\,dy`. That is
all a conversion between spherical-overdensity mass definitions needs from it
(:class:`~hmf.core.mass_conversion.ProfileConversion`): a halo of concentration
:math:`c = r_\Delta/r_s` in one definition has the concentration :math:`c'` with
:math:`F(c') = qF(c)` in a definition of q times the halo density, with
:math:`F = \mu/x^3`, and the mass ratio :math:`\mu(c')/\mu(c)` (see
:mod:`hmf.core._kernels.mass_conversion`).

So a :class:`DensityProfile` gives :math:`\ln\mu(x)` and its slope
:math:`\psi(x) = d\ln\mu/d\ln x` (:meth:`DensityProfile.ln_mu_and_slope_kernel`), and
solves for :math:`c'`, with the Jacobian
(:meth:`DensityProfile.convert_concentration_kernel`). Its density must fall outwards
(:math:`\psi < 3`), so that the root is unique. It is a model kind
(``DensityProfile.get("NFW")``), so other packages (e.g. halomod, which adds Fourier
transforms and truncations to its profiles) subclass these profiles rather than
defining their own. Implemented:

* :class:`NFW` (alias ``"NFW"``): Navarro, Frenk & White (1997).

A new profile implements :meth:`DensityProfile._ln_mu_and_slope` (on plain arrays,
unchecked), and may give a better starting point for the solve than the default
(:meth:`DensityProfile._start`, the local power law of F) and fewer Newton steps
(:attr:`DensityProfile.newton_steps`).
"""

from __future__ import annotations

import abc
from typing import ClassVar

import attrs
import numpy as np
import numpy.typing as npt

from . import _references as refs
from ._kernels import mass_conversion as _k
from .domain import Domain, Interval, check_extent
from .model import Model
from .units import unit_boundary

__all__ = ["NFW", "DensityProfile"]

FloatArray = npt.NDArray[np.float64]

_POSITIVE = Interval(0, None, lower_open=True)


@attrs.frozen(kw_only=True)
class DensityProfile(Model, kind=True):
    r"""The kind of halo density profiles: their enclosed mass :math:`\mu(x)`.

    Subclasses implement :meth:`_ln_mu_and_slope`. See the module documentation.
    """

    #: Where the profile can be evaluated: :math:`x = r/r_s > 0`.
    valid_domain: ClassVar[Domain] = Domain({"x": (0, None, "(]")}, source="x = r / r_s > 0.")

    #: Profiles are not calibrated against simulations (their concentrations are).
    calibration_domain: ClassVar[Domain | None] = None

    #: The Newton steps of :meth:`convert_concentration_kernel` from :meth:`_start`.
    #: Eight reach round-off from the default start for 1e-2 < c < 1e3 and
    #: 1e-3 < q < 1e3 for the profiles tested; a profile with a better start sets fewer.
    newton_steps: ClassVar[int] = 8

    @abc.abstractmethod
    def _ln_mu_and_slope(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        r""":math:`\ln\mu(x)` and :math:`\psi(x)`, on plain arrays (unchecked, x > 0)."""

    def _start(self, c: FloatArray, ln_q: FloatArray) -> FloatArray:
        """The solve's start: F as its local power law at c (``power_law_start``)."""
        _, psi = self._ln_mu_and_slope(c)
        return _k.power_law_start(c, ln_q, psi)

    def ln_mu_and_slope_kernel(self, x: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
        r""":math:`\ln\mu(x)` and :math:`\psi(x) = d\ln\mu/d\ln x`, at kernel level.

        Parameters
        ----------
        x
            :math:`r/r_s`, dimensionless: a plain array.

        Returns
        -------
        ln_mu, psi : numpy.ndarray
            With the shape of ``x``.

        Raises
        ------
        DomainError
            If an x is not finite and > 0.
        """
        x = check_extent(
            "x", x, type(self).valid_domain, where=f"{type(self).__name__}.ln_mu_and_slope_kernel"
        )
        return self._ln_mu_and_slope(x)

    def convert_concentration_kernel(
        self, c: npt.ArrayLike, *, slope: npt.ArrayLike, ln_q: npt.ArrayLike
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        r"""Convert haloes of concentration c to a definition of q times the halo density.

        :func:`~hmf.core._kernels.mass_conversion.profile_convert` with this profile:
        the new concentration :math:`c'` (:math:`F(c') = qF(c)`), the mass ratio and
        the analytic Jacobian, elementwise.

        Parameters
        ----------
        c
            The concentrations in the old definition (> 0): a plain array.
        slope
            :math:`d\ln c/d\ln M` of the concentrations, broadcast with ``c``.
        ln_q
            :math:`\ln(\rho_{\rm new}/\rho_{\rm old})`, broadcast with ``c``.

        Returns
        -------
        ln_mass_ratio, c_new, dln_m_new_dln_m, dln_c_new_dln_m : numpy.ndarray
            :math:`\ln(M'/M)`, :math:`c'`, :math:`d\ln M'/d\ln M` and
            :math:`d\ln c'/d\ln M`, with the broadcast shape.

        Raises
        ------
        DomainError
            If a concentration is not finite and > 0, or a slope or ``ln_q`` is not
            finite.
        """
        where = f"{type(self).__name__}.convert_concentration_kernel"
        c = check_extent("c", c, _POSITIVE, where=where)
        slope = check_extent("slope", slope, where=where)
        ln_q = check_extent("ln_q", ln_q, where=where)
        c_start = self._start(*np.broadcast_arrays(c, ln_q))
        return _k.profile_convert(
            self._ln_mu_and_slope, c, slope, ln_q, c_start, type(self).newton_steps
        )

    @unit_boundary()
    def mu(self, x: npt.ArrayLike) -> FloatArray:
        r"""The enclosed mass in units of :math:`4\pi\rho_s r_s^3`, :math:`\mu(x)`.

        Parameters
        ----------
        x : float or array_like
            :math:`r/r_s` (> 0).

        Returns
        -------
        numpy.float64 or numpy.ndarray
            :math:`\mu(x)`.

        Raises
        ------
        DomainError
            If an x is not finite and > 0.
        """
        return np.exp(self.ln_mu_and_slope_kernel(x)[0])


@attrs.frozen(kw_only=True)
class NFW(DensityProfile, alias="NFW"):
    r"""The Navarro, Frenk & White (1997) profile, :math:`f(x) = 1/[x(1+x)^2]`.

    :math:`\mu(x) = \ln(1+x) - x/(1+x)`. The solve starts from the fit of Hu &
    Kravtsov (2003, eq. C11), so four Newton steps reach round-off (about
    :math:`10^{-13}` in c).
    """

    references: ClassVar[tuple[str, ...]] = (refs.NFW97, refs.HU03)
    parameter_source: ClassVar[str] = "Navarro, Frenk & White 1997; no parameters."
    newton_steps: ClassVar[int] = _k.NEWTON_STEPS

    def _ln_mu_and_slope(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        return _k.nfw_ln_mu_and_slope(x)

    def _start(self, c: FloatArray, ln_q: FloatArray) -> FloatArray:
        """Hu & Kravtsov (2003), eq. C11."""
        return _k.nfw_start(c, ln_q)
