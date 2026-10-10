"""Analytic power spectra for the MassVariance tests, as PowerSource implementations.

These are deliberately written here, independently of hmf (v3 or v4), so that tests
against them are physical tests, not comparisons of the code with itself.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import cached_property
from types import MappingProxyType

import attrs
import numpy as np

from hmf.core.species import MATTER_SPECIES
from hmf.core.transfer_models import TransferModel, TransferSolution
from hmf.core.units import H0_unit, UnitContext

#: The critical density today, in Msun h^2 / Mpc^3.
RHO_CRIT0 = 2.775366e11


@attrs.frozen
class AnalyticPower:
    """A PowerSource from a function of k (h/Mpc) returning P ((Mpc/h)^3)."""

    function: Callable[[np.ndarray], np.ndarray]
    rho_mean0: float = 0.3 * RHO_CRIT0
    H0: float = 70.0
    #: No user table: an analytic function is never extrapolated.
    table_range = None

    def ln_power_kernel(self, ln_k):
        # A non-positive power has no log: it gives -inf or NaN, which MassVariance rejects.
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(self.function(np.exp(np.asarray(ln_k, dtype=float))))

    @cached_property
    def _unit_context(self) -> UnitContext:
        return UnitContext(self.H0 * H0_unit)


@attrs.frozen
class PowerLaw:
    """P(k) = amplitude * k**n."""

    n: float
    amplitude: float = 1.0

    def __call__(self, k):
        return self.amplitude * k**self.n


@attrs.frozen
class EisensteinHuNoWiggle:
    """Eisenstein & Hu (1998) no-wiggle transfer function times k**n_s (eqs. 26-31)."""

    omega_m: float = 0.3
    omega_b: float = 0.045
    h: float = 0.7
    n_s: float = 0.965
    t_cmb: float = 2.7255

    def transfer(self, k):
        om, ob, h = self.omega_m, self.omega_b, self.h
        theta = self.t_cmb / 2.7
        omh2, obh2 = om * h**2, ob * h**2
        fb = ob / om
        s = 44.5 * math.log(9.83 / omh2) / math.sqrt(1 + 10 * obh2**0.75)  # Mpc
        alpha = 1 - 0.328 * math.log(431 * omh2) * fb + 0.38 * math.log(22.3 * omh2) * fb**2
        kh = k * h  # 1/Mpc
        gamma_eff = om * h * (alpha + (1 - alpha) / (1 + (0.43 * kh * s) ** 4))
        q = k * theta**2 / gamma_eff
        l0 = np.log(2 * math.e + 1.8 * q)
        c0 = 14.2 + 731 / (1 + 62.5 * q)
        return l0 / (l0 + c0 * q**2)

    def __call__(self, k):
        return k**self.n_s * self.transfer(k) ** 2


@attrs.frozen
class WithBAO:
    """A smooth spectrum times a damped BAO wiggle, 1 + A sin(k s) / (k s) exp(-(k/k_d)^2)."""

    smooth: Callable[[np.ndarray], np.ndarray] = EisensteinHuNoWiggle()
    amplitude: float = 0.5
    s: float = 105.0  # Mpc/h
    k_damp: float = 0.2  # h/Mpc

    def __call__(self, k):
        x = k * self.s
        wiggle = self.amplitude * np.sinc(x / math.pi) * np.exp(-((k / self.k_damp) ** 2))
        return self.smooth(k) * (1 + wiggle)


@attrs.frozen
class WDMTruncated:
    """A smooth spectrum truncated as by warm dark matter (Bode et al. 2001 form)."""

    smooth: Callable[[np.ndarray], np.ndarray] = EisensteinHuNoWiggle()
    alpha: float = 0.05  # Mpc/h
    nu: float = 1.12

    def __call__(self, k):
        return self.smooth(k) * (1 + (self.alpha * k) ** (2 * self.nu)) ** (-10 / self.nu)


@attrs.frozen(kw_only=True)
class UnitTransfer(TransferModel, abstract=True):
    """T(k) = 1 for every species, so a Transfer stage's unnormalised power is k**n_s.

    A pure power law, for closed-form tests of the stages built on Transfer. It is not
    registered (``abstract=True``), so the registries other tests check are unchanged;
    stages take it as an instance.
    """

    def solve(self, cosmology, k_accuracy, *, disk_cache=None):
        def ln_t(k):
            return np.zeros_like(np.asarray(k, dtype=float))

        return TransferSolution(MappingProxyType(dict.fromkeys(MATTER_SPECIES, ln_t)))
