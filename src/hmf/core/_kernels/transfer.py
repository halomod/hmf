"""Kernels of the transfer function: fitting formulae and tabulated transfer functions.

Every wavenumber here is in h/Mpc (the canonical unit), as a plain array, unless its
name says otherwise (``k_mpc`` is in 1/Mpc). Transfer functions are dimensionless
and normalised to 1 as k -> 0. Logarithms are natural (``ln_``).

The fitting formulae (:func:`ln_t_eh98`, :func:`ln_t_eh98_no_wiggle`,
:func:`ln_t_bbks`, :func:`ln_t_bond_efs`) take their cosmological input as plain
floats, and their scales (sound horizon, k_eq, ...) from :func:`eh98_scales`.

A transfer function tabulated by a Boltzmann code (or given by the user) is
interpolated by :class:`TabulatedTransfer`, which also extrapolates it beyond the
table on both sides: see its documentation.
"""

from __future__ import annotations

from typing import NamedTuple

import attrs
import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline, PPoly

__all__ = [
    "EH98Scales",
    "TabulatedTransfer",
    "eh98_scales",
    "ln_t_bbks",
    "ln_t_bond_efs",
    "ln_t_eh98",
    "ln_t_eh98_no_wiggle",
    "tabulate_transfer",
]

Array = npt.NDArray[np.float64]


class EH98Scales(NamedTuple):
    """The scales of the Eisenstein & Hu (1998) fit, from :func:`eh98_scales`.

    Lengths are in Mpc and wavenumbers in 1/Mpc (no h), as in the paper.
    """

    h: float
    omh2: float
    obh2: float
    f_baryon: float
    theta_cmb: float
    z_eq: float
    k_eq: float
    z_drag: float
    r_drag: float
    r_eq: float
    sound_horizon: float
    sound_horizon_fit: float
    k_silk: float
    alpha_c: float
    beta_c: float
    alpha_b: float
    beta_b: float
    beta_node: float
    alpha_gamma: float


def eh98_scales(*, h: float, omega_m: float, omega_b: float, t_cmb: float) -> EH98Scales:
    """Compute the scales of the Eisenstein & Hu (1998) transfer function.

    Parameters
    ----------
    h
        The dimensionless Hubble parameter.
    omega_m
        The density of matter that clusters (CDM + baryons) today, in units of the
        critical density (astropy's ``Om0``).
    omega_b
        The density of baryons today, in units of the critical density.
    t_cmb
        The CMB temperature today, in K.

    Returns
    -------
    EH98Scales
        The equations of EH98 they come from are noted in the source.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return _eh98_scales(
            np.float64(h), np.float64(omega_m), np.float64(omega_b), np.float64(t_cmb)
        )


def _eh98_scales(
    h: np.float64, omega_m: np.float64, omega_b: np.float64, t_cmb: np.float64
) -> EH98Scales:
    """:func:`eh98_scales` on numpy floats (so that Ob0 = 0 gives NaN, not an error).

    Without baryons the scales of the BAO (sound horizon, ...) are NaN, but those of
    the no-wiggle fit are well defined.
    """
    omh2 = omega_m * h**2
    obh2 = omega_b * h**2
    f_baryon = omega_b / omega_m
    theta = t_cmb / 2.7

    z_eq = 2.50e4 * omh2 * theta**-4  # Eq. 2 (this is really 1 + z_eq)
    k_eq = 7.46e-2 * omh2 * theta**-2  # Eq. 3, 1/Mpc

    b1 = 0.313 * omh2**-0.419 * (1 + 0.607 * omh2**0.674)  # Eq. 4
    b2 = 0.238 * omh2**0.223
    z_drag = 1291 * omh2**0.251 / (1 + 0.659 * omh2**0.828) * (1 + b1 * obh2**b2)

    r_drag = 31.5 * obh2 * theta**-4 * (1000 / (1 + z_drag))  # Eq. 5
    r_eq = 31.5 * obh2 * theta**-4 * (1000 / z_eq)

    # Eq. 6, Mpc
    sound_horizon = (
        2
        / (3 * k_eq)
        * np.sqrt(6 / r_eq)
        * np.log((np.sqrt(1 + r_drag) + np.sqrt(r_drag + r_eq)) / (1 + np.sqrt(r_eq)))
    )
    # Eq. 26, Mpc
    sound_horizon_fit = 44.5 * np.log(9.83 / omh2) / np.sqrt(1 + 10 * obh2**0.75)

    k_silk = 1.6 * obh2**0.52 * omh2**0.73 * (1 + (10.4 * omh2) ** -0.95)  # Eq. 7, 1/Mpc

    a1 = (46.9 * omh2) ** 0.670 * (1 + (32.1 * omh2) ** -0.532)  # Eq. 11
    a2 = (12.0 * omh2) ** 0.424 * (1 + (45.0 * omh2) ** -0.582)
    alpha_c = a1**-f_baryon * a2 ** (-(f_baryon**3))

    bb1 = 0.944 / (1 + (458 * omh2) ** -0.708)  # Eq. 12
    bb2 = (0.395 * omh2) ** -0.0266
    beta_c = 1 / (1 + bb1 * ((1 - f_baryon) ** bb2 - 1))

    y = z_eq / (1 + z_drag)  # Eqs. 14-15
    g_y = y * (
        -6 * np.sqrt(1 + y) + (2 + 3 * y) * np.log((np.sqrt(1 + y) + 1) / (np.sqrt(1 + y) - 1))
    )
    alpha_b = 2.07 * k_eq * sound_horizon * (1 + r_drag) ** -0.75 * g_y

    beta_node = 8.41 * omh2**0.435  # Eq. 23
    beta_b = 0.5 + f_baryon + (3 - 2 * f_baryon) * np.sqrt((17.2 * omh2) ** 2 + 1)  # Eq. 24

    alpha_gamma = (  # Eq. 31
        1 - 0.328 * np.log(431 * omh2) * f_baryon + 0.38 * np.log(22.3 * omh2) * f_baryon**2
    )

    return EH98Scales(
        h=float(h),
        omh2=float(omh2),
        obh2=float(obh2),
        f_baryon=float(f_baryon),
        theta_cmb=float(theta),
        z_eq=float(z_eq),
        k_eq=float(k_eq),
        z_drag=float(z_drag),
        r_drag=float(r_drag),
        r_eq=float(r_eq),
        sound_horizon=float(sound_horizon),
        sound_horizon_fit=float(sound_horizon_fit),
        k_silk=float(k_silk),
        alpha_c=float(alpha_c),
        beta_c=float(beta_c),
        alpha_b=float(alpha_b),
        beta_b=float(beta_b),
        beta_node=float(beta_node),
        alpha_gamma=float(alpha_gamma),
    )


def ln_t_eh98(k: Array, scales: EH98Scales) -> Array:
    """The Eisenstein & Hu (1998) transfer function, with baryon acoustic oscillations.

    Parameters
    ----------
    k
        Wavenumbers, in h/Mpc.
    scales
        From :func:`eh98_scales`.

    Returns
    -------
    numpy.ndarray
        ln T(k): EH98 Eqs. 16-24.
    """
    s = scales
    k_mpc = np.asarray(k, dtype=float) * s.h
    q = k_mpc / (13.41 * s.k_eq)  # Eq. 10
    ks = k_mpc * s.sound_horizon

    def t0_tilde(alpha: float, beta: float) -> Array:  # Eqs. 19-20
        ln_part = np.log(np.e + 1.8 * beta * q)
        c = 14.2 / alpha + 386 / (1 + 69.9 * q**1.08)
        out: Array = ln_part / (ln_part + c * q**2)
        return out

    f = 1 / (1 + (ks / 5.4) ** 4)  # Eq. 18
    t_c = f * t0_tilde(1, s.beta_c) + (1 - f) * t0_tilde(s.alpha_c, s.beta_c)  # Eq. 17

    s_tilde = s.sound_horizon / (1 + (s.beta_node / ks) ** 3) ** (1 / 3)  # Eq. 22
    ks_tilde = k_mpc * s_tilde
    t_b = (  # Eq. 21
        t0_tilde(1, 1) / (1 + (ks / 5.2) ** 2)
        + s.alpha_b / (1 + (s.beta_b / ks) ** 3) * np.exp(-((k_mpc / s.k_silk) ** 1.4))
    ) * np.sinc(ks_tilde / np.pi)

    out: Array = np.log(s.f_baryon * t_b + (1 - s.f_baryon) * t_c)  # Eq. 16
    return out


def ln_t_eh98_no_wiggle(k: Array, scales: EH98Scales) -> Array:
    """The Eisenstein & Hu (1998) "no-wiggle" transfer function, without BAO.

    Parameters
    ----------
    k
        Wavenumbers, in h/Mpc.
    scales
        From :func:`eh98_scales`.

    Returns
    -------
    numpy.ndarray
        ln T(k): EH98 Eqs. 26-31, with the shape parameter of Eq. 30 and q of Eq. 28.
    """
    s = scales
    k_mpc = np.asarray(k, dtype=float) * s.h
    ks = k_mpc * s.sound_horizon_fit
    gamma_eff_h = s.omh2 * (s.alpha_gamma + (1 - s.alpha_gamma) / (1 + (0.43 * ks) ** 4))
    q = k_mpc * s.theta_cmb**2 / gamma_eff_h  # Eq. 28, with k/h in h/Mpc and Gamma h
    l0 = np.log(2 * np.e + 1.8 * q)  # Eq. 29
    c0 = 14.2 + 731 / (1 + 62.5 * q)
    out: Array = np.log(l0 / (l0 + c0 * q**2))
    return out


def ln_t_bbks(k: Array, *, gamma: float, a: float, b: float, c: float, d: float, e: float) -> Array:
    """The Bardeen et al. (1986) transfer function (their Eq. G3).

    Parameters
    ----------
    k
        Wavenumbers, in h/Mpc.
    gamma
        The shape parameter, Gamma (in h/Mpc), so that q = k / Gamma.
    a, b, c, d, e
        The coefficients of the fit.

    Returns
    -------
    numpy.ndarray
        ln T(k). It uses ``log1p``, so it is accurate as k -> 0, where T -> 1.
    """
    q = np.asarray(k, dtype=float) / gamma
    aq = a * q
    # ln(1 + aq) / (aq) -> 1 as q -> 0; np.log1p keeps it accurate there.
    ln_ratio = np.log(np.where(aq > 0, np.log1p(aq) / np.where(aq > 0, aq, 1.0), 1.0))
    out: Array = ln_ratio - 0.25 * np.log1p(b * q + (c * q) ** 2 + (d * q) ** 3 + (e * q) ** 4)
    return out


def ln_t_bond_efs(k: Array, *, gamma: float, a: float, b: float, c: float, nu: float) -> Array:
    """The Bond & Efstathiou (1984) transfer function, in the form of EBW92 (their Eq. 7).

    Parameters
    ----------
    k
        Wavenumbers, in h/Mpc.
    gamma
        The shape parameter, Gamma (in h/Mpc), so that q = k / Gamma.
    a, b, c
        The coefficients of the fit, in Mpc/h at Gamma = 1.
    nu
        The exponent of the fit.

    Returns
    -------
    numpy.ndarray
        ln T(k).
    """
    q = np.asarray(k, dtype=float) / gamma
    out: Array = -np.log1p((a * q + (b * q) ** 1.5 + (c * q) ** 2) ** nu) / nu
    return out


# ---------------------------------------------------------------------------------
# Tabulated transfer functions
# ---------------------------------------------------------------------------------


@attrs.frozen(eq=False)
class TabulatedTransfer:
    r"""A tabulated transfer function, interpolated and extrapolated smoothly.

    The table is stored as the *residual* from the EH98 no-wiggle fit,
    :math:`R(\ln k) = \ln T(k) - \ln T_{\rm EH}(k)`, which is smooth and nearly
    constant outside the BAO range. It is evaluated as
    :math:`\ln T = \ln T_{\rm EH} + R`, with :math:`R`:

    * inside the table, a cubic spline through the nodes whose slope at each end is
      clamped: to zero at the low end, and to the slope fitted to the last
      ``n_slope`` nodes at the high end;
    * below the table, constant, so that :math:`T \to T_{\rm EH} \to 1` as
      :math:`k \to 0`;
    * above the table (beyond the Boltzmann code's k_max), the EH shape rescaled in
      amplitude and tilt to match the table's value **and** slope at the join, with
      the tilt correction decaying over ``decay_ln_k`` e-folds:
      :math:`R = R_j + R'_j L (1 - e^{-(\ln k - \ln k_j)/L})`. So
      :math:`d\ln T/d\ln k` is continuous at the join, and far above it the shape is
      that of EH98.

    The table is normalised so that :math:`T \to 1` as :math:`k \to 0` (it subtracts
    the residual at the first node).

    Build it with :func:`tabulate_transfer`. Its attributes are read-only arrays.
    """

    #: ln of the nodes' wavenumbers, in h/Mpc.
    ln_k: Array
    #: The residual from EH98 no-wiggle at the nodes, normalised to 0 at the first node.
    residual: Array
    #: The scales of the EH98 no-wiggle reference.
    scales: EH98Scales
    #: The slope dR/dln k at the high end.
    slope_high: float
    #: The e-folding scale of the decay of the tilt correction above the table.
    decay_ln_k: float
    #: The spline of the residual, as piecewise-polynomial coefficients.
    _coefficients: Array

    @property
    def k_min(self) -> float:
        """The smallest wavenumber of the table, in h/Mpc."""
        return float(np.exp(self.ln_k[0]))

    @property
    def k_max(self) -> float:
        """The largest wavenumber of the table (the join), in h/Mpc."""
        return float(np.exp(self.ln_k[-1]))

    def ln_residual(self, ln_k: Array) -> Array:
        """The residual R(ln k) from the EH98 no-wiggle fit (see the class docs)."""
        ln_k = np.asarray(ln_k, dtype=float)
        x0, xj = self.ln_k[0], self.ln_k[-1]
        inside = PPoly.construct_fast(self._coefficients, self.ln_k, extrapolate=False)(
            np.clip(ln_k, x0, xj)
        )
        el = self.decay_ln_k
        above = self.residual[-1] + self.slope_high * el * -np.expm1(
            -np.maximum(ln_k - xj, 0.0) / el
        )
        out: Array = np.where(ln_k < x0, self.residual[0], np.where(ln_k > xj, above, inside))
        return out

    def ln_t(self, k: Array) -> Array:
        """Ln T(k), with k in h/Mpc."""
        k = np.asarray(k, dtype=float)
        out: Array = ln_t_eh98_no_wiggle(k, self.scales) + self.ln_residual(np.log(k))
        return out


def tabulate_transfer(
    k: Array,
    t: Array,
    scales: EH98Scales,
    *,
    n_slope: int = 4,
    decay_ln_k: float = 1.0,
) -> TabulatedTransfer:
    """Build a :class:`TabulatedTransfer` from a table of T(k).

    Parameters
    ----------
    k
        Wavenumbers of the table, in h/Mpc, strictly increasing.
    t
        The transfer function at ``k``, in any normalisation; it must be positive.
    scales
        The EH98 scales of the cosmology (from :func:`eh98_scales`), for the
        no-wiggle reference the table is stored relative to.
    n_slope
        The number of nodes at the high end to fit the slope of the residual to
        (a straight line in ln k, by least squares). At least 2.
    decay_ln_k
        The e-folding scale, in ln k, of the tilt correction above the table.

    Returns
    -------
    TabulatedTransfer

    Raises
    ------
    ValueError
        If ``k`` is not strictly increasing and positive, ``t`` is not positive, or
        there are fewer than ``max(n_slope, 4)`` nodes.
    """
    k = np.asarray(k, dtype=float)
    t = np.asarray(t, dtype=float)
    if k.ndim != 1 or k.shape != t.shape:
        raise ValueError(
            f"k and T must be 1-D arrays of the same length, got {k.shape}, {t.shape}."
        )
    if n_slope < 2 or decay_ln_k <= 0:
        raise ValueError("n_slope must be >= 2 and decay_ln_k > 0.")
    if k.size < max(n_slope, 4):
        raise ValueError(f"A tabulated transfer function needs at least {max(n_slope, 4)} nodes.")
    if not (np.all(k > 0) and np.all(np.diff(k) > 0)):
        raise ValueError(
            "The wavenumbers of a tabulated transfer function must be positive and "
            "strictly increasing."
        )
    if not np.all(t > 0):
        raise ValueError("A tabulated transfer function must be positive.")

    ln_k = np.log(k)
    residual = np.log(t) - ln_t_eh98_no_wiggle(k, scales)
    residual = residual - residual[0]

    slope_high = float(np.polyfit(ln_k[-n_slope:], residual[-n_slope:], 1)[0])
    spline = CubicSpline(ln_k, residual, bc_type=((1, 0.0), (1, slope_high)))

    for arr in (ln_k, residual):
        arr.flags.writeable = False
    coefficients = np.asarray(spline.c)
    coefficients.flags.writeable = False
    return TabulatedTransfer(
        ln_k=ln_k,
        residual=residual,
        scales=scales,
        slope_high=slope_high,
        decay_ln_k=float(decay_ln_k),
        coefficients=coefficients,
    )
