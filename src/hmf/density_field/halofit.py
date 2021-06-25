"""
Implements the HALOFIT (Smith+2003, Takahashi+2012) method.

This code was heavily influenced by the `HaloFit` class from the
`chomp` python package by Christopher Morrison, Ryan Scranton
and Michael Schneider (https://code.google.com/p/chomp/). It has
been modified to improve its integration with this package.
"""
import numpy as np
import warnings
from scipy.integrate import simps as _simps
from scipy.interpolate import InterpolatedUnivariateSpline as _spline
from scipy.optimize import minimize
from typing import Tuple

from ..cosmology.cosmo import Cosmology as csm


def _get_spec(
    k: np.ndarray, delta_k: np.ndarray, sigma_8=None
) -> Tuple[float, float, float]:
    """
    Calculate nonlinear wavenumber, effective spectral index and curvature
    of the power spectrum.

    Parameters
    ----------
    k : array_like
        Wavenumbers
    delta_k : array_like
        Dimensionless power spectrum at `k`
    sigma_8 : scalar
        RMS linear density fluctuations in spheres of radius 8 Mpc/h at z=0. Not
        used any more at all!

    Returns
    -------
    knl : float
        Non-linear wavenumber
    n_eff : float
        Effective spectral index
    n_curv : float
        Curvature of the spectrum
    """
    # Initialize sigma spline
    def get_log_sigma2(lnr):
        R = np.exp(lnr)
        integrand = delta_k * np.exp(-((k * R) ** 2))
        return np.log(_simps(integrand, np.log(k)))

    def get_sigma_abs(lnr):
        return np.abs(get_log_sigma2(lnr))

    res = minimize(
        get_sigma_abs, x0=[1.0], options={"xatol": np.log(1.1)}, method="Nelder-Mead"
    )

    if not res.success:
        warnings.warn(
            f"Could not determine non-linear scale! Failed with error: {res.message}. "
            f"Continuing with best-fit non-linear scale: r_nl={np.exp(res.x)}, with log_sigma^2 = {res.fun}"
        )

    rnl = np.exp(res.x)
    knl = 1 / rnl

    lnr = np.linspace(np.log(0.75 * rnl), np.log(1.25 * rnl), 20)
    lnsig = [get_log_sigma2(r) for r in lnr]
    sig_of_r = _spline(lnr, lnsig, k=5)
    dev1, dev2 = sig_of_r.derivatives(np.log(rnl))[1:3]

    n_eff = -dev1 - 3.0
    n_curv = -dev2

    return knl, n_eff, n_curv


def halofit(k, delta_k, sigma_8=None, z=0, cosmo=None, takahashi=True):
    """
    Implementation of HALOFIT (Smith+2003).

    Parameters
    ----------
    k : array_like
        Wavenumbers [h/Mpc].
    delta_k : array_like
        Dimensionless power (linear) at `k`.
    sigma_8 : float
        RMS linear density fluctuations in spheres of radius 8 Mpc/h at z=0. Not used
        at all.
    z : float
        Redshift
    cosmo : :class:`hmf.cosmo.Cosmology` instance, optional
        An instance of either the `Cosmology` class provided in the `hmf` package, or
        any subclass of `FLRW` from `astropy`. Defualt is the default cosmology from
        the :mod:`hmf.cosmo` module.
    takahashi : bool, optional
        Whether to use updated parameters from Takahashi+2012. Otherwise use
        original from Smith+2003.

    Returns
    -------
    nonlinear_delta_k : array_like
        Dimensionless power at `k`, with nonlinear corrections applied.
    """
    if sigma_8 is not None:
        warnings.warn("sigma_8 is not used any more, and will be removed in v4")

    if cosmo is None:
        cosmo = csm()

    # Get physical parameters
    rknl, neff, rncur = _get_spec(k, delta_k)

    # Only apply the model to higher wavenumbers
    mask = k > 0.005
    plin = delta_k[mask]
    k = k[mask]

    # Define the cosmology at redshift
    omegamz = cosmo.Om(z)
    omegavz = cosmo.Ode(z)

    w = cosmo.w(z)
    fnu = cosmo.Onu0 / cosmo.Om0

    if takahashi:
        a = 10 ** (
            1.5222
            + 2.8553 * neff
            + 2.3706 * neff ** 2
            + 0.9903 * neff ** 3
            + 0.2250 * neff ** 4
            + -0.6038 * rncur
            + 0.1749 * omegavz * (1 + w)
        )
        b = 10 ** (
            -0.5642
            + 0.5864 * neff
            + 0.5716 * neff ** 2
            + -1.5474 * rncur
            + 0.2279 * omegavz * (1 + w)
        )
        c = 10 ** (0.3698 + 2.0404 * neff + 0.8161 * neff ** 2 + 0.5869 * rncur)
        gam = 0.1971 - 0.0843 * neff + 0.8460 * rncur
        alpha = np.abs(6.0835 + 1.3373 * neff - 0.1959 * neff ** 2 + -5.5274 * rncur)
        beta = (
            2.0379
            - 0.7354 * neff
            + 0.3157 * neff ** 2
            + 1.2490 * neff ** 3
            + 0.3980 * neff ** 4
            - 0.1682 * rncur
            + fnu * (1.081 + 0.395 * neff ** 2)
        )
        xmu = 0.0
        xnu = 10 ** (5.2105 + 3.6902 * neff)

    else:
        a = 10 ** (
            1.4861
            + 1.8369 * neff
            + 1.6762 * neff ** 2
            + 0.7940 * neff ** 3
            + 0.1670 * neff ** 4
            + -0.6206 * rncur
        )
        b = 10 ** (0.9463 + 0.9466 * neff + 0.3084 * neff ** 2 + -0.94 * rncur)
        c = 10 ** (-0.2807 + 0.6669 * neff + 0.3214 * neff ** 2 - 0.0793 * rncur)
        gam = 0.8649 + 0.2989 * neff + 0.1631 * rncur
        alpha = np.abs(1.3884 + 0.3700 * neff - 0.1452 * neff ** 2)
        beta = 0.8291 + 0.9854 * neff + 0.3401 * neff ** 2
        xmu = 10 ** (-3.5442 + 0.1908 * neff)
        xnu = 10 ** (0.9589 + 1.2857 * neff)

    if np.abs(1 - omegamz) > 0.01:
        f1a = omegamz ** -0.0732
        f2a = omegamz ** -0.1423
        f3a = omegamz ** 0.0725
        f1b = omegamz ** -0.0307
        f2b = omegamz ** -0.0585
        f3b = omegamz ** 0.0743
        frac = omegavz / (1 - omegamz)
        if takahashi:
            f1 = f1b
            f2 = f2b
            f3 = f3b
        else:
            f1 = frac * f1b + (1 - frac) * f1a
            f2 = frac * f2b + (1 - frac) * f2a
            f3 = frac * f3b + (1 - frac) * f3a
    else:
        f1 = f2 = f3 = 1.0

    y = k / rknl

    ph = a * y ** (f1 * 3) / (1 + b * y ** f2 + (f3 * c * y) ** (3 - gam))
    ph = (
        ph
        / (1 + xmu / y + xnu * y ** -2)
        * (1 + fnu * (0.977 - 18.015 * (cosmo.Om0 - 0.3)))
    )

    plinaa = plin * (1 + fnu * 47.48 * k ** 2 / (1 + 1.5 * k ** 2))
    pq = (
        plin
        * (1 + plinaa) ** beta
        / (1 + plinaa * alpha)
        * np.exp(-y / 4.0 - y ** 2 / 8.0)
    )
    pnl = pq + ph

    # We have to copy so the original data is not overwritten, giving unexpected results.
    nonlinear_delta_k = delta_k.copy()
    nonlinear_delta_k[mask] = pnl

    return nonlinear_delta_k
