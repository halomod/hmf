"""
Implements the HALOFIT (Smith+2003, Takahashi+2012) method.

This code was heavily influenced by the `HaloFit` class from the
`chomp` python package by Christopher Morrison, Ryan Scranton
and Michael Schneider (https://code.google.com/p/chomp/). It has
been modified to improve its integration with this package.

Notes
-----
When comparing the output of this module to other codes that implement HALOFIT
(e.g. nbodykit via CLASS, or CAMB), differences of order a few percent may be
observed.  These differences arise almost entirely from the **input linear power
spectrum**, not from the HALOFIT algorithm itself.  Specifically:

- nbodykit/CLASS and CAMB use Boltzmann-solver transfer functions, whereas
  ``hmf`` defaults to the Eisenstein-Hu (EH) fitting formula for the transfer
  function.
- Different codes may also use different values of σ₈ if they normalise the
  spectrum differently.

Benchmarks: when the **same** linear power spectrum (``k``, ``Δ²(k)``) is
passed to both this implementation and to CAMB's Takahashi+2012 halofit, the
resulting non-linear power spectra agree to better than 0.3% across the range
0.01 ≲ k ≲ 30 h/Mpc.  Any larger discrepancy seen in practice is therefore
attributable to differences in the linear input, not to this implementation.
"""

import warnings

import numpy as np
from scipy.integrate import simpson as _simps
from scipy.optimize import brentq

from ..cosmology.cosmo import Cosmology as CosmologyClass


def _sigma2_moments(
    lnr: np.ndarray, k: np.ndarray, delta_k: np.ndarray, nmax: int = 2
) -> np.ndarray:
    r"""
    Gaussian-smoothed variance and its moments, vectorised over radius.

    Computes, for every :math:`R = e^{\ln R}` at once,

    .. math::

        s_n(R) = \int \Delta^2(k)\, (kR)^{2n}\, e^{-(kR)^2}\, \mathrm{d}\ln k,
        \qquad n = 0, \ldots, n_\mathrm{max},

    so that :math:`s_0 = \sigma^2(R)`.

    Parameters
    ----------
    lnr : array_like
        Natural log of the smoothing radii [Mpc/h], any shape.
    k : array_like
        Wavenumbers [h/Mpc], 1D.
    delta_k : array_like
        Dimensionless power spectrum at `k`.
    nmax : int, optional
        Highest moment to compute.

    Returns
    -------
    moments : np.ndarray
        Array of shape ``(nmax + 1,) + np.shape(lnr)`` holding :math:`s_0, \ldots,
        s_{n_\mathrm{max}}`.
    """
    x = (k * np.exp(np.asarray(lnr))[..., None]) ** 2
    integrands = delta_k * np.exp(-x) * np.stack([x**n for n in range(nmax + 1)])
    return _simps(integrands, x=np.log(k), axis=-1)


def _get_spec(k: np.ndarray, delta_k: np.ndarray, sigma_8=None) -> tuple[float, float, float]:
    r"""
    Calculate spectral parameters from power spectrum.

    Computes the nonlinear wavenumber, effective spectral index, and curvature
    of the power spectrum following Smith+2003 / Takahashi+2012.

    The non-linear wavenumber ``k_nl = 1/R_nl`` is defined by the condition
    σ²(R_nl) = 1, where the variance is computed using a Gaussian window
    function:

    .. math::

        \sigma^2(R) = \int \Delta^2(k)\, e^{-(kR)^2}\, \mathrm{d}\ln k

    The effective spectral index and curvature are the first and second
    logarithmic derivatives of σ² evaluated at R_nl:

    .. math::

        n_\mathrm{eff} = -\frac{\mathrm{d}\ln\sigma^2}{\mathrm{d}\ln R}
                          \bigg|_{R_\mathrm{nl}} - 3, \qquad
        C = -\frac{\mathrm{d}^2\ln\sigma^2}{\mathrm{d}(\ln R)^2}
            \bigg|_{R_\mathrm{nl}}

    Since σ²(R) decreases monotonically with R, R_nl is bracketed by tabulating
    σ²(R) on a coarse grid (in a single vectorised integration) and then refined
    with a root-find (:func:`scipy.optimize.brentq`) on ln σ²(ln R). The derivatives are
    evaluated analytically from the moments of the Gaussian window (see
    :func:`_sigma2_moments`), rather than by numerical differentiation.

    Parameters
    ----------
    k : array_like
        Wavenumbers
    delta_k : array_like
        Dimensionless power spectrum at `k`
    sigma_8 : scalar
        RMS linear density fluctuations in spheres of radius 8 Mpc/h at z=0.
        Not used any more at all!

    Returns
    -------
    knl : float
        Non-linear wavenumber
    n_eff : float
        Effective spectral index
    n_curv : float
        Curvature of the spectrum

    """

    def log_sigma2(lnr):
        return np.log(_sigma2_moments(lnr, k, delta_k, nmax=0)[0])

    # Tabulate ln sigma^2 on a coarse grid of R covering the scales probed by the
    # k-range (with some margin, since the Gaussian window still has support a little
    # beyond 1/k). sigma^2 decreases monotonically with R, so the sign change of
    # ln sigma^2 brackets R_nl, which is then refined with a root-find.
    lnr_grid = np.arange(-np.log(k.max()) - 2.0, -np.log(k.min()) + 2.0, 1.0)
    with np.errstate(divide="ignore"):
        lnsig_grid = log_sigma2(lnr_grid)

    crossing = np.flatnonzero((lnsig_grid[:-1] > 0) & (lnsig_grid[1:] <= 0))
    if crossing.size:
        i = crossing[0]
        lnrnl = brentq(log_sigma2, lnr_grid[i], lnr_grid[i + 1], xtol=1e-10)
    else:
        lnrnl = lnr_grid[np.argmin(np.abs(lnsig_grid))]
        warnings.warn(
            "Could not determine non-linear scale: sigma(R) does not cross unity for "
            f"R in [{np.exp(lnr_grid[0]):.3g}, {np.exp(lnr_grid[-1]):.3g}] Mpc/h. "
            f"Continuing with r_nl={np.exp(lnrnl):.3g}; HALOFIT results will be unreliable.",
            stacklevel=2,
        )

    s0, s1, s2 = _sigma2_moments(lnrnl, k, delta_k)
    # d ln(s0) / d ln R and d^2 ln(s0) / d (ln R)^2, using ds_n/dlnR = 2n s_n - 2 s_{n+1}.
    dev1 = -2 * s1 / s0
    dev2 = 4 * (s2 - s1) / s0 - dev1**2

    n_eff = -dev1 - 3.0
    n_curv = -dev2

    return 1 / np.exp(lnrnl), n_eff, n_curv


def halofit(k, delta_k, *, sigma_8=None, z=0, cosmo=None, takahashi=True):
    """
    Implementation of HALOFIT (Smith+2003, Takahashi+2012).

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
    cosmo : :class:`astropy.cosmology.FLRW` or :class:`hmf.cosmology.Cosmology`, optional
        The cosmology used for the redshift-dependent density parameters. Either any
        astropy ``FLRW`` instance, or an hmf :class:`~hmf.cosmology.Cosmology`
        framework (in which case its :attr:`~hmf.cosmology.Cosmology.cosmo` is used,
        including any ``cosmo_params``). Default is the default cosmology of the
        :class:`~hmf.cosmology.Cosmology` framework (Planck18).
    takahashi : bool, optional
        Whether to use updated parameters from Takahashi+2012. Otherwise use
        original from Smith+2003.

    Returns
    -------
    nonlinear_delta_k : array_like
        Dimensionless power at `k`, with nonlinear corrections applied.

    Notes
    -----
    When comparing against other codes (e.g. nbodykit via CLASS, or CAMB) one
    may observe differences of a few percent.  Benchmarks show that when the
    **same** linear power spectrum is passed to this function and to CAMB's
    Takahashi halofit, the results agree to better than 0.3% for
    0.01 ≲ k ≲ 30 h/Mpc.  Larger observed differences are therefore due to
    the codes using different input linear power spectra (e.g. Eisenstein-Hu
    vs. a full Boltzmann solver), not to a difference in the halofit algorithm
    itself.  Use the CAMB transfer model (``transfer_model="CAMB"``) in the
    :class:`~hmf.density_field.transfer.Transfer` framework to minimise such
    differences.
    """
    if sigma_8 is not None:
        warnings.warn("sigma_8 is not used any more, and will be removed in v4", stacklevel=2)

    if cosmo is None:
        cosmo = CosmologyClass().cosmo
    elif isinstance(cosmo, CosmologyClass):
        cosmo = cosmo.cosmo

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
            + 2.3706 * neff**2
            + 0.9903 * neff**3
            + 0.2250 * neff**4
            + -0.6038 * rncur
            + 0.1749 * omegavz * (1 + w)
        )
        b = 10 ** (
            -0.5642
            + 0.5864 * neff
            + 0.5716 * neff**2
            + -1.5474 * rncur
            + 0.2279 * omegavz * (1 + w)
        )
        c = 10 ** (0.3698 + 2.0404 * neff + 0.8161 * neff**2 + 0.5869 * rncur)
        gam = 0.1971 - 0.0843 * neff + 0.8460 * rncur
        alpha = np.abs(6.0835 + 1.3373 * neff - 0.1959 * neff**2 + -5.5274 * rncur)
        beta = (
            2.0379
            - 0.7354 * neff
            + 0.3157 * neff**2
            + 1.2490 * neff**3
            + 0.3980 * neff**4
            - 0.1682 * rncur
            + fnu * (1.081 + 0.395 * neff**2)
        )
        xmu = 0.0
        xnu = 10 ** (5.2105 + 3.6902 * neff)

    else:
        a = 10 ** (
            1.4861
            + 1.8369 * neff
            + 1.6762 * neff**2
            + 0.7940 * neff**3
            + 0.1670 * neff**4
            + -0.6206 * rncur
        )
        b = 10 ** (0.9463 + 0.9466 * neff + 0.3084 * neff**2 + -0.94 * rncur)
        c = 10 ** (-0.2807 + 0.6669 * neff + 0.3214 * neff**2 - 0.0793 * rncur)
        gam = 0.8649 + 0.2989 * neff + 0.1631 * rncur
        alpha = np.abs(1.3884 + 0.3700 * neff - 0.1452 * neff**2)
        beta = 0.8291 + 0.9854 * neff + 0.3401 * neff**2
        xmu = 10 ** (-3.5442 + 0.1908 * neff)
        xnu = 10 ** (0.9589 + 1.2857 * neff)

    if np.abs(1 - omegamz) > 0.01:
        f1a = omegamz**-0.0732
        f2a = omegamz**-0.1423
        f3a = omegamz**0.0725
        f1b = omegamz**-0.0307
        f2b = omegamz**-0.0585
        f3b = omegamz**0.0743
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

    ph = a * y ** (f1 * 3) / (1 + b * y**f2 + (f3 * c * y) ** (3 - gam))
    ph = ph / (1 + xmu / y + xnu * y**-2) * (1 + fnu * (0.977 - 18.015 * (cosmo.Om0 - 0.3)))

    plinaa = plin * (1 + fnu * 47.48 * k**2 / (1 + 1.5 * k**2))
    pq = plin * (1 + plinaa) ** beta / (1 + plinaa * alpha) * np.exp(-y / 4.0 - y**2 / 8.0)
    pnl = pq + ph

    # We have to copy so the original data is not overwritten, giving unexpected results.
    nonlinear_delta_k = delta_k.copy()
    nonlinear_delta_k[mask] = pnl

    return nonlinear_delta_k
