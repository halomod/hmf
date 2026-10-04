"""A supporting module with a routine to integrate the differential hmf in a robust manner."""

import warnings

import numpy as np
import scipy.integrate as intg
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from ..exceptions import HMFExtrapolationWarning

#: Fraction of the full integral above which the extrapolated tail triggers a warning.
_EXTRAPOLATION_WARN_FRACTION = 1e-3


class NaNError(Exception):
    """Integrator hit a NaN."""


def hmf_integral_gtm(m, dndm, mass_density=False):
    r"""
    Cumulatively integrate dn/dm.

    Parameters
    ----------
    M : array_like
        Array of masses.
    dndm : array_like
        Array of dn/dm (corresponding to M)
    mass_density : bool, `False`
        Whether to calculate mass density (or number density).

    Returns
    -------
    ngtm : array_like
        Cumulative integral of dndm.

    Examples
    --------
    Using a simple power-law mass function:

    >>> import numpy as np
    >>> m = np.logspace(10,18,500)
    >>> dndm = m**-2
    >>> ngtm = hmf_integral_gtm(m,dndm)
    >>> np.allclose(ngtm,1/m) #1/m is the analytic integral to infinity.
    True

    The function always integrates to m=1e18, and extrapolates if data are not
    provided (here emitting a :class:`~hmf.HMFExtrapolationWarning`):

    >>> m = np.logspace(10,12,500)
    >>> dndm = m**-2
    >>> ngtm = hmf_integral_gtm(m,dndm)
    >>> np.allclose(ngtm,1/m) #1/m is the analytic integral to infinity.
    True

    Notes
    -----
    ``m`` is assumed to be log-uniformly spaced. The integral above the largest
    mass, ``m[-1]``, up to :math:`10^{18}`, is only included if ``m[-1]`` is more than
    three grid steps below :math:`10^{18}` (i.e. ``m[-1] < 1e18 * m[0] / m[3]``);
    otherwise it is taken to be zero. When it is included, ``dndm`` is extrapolated
    from ``m[-1]`` to :math:`10^{18}` as a power law, by linearly extending the last
    segment of :math:`\ln(m\,dn/dm)` against :math:`\ln m`. Since a real mass function
    is exponentially suppressed at high mass, this power-law tail *overestimates* the
    true contribution. A :class:`~hmf.HMFExtrapolationWarning` is emitted when the
    extrapolated tail is more than 0.1% of the full integral above ``m[0]``; supply
    ``m`` extending to :math:`\sim 10^{18}` to avoid it. Note that the cumulative values
    within the last few grid points below ``m[-1]`` are always dominated by the tail,
    even when it is negligible for the total.
    """
    n = len(m)

    # Eliminate NaN's
    mask = np.isfinite(dndm)
    m = m[mask]
    dndm = dndm[mask]
    dndlnm = m * dndm

    if len(m) < 4:
        raise NaNError(
            f"There are too few real numbers in dndm: len(dndm) = {n}, #NaN's = {n - len(m)}"
        )

    # Calculate the mass function (and its integral) from the highest M up to 10**18
    if m[-1] < m[0] * 10**18 / m[3]:
        m_upper = np.arange(np.log(m[-1]), np.log(10**18), np.log(m[1]) - np.log(m[0]))
        mf_func = Spline(np.log(m), np.log(dndlnm), k=1)
        mf = mf_func(m_upper)

        if not mass_density:
            int_upper = intg.simpson(np.exp(mf), dx=m_upper[2] - m_upper[1])
        else:
            int_upper = intg.simpson(np.exp(m_upper + mf), dx=m_upper[2] - m_upper[1])
        total = int_upper + intg.trapezoid(dndlnm * m if mass_density else dndlnm, np.log(m))
        if int_upper > _EXTRAPOLATION_WARN_FRACTION * total:
            warnings.warn(
                f"hmf_integral_gtm: extrapolated dn/dm as a power law from m={m[-1]:.3g} to "
                f"m=1e18; this tail is {int_upper / total:.2g} of the integral above "
                f"m={m[0]:.3g}. Supply masses up to ~1e18 to avoid extrapolating.",
                HMFExtrapolationWarning,
                stacklevel=2,
            )
    else:
        int_upper = 0

    # Calculate the cumulative integral (backwards) of [m*]dndlnm
    if not mass_density:
        ngtm = np.concatenate(
            (
                intg.cumulative_trapezoid(dndlnm[::-1], dx=np.log(m[1] / m[0]))[::-1],
                np.zeros(1),
            )
        )
    else:
        ngtm = np.concatenate(
            (
                intg.cumulative_trapezoid(m[::-1] * dndlnm[::-1], dx=np.log(m[1]) - np.log(m[0]))[
                    ::-1
                ],
                np.zeros(1),
            )
        )

    return ngtm + int_upper
