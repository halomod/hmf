"""
Module for dealing with sampled mass functions.

Provides routines for sampling theoretical functions, and for binning sampled data.
"""
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as _spline

from ..mass_function import hmf


def _prepare_mf(log_mmin, **mf_kwargs):
    h = hmf.MassFunction(Mmin=log_mmin, **mf_kwargs)
    mask = h.ngtm > 0
    icdf = _spline((h.ngtm[mask] / h.ngtm[0])[::-1], np.log10(h.m[mask][::-1]), k=3)

    return icdf, h


def _choose_halo_masses_num(N, icdf):
    # Generate random variates from 0 to maxcum
    x = np.random.random(int(N))

    # Generate halo masses from mf distribution
    m = 10 ** icdf(x)
    return m


def sample_mf(N, log_mmin, sort=False, **mf_kwargs):
    """
    Create a sample of halo masses from a theoretical mass function.

    Parameters
    ----------
    N : int
        Number of samples to draw
    log_mmin : float
        Log10 of the minimum mass to sample [Msun/h]
    sort : bool, optional
        Whether to sort (in descending order of mass) the output masses.
    mf_kwargs : keywords
        Anything passed to :class:`hmf.MassFunction` to create the mass function
        which is sampled.

    Returns
    -------
    m : array_like
        The masses
    hmf : `hmf.MassFunction` instance
        The instance used to define the mass function.

    Examples
    --------
    Simplest example:

    >>> m,hmf = sample_mf(1e5,11.0)

    Or change the mass function:

    >>> m,hmf = sample_mf(1e6,10.0,hmf_model="PS",Mmax=17)
    """
    icdf, h = _prepare_mf(log_mmin, **mf_kwargs)

    m = _choose_halo_masses_num(N, icdf)

    if sort:
        m.sort()

    return m[::-1], h


def dndm_from_sample(m, V, nm=None, bins=50):
    """
    Generate a binned dn/dm from a sample of halo masses.

    Parameters
    ----------
    m : array_like
        A sample of masses
    V : float
        Physical volume of the sample
    nm : array_like
        A multiplicity of each of the masses -- useful for
        samples from simulations in which the number of unique masses
        is much smaller than the total sample.
    bins : int or array
        Specifies bins (in log10-space!) for the sample.
        See `numpy.histogram` for more details.

    Returns
    -------
    centres : array_like
        The centres of the bins.
    hist : array_like
        The value of dn/dm in each bin.

    Notes
    -----
    The "centres" of the bins are located as the midpoint in log10-space.

    If one does not have the volume, it can be calculated as N/n(>mmin).
    """
    hist, edges = np.histogram(np.log10(m), bins, weights=nm)
    centres = (edges[1:] + edges[:-1]) / 2
    dx = centres[1] - centres[0]
    hist = hist.astype("float") / (10 ** centres * float(V) * dx * np.log(10))

    if hist[0] == 0:
        try:
            hist0 = np.where(hist != 0)[0][0]
            hist[hist0] = np.nan
        except IndexError:
            pass
    if hist[-1] == 0:
        try:
            histN = np.where(hist != 0)[0][-1]
            hist[histN] = np.nan
        except IndexError:
            pass

    return centres, hist
