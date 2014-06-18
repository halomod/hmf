'''
Functions for sampling the mass function with a given number of particles.
'''
import numpy as np
from hmf import MassFunction
from scipy.integrate import simps, cumtrapz
from scipy.interpolate import InterpolatedUnivariateSpline as spline

def _prepare_mf(M_min, M_max, mf_kwargs, boxsize=None):
    # Create the mass function object
    M = np.linspace(M_min, M_max, 500)
    if boxsize is not None:
        mf_kwargs['lnk'] = np.linspace(np.log(2 * np.pi / boxsize), 20, 500)

    mf_obj = MassFunction(M=M, **mf_kwargs)

    # Get the total density within limits
    total_rho = simps(mf_obj.M * mf_obj.dndlnm, M) * np.log(10)
    frac_in_bounds = total_rho / (mf_obj.cosmo.omegam * 2.7755e11)

    cumfunc = cumtrapz(mf_obj.dndlnm, M, initial=1e-20) * np.log(10)

    cdf = spline(M, cumfunc, k=3)
    icdf = spline(cumfunc, M, k=3)

    print "prepare mf: ", total_rho, frac_in_bounds, M_min, M_max, cumfunc.min(), cumfunc.max()
    return cdf, icdf, mf_obj, frac_in_bounds

def _choose_halo_masses(cdf, icdf, M_min, M_max, omegam, vol, frac, tol):
    M_min = 10 ** M_min
    M_max = 10 ** M_max
    diff = 4
    m_tot = omegam * 2.7755e11 * vol * frac
    m_tot_temp = omegam * 2.7755e11 * vol * frac
    j = 0


    while diff > tol:
        # Figure out expected number of halos to get back total mass
        maxcum = cdf(np.log10(min(M_max, m_tot_temp)))
        N_expected = int(maxcum * vol)
        # Generate random variates from 0 to maxcum
        x = np.random.random(N_expected) * maxcum

        # Generate halo masses from mf distribution
        m = 10 ** icdf(x)

        # Make sure we don't have more or less mass than needed
        cumsum = np.cumsum(m)
        print "choose halo masses:", maxcum, N_expected, np.log10([m.min(), m.max()]), cumsum
        try:
            cross_ind = np.where(cumsum > m_tot_temp)[0][0]
        except IndexError:
            cross_ind = len(cumsum) - 1

        over = abs(cumsum[cross_ind] - m_tot_temp) / m_tot
        under = abs(cumsum[cross_ind - 1] - m_tot_temp) / m_tot

        if over < tol and under < tol:
            if over < under:
                m = m[:cross_ind ]
            else:
                m = m[:cross_ind + 1]
            diff = over
        elif over < tol:
            m = m[:cross_ind + 1]
            diff = over
        else:
            m = m[:cross_ind]
            diff = under
            m_tot_temp -= cumsum[cross_ind - 1]

        # Save the halo masses
        if j == 0:
            cell_masses = m
        else:
            cell_masses = np.concatenate((cell_masses, m))

        j += 1

    return cell_masses

def _choose_halo_masses_num(cdf, icdf, m_max, vol=None, N=None):
    maxcum = cdf(m_max)
    if vol is not None:
        print "getting N in choose"
        N = int(maxcum * vol)

    # Generate random variates from 0 to maxcum
    x = np.random.random(N) * maxcum

    # Generate halo masses from mf distribution
    m = 10 ** icdf(x)
    return m


def sample_mf(simvars=None, nvars=None, tol=3, match='mass',
              sort=False, **mf_kwargs):
    """
    Create a sample of halo masses from the theoretical mass function.
    
    There are two ways of calling the function. The first is to treat it as
    sampling within a simulation box, and the second is to purely get a sample
    of length N within the halo mass bounds given. For the first, :attr:`simvars`
    must be passed, for the second :attr:`nvars` must be passed. If both are 
    passed, simvars is used.
    
    Parameters
    ----------
    simvars : dict
        A dictionary defining four values:
            boxsize : boxsize of the simulation in Mpc/h
            npart   : number of particles in the simulations
            m_min   : minimum halo mass desired, in units of the particle mass
            mfrac   : maximum halo mass desired, in units of the simulation mass
            
    nvars : dict
        A dictionary defining three values:
            n       : number of halos to sample
            m       : total mass to sample (if n not provided)
            m_min   : minimum halo mass in log10(M_sun)
            m_max   : maximum halo mass in log10(M_sun)
            
    tol : int or float, default 3
        If sampling to match a total mass, this defines the tolerance within 
        which the sample is deemed to match it.
        If int, the tolerance is within tol*m_min of the total mass.
        If float, tol is a fraction of the total mass.
        
    match : str, {'mass','num'}
        What you want to match
        
    mf_kwargs : keywords
        Anything passed to :class:`MassFunction` to create the mass function 
        which is sampled.
        
    """
    hard_M_max = 16  # numerical error after this

    omegam = MassFunction(**mf_kwargs).cosmo.omegam

    if simvars is not None:
        boxsize = simvars.pop('boxsize')
        npart = simvars.pop('npart')
        m_min = simvars.pop('m_min', 20)
        mfrac = simvars.pop('mfrac', 0.01)
        vol = boxsize ** 3
        m_tot = omegam * vol * 2.7755e11
        mpart = m_tot / npart
        m_min = np.log10(m_min * mpart)
        m_max = np.log10(m_tot * mfrac)
    else:
        n = nvars.pop("n", None)
        if n is None:
            m_tot = nvars.pop("m")
            vol = m_tot / (omegam * 2.7755e11)
        else:
            m_tot = None
            vol = None
        m_min = nvars.pop("m_min", 11)
        m_max = nvars.pop("m_max", 16)
        boxsize = vol ** (1. / 3.)



    m_max = min(m_max, hard_M_max)
    cdf, icdf, hmf, frac_in_bounds = _prepare_mf(m_min, m_max,
                                                 mf_kwargs, boxsize)

    if type(tol) is int and m_tot is not None:
        tol *= 10 ** m_min / (m_tot * frac_in_bounds)


    if match == "mass":
        m = _choose_halo_masses(cdf, icdf, m_min, m_max, omegam, vol,
                                frac_in_bounds, tol)
    elif match == "num":
        m = _choose_halo_masses_num(cdf, icdf, m_max * frac_in_bounds, N=n, vol=vol)

    if sort:
        m.sort()

    return m[::-1], hmf, frac_in_bounds
