'''
This module contains numerous functions for fitting hmf models to data. 
It uses MCMC techniques to do so.
'''
#===============================================================================
# IMPORTS
#===============================================================================
import numpy as np
import emcee
from hmf import MassFunction
from scipy.stats import norm
import sys
# from scipy.optimize import minimize
# import copy

#===============================================================================
# The Model
#===============================================================================
def model(parm, initial, pert, attrs, data, sd):
    """
    Calculate the log probability of a HMF model given data
    
    Parameters
    ----------
    parm : sequence
        The position of the model. Takes arbitrary parameters.
        
    initial : dict
        A dictionary with keys as names of parameters in ``parm`` and values
        as a 4-item list. The first item is the guessed value for the parameter,
        the second is a string identifying whether the prior is normal (``norm``)
        or uniform (``unif``). The third and fourth are the uniform lower and
        upper boundaries or the mean and standard deviation of the normal
        distribution.
        
    pert : ``hmf.Perturbations`` instance
        A fully realised instance is best as it may make updating it faster
        
    attrs : list
        A list of the names of parameters passed in ``parm``. Corresponds to the
        keys of ``initial``.
        
    data : array_like
        The measured HMF.
        
    sd : array_like
        Uncertainty in the measured HMF
        
    Returns
    -------
    ll : float
        The log likelihood of the model at the given position.
        
    """
    ll = 0

    # First check all values are inside boundaries (if bounds given), and
    # Get the logprob of the priors-- uniform priors don't count here
    # We reflect the variables off the edge if we need to
    i = 0
    for k, v in initial.iteritems():
        j = 0
        if v[1] == "norm":

            ll += norm.logpdf(parm[i], loc=v[2], scale=v[3])
            print ll
        elif v[1] == "unif":
            if parm[i] < v[2] or parm[i] > v[3]:
                return -np.inf
        i += 1

    # Rebuild the hod dict from given vals
    hmfdict = {attr:val for attr, val in zip(attrs, parm)}
    pert.update(**hmfdict)

    # The logprob of the model
    model = pert.corr_gal.copy()  # data + np.random.normal(scale=0.1)

    ll += np.sum(norm.logpdf(data, loc=model, scale=sd))

    return ll, parm

def fit_hmf(M, data, sd, initial, nwalkers=100, nsamples=100, burnin=10,
           thin=50, nthreads=1, filename=None, **hmfkwargs):
    """
    Run an MCMC procedure to fit a model HMF to data
    
    Parameters
    ----------
    M : array_like
        The masses at which to perform analysis. Must be the same as the input
        data
        
    data : array_like
        The measured HMF at ``M``
        
    sd : array_like
        The uncertainty in the measured HMF
        
    initial : dict
        A dictionary with keys as names of parameters in ``parm`` and values
        as a 4-item list. The first item is the guessed value for the parameter,
        the second is a string identifying whether the prior is normal (``norm``)
        or uniform (``unif``). The third and fourth are the uniform lower and
        upper boundaries or the mean and standard deviation of the normal
        distribution.
        
    nwalkers : int, default 100
        Number of walkers to use for Affine-Invariant Ensemble Sampler
        
    nsamples : int, default 100
        Number of samples that *each walker* will perform.
        
    burnin : int, default 10
        Number of samples from each walker that will be initially erased as burnin
        
    thin : int, default 50
        Keep 1 in every ``thin`` samples.
        
    nthreads : int, default 1
        Number of threads to use in sampling.
        
    \*\*hmfkwargs : arguments
        Any argument that could be sent to ``hmf.Perturbations``
        
    Returns
    -------
    flatchain : array_like, ``shape = (len(initial),nwalkers*nsamples)``
        The MCMC chain, with each parameter as a column
        
    acceptance_fraction : float
        The acceptance fraction for the MCMC run. Should be ....
    
        
    """
    if len(initial) == 0:
        raise ValueError("initial must be at least length 1")

    # Get the number of variables for MCMC
    ndim = len(initial)

    # Save which attributes are updatable for hmf as a list
    attrs = [k for k in initial]

    # Check that attrs are all applicable
    for a in attrs:
        if a not in ["wdm_mass", "delta_halo", "sigma_8", "n", "omegab", 'omegac',
                     "omegav", "omegak", "H0"]:
            raise ValueError(a + " is not a valid variable for MCMC in hmf")

    # Make sure hmfkwargs is ok
    if nthreads > 1:
        numthreads = 1
    else:
        numthreads = 0
    if 'NumThreads' in hmfkwargs:
        del hmfkwargs['NumThreads']

    # Initialise the HOD object - use all available cpus for this
    pert = Perturbations(M=M, ThreadNum=nthreads, **hmfkwargs)

    # It's better to get a dndm instance now and then the updates are faster
    pert.dndm

    # Now update numthreads for MCMC parallelisation
    pert.update(ThreadNum=numthreads)

    # Get an array of initial values
    initial_val = np.array([val[0] for k, val in initial.iteritems()])

    # Get an initial value for all walkers, around a small ball near the initial guess
    stacked_val = initial_val
    for i in range(nwalkers - 1):
        stacked_val = np.vstack((initial_val, stacked_val))
    p0 = stacked_val * np.random.normal(loc=1.0, scale=0.2, size=ndim * nwalkers).reshape((nwalkers, ndim))

    # Create the sampler object
    sampler = emcee.EnsembleSampler(nwalkers, ndim, model,
                                    args=[initial, pert, attrs, data, sd],
                                    threads=nthreads)

    # Run a burn-in
    if burnin:
        pos, prob, state = sampler.run_mcmc(p0, burnin)
        sampler.reset()
    else:
        pos = p0

    # Run the actual run
    sampler.run_mcmc(pos, nsamples, thin=thin)

    return sampler.flatchain, np.mean(sampler.acceptance_fraction)
