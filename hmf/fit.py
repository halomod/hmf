'''
Created on Oct 14, 2013

@author: Steven

This module contains numerous functions for fitting HOD models to correlation
function data. It uses MCMC techniques to do so.
'''
#===============================================================================
# IMPORTS
#===============================================================================
import numpy as np
import emcee
from scipy.stats import norm
from scipy.optimize import minimize
from multiprocessing import cpu_count
import time
import cosmolopy as cp
import warnings
import pickle
from numbers import Number
import copy

#===============================================================================
# The Model
#===============================================================================
def model(parm, priors, h, attrs, data, quantity, sigma, blobs=None,
          verbose=0, store_class=False, relax=False):
    """
    Calculate the log probability of a :class:`~hmf.MassFunction` model given data
    
    Parameters
    ----------
    parm : list of floats
        The position of the model. Takes arbitrary parameters.
        
    priors : list of prior classes
        A list containing instances of :class:`.Prior` subclasses. These specify 
        the prior information on each parameter.
        
    h : instance of :class:`~cosmo.Cosmology` subclass
        An instance of any subclass of :class:`~cosmo.Cosmology` with the 
        desired options set. Variables of the estimation are updated within the 
        routine.  
        
    attrs : list
        A list of the names of parameters passed in :attr:`.parm`.
        
    data : array_like
        The data to be compared to -- must be the same length as the 
        
    quantity : str
        The quantity to be compared (eg. ``"dndm"``)
        
    sigma : array_like
        If a vector, this is taken to be the standard deviation of the data. If
        a matrix, it is taken to be the covariance matrix of the data.
        
    blobs : list of str
        Names of quantities to be returned along with the chain. Must be a
        
    verbose : int, default 0
        How much to write to screen.
        
    store_class : bool, default False
        If True, return an ordered list of HaloModel objects. 
        
    relax : bool, default False
        If relax is true, the call to get the quantity is wrapped in a try:except:.
        If an error occurs, the lognorm is set to -inf, rather than raising an exception.
        This can be helpful if a flat prior is used on cosmology, for which extreme
        values can sometimes cause exceptions. 
        
    Returns
    -------
    ll : float
        The log likelihood of the model at the given position.
        
    """
    ll = 0

    p = copy.copy(parm)
    for prior in priors:
        # A uniform prior doesn't change likelihood but returns -inf if outside bounds
        if isinstance(prior, Uniform):
            index = attrs.index(prior.name)
            if parm[index] < prior.low or parm[index] > prior.high:
                return -np.inf, blobs
            # If it is a log distribution, un-log it for use.
            if isinstance(prior, Log):
                p[index] = 10 ** parm[index]

        elif isinstance(prior, Normal):
            index = attrs.index(prior.name)
            ll += norm.logpdf(parm[index], loc=prior.mean, scale=prior.sd)

        elif isinstance(prior, MultiNorm):
            indices = [attrs.index(name) for name in prior.name]
            ll += _lognormpdf(np.array(parm[indices]), np.array(prior.means),
                              prior.cov)

    # Rebuild the hod dict from given vals
    # Any attr starting with <name>: is put into a dictionary.
    hoddict = {}
    for attr, val in zip(attrs, p):
        if ":" in attr:
            if attr.split(":")[0] not in hoddict:
                hoddict[attr.split(":")[0]] = {}

            hoddict[attr.split(":")[0]][attr.split(":")[1]] = val
        else:
            hoddict[attr] = val

    # Update the actual model
    h.update(**hoddict)

    # Get the quantity to compare (if exceptions are raised, treat properly)
    try:
        q = getattr(h, quantity)
    except Exception as e:
        if relax:
            print "WARNING: PARAMETERS FAILED, RETURNING INF: ", zip(attrs, parm)
            print "EXCEPTION RAISED: ", e
            return -np.inf, blobs
        else:
            raise e

    # The logprob of the model
    if len(sigma.shape) == 2:
        ll += _lognormpdf(q, data, sigma)
    elif len(sigma.shape) == 1:
        ll += np.sum(norm.logpdf(data, loc=q, scale=sigma))
    else:
        raise ValueError("sigma must be an array of 1 or 2 dimensions, but has %s dim" % len(sigma.shape))

    if verbose > 0:
        print "Likelihood: ", ll
    if verbose > 1 :
        print "Update Dictionary: ", hoddict

    # Get blobs to return as well.
    if blobs is not None or store_class:
        out = []
        if store_class:
            out.append(h)
        for b in blobs:
            if ":" not in b:
                out.append(getattr(h, b))
            elif ":" in b:
                out.append(getattr(h, b.split(":")[0])[b.split(":")[1]])
        return ll, out
    else:
        return ll

def fit_hod(data, priors, h, guess=[], nwalkers=100, nsamples=100, burnin=0,
            nthreads=0, blobs=None, prefix=None, chunks=None, verbose=0,
            find_peak_first=False, sd=None, covar=None,
            quantity="projected_corr_gal", store_class=False, relax=False,
            initial_pos=None, **kwargs):
    """
    Estimate the parameters in :attr:`.priors` using AIES MCMC.
    
    This routine uses the emcee package to run an MCMC procedure, fitting 
    parameters passed in :attr:`.priors` to the given galaxy correlation 
    function.
    
    Parameters
    ----------
    data : array_like
        The measured correlation function at :attr:`r`
                
    priors : list of prior classes
        A list containing instances of :class:`.Uniform`, :class:`.Normal` or 
        :class:`.MultiNorm` classes. These specify the prior information on each 
        parameter.
    
    h : instance of :class:`~halo_model.HaloModel`
        This instance will be updated with the variables of the minimization.
        Other desired options should have been set upon instantiation.
        
    guess : array_like, default []
        Where to start the chain. If empty, will get central values from the
        distributions.
        
    nwalkers : int, default 100
        Number of walkers to use for Affine-Invariant Ensemble Sampler
        
    nsamples : int, default 100
        Number of samples that *each walker* will perform.
        
    burnin : int, default 10
        Number of samples from each walker that will be initially erased as 
        burnin. Note, this performs *additional* iterations, rather than 
        consuming iterations from :attr:`.nsamples`.
                
    nthreads : int, default 0
        Number of threads to use in sampling. If nought, will automatically 
        detect number of cores available.
        
    blobs : list of str
        Names of quantities to be returned along with the chain
        MUST be immediate properties of the :class:`HaloModel` class.
        
    prefix : str, default ``None``
        The prefix for files to which to write results sequentially. If ``None``,
        will not write anything out.
        
    chunks : int, default ``None``
        Number of samples to run before appending results to file. Only
        applicable if :attr:`.filename` is provided.
        
    verbose : int, default 0
        The verbosity level.
        
    find_peak_first : bool, default False
        Whether to perform a minimization routine before using MCMC to find a 
        good guess to begin with. Could reduce necessary burn-in.
        
    sd : array_like, default ``None``
        Uncertainty in the measured correlation function
        
    covar : 2d array, default ``None``
        Covariance matrix of the data. Either `sd` or `covar` must be given,
        but if both are given, `covar` takes precedence.
    
    store_class : bool, default False
        If True, return an ordered list of HaloModel objects. 
    
    relax : bool, default False
        If relax is true, the call to get the quantity is wrapped in a try:except:.
        If an error occurs, the lognorm is set to -inf, rather than raising an exception.
        This can be helpful if a flat prior is used on cosmology, for which extreme
        values can sometimes cause exceptions. 
        
    dependent_params : ``func``, default None
        A function which returns a dictionary of values for parameters which
        are dependent on combinations of other parameters for a given dataset.
        The signature is ``func(h,dict)``, where h is an instance of `HaloModel`
        pre-updating with new parameters, and dict is the dictionary that will
        serve to update `h` (pre-update with this function).
        
    initial_pos : array_like shape=``(nparams,nwalkers)``, default None
        Starting positions of the parameters (for each walker). If None,
        these will be calculated in a small ball around the guess for each.
        This is useful for re-starting the calculation from a saved position.
        
    \*\*kwargs :
        Arguments passed to :func:`fit_hod_minimize` if :attr:`find_peak_first`
        is ``True``.
        
    Returns
    -------
    flatchain : array_like, ``shape = (len(initial),nwalkers*nsamples)``
        The MCMC chain, with each parameter as a column. Note that each walker 
        is semi-independent and they are interleaved. Calling 
        ``flatchain.reshape((nwalkers, -1, ndim))`` will retrieve the proper
        structure.
        
    acceptance_fraction : float
        The acceptance fraction for the MCMC run. Should be between 0.2 and 0.5.
    
    """

    if len(priors) == 0:
        raise ValueError("priors must be at least length 1")

    # Save which attributes are updatable for HaloModel as a list
    attrs = []
    for prior in priors:
        if isinstance(prior.name, basestring):
            attrs += [prior.name]
        else:
            attrs += prior.name

    # Get the number of variables for MCMC
    ndim = len(attrs)

    # Ensure guess was set correctly.
    if guess and len(guess) != ndim:
            warnings.warn("Guess was set incorrectly: %s" % guess)
            guess = []

    # Ensure no burn-in if restarting from old run
    if initial_pos is not None:
        burnin = 0

    # If using CAMB, nthreads MUST BE 1
    if h.transfer_fit == "CAMB":
        for pp in h.pycamb_dict:
            if pp in attrs:
                nthreads = 1
    elif not nthreads:
        # auto-calculate the number of threads to use if not set.
        nthreads = cpu_count()

    # This just makes sure that the caching works
    getattr(h, quantity)


    # Setup the Ensemble Sampler
    if covar is not None:
        arglist = [priors, h, attrs, data, quantity, blobs, None, covar, verbose,
                   store_class, relax]
    else:
        arglist = [priors, h, attrs, data, quantity, blobs, sd, None, verbose,
                   store_class, relax]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, model,
                                    args=arglist,
                                    threads=nthreads)

    # Get initial positions if required
    if initial_pos is None:
        initial_pos = get_initial_pos(guess, priors, nwalkers, find_peak_first,
                                      data, sd, h, verbose, **kwargs)
        extend = False
    else:
        extend = True

    # Run a burn-in
    if burnin:
        if type(burnin) == int:
            initial_pos, lnprob, rstate = sampler.run_mcmc(initial_pos, burnin)
            sampler.reset()

        else:
            sampler.run_mcmc(initial_pos, burnin[0])
            print burnin[1] * np.max(sampler.acor), sampler.iterations
            while burnin[1] * np.max(sampler.acor) > sampler.iterations:
                initial_pos, lnprob, rstate, blobs0 = sampler.run_mcmc(None, 5)
                print burnin[1] * np.max(sampler.acor), sampler.iterations
                if sampler.iterations > burnin[2]:
                    warnings.warn("Burnin FAILED... continuing (acor=%s)" % (np.max(sampler.acor)))

            if verbose > 0:
                burnin = sampler.iterations
                print "Used %s samples for burnin" % sampler.iterations
            sampler.reset()
    else:
        lnprob = None
        rstate = None
        blobs0 = None
#     else: TODO
#         try:
#             lnprob = np.genfromtxt(prefix+"likelihoods")[-nwalkers:,:]
#             rstate = None
#             blobs0
#
    # Run the actual run
    if prefix is None:
        sampler.run_mcmc(initial_pos, nsamples)
    else:
        header = "# " + "\t".join(attrs) + "\n"

        if not extend:
            with open(prefix + "chain", "w") as f:
                f.write(header)
        else:
            with open(prefix + "chain", 'r') as f:
                nsamples -= (sum(1 for line in f) - 1) / nwalkers

        print "NSAMPLES: ", nsamples
        # If storing the whole class, add the label to front of blobs
        if store_class:
            try:
                blobs = ["HaloModel"] + blobs
            except TypeError:
                blobs = ["HaloModel"]

        if chunks == 0 or chunks > nsamples:
            chunks = nsamples

        start = time.time()
        for i, result in enumerate(sampler.sample(initial_pos, iterations=nsamples,
                                                  lnprob0=lnprob, rstate0=rstate,
                                                  blobs0=blobs0)):
            if (i + 1) % chunks == 0 or i + 1 == nsamples:
                if verbose:
                    print "Done ", 100 * float(i + 1) / nsamples ,
                    print "%. Time per sample: ", (time.time() - start) / ((i + 1) * nwalkers)

                # Write out files
                write_iter(sampler, i, nwalkers, chunks, prefix, blobs, extend)

    return sampler

def fit_hod_minimize(data, priors, h, sd=None, covar=None, guess=[], verbose=0,
                     method="Nelder-Mead", disp=False, maxiter=30, tol=None):
    """
    Run an optimization procedure to fit a model correlation function to data.
    
    Parameters
    ----------
    data : array_like
        The measured correlation function at :attr:`r`
        
    h : instance of :class:`~halo_model.HaloModel`
        This instance will be updated with the variables of the minimization.
        Other desired options should have been set upon instantiation.
        
    sd : array_like
        The uncertainty in the measured correlation function, same length as 
        :attr:`r`.
    
    covar : 2d array, default ``None``
        Covariance matrix of the data. Either `sd` or `covar` must be given,
        but if both are given, `covar` takes precedence.
           
    priors : list of prior classes
        A list containing instances of :class:`.Uniform`, :class:`.Normal` or 
        :class:`.MultiNorm` classes. These specify the prior information on each 
        parameter.
        
    guess : array_like, default []
        Where to start the chain. If empty, will get central values from the
        distributions.
        
    verbose : int, default 0
        The verbosity level. 
        
    method : str, default ``"Nelder-Mead"``
        The optimizing routine (see `scipy.optimize.minimize` for details).
        
    disp : bool, default False
        Whether to display optimization information while running.
        
    maxiter : int, default 30
        Maximum number of iterations
        
    tol : float, default None
        Tolerance for termination
        
    Returns
    -------
    res : instance of :class:`scipy.optimize.Result`
        Contains the results of the minimization. Important attributes are the
        solution vector :attr:`x`, the number of iterations :attr:`nit`, whether
        the minimization was a success :attr:`success`, and the exit message 
        :attr:`message`.
         
    """
    # Save which attributes are updatable for HOD as a list
    attrs = []
    for prior in priors:
        if isinstance(prior.name, basestring):
            attrs += [prior.name]
        else:
            attrs += prior.name


    # Set guess if not set
    if len(guess) != len(attrs):
        guess = []
        for prior in priors:
            if isinstance(prior, Uniform):
                guess += [(prior.high + prior.low) / 2]
            elif isinstance(prior, Normal):
                guess += [prior.mean]
            elif isinstance(prior, MultiNorm):
                guess += prior.means.tolist()

    guess = np.array(guess)

    def negmod(*args):
        return -model(*args)

    res = minimize(negmod, guess, (priors, h, attrs, data, sd, covar, verbose), tol=tol,
                   method=method, options={"disp":disp, "maxiter":maxiter})

    return res

# _vars = ["wdm_mass", "delta_halo", "sigma_8", "n", "omegab", 'omegac',
#          "omegav", "omegak", "H0", "M_1", 'alpha', "M_min", 'gauss_width',
#          'M_0', 'fca', 'fcb', 'fs', 'delta', 'x', 'omegab_h2', 'omegac_h2', 'h']

#===============================================================================
# Function for getting initial positions
#===============================================================================
def get_initial_pos(guess, priors, nwalkers, find_peak_first=False, data=None,
                    sd=None, h=None, verbose=0, **kwargs):

    # Set guess if not set
    if not guess:
        for prior in priors:
            if isinstance(prior, Uniform):
                guess += [(prior.high + prior.low) / 2]
            elif isinstance(prior, Normal):
                guess += [prior.mean]
            elif isinstance(prior, MultiNorm):
                guess += prior.means.tolist()

    guess = np.array(guess)

    if find_peak_first:
        res = fit_hod_minimize(data, sd, priors, h, guess=guess,
                               verbose=verbose, **kwargs)
        guess = res.x

    # Get an initial value for all walkers, around a small ball near the initial guess
    stacked_val = guess.copy()
    for i in range(nwalkers - 1):
        stacked_val = np.vstack((guess, stacked_val))

    i = 0
    for prior in priors:
        if isinstance(prior, Uniform):
            stacked_val[:, i] += np.random.normal(loc=0.0, scale=0.05 *
                                                  min((guess[i] - prior.low),
                                                      (prior.high - guess[i])),
                                                  size=nwalkers)
            i += 1
        elif isinstance(prior, Normal):
            stacked_val[:, i] += np.random.normal(loc=0.0, scale=prior.sd,
                                                  size=nwalkers)
            i += 1
        elif isinstance(prior, MultiNorm):
            for j in range(len(prior.name)):
                stacked_val[:, i] += np.random.normal(loc=0.0, scale=np.sqrt(prior.cov[j, j]),
                                                      size=nwalkers)
                i += 1

    return stacked_val

#===============================================================================
# Write out sequential results
#===============================================================================
def write_iter(sampler, i, nwalkers, chunks, prefix, blobs, extend):
    # The reshaping and transposing here is important to get the output correct
    fc = np.transpose(sampler.chain[:, (i + 1 - chunks):i + 1, :], (1, 0, 2)).reshape((nwalkers * chunks, -1))
    ll = sampler.lnprobability[:, (i + 1 - chunks):i + 1].T.flatten()

    with open(prefix + "chain", "a") as f:
        np.savetxt(f, fc)

    with open(prefix + "likelihoods", "a") as f:
        np.savetxt(f, ll)

    if blobs:
        # All floats go together.
        ind_float = [ii for ii, b in enumerate(sampler.blobs[0][0]) if isinstance(b, Number)]
        if not extend and ind_float:
            with open(prefix + "derived_parameters", "w") as f:
                f.write("# %s\n" % ("\t".join([blobs[ii] for ii in ind_float])))

        if ind_float:
            numblobs = np.array([[[b[ii] for ii in ind_float] for b in c]
                                 for c in sampler.blobs[(i + 1 - chunks):i + 1]])

            # Write out numblobs
            sh = numblobs.shape
            numblobs = numblobs.reshape(sh[0] * sh[1], sh[2])
            with open(prefix + "derived_parameters", "a") as f:
                np.savetxt(f, numblobs)

        # Everything else gets treated with pickle
        pickledict = {}
        # If file already exists, read in those blobs first
        if extend:
            with open(prefix + "blobs", "r") as f:
                pickledict = pickle.load(f)

        # Append current blobs
        if len(ind_float) != len(blobs):
            ind_pickle = [ii for ii in range(len(blobs)) if ii not in ind_float]
            for ii in ind_pickle:
                if not pickledict:
                    pickledict[blobs[ii]] = []
                for c in sampler.blobs:
                    pickledict[blobs[ii]].append([b[ii] for b in c])

        # Write out pickle blobs
        if pickledict:
            with open(prefix + "blobs", 'w') as f:
                pickle.dump(pickledict, f)

#===============================================================================
# Some helpful functions to pass as dependent_params
#===============================================================================
def change_r(h, d, **kwargs):
    """
    Modifies the r...
    """
    input_z = kwargs['z']
    h.update(**d)
    r = cp.distance.comoving_distance(input_z, **h.cosmolopy_dict) * h.h
    # return rmin as vector, so we use correct values, not just logspace.
    return {"rmin":r}

#===============================================================================
# Classes for different prior models
#===============================================================================
class Uniform(object):
    """
    A Uniform prior.
    
    Parameters
    ----------
    param : str
        The name of the parameter
    
    low : float
        The lower bound of the parameter
        
    high : float
        The upper bound of the parameter
        
    """
    def __init__(self, param, low, high):
        self.name = param
        self.low = low
        self.high = high

class Log(Uniform):
    pass

class Normal(object):
    """
    A Gaussian prior.
    
    Parameters
    ----------
    param : str
        Name of the parameter
        
    mean : float
        Mean of the prior distribution
        
    sd : float
        The standard deviation of the prior distribution
    """
    def __init__(self, param, mean, sd):
        self.name = param
        self.mean = mean
        self.sd = sd

class MultiNorm(object):
    """
    A Multivariate Gaussian prior
    
    Parameters
    ----------
    params : list of str
        Names of the parameters (in order)
        
    means : list of float
        Mean vector of the prior distribution
        
    cov : ndarray
        Covariance matrix of the prior distribution
    """
    def __init__(self, params, means, cov):
        self.name = params
        self.means = means
        self.cov = cov

def _lognormpdf(x, mu, S):
    """ Log of Multinormal PDF at x, up to scale-factors."""
    err = x - mu
    return -0.5 * np.linalg.solve(S, err).T.dot(err)

