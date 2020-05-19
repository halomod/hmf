"""
Created on Oct 14, 2013

@author: Steven

This module contains numerous functions for fitting HOD models to correlation
function data. It uses MCMC techniques to do so.
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from multiprocessing import cpu_count
import time
import warnings

import copy
import traceback
from ..density_field import transfer_models as tm


try:
    from emcee import EnsembleSampler as es

    HAVE_EMCEE = True

    # The following redefines the EnsembleSampler so that the pool object is not
    # pickled along with it (it can't be).
    def should_pickle(k):
        return k != "pool"

    class EnsembleSampler(es):
        def __getstate__(self):
            return {k: v for (k, v) in list(self.__dict__.items()) if should_pickle(k)}


except ImportError:
    HAVE_EMCEE = False


def model(parm, h, self):
    """
    Calculate the log probability of a model `h`
    [instance of :class:`hmf._framework.Framework`] with parameters ``parm``.

    At the moment, this is a little hacky, because the parameters have to
    be the first argument (for both Minimize and MCMC), so we use a
    function and pass self last.

    Parameters
    ----------
    parm : list of floats
        The position of the model. Takes arbitrary parameters.

    h : instance of :class:`~_framework.Framework`
        An instance of any subclass of :class:`~_framework.Framework` with the
        desired options set. Variables of the estimation are updated within the
        routine.

    Returns
    -------
    ll : float
        The log likelihood of the model at the given position.
    """
    if self.verbose > 1:
        print(("Params: ", list(zip(self.attrs, parm))))

    ll = 0
    p = copy.copy(parm)
    for prior in self.priors:
        if type(prior.name) == list:
            index = [self.attrs.index(name) for name in prior.name]
        else:
            index = self.attrs.index(prior.name)
        ll += prior.ll(parm[index])
    if np.isinf(ll):
        return ret_arg(ll, self.blobs)

    # If it is a log distribution, un-log it for use.
    if isinstance(prior, Log):
        p[index] = 10 ** parm[index]

    # Rebuild the hod dict from given vals
    # Any attr starting with <name>: is put into a dictionary.
    param_dict = {}
    for attr, val in zip(self.attrs, p):
        if ":" in attr:
            if attr.split(":")[0] not in param_dict:
                param_dict[attr.split(":")[0]] = {}

            param_dict[attr.split(":")[0]][attr.split(":")[1]] = val
        else:
            param_dict[attr] = val

    # Update the actual model
    try:  # This try: except: should capture poor parameter choices quickly.
        h.update(**param_dict)
    except ValueError as e:
        if self.relax:
            print(
                (
                    "WARNING: PARAMETERS FAILED ON UPDATE, RETURNING INF: ",
                    list(zip(self.attrs, parm)),
                )
            )
            print(e)
            print((traceback.format_exc()))
            return ret_arg(-np.inf, self.blobs)
        else:
            print((traceback.format_exc()))
            raise e

    # Get the quantity to compare (if exceptions are raised, treat properly)
    try:
        q = [getattr(h, qq) for qq in self.quantities]
    except Exception as e:
        if self.relax:
            print(
                (
                    "WARNING: PARAMETERS FAILED WHEN CALCULATING QUANTITY, RETURNING INF: ",
                    list(zip(self.attrs, parm)),
                )
            )
            print(e)
            print((traceback.format_exc()))
            return ret_arg(-np.inf, self.blobs)
        else:
            print((traceback.format_exc()))
            raise e

    ll += self.ll_func(q, self.data, **self.usr_kwargs)

    if self.verbose:
        print(("Likelihood: ", ll))
    if self.verbose > 1:
        print(("Update Dictionary: ", param_dict))
    if self.verbose > 2:
        print(("Final Quantities: ", q))

    # Get blobs to return as well.
    if self.blobs is not None:
        out = []
        for b in self.blobs:
            if ":" not in b:
                out.append(getattr(h, b))
            elif ":" in b:
                out.append(getattr(h, b.split(":")[0])[b.split(":")[1]])
        return ll, out
    else:
        return ll


def chi_squared(q, data, sigma):
    """
    A simple log-likelihood function which takes a list of data, along with a list of models at that data, and an
    uncertainty at each of those points, and does chi-squared likelihood

    Parameters
    ----------
    q : list
        A list of model values that correspond to the data

    data : list
        A list of data values that correspond to q

    sigma : list
        A list of uncertainties on the data. Each entry can be a single number, a 1d array, or a 2D array if the data
        is covariant.

    Returns
    -------
    ll : float
        The log-likelihood of the model.

    """

    ll = 0
    for qq, dd, ss in zip(q, data, sigma):
        if len(ss.shape) == 2:
            ll += _lognormpdf(qq, dd, ss)
        else:
            ll += np.sum(norm.logpdf(dd, loc=qq, scale=ss))

    return ll


def ret_arg(ll, blobs):
    if blobs is None:
        return ll
    else:
        return ll, blobs


class Fit(object):
    """
    Parameters
    ----------
    priors : list of prior classes
        A list containing instances of :class:`.Prior` subclasses. These specify
        the prior information on each parameter.

    data : array_like
        The data to be compared to -- must be the same length as the intended
        quantity. Also must be free from NaN values or a ValueError will be raised.

    quantity : str
        The quantity to be compared (eg. ``"dndm"``)

    constraints : dict
        A dictionary with keys being quantity names, and values being length 2
        collections with element 0 the desired value of the quantity, and
        element 1 the uncertainty. This is used in addition to the data to
        calculate the likelihood

    sigma : array_like
        If a vector, this is taken to be the standard deviation of the data. If
        a matrix, it is taken to be the covariance matrix of the data.

    blobs : list of str
        Names of quantities to be returned along with the chain. Must be a

    verbose : int, default 0
        How much to write to screen.

    relax : bool, default False
        If relax is true, the call to get the quantity is wrapped in a try:except:.
        If an error occurs, the lognorm is set to -inf, rather than raising an exception.
        This can be helpful if a flat prior is used on cosmology, for which extreme
        values can sometimes cause exceptions.
    """

    def __init__(
        self,
        priors,
        data,
        quantities,
        ll_func=chi_squared,
        ll_kwargs={},
        guess=[],
        blobs=None,
        verbose=0,
        relax=False,
    ):
        if len(priors) == 0:
            raise ValueError("priors must be at least length 1")
        else:
            self.priors = priors

        # Save which attributes are updatable as a list
        self.attrs = []
        for prior in self.priors:
            if isinstance(prior.name, str):
                self.attrs += [prior.name]
            else:
                self.attrs += prior.name

        # Get the number of variables for MCMC
        self.ndim = len(self.attrs)

        # Ensure guess was set correctly.
        if guess and len(guess) != self.ndim:
            warnings.warn("Guess was set incorrectly: %s" % guess)
            guess = []

        self.guess = self.get_guess(guess)
        if np.any(np.isnan(data)):
            raise ValueError("The data must contain no NaN values")

        self.ll_func = ll_func
        self.usr_kwargs = ll_kwargs
        self.data = data
        self.quantities = quantities
        self.blobs = blobs
        self.verbose = verbose
        self.relax = relax

    def get_guess(self, guess):
        # Set guess if not set
        if not guess:
            for prior in self.priors:
                if isinstance(prior, Uniform):
                    guess += [(prior.high + prior.low) / 2]
                elif isinstance(prior, Normal):
                    guess += [prior.mean]
                elif isinstance(prior, MultiNorm):
                    guess += prior.means.tolist()
        return np.array(guess)

    def model(self, p, h):
        return model(p, h, self)


class MCMC(Fit):
    def __init__(self, *args, **kwargs):
        if not HAVE_EMCEE:
            raise TypeError(
                "You need emcee to use this class, aborting. ['pip install emcee']"
            )

        super(MCMC, self).__init__(*args, **kwargs)

    def fit(
        self,
        sampler=None,
        h=None,
        nwalkers=100,
        nsamples=100,
        burnin=0,
        nthreads=0,
        chunks=None,
    ):
        """
        Estimate the parameters in :attr:`.priors` using AIES MCMC.

        This routine uses the emcee package to run an MCMC procedure, fitting
        parameters passed in :attr:`.priors` to the given quantity.

        Parameters
        ----------
        sampler : instance of :class:`EnsembleSampler`
            A sampler instance, which may already include samples from a previous
            run.

        h : instance of :class:`~hmf._framework.Framework` subclass, optional
            This instance will be updated with the variables of the minimization.
            Other desired options should have been set upon instantiation.
            Needed if `sampler` not present.

        nwalkers : int
            Number of walkers to use for Affine-Invariant Ensemble Sampler.

        nsamples : int, optional
            Number of samples that *each walker* will perform.

        burnin : int, optional
            Number of samples from each walker that will be initially erased as
            burnin. Note, this performs *additional* iterations, rather than
            consuming iterations from `nsamples`.

        nthreads : int, optional
            Number of threads to use in sampling. If nought, will automatically
            detect number of cores available.

        chunks : int, optional
            Number of samples to run before appending results to file. Only
            applicable if :attr:`.filename` is provided.


        Yields
        ------
        sampler : :class:`EnsembleSampler` object
            The full sampling object, with chain, blobs, acceptance fraction etc.
        """
        if sampler is None and h is None:
            raise ValueError("Either sampler or h must be given")

        # If using CAMB, nthreads MUST BE 1
        if h.transfer_model == "CAMB" or h.transfer_model == tm.CAMB:
            if any(p.startswith("cosmo_params:") for p in self.attrs):
                nthreads = 1

        if not nthreads:
            # auto-calculate the number of threads to use if not set.
            nthreads = cpu_count()

        # This just makes sure that the caching works
        [getattr(h, qq) for qq in self.quantities]

        initial_pos = None
        if sampler is not None:
            if sampler.iterations > 0:
                initial_pos = sampler.chain[:, -1, :]
        else:
            # Note, sampler CANNOT be an attribute of self, since self is passed to emcee.
            sampler = EnsembleSampler(
                nwalkers, self.ndim, model, args=[h, self], threads=nthreads
            )

        # Get initial positions
        if initial_pos is None:
            initial_pos = self.get_initial_pos(nwalkers)

        # Run a burn-in
        # If there are some samples already in the sampler, only run the difference.
        if burnin:
            initial_pos, lnprob, rstate, blobs0 = self._run_burnin(burnin, initial_pos)
        else:
            lnprob = None
            rstate = None
            blobs0 = None

        # Run the actual run
        if chunks == 0 or chunks > nsamples or chunks is None:
            chunks = nsamples

        for i, result in enumerate(
            sampler.sample(
                initial_pos,
                iterations=nsamples,
                lnprob0=lnprob,
                rstate0=rstate,
                blobs0=blobs0,
            )
        ):
            if (i + 1) % chunks == 0 or i + 1 == nsamples:
                yield sampler
        #
        # self.__sampler = sampler

    # def get_and_del_sampler(self):
    #     """
    #     Returns the sampler object if it exists (ie. fit has been called) and deletes it.
    #
    #     This must be used to get the sampler if no chunks are being used. That is
    #
    #     ```
    #     F = MCMC(...)
    #     F.fit()
    #     sampler = F.get_and_del_sampler()
    #     ```
    #
    #     After being assigned, it is deleted since it cannot exist in the class
    #     when `.fit()` is called.
    #     """
    #     sampler = self.__sampler
    #     del self.__sampler
    #     return sampler

    def get_initial_pos(self, nwalkers):
        # Get an initial value for all walkers, around a small ball near the initial guess
        stacked_val = self.guess.copy()
        for i in range(nwalkers - 1):
            stacked_val = np.vstack((self.guess, stacked_val))

        i = 0
        for prior in self.priors:
            if isinstance(prior, Uniform):
                stacked_val[:, i] += np.random.normal(
                    loc=0.0,
                    scale=0.05
                    * min((self.guess[i] - prior.low), (prior.high - self.guess[i])),
                    size=nwalkers,
                )
                i += 1
            elif isinstance(prior, Normal):
                stacked_val[:, i] += np.random.normal(
                    loc=0.0, scale=prior.sd, size=nwalkers
                )
                i += 1
            elif isinstance(prior, MultiNorm):
                for j in range(len(prior.name)):
                    stacked_val[:, i] += np.random.normal(
                        loc=0.0, scale=np.sqrt(prior.cov[j, j]), size=nwalkers
                    )
                    i += 1

        return stacked_val

    def _run_burnin(self, burnin, initial_pos):
        if type(burnin) == int:
            if burnin - self.sampler.iterations > 0:
                initial_pos, lnprob, rstate, blobs0 = self.sampler.run_mcmc(
                    initial_pos, burnin - self.sampler.iterations
                )
                self.sampler.reset()
        else:
            if burnin[0] - self.sampler.iterations > 0:
                initial_pos, lnprob, rstate, blobs0 = self.sampler.run_mcmc(
                    initial_pos, burnin[0] - self.sampler.iterations
                )
            else:
                return initial_pos, None, None, None

            it_needed = burnin[1] * np.max(self.sampler.acor)
            while (
                it_needed > self.sampler.iterations or it_needed < 0
            ):  # if negative, probably ran fewer samples than lag.
                initial_pos, lnprob, rstate, blobs0 = self.sampler.run_mcmc(
                    initial_pos, burnin[0] / 2
                )
                it_needed = burnin[1] * np.max(self.sampler.acor)
                if self.sampler.iterations > burnin[2]:
                    warnings.warn("Burnin FAILED... continuing (acor=%s)" % (it_needed))

            if self.verbose > 0:
                burnin = self.sampler.iterations
                print(("Used %s samples for burnin" % self.sampler.iterations))
            self.sampler.reset()
        return initial_pos, lnprob, rstate, blobs0


# ===========================================================
# Minimize Fitting Routine
# ===========================================================
class Minimize(Fit):
    def __init__(self, *args, **kwargs):
        super(Minimize, self).__init__(*args, **kwargs)
        self.original_blobs = self.blobs + []  # add [] to copy it
        self.blobs = None

    def fit(self, h, disp=False, maxiter=50, tol=None, **minimize_kwargs):
        r"""
        Run an optimization procedure to fit a model to data.

        Parameters
        ----------
        h : instance of :class:`~hmf.framework.Framework` subclass
            This instance will be updated with the variables of the minimization.
            Other desired options should have been set upon instantiation.

        method : str, default ``"Nelder-Mead"``
            The optimizing routine (see `scipy.optimize.minimize` for details).

        disp : bool, default False
            Whether to display optimization information while running.

        maxiter : int, default 30
            Maximum number of iterations

        tol : float, default None
            Tolerance for termination

        kwargs :
            Arguments passed directly to :func:`scipy.optimize.minimize`.

        Returns
        -------
        res : instance of :class:`scipy.optimize.Result`
            Contains the results of the minimization. Important attributes are the
            solution vector :attr:`x`, the number of iterations :attr:`nit`, whether
            the minimization was a success :attr:`success`, and the exit message
            :attr:`message`.

        """
        # try to set some bounds
        bounds = []
        for p in self.priors:
            if type(p.name) is list:
                bounds += p.bounds()
            else:
                bounds.append(p.bounds())

        res = minimize(
            self.negmod,
            self.guess,
            (h,),
            tol=tol,
            options={"disp": disp, "maxiter": maxiter},
            **minimize_kwargs
        )
        if hasattr(res, "hess_inv"):
            self.cov_matrix = res.hess_inv

        return res

    def negmod(self, *args):
        ll = self.model(*args)
        if np.isinf(ll):
            return 1e30
        else:
            return -ll


# ===============================================================================
# Classes for different prior models
# ===============================================================================
class Prior(object):
    def ll(self, param):
        """
        Returns the log-likelihood of the given parameter given the Prior
        """
        pass

    def guess(self, *p):
        """
        Returns an "initial guess" for the prior
        """
        pass


class Uniform(Prior):
    r"""
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

    def ll(self, param):
        if param < self.low or param > self.high:
            return -np.inf
        else:
            return 0

    def guess(self, *p):
        return (self.low + self.high) / 2

    def bounds(self):
        return (self.low, self.high)


class Log(Uniform):
    pass


class Normal(Prior):
    r"""
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

    def ll(self, param):
        return norm.logpdf(param, loc=self.mean, scale=self.sd)

    def guess(self, *p):
        return self.mean

    def bounds(self):
        return (self.mean - 5 * self.sd, self.mean + 5 * self.sd)


class MultiNorm(Prior):
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

    def __init__(self, params, mean, cov):
        self.name = params
        self.mean = mean
        self.cov = cov

    def ll(self, params):
        """
        Here params should be a dict of key:values
        """
        # params = np.array([params[k] for k in self.name])
        return _lognormpdf(params, self.mean, self.cov)

    def guess(self, *p):
        """
        p should be the parameter name
        """
        return self.mean[self.name.index(p[0])]

    def bounds(self):
        return [
            (m + 5 * sd, m - 5 * sd)
            for m, sd in zip(self.mean, np.sqrt(np.diag(self.cov)))
        ]


def _lognormpdf(x, mu, S):
    """ Log of Multinormal PDF at x, up to scale-factors."""
    err = x - mu
    return -0.5 * np.linalg.solve(S, err).T.dot(err)


# ===============================================================================
# COVARIANCE DATA FROM CMB MISSIONS
# ===============================================================================
# Some data from CMB missions.
# All cov and mean data is in order of ["omegab_h2", "omegac_h2", "n", "sigma_8", "H0"]
class CosmoCovData(object):
    def __init__(self, cov, mean, params):
        self.cov = cov
        self.mean = mean
        self.params = params

    def get_cov(self, *p):
        """
        Return covariance matrix of given parameters *p
        """
        if any(str(pp) not in self.params for pp in p):
            raise AttributeError("One or more parameters passed are not in the data")

        indices = [self.params.index(str(k)) for k in p]
        return self.cov[indices, :][:, indices]

    def get_mean(self, *p):
        indices = [self.params.index(str(k)) for k in p]
        return self.mean[indices]

    def get_std(self, *p):
        cov = self.get_cov(*p)
        return np.sqrt([(cov[i, i]) for i in range(cov.shape[0])])

    def get_normal_priors(self, *p):
        std = self.get_std(*p)
        mean = self.get_mean(*p)
        return [
            Normal("cosmo_params:" + pp, m, s)
            if p not in ["sigma_8", "n"]
            else Normal(pp, m, s)
            for pp, m, s in zip(p, mean, std)
        ]

    def get_cov_prior(self, *p):
        cov = self.get_cov(*p)
        mean = self.get_mean(*p)
        p = ["cosmo_params:" + pp if pp not in ["sigma_8", "n"] else pp for pp in p]
        return MultiNorm(p, mean, cov)


class FlatCovData(CosmoCovData):
    def __init__(self, cov, mean):
        params = ["Om0", "Ob0", "sigma_8", "n", "H0"]
        super(FlatCovData, self).__init__(cov, mean, params)


WMAP3 = FlatCovData(
    cov=np.array(
        [
            [1.294e-03, 1.298e-04, 1.322e-03, -1.369e-04, -1.153e-01],
            [1.298e-04, 1.361e-05, 1.403e-04, -7.666e-06, -1.140e-02],
            [1.322e-03, 1.403e-04, 2.558e-03, 2.967e-04, -9.972e-02],
            [-1.369e-04, -7.666e-06, 2.967e-04, 2.833e-04, 2.289e-02],
            [-1.153e-01, -1.140e-02, -9.972e-02, 2.289e-02, 1.114e01],
        ]
    ),
    mean=np.array([2.409e-01, 4.182e-02, 7.605e-01, 9.577e-01, 7.321e01]),
)

WMAP5 = FlatCovData(
    cov=np.array(
        [
            [9.514e-04, 9.305e-05, 8.462e-04, -1.687e-04, -8.107e-02],
            [9.305e-05, 9.517e-06, 8.724e-05, -1.160e-05, -7.810e-03],
            [8.462e-04, 8.724e-05, 1.339e-03, 1.032e-04, -6.075e-02],
            [-1.687e-04, -1.160e-05, 1.032e-04, 2.182e-04, 2.118e-02],
            [-8.107e-02, -7.810e-03, -6.075e-02, 2.118e-02, 7.421e00],
        ]
    ),
    mean=np.array([2.597e-01, 4.424e-02, 7.980e-01, 9.634e-01, 7.180e01]),
)

WMAP7 = FlatCovData(
    cov=np.array(
        [
            [8.862e-04, 8.399e-05, 7.000e-04, -2.060e-04, -7.494e-02],
            [8.399e-05, 8.361e-06, 7.000e-05, -1.500e-05, -7.003e-03],
            [7.000e-04, 7.000e-05, 1.019e-03, 4.194e-05, -4.987e-02],
            [-2.060e-04, -1.500e-05, 4.194e-05, 2.103e-04, 2.300e-02],
            [-7.494e-02, -7.003e-03, -4.987e-02, 2.300e-02, 6.770e00],
        ]
    ),
    mean=np.array([2.675e-01, 4.504e-02, 8.017e-01, 9.634e-01, 7.091e01]),
)

WMAP9 = FlatCovData(
    cov=np.array(
        [
            [6.854e-04, 6.232e-05, 4.187e-04, -2.180e-04, -5.713e-02],
            [6.232e-05, 5.964e-06, 4.048e-05, -1.643e-05, -5.134e-03],
            [4.187e-04, 4.048e-05, 5.644e-04, -1.037e-05, -2.945e-02],
            [-2.180e-04, -1.643e-05, -1.037e-05, 1.766e-04, 2.131e-02],
            [-5.713e-02, -5.134e-03, -2.945e-02, 2.131e-02, 5.003e00],
        ]
    ),
    mean=np.array([2.801e-01, 4.632e-02, 8.212e-01, 9.723e-01, 6.998e01]),
)

Planck13 = FlatCovData(
    cov=np.array(
        [
            [3.884e-04, 3.017e-05, -1.508e-04, -1.619e-04, -2.834e-02],
            [3.017e-05, 2.459e-06, -9.760e-06, -1.236e-05, -2.172e-03],
            [-1.508e-04, -9.760e-06, 7.210e-04, 1.172e-04, 1.203e-02],
            [-1.619e-04, -1.236e-05, 1.172e-04, 8.918e-05, 1.196e-02],
            [-2.834e-02, -2.172e-03, 1.203e-02, 1.196e-02, 2.093e00],
        ]
    ),
    mean=np.array([3.138e-01, 4.861e-02, 8.339e-01, 9.617e-01, 6.741e01]),
)

Planck15 = FlatCovData(
    cov=np.array(
        [
            [1.021e-04, 8.034e-06, -5.538e-05, -4.492e-05, -7.479e-03],
            [8.034e-06, 6.646e-07, -3.924e-06, -3.542e-06, -5.803e-04],
            [-5.538e-05, -3.924e-06, 3.308e-04, 4.343e-05, 4.250e-03],
            [-4.492e-05, -3.542e-06, 4.343e-05, 2.940e-05, 3.291e-03],
            [-7.479e-03, -5.803e-04, 4.250e-03, 3.291e-03, 5.531e-01],
        ]
    ),
    mean=np.array([3.114e-01, 4.888e-02, 8.460e-01, 9.669e-01, 6.758e01]),
)
