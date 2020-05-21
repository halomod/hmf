"""
Created on 27/02/2015

@author: Steven Murray
"""

import sys
import os
from configparser import ConfigParser as cfg
from functools import reduce
import numpy as np
from . import fit
import json
import time
import errno
from multiprocessing import cpu_count
from os.path import join
import warnings
import pickle
import copy
from emcee.interruptible_pool import InterruptiblePool

from ..density_field import transfer_models as tm

cfg.optionxform = str


def secondsToStr(t):
    return "%d:%02d:%02d.%03d" % reduce(
        lambda ll, b: divmod(ll[0], b) + ll[1:], [(t * 1000,), 1000, 60, 60]
    )


class CLIError(Exception):
    """Generic exception to raise and log different fatal errors."""

    def __init__(self, msg):
        super(CLIError).__init__(type(self))
        self.msg = "E: %s" % msg

    def __str__(self):
        return self.msg

    def __unicode__(self):
        return self.msg


def import_class(cl):
    d = cl.rfind(".")
    classname = cl[d + 1 : len(cl)]
    m = __import__(cl[0:d], globals(), locals(), [classname])
    return getattr(m, classname)


class CLIRunner(object):
    """
    A class which imports and interprets a config file and runs a fit.
    """

    def __init__(self, config, prefix="", restart=False, verbose=0):

        self.verbose = verbose

        self.prefix = prefix
        if self.prefix and not self.prefix.endswith("."):
            self.prefix += "."

        # READ CONFIG FILE
        # NOTE param_dict just contains variables of the actual fit.
        param_dict = self.read_config(config)

        # ## Make output directory
        if self.outdir:
            try:
                os.makedirs(self.outdir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        self.full_prefix = join(self.outdir, self.prefix)

        # ## Import observed data
        self.x, self.y, self.sigma = self.get_data()

        # Get params that are part of a dict
        self.priors, self.keys, self.guess = self.param_setup(param_dict)

        self.sampler = self._get_previous_sampler() if restart else None

    def read_config(self, fname):
        config = cfg()
        config.read(fname)

        # Convert config to a dict
        res = {s: dict(config.items(s)) for s in config.sections()}
        if "outdir" not in res["IO"]:
            res["IO"]["outdir"] = ""
        if "covar_data" not in res["cosmo_paramsParams"]:
            res["cosmo_paramsParams"]["covar_data"] = ""

        # Run Options
        self.quantity = res["RunOptions"].pop("quantity")
        self.xval = res["RunOptions"].pop("xval")
        self.framework = res["RunOptions"].pop("framework")
        self.relax = bool(res["RunOptions"].pop("relax"))
        self.nthreads = int(res["RunOptions"].pop("nthreads"))

        # Derived params and quantities
        dparams = json.loads(res["RunOptions"].pop("der_params"))
        dquants = json.loads(res["RunOptions"].pop("der_quants"))

        self.n_dparams = len(dparams)
        self.blobs = dparams + dquants

        # Fit-options
        self.fit_type = res["FitOptions"].pop("fit_type")

        # MCMC-specific
        self.nwalkers = int(res["MCMC"].pop("nwalkers"))
        self.nsamples = int(res["MCMC"].pop("nsamples"))
        self.burnin = json.loads(res["MCMC"].pop("burnin"))

        # Downhill-specific
        self.downhill_kwargs = {
            k: json.loads(v) for k, v in list(res["Downhill"].items())
        }
        del res["Downhill"]

        # IO-specific
        self.outdir = res["IO"].pop("outdir", None)
        self.chunks = int(res["IO"].pop("chunks"))

        # Data-specific
        self.data_file = res["Data"].pop("data_file")
        self.cov_file = res["Data"].pop("cov_file", None)

        # Model-specific
        self.model = res.pop("Model")
        self.model_pickle = self.model.pop("model_file", None)
        for k in self.model:
            self.model[k] = json.loads(self.model[k])

        self.constraints = {
            k: json.loads(v) for k, v in list(res["Constraints"].items())
        }

        return {k: res.pop(k) for k in list(res.keys()) if k.endswith("Params")}

    def get_data(self):
        """
        Import the data to be compared to (both data and var/covar)

        Returns
        -------
        float array:
            array of x values.

        float array:
            array of y values.

        float array or None:
            Standard Deviation of y values or None if covariance is provided

        float array or None:
            Covariance of y values, or None if not provided.
        """
        data = np.genfromtxt(self.data_file)

        x = data[:, 0]
        y = data[:, 1]

        sigma = np.genfromtxt(self.cov_file) if self.cov_file else None
        if sigma is None:
            try:
                sigma = data[:, 2]
            except IndexError:
                raise ValueError(
                    """
Either a univariate standard deviation, or multivariate cov matrix must be provided.
        """
                )

        return x, y, sigma

    def param_setup(self, params):
        """
        Takes a dictionary of input parameters, with keys defining the parameters
        and the values defining various aspects of the priors, and converts them
        to usable Prior() instances, along with keys and guesses.

        Note that here, *only* cosmological parameters are able to be set as
        multivariate normal priors (this is not true in general, but for the CLI
        it is much simpler). All other parameters may be set as Normal or Uniform
        priors.

        Returns
        -------
        priors : list
            A list of Prior() classes corresponding to each parameter specified.
            Names in these will be prefixed by "<dict>:" for parameters required
            to pass to dictionaries.

        keys : list
            A list of of parameter names (without prefixes)

        guess : list
            A list containing an initial guess for each parameter.
        """
        # Set-up returned lists of parameters
        priors = []
        keys = []

        # Get covariance data for the cosmology (ie. name of CMB mission if provided)
        covdata = params["cosmo_paramsParams"].pop("covar_data", None)
        if covdata:
            try:
                cosmo_cov = getattr(sys.modules["hmf.fitting.fit"], covdata)
            except AttributeError:
                raise AttributeError("%s is not a valid cosmology dataset" % covdata)
            except Exception:
                raise

        # Deal specifically with cosmology priors, separating types
        original_cpars = copy.copy(params["cosmo_paramsParams"])
        cosmo_priors = {
            k: json.loads(v) for k, v in list(params["cosmo_paramsParams"].items())
        }
        # the following rely on covdata
        cov_vars = {k: v for k, v in list(cosmo_priors.items()) if v[0] == "cov"}
        norm_vars = {
            k: v
            for k, v in list(cosmo_priors.items())
            if (v[0] == "norm" and len(v) == 2)
        }

        # remove these to be left with normal stuff
        for k in list(cov_vars.keys()) + list(norm_vars.keys()):
            del params["cosmo_paramsParams"][k]

        # sigma_8 and n are special cosmology parameters that don't nest
        if "sigma_8" in params["cosmo_paramsParams"]:
            params["OtherParams"]["sigma_8"] = params["cosmo_paramsParams"].pop(
                "sigma_8"
            )
        if "n" in params["cosmo_paramsParams"]:
            params["OtherParams"]["n"] = params["cosmo_paramsParams"].pop("n")

        # Now we should have all actual cosmo_params labelled as such in the
        # `params` dict, with sigma_8 and n in OtherParams as necessary.

        if cov_vars:
            priors += [cosmo_cov.get_cov_prior(*cov_vars)]
        if norm_vars:
            priors += cosmo_cov.get_normal_priors(*norm_vars)

        # All non-cosmology-covariance-dependent stuff that is top-level
        otherparams = params["OtherParams"]
        for param, val in list(otherparams.items()):
            priors += self.set_prior(param, val)

        # All non-cosmology-covariance-dependent stuff that is nested
        for k, v in list(params.items()):
            if k != "OtherParams":
                for kk, vv in list(v.items()):
                    priors += self.set_prior(k[:-6] + ":" + kk, vv)

        # Create list of all the names of parameters (pure name without :)
        for prior in priors:
            if isinstance(prior.name, str):
                keys += [prior.name]
            else:
                keys += prior.name
        keys = [k.split(":")[-1] for k in keys]

        # for guessing, we need all keys back in params
        params["cosmo_paramsParams"] = original_cpars
        guess = self.get_guess(params, keys, priors)

        print(("KEY NAMES: ", keys))
        print(("INITIAL GUESSES: ", guess))

        return priors, keys, guess

    def get_guess(self, params, keys, priors):
        # Get all parameters to be set as a flat dictionary
        allparams = {}
        for pset, vset in list(params.items()):
            for p, val in list(vset.items()):
                allparams[p] = val

        # Get the guesses
        guess = []
        for i, p in enumerate(priors):
            if isinstance(p.name, list):
                # get raw input
                for j, k in enumerate(p.name):
                    val = json.loads(allparams[k.split(":")[-1]])
                    if val[-1] is None or len(val) == 1:
                        guess.append(p.guess(k))
                    else:
                        guess.append(val[-1])
            else:

                val = json.loads(allparams[p.name.split(":")[-1]])
                if val[-1] is None or len(val) == 1:
                    guess.append(p.guess())
                else:
                    guess.append(val[-1])

        return guess

    def set_prior(self, param, val):
        val = json.loads(val)
        if val[0] == "unif":
            x = fit.Uniform(param, val[1], val[2])
        elif val[0] == "norm":
            x = fit.Normal(param, val[1], val[2])
        elif val[0] == "log":
            x = fit.Log(param, val[1], val[2])

        return [x]

    def _get_previous_sampler(self):
        """
        Tries to find a pickled sampler in the current directory to use.
        """
        try:
            with open(self.prefix + "sampler.pickle") as f:
                h = pickle.load(f)
            # check that it lines up with current Parameters
            if h.k != self.nwalkers:
                warnings.warn(
                    f"Imported previous chain had different number of walkers {h.k} "
                    f"than specified {self.nwalkers}"
                )
                return None
            else:
                # WE DO THE FOLLOWING IN FIT.PY, BUT HAVE TO DO IT HERE TO update
                # THE PICKLED OBJECT, SINCE THE POOL CANNOT BE SAVED
                if h.args[0].transfer_fit in ["CAMB", tm.CAMB] and any(
                    p.startswith("cosmo_params:") for p in self.keys
                ):
                    nthreads = 1

                if not nthreads:
                    # auto-calculate the number of threads to use if not set.
                    nthreads = cpu_count()

                if nthreads != 1:
                    h.pool = InterruptiblePool(nthreads)
                return h
        except Exception:
            return None

    def _setup_x(self, instance):
        if self.xval == "M":
            assert np.allclose(np.diff(np.diff(np.log10(self.x))), 0)
            dlog10m = np.log10(self.x[1] / self.x[0])
            instance.update(
                Mmin=np.log10(self.x[0]),
                Mmax=np.log10(self.x[-1]) + 0.2 * dlog10m,
                dlog10m=dlog10m,
            )
        elif self.xval == "k":
            assert np.allclose(np.diff(np.diff(np.log10(self.x))), 0)
            dlnk = np.log(self.x[1] / self.x[0])
            instance.update(
                lnk_min=np.log(self.x[0]),
                lnk_max=np.log(self.x[-1]) + 0.2 * dlnk,
                dlnk=dlnk,
            )

        return instance

    def _setup_instance(self):
        if self.model_pickle:
            with open(self.model_pickle) as f:
                instance = pickle.load(f)
        else:
            # Create the proper framework
            instance = import_class(self.framework)(**self.model)

        # Set up x-variable in Framework
        instance = self._setup_x(instance)

        # pre-get the quantity
        getattr(instance, self.quantity)

        # Write out a pickle file of the model
        with open(self.full_prefix + "model.pickle", "w") as f:
            pickle.dump(instance, f)

        return instance

    def run(self):
        if self.fit_type == "opt":
            self.run_downhill()
        else:
            self.run_mcmc()

    def run_downhill(self, instance=None):
        """
        Runs a simple downhill-gradient fit.
        """
        fitter = fit.Minimize(
            priors=self.priors,
            data=self.y,
            quantity=self.quantity,
            constraints=self.constraints,
            sigma=self.sigma,
            guess=self.guess,
            blobs=self.blobs,
            verbose=self.verbose,
            relax=self.relax,
        )

        if instance is None:
            instance = self._setup_instance()

        result = fitter.fit(instance, **self.downhill_kwargs)
        print(("Optimization Result: ", result))

        self._write_opt_log(result)
        return result

    def run_mcmc(self):
        """
        Runs the MCMC fit
        """
        if self.sampler is not None:
            instance = None
            prev_samples = self.sampler.iterations
        else:
            instance = self._setup_instance()
            if self.fit_type == "both":
                optres = self.run_downhill(instance)
                if optres.success:
                    self.guess = list(optres.x)
            prev_samples = 0

        self._write_log_pre()

        fitter = fit.MCMC(
            priors=self.priors,
            data=self.y,
            quantity=self.quantity,
            constraints=self.constraints,
            sigma=self.sigma,
            guess=self.guess,
            blobs=self.blobs,
            verbose=self.verbose,
            relax=self.relax,
        )

        start = time.time()
        if self.chunks == 0:
            self.chunks = self.nsamples - prev_samples
        nchunks = np.ceil((self.nsamples - prev_samples) / float(self.chunks))
        for i, s in enumerate(
            fitter.fit(
                self.sampler,
                instance,
                self.nwalkers,
                self.nsamples - prev_samples,
                self.burnin,
                self.nthreads,
                self.chunks,
            )
        ):
            # Write out files
            self.write_iter_pickle(s)
            print(
                (
                    "Done {0}%. Time per sample: {1}".format(
                        100 * float(i + 1) / nchunks,
                        (time.time() - start) / ((i + 1) * self.chunks * self.nwalkers),
                    )
                )
            )

        total_time = time.time() - start

        self._write_log_post(s, total_time)
        self._write_data(s)

    def write_iter_pickle(self, sampler):
        """
        Write out a pickle version of the sampler every chunk.
        """
        with open(self.full_prefix + "sampler.pickle", "w") as f:
            pickle.dump(sampler, f)

    def _write_opt_log(self, result):
        with open(self.full_prefix + "opt.log", "w") as f:
            for k, r in zip(self.keys, result.x):
                f.write("%s: %s\n" % (k, r))
            f.write(str(result))
            # if hasattr(result,"hess_inv"):
            #     f.write("Covariance Matrix:\n")
            #     f.write(str(result.hess_inv))
            # f.write("Success: %s\n"%result.success)
            # f.write("Iterations Required: %s\n"%result.nit)
            # f.write("Func. Evaluations: %s\n"%result.nfev)
            # f.write("Message: %s\n"%result.message)

    def _write_data(self, sampler):
        """
        Writes out chains and other data to longer-term readable files (ie ASCII)
        """
        with open(self.full_prefix + "chain", "w") as f:
            np.savetxt(f, sampler.flatchain, header="\t".join(self.keys))

        with open(self.full_prefix + "likelihoods", "w") as f:
            np.savetxt(f, sampler.lnprobability.T)

        # We can write out any blobs that are parameters
        if self.blobs:
            if self.n_dparams:
                numblobs = np.array(
                    [
                        [[b[ii] for ii in range(self.n_dparams)] for b in c]
                        for c in sampler.blobs
                    ]
                )

                # Write out numblobs
                sh = numblobs.shape
                numblobs = numblobs.reshape(sh[0] * sh[1], sh[2])
                with open(self.full_prefix + "derived_parameters", "w") as f:
                    np.savetxt(
                        f,
                        numblobs,
                        header="\t".join(
                            [self.blobs[ii] for ii in range(self.n_dparams)]
                        ),
                    )

    def _write_log_pre(self):
        with open(self.full_prefix + "log", "w") as f:
            f.write("Nsamples:  %s\n" % self.nsamples)
            f.write("Nwalkers: %s\n" % self.nwalkers)
            f.write("Burnin: %s\n" % self.burnin)
            f.write("Parameters: %s\n" % self.keys)

    def _write_log_post(self, sampler, total_time):
        with open(self.full_prefix + "log", "a") as f:
            f.write("Total Time: %s\n" % secondsToStr(total_time))
            if isinstance(self.burnin, int):
                f.write(
                    "Average time: %s\n"
                    % (
                        total_time
                        / (self.nwalkers * self.nsamples + self.nwalkers * self.burnin)
                    )
                )
            else:
                f.write(
                    "Average time (discounting burnin): %s\n"
                    % (total_time / (self.nwalkers * self.nsamples))
                )
            f.write("Mean values = %s\n" % np.mean(sampler.chain, axis=0))
            f.write("Std. Dev = %s\n" % np.std(sampler.chain, axis=0))
            f.write("Covariance Matrix: %s\n" % np.cov(sampler.flatchain.T))
            f.write("Acceptance Fraction: %s\n" % sampler.acceptance_fraction)
            f.write("Acorr: %s\n" % json.dumps(sampler.acor.tolist()))
