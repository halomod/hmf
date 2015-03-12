'''
Created on 27/02/2015

@author: Steven Murray
'''

import sys
import os
from ConfigParser import SafeConfigParser as cfg
cfg.optionxform = str
import numpy as np
import fit
import json
import time
import errno
from os.path import join
import warnings
from emcee import autocorr
import pickle

class CLIError(Exception):
    '''Generic exception to raise and log different fatal errors.'''
    def __init__(self, msg):
        super(CLIError).__init__(type(self))
        self.msg = "E: %s" % msg
    def __str__(self):
        return self.msg
    def __unicode__(self):
        return self.msg

def import_class(cl):
    d = cl.rfind(".")
    classname = cl[d + 1:len(cl)]
    m = __import__(cl[0:d], globals(), locals(), [classname])
    return getattr(m, classname)

class CLIRunner(object):
    """
    A class which imports and interprets a config file and runs a fit.
    """

    def __init__(self, config, prefix="", restart=False, verbose=0):

        self.verbose = verbose

        self.prefix = prefix
        if self.prefix:
            if not self.prefix.endswith("."):
                self.prefix += "."

        ### READ CONFIG FILE ###
        # NOTE param_dict just contains variables of the actual fit.
        param_dict = self.read_config(config)

        # ## Make output directory
        if self.outdir:
            try:
                os.makedirs(self.outdir)
            except OSError, e:
                if e.errno != errno.EEXIST:
                    raise

        self.full_prefix = join(self.outdir, self.prefix)

        # ## Import observed data
        self.x, self.y, self.sigma = self.get_data()

        # Get params that are part of a dict (eg. HOD)

        self.priors, self.keys, self.guess = self.param_setup(param_dict)



        if restart:
            self.initial, self.prev_samples = self.get_initial()
        else:
            self.initial, self.prev_samples = None, 0


    def read_config(self, fname):
        config = cfg()
        config.read(fname)

        # Convert config to a dict
        res = {s:dict(config.items(s)) for s in config.sections()}
        if "outdir" not in res["IO"]:
            res["IO"]["outdir"] = ""
        if "covar_data" not in res["cosmo_paramsParams"]:
            res["cosmo_paramsParams"]['covar_data'] = ''

        # Set simple parameters
        self.quantity = res["RunOptions"].pop("quantity")
        self.xval = res["RunOptions"].pop("xval")
        self.blobs = json.loads(res["RunOptions"].pop("der_params"))
        self.framework = res["RunOptions"].pop("framework")
        self.relax = bool(res["RunOptions"].pop("relax"))
        self.nthreads = int(res["RunOptions"].pop("nthreads"))

        self.nwalkers = int(res["MCMC"].pop("nwalkers"))
        self.nsamples = int(res["MCMC"].pop("nsamples"))
        self.burnin = json.loads(res["MCMC"].pop("burnin"))


        self.outdir = res["IO"].pop("outdir", None)
        self.chunks = int(res["IO"].pop("chunks"))
        self.verbose = int(res["IO"].pop("verbose"))

        self.data_file = res["Data"].pop("data_file")
        self.cov_file = res["Data"].pop("cov_file", None)

        self.model = res.pop("Model")
        self.model_pickle = self.model.pop("model_file", None)
        for k in self.model:
            try:
                self.model[k] = json.loads(self.model[k])
            except:
                pass

        self.constraints = {k:json.loads(v) for k, v in res["Constraints"].iteritems()}

        param_dict = {k:res.pop(k) for k in res.keys() if k.endswith("Params")}
        return param_dict

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

        if self.cov_file:
            sigma = np.genfromtxt(self.cov_file)
        else:
            sigma = None

        if sigma is None:
            try:
                sigma = data[:, 2]
            except IndexError:
                raise ValueError("""
Either a univariate standard deviation, or multivariate cov matrix must be provided.
        """)

        return x, y, sigma

    def param_setup(self, params):
        """
        Takes a dictionary of input parameters, with keys defining the parameters
        and the values defining various aspects of the priors, and converts them
        to useable Prior() instances, along with keys and guesses.
        
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
                cosmo_cov = getattr(sys.modules["hmf.fit"], covdata)
            except AttributeError:
                raise AttributeError("%s is not a valid cosmology dataset" % covdata)
            except Exception:
                raise

        # Deal specifically with cosmology priors, separating types
        cosmo_priors = {k:json.loads(v) for k, v in params["cosmo_paramsParams"].iteritems()}
        # the following rely on covdata
        cov_vars = {k:v for k, v in cosmo_priors.iteritems() if v[0] == "cov"}
        norm_vars = {k:v for k, v in cosmo_priors.iteritems() if (v[0] == "norm" and len(v) == 2)}
        # remove these to be left with normal stuff
        for k in cov_vars.keys() + norm_vars.keys():
            del params["cosmo_paramsParams"][k]

        if cov_vars:
            priors += cosmo_cov.get_cov_prior(*cov_vars)
        if norm_vars:
            priors += cosmo_cov.get_normal_priors(*norm_vars)

        # sigma_8 and n are special cosmology parameters that don't nest
        if "sigma_8" in params["cosmo_paramsParams"]:
            params["OtherParams"]["sigma_8"] = params["cosmo_paramsParams"].pop("sigma_8")
        if "n" in params["cosmo_paramsParams"]:
            params["OtherParams"]["n"] = params["cosmo_paramsParams"].pop("n")

        # All non-cosmology-covariance-dependent stuff that is top-level
        otherparams = params["OtherParams"]
        for param, val in otherparams.iteritems():
            priors += self.set_prior(param, val)


        # All non-cosmology-covariance-dependent stuff that is nested
        for k, v in params.iteritems():
            if k != "OtherParams":
                for kk, vv in v.iteritems():
                    priors += self.set_prior(k[:-6] + ":" + kk, vv)

        # Create list of all the names of parameters (pure name without :)
        for prior in priors:
            if isinstance(prior.name, basestring):
                keys += [prior.name]
            else:
                keys += prior.name
        keys = [k.split(":")[-1] for k in keys]

        guess = self.get_guess(params, keys, priors)

        print "KEY NAMES: ", keys
        print "INITIAL GUESSES: ", guess

        return priors, keys, guess

    def get_guess(self, params, keys, priors):
        # Get all parmeters to be set as a flat dictionary
        allparams = {}
        for pset, vset in params.iteritems():
            for p, val in vset.iteritems():
                allparams[p] = val

            # Get the guesses
        guess = []
        for i, k in enumerate(keys):
            val = json.loads(allparams[k])
            if val[-1] is None:
                guess.append(priors[i].guess(k))
            else:
                guess.append(val[-1])
        return guess

    def set_prior(self, param, val):
        val = json.loads(val)
        if val[0] == 'unif':
            x = fit.Uniform(param, val[1], val[2])
        elif val[0] == 'norm':
            x = fit.Normal(param, val[1], val[2])
        elif val[0] == "log":
            x = fit.Log(param, val[1], val[2])

        return [x]

    def get_initial(self):
        """
        Tries to find a chain in the current directory to use.
        """
        try:
            x = np.genfromtxt(self.prefix + "chain")
            nsamples = x.shape[0]
            return x[-self.nwalkers:, :], nsamples
        except:
            warnings.warn("Problem importing old file, starting afresh")
            return None, 0

    def _setup_x(self, instance):
        if self.xval == "M":
            assert np.allclose(np.diff(np.diff(np.log10(self.x))), 0)
            dlog10m = np.log10(self.x[1] / self.x[0])
            instance.update(Mmin=np.log10(self.x[0]), Mmax=np.log10(self.x[-1]) + 0.2 * dlog10m, dlog10m=dlog10m)
        elif self.xval == "k":
            assert np.allclose(np.diff(np.diff(np.log10(self.x))), 0)
            dlnk = np.log(self.x[1] / self.x[0])
            instance.update(lnk_min=np.log(self.x[0]), lnk_max=np.log(self.x[-1]) + 0.2 * dlnk, dlnk=dlnk)

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
        q = getattr(instance, self.quantity)

        # Apply the units of the quantity to the data
        if hasattr(q, "unit"):
            self.y *= q.unit
            self.sigma *= q.unit ** len(self.sigma.shape)

        # Apply units of constraints
        for k in self.constraints:
            unit = getattr(getattr(instance, k), "unit", None)
            if unit:
                self.constraints[k][0] *= unit
                self.constraints[k][1] *= unit

        return instance

    def run(self):
        """
        Runs the MCMC fit
        """
        instance = self._setup_instance()

        # # Write out a pickle file.
        with open(self.full_prefix + "model.pickle", 'w') as f:
            pickle.dump(instance, f)

        start = time.time()
        fitter = fit.MCMC(priors=self.priors, data=self.y, quantity=self.quantity,
                          constraints=self.constraints, sigma=self.sigma,
                          guess=self.guess, blobs=self.blobs,
                          verbose=self.verbose, relax=self.relax)

        s = fitter.fit(instance, nwalkers=self.nwalkers, nsamples=self.nsamples,
                       burnin=self.burnin, nthreads=self.nthreads,
                       prefix=self.full_prefix, chunks=self.chunks,
                       initial_pos=self.initial)

        # Grab acceptance fraction from initial run if possible
        new_accepted = np.mean(s.acceptance_fraction) * self.nsamples * self.nwalkers

        if self.initial is not None:
            try:
                with open(self.full_prefix + "log", 'r') as f:
                    for line in f:
                        if line.startswith("Acceptance Fraction:"):
                            af = float(line[20:])
                            naccepted = af * self.prev_samples
            except IOError:
                naccepted = 0
                self.prev_samples = 0
        else:
            naccepted = 0
        acceptance = (naccepted + new_accepted) / (self.prev_samples + self.nwalkers * self.nsamples)

        # Ditch the sampler from memory
        del s

        # Read in total chain
        chain = np.genfromtxt(self.full_prefix + "chain").reshape((self.nwalkers, self.nsamples, -1))
        acorr = autocorr.integrated_time(np.mean(chain, axis=0), axis=0,
                                         window=50, fast=False)
        chain = chain.reshape((self.nwalkers * self.nsamples, -1))
        # Write out the logfile
        with open(self.full_prefix + "log", 'w') as f:
            if isinstance(self.burnin, int):
                f.write("Average time: %s\n" % ((time.time() - start) / (self.nwalkers * self.nsamples + self.nwalkers * self.burnin)))
            else:
                f.write("Average time (discounting burnin): %s\n" % ((time.time() - start) / (self.nwalkers * self.nsamples)))
            f.write("Nsamples:  %s\n" % self.nsamples)
            f.write("Nwalkers: %s\n" % self.nwalkers)
            f.write("Mean values = %s\n" % np.mean(chain, axis=0))
            f.write("Covariance Matrix: %s\n" % np.cov(chain.T))
            f.write("Acceptance Fraction: %s\n" % acceptance)
            f.write("Acorr: %s\n" % json.dumps(acorr.tolist()))


