"""
Created on 16/02/2015

@author: Steven
"""
import numpy as np
import inspect
import os
import sys

from src.hmf import MassFunction
from src.hmf import fit

try:
    import emcee

    HAVE_EMCEE = True
except ImportError:
    HAVE_EMCEE = False


RUN_MCMC = False

LOCATION = "/".join(
    os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split(
        "/"
    )[:-1]
)
sys.path.insert(0, LOCATION)


def test_circular_minimize():
    h = MassFunction(sigma_8=0.8, mf_fit="ST")
    dndm = h.dndm.copy()
    f = fit.Minimize(
        priors=[fit.Uniform("sigma_8", 0.6, 1.0)],
        data=dndm,
        quantity="dndm",
        sigma=dndm / 5,
        guess=[0.9],
        blobs=None,
        verbose=0,
        store_class=False,
        relax=False,
    )
    res = f.fit(h)
    print("Diff: ", np.abs(res.x - 0.8))
    assert np.abs(res.x - 0.8) < 0.01


if HAVE_EMCEE and RUN_MCMC:

    def test_circular_emcee():
        h = MassFunction(sigma_8=0.8, mf_fit="ST")
        dndm = h.dndm.copy()
        f = fit.MCMC(
            priors=[fit.Uniform("sigma_8", 0.6, 1.0)],
            data=dndm,
            quantity="dndm",
            sigma=dndm / 5,
            guess=[0.8],
            blobs=None,
            verbose=0,
            store_class=False,
            relax=False,
        )
        sampler = f.fit(h, nwalkers=16, nsamples=15, burnin=0, nthreads=0)
        print("Diff: ", np.abs(np.mean(sampler.chain) - 0.8))
        assert np.abs(np.mean(sampler.chain) - 0.8) < 0.01
