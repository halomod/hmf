'''
Created on 16/02/2015

@author: Steven
'''
#===============================================================================
# IMPORTS
#===============================================================================
import numpy as np
import inspect
import os
LOCATION = "/".join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split("/")[:-1])
import sys
sys.path.insert(0, LOCATION)
from hmf import MassFunction
from hmf import fit

def test_circular_minimize():
    h = MassFunction(sigma_8=0.8, mf_fit="ST")
    dndm = h.dndm.copy()
    f = fit.Minimize(priors=[fit.Uniform("sigma_8", 0.6, 1.0)],
                     data=dndm, quantity="dndm", sigma=dndm / 5,
                     guess=[0.9], blobs=None,
                     verbose=0, store_class=False, relax=False)
    res = f.fit(h)
    print "Diff: ", np.abs(res.x - 0.8)
    assert np.abs(res.x - 0.8) < 0.01
