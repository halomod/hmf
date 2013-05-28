'''
Created on May 22, 2013

@author: Steven
'''

import timeit

setup_setup = """
from hmf.Perturbations import Perturbations
import numpy as np
"""
setup_time = timeit.timeit('M = np.linspace(10,15,50); Perturbations(M)', setup=setup_setup, number=10)

run_setup = """
from hmf.Perturbations import Perturbations
import numpy as np
M = np.linspace(10, 15, 50);
pert = Perturbations(M)
"""
mf_time = timeit.repeat('pert.MassFunction()', setup=run_setup, number=200, repeat=3)

set_transfer_cosmo_time = timeit.repeat('pert.set_transfer_cosmo(H0=70.0)', setup=run_setup, number=30, repeat=3)
set_kbounds_time = timeit.repeat('pert.set_kbounds(sigma_8=0.8)', setup=run_setup, number=30, repeat=3)
set_WDM_time = timeit.repeat('pert.set_WDM(WDM=0.5)', setup=run_setup, number=30, repeat=3)
set_z_time = timeit.repeat('pert.set_z(z=1.0)', setup=run_setup, number=50, repeat=3)

print "For 50 M's:"
print "----------------------------------------------------"
print "setup             : ", setup_time
print "set_transfer_cosmo: ", min(set_transfer_cosmo_time)
print "set_kbounds       : ", min(set_kbounds_time)
print "set_WDM           : ", min(set_WDM_time)
print "set_z             : ", min(set_z_time)
print "----------------------------------------------------"
