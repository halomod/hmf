'''
Created on Sep 10, 2013

@author: Steven

This module is a simple script that runs hmf, genmf 
and cosmolopy to get mass functions for different cosmologies 
and redshifts etc. It simply times them to compare.
'''

from cosmolopy import cp
import numpy as np
import matplotlib.pyplot as plt
import sys
from hmf import Perturbations
import time

redshifts = [0.0, 1.0]
fits = ["ST", "PS"]
s8s = [0.8, 0.9]
omegavs = [0.7, 0.6]

#===============================================================================
# Do hmf first
#===============================================================================
# First do a "fair" comparison of a single calculation
start = time.time()
pert = Perturbations(M=np.linspace(3, 17, 1401), k_bounds=np.exp([-21, 21]),
                     omegav=0.7, omegab=0.05, omegac=0.25, H0=70.0, n=1.0,
                     sigma_8=0.8)

#pert.dndlnm
pert.ngtm

time_hmf_1 = time.time() - start

#Now do a comparison of 2 redshifts, 2 fitting functions, 2 sigma_8s and 2 cosmos
start = time.time()
pert = Perturbations(M=np.linspace(3, 17, 1401), k_bounds=np.exp([-21, 21]),
                     omegav=0.7, omegab=0.05, omegac=0.25, H0=70.0, n=1.0,
                     sigma_8=0.8)
for z in redshifts:
    pert.update(z=z)
    pert.ngtm

for fit in fits:
    pert.update(mf_fit=fit)
    pert.ngtm

for s8 in s8s:
    pert.update(sigma_8=s8)
    pert.ngtm

for omegav in omegavs:
    pert.update(omegav=omegav, omegam=1 - omegav - 0.05)
    pert.ngtm

time_hmf_combo = time.time() - start


#===============================================================================
# Now do genmf
#===============================================================================
#First make the same power spectrum
start = time.time()
pert = Perturbations(M=np.linspace(3, 17, 1401), k_bounds=np.exp([-21, 21]),
                     omegav=0.7, omegab=0.05, omegac=0.25, H0=70.0, n=1.0,
                     sigma_8=0.8)
np.savetxt("data/code_comparison_power", np.vs)

