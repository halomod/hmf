'''
This script explores the projected differences between using an EH and CAMB
transfer function. 
'''

from hmf import Perturbations
import time
import numpy as np

omegab = [0.02, 0.05, 0.1]
omegac = [0.2, 0.25, 0.5]
H0 = [50, 70, 90]
n = [0.8, 0.9, 1.0]


pert_camb = Perturbations(transfer_fit="CAMB")
pert_EH = Perturbations(transfer_fit="EH")
camb_time = 0.0
eh_time = 0.0
for ob in omegab:
    for oc in omegac:
        for h in H0:
            for nn in n:
                pert_camb.update(omegab=ob, omegac=oc, H0=h, n=nn)
                pert_EH.update(omegab=ob, omegac=oc, H0=h, n=nn)

                start = time.time()
                camb = pert_camb.dndm
                camb_time += time.time() - start

                start = time.time()
                eh = pert_EH.dndm
                eh_time += time.time() - start

                print "For ", ob, oc, h, nn
                print camb[0] / eh[0], camb[250] / eh[250], camb[500] / eh[500], eh_time / camb_time
