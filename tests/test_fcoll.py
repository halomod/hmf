'''
Created on Sep 9, 2013

@author: Steven

This module provides some tests of mgtm/mean_dens against analytic f_coll.

As such, it is the best test of all calculations after sigma.
'''

from hmf import Perturbations
import numpy as np
from scipy.special import erfc

class TestFcoll(object):

    def check_fcoll(self, pert, fit):
        if fit == "PS":
            anl = fcoll_PS(pert.sigma, pert.cosmo_params['delta_c'])
            num = pert.mgtm / pert.cosmo_params['mean_dens']

        elif fit == "Peacock":
            anl = fcoll_Peacock(pert.sigma, pert.cosmo_params['delta_c'])
            num = pert.mgtm / pert.cosmo_params['mean_dens']

        err = np.abs((num - anl) / anl)
        print np.max(err)
        print num, anl
        assert np.max(err) < 0.05

    def test_fcolls(self):

        pert = Perturbations(M=np.linspace(10, 15, 1301))
        fits = ['PS', 'Peacock']

        for fit in fits:
            pert.update(mf_fit=fit)
            yield self.check_fcoll, pert, fit



def fcoll_PS(sigma, delta_c):
    return erfc(delta_c / sigma / np.sqrt(2))

def fcoll_Peacock(sigma, delta_c):
    nu = delta_c / sigma
    a = 1.529
    b = 0.704
    c = 0.412

    return (1 + a * nu ** b) ** -1 * np.exp(-c * nu ** 2)
