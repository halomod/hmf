'''
Created on May 16, 2013

@author: Steven
'''
import nose
from nose import with_setup
from hmf.Perturbations import Perturbations
import numpy as np

class testclass(object):

    M = np.linspace(8, 15, 70)
    pert = Perturbations(M)


    def test_the_test(self):
        i = 3
        assert i == 3

    def test_init_transfer_cosmo(self):
        items = ['w_lam', 'omegab', 'omegac', 'omegav',
                 'omegan', 'H0', 'cs2_lam', 'TCMB', 'yhe',
                 'Num_Nu_massless', 'omegak']
        for item in items:
            assert item in self.pert.transfer_cosmo

    def test_init_transfer_cosmo_reion(self):
        if self.pert.transfer_options['reion__use_optical_depth']:
            assert 'reion__optical_depth' in self.pert.transfer_cosmo
            assert 'reion__redshift' not in self.pert.transfer_cosmo
        else:
            assert 'reion__optical_depth' in self.pert.transfer_cosmo
            assert 'reion__redshift' not in self.pert.transfer_cosmo

    def test_init_transfer_options(self):
        items = ['Num_Nu_massive' , 'reion__fraction' , 'reion__delta_redshift',
                 'Scalar_initial_condition' , 'scalar_amp' , 'scalar_running',
                 'tensor_index', 'tensor_ratio', 'lAccuracyBoost', 'lSampleBoost',
                 'AccuracyBoost', 'WantScalars', 'WantTensors', 'reion__reionization',
                 'reion__use_optical_depth', 'w_perturb', 'DoLensing']
        for item in items:
            assert item in self.pert.transfer_options

    def test_init_dlogm(self):
        assert self.pert.dlogM > 0.09
        assert self.pert.dlogM < 1.1

    def test_init_transfer_file(self):
        assert self.pert.transfer_file is None

    def test_init_extrapolate(self):
        assert self.pert.extrapolate

    def test_init_z(self):
        assert self.pert.z == 0.0

    def test_init_wdm(self):
        assert self.pert.WDM is None


    def test_set_transfer_cosmo(self):
        self.pert.set_transfer_cosmo(omegav=0.7)
        assert self.pert.camb_dict['omegav'] == 0.7

        self.pert.set_transfer_cosmo(H0=80.0)
        assert self.pert.camb_dict['H0'] == 80.0

    def test_Setup(self):
        assert len(self.pert.k_original) > 10
        assert len(self.pert.k_original) == len(self.pert.Transfer)
        assert len(self.pert.Transfer.shape) == 1

    #def test_interpolate(self):
    #    tol = 0.01
    #    for i, item in enumerate(self.pert.Transfer):
    #        assert abs(self.pert.transfer_function(self.pert.k_original)[i] - item) < tol



