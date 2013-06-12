'''
Created on May 16, 2013

@author: Steven
'''
import nose
from nose import with_setup
from hmf.Perturbations import Perturbations
import numpy as np

redshifts = [ 0, 10]
delta_virs = [10, 1000, 1000000]
wdms = [0.0, 100]
ffs = ['PS', 'ST', 'Courtin', 'Jenkins', 'Warren', 'Reed03', 'Reed07', 'Angulo', 'Tinker',
        'Watson', 'Crocce', 'Bhattacharya', "Angulo_Bound", "Watson_FoF"]
alternate_models = ['sys.exit()', "x+1"]
kbounds = [[0.00001, 0.1], [10, 21]]
mrange = [[3, 18]]
delta_cs = [1, 3]
ns = [-4, 3]
sigma_8s = [0.1, 1000]
H0s = [10, 500]
omegabs = [0.005, 0.65]
omegacs = [0.02, 2.0]
omegavs = [0, 1.6]
omegans = [0, 0.7]

def test_default():

    M = np.linspace(8, 15, 70)
    pert = Perturbations(M)
    assert check_k(pert.k)
    assert check_mf(pert.MassFunction())
    assert check_vfv(pert.vfv)

def test_z():
    M = np.linspace(8, 15, 70)
    pert = Perturbations(M, z=redshifts[0])

    assert check_k(pert.k)
    assert check_mf(pert.MassFunction())
    assert check_vfv(pert.vfv)

    for z in redshifts[1:]:
        pert.update(z=z)

        assert check_k(pert.k)
        assert check_mf(pert.MassFunction())
        assert check_vfv(pert.vfv)

def test_delta_virs():
    M = np.linspace(8, 15, 70)
    pert = Perturbations(M)

    for dv in delta_virs:
        assert check_mf(pert.MassFunction(overdensity=dv))
        assert check_vfv(pert.vfv)

def test_wdm():
    M = np.linspace(8, 15, 70)
    pert = Perturbations(M, wdm=wdms[0])

    assert check_k(pert.k)
    assert check_mf(pert.MassFunction())
    assert check_vfv(pert.vfv)

    for wdm in wdms[1:]:
        pert.update(wdm=wdm)

        assert check_k(pert.k)
        assert check_mf(pert.MassFunction())
        assert check_vfv(pert.vfv)

def test_ff():
    M = np.linspace(8, 15, 70)
    pert = Perturbations(M)

    for ff in ffs:
        assert check_mf(pert.MassFunction(fsigma=ff))
        assert check_vfv(pert.vfv)


def test_alternate_models():
    M = np.linspace(8, 15, 70)
    pert = Perturbations(M)

    assert check_k(pert.k)
    assert check_mf(pert.MassFunction())
    assert check_vfv(pert.vfv)

    for am in alternate_models:
       assert check_mf(pert.MassFunction(user_model=am))
       assert check_vfv(pert.vfv)

def test_kbounds():
    M = np.linspace(8, 15, 70)
    pert = Perturbations(M, k_bounds=kbounds[0])

    assert check_k(pert.k)
    assert check_mf(pert.MassFunction())
    assert check_vfv(pert.vfv)

    for bounds in kbounds[1:]:
        pert.update(k_bounds=bounds)

        assert check_k(pert.k)
        assert check_mf(pert.MassFunction())
        assert check_vfv(pert.vfv)

def test_mrange():
    for m in mrange:
        M = np.arange(m[0], m[1], 0.1)
        pert = Perturbations(M)

        assert check_k(pert.k)
        assert check_mf(pert.MassFunction())
        assert check_vfv(pert.vfv)

def test_deltac():
    M = np.linspace(8, 15, 70)
    pert = Perturbations(M, delta_c=delta_cs[0])

    assert check_k(pert.k)
    assert check_mf(pert.MassFunction())
    assert check_vfv(pert.vfv)

    for dc in delta_cs[1:]:
        assert check_mf(pert.MassFunction(delta_c=dc))
        assert check_vfv(pert.vfv)

def test_n():
    M = np.linspace(8, 15, 70)
    pert = Perturbations(M, n=ns[0])

    assert check_k(pert.k)
    assert check_mf(pert.MassFunction())
    assert check_vfv(pert.vfv)

    for n in ns[1:]:
        pert.update(n=n)

        assert check_k(pert.k)
        assert check_mf(pert.MassFunction())
        assert check_vfv(pert.vfv)

def test_sigma_8():
    M = np.linspace(8, 15, 70)
    pert = Perturbations(M, sigma_8=sigma_8s[0])

    assert check_k(pert.k)
    assert check_mf(pert.MassFunction())
    assert check_vfv(pert.vfv)

    for s8 in sigma_8s[1:]:
        pert.update(sigma_8=s8)

        assert check_k(pert.k)
        assert check_mf(pert.MassFunction())
        assert check_vfv(pert.vfv)

def test_H0():
    M = np.linspace(8, 15, 70)
    pert = Perturbations(M, H0=H0s[0])

    assert check_k(pert.k)
    assert check_mf(pert.MassFunction())
    assert check_vfv(pert.vfv)

    for h in H0s[1:]:
        pert.update(H0=h)

        assert check_k(pert.k)
        assert check_mf(pert.MassFunction())
        assert check_vfv(pert.vfv)

def test_ob():
    M = np.linspace(8, 15, 70)
    pert = Perturbations(M, omegab=omegabs[0])

    assert check_k(pert.k)
    assert check_mf(pert.MassFunction())
    assert check_vfv(pert.vfv)

    for ob in omegabs[1:]:
        pert.update(omegab=ob)

        assert check_k(pert.k)
        assert check_mf(pert.MassFunction())
        assert check_vfv(pert.vfv)

def test_oc():
    M = np.linspace(8, 15, 70)
    pert = Perturbations(M, omegac=omegacs[0])

    assert check_k(pert.k)
    assert check_mf(pert.MassFunction())
    assert check_vfv(pert.vfv)

    for oc in omegacs[1:]:
        pert.update(omegac=oc)

        assert check_k(pert.k)
        assert check_mf(pert.MassFunction())
        assert check_vfv(pert.vfv)

def test_ov():
    M = np.linspace(8, 15, 70)
    pert = Perturbations(M, omegav=omegavs[0])

    assert check_k(pert.k)
    assert check_mf(pert.MassFunction())
    assert check_vfv(pert.vfv)

    for ov in omegavs[1:]:
        pert.update(omegav=ov)

        assert check_k(pert.k)
        assert check_mf(pert.MassFunction())
        assert check_vfv(pert.vfv)

def test_on():
    M = np.linspace(8, 15, 70)
    pert = Perturbations(M, omegan=omegans[0])

    assert check_k(pert.k)
    assert check_mf(pert.MassFunction())
    assert check_vfv(pert.vfv)

    for on in omegans[1:]:
        pert.update(omegan=on)

        assert check_k(pert.k)
        assert check_mf(pert.MassFunction())
        assert check_vfv(pert.vfv)

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



def check_k(k):

    return True  #k[1] - k[0] == k[2] - k[1]

def check_mf(mf):
    return np.all(np.logical_or(mf > 0, np.isnan(mf)))

def check_vfv(vfv):
    return np.all(np.logical_or(vfv < 1 , np.isnan(vfv)))



