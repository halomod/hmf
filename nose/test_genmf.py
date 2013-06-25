'''
Created on Jun 20, 2013

@author: Steven
'''
import numpy as np
from hmf.Perturbations import Perturbations
import unittest

class TestRedshift0(unittest.TestCase):

    def __init__(self):
        self.genmf = np.genfromtxt("data/final_testing_out_z0_om3_ol7_s88_fsigST.dat")[::-1]
        self.M = np.linspace(3, 17, 1401)
        self.pert = Perturbations(self.M, omegab=0.05, omegac=0.25, omegav=0.7, sigma_8=0.8, n=1, k_bounds=[np.exp(-21), np.exp(21)])
        self.mf = self.pert.MassFunction(fsigma="ST")

   # def teardown(self):
   #     pass


    def test_sigma(self):
        print np.mean(np.abs(self.pert.sigma - self.genmf[:, 5]) / self.pert.sigma)
        print (self.pert.sigma / self.genmf[:, 5])[:20]
        assert np.mean(np.abs(self.pert.sigma - self.genmf[:, 5]) / self.pert.sigma) < 0.01

    def test_lnsigma(self):
        print np.mean(np.abs(self.pert.lnsigma - self.genmf[:, 3]) / self.pert.lnsigma)
        assert np.mean(np.abs(self.pert.lnsigma - self.genmf[:, 3]) / self.pert.lnsigma) < 0.01

    def test_n_eff(self):
        print np.mean(np.abs(self.pert.n_eff - self.genmf[:, 6]) / self.pert.n_eff)
        assert np.mean(np.abs(self.pert.n_eff - self.genmf[:, 6]) / self.pert.n_eff) < 0.01

    def test_f(self):
        print np.mean(np.abs(self.pert.vfv - self.genmf[:, 4]) / self.pert.vfv)
        print (self.pert.vfv / self.genmf[:, 4])[:20]
        assert np.mean(np.abs(self.pert.vfv - self.genmf[:, 4]) / self.pert.vfv) < 0.01

    def test_hmf(self):
        a = np.mean(np.abs(self.mf - 10 ** self.genmf[:, 1]) / self.mf)
        print a
        assert a < 0.01

    def test_ngtm(self):
        ngtm = self.pert.NgtM(self.mf)
        a = np.mean(np.abs(ngtm - self.genmf[:, 2]) / ngtm)
        print a
        assert  a < 0.01

    def test_PS(self):
        mf = self.pert.MassFunction(fsigma="PS")

        genmf = np.genfromtxt("data/final_testing_out_z0_om3_ol7_s88_fsigPS.dat")[:, 1]

        a = np.mean(np.abs(mf - genmf) / mf)
        print a
        assert a < 0.01

#    def test_Jenkins(self):
#        mf = self.pert.MassFunction(fsigma="Jenkins")
#        genmf = np.genfromtxt("data/final_testing_out_z0_om3_ol7_s88_fsigJ.dat")[:, 1]
#        a = np.mean(np.abs(mf - genmf) / mf)
#        print a
#        assert a < 0.01

    def test_Warren(self):
        mf = self.pert.MassFunction(fsigma="Warren")
        genmf = np.genfromtxt("data/final_testing_out_z0_om3_ol7_s88_fsigW.dat")[:, 1]
        a = np.mean(np.abs(mf - genmf) / mf)
        print a
        assert a < 0.01

    def test_R03(self):
        mf = self.pert.MassFunction(fsigma="Reed03")
        genmf = np.genfromtxt("data/final_testing_out_z=0_om3_ol7_s88_fsigR03.dat")[:, 1]
        a = np.mean(np.abs(mf - genmf) / mf)
        print a
        assert a < 0.01

    def test_R07(self):
        mf = self.pert.MassFunction(fsigma="Reed07")
        genmf = np.genfromtxt("data/final_testing_out_z=0_om3_ol7_s88_fsigR07.dat")[:, 1]
        a = np.mean(np.abs(mf - genmf) / mf)
        print a
        assert a < 0.01

    def test_PS_n(self):
        mf = self.pert.MassFunction(fsigma="PS")
        mf = self.pert.NgtM(mf)
        genmf = np.genfromtxt("data/final_testing_out_z0_om3_ol7_s88_fsigPS.dat")[:, 2]
        a = np.mean(np.abs(mf - genmf) / mf)
        print a
        assert a < 0.01

#    def test_Jenkins_n(self):
#        mf = self.pert.MassFunction(fsigma="Jenkins")
#        mf = self.pert.NgtM(mf)
#        genmf = np.genfromtxt("data/final_testing_out_z0_om3_ol7_s88_fsigJ.dat")[:, 2]
#        a = np.mean(np.abs(mf - genmf) / mf)
#        print a
#        assert a < 0.01

    def test_Warren_n(self):
        mf = self.pert.MassFunction(fsigma="Warren")
        mf = self.pert.NgtM(mf)
        genmf = np.genfromtxt("data/final_testing_out_z0_om3_ol7_s88_fsigW.dat")[:, 2]
        a = np.mean(np.abs(mf - genmf) / mf)
        print a
        assert  a < 0.01

    def test_R03_n(self):
        mf = self.pert.MassFunction(fsigma="Reed03")
        mf = self.pert.NgtM(mf)
        genmf = np.genfromtxt("data/final_testing_out_z0_om3_ol7_s88_fsigR03.dat")[:, 2]
        a = np.mean(np.abs(mf - genmf) / mf)
        print a
        assert  a < 0.01

    def test_R07_n(self):
        mf = self.pert.MassFunction(fsigma="Reed07")
        mf = self.pert.NgtM(mf)
        genmf = np.genfromtxt("data/final_testing_out_z0_om3_ol7_s88_fsigR07.dat")[:, 2]
        a = np.mean(np.abs(mf - genmf) / mf)
        print a
        assert  a < 0.01


class TestRedshift_2:

    def __init__(self):
        self.genmf = np.genfromtxt("data/final_testing_out_z2_om3_ol7_s88_fsigST.dat")
        self.M = np.linspace(3, 17, 1401)
        self.pert = Perturbations(self.M)
        self.pert.update(omegab=0.05, omegac=0.25, omegav=0.7, sigma_8=0.8, n=1, k_bounds=[np.exp(-21), np.exp(21)], z=2)
        self.mf = self.pert.MassFunction(fsigma="ST")



    def test_sigma(self):
        a = np.mean(np.abs(self.pert.sigma - self.genmf[:, 5]) / self.pert.sigma)
        print a
        assert a < 0.01

    def test_lnsigma(self):
        a = np.mean(np.abs(self.pert.lnsigma - self.genmf[:, 3]) / self.pert.lnsigma)
        print a
        assert a < 0.01

    def test_n_eff(self):
        a = np.mean(np.abs(self.pert.n_eff - self.genmf[:, 6]) / self.pert.n_eff)
        print a
        assert  a < 0.01

    def test_f(self):
        a = np.mean(np.abs(self.pert.vfv - self.genmf[:, 4]) / self.pert.vfv)
        print a
        assert  a < 0.01

    def test_hmf(self):
        a = np.mean(np.abs(self.mf - self.genmf[:, 1]) / self.mf)
        print a
        assert a < 0.01

    def test_ngtm(self):
        ngtm = self.pert.NgtM(self.mf)
        a = np.mean(np.abs(ngtm - self.genmf[:, 2]) / ngtm)
        print a
        assert  a < 0.01


if __name__ == '__main__':
    unittest.main()
