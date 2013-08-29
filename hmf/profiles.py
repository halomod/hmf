"""
A few halo density profiles and their normalised  fourier-transform pairs, along with concentration-mass relations.

Each density profile is a function of r,m,z. each fourier pair is a function of k,m,z and each mass relation is a function of mass, and outputs r_s as well if you want it to.
"""
import numpy as np
import scipy.special as sp

class profiles(object):

    def __init__(self, mean_dens, delta_vir, profile='nfw', cm_relation='duffy'):

        self.mean_dens = mean_dens
        self.delta_vir = delta_vir

        if profile == 'nfw':
            self.rho = self.rho_nfw
            self.u = self.u_nfw

        if cm_relation == 'duffy':
            self.cm_relation = self.cm_duffy

    def mvir_to_rvir(self, m):

        return (3 * m / (4 * np.pi * self.delta_vir * self.mean_dens)) ** (1. / 3.)


    def rho_nfw(self, r, m, z):

        c, r_s = self.cm_relation(m, z, get_rs=True)

        x = r / r_s

        return 1 / (x * (1 + x) ** 2)

    def u_nfw(self, k, m, z):
        c, r_s = self.cm_relation(m, z, get_rs=True)

        K = k * r_s

        asi, ac = sp.sici((1 + c) * K)
        bs, bc = sp.sici(K)

        return (np.sin(K) * (asi - bs) - np.sin(c * K) / ((1 + c) * K) + np.cos(K) * (ac - bc)) / (np.log(1 + c) - c / (1 + c))

    def cm_duffy(self, m, z, get_rs=True):
        c = 6.71 * (m / (2.0 * 10 ** 12)) ** -0.091 * (1 + z) ** -0.44

        rvir = self.mvir_to_rvir(m)

        if get_rs:
            return c, rvir / c
        else:
            return c






