"""
Analytic functions for this test are defined in "analytic_filter.ipynb" in the development/ directory.
"""

import inspect
import os
LOCATION = "/".join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split("/")[:-1])
import sys
sys.path.insert(0, LOCATION)
from hmf import filters
import numpy as np
from numpy import sin, cos, pi
import warnings
from scipy.special import gamma

# Need to do the following to catch repeated warnings.
warnings.simplefilter("always",UserWarning)

class TestTopHat(object):
    def __init__(self):
        k = np.logspace(-6,0,10000)
        pk = k**2
        self.cls = filters.TopHat(k,pk)

    def test_sigma(self):
        R = 1.0
        true =(9*R**2*sin(R)**2/2 + 9*R**2*cos(R)**2/2 + 9*R*sin(R)*cos(R)/2 - 9*sin(R)**2)/(2*pi**2*R**6)

        print true,self.cls.sigma(R)**2
        assert np.isclose(self.cls.sigma(R)[0]**2,true)

    def test_sigma1(self):
        R = 1.0
        true = (3*R**2*sin(R)**2/2 + 3*R**2*cos(R)**2/2 + 9*R*sin(R)*cos(R)/2 - 9*sin(R)**2/4 + 45*cos(R)**2/4 - 45*sin(R)*cos(R)/(4*R))/(2*pi**2*R**6)


        print true,self.cls.sigma(R,1)**2
        assert np.isclose(self.cls.sigma(R,1)[0]**2,true)

    def test_dwdlnkr(self):
        x = 1.0
        true = x*(3*sin(x)/x**2 - 3*(-3*x*cos(x) + 3*sin(x))/x**4)
        assert np.isclose(self.cls.dw_dlnkr(x),true)

    def test_dlnssdlnr(self):
        R = 1.0
        true = 2*R**4*(-45*sin(R)**2/(4*R**2) - 27*cos(R)**2/(4*R**2) - 81*sin(R)*cos(R)/(4*R**3) + 27*sin(R)**2/R**4)/(9*R**2*sin(R)**2/2 + 9*R**2*cos(R)**2/2 + 9*R*sin(R)*cos(R)/2 - 9*sin(R)**2)

        print true, self.cls.dlnss_dlnr(R)
        assert np.isclose(self.cls.dlnss_dlnr(R),true)

class TestSharpK(object):
    def __init__(self):
        k = np.logspace(-6,0,10000)
        pk = k**2
        self.cls = filters.SharpK(k,pk)

    def test_sigma(self):
        R = 1.0
        t = 2 + 2 + 1
        true = 1./(2*pi**2 * t * R**t)

        print true,self.cls.sigma(R)**2
        assert np.isclose(self.cls.sigma(R)[0]**2,true)

    def test_sigma1(self):
        R = 1.0
        t = 4 + 2 + 1
        true = 1./(2*pi**2 * t * R**t)

        print true,self.cls.sigma(R,1)**2
        assert np.isclose(self.cls.sigma(R,1)[0]**2,true)


    def test_dlnssdlnr(self):
        R = 1.0
        t = 2 + 2 + 1
        sigma2 = 1./(2*pi**2 * t * R**t)
        true = -1./(2*pi**2*sigma2 * R**(3+2))

        print true, self.cls.dlnss_dlnr(R)
        assert np.isclose(self.cls.dlnss_dlnr(R),true)


    def test_sigma_R3(self):
        R = 3.0
        t = 2 + 2 + 1
        true = 1./(2*pi**2 * t * R**t)

        print true,self.cls.sigma(R)**2
        assert np.isclose(self.cls.sigma(R)[0]**2,true)

    def test_sigma1_R3(self):
        R = 3.0
        t = 4 + 2 + 1
        true = 1./(2*pi**2 * t * R**t)

        print true,self.cls.sigma(R,1)**2
        assert np.isclose(self.cls.sigma(R,1)[0]**2,true)


    def test_dlnssdlnr_R3(self):
        R = 3.0
        t = 2 + 2 + 1
        sigma2 = 1./(2*pi**2 * t * R**t)
        true = -1./(2*pi**2*sigma2 * R**(3+2))

        print true, self.cls.dlnss_dlnr(R)
        assert np.isclose(self.cls.dlnss_dlnr(R),true)


    def test_sigma_Rhalf(self):
        # should also raise a warning
        R = 0.5
        thisr = 1./self.cls.k.max()
        t = 2 + 2 + 1
        true = 1./(2*pi**2 * t * thisr**t)

        with warnings.catch_warnings(record=True) as w:
            s2 = self.cls.sigma(R)[0]**2
            assert w
        print s2,true
        assert np.isclose(s2,true)


    def test_sigma1_Rhalf(self):
        # should also raise a warning
        R = 0.5
        thisr = 1./self.cls.k.max()

        t = 4 + 2 + 1
        true = 1./(2*pi**2 * t * thisr**t)

        with warnings.catch_warnings(record=True) as w:
            s2 = self.cls.sigma(R,1)[0]**2
            assert w

        print s2,true
        assert np.isclose(s2,true)


    def test_dlnssdlnr_Rhalf(self):
        R = 3.0
        t = 2 + 2 + 1
        sigma2 = 1./(2*pi**2 * t * R**t)
        true = -1./(2*pi**2*sigma2 * R**(3+2))

        print true, self.cls.dlnss_dlnr(R)
        assert np.isclose(self.cls.dlnss_dlnr(R),true)


class TestGaussian(object):
    def __init__(self):
        k = np.logspace(-6,1,80)
        pk = k**2
        self.cls = filters.Gaussian(k,pk)


    def test_sigma(self):
        R = 10.0
        true = 3./(16*pi**(3./2.)*R**5)

        print true,self.cls.sigma(R)**2
        assert np.isclose(self.cls.sigma(R)[0]**2,true)

    def test_sigma1(self):
        R = 10.0
        true = 15/(32*pi**(3./2.)*R**7)

        print true,self.cls.sigma(R,1)**2
        assert np.isclose(self.cls.sigma(R,1)[0]**2,true)


    def test_dlnssdlnr(self):
        R = 10.0
        true = -5

        print true, self.cls.dlnss_dlnr(R)
        assert np.isclose(self.cls.dlnss_dlnr(R),true)
