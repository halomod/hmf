import inspect
import os
LOCATION = "/".join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split("/")[:-1])
import sys
sys.path.insert(0, LOCATION)
from hmf import filters
import numpy as np


class TestTopHat(object):
    def __init__(self):
        k = np.logspace(-6,0,10000)
        pk = k**2
        self.cls = filters.TopHat(k,pk)

    def test_sigma(self):
        R = 1.0
        true =9*(R**2*np.sin(R)**2/2 + R**2*np.cos(R)**2/2 + R*np.sin(R)*np.cos(R)/2 - np.sin(R)**2)/(2*np.pi**2*R**6)

        print true,self.cls.sigma(R)**2
        assert np.isclose(self.cls.sigma(R)[0]**2,true)

    def test_sigma1(self):
        R = 1.0
        true = 9*(R**2*np.sin(R)**2/6 + R**2*np.cos(R)**2/6 + R*np.sin(R)*np.cos(R)/2 - np.sin(R)**2/4 + 5*np.cos(R)**2/4 - 5*np.sin(R)*np.cos(R)/(4*R))/(2*np.pi**2*R**6)

        print true,self.cls.sigma(R,1)**2
        assert np.isclose(self.cls.sigma(R,1)[0]**2,true)
