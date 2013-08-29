'''
Created on Aug 7, 2013

@author: Steven

Implements the Quadrature method with zeros of bessel functions to calculate the zeroth-order hankel transform of a function f()
'''
import numpy as np

def bessel_spher_0(x):
    """ The spherical bessel function of order 0 """
    return np.sin(x) / x

def roots(N):
    return (range(N) + 1) * 2 * np.pi

def bessel_spher_1(x):
    """The spherical besel function of order 1"""
    return np.sin(x) / x ** 2 - np.cos(x) / x

def weight(N):
    return 2 / (np.pi ** 2 * roots(N) * bessel_spher_1(np.pi * roots(N)))

def psi(t):
    return t * np.tanh(np.pi * np.sinh(t) / 2)

def d_psi(t):
    a = (np.pi * t * np.cosh(t) + np.sinh(np.pi * np.sinh(t))) / (1 + np.cosh(np.pi * np.sinh(t)));
    a[np.isnan(a)] = 1
    return a

def hankel_transform(f, N, h):
    """
    Performs a spherical hankel transform of order 0, returning the result and error estimate
    """
    r = roots(N)
    summation = weight(N) * f(np.pi * psi(h * r) / h) * bessel_spher_0(np.pi * psi(h * r) / h) * d_psi(h * r)
    return np.sum(summation) * np.pi, np.pi * summation[-1]
