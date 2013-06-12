'''
Created on Apr 20, 2012

@author: Steven Murray
@contact: steven.murray@uwa.edu.au

Last Updated: 12 March 2013

This module contains 4 functions which all have the purpose of importing the
transfer function (or running CAMB on the fly) for
input to the Perturbations class (in Perturbations.py)

The functions are:
    1. SetParameter: modifies the params.ini file for use in CAMB with specified parameters
    2. ImportTransferFunction: Merely reads a CAMB transfer file and outputs ln(k) and ln(T).
    3. CAMB: runs CAMB through the OS (camb must be compiled previously)
    4. Setup: A driver for the above functions. Returns ln(k) and ln(T)
'''

###############################################################################
# Some simple imports
###############################################################################
import numpy as np

import pycamb

###############################################################################
# The function definitions
###############################################################################
def check_kR(min_m, max_m, mean_dens, mink, maxk):

    #Define mass from radius function
    def M(r):
        return 4 * np.pi * r ** 3 * mean_dens / 3

    #Define min and max radius
    min_r = (3 * min_m / (4 * np.pi * mean_dens)) ** (1. / 3.)
    max_r = (3 * max_m / (4 * np.pi * mean_dens)) ** (1. / 3.)

    errmsg1 = \
"""
Please make sure the product of minimum radius and maximum k is > 3.
If it is not, then the mass variance could be extremely inaccurate.
                    
"""

    errmsg2 = \
"""
Please make sure the product of maximum radius and minimum k is < 0.1
If it is not, then the mass variance could be inaccurate.
                    
"""

    if maxk * min_r < 3:
        error1 = errmsg1 + "This means extrapolating k to " + str(3 / min_r) + " or using min_M > " + str(np.log10(M(3.0 / maxk)))
    else:
        error1 = None

    if mink * max_r > 0.1:
        error2 = errmsg2 + "This means extrapolating k to " + str(0.1 / max_r) + " or using max_M < " + str(np.log10(M(0.1 / mink)))
    else:
        error2 = None

    return error1, error2

def ImportTransferFunction(transfer_file):
    """
    Imports the Transfer Function file to be analysed, and returns the pair ln(k), ln(T)
    
    Input: "transfer_file": full path to the file containing the transfer function (from camb).
    
    Output: ln(k), ln(T)
    """

    transfer = np.loadtxt(transfer_file)
    k = transfer[:, 0]
    T = transfer[:, 1]

    k = np.log(k)
    T = np.log(T)


    return k, T


def Setup(transfer_file, camb_dict):
    """
    A convenience function used to fully setup the workspace in the 'usual' way
    """
    #If no transfer file uploaded, but it was custom, execute CAMB
    if transfer_file is None:
        k, T, sig8 = pycamb.transfers(**camb_dict)
        T = np.log(T[1, :, 0])
        k = np.log(k)

    else:
        #Import the transfer file wherever it is.
        k, T = ImportTransferFunction(transfer_file)

    return k, T

