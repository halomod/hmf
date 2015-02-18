'''
A collection of functions which do some of the core work of the HMF calculation.

The routines here could be made more 'elegant' by taking `MassFunction` or 
`Transfer` objects as arguments, but we keep them simple for the sake of 
flexibility.
'''

#===============================================================================
# Imports
#===============================================================================
import logging
logger = logging.getLogger('hmf')
#===============================================================================
# Functions
#===============================================================================
# def check_kr(min_m, max_m, mean_dens, mink, maxk):
#     """
#     Check the bounds of the product of k*r
#
#     If the bounds are not high/low enough, then there can be information loss
#     in the calculation of the mass variance. This routine returns a warning
#     indicating the necessary adjustment for requisite accuracy.
#
#     See http://arxiv.org/abs/1306.6721 for details.
#     """
#     # Define min and max radius
#     min_r = mass_to_radius(min_m, mean_dens)
#     max_r = mass_to_radius(max_m, mean_dens)
#
#     if np.exp(maxk) * min_r < 3:
#         logger.warn("r_min (%s) * k_max (%s) < 3. Mass variance could be inaccurate." % (min_r, np.exp(maxk)))
#     elif np.exp(mink) * max_r > 0.1:
#         logger.warn("r_max (%s) * k_min (%s) > 0.1. Mass variance could be inaccurate." % (max_r, np.exp(mink)))



