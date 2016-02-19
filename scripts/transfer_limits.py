"""
This script calculates the transfer function using all four transfer fits for
different values of the density parameters, testing if realistic results are 
returned.

It seems that CAMB uses the physical constants to actual do most of the 
calculations -- ie. bounds in density space are constant with omegab_h2 etc, 
not omegab.
"""

from hmf import _Tr
import numpy as np
import matplotlib.pyplot as plt

def get_t(param, parm_name, other, **kwargs):
    if parm_name == "omegab_h2":
        t = _Tr(omegab_h2=param, omegac_h2=other, **kwargs)
    elif parm_name == "omegac_h2":
        t = _Tr(omegab_h2=other, omegac_h2=param, **kwargs)
    elif parm_name == "both":
        t = _Tr(omegab_h2=other*param, omegac_h2=(1 - other)*param, **kwargs)

    return t

def find_good(start, scale, parm_name, other, maxiter=20, **kwargs):
    param = start
    initial_scale = scale
    t = get_t(param, parm_name, other, **kwargs)
    rubbish = np.any(np.isnan(t._lnP_0)) or np.any(np.isinf(t._lnP_0))
    diff = 0.0
    first_non_rubbish = False
    last_was_rubbish = rubbish
    i = 0
    while rubbish or diff > 1e-4 or diff < 0.0:
        i += 1
        param += scale
        t = get_t(param, parm_name, other, **kwargs)
        rubbish = np.any(np.isnan(t._lnP_0)) or np.any(np.isinf(t._lnP_0))
        diff = scale * np.sign(initial_scale)
        if not rubbish:
            first_non_rubbish = True

        if first_non_rubbish:
            scale /= 1.5

        if last_was_rubbish != rubbish:
            scale = -scale
        last_was_rubbish = rubbish
        if i == maxiter:
            print "warning: maxiter reached"
            break

    return param

start = {"omegab_h2_lower":[0.0009, 0.001, 0.125],
         "omegab_h2_upper":[0.325, -0.05, 0.125],
         "omegac_h2_lower":[0.0, 0.01, 0.025],
         "omegac_h2_upper":[0.45, -0.05, 0.025],
         "both_lower":[0.006, 0.0005, 1. / 6.],
         "both_upper":[0.48, -0.05, 1. / 6.]}

# print """
# ===============================================================================
#         FLAT, h=1
# ===============================================================================
# """
# kwargs = {"force_flat":True, "h":1}
# for name in ["omegab_h2", "omegac_h2", "both"]:
#     for lim in ["lower", "upper"]:
#         k = name + "_" + lim
#         v = start[k]
#         good = find_good(v[0], v[1], name, v[2], **kwargs)
#         print "Good ", k, ": ", good

print """
===============================================================================
        FLAT, h=0.5
===============================================================================
"""
h = 0.5
kwargs = {"force_flat":True, "h":h}
start = {"omegab_h2_lower":[0.0009, 0.001, 0.25 * h ** 2],
         "omegab_h2_upper":[(1 - 0.25) * h ** 2, -0.05, 0.25 * h ** 2],
         "omegac_h2_lower":[0.0, 0.01, 0.05 * h ** 2],
         "omegac_h2_upper":[(1 - 0.05) * h ** 2, -0.05, 0.05 * h ** 2],
         "both_lower":[0.006, 0.0005, 1. / 6.],
         "both_upper":[h ** 2, -0.05, 1. / 6.]}
for name in ["omegab_h2", "omegac_h2", "both"]:
    for lim in ["lower", "upper"]:
        k = name + "_" + lim
        v = start[k]
        good = find_good(v[0], v[1], name, v[2], **kwargs)
        print "Good ", k, ": ", good
# print "good omegab: ", omegab
# print "=============================================="
# print "             omegab alone lower  "
# print "=============================================="
# # I know that omegab = 0.0011 gives a segfault so don't even try that (or
# # anything below that)
# omegab = 0.0012
# scale = 0.001
#
# t = Transfer(omegab=omegab, omegac=0.25, force_flat=True, h=1)
# rubbish = np.any(np.isnan(t._lnP_0)) or np.any(np.isinf(t._lnP_0))
# diff = 0.0
# first_non_rubbish = False
# last_was_rubbish = rubbish
# while rubbish or diff > 1e-4:
#     omegab += scale
#     t = Transfer(omegab=omegab, omegac=0.25, force_flat=True, h=1)
#     rubbish = np.any(np.isnan(t._lnP_0)) or np.any(np.isinf(t._lnP_0))
#     diff = abs(omegab - omegab / scale)
#     if not rubbish:
#         first_non_rubbish = True
#
#     if first_non_rubbish:
#         scale /= 1.5
#
#     if last_was_rubbish != rubbish:
#         scale = -scale
#     last_was_rubbish = rubbish
#
# print "good omegab: ", omegab
#
#
# print "=============================================="
# print "             omegab alone upper  "
# print "=============================================="
# omegab = 0.75
# scale = -0.05
#
# t = Transfer(omegab=omegab, omegac=0.25, force_flat=True, h=1)
# rubbish = np.any(np.isnan(t._lnP_0)) or np.any(np.isinf(t._lnP_0))
# diff = 0.0
# first_non_rubbish = False
# last_was_rubbish = rubbish
# while rubbish or diff > 1e-4:
#     omegab += scale
#     t = Transfer(omegab=omegab, omegac=0.25, force_flat=True, h=1)
#     rubbish = np.any(np.isnan(t._lnP_0)) or np.any(np.isinf(t._lnP_0))
#     diff = abs(omegab - omegab / scale)
#     if not rubbish:
#         first_non_rubbish = True
#
#     if first_non_rubbish:
#         scale /= 1.5
#
#     if last_was_rubbish != rubbish:
#         scale = -scale
#     last_was_rubbish = rubbish
#
# print "good omegab: ", omegab
#
#
# print "=============================================="
# print "             omegac alone lower  "
# print "=============================================="
# omegac = 0.0
# scale = 0.01
#
# t = Transfer(omegab=0.05, omegac=omegac, force_flat=True, h=1)
# rubbish = np.any(np.isnan(t._lnP_0)) or np.any(np.isinf(t._lnP_0))
# diff = 0.0
# first_non_rubbish = False
# last_was_rubbish = rubbish
# while rubbish or diff > 1e-4:
#     omegab *= scale
#     t = Transfer(omegab=omegab, omegac=0.25, force_flat=True, h=1)
#     rubbish = np.any(np.isnan(t._lnP_0)) or np.any(np.isinf(t._lnP_0))
#     diff = abs(omegab - omegab / scale)
#     if not rubbish:
#         first_non_rubbish = True
#
#     if first_non_rubbish:
#         scale -= (scale - 1.0) / 1.2
#
#     if last_was_rubbish != rubbish:
#         scale = 1.0 / scale
#     last_was_rubbish = rubbish
#
# print "good omegab: ", omegab
#
# for omegab in np.linspace(0.0012, 0.75, 10):
#     t = Transfer(omegab=omegab, omegac=0.25, force_flat=True, h=1)
#     anynan = np.any(np.isnan(t._lnP_0))
#     anyinf = np.any(np.isnan(t._lnP_0))
#     if anynan or anyinf:
#         print omegab, ":  NaNs: ", anynan, ". Infs: ", anyinf
#     plt.plot(t.lnk, t._lnP_0, label=str(omegab))
#
# plt.legend()
# plt.savefig("omegab_alone.pdf")
# plt.clf()
#
# print "=============================================="
# print "             omegac alone   "
# print "=============================================="
# for omegac in np.linspace(0.05, 0.95, 10):
#     t = Transfer(omegab=0.05, omegac=omegac, force_flat=True, h=1)
#     anynan = np.any(np.isnan(t._lnP_0))
#     anyinf = np.any(np.isnan(t._lnP_0))
#     if anynan or anyinf:
#         print omegac, ":  NaNs: ", anynan, ". Infs: ", anyinf
#     plt.plot(t.lnk, t._lnP_0, label=str(omegac))
#
# plt.legend()
# plt.savefig("omegac_alone.pdf")
# plt.clf()
#
#
# # --------- omegac + omegab ----------------------------------------------------
# print "=============================================="
# print "             omegac + omegab   "
# print "=============================================="
# for omegam in np.linspace(0.012, 1.0, 10):
#     omegab = omegam / 6.0
#     omegac = omegab * 5.0
#     t = Transfer(omegab=omegab, omegac=omegac, force_flat=True, h=1)
#     anynan = np.any(np.isnan(t._lnP_0))
#     anyinf = np.any(np.isnan(t._lnP_0))
#     if anynan or anyinf:
#         print omegam, ":  NaNs: ", anynan, ". Infs: ", anyinf
#     plt.plot(t.lnk, t._lnP_0, label=str(omegam))
#
# plt.legend()
# plt.savefig("omegam.pdf")
# plt.clf()
