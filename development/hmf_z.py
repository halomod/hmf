"""
Info script for Steven.
Results show issues at higher redshift.
"""

import hmf
from astropy.cosmology import LambdaCDM
from astropy.io import ascii
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import subprocess
import glob
import os


# --- Set parameter values. ---

logMmin = 12.5
logMmax = 16
z       = 0.01

hmf_model = "Tinker08"
delta_wrt = "crit"
delta_h   = 500.0

H0  = 67.11
Om0 = 0.3175
Ob0 = 0.049
Ode0 = 0.6825

sigma_8 = 0.8344
n       = 0.9624
transfer_model = "EH"

# -----------------------------


# --- Create cosmology and set its parameter values. ---

cosmo_model = LambdaCDM(H0=H0, Om0=Om0, Ob0=Ob0, Ode0=Ode0)
print ""
print "Cosmological model and parameter values:"
print cosmo_model
print "sigma_8 =", sigma_8, ", n =", n, ", transfer_model =", transfer_model

# -----------------------------------------------------


# --- Create mass function. ---

mf = hmf.MassFunction(Mmin=logMmin, Mmax=logMmax, z=z,
                      hmf_model=hmf_model, delta_wrt=delta_wrt, delta_h=delta_h,
                      cosmo_model=cosmo_model,
                      sigma_8=sigma_8, n=n, transfer_model=transfer_model)

print ""
print "Mass function and parameter values:"
print mf.parameter_values
print ""

# -----------------------------

# Different zs:
#zmin = np.log10(0.01)
#zmax = np.log10(2)

zmin = 0.01
zmax = 2
nz = 200
outdir = 'Plots/'
if not os.path.exists(outdir):
    print "The directory ", outdir, "does not exist. Creating it..."
    os.makedirs(outdir)

filnam = outdir + 'hmf_challenge'
exten_ima = '.jpg'
all_imas = filnam + "*" + exten_ima
all_imas_gl = glob.glob(all_imas)
#args = 'rm -f' + all_imas_gl
#outcmd = subprocess.Popen(args)
# Doesn't work for some reason (also don't want to set shell=True).

for filerm in all_imas_gl:
    os.remove(filerm)

i = 0
print "Calculating and plotting mass functions at", nz, "redshifts..."
#for z in np.logspace(zmin,zmax,nz):

for z in np.linspace(zmin,zmax,nz):
    mf.update(z=z)
    plt.plot(mf.m,mf.dndm,color="darkblue",alpha=1)
    outfil = filnam + str(i).zfill(3)
    plt.xlim(1e12, 1e16)
    plt.ylim(1e-50, 1e-15)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"Mass, $[h^{-1}M_\odot]$")
    plt.ylabel(r"$dn/dm$, $[h^{4}{\rm Mpc}^{-3}M_\odot^{-1}]$")
    plt.text(5e12, 1e-40, '$z = $'+str(format(z, '6.4f')), fontsize=15)
    plt.savefig(outfil + exten_ima)
    plt.close()
    i = i + 1

movie_nam = filnam + '.mp4'
ffmpeg_out = 'ffmpeg.out'
outcmd = subprocess.call(['rm', '-f', movie_nam])
argus = ['ffmpeg', '-f', 'image2', '-r', '10', '-pattern_type', 'glob', '-i', all_imas, movie_nam]
f_std = open(ffmpeg_out, 'w')
outcmd = subprocess.Popen(argus, stdout=f_std, stderr=subprocess.STDOUT)
f_std.close()

