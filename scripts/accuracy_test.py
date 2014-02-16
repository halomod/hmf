#!/usr/local/bin/python2.7
# encoding: utf-8
'''
scripts.accuracy_test -- shortdesc

scripts.accuracy_test is a description

It defines classes_and_methods
'''

import sys
import os
import traceback

from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter

import hmf
import numpy as np
import time
import itertools
import re
import pandas

__all__ = []
__version__ = 0.1
__date__ = 2014 - 02 - 13
__updated__ = 2014 - 02 - 13

DEBUG = 0
TESTRUN = 0
PROFILE = 0

class CLIError(Exception):
    '''Generic exception to raise and log different fatal errors.'''
    def __init__(self, msg):
        super(CLIError).__init__(type(self))
        self.msg = "E: %s" % msg
    def __str__(self):
        return self.msg
    def __unicode__(self):
        return self.msg

def main(argv=None):
    '''Generate halo mass functions and write them to file (BETA).'''

    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    program_name = os.path.basename(sys.argv[0])
    program_version = "v%s" % __version__
    program_build_date = str(__updated__)
    program_version_message = '%%(prog)s %s (%s)\n\nHISTORY\n------%s' % (program_version, program_build_date, HISTORY)
    program_shortdesc = __import__('__main__').__doc__.split("\n")[1]
    program_license = '''%s

    Copyright (c) 2014 Steven Murray

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.

USAGE
''' % (program_shortdesc)

    try:
        h = hmf.MassFunction()
        m_attrs = ["dndlog10m", "lnsigma", "n_eff", "sigma",
                   "dndm", "ngtm", "fsigma", "mgtm", "nltm", "dndlnm",
                   "how_big", "mltm", "_sigma_0", "_dlnsdlnm"]
        k_attrs = ["power", "delta_k", "transfer", "nonlinear_power",
                   "_lnP_0", "_lnP_cdm_0", "_lnT_cdm", "_unnormalised_lnP",
                   "_unnormalised_lnT"]
        # Setup argument parser
        parser = ArgumentParser(description=program_license, formatter_class=RawDescriptionHelpFormatter)
        parser.add_argument("-v", "--verbose", dest="verbose", action="count", help="set verbosity level [default: %(default)s]")
        parser.add_argument('-V', '--version', action='version', version=program_version_message)

        # HMF specific arguments
        config = parser.add_argument_group("Config", "Variables of Configuration")
        config.add_argument("--max-diff-tol", type=float, default=100.0, help="Minimum maximum diff to write out")
        config.add_argument("--rms-diff-tol", type=float, default=100.0, help="Minimum rms diff to write out")


        # config.add_argument("filename", help="filename to write to")
        config.add_argument("--quantities", nargs="*", default=["dndm"],
                            choices=m_attrs + k_attrs)
        accuracy_args = parser.add_argument_group("Accuracy Args", "Arguments affecting the accuracy of output")

        # accuracy_args.add_argument("--M-min", nargs="*", type=float,
        #                    help="minimum log10 M [default: %s]" %
        #                    np.log10(h.M[0]))
        # accuracy_args.add_argument("--M-max", nargs="*", type=float,
        #                    help="maximum log10 M [default: %s]" %
        #                    np.log10(h.M[-1]))
        # accuracy_args.add_argument("--M-num", nargs="*", type=int,
        #                    help="length of M vector [default: %s]" %
        #                    len(h.M))
        accuracy_args.add_argument("--lnk-min", nargs="*", type=float,
                                   help="the maximum wavenumber [default: %s]" %
                                   h.transfer.lnk[0])
        accuracy_args.add_argument("--lnk-max", nargs="*", type=float,
                                   help="the minimum wavenumber [default: %s]" %
                                   h.transfer.lnk[0])
        accuracy_args.add_argument("--lnk-num", nargs="*", type=int,
                                   help="the number of wavenumber [default: %s]" %
                                   len(h.transfer.lnk))
        accuracy_args.add_argument("--lAccuracyBoost", nargs="*", type=float,
                            help="[CAMB] optional accuracy boost [default: %s]" % h.transfer._camb_options["lAccuracyBoost"])
        accuracy_args.add_argument("--AccuracyBoost", nargs="*", type=float,
                            help="[CAMB] optional accuracy boost [default: %s]" % h.transfer._camb_options["AccuracyBoost"])

        accuracy_args.add_argument("--transfer--k-per-logint", nargs="*", type=float,
                            help="[CAMB] number of estimated wavenumbers per interval [default: %s]" % h.transfer._camb_options["transfer__k_per_logint"])
        accuracy_args.add_argument("--transfer--kmax", nargs='*', type=float,
                            help="[CAMB] maximum wavenumber to estimate [default: %s]" % h.transfer._camb_options["transfer__kmax"])
        accuracy_args.add_argument("--mv-scheme", nargs="*", choices=['trapz', 'simps', 'romb'])

        hmfargs = parser.add_argument_group("HMF", "HMF-specific arguments")
        hmfargs.add_argument("--mf-fit", choices=hmf.Fits.mf_fits,
                            help="fitting function to use. 'all' uses all of them [default: %s]" % h.mf_fit)
        hmfargs.add_argument("--delta-h", type=float,
                            help="overdensity of halo w.r.t delta_wrt [default %s]" % h.delta_wrt)
        hmfargs.add_argument("--delta-wrt", choices=["mean", "crit"],
                            help="what delta_h is with respect to [default: %s]" % h.delta_h)
        hmfargs.add_argument("--user-fit", help="a custom fitting function defined as a string in terms of x for sigma [default: %s]" % "'" + h.user_fit + "'")
        hmfargs.add_argument("--no-cut-fit", action="store_true", help="whether to cut the fitting function at tested boundaries")
        hmfargs.add_argument("--z2", type=float, help="upper redshift for volume weighting")
        hmfargs.add_argument("--nz", type=float, help="number of redshift bins for volume weighting")
        hmfargs.add_argument("--delta-c", type=float, help="critical overdensity for collapse [default: %s]" % h.delta_c)

        # # Transfer-specific arguments
        transferargs = parser.add_argument_group("Transfer", "Transfer-specific arguments")
        transferargs.add_argument("--z", type=float, help="redshift of analysis [default: %s]" % h.transfer.z)
        transferargs.add_argument("--wdm-mass", type=float, help="warm dark matter mass (0 is CDM)")
        transferargs.add_argument("--transfer-fit", choices=hmf.transfer.Transfer.fits,
                                  help="which fit for the transfer function to use ('all' uses all of them) [default: %s]" % h.transfer.transfer_fit)

        cambargs = parser.add_argument_group("CAMB", "CAMB-specific arguments")
        cambargs.add_argument("--Scalar-initial-condition", type=int, choices=[1, 2, 3, 4, 5],
                              help="[CAMB] initial scalar perturbation mode [default: %s]" % h.transfer._camb_options["Scalar_initial_condition"])
        cambargs.add_argument("--ThreadNum", type=int,
                              help="number of threads to use (0 is automatic detection) [default: %s]" % h.transfer._camb_options["ThreadNum"])
        cambargs.add_argument("--w-perturb", action="store_true", help="[CAMB] whether w should be perturbed or not")

        # # Cosmo-specific arguments
        cosmoargs = parser.add_argument_group("Cosmology", "Cosmology arguments")
        cosmoargs.add_argument("--default", choices=['planck1_base'],
                               help="base cosmology to use [default: %s]" % h.transfer.cosmo.default)
        cosmoargs.add_argument("--force-flat", action="store_true",
                               help="force cosmology to be flat (changes omega_lambda) [default: %s]" % h.transfer.cosmo.force_flat)
        cosmoargs.add_argument("--sigma-8", type=float, help="mass variance in top-hat spheres with r=8")
        cosmoargs.add_argument("--n", type=float, help="spectral index")
        cosmoargs.add_argument("--w", type=float, help="dark energy equation of state")
        cosmoargs.add_argument("--cs2-lam", type=float, help="constant comoving sound speed of dark energy")

        h_group = cosmoargs.add_mutually_exclusive_group()
        h_group.add_argument("--h", type=float, help="The hubble parameter")
        h_group.add_argument("--H0", type=float, help="The hubble constant")

        omegab_group = cosmoargs.add_mutually_exclusive_group()
        omegab_group.add_argument("--omegab", type=float, help="baryon density")
        omegab_group.add_argument("--omegab-h2", type=float, help="baryon density by h^2")

        omegac_group = cosmoargs.add_mutually_exclusive_group()
        omegac_group.add_argument("--omegac", type=float, help="cdm density")
        omegac_group.add_argument("--omegac-h2", type=float, help="cdm density by h^2")
        omegac_group.add_argument("--omegam", type=float, help="total matter density")

        cosmoargs.add_argument("--omegav", type=float, help="the dark energy density")

        # Process arguments
        args = parser.parse_args()

        # # Process the arguments
        kwargs = {}
        for arg in ["omegab", "omegab_h2", "omegac", "omegac_h2", "omegam", "h", "H0",
                    "sigma_8", "n", "w", "cs2_lam", "omegav", "ThreadNum", "transfer__kmax",
                    "transfer__k_per_logint", "AccuracyBoost", "lAccuracyBoost",
                    "Scalar_initial_condition", "z", "z2", "nz", "delta_c", "user_fit", "delta_h",
                    "delta_wrt", 'mf_fit', 'transfer_fit', 'mv_scheme']:
            if getattr(args, arg) is not None:
                kwargs[arg] = make_scalar(getattr(args, arg))

        if args.w_perturb:
            kwargs['w_perturb'] = True

        if args.user_fit is not None:
            kwargs['mf_fit'] = 'user_model'

        if args.no_cut_fit:
            kwargs['cut_fit'] = not args.no_cut_fit

#         if args.M_min is not None and args.M_max is not None and args.M_num is not None:
#             kwargs["M"] = []
#             for mmin in args.M_min:
#                 for mmax in args.M_max:
#                     for mnum in args.M_num:
#                         kwargs["M"].append(np.linspace(mmin, mmax, mnum))

        if args.lnk_min is not None and args.lnk_max is not None and args.lnk_num is not None:
            kwargs["lnk"] = []
            for kmin in args.M_min:
                for kmax in args.M_max:
                    for knum in args.M_num:
                        kwargs["lnk"].append(np.linspace(kmin, kmax, knum))

        m_att = [a for a in args.quantities if a in m_attrs]
        k_att = [a for a in args.quantities if a in k_attrs]

        # Create a high-res object
        start = time.time()
        testbed = hmf.MassFunction(# M=np.linspace(3, 18, 5000),
                                   transfer__k_per_logint=20,
                                   transfer__kmax=100,
                                   lnk=np.linspace(-20, 20, 4097),
                                   lAccuracyBoost=2,
                                   AccuracyBoost=2,
                                   mv_scheme='romb')

        # Initialise all required quantities
        for quantity in m_att:
            getattr(testbed, quantity)
        for quantity in k_att:
            getattr(testbed.transfer, quantity)

        hires_time = time.time() - start

        # Set up results dictionaries
        results = {}
        for quantity in args.quantities:
            results[quantity] = {}

        times = {}


        # Save all the variables into dictionary/list of dicts for use
        listkwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, (list, tuple)):
                listkwargs[k] = kwargs.pop(k)

        # Now we have separated kwargs into listy things and singular items
        vallist = [v for k, v in listkwargs.iteritems()]
        final_list = [dict(zip(listkwargs.keys(), v)) for v in itertools.product(*vallist)]

        for vals in final_list:
            # We make a fresh one each time to get the timing right.
            kwargs.update(vals)
            h = hmf.MassFunction(**kwargs)

            if len(final_list) > 1:
                label = str(vals)
            elif kwargs:
                label = str(kwargs)
            else:
                label = h.mf_fit


            label = label.replace("{", "").replace("}", "").replace("'", "")
            label = label.replace("_", "").replace(": ", "").replace(", ", "_")
            label = label.replace("mffit", "").replace("transferfit", "").replace("delta_wrt", "").replace("\n", "")

            # The following lines transform the M and lnk parts
            while "[" in label:
                label = re.sub("[\[].*?[\]]", "", label)
            label = label.replace("array", "")
            label = label.replace("mvscheme", "")
            label = label.replace("M()", "M(" + str(np.log10(h.M[0])) + ", " + str(np.log10(h.M[-1])) + ", " +
                          str(np.log10(h.M[1]) - np.log10(h.M[0])) + ")")
            label = label.replace("lnk()", "lnk(" + str(h.transfer.lnk[0]) + ", " + str(h.transfer.lnk[-1]) + ", " +
                          str(h.transfer.lnk[1] - h.transfer.lnk[0]) + ")")

            start = time.time()
            for att in m_att:
                results[quantity][label] = getattr(h, att)
            for att in k_att:
                results[quantity][label] = getattr(h.transfer, att)

            times[label] = time.time() - start

        labels = [label for label in times]
        print labels
        # Now write out/plot results
        for quantity in args.quantities:
            if quantity in m_att:
                test = getattr(testbed, quantity)
            elif quantity in k_att:
                test = getattr(testbed.transfer, quantity)

            df = pandas.DataFrame(np.zeros((len(labels), 3)),
                                  columns=['MaxDiff', "RMSDiff", "Time"],
                                  index=labels)
            df["Time"] = [t for l, t in times.iteritems()]
            print "============ " + quantity + " ================"
            for label, res in results[quantity].iteritems():
                # Maximum difference
                df["MaxDiff"][label] = np.max(np.abs(res / test - 1))
                # rms difference
                df["RMSDiff"][label] = np.sqrt(np.mean((res / test - 1) ** 2))

            df.sort("Time", inplace=True)
            print df[np.logical_and(df["MaxDiff"] < args.max_diff_tol,
                                    df["RMSDiff"] < args.rms_diff_tol)]
        return 0
    except KeyboardInterrupt:
        ### handle keyboard interrupt ###
        return 0
    except Exception, e:
        if DEBUG or TESTRUN:
            raise(e)
        traceback.print_exc()
        indent = len(program_name) * " "
        sys.stderr.write(program_name + ": " + repr(e) + "\n")
        sys.stderr.write(indent + "  for help use --help\n")
        return 2

def make_scalar(a):
    if isinstance(a, (list, tuple)):
        if len(a) == 1:
            a = a[0]
    return a

HISTORY = """
0.0.1 - Not even working yet
"""
if __name__ == "__main__":
    if DEBUG:
        sys.argv.append("-h")
        sys.argv.append("-v")
    if TESTRUN:
        import doctest
        doctest.testmod()
    if PROFILE:
        import cProfile
        import pstats
        profile_filename = 'scripts.accuracy_test_profile.txt'
        cProfile.run('main()', profile_filename)
        statsfile = open("profile_stats.txt", "wb")
        p = pstats.Stats(profile_filename, stream=statsfile)
        stats = p.strip_dirs().sort_stats('cumulative')
        stats.print_stats()
        statsfile.close()
        sys.exit(0)
    sys.exit(main())



