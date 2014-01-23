"""
This module contains a single class, `Transfer`, which provides methods to 
calculate the transfer function, matter power spectrum and several other 
related quantities. 
"""
import numpy as np
from cosmo import Cosmology
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import cosmolopy as cp
import scipy.integrate as integ

# import cosmolopy.density as cden
import tools
try:
    import pycamb
    HAVE_PYCAMB = True
except ImportError:
    HAVE_PYCAMB = False

class Transfer(object):
    '''
    Neatly deals with different transfer functions and their routines.
    
    The purpose of this class is to calculate transfer functions, power spectra
    and several tightly associated quantities using many of the available fits
    from the literature. 
        
    Importantly, it contains the means to calculate the transfer function using the
    popular CAMB code, the Eisenstein-Hu fit (1998), the BBKS fit or the Bond and
    Efstathiou fit (1984). Furthermore, it can calculate non-linear corrections
    using the halofit model (with updated parameters from Takahashi2012).
    
    The primary feature of this class is to wrap all the methods into a unified
    interface. On top of this, the class implements optimized updates of 
    parameters which is useful in, for example, MCMC code which covers a
    large parameter-space. Calling the `nonlinear_power` does not re-evaluate
    the entire transfer function, rather it just calculates the corrections, 
    improving performance.
    
    To update parameters optimally, use the update() method. 
    All output quantities are calculated only when needed (but stored after 
    first calculation for quick access).
    
    
    Parameters
    ----------
    lnk : array_like, default ``linspace(log(1e-8),log(2e4),250)``
        Defines logarithmic wavenumbers, *k* [units :math:`h Mpc^{-1}`]. 
        This array is integrated over for normalisation. 
        
    z : float, optional, default ``0.0``
        The redshift of the analysis.
                   
    wdm_mass : float, optional, default ``None``
        The warm dark matter particle size in *keV*, or ``None`` for CDM.
                                                                          
    transfer_fit : str, { ``"CAMB"``, ``"EH"``, ``"bbks"``, ``"bond_efs"``} 
        Defines which transfer function fit to use. If not defined from the
        listed options, it will be treated as a filename to be read in. In this
        case the file must contain a transfer function in CAMB output format. 
           
    initial_mode : int, {1,2,3,4,5}
        (CAMB-only) Initial scalar perturbation mode (adiabatic=1, CDM iso=2, 
        Baryon iso=3,neutrino density iso =4, neutrino velocity iso = 5) 
        
    lAccuracyBoost : float, optional, default ``1.0``
        (CAMB-only) Larger to keep more terms in the hierarchy evolution
    
    AccuracyBoost : float, optional, default ``1.0``
        (CAMB-only) Increase accuracy_boost to decrease time steps, use more k 
        values,  etc.Decrease to speed up at cost of worse accuracy. 
        Suggest 0.8 to 3.
        
    w_perturb : bool, optional, default ``False``
        (CAMB-only) 
    
    k_per_logint : int, optional, default ``11``
        (CAMB-only) Number of wavenumbers estimated per log interval by CAMB
        Default of 11 gets best performance for requisite accuracy of mass function.
        
    kmax : float, optional, default ``0.25``
        (CAMB-only) Maximum value of the wavenumber.
        Default of 0.25 is high enough for requisite accuracy of mass function.
        
    ThreadNum : int, optional, default ``0``
        (CAMB-only) Number of threads to use for calculation of transfer 
        function by CAMB. Default 0 automatically determines the number.
                       
    kwargs : keywords
        The ``**kwargs`` take any cosmological parameters desired, which are 
        input to the `hmf.cosmo.Cosmology` class. `hmf.Perturbations` uses a 
        default parameter set from the first-year PLANCK mission, with optional 
        modifications by the user. Here is a list of parameters currently 
        available (and their defaults in `Transfer`):       
                 
        :sigma_8: [0.8344] The normalisation. Mass variance in top-hat spheres 
            with :math:`R=8Mpc h^{-1}`   
        :n: [0.9624] The spectral index 
        :w: [-1] The dark-energy equation of state
        :cs2_lam: [1] The constant comoving sound speed of dark energy
        :t_cmb: [2.725] Temperature of the CMB
        :y_he: [0.24] Helium fraction
        :N_nu: [3.04] Number of massless neutrino species
        :N_nu_massive: [0] Number of massive neutrino species
        :delta_c: [1.686] The critical overdensity for collapse
        :H0: [67.11] The hubble constant
        :h: [``H0/100.0``] The hubble parameter
        :omegan: [0] The normalised density of neutrinos
        :omegab_h2: [0.022068] The normalised baryon density by ``h**2``
        :omegac_h2: [0.12029] The normalised CDM density by ``h**2``
        :omegav: [0.6825] The normalised density of dark energy
        :omegab: [``omegab_h2/h**2``] The normalised baryon density
        :omegac: [``omegac_h2/h**2``] The normalised CDM density     
        :force_flat: [False] Whether to force the cosmology to be flat (affects only ``omegav``)
        :default: [``"planck1_base"``] A default set of cosmological parameters
    '''

    fits = ["CAMB", "EH", "bbks", "bond_efs"]
    _cp = ["sigma_8", "n", "w", "cs2_lam", "t_cmb", "y_he", "N_nu",
           "omegan", "H0", "h", "omegab",
           "omegac", "omegav", "omegab_h2", "omegac_h2",
           "force_flat", "default"]

    def __init__(self, z=0.0, lnk=None,
                 wdm_mass=None, transfer_fit='CAMB',
                 initial_mode=1, lAccuracyBoost=1,
                 AccuracyBoost=1, w_perturb=False, k_per_logint=11,
                 kmax=0.25, ThreadNum=0, **kwargs):
        '''
        Initialises some parameters
        '''

        if lnk is None:
            lnk = np.linspace(np.log(1e-8), np.log(2e4), 250)
        # Set up a simple dictionary of cosmo params which can be later updated
        if "default" not in kwargs:
            kwargs["default"] = "planck1_base"

        self._cpdict = {k:v for k, v in kwargs.iteritems() if k in Transfer._cp}
        self._camb_options = {'Scalar_initial_condition' : initial_mode,
                              'scalar_amp'      : 1E-9,
                              'lAccuracyBoost' : lAccuracyBoost,
                              'AccuracyBoost'  : AccuracyBoost,
                              'w_perturb'      : w_perturb,
                              'transfer__k_per_logint': k_per_logint,
                              'transfer__kmax':kmax,
                              'ThreadNum':ThreadNum}



        # Set all given parameters
        self.lnk = lnk
        self.wdm_mass = wdm_mass
        self.z = z
        self.transfer_fit = transfer_fit
        self.cosmo = Cosmology(**self._cpdict)

        # Here we store the values (with defaults) into _cpdict so they can be updated later.
        if "omegab" in kwargs:
            actual_cosmo = {k:v for k, v in self.cosmo.__dict__.iteritems()
                            if k in Transfer._cp and k not in ["omegab_h2", "omegac_h2"]}
        else:
            actual_cosmo = {k:v for k, v in self.cosmo.__dict__.iteritems()
                            if k in Transfer._cp and k not in ["omegab", "omegac"]}
        if "h" in kwargs:
            del actual_cosmo["H0"]
        elif "H0" in kwargs:
            del actual_cosmo["h"]
        self._cpdict.update(actual_cosmo)

    def update(self, **kwargs):
        """
        Update the class optimally with given arguments.
        
        Accepts any argument that the constructor takes
        """
        # First update the cosmology
        cp = {k:v for k, v in kwargs.iteritems() if k in self._cp}
        if cp:
            true_cp = {}
            for k, v in cp.iteritems():
                if k not in self._cpdict:
                    true_cp[k] = v
                elif k in self._cpdict:
                    if v != self._cpdict[k]:
                        true_cp[k] = v

            self._cpdict.update(true_cp)
            # Delete the entries we've used from kwargs
            for k in cp:
                del kwargs[k]

            # Now actually update the Cosmology class
            self.cosmo = Cosmology(**self._cpdict)

            # The following two parameters don't necessitate a complete recalculation
            if "n" in true_cp:
                try: del self._unnormalised_lnP
                except AttributeError: pass
            if "sigma_8" in true_cp:
                try: del self._lnP_cdm_0
                except AttributeError: pass
                try: del self._lnT_cdm
                except AttributeError: pass

            # All other parameters mean recalculating everything :(
            for item in ["omegab", "omegac", "h", "H0", "omegab_h2", "omegac_h2"]:
                if item in true_cp:
                    del self._unnormalised_lnT

        # Now do the other parameters
        for key, val in kwargs.iteritems():  # only camb options should be left
            # CAMB OPTIONS
            if key in self._camb_options:
                if self._camb_options[key] != val:
                    self._camb_options.update({key:val})
                    if key != "ThreadNum":
                        del self._unnormalised_lnT
            # ANYTHING ELSE
            else:
                if "_Transfer__" + key not in self.__dict__:
                    print "WARNING: ", key, " is not a valid parameter for the Transfer class"
                else:
                    if np.any(getattr(self, key) != val):
                        setattr(self, key, val)  # doing it this way enables value-checking

        # Some extra logic for deletes
        if ('omegab' in cp or 'omegac' in cp or 'omegav' in cp) and self.z > 0:
            del self.growth
        elif 'z' in kwargs:
            if kwargs['z'] == 0:
                del self.growth

    # ---- SET PROPERTIES --------------------------------
    @property
    def lnk(self):
        return self.__lnk

    @lnk.setter
    def lnk(self, val):
        try:
            if len(val) < 100:
                raise ValueError("lnk should have more than 100 steps!")
        except TypeError:
            raise TypeError("lnk must be a sequence")

        if np.any(np.abs(np.diff(val, 2)) > 1e-5) or val[1] < val[0]:
            raise ValueError("lnk must be a linearly increasing array!")

        del self._unnormalised_lnT
        self.__lnk = val

    @property
    def z(self):
        return self.__z

    @z.setter
    def z(self, val):
        try:
            val = float(val)
        except ValueError:
            raise ValueError("z must be a number (", val, ")")

        if val < 0:
            raise ValueError("z must be > 0 (", val, ")")

        # Delete stuff dependent on it
        del self.growth
        self.__z = val

    @property
    def wdm_mass(self):
        return self.__wdm_mass

    @wdm_mass.setter
    def wdm_mass(self, val):
        if val is None:
            self.__wdm_mass = val
            return
        try:
            val = float(val)
        except ValueError:
            raise ValueError("wdm_mass must be a number (", val, ")")

        if val <= 0:
            raise ValueError("wdm_mass must be > 0 (", val, ")")

        # delete stuff dependent on it
        del self._lnP_0
        del self.transfer

        self.__wdm_mass = val

    @property
    def transfer_fit(self):
        return self.__transfer_fit

    @transfer_fit.setter
    def transfer_fit(self, val):
        if not HAVE_PYCAMB and val == "CAMB":
            raise ValueError("You cannot use the CAMB transfer since pycamb isn't installed")

        del self._unnormalised_lnT
        self.__transfer_fit = val


    # ---- DERIVED PROPERTIES AND FUNCTIONS ---------------
    def _check_low_k(self, lnk, lnT):
        """
        Check convergence of transfer function at low k.
        
        Unfortunately, some versions of CAMB produce a transfer which has a
        turn-up at low k, which is what we seek to cut out here.
        
        Parameters
        ----------
        lnk : array_like
            Value of log(k)
            
        lnT : array_like
            Value of log(transfer)
        """

        start = 0
        for i in range(len(lnk) - 1):
            if abs((lnT[i + 1] - lnT[i]) / (lnk[i + 1] - lnk[i])) < 0.01:
                start = i
                break
        lnT = lnT[start:-1]
        lnk = lnk[start:-1]

        return lnk, lnT

    # ---- TRANSFER FITS -------------------------------------------------------
    def _from_file(self, k):
        """
        Import the transfer function from file (must be CAMB format)
        
        .. note :: This should not be called by the user!
        """
        T = np.log(np.genfromtxt(self.transfer_fit)[:, [0, 6]].T)
        lnk, lnT = self._check_low_k(T[0, :], T[1, :])
        return spline(lnk, lnT, k=1)(k)

    def _CAMB(self, k):
        """
        Generate transfer function with CAMB
        
        .. note :: This should not be called by the user!
        """
        cdict = dict(self.cosmo.pycamb_dict(),
                     **self._camb_options)
        T = pycamb.transfers(**cdict)[1]
        T = np.log(T[[0, 6], :, 0])

        lnk, lnT = self._check_low_k(T[0, :], T[1, :])

        return spline(lnk, lnT, k=1)(k)

    def _EH(self, k):
        """
        Eisenstein-Hu transfer function
        
        .. note :: This should not be called by the user!
        """

        T = np.log(cp.perturbation.transfer_function_EH(np.exp(k) * self.cosmo.h,
                                    **self.cosmo.cosmolopy_dict())[1])
        return T

    def _bbks(self, k):
        """
        BBKS transfer function.

        .. note :: This should not be called by the user!
        """
        Gamma = self.cosmo.omegam * self.cosmo.h
        q = np.exp(k) / Gamma * np.exp(self.cosmo.omegab + np.sqrt(2 * self.cosmo.h) *
                               self.cosmo.omegab / self.cosmo.omegam)
        return np.log((np.log(1.0 + 2.34 * q) / (2.34 * q) *
                (1 + 3.89 * q + (16.1 * q) ** 2 + (5.47 * q) ** 3 +
                 (6.71 * q) ** 4) ** (-0.25)))

    def _bond_efs(self, k):
        """
        Bond and Efstathiou transfer function.
        
        .. note :: This should not be called by the user!
        """

        omegah2 = 1.0 / (self.cosmo.omegam * self.cosmo.h ** 2)

        a = 6.4 * omegah2
        b = 3.0 * omegah2
        c = 1.7 * omegah2
        nu = 1.13
        k = np.exp(k)
        return np.log((1 + (a * k + (b * k) ** 1.5 + (c * k) ** 2) ** nu) ** (-1 / nu))

    @property
    def _unnormalised_lnT(self):
        """
        The un-normalised transfer function
        
        This wraps the individual transfer_fit methods to provide unified access.
        """
        try:
            return self.__unnormalised_lnT
        except AttributeError:
            try:
                self.__unnormalised_lnT = getattr(self, "_" + self.transfer_fit)(self.lnk)
            except AttributeError:
                self.__unnormalised_lnT = self._from_file(self.lnk)
            return self.__unnormalised_lnT

    @_unnormalised_lnT.deleter
    def _unnormalised_lnT(self):
        try:
            del self.__unnormalised_lnT
            del self._unnormalised_lnP
            del self._lnT_cdm
        except AttributeError:
            pass

    @property
    def _unnormalised_lnP(self):
        """
        Un-normalised CDM log power at :math:`z=0` [units :math:`Mpc^3/h^3`]
        """
        try:
            return self.__unnormalised_lnP
        except AttributeError:
            self.__unnormalised_lnP = self.cosmo.n * self.lnk + 2 * self._unnormalised_lnT
            return self.__unnormalised_lnP

    @_unnormalised_lnP.deleter
    def _unnormalised_lnP(self):
        try:
            del self.__unnormalised_lnP
            del self._lnP_cdm_0
        except AttributeError:
            pass

    @property
    def _lnP_cdm_0(self):
        """
        Normalised CDM log power at z=0 [units :math:`Mpc^3/h^3`]
        """
        try:
            return self.__lnP_cdm_0
        except AttributeError:
            self.__lnP_cdm_0 = tools.normalize(self.cosmo.sigma_8,
                                               self._unnormalised_lnP,
                                               self.lnk, self.cosmo.mean_dens)[0]
            return self.__lnP_cdm_0

    @_lnP_cdm_0.deleter
    def _lnP_cdm_0(self):
        try:
            del self.__lnP_cdm_0
            del self._lnP_0
        except AttributeError:
            pass

    @property
    def _lnT_cdm(self):
        """
        Normalised CDM log transfer function
        """
        try:
            return self.__lnT_cdm
        except AttributeError:
            self.__lnT_cdm = tools.normalize(self.cosmo.sigma_8,
                                             self._unnormalised_lnT,
                                             self.lnk, self.cosmo.mean_dens)
            return self.__lnT_cdm

    @_lnT_cdm.deleter
    def _lnT_cdm(self):
        try:
            del self.__lnT_cdm
            del self.transfer
        except AttributeError:
            pass

    @property
    def _lnP_0(self):
        """
        Normalised log power at :math:`z=0` (for CDM/WDM)
        """
        try:
            return self.__lnP_0
        except AttributeError:
            if self.wdm_mass is not None:
                self.__lnP_0 = tools.wdm_transfer(self.wdm_mass, self._lnP_cdm_0,
                                                  self.lnk, self.cosmo.h, self.cosmo.omegac)
            else:
                self.__lnP_0 = self._lnP_cdm_0
            return self.__lnP_0

    @_lnP_0.deleter
    def _lnP_0(self):
        try:
            del self.__lnP_0
            del self.power
        except AttributeError:
            pass

    @property
    def growth(self):
        r"""
        The growth factor :math:`d(z)`
        
        This is calculated (see Lukic 2007) as
        
        .. math:: d(z) = \frac{D^+(z)}{D^+(z=0)}
                
        where
        
        .. math:: D^+(z) = \frac{5\Omega_m}{2}\frac{H(z)}{H_0}\int_z^{\infty}{\frac{(1+z')dz'}{[H(z')/H_0]^3}}
        
        and
        
        .. math:: H(z) = H_0\sqrt{\Omega_m (1+z)^3 + (1-\Omega_m)}
        
        """
        try:
            return self.__growth
        except:
            if self.z > 0:
                self.__growth = tools.growth_factor(self.z, self.cosmo)
            else:
                self.__growth = 1.0
            return self.__growth

    @growth.deleter
    def growth(self):
        try:
            del self.__growth
            del self.power
        except:
            pass

    @property
    def power(self):
        """
        Normalised log power spectrum [units :math:`Mpc^3/h^3`]
        """
        try:
            return self.__power
        except AttributeError:
            self.__power = 2 * np.log(self.growth) + self._lnP_0
            return self.__power

    @power.deleter
    def power(self):
        try:
            del self.__power
            del self.delta_k
        except AttributeError:
            pass

    @property
    def transfer(self):
        """
        Normalised log transfer function for CDM/WDM
        """
        try:
            return self.__transfer
        except AttributeError:
            self.__transfer = tools.wdm_transfer(self.wdm_mass, self._lnT_cdm,
                                                 self.lnk, self.cosmo.h, self.cosmo.omegac)
            return self.__transfer

    @transfer.deleter
    def transfer(self):
        try:
            del self.__transfer
        except AttributeError:
            pass

    @property
    def delta_k(self):
        r"""
        Dimensionless power spectrum, :math:`\Delta_k = \frac{k^3 P(k)}{2\pi^2}`
        """
        try:
            return self.__delta_k
        except AttributeError:
            self.__delta_k = 3 * self.lnk + self.power - np.log(2 * np.pi ** 2)
            return self.__delta_k

    @delta_k.deleter
    def delta_k(self):
        try:
            del self.__delta_k
            del self.nonlinear_power
        except AttributeError:
            pass

    @property
    def nonlinear_power(self):
        """
        Non-linear log power [units :math:`Mpc^3/h^3`]
        
        Non-linear corrections come from HALOFIT (Smith2003) with updated
        parameters from Takahashi2012. 
        
        This code was heavily influenced by the HaloFit class from the 
        `chomp` python package by Christopher Morrison, Ryan Scranton 
        and Michael Schneider (https://code.google.com/p/chomp/). It has 
        been modified to improve its integration with this package.        
        """
        try:
            return self.__nonlinear_power
        except:
            k = np.exp(self.lnk)
            delta_k = np.exp(self.delta_k)

            # Define the cosmology at redshift
            omegam = cp.density.omega_M_z(self.z, **self.cosmo.cosmolopy_dict())
            omegav = self.cosmo.omegav / cp.distance.e_z(self.z, **self.cosmo.cosmolopy_dict()) ** 2
            w = self.cosmo.w

            f1 = omegam ** -0.0307
            f2 = omegam ** -0.0585
            f3 = omegam ** 0.0743

            # Initialize sigma spline
            lnr = np.linspace(np.log(0.1), np.log(10.0), 1000)
            lnsig = np.empty(1000)

            for i, r in enumerate(lnr):
                R = np.exp(r)
                integrand = delta_k * np.exp(-(k * R) ** 2)
                sigma2 = integ.simps(integrand, np.log(k))
                lnsig[i] = np.log(sigma2)

            r_of_sig = spline(lnsig[::-1], lnr[::-1], k=5)
            ks = 1.0 / np.exp(r_of_sig(0.0))
            sig_of_r = spline(lnr, lnsig, k=5)

            (dev1, dev2) = sig_of_r.derivatives(np.log(1.0 / ks))[1:3]

            neff = -dev1 - 3.0
            C = -dev2

            a_n = 10 ** (1.5222 + 2.8553 * neff + 2.3706 * neff ** 2 +
                    0.9903 * neff ** 3 + 0.2250 * neff ** 4 +
                    - 0.6038 * C + 0.1749 * omegav * (1 + w))
            b_n = 10 ** (-0.5642 + 0.5864 * neff + 0.5716 * neff ** 2 +
                    - 1.5474 * C + 0.2279 * omegav * (1 + w))
            c_n = 10 ** (0.3698 + 2.0404 * neff + 0.8161 * neff ** 2 + 0.5869 * C)
            gamma_n = 0.1971 - 0.0843 * neff + 0.8460 * C
            alpha_n = np.fabs(6.0835 + 1.3373 * neff - 0.1959 * neff ** 2 +
                    - 5.5274 * C)
            beta_n = (2.0379 - 0.7354 * neff + 0.3157 * neff ** 2 +
                      1.2490 * neff ** 3 + 0.3980 * neff ** 4 - 0.1682 * C)
            mu_n = 0.0
            nu_n = 10 ** (5.2105 + 3.6902 * neff)

            y = k / ks
            fk = y / 4.0 + y ** 2 / 8.0

            delta2Q = delta_k * ((1 + delta_k) ** beta_n /
                                 (1 + alpha_n * delta_k) * np.exp(-fk))

            delta2Hprime = a_n * y ** (3 * f1) / (1 + b_n * y ** f2 +
                                                  (c_n * f3 * y) ** (3 - gamma_n))

            delta2H = delta2Hprime / (1 + mu_n / y + nu_n / y ** 2)

            self.__nonlinear_power = np.log(2.0 * np.pi ** 2 / k ** 3 *
                                            (delta2Q + delta2H))
            return self.__nonlinear_power

    @nonlinear_power.deleter
    def nonlinear_power(self):
        try:
            del self.__nonlinear_power
        except AttributeError:
            pass
