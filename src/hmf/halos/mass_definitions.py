"""A model for mass definitions.

This is primarily inspired by Benedikt Diemer's COLOSSUS code:
https://bdiemer.bitbucket.io/colossus/halo_mass_defs.html
"""
import astropy.units as u
import numpy as np
import scipy as sp
import warnings
from astropy.cosmology import Planck15
from typing import Optional

from .._internals import _framework
from ..cosmology import Cosmology

__all__ = [
    "FOF",
    "MassDefinition",
    "OptimizationException",
    "SOCritical",
    "SOMean",
    "SOVirial",
    "SphericalOverdensity",
]


@_framework.pluggable
class MassDefinition(_framework.Component):
    """A base class for a Mass Definition."""

    @staticmethod
    def critical_density(z=0, cosmo=Planck15):
        """Get the critical density of the Universe at redshift z, [h^2 Msun/Mpc^3]."""
        return (cosmo.critical_density(z) / cosmo.h ** 2).to(u.Msun / u.Mpc ** 3).value

    @classmethod
    def mean_density(cls, z=0, cosmo=Planck15):
        """Get the mean density of the Universe at redshift z, [h^2 Msun / Mpc^3]."""
        return cosmo.Om(z) * cls.critical_density(z, cosmo)

    def halo_density(self, z=0, cosmo=Planck15):
        r"""
        The density of haloes under this definition.

        May not exist in some definitions. Units are :math:`M_\odot h^2/{\rm Mpc}^3`.
        """
        raise AttributeError("halo_density does not exist for this Mass Definition")

    @property
    def colossus_name(self):
        """The name of the mass definition in Colossus format, if applicable."""
        return None

    def halo_overdensity_mean(self, z=0, cosmo=Planck15):
        return self.halo_density(z, cosmo) / self.mean_density(z, cosmo)

    def halo_overdensity_crit(self, z=0, cosmo=Planck15):
        return self.halo_density(z, cosmo) / self.critical_density(z, cosmo)

    def m_to_r(self, m, z=0, cosmo=Planck15):
        r"""
        Return the radius corresponding to m for this mass definition

        Parameters
        ----------
        m : float or array_like
            The mass to convert to radius. Should be in the same units (modulo volume)
            as  :meth:`halo_density`.

        Notes
        -----
        Computed as :math:`\left(\frac{3m}{4\pi \rho_{\rm halo}\right)`.
        """
        try:
            return (3 * m / (4 * np.pi * self.halo_density(z, cosmo))) ** (1.0 / 3.0)
        except AttributeError:
            raise AttributeError(
                f"{self.__class__.__name__} cannot convert mass to radius."
            )

    def r_to_m(self, r, z=0, cosmo=Planck15):
        r"""
        Return the mass corresponding to r for this mass definition

        Parameters
        ----------
        r : float or array_like
            The radius to convert to mass. Units should be compatible with
            :meth:`halo_density`.

        Notes
        -----
        Computed as :math:`\frac{4\pi r^3}{3} \rho_{\rm halo}`.
        """
        try:
            return 4 * np.pi * r ** 3 * self.halo_density(z, cosmo) / 3
        except AttributeError:
            raise AttributeError(
                f"{self.__class__.__name__} cannot convert radius to mass."
            )

    def _duffy_concentration(self, m, z=0):
        a, b, c, ms = 6.71, -0.091, 0.44, 2e12
        return a / (1 + z) ** c * (m / ms) ** b

    def change_definition(
        self, m: np.ndarray, mdef, profile=None, c=None, z=0, cosmo=Planck15
    ):
        r"""
        Change the spherical overdensity mass definition.

        This requires using a profile, for which the `halomod` package must be used.

        Parameters
        ----------
        m : float or array_like
            The halo mass to be changed, in :math:`M_\odot/h`. Must be
            broadcastable with `c`, if provided.
        mdef : :class:`MassDefinition` subclass instance
            The mass definition to which to change.
        profile : :class:`halomod.profiles.Profile` instance, optional
            An instantiated profile object from which to calculate the expected
            definition change. If not provided, a mocked NFW profile is used.
        c : float or array_like, optional
            The concentration(s) of the halos given. If not given, the concentrations
            will be automatically calculated using the profile object.

        Returns
        -------
        m_f : float or array_like
            The masses of the halos in the new definition.
        r_f : float or array_like
            The radii of the halos in the new definition.
        c_f : float or array_like
            The concentrations of the halos in the new definition.
        """
        if (
            c is not None
            and not np.isscalar(c)
            and not np.isscalar(m)
            and len(m) != len(c)
        ):
            raise ValueError(
                "If both m and c are arrays, they must be of the same length"
            )
        if c is not None and np.isscalar(c) and not np.isscalar(m):
            c = np.ones_like(m) * c
        if c is not None and np.isscalar(m) and not np.isscalar(c):
            m = np.ones_like(m) * m
        if c is not None:
            c = np.atleast_1d(c)
        m = np.atleast_1d(m)

        if profile is None:
            try:
                from halomod.concentration import Duffy08
                from halomod.profiles import NFW

                profile = NFW(
                    cm_relation=Duffy08(cosmo=Cosmology(cosmo)), mdef=self, z=z
                )
            except ImportError:
                raise ImportError(
                    "Cannot change mass definitions without halomod installed!"
                )

        if profile.z != z:
            warnings.warn(
                f"Redshift of given profile ({profile.z})does not match redshift "
                f"passed to change_definition(). Using the redshift directly passed."
            )
            profile.z = z

        if c is None:
            c = profile.cm_relation(m)

        rs = self.m_to_r(m, z, cosmo) / c
        rhos = profile._rho_s(c)

        if not hasattr(rhos, "__len__"):
            rhos = [rhos]
            c = [c]

        c_new = np.array(
            [
                _find_new_concentration(
                    rho, mdef.halo_density(z, cosmo), profile._h, cc
                )
                for rho, cc in zip(rhos, c)
            ]
        )

        if len(c_new) == 1:
            c_new = c_new[0]

        r_new = c_new * rs

        if len(r_new) == 1:
            r_new = r_new[0]

        return mdef.r_to_m(r_new, z, cosmo), r_new, c_new

    def __eq__(self, other):
        """Test equality with another object."""
        return (
            self.__class__.__name__ == other.__class__.__name__
            and self.params == other.params
        )


class SphericalOverdensity(MassDefinition):
    """An abstract base class for all spherical overdensity mass definitions."""

    pass

    def __str__(self):
        """Describe the overdensity in standard notation."""
        return f"{self.__class__.__name__}({self.params['overdensity']})"


class SOGeneric(SphericalOverdensity):
    """A generic SO definition which can claim equality with any SO."""

    def __init__(self, preferred: Optional[SphericalOverdensity] = None, **kwargs):
        super().__init__(**kwargs)
        self.preferred = preferred

    def __eq__(self, other):
        """Test equality with another object."""
        return isinstance(other, SphericalOverdensity)

    def __str__(self):
        return "SOGeneric"


class SOMean(SphericalOverdensity):
    """A mass definition based on spherical overdensity wrt mean background density."""

    _defaults = {"overdensity": 200}

    def halo_density(self, z=0, cosmo=Planck15):
        """The density of haloes under this definition."""
        return self.params["overdensity"] * self.mean_density(z, cosmo)

    @property
    def colossus_name(self):
        return f"{int(self.params['overdensity'])}m"


class SOCritical(SphericalOverdensity):
    """A mass definition based on spherical overdensity wrt critical density."""

    _defaults = {"overdensity": 200}

    def halo_density(self, z=0, cosmo=Planck15):
        """The density of haloes under this definition."""
        return self.params["overdensity"] * self.critical_density(z, cosmo)

    @property
    def colossus_name(self):
        return f"{int(self.params['overdensity'])}c"


class SOVirial(SphericalOverdensity):
    """A mass definition based on spherical overdensity.

    Density threshold isgiven by Bryan and Norman (1998).
    """

    def halo_density(self, z=0, cosmo=Planck15):
        """The density of haloes under this definition."""
        x = cosmo.Om(z) - 1
        overdensity = 18 * np.pi ** 2 + 82 * x - 39 * x ** 2
        return overdensity * self.mean_density(z, cosmo) / cosmo.Om(z)

    @property
    def colossus_name(self):
        return "vir"

    def __str__(self):
        """Describe the halo definition in standard notation."""
        return "SOVirial"


class FOF(MassDefinition):
    """A mass definition based on FroF networks with given linking length."""

    _defaults = {"linking_length": 0.2}

    def halo_density(self, z=0, cosmo=Planck15):
        r"""
        The density of halos under this mass definition.

        Note that for FoF halos, this is very approximate. We follow [1]_ and define
        :math:`rho_{FOF} = 9/(2\pi b^3) \rho_m`, with *b* the linking length. This
        assumes all groups are spherical and singular isothermal spheres.

        References
        ----------
        .. [1] White, Martin, Lars Hernquist, and Volker Springel. “The Halo Model and
           Numerical Simulations.” The Astrophysical Journal 550, no. 2 (April 2001):
           L129–32. https://doi.org/10.1086/319644.
        """
        overdensity = 9 / (2 * np.pi * self.params["linking_length"] ** 3)
        return overdensity * self.mean_density(z, cosmo)

    @property
    def colossus_name(self):
        return "fof"

    def __str__(self):
        """Describe the halo definition in standard notation."""
        return f"FoF(l={self.params['linking_length']})"


def from_colossus_name(name):
    if name == "vir":
        return SOVirial()
    elif name.endswith("c"):
        return SOCritical(overdensity=int(name[:-1]))
    elif name.endswith("m"):
        return SOMean(overdensity=int(name[:-1]))
    elif name == "fof":
        return FOF()
    else:
        raise ValueError(f"name '{name}' is an unknown mass definition to colossus.")


def _find_new_concentration(rho_s, halo_density, h=None, x_guess=5.0):
    r"""
    Find :math:`x=r/r_{\\rm s}` where the enclosed density has a particular value.

    .. note :: This is almost exactly the same code as profileNFW.xDelta from COLOSSUS.
               It may one day be changed to literally just call that function. For now
               it just sits here to be called whenever halomod is not installed

    Parameters
    ----------
    rho_s: float
        The central density in physical units :math:`M_{\odot} h^2 / {\\rm Mpc}^3`.
    halo_density: float
        The desired enclosed density threshold in physical units
        :math:`M_{\odot} h^2 / {\\rm Mpc}^3`.
    h : callable
        Return the enclosed density as function of r/r_s.

    Returns
    -------
    x: float
        The radius in units of the scale radius, :math:`x=r/r_{\\rm s}`, where the
        enclosed density reaches ``density_threshold``.
    """

    # A priori, we have no idea at what radius the result will come out, but we need to
    # provide lower and upper limits for the root finder. To balance stability and
    # performance, we do so iteratively: if there is no result within relatively
    # aggressive limits, we try again with more conservative limits.
    args = rho_s, halo_density
    x = None
    i = 0
    XDELTA_GUESS_FACTORS = [5.0, 10.0, 20.0, 100.0, 10000.0]

    if h is None:

        def h(x):
            return np.log(1.0 + x) - x / (1.0 + x)

    fnc = (
        lambda x, rhos, density_threshold: rhos * h(x) * 3.0 / x ** 3
        - density_threshold
    )

    while x is None and i < len(XDELTA_GUESS_FACTORS):
        try:
            xmin = x_guess / XDELTA_GUESS_FACTORS[i]
            xmax = x_guess * XDELTA_GUESS_FACTORS[i]
            x = sp.optimize.brentq(fnc, xmin, xmax, args)
        except Exception:
            i += 1

    if x is None:
        raise OptimizationException(
            "Could not determine x where the density threshold %.2f is satisfied."
            % halo_density
        )

    return x


class OptimizationException(Exception):
    """Exception class related to failed optimization."""

    pass
