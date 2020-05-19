"""
A model for mass definitions.

This is primarily inspired by Benedikt Diemer's COLOSSUS code: https://bdiemer.bitbucket.io/colossus/halo_mass_defs.html
"""
from .._internals import _framework
import numpy as np
import scipy as sp
import astropy.units as u
from astropy.cosmology import Planck15

__all__ = [
    "FOF",
    "MassDefinition",
    "OptimizationException",
    "SOCritical",
    "SOMean",
    "SOVirial",
    "SphericalOverdensity",
]


class MassDefinition(_framework.Component):
    """
    A base class for a Mass Definition.

    Parameters
    ----------
    cosmo: :class:`astropy.cosmology.FLRW` instance, optional
        The cosmology within which the haloes will be embedded.
    z : float, optional
        The redshift at which the mass definition is defined.
    model_parameters :
        Any parameters that uniquely affect the chosen Mass Definition model. These
        merge with `_defaults` to create the `.params` dictionary.
    """

    def __init__(self, cosmo=Planck15, z=0, **model_parameters):
        super(MassDefinition, self).__init__(**model_parameters)
        self.cosmo = cosmo
        self.z = z

        # Copy this from cosmology class to have it easily available here.
        self.mean_density0 = (
            (self.cosmo.Om0 * self.cosmo.critical_density0 / self.cosmo.h ** 2)
            .to(u.Msun / u.Mpc ** 3)
            .value
        )
        self.mean_density = (
            (self.cosmo.Om(z) * self.cosmo.critical_density(z) / self.cosmo.h ** 2)
            .to(u.Msun / u.Mpc ** 3)
            .value
        )

    @property
    def halo_density(self):
        r"""
        The density of haloes under this definition.

        May not exist in some definitions. Units are :math:`M_\odot h^2/{\rm Mpc}^3`.
        """
        raise AttributeError(
            "The overdensity attribute does not exist for this Mass Definition"
        )

    def m_to_r(self, m):
        r"""
        Return the radius corresponding to m for this mass definition

        Parameters
        ----------
        m : float or array_like
            The mass to convert to radius. Should be in the same units (modulo volume) as  :meth:`halo_density`.

        Notes
        -----
        Computed as :math:`\left(\frac{3m}{4\pi \rho_{\rm halo}\right)`.
        """
        try:
            return (3 * m / (4 * np.pi * self.halo_density)) ** (1.0 / 3.0)
        except AttributeError:
            raise AttributeError(
                "This Mass Definition has no way to convert mass to radius."
            )

    def r_to_m(self, r):
        r"""
        Return the mass corresponding to r for this mass definition

        Parameters
        ----------
        r : float or array_like
            The radius to convert to mass. Units should be compatible with :meth:`halo_density`.

        Notes
        -----
        Computed as :math:`\frac{4\pi r^3}{3} \rho_{\rm halo}`.
        """
        try:
            return 4 * np.pi * r ** 3 * self.halo_density / 3
        except AttributeError:
            raise AttributeError(
                "This Mass Definition has no way to convert radius to mass."
            )

    def _duffy_concentration(self, m):
        a, b, c, ms = 6.71, -0.091, 0.44, 2e12
        return a / (1 + self.z) ** c * (m / ms) ** b

    def change_definition(self, m, mdef, profile=None, c=None):
        r"""
        Change the spherical overdensity mass definition.

        This requires using a profile, for which the `halomod` package must be used.
        If `halomod` is not installed, the default here is to use the NFW profile and
        the Duffy+08 concentration-mass relation, for which a hardcoded solution is
        inbuilt to this method.

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
            # Use an NFW profile.
            if c is None:
                c = self._duffy_concentration(m)

            rs = self.m_to_r(m) / c
            rhos = m / rs ** 3 / 4.0 / np.pi / (np.log(1.0 + c) - c / (1.0 + c))

            c_new = np.array(
                [
                    _find_new_concentration(rho, mdef.halo_density, x_guess=cc)
                    for rho, cc in zip(rhos, c)
                ]
            )
        else:
            if c is None:
                c = profile.cm_relation(m)

            rs = profile._rs_from_m(m, c)
            rhos = profile._rho_s(c)
            c_new = np.array(
                [
                    _find_new_concentration(rho, mdef.halo_density, profile._h, cc)
                    for rho, cc in zip(rhos, c)
                ]
            )

        if len(c_new) == 1:
            c_new = c_new[0]

        r_new = c_new * rs

        if len(r_new) == 1:
            r_new = r_new[0]

        return mdef.r_to_m(r_new), r_new, c_new


class SphericalOverdensity(MassDefinition):
    """An abstract base class for all spherical overdensity mass definitions."""

    pass


class SOMean(SphericalOverdensity):
    """A mass definition based on spherical overdensity wrt mean background density."""

    _defaults = {"overdensity": 200}

    @property
    def halo_density(self):
        """The density of haloes under this definition."""
        return self.params["overdensity"] * self.mean_density


class SOCritical(SphericalOverdensity):
    """A mass definition based on spherical overdensity wrt critical density."""

    _defaults = {"overdensity": 200}

    @property
    def halo_density(self):
        """The density of haloes under this definition."""
        return self.params["overdensity"] * self.mean_density / self.cosmo.Om(self.z)


class SOVirial(SphericalOverdensity):
    """A mass definition based on spherical overdensity.

     Density threshold isgiven by Bryan and Norman (1998).
     """

    @property
    def halo_density(self):
        """The density of haloes under this definition."""
        x = self.cosmo.Om(self.z) - 1
        overdensity = 18 * np.pi ** 2 + 82 * x - 39 * x ** 2
        return overdensity * self.mean_density / self.cosmo.Om(self.z)


class FOF(MassDefinition):
    """A mass definition based on Friends-of-Friends networks with given linking length."""

    _defaults = {"linking_length": 0.2}

    @property
    def halo_density(self):
        r"""
        The density of haloes under this mass definition.

        Note that for FoF haloes, this is very approximate. We follow [1]_ and define
        :math:`rho_{FOF} = 9/(2\pi b^3) \rho_m`, with *b* the linking length. This
        assumes all groups are spherical and singular isothermal spheres.

        References
        ----------
        .. [1] White, Martin, Lars Hernquist, and Volker Springel. “The Halo Model and
           Numerical Simulations.” The Astrophysical Journal 550, no. 2 (April 2001):
           L129–32. https://doi.org/10.1086/319644.
        """
        overdensity = 9 / (2 * np.pi * self.params["linking_length"] ** 3)
        return overdensity * self.mean_density


def _find_new_concentration(rho_s, halo_density, h=None, x_guess=5.0):
    r"""
    Find :math:`x=r/r_{\\rm s}` where the enclosed density has a particular value.

    .. note :: This is almost exactly the same code as profileNFW.xDelta from COLOSSUS. It
               may one day be changed to literally just call that function. For now it just
               sits here to be called whenever halomod is not installed

    Parameters
    ----------
    rho_s: float
        The central density in physical :math:`M_{\odot} h^2 / {\\rm kpc}^3`.
    halo_density: float
        The desired enclosed density threshold in physical :math:`M_{\odot} h^2 / {\\rm kpc}^3`.
        This number can be generated from a mass definition and redshift using the
        :func:`~halo.mass_so.densityThreshold` function.
    h : callable
        Return the enclosed density as function of r/r_s.

    Returns
    -------
    x: float
        The radius in units of the scale radius, :math:`x=r/r_{\\rm s}`, where the enclosed
        density reaches ``density_threshold``.
    """

    # A priori, we have no idea at what radius the result will come out, but we need to
    # provide lower and upper limits for the root finder. To balance stability and performance,
    # we do so iteratively: if there is no result within relatively aggressive limits, we
    # try again with more conservative limits.
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
    pass
