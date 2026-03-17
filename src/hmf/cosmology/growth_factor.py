"""
Module defining calculations of the cosmological growth factor and growth rate.

The primary class, :class:`GrowthFactor`, intelligently dispatches to a number of
different methods for calculating the growth factor, depending on the cosmology and
the parameters. The most general method, which is applicable for any FLRW cosmology, is
:class:`ODEGrowthFactor`, which solves the full ODE for the growth factor. This is the
most general method, but also the slowest. For cosmologies with negligible radiation
density, the growth factor can be calculated using the integral form defined in
:class:`IntegralGrowthFactor`. If the cosmology is also *flat*, then the growth factor
can be calculated using the analytical formulae implemented in
:class:`Eisensten97GrowthFactor`, which is the fastest method. Various other
limiting cases and assumptions are also implemented. If using the base :class:`GrowthFactor` class,
the appropriate method will be chosen automatically such that it uses the fastest
available *correct* method.

In addition, a couple of approximate formulae are included, for example
:class:`Carroll1992` and :class:`GenMFGrowth`, which are not exact for any cosmology
but are very fast to compute and quite accurate across a broad range of cosmologies.

Full details of the formulae, their assumptions and limitations, and derivations, are
given in the `technical documentation <https://hmf.readthedocs.io/en/latest/technical/growth_factors.html>`_.
"""

import warnings
from abc import abstractmethod
from functools import cached_property
from typing import Any, ClassVar

import numpy as np
from astropy import cosmology
from scipy.integrate import solve_ivp
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from .._internals._framework import Component as Cmpt
from .._internals._framework import pluggable
from .._internals._utils import inherit_docstrings as _inherit

try:
    import camb

    HAVE_CAMB = True
except ImportError:  # pragma: nocover
    HAVE_CAMB = False


@pluggable
class _GrowthFactor(Cmpt):
    r"""General class for a growth factor calculation."""

    _defaults: ClassVar[dict[str, float]] = {"dlna": 0.01, "amin": 1e-8}

    def __init__(self, cosmo: cosmology.FLRW, **model_parameters):
        self.cosmo = cosmo
        super().__init__(**model_parameters)

    @cached_property
    def _lna(self):
        lna = np.arange(np.log(self.params["amin"]), 0, self.params["dlna"])
        if lna[-1] != 0:
            lna = np.concatenate((lna, [0]))
        return lna

    @cached_property
    def _zvec(self):
        return 1.0 / np.exp(self._lna) - 1.0

    @abstractmethod
    def _d_plus_unnormalized(self, z):
        """Compute the unnormalized growth factor, D^+(a)."""
        raise NotImplementedError  # pragma: nocover

    @cached_property
    def _d_plus0(self) -> float:
        """The un-normalized growth factor at z=0."""
        return self._d_plus_unnormalized(0.0)

    def growth_factor(self, z):
        r"""
        Compute the normalized growth factor, :math:`d(a) = D^+(a)/D^+(a=1)`.

        Parameters
        ----------
        z : array_like
            Redshift.

        Returns
        -------
        gf : array_like
            The growth factor at `z`.
        """
        return self._d_plus_unnormalized(z) / self._d_plus0

    @cached_property
    def _growth_rate_spline(self):
        """Calculate the growth rate, dln(d)/dln(a), using the spline derivative.

        This is here so that sub-classes can use it. But they need not, if there is a
        faster/analytic way to calculate the growth rate.
        """
        return Spline(self._lna, np.log(self.growth_factor(self._zvec))).derivative()

    def growth_rate(self, z) -> float | np.ndarray:
        r"""
        Compute the growth rate, :math:`f(a) = d\ln D^+ / d\ln a`.

        Parameters
        ----------
        z : array_like
            Redshift.

        Returns
        -------
        gr : array_like
            The growth rate at `z`.
        """
        # The bog standard method will just be numerical derivative.
        return self._growth_rate_spline(np.log(1 / (1 + z)))


class GrowthFactor(_GrowthFactor):
    r"""Growth factor calculation that intelligently chooses which method to use.

    This class chooses which growth factor calculation to use based on the cosmology and
    the parameters. It prioritises first accuracy, then performance, conditional on
    the parameters. In particular, it uses the following logic to compute the growth factor:

    - If the dark energy equation of state is not w = -1, use the ODE solution; otherwise
    - If the cosmology has no radiation component, or the redshift is low enough such
      that the radiation component can be neglected:
      - If the cosmology is flat, use analytic forms of Eisenstein 1997
      - Otherwise, use the integral form defined in Heath 1977.
    - Otherwise, solve the full ODE numerically (also presented in the docs).

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create empty attributes for the different growth factor calculations,
        # which will be filled in as needed.
        self._ode_growth_factor = None
        self._integral_growth_factor = None
        self._eisenstein_growth_factor = None

    def _d_plus_unnormalized(self, z):
        """Calculate the growth factor at redshift z.

        See class documentation for the logic tree used here.
        """
        a = 1 / (1 + z)

        if self._ode_growth_factor is not None:
            # Might as well use it since it's already been initialized and it's the
            # most accurate.
            return self._ode_growth_factor._d_plus_unnormalized(z)

        if not isinstance(self.cosmo, cosmology.LambdaCDM):
            self._ode_growth_factor = ODEGrowthFactor(self.cosmo, **self.params)
            return self._ode_growth_factor._d_plus_unnormalized(z)

        if (np.all(z < 5) or self.cosmo.Ogamma0 == 0) and isinstance(
            self.cosmo, cosmology.FlatLambdaCDM
        ):
            # At low z for a flat cosmology, we can use the formulae in Eisenstein 1997,
            # which is very fast.
            self._eisenstein_growth_factor = Eisenstein97GrowthFactor(self.cosmo, **self.params)
            return self._eisenstein_growth_factor._d_plus_unnormalized(z)

        if self.cosmo.Ogamma0 == 0 or np.all(a > 100 * self.cosmo.Ogamma0 / self.cosmo.Om0):
            self._integral_growth_factor = IntegralGrowthFactor(self.cosmo, **self.params)
            return self._integral_growth_factor._d_plus_unnormalized(z)

        self._ode_growth_factor = ODEGrowthFactor(self.cosmo, **self.params)
        return self._ode_growth_factor._d_plus_unnormalized(z)

    def growth_rate(self, z):
        """Calculate the growth rate at redshift z.

        See class documentation for the logic tree used here.
        """
        a = 1 / (1 + z)

        if self._ode_growth_factor is not None:
            # Might as well use it since it's already been initialized and it's the
            # most accurate.
            return self._ode_growth_factor.growth_rate(z)

        if not isinstance(self.cosmo, cosmology.LambdaCDM):
            self._ode_growth_factor = ODEGrowthFactor(self.cosmo, **self.params)
            return self._ode_growth_factor.growth_rate(z)

        if (np.all(z < 5) or self.cosmo.Ogamma0 == 0) and isinstance(
            self.cosmo, cosmology.FlatLambdaCDM
        ):
            # At low z for a flat cosmology, we can use the formulae in Eisenstein 1997,
            # which is very fast.
            self._eisenstein_growth_factor = Eisenstein97GrowthFactor(self.cosmo, **self.params)
            return self._eisenstein_growth_factor.growth_rate(z)

        if self.cosmo.Ogamma0 == 0 or np.all(a > 100 * self.cosmo.Ogamma0 / self.cosmo.Om0):
            self._integral_growth_factor = IntegralGrowthFactor(self.cosmo, **self.params)
            return self._integral_growth_factor.growth_rate(z)

        self._ode_growth_factor = ODEGrowthFactor(self.cosmo, **self.params)
        return self._ode_growth_factor.growth_rate(z)


class ODEGrowthFactor(_GrowthFactor):
    r"""
    Growth factor calculation that solves the full ODE.

    The ODE solved here is given in many references, e.g. Peebles 1980 "The Large
    Scale Structure of the Universe". The implementation here does not assume either
    flatness, a cosmological constant, or negligible radiation density, and so should be
    accurate for any FLRW cosmology. The growth factor is normalised such that D+(a) = 1
    at a=1.

    Notes
    -----
    The growth factor is calculated by solving the ODE for the growth function
    :math:`G(a) = D+(a)/a`, which is given by

    .. math:: G''(a) + \left(\frac{H'(a)}{H(a)} + \frac{5}{a}\right) G'(a) +
        \left(\frac{3}{a^2} + \frac{H'(a)}{a H(a)} -
        \frac{3}{2} \frac{\Omega_{m,0}}{a^5 H(a)^2}\right) G(a) = 0.

    The growth factor is then given by :math:`D^+(a) = G(a) \cdot a`. We use
    astropy to calculate :math:`H(a)` and its derivative, which ensures that the growth
    factor is calculated self-consistently with the cosmology. The growth factor is
    normalised such that :math:`D^+(a=1) = 1`.

    The ODE is solved using scipy's `solve_ivp` function, with initial conditions set in
    the matter-dominated era, where :math:`G(a) \approx 1` and :math:`G'(a) \approx 0`.
    Although these initial conditions are used for the ODE solution, the growth factor
    is ultimately normalised such that :math:`D^+(a=1) = 1`, so the final result is not
    sensitive to the exact choice of initial conditions (beyond the condition on
    the normalisation).

    The growth rate is calculated by taking the derivative of the growth factor spline,
    which is equivalent to the expression given in many references, e.g. Carroll et al.
    1992, eq. 30:

    .. math:: \frac{d\ln D^+}{d\ln a}.

    In this class this is done using the spline derivative, on the spline of the growth
    factor itself, which ensures that the growth rate is consistent with the growth
    factor.

    Since the ODE solution is numerical, it must be evaluated on a grid, and since it
    is solved as an initial value problem, the grid must start at some early time.
    The parameters `dlna` and `amin` control the grid on which the ODE is solved, and so
    the accuracy of the solution. The grid is in log-space for the scale factor, and so
    `dlna` controls the step-size in log-space, while `amin` controls how far back in
    time the solution is calculated. The default values should be sufficient for most
    cosmologies.
    """

    @cached_property
    def _dlnefunc_dlna(self):
        """Compute the derivative of ln(H) with respect to ln(a).

        This is used later for computing H'(a)/H(a) == (1/a)*dlnH/dlna for the growth
        rate calculation.
        """
        return Spline(self._lna, np.log(self.cosmo.efunc(self._zvec))).derivative()

    @cached_property
    def _ode_solution(self):
        # This function implementation borrowed heavily from the Colossus code
        # by Benedikt Diemer

        def ode(a, y):
            D, Dp = y

            z = 1 / a - 1
            lna = np.log(a)

            dlnhdlna = self._dlnefunc_dlna(lna)
            return [
                Dp,
                -(3 / a + dlnhdlna / a) * Dp
                + D * 3 * self.cosmo.Om0 / (2 * self.cosmo.efunc(z) ** 2 * a**5),
            ]

        a = np.exp(self._lna)

        # The initial values here assume we are well back into the radiation
        # dominated era. The asymptotic solution in this era is a constant 2q/3 where
        # q is the radiation to matter ratio. If there is no radiation, then we are in
        # the matter dominated era and the solution is just a (i.e. with a derivative
        # of unity)
        aeq = self.cosmo.Ogamma0 / self.cosmo.Om0
        sol = solve_ivp(
            ode,
            (a.min(), a.max()),
            [2 * aeq / 3 if aeq > 0 else a.min(), 0.0 if aeq > 0 else 1.0],
            t_eval=a,
            atol=1e-6,
            rtol=1e-6,
            vectorized=True,
        )

        D = sol["y"][0, :]

        if (sol["status"] != 0) or (D.shape[0] != a.shape[0]):
            raise Exception("The calculation of the growth factor failed.")

        return (Spline(self._lna, np.log(D)), Spline(D, self._zvec))

    def _d_plus_unnormalized(self, z):
        """Calculate the unnormalized growth factor D^+(a) at redshift z."""
        a = 1 / (1 + z)
        lna = np.log(a)

        # Interpolate the solution to get D^+(a) at the desired redshift(s).
        return np.exp(self._ode_solution[0](lna))

    def growth_factor_inverse(self, d):
        r"""
        Inverse of the growth factor function, i.e. z as a function of growth factor.

        Parameters
        ----------
        d : float | np.ndarray
            The normalised growth factor

        Returns
        -------
        float | np.ndarray
            The redshift corresponding to the given growth factor.
        """
        return self._ode_solution[1](d)

    @cached_property
    def _growth_rate_spline(self):
        r"""Calculate the growth rate, dln(d)/dln(a).

        Given the integral form of the growth factor, we can write out the growth
        rate analytically as

        .. math:: \frac{d\ln D^+}{d\ln a} = dlnH/dln(a)
            + 2.5 \frac{\Omega_m(a)}{a^2 H^2(a) D^+(a)}.
        """
        return self._ode_solution[0].derivative()

    def growth_rate(self, z: float | np.ndarray) -> float | np.ndarray:
        """
        Growth rate, dln(D+)/dln(a).

        Parameters
        ----------
        z : float | np.ndarray
            The redshift
        """
        lna = np.log(1 / (1 + z))
        return self._growth_rate_spline(lna)


class IntegralGrowthFactor(_GrowthFactor):
    r"""Growth factor computed using the integral of Heath 1977.

    This growth factor is only applicable when the radiation density is negligible, and
    so is only accurate at low redshifts in cosmologies with radiation.
    """

    _defaults: ClassVar[dict[str, float]] = {"dlna": 0.01, "amin": 1e-8}

    @cached_property
    def _lna(self):
        lna = np.arange(np.log(self.params["amin"]), 0, self.params["dlna"])
        if lna[-1] != 0:
            lna = np.concatenate((lna, [0]))
        return lna

    @cached_property
    def _zvec(self):
        return 1.0 / np.exp(self._lna) - 1.0

    @cached_property
    def _integral(self):
        r"""The integral :math:`\int_0^a da' / (a'^3 E(a')^3)`.

        Parameters
        ----------
        a : array_like
            Scale factor(s) at which to evaluate the integral.
        """
        a = np.exp(self._lna)
        return Spline(self._lna, a / (a * self.cosmo.efunc(self._zvec)) ** 3).antiderivative()

    def _d_plus_unnormalized(self, z: float | np.ndarray) -> float | np.ndarray:
        r"""
        Compute the un-normalized growth factor, D^+(a).

        This function computes the integral form of the growth factor given in Heath
        1977, which is valid for (both flat and non-flat) cosmologies with negligible
        radiation density and a constant dark energy equation of state.

        This is normalized such that :math:`D^+(a->0) = a` in the matter-dominated era.
        This can depart significantly from this behaviour at very high redshifts in
        cosmologies with radiation, so the normalization is not exact.

        Parameters
        ----------
        z
            The redshift

        Returns
        -------
        dplus
            The un-normalised growth factor -- same type as ``z``.
        """
        a_min = np.exp(self._lna).min()
        a = 1 / (1 + z)
        lna = np.log(a)

        if np.any(a < self.cosmo.Ogamma0 / self.cosmo.Om0 * 50):
            warnings.warn(
                f"The IntegralGrowthFactor is not accurate at high redshifts "
                f"({np.max(z):.2f}) in cosmologies with radiation. Consider using the "
                "ODEGrowthFactor instead.",
                stacklevel=2,
            )
        if not isinstance(self.cosmo, cosmology.LambdaCDM):
            warnings.warn(
                "The IntegralGrowthFactor is only accurate for cosmologies with a "
                "constant dark energy equation of state. Consider using the "
                "ODEGrowthFactor instead.",
                stacklevel=2,
            )

        if np.any(z < 0):
            raise ValueError("Redshifts <0 not supported")
        if np.any(a < a_min):
            raise ValueError(f"Cannot compute integral for z > {1 / a_min - 1}. Set amin lower.")

        return self._integral(lna) * self.cosmo.efunc(z) * 2.5 * self.cosmo.Om0

    def _efunc(self, z):
        """Calculate E(z) = H(z)/H0.

        Sub-classes can over-ride this method, in order to apply specific assumptions
        from their particular models. In this class, we keep it general. This will
        mean that both the growth factor and growth rate are consistent with each other
        and also that each is *incorrect* if either the cosmology is not a cosmo constant
        or has non-negligible radiation, since the formula for the growth factor is only
        accurate under those assumptions.
        """
        return self.cosmo.efunc(z)

    @cached_property
    def _dlnhdlna(self):
        return Spline(self._lna, np.log(self._efunc(self._zvec))).derivative()

    def growth_rate(self, z):
        """Calculate the growth rate, using the integral form of the growth factor."""
        a = 1 / (1 + z)

        # Note that we use _d_plus_unnormalized instead of the normalized growth_factor.
        # This is because the coefficients in front of the second term are chosen
        # based on the normalization specified in _d_plus_unnormalized (i.e. 5/2 Om0).
        # This is a convenient choice so that D(a) -> a for a -> 0.
        return self._dlnhdlna(np.log(a)) + 2.5 * self.cosmo.Om0 / (
            a**2 * self._efunc(z) ** 2 * self._d_plus_unnormalized(z)
        )


class Eisenstein97GrowthFactor(IntegralGrowthFactor):
    r"""Growth factor calculated using the formulae in Eisenstein 1997.

    This is the result from Eqs 8-10.

    This growth factor is only applicable for flat cosmology with a cosmological
    constant and negligible radiation (i.e. low redshifts).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _efunc(self, z):
        """Calculate E(z) = H(z)/H0 under assumptions of the Eisenstein+97 formulae."""
        a = 1 / (1 + z)
        return np.sqrt(self.cosmo.Om0 * a**-3 + (1 - self.cosmo.Om0))

    def _dlnhdlna(self, lna):
        # We want the growth rate to be consistent with the growth factor, so we
        # calculate a*H'(a)/H(a) under the same assumptions as directly assumed in the
        # analytic formula for the growth factor, rather than
        # using the cosmology given. This ensures that the growth rate is consistent
        # with the growth factor, even if the growth factor is not exact.
        # The formula for D+ in Eisenstein 1997 assumes both flatness and negligible
        # radiation, so we calculate a*H'(a)/H(a) under the same assumptions.
        a = np.exp(lna)
        Hsq = self._efunc(1 / a - 1) ** 2
        return -1.5 * self.cosmo.Om0 * a**-3 / Hsq

    def _d_plus_unnormalized(self, z: float | np.ndarray) -> float | np.ndarray:
        from scipy.special import ellipeinc, ellipkinc

        if np.any(z > 100) and self.cosmo.Ogamma0 > 0:
            warnings.warn(
                "The Eisenstein97GrowthFactor is not accurate at high redshifts (z > 100)."
                " Consider using the ODEGrowthFactor instead.",
                stacklevel=2,
            )
        if not isinstance(self.cosmo, cosmology.FlatLambdaCDM):
            raise ValueError("Eisenstein97GrowthFactor only supports flat LambdaCDM cosmologies")

        a = 1 / (1 + z)
        v = 1 / a * (self.cosmo.Om0 / self.cosmo.Ode0) ** (1 / 3)

        beta = np.arccos((v + 1 - np.sqrt(3)) / (v + 1 + np.sqrt(3)))

        efunc = ellipeinc(beta, np.sin(75 * np.pi / 180) ** 2)
        ffunc = ellipkinc(beta, np.sin(75 * np.pi / 180) ** 2)

        term1 = 3**0.25 * np.sqrt(1 + v**3) * (efunc - 1 / (3 + np.sqrt(3)) * ffunc)
        term2 = (1 - (np.sqrt(3) + 1) * v**2) / (v + 1 + np.sqrt(3))

        # Watch out for loss of precision at high v.
        v = np.atleast_1d(v)
        term1 = np.atleast_1d(term1)
        term2 = np.atleast_1d(term2)

        mask = v > 8
        result = np.empty_like(v)
        result[mask] = 1 - (2 / 11) * v[mask] ** -3 + (16 / 187) * v[mask] ** -6
        result[~mask] = (5 / 3) * v[~mask] * (term1[~mask] + term2[~mask])
        return result * a


class Heath77GrowthFactor(IntegralGrowthFactor):
    r"""Growth factor calculated using the formulae in Heath 1977.

    These results apply when Lambda = 0 and the radiation density is negligible, and is
    given in Eq 13 of Heath 1977.
    """

    def _d_plus_unnormalized(self, z: float | np.ndarray) -> float | np.ndarray:
        if self.cosmo.Ogamma0 > 0 and np.any(
            z > 1 / (100 * self.cosmo.Ogamma0 / self.cosmo.Om0) - 1
        ):
            warnings.warn(
                "The Heath77GrowthFactor is not accurate at high redshifts (z > "
                "100*Ogamma0/Om0 - 1) in cosmologies with radiation. Consider using the"
                " ODEGrowthFactor instead.",
                stacklevel=2,
            )
        if (hasattr(self.cosmo, "w0") and self.cosmo.w0 != -1) or (
            hasattr(self.cosmo, "wa") and self.cosmo.wa != 0
        ):
            warnings.warn(
                "The Heath77GrowthFactor is only accurate for cosmologies with a "
                "constant dark energy equation of state. Consider using the "
                "ODEGrowthFactor instead.",
                stacklevel=2,
            )

        sigma0 = self.cosmo.Om0 / 2

        p = (2 * sigma0 * z + 1) * (1 + z) ** 2
        if sigma0 > 0.5:
            theta = np.arccos((sigma0 * z - sigma0 + 1) / (sigma0 * (1 + z)))
        elif sigma0 < 0.5:
            theta = np.arccosh((sigma0 * z - sigma0 + 1) / (sigma0 * (1 + z)))
        elif sigma0 == 0.5 and self.cosmo.Ode0 == 0:
            return 1 / (1 + z)  # Einstein-de Sitter case
        else:
            raise ValueError("Heath77GrowthFactor cannot compute OmegaM = 1 case with Lambda!=0.")

        term1 = (6 * sigma0 * z + 4 * sigma0 + 1) / np.abs(2 * sigma0 - 1)
        term2 = 3 * theta * sigma0 * np.sqrt(p) / np.abs(2 * sigma0 - 1) ** (3 / 2)

        return term1 - term2


@_inherit
class FromFile(_GrowthFactor):
    r"""
    Import a growth factor from file.

    .. note:: The file should be in the same format as output from CAMB,
              or else in two-column ASCII format (z,d).

    Parameters
    ----------
    \*\*model_parameters : unpack-dict
        Parameters specific to this model. In this case, available
        parameters are the following. To see their default values,
        check the :attr:`_defaults` class attribute.

        :fname: str
            Location of the file to import.
    """

    supported_cosmos = (cosmology.LambdaCDM, cosmology.w0waCDM, cosmology.wCDM)
    _defaults: ClassVar[dict[str, Any]] = {"dlna": 0.01, "amin": 1e-8, "fname": ""}

    def _d_plus_unnormalized(self, z):
        growth = np.genfromtxt(self.params["fname"])[:, [0, 1]].T

        z_out = growth[0, :]
        d_out = growth[1, :]
        return Spline(z_out, d_out, k=1)(z)


@_inherit
class FromArray(FromFile):
    r"""
    Use a spline over a given array to define the growth factor.

    Parameters
    ----------
    \*\*model_parameters : unpack-dict
        Parameters specific to this model. In this case, available
        parameters are the following. To see their default values,
        check the :attr:`_defaults` class attribute.

        :z: array
            Redshifts

        :d: array
            The growth factor at `z`.
    """

    supported_cosmos = (cosmology.LambdaCDM, cosmology.w0waCDM, cosmology.wCDM)
    _defaults: ClassVar[dict[str, Any]] = {"dlna": 0.01, "amin": 1e-8, "z": None, "d": None}

    def _d_plus_unnormalized(self, z):
        z_out = self.params["z"]
        d_out = self.params["d"]

        if z_out is None or d_out is None:
            raise ValueError("You must supply an array for both z and d for this Growth model")
        if len(z_out) != len(d_out):
            raise ValueError("z and d must have same length")

        return Spline(z_out, d_out, k=1)(z)


@_inherit
class GenMFGrowth(_GrowthFactor):
    r"""
    Port of growth factor routines found in the ``genmf`` code.

    Parameters
    ----------
    cosmo : ``astropy.cosmology.FLRW`` instance
        Cosmological model.
    \*\*model_parameters : unpack-dict
        Parameters specific to this model. In this case, available
        parameters are as follows.To see their default values, check
        the :attr:`_defaults` class attribute.

        :dz: Step-size for redshift integration
        :zmax: Maximum redshift to integrate to. Only used for :meth:`growth_factor_fn`.
    """

    def _general_case(self, w, x):
        x = np.atleast_1d(x)
        xn_vec = np.linspace(0, x.max(), 1000)

        func = Spline(xn_vec, (xn_vec / (xn_vec**3 + 2)) ** 1.5)

        g = np.array([func.integral(0, y) for y in x])
        return ((x**3.0 + 2.0) ** 0.5) * (g / x**1.5)

    def _d_plus_unnormalized(self, z):
        """
        Compute the unnormalized growth factor, :math:`D^+(a)`.

        This uses an approximation only valid in closed or
        flat cosmologies, ported from ``genmf``.

        Parameters
        ----------
        z : array_like
            Redshift.

        Returns
        -------
        gf : array_like
            The growth factor at `z`.
        """
        a = 1 / (1 + z)
        w = 1 / self.cosmo.Om0 - 1.0

        if not isinstance(self.cosmo, cosmology.LambdaCDM):
            raise ValueError(
                "The GenMFGrowth factor is only accurate for LambdaCDM cosmologies. "
                "Consider using the ODEGrowthFactor instead."
            )

        if not isinstance(self.cosmo, cosmology.FlatLambdaCDM) and not self.cosmo.Ok0 >= 0:
            raise ValueError("GenMFGrowth only supports flat or open LambdaCDM cosmologies")

        if self.cosmo.Om0 == 1:
            return a
        if self.cosmo.Ode0 > 0:
            xn = (2.0 * w) ** (1.0 / 3)
            aofxn = self._general_case(w, xn)
            x = a * xn
            aofx = self._general_case(w, x)
            return aofx / aofxn
        dn = 1 + 3 / w + (3 * ((1 + w) ** 0.5) / w**1.5) * np.log((1 + w) ** 0.5 - w**0.5)
        x = w * a
        return (1 + 3 / x + (3 * ((1 + x) ** 0.5) / x**1.5) * np.log((1 + x) ** 0.5 - x**0.5)) / dn


@_inherit
class Carroll1992(GrowthFactor):
    r"""
    Analytic approximation for the growth factor from Carroll et al. 1992.

    This formula is based on a formula in Lahav+1991 and Schechter and Lightman 1991.

    This formula is explicitly only valid at z=0, and a warning will be raised if you
    try to use it at z>0. However, the formula is actually pretty accurate at
    non-zero redshifts if redshift-dependent values for Omega_m and Omega_L are used.
    """

    def _d_plus_unnormalized(self, z):
        """Calculate the unnormalized growth factor."""
        a = 1 / (1 + z)
        Omega_m = self.cosmo.Om(z)  # om / denom
        Omega_L = self.cosmo.Ode(z)  # / denom

        coeff = 5.0 * Omega_m / (2.0 / a)
        term1 = Omega_m ** (4.0 / 7.0)
        term3 = (1.0 + 0.5 * Omega_m) * (1.0 + Omega_L / 70.0)
        return coeff / (term1 - Omega_L + term3)

    def growth_rate(self, z):
        """
        Growth rate, dln(d)/dln(a).

        Parameters
        ----------
        z : float
            The redshift
        """
        Omega_m = self.cosmo.Om(z)
        Omega_L = self.cosmo.Ode(z)

        return Omega_m ** (4.0 / 7.0) + Omega_L / 70.0 * (1.0 + Omega_m / 2.0)


if HAVE_CAMB:

    @_inherit
    class CambGrowth(GrowthFactor):
        """
        Growth factor computed using CAMB at k/h = 1.0.

        Recommended for non-LambdaCDM cosmologies (e.g., wCDM) as it correctly
        deals with their growth evolution. For standard LCDM, other classes are
        preferred since this class requires re-calculating the transfer function.

        """

        supported_cosmos = (cosmology.LambdaCDM, cosmology.w0waCDM, cosmology.wCDM)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # Save the CAMB object properly for use
            # Set the cosmology
            self.p = self._get_camb_params(self.cosmo)

        @classmethod
        def _get_camb_params(cls, cosmo: cosmology.FLRW) -> camb.CAMBparams:
            p = camb.CAMBparams()

            if cosmo.Ob0 is None:
                raise ValueError(
                    "If using CAMB, the baryon density must be set explicitly in the cosmology."
                )

            if cosmo.Tcmb0.value == 0:
                raise ValueError(
                    "If using CAMB, the CMB temperature must be set explicitly in the cosmology."
                )

            p.set_cosmology(
                H0=cosmo.H0.value,
                ombh2=cosmo.Ob0 * cosmo.h**2,
                omch2=(cosmo.Om0 - cosmo.Ob0) * cosmo.h**2,
                mnu=sum(cosmo.m_nu.value),
                neutrino_hierarchy="degenerate",
                omk=cosmo.Ok0,
                nnu=cosmo.Neff,
                standard_neutrino_neff=cosmo.Neff,
                TCMB=cosmo.Tcmb0.value,
            )

            p.WantTransfer = True

            # Set the DE equation of state. We only support constant w.
            if hasattr(cosmo, "w0"):
                p.set_dark_energy(w=cosmo.w0)

            return p

        @cached_property
        def _camb_transfers(self):
            return camb.get_transfer_functions(self.p)

        @cached_property
        def _t0(self):
            """The Transfer function at z=0."""
            return self._camb_transfers.get_redshift_evolution(0.01, 0.0, ["delta_tot"])[0][0]

        def growth_factor(self, z):
            """
            Calculate :math:`d(a) = D^+(a)/D^+(a=1)`.

            Parameters
            ----------
            z : float
                The redshift

            Returns
            -------
            float
                The normalised growth factor.
            """
            growth = (
                self._camb_transfers.get_redshift_evolution(0.01, z, ["delta_tot"]).flatten()
                / self._t0
            )
            if len(growth) == 1:
                return growth[0]
            return growth

        def __getstate__(self):
            """Get the state of the object, converting the CAMBparams to a dict."""
            dct = self.__dict__.copy()

            # Can't pickle/copy CAMBparams or CAMBResults
            del dct["p"]
            if "_camb_transfers" in dct:
                del dct["_camb_transfers"]

            return dct

        def __setstate__(self, state):
            """Set the state of the object, reconstructing the CAMBparams object."""
            self.__dict__ = state

            self.p = self._get_camb_params(self.cosmo)
