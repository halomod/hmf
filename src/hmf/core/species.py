r"""The matter species of the transfer and growth models, and their mean densities.

Every transfer and growth model gives each matter species (:data:`Species`):
``"cb"`` (CDM + baryons) and ``"tot"`` (total matter, including massive neutrinos).
:data:`MATTER_SPECIES` lists them, and :func:`check_species` validates one.

The density parameter of a species today (:func:`omega_m0`) and at z
(:func:`omega_m`), and its mean comoving density today (:func:`rho_mean0`), come
from the astropy cosmology. Astropy's ``Om0`` is CDM + baryons only: it counts
massive neutrinos in ``Onu0``, with the massless ones. So ``"cb"`` is ``Om0``, and
``"tot"`` adds the density of the massive neutrinos today,
:math:`\Omega_{\nu,0}h^2 \approx \sum m_\nu / 93.14\,{\rm eV}`.

The functions take and return plain floats and arrays: :func:`rho_mean0` is in the
canonical density unit (:data:`~hmf.core.units.rho_unit`), and the density
parameters are dimensionless.

Examples
--------
>>> from astropy.cosmology import Planck18
>>> from hmf.core import species
>>> species.MATTER_SPECIES
('cb', 'tot')
>>> species.omega_m0(Planck18, "tot") > species.omega_m0(Planck18, "cb")
True
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal, get_args

import astropy.units as u
import numpy as np
import numpy.typing as npt
from astropy.cosmology import FLRW

from .units import RHO_CRIT0_H2, rho_unit

__all__ = [
    "CAMB_COLUMNS",
    "MATTER_SPECIES",
    "Species",
    "check_species",
    "omega_m",
    "omega_m0",
    "rho_mean0",
]

#: A matter species: ``"cb"`` (CDM + baryons) or ``"tot"`` (total matter).
Species = Literal["cb", "tot"]

#: The matter species every model provides, in order.
MATTER_SPECIES: tuple[str, ...] = get_args(Species)

#: The zero-based column of each species in CAMB's transfer-function output: the
#: columns of a CAMB transfer file, and the rows of ``MatterTransferData.transfer_data``
#: (CAMB's 1-based ``Transfer_tot`` and ``Transfer_nonu``, minus 1). Column 0 is k.
CAMB_COLUMNS: Mapping[str, int] = MappingProxyType({"tot": 6, "cb": 7})

#: The energy density of one massless neutrino species relative to the photons',
#: :math:`\frac78(4/11)^{4/3}`, before astropy's factor ``Neff / N`` per species.
_MASSLESS_NU_PER_PHOTON = 7 / 8 * (4 / 11) ** (4 / 3)

#: The critical density today, as a float in the canonical density unit.
_RHO_CRIT0_H2 = float(RHO_CRIT0_H2.to_value(rho_unit))


def check_species(species: str) -> str:
    """Raise ``ValueError`` unless ``species`` is one of :data:`MATTER_SPECIES`.

    Parameters
    ----------
    species
        The species to check.

    Returns
    -------
    str
        ``species``.

    Raises
    ------
    ValueError
        If it is not a matter species.
    """
    if species not in MATTER_SPECIES:
        raise ValueError(f"species must be one of {MATTER_SPECIES}, got {species!r}.")
    return species


def _omega_massive_nu0(cosmology: FLRW) -> float:
    """The density parameter of the massive neutrinos today (0 if there are none).

    Astropy's ``Onu0`` holds every neutrino species, each with ``Neff / N`` of the
    density of one standard species; the massless ones are radiation, so they are
    subtracted.
    """
    if not cosmology.has_massive_nu:
        return 0.0
    m_nu = np.atleast_1d(cosmology.m_nu.to_value(u.eV))
    n_massless = np.count_nonzero(m_nu == 0)
    massless = cosmology.Ogamma0 * _MASSLESS_NU_PER_PHOTON * cosmology.Neff / m_nu.size
    return float(cosmology.Onu0 - massless * n_massless)


def omega_m0(cosmology: FLRW, species: str = "cb") -> float:
    r"""The density parameter of a matter species today.

    Parameters
    ----------
    cosmology
        The cosmology.
    species
        ``"cb"`` (CDM + baryons: astropy's ``Om0``) or ``"tot"`` (also the massive
        neutrinos).

    Returns
    -------
    float
        :math:`\Omega_{s,0}`.

    Raises
    ------
    ValueError
        If ``species`` is not a matter species.
    """
    if check_species(species) == "cb":
        return float(cosmology.Om0)
    return float(cosmology.Om0) + _omega_massive_nu0(cosmology)


def rho_mean0(cosmology: FLRW, species: str = "cb") -> float:
    r"""The mean comoving density of a matter species, in Msun h^2 / Mpc^3.

    A plain float in the canonical density unit
    (:data:`~hmf.core.units.rho_unit`), for library code.

    Parameters
    ----------
    cosmology
        The cosmology.
    species
        ``"cb"`` (CDM + baryons) or ``"tot"`` (also the massive neutrinos).

    Returns
    -------
    float
        :math:`\Omega_{s,0}\,\rho_{c,0}`, in Msun h^2 / Mpc^3.

    Raises
    ------
    ValueError
        If ``species`` is not a matter species.
    """
    return omega_m0(cosmology, species) * _RHO_CRIT0_H2


def omega_m(cosmology: FLRW, z: npt.ArrayLike, species: str = "cb") -> npt.NDArray[np.float64]:
    r"""The density parameter of a matter species at redshift z.

    .. math:: \Omega_s(z) = \frac{\Omega_{s,0}(1+z)^3}{E^2(z)},

    with astropy's :math:`E(z) = H(z)/H_0`, which includes radiation, curvature and
    dark energy. For ``"cb"`` this is astropy's ``Om(z)``. Massive neutrinos
    (``"tot"``) are counted as non-relativistic matter, with their density today,
    which holds while they are non-relativistic: for z below about
    :math:`m_\nu / 5\times10^{-4}\,{\rm eV}` (z ~ 100 for 0.06 eV).

    Parameters
    ----------
    cosmology
        The cosmology.
    z
        Redshift(s), dimensionless.
    species
        ``"cb"`` (CDM + baryons) or ``"tot"`` (also the massive neutrinos).

    Returns
    -------
    numpy.ndarray
        :math:`\Omega_s(z)`, dimensionless, with the shape of ``z`` (0-d for a
        scalar).

    Raises
    ------
    ValueError
        If ``species`` is not a matter species.
    """
    z = np.asarray(z, dtype=np.float64)
    out: npt.NDArray[np.float64] = (
        omega_m0(cosmology, species) * (z + 1.0) ** 3 * cosmology.inv_efunc(z) ** 2
    )
    return out
