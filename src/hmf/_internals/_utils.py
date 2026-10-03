import warnings
from inspect import getmembers, ismethod

import numpy as np
from astropy.cosmology import FLRW

#: Matter fields that the CAMB-based models can compute. ``"tot"`` is total matter
#: (CDM + baryons + massive neutrinos) and ``"cb"`` is CDM + baryons only.
MATTER_SPECIES = ("tot", "cb")


def inherit_docstrings(cls):
    """Make docstrings inheritable for a class."""
    for name, func in getmembers(cls, ismethod):
        if func.__doc__:
            continue
        if name.startswith("__"):
            continue
        for parent in cls.__mro__[1:]:
            if hasattr(parent, name):
                func.__func__.__doc__ = getattr(parent, name).__doc__
    return cls


def resolve_matter_species(species: str | None, cosmo: FLRW, model: str) -> str:
    """Validate a ``matter_species`` parameter, resolving the default ``None`` to ``"cb"``.

    When ``species`` is ``None`` and the cosmology has massive neutrinos, so that the
    choice matters, a warning explains the (changed) default and how to silence it.

    Parameters
    ----------
    species
        The ``matter_species`` given by the user: ``"tot"``, ``"cb"`` or ``None``.
    cosmo
        The cosmology of the model.
    model
        Name of the model, used in the warning.

    Returns
    -------
    str
        Either ``"tot"`` or ``"cb"``.

    Raises
    ------
    ValueError
        If ``species`` is not ``None``, ``"tot"`` or ``"cb"``.
    """
    if species is None:
        if cosmo.has_massive_nu:
            warnings.warn(
                f"matter_species was not set for {model}, so it defaults to 'cb' "
                "(CDM + baryons). Earlier versions of hmf used 'tot' (total matter, "
                "including massive neutrinos). 'cb' is the field that halo mass function "
                "and bias fits are universal in (Costanzi+2013, Castorina+2014), and it "
                "matches hmf's mean density, which excludes neutrinos. sigma_8 still "
                "normalises the total matter field (sigma_8_species='tot' by default). "
                "Set matter_species to 'cb' or 'tot' explicitly to silence this warning.",
                stacklevel=3,
            )
        return "cb"

    if species not in MATTER_SPECIES:
        raise ValueError(f"matter_species must be one of {list(MATTER_SPECIES)}, got {species!r}.")
    return species


def set_camb_cosmology(params, cosmo: FLRW) -> None:
    """Set the background cosmology of a ``camb.CAMBparams`` from an astropy cosmology.

    The neutrino masses are passed to CAMB species by species, not just as their
    sum: each distinct non-zero mass in ``cosmo.m_nu`` becomes a CAMB mass eigenstate,
    with as many (degenerate) species as share that mass, and the zero-mass entries
    are massless. So e.g. ``m_nu=[0.1, 0.1, 0.1]`` eV is three massive species of
    0.1 eV, not one of 0.3 eV (CAMB's default when only the sum is given), which
    matters for the free-streaming scale even though the neutrino density is the same.

    Parameters
    ----------
    params : camb.CAMBparams
        The CAMB parameters to update in place.
    cosmo
        The cosmology. Its baryon density must be set.
    """
    m_nu = np.zeros(0) if cosmo.m_nu is None else np.atleast_1d(cosmo.m_nu.value)
    masses, counts = np.unique(m_nu[m_nu > 0], return_counts=True)
    num_massive = int(counts.sum())

    params.set_cosmology(
        H0=cosmo.H0.value,
        ombh2=cosmo.Ob0 * cosmo.h**2,
        omch2=(cosmo.Om0 - cosmo.Ob0) * cosmo.h**2,
        mnu=float(m_nu.sum()),
        neutrino_hierarchy="degenerate",
        num_massive_neutrinos=num_massive,
        omk=cosmo.Ok0,
        nnu=cosmo.Neff,
        standard_neutrino_neff=cosmo.Neff,
        TCMB=cosmo.Tcmb0.value,
    )

    if len(masses) > 1:
        # CAMB has put all massive species in one eigenstate. Split them into one
        # eigenstate per distinct mass, keeping the same degeneracy per species (and
        # so the same Neff) and giving each eigenstate its share of the mass sum.
        degeneracy = params.nu_mass_degeneracies[0] / num_massive
        params.nu_mass_eigenstates = len(masses)
        params.nu_mass_numbers = [int(c) for c in counts]
        params.nu_mass_degeneracies = [c * degeneracy for c in counts]
        params.nu_mass_fractions = [c * m / m_nu.sum() for c, m in zip(counts, masses, strict=True)]
