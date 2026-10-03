import warnings
from inspect import getmembers, ismethod

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
