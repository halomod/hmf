"""The matter species of the transfer and growth models.

Every transfer and growth model gives each matter species: ``"cb"`` (CDM + baryons)
and ``"tot"`` (total matter, including massive neutrinos).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal, get_args

__all__ = ["CAMB_COLUMNS", "MATTER_SPECIES", "Species", "check_species"]

#: A matter species: ``"cb"`` (CDM + baryons) or ``"tot"`` (total matter).
Species = Literal["cb", "tot"]

#: The matter species every model provides, in order.
MATTER_SPECIES: tuple[str, ...] = get_args(Species)

#: The zero-based column of each species in CAMB's transfer-function output: the
#: columns of a CAMB transfer file, and the rows of ``MatterTransferData.transfer_data``
#: (CAMB's 1-based ``Transfer_tot`` and ``Transfer_nonu``, minus 1). Column 0 is k.
CAMB_COLUMNS: Mapping[str, int] = MappingProxyType({"tot": 6, "cb": 7})


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
