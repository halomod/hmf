"""The opt-in disk cache of Boltzmann-code output.

Running CAMB or CLASS is the slowest step of most hmf calculations. With a
:class:`DiskCache`, the output of each run is saved to disk, under a key that is a
content hash of everything the run depends on, and reused by later runs with the same
input, in this process or any other::

    from hmf.core.cache import DiskCache
    from hmf.core.transfer import Transfer

    transfer = Transfer(disk_cache=DiskCache())  # or disk_cache=True

The key is the SHA-256 hash of the canonical serialisation of

* the exact input of the Boltzmann code (built from the transfer or growth model,
  the cosmology and the accuracy settings: the run is a function of this input only),
* the name and version of the Boltzmann code (``camb`` or ``classy``), and
* the version of hmf,

so a new version of either invalidates the cache. Entries are written atomically (to
a temporary file in the same directory, which is then renamed), so concurrent
processes can share a cache directory: each sees either a complete entry or none. An
entry that can't be read (e.g. truncated by a full disk) is treated as missing, and
overwritten.

The cache is never cleaned up automatically; delete the directory to clear it.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import attrs
import numpy as np
import numpy.typing as npt

from ._fields import add_parameters_section, field

__all__ = ["CACHE_DIR_ENV", "DiskCache", "default_cache_dir"]

#: The environment variable that overrides the default cache directory.
CACHE_DIR_ENV = "HMF_CACHE_DIR"


def default_cache_dir() -> Path:
    """The default cache directory.

    It is ``$HMF_CACHE_DIR`` if that is set, else ``$XDG_CACHE_HOME/hmf`` if that is
    set, else ``~/.cache/hmf``.

    Returns
    -------
    pathlib.Path
    """
    if os.environ.get(CACHE_DIR_ENV):
        return Path(os.environ[CACHE_DIR_ENV]).expanduser()
    if os.environ.get("XDG_CACHE_HOME"):
        return Path(os.environ["XDG_CACHE_HOME"]).expanduser() / "hmf"
    return Path.home() / ".cache" / "hmf"


@attrs.frozen(kw_only=True)
class DiskCache:
    """Where, and whether, to keep Boltzmann-code output on disk.

    Pass one as the ``disk_cache`` of a :class:`~hmf.core.transfer.Transfer` or
    :class:`~hmf.core.growth.Growth` stage. See the module documentation for the
    key and the guarantees.
    """

    directory: Path = field(
        factory=default_cache_dir,
        converter=lambda p: Path(p).expanduser(),
        doc=(
            "The cache directory. Defaults to $HMF_CACHE_DIR, $XDG_CACHE_HOME/hmf or "
            "~/.cache/hmf. It is created when the first entry is written."
        ),
    )

    def path(self, key: str) -> Path:
        """The file holding the entry with content hash ``key``."""
        return self.directory / "boltzmann" / f"{key}.npz"

    def load(self, key: str) -> dict[str, npt.NDArray[Any]] | None:
        """Load the arrays stored under ``key``.

        Returns
        -------
        dict or None
            The arrays by name, or ``None`` if there is no (readable) entry.
        """
        path = self.path(key)
        try:
            with np.load(path, allow_pickle=False) as data:
                return {name: data[name] for name in data.files}
        except (OSError, ValueError, EOFError, KeyError):
            return None

    def store(self, key: str, arrays: Mapping[str, npt.ArrayLike]) -> Path:
        """Store ``arrays`` under ``key``, atomically.

        Parameters
        ----------
        key
            The content hash.
        arrays
            Arrays by name.

        Returns
        -------
        pathlib.Path
            The file written.
        """
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{key}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                savez: Any = np.savez  # numpy's stubs reject **arrays (allow_pickle)
                savez(f, **{name: np.asarray(a) for name, a in arrays.items()})
            Path(tmp).replace(path)
        except BaseException:
            with contextlib.suppress(OSError):
                Path(tmp).unlink()
            raise
        return path


add_parameters_section(DiskCache)


def to_disk_cache(value: DiskCache | bool | str | os.PathLike[str] | None) -> DiskCache | None:
    """Convert the ``disk_cache`` field of a stage.

    ``None`` or ``False`` disables the cache, ``True`` uses the default directory, and
    a path uses that directory.
    """
    if value is None or value is False:
        return None
    if value is True:
        return DiskCache()
    if isinstance(value, DiskCache):
        return value
    return DiskCache(directory=Path(value))
