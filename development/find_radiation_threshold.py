"""Determine a safe radiation threshold for the growth-factor selector.

This utility compares `Tinker08` halo mass functions computed with
`Eisenstein97GrowthFactor` and `ODEGrowthFactor` over a redshift and mass grid, then
reports the largest radiation-density threshold for which `dndlnm` stays within a
chosen relative tolerance for all masses below a configurable upper limit.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
from astropy.cosmology import FlatLambdaCDM

from hmf import MassFunction


@dataclass(frozen=True)
class ThresholdConfig:
    """Configuration for the radiation-threshold scan.

    Parameters
    ----------
    z_min
        Minimum redshift included in the scan.
    z_max
        Maximum redshift included in the scan.
    dz
        Redshift step size.
    mass_limit
        Maximum halo mass, in `Msun / h`, included in the error check.
    mmin
        Minimum log10 halo mass passed to `MassFunction`.
    mmax
        Maximum log10 halo mass passed to `MassFunction`.
    dlog10m
        Log10 halo-mass spacing passed to `MassFunction`.
    tolerance
        Maximum allowed relative difference in `dndlnm`.
    hmf_model
        Halo mass function model to evaluate.
    H0
        Hubble constant for the comparison cosmology.
    Om0
        Matter density parameter at `z=0`.
    Ob0
        Baryon density parameter at `z=0`.
    Tcmb0
        CMB temperature for the comparison cosmology.
    sigma_8
        Present-day amplitude of matter fluctuations.
    n
        Primordial scalar spectral index.
    """

    z_min: float = 0.0
    z_max: float = 12.0
    dz: float = 0.1
    mass_limit: float = 1.0e14
    mmin: float = 10.0
    mmax: float = 14.2
    dlog10m: float = 0.02
    tolerance: float = 0.01
    hmf_model: str = "Tinker08"
    H0: float = 67.74
    Om0: float = 0.3089
    Ob0: float = 0.0486
    Tcmb0: float = 2.7255
    sigma_8: float = 0.8159
    n: float = 0.9667

    @property
    def cosmo(self) -> FlatLambdaCDM:
        """Comparison cosmology used for the scan."""
        return FlatLambdaCDM(
            H0=self.H0,
            Om0=self.Om0,
            Ob0=self.Ob0,
            Tcmb0=self.Tcmb0,
        )

    @property
    def redshifts(self) -> np.ndarray:
        """Redshift grid used for the scan."""
        return np.arange(self.z_min, self.z_max + self.dz / 2, self.dz)


def radiation_density(cosmo: FlatLambdaCDM, z: float) -> float:
    """Compute the fractional radiation density at redshift `z`.

    Parameters
    ----------
    cosmo
        Cosmology used for the comparison.
    z
        Redshift at which to evaluate the radiation fraction.

    Returns
    -------
    float
        Radiation density fraction, matching the selector logic used by
        `GrowthFactor.radiation_density`.
    """
    return float((cosmo.Ogamma0 + cosmo.Onu0) * (1 + z) ** 4 * cosmo.inv_efunc(z) ** 2)


def build_mass_function(z: float, growth_model: str, config: ThresholdConfig) -> MassFunction:
    """Construct a comparison `MassFunction` instance.

    Parameters
    ----------
    z
        Redshift of the calculation.
    growth_model
        Growth-factor model name to pass to `MassFunction`.
    config
        Configuration for the threshold scan.

    Returns
    -------
    MassFunction
        Configured mass function instance.
    """
    return MassFunction(
        z=z,
        hmf_model=config.hmf_model,
        growth_model=growth_model,
        transfer_model="EH",
        mdef_model="SOMean",
        mdef_params={"overdensity": 200},
        Mmin=config.mmin,
        Mmax=config.mmax,
        dlog10m=config.dlog10m,
        cosmo_params={
            "H0": config.H0,
            "Om0": config.Om0,
            "Ob0": config.Ob0,
            "Tcmb0": config.Tcmb0,
        },
        sigma_8=config.sigma_8,
        n=config.n,
    )


def max_relative_dndlnm_error(z: float, config: ThresholdConfig) -> float:
    """Return the maximum relative `dndlnm` error at one redshift.

    Parameters
    ----------
    z
        Redshift at which to compare the growth models.
    config
        Configuration for the threshold scan.

    Returns
    -------
    float
        Maximum relative difference between the Eisenstein and ODE `dndlnm`
        predictions for masses below `config.mass_limit`.
    """
    eisenstein = build_mass_function(z, "Eisenstein97GrowthFactor", config)
    ode = build_mass_function(z, "ODEGrowthFactor", config)

    mask = eisenstein.m < config.mass_limit
    rel = np.abs(eisenstein.dndlnm[mask] - ode.dndlnm[mask]) / ode.dndlnm[mask]
    return float(np.max(rel))


def find_safe_threshold(config: ThresholdConfig) -> list[tuple[float, float, float]]:
    """Scan the grid and collect redshift, radiation density, and HMF error.

    Parameters
    ----------
    config
        Configuration for the threshold scan.

    Returns
    -------
    list of tuple
        Tuples of `(z, radiation_density, max_relative_error)`.
    """
    return [
        (
            float(z),
            radiation_density(config.cosmo, float(z)),
            max_relative_dndlnm_error(float(z), config),
        )
        for z in config.redshifts
    ]


def write_line(text: str = "") -> None:
    """Write one line to standard output."""
    sys.stdout.write(f"{text}\n")


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the utility."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z-min", type=float, default=ThresholdConfig.z_min)
    parser.add_argument("--z-max", type=float, default=ThresholdConfig.z_max)
    parser.add_argument("--dz", type=float, default=ThresholdConfig.dz)
    parser.add_argument("--mass-limit", type=float, default=ThresholdConfig.mass_limit)
    parser.add_argument("--mmin", type=float, default=ThresholdConfig.mmin)
    parser.add_argument("--mmax", type=float, default=ThresholdConfig.mmax)
    parser.add_argument("--dlog10m", type=float, default=ThresholdConfig.dlog10m)
    parser.add_argument("--tolerance", type=float, default=ThresholdConfig.tolerance)
    parser.add_argument("--hmf-model", type=str, default=ThresholdConfig.hmf_model)
    return parser


def main() -> int:
    """Run the threshold scan and print a compact report."""
    args = build_parser().parse_args()
    config = ThresholdConfig(
        z_min=args.z_min,
        z_max=args.z_max,
        dz=args.dz,
        mass_limit=args.mass_limit,
        mmin=args.mmin,
        mmax=args.mmax,
        dlog10m=args.dlog10m,
        tolerance=args.tolerance,
        hmf_model=args.hmf_model,
    )

    rows = find_safe_threshold(config)
    safe_rows = [row for row in rows if row[2] <= config.tolerance]
    failing_rows = [row for row in rows if row[2] > config.tolerance]

    write_line("z radiation_density max_rel_dndlnm_error")
    for z, rad, err in rows:
        write_line(f"{z:5.2f} {rad:17.10f} {err:20.10f}")

    if not safe_rows:
        write_line("\nNo safe redshifts found on the requested grid.")
        return 1

    z_safe, threshold, err_safe = safe_rows[-1]
    write_line(
        "\nLargest safe threshold on scanned grid: "
        f"{threshold:.10f} at z={z_safe:.2f} with max_rel_error={err_safe:.6f}"
    )

    if failing_rows:
        z_fail, rad_fail, err_fail = failing_rows[0]
        write_line(
            "First failing point: "
            f"z={z_fail:.2f}, radiation_density={rad_fail:.10f}, max_rel_error={err_fail:.6f}"
        )
    else:
        write_line(
            "No failing points found on the scanned grid; increase --z-max to probe further."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
