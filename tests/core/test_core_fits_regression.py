"""Regression of hmf.core.fits against the hmf 3.x fitting functions.

This is a cross-check of the port, not a physical test (see ``test_core_fits.py``):
agreement only shows that v4 computes what v3 computed. The v3 code in this
repository's ``hmf.mass_function.fitting_functions`` is that of the v3.7.2 release
(``git diff v3.7.2 -- src/hmf/mass_function/fitting_functions.py`` changes only a
comment).

Intentional differences from 3.7.2:

* ``Manera``: p = 0.248 (Manera et al. 2010, Table 2, linking length 0.2), not
  0.289 (the l_link = 0.15 row). With p = 0.289 the two agree.
* ``Bocquet200c*`` and ``Bocquet500c*``: v3 multiplied f(sigma) by the mass ratio
  M_Delta / M200m; v4's ``fsigma`` is the eq. 3 form only, and the ratio is
  ``mass_ratio_to_200m``. Their product agrees with v3.
* ``Tinker10`` at a non-integer overdensity: v3 truncated Delta with ``int()`` when
  deciding whether it is tabulated (so Delta = 200.5 used the Delta = 200 row and
  alpha); v4 interpolates. Only integer Delta are compared here.
* ``Tinker10``'s ``terminate=False`` (clamping unphysical parameters to arbitrary
  values) is not ported: v4 always raises.
"""

import warnings

import attrs
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

from hmf.core import fits
from hmf.core.fits import FittingFunction
from hmf.core.units import Msun_h, dndm_unit, number_density_unit
from hmf.exceptions import HMFExtrapolationWarning
from hmf.halos import mass_definitions as md
from hmf.mass_function import fitting_functions as ff

DELTA_C = 1.68647
COSMO = FlatLambdaCDM(H0=67.7, Om0=0.31)
SIGMA = np.geomspace(0.25, 4.0, 25)
M = np.geomspace(1e9, 1e16, 25)  # Msun/h; only Bocquet's mass conversion uses it
N_EFF = np.linspace(-2.6, -1.0, 25)

NAMES = sorted(FittingFunction.get_aliases())
SO_FITS = {"Tinker08", "Tinker10", "Watson", "Behroozi"}


def _redshifts(name):
    return (6.0, 9.5, 19.0) if name == "Yung24" else (0.0, 0.3, 1.0, 2.5, 7.0)


def _deltas(name):
    return (200.0, 340.0, 1000.0, 3200.0) if name in SO_FITS else (200.0,)


def _v3(name, z, delta, **params):
    kwargs = {
        "nu2": (DELTA_C / SIGMA) ** 2,
        "m": M,
        "z": z,
        "n_eff": N_EFF,
        "cosmo": COSMO,
        "delta_c": DELTA_C,
    }
    if name in SO_FITS:
        kwargs["mass_definition"] = md.SOMean(overdensity=delta)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", HMFExtrapolationWarning)
        fit = getattr(ff, name)(**kwargs, **params)
    f = fit.fsigma
    if hasattr(fit, "convert_mass"):
        f = f / fit.convert_mass()
    return f


def _v4(model, z, delta):
    return model.fsigma(
        SIGMA,
        z=z,
        omega_m_z=COSMO.Om(z),
        delta_halo=delta,
        delta_c=DELTA_C,
        n_eff=N_EFF,
        m=M * Msun_h,
    )


@pytest.mark.parametrize("name", NAMES)
def test_fsigma_matches_v3(name):
    model = FittingFunction.get(name)()
    if name == "Manera":
        model = attrs.evolve(model, p=0.289)
    for z in _redshifts(name):
        for delta in _deltas(name):
            # Tolerance: the same formulas, up to floating-point rounding.
            np.testing.assert_allclose(
                _v4(model, z, delta), _v3(name, z, delta), rtol=1e-12, err_msg=f"{z=}, {delta=}"
            )


def test_manera_default_differs_from_v3():
    """v4 fixes Manera's p (see the module docstring)."""
    assert fits.Manera().p == 0.248
    assert ff.Manera._defaults["p"] == 0.289
    assert not np.allclose(_v4(fits.Manera(), 0.0, 200.0), _v3("Manera", 0.0, 200.0))


@pytest.mark.parametrize(
    "name", ["Bocquet200cDMOnly", "Bocquet200cHydro", "Bocquet500cDMOnly", "Bocquet500cHydro"]
)
@pytest.mark.parametrize("z", [0.0, 1.0, 2.0])
def test_bocquet_mass_conversion_matches_v3(name, z):
    """v3's f(sigma) is v4's f(sigma) times mass_ratio_to_200m."""
    model = FittingFunction.get(name)()
    ratio = model.mass_ratio_to_200m(M * Msun_h, z=z, omega_m0=COSMO.Om0, h=COSMO.h)
    fit = getattr(ff, name)(nu2=(DELTA_C / SIGMA) ** 2, m=M, z=z, cosmo=COSMO, delta_c=DELTA_C)
    np.testing.assert_allclose(_v4(model, z, 200.0) * ratio, fit.fsigma, rtol=1e-12)


def test_behroozi_modify_dndm_matches_v3():
    m = np.geomspace(1e10, 1e15, 20)
    dndm = m**-1.9
    ngtm = m**-0.9 / 0.9
    for z in (0.0, 4.0, 8.0):
        v3 = ff.Behroozi(nu2=np.ones(20), z=z, mass_definition=md.SOVirial())._modify_dndm(
            m / 0.7, dndm, z, ngtm, h=0.7
        )
        v4 = fits.Behroozi().modify_dndm(
            m * Msun_h, dndm * dndm_unit, z=z, ngtm=ngtm * number_density_unit, h=0.7
        )
        np.testing.assert_allclose(v4.to_value(dndm_unit), v3, rtol=1e-12)


@pytest.mark.parametrize("name", NAMES)
def test_normalized_matches_v3(name):
    """The `normalized` flag of the default instances agrees with v3's class flag."""
    assert FittingFunction.get(name)().normalized == getattr(ff, name).normalized
