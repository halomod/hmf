"""Physical tests of the baryon corrections to the BBKS transfer function."""

import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

from hmf.density_field.transfer_models import BBKS

LNK = np.linspace(np.log(1e-4), np.log(10.0), 200)

NO_BARYONS = {"use_sugiyama_baryons": False, "use_liddle_baryons": False}
SUGIYAMA = {"use_sugiyama_baryons": True, "use_liddle_baryons": False}
LIDDLE = {"use_sugiyama_baryons": False, "use_liddle_baryons": True}


@pytest.mark.parametrize("params", [SUGIYAMA, LIDDLE], ids=["sugiyama", "liddle"])
def test_baryon_correction_vanishes_without_baryons(params):
    """With Omega_b -> 0, both corrections reduce to the bare Gamma = Omega_m h."""
    cosmo = FlatLambdaCDM(H0=67.0, Om0=0.3, Ob0=1e-10)

    np.testing.assert_allclose(
        BBKS(cosmo, **params).lnt(LNK), BBKS(cosmo, **NO_BARYONS).lnt(LNK), rtol=0, atol=1e-8
    )


def test_liddle_equals_sugiyama_at_h_half():
    """At h = 0.5 the published and preprint Sugiyama corrections coincide.

    Since sqrt(2h) = 1, the published Sugiyama (1995) form,
    Gamma = Omega_m h exp[-Omega_b (1 + sqrt(2h)/Omega_m)] (as quoted by Meiksin,
    White & Peacock 1999, eq. 4), equals the preprint form
    Gamma = Omega_m h exp[-Omega_b (1 + 1/Omega_m)] exactly.
    """
    cosmo = FlatLambdaCDM(H0=50.0, Om0=0.3, Ob0=0.05)

    np.testing.assert_allclose(
        BBKS(cosmo, **LIDDLE).lnt(LNK), BBKS(cosmo, **SUGIYAMA).lnt(LNK), rtol=1e-12
    )


@pytest.mark.parametrize("params", [NO_BARYONS, SUGIYAMA, LIDDLE])
def test_transfer_unity_at_low_k(params):
    """T(k) -> 1 on scales much larger than the horizon at equality."""
    cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315, Ob0=0.0493)

    t = np.exp(BBKS(cosmo, **params).lnt(np.log(np.array([1e-6, 1e-5]))))

    np.testing.assert_allclose(t, 1.0, atol=1e-3)
