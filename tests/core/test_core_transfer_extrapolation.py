"""Physical tests of the extrapolation of tabulated transfer functions beyond k_max.

Above the Boltzmann code's last wavenumber k_j, T(k) follows the EH98 no-wiggle
shape, matched to the table in value *and* logarithmic slope (hmf v3 matched the
value only, which left a kink in the slope). These tests check:

* d ln T / d ln k is continuous at the join (and the v3-style join is not, so the
  test can tell);
* sigma(M) with a top-hat filter, integrated here (not with hmf's own code), has no
  feature at the mass of the join: it agrees with a run that needs no extrapolation
  there;
* the extrapolation follows a CAMB run to higher k_max.
"""

import numpy as np
import pytest
from astropy.cosmology import Planck18
from scipy.integrate import simpson

from hmf.core._kernels import transfer as kt
from hmf.core.transfer import Transfer
from hmf.core.transfer_models import CAMB, _eh98_scales
from hmf.core.units import h_Mpc


@pytest.fixture(scope="module")
def default():
    return Transfer(model=CAMB())


@pytest.fixture(scope="module")
def deep():
    """A run to k_max = 100 h/Mpc: no extrapolation below 100 h/Mpc."""
    return Transfer(model=CAMB(k_max=100 * h_Mpc))


def amplitude_only_join(transfer):
    """Ln T(k) with the v3 extrapolation: EH98 rescaled in amplitude only."""
    sol = transfer.solution
    k_j = sol.k_max_table
    scales = _eh98_scales(transfer.cosmology)

    def ln_t(k):
        out = sol.ln_transfer(k).copy()
        above = k > k_j
        offset = (
            sol.ln_transfer(np.array([k_j]))[0] - kt.ln_t_eh98_no_wiggle(np.array([k_j]), scales)[0]
        )
        out[above] = kt.ln_t_eh98_no_wiggle(k[above], scales) + offset
        return out

    return ln_t


def slope_jump(ln_t, k_j, eps):
    """The difference of the one-sided slopes of ln T at k_j."""
    x = np.log(k_j)
    f = lambda xx: ln_t(np.exp(np.array([xx])))[0]  # noqa: E731
    below = (f(x) - f(x - eps)) / eps
    above = (f(x + eps) - f(x)) / eps
    return above - below, below


@pytest.mark.parametrize("species", ["cb", "tot"])
def test_log_slope_is_continuous_at_the_join(default, species):
    sol = default.solution
    k_j = sol.k_max_table
    for eps in (1e-3, 1e-4):
        jump, slope = slope_jump(lambda k: sol.ln_transfer(k, species), k_j, eps)
        # Tolerance: with a continuous slope, one-sided differences differ by
        # O(eps * |jump in d2 ln T / d ln k2|); measured 3.6e-5 (eps = 1e-3) and
        # 3.6e-6 (eps = 1e-4), against a slope of -1.82.
        assert abs(jump) < 0.1 * eps * abs(slope)


def test_amplitude_only_join_has_a_kink(default):
    """The v3 join fails the same test: its slope jumps by 3.7e-3 at the join."""
    jump, slope = slope_jump(amplitude_only_join(default), default.solution.k_max_table, 1e-4)
    assert abs(jump) > 100 * 0.1 * 1e-4 * abs(slope)


def test_extrapolation_follows_a_deeper_run(default, deep):
    """Beyond k_max = 20 h/Mpc the extrapolated T(k) tracks a run to 100 h/Mpc."""
    k = np.geomspace(default.solution.k_max_table, 60, 30)
    diff = default.solution.ln_transfer(k) - deep.solution.ln_transfer(k)
    # Tolerance: measured max 1.4e-3 (at 60 h/Mpc; 2e-4 up to 40 h/Mpc). The EH98
    # shape departs from CAMB's slowly above ~50 h/Mpc.
    np.testing.assert_allclose(diff, 0, atol=3e-3)


def _tophat(x):
    xs = np.where(x < 1e-2, 1.0, x)
    return np.where(x < 1e-2, 1 - x**2 / 10, 3 * (np.sin(xs) - xs * np.cos(xs)) / xs**3)


def _dln_sigma_dln_m(ln_t, n_s, ln_r):
    """D ln sigma / d ln M for a top-hat filter, by direct quadrature in ln k."""
    ln_k = np.arange(np.log(1e-5), np.log(1e5), 0.005)
    k = np.exp(ln_k)
    p = k**n_s * np.exp(2 * ln_t(k))
    sigma2 = np.array([simpson(p * k**3 * _tophat(k * r) ** 2, x=ln_k) for r in np.exp(ln_r)])
    return np.gradient(0.5 * np.log(sigma2), 3 * ln_r)


def test_tophat_sigma_has_no_feature_at_the_join_mass(default, deep):
    """Dln sigma/dln M agrees with the deeper run around the mass of the join.

    The join is at k_j = 21 h/Mpc, i.e. R_j = 1/k_j ~ 0.05 Mpc/h (M_j ~ 4e7 Msun/h
    for Planck18). Masses a few times above M_j are where a slope kink at k_j would
    show; there the matched join agrees with the deeper run to ~1e-6, ten to fifty
    times better than the v3 join, whose error (up to 1.2e-4) is visible at the 1e-4
    accuracy target of v4. (Further below M_j, the EH98 shape itself departs from
    CAMB's above ~50 h/Mpc: raise ``k_max`` for those masses.)
    """
    k_j = default.solution.k_max_table
    ln_r = np.arange(np.log(1 / k_j) - 0.5, np.log(1 / k_j) + 2.5, 0.05)
    reference = _dln_sigma_dln_m(deep.solution.ln_transfer, default.n_s, ln_r)
    matched = _dln_sigma_dln_m(default.solution.ln_transfer, default.n_s, ln_r) - reference
    v3 = _dln_sigma_dln_m(amplitude_only_join(default), default.n_s, ln_r) - reference

    near = ln_r >= np.log(1.6 / k_j)
    above = ln_r >= np.log(2.0 / k_j)
    # Tolerances: measured max |matched| 1.1e-5 (R >= 1.6 R_j) and 1.3e-6 (R >= 2 R_j);
    # the v3 join gives 1.2e-4 and 6.9e-5 there.
    assert np.abs(matched[near]).max() < 2e-5
    assert np.abs(matched[above]).max() < 3e-6
    assert np.abs(v3[near]).max() > 5 * np.abs(matched[near]).max()
    assert np.abs(v3[above]).max() > 10 * np.abs(matched[above]).max()


def test_join_with_massive_neutrinos_is_smooth_for_both_species():
    from astropy import units as u
    from astropy.cosmology import FlatLambdaCDM

    cosmo = FlatLambdaCDM(H0=67.7, Om0=0.31, Ob0=0.049, Tcmb0=2.7255, m_nu=[0, 0, 0.3] * u.eV)
    sol = Transfer(cosmology=cosmo, model="CAMB").solution
    for species in ("cb", "tot"):
        jump, slope = slope_jump(lambda k, s=species: sol.ln_transfer(k, s), sol.k_max_table, 1e-4)
        assert abs(jump) < 0.1 * 1e-4 * abs(slope)


def test_planck18_is_the_default_cosmology(default):
    assert default.cosmology is Planck18
