import inspect
import itertools

import numpy as np
import pytest
from colossus.cosmology.cosmology import setCosmology
from colossus.lss.mass_function import massFunction

from hmf import MassFunction
from hmf.mass_function import fitting_functions as ff

allfits = [
    o
    for n, o in inspect.getmembers(
        ff,
        lambda member: (
            inspect.isclass(member)
            and issubclass(member, ff.BaseFittingFunction)
            and member is not ff.BaseFittingFunction
            and member is not ff.PS
            and member is not ff.Yung24  # only valid at z=6-19; tested separately
        ),
    )
]


def _conversion_is_active(hmf: MassFunction) -> bool:
    """Whether `MassFunction.dndm` will apply mass-definition conversion."""
    return (
        hmf.hmf.measured_mass_definition is not None
        and hmf.hmf.measured_mass_definition != hmf.mdef
        and not hmf.disable_mass_conversion
    )


@pytest.fixture(scope="module")
def hmf():
    return MassFunction(
        Mmin=10,
        Mmax=15,
        dlog10m=0.1,
        lnk_min=-16,
        lnk_max=10,
        dlnk=0.01,
        hmf_model="PS",
        z=0.0,
        sigma_8=0.8,
        n=1,
        cosmo_params={"Om0": 0.3, "H0": 70.0, "Ob0": 0.05},
        transfer_model="EH",
    )


@pytest.fixture(scope="module")
def ps_max(hmf):
    hmf.update(hmf_model="PS")
    return hmf.fsigma.max()


@pytest.mark.parametrize(("redshift", "fit"), itertools.product([0.0, 2.0], allfits))
def test_allfits(hmf, ps_max, redshift, fit):
    """
    Test all implemented fits for correct form and behavior.

    Tests that:
    1) the maximum fsigma is less than in the PS formula (which is known to overestimate)
    2) the slope is positive below this maximum
    3) the slope is negative above this maximum

    Since it calls each class, any blatant errors should also pop up.

    """
    hmf.update(z=redshift, hmf_model=fit)
    maxarg = np.argmax(hmf.fsigma)
    assert ps_max >= hmf.fsigma[maxarg]
    assert np.all(np.diff(hmf.fsigma[:maxarg]) >= 0)
    assert np.all(np.diff(hmf.fsigma[maxarg:]) <= 0)


def test_tinker08_dh():
    h = MassFunction(
        hmf_model="Tinker08",
        mdef_model="SOMean",
        mdef_params={"overdensity": 200},
        transfer_model="EH",
    )
    h1 = MassFunction(
        hmf_model="Tinker08",
        mdef_model="SOMean",
        mdef_params={"overdensity": 200.1},
        transfer_model="EH",
    )

    assert np.allclose(h.fsigma, h1.fsigma, rtol=1e-2)


def test_tinker10_dh():
    h = MassFunction(hmf_model="Tinker10", transfer_model="EH")
    h1 = MassFunction(
        hmf_model="Tinker10",
        mdef_model="SOMean",
        mdef_params={"overdensity": 200.1},
        transfer_model="EH",
    )

    assert np.allclose(h.fsigma, h1.fsigma, rtol=1e-2)


def test_tinker10_neg_gam():
    with pytest.raises(ValueError):
        h = MassFunction(hmf_model="Tinker10", hmf_params={"gamma_200": -1}, transfer_model="EH")
        h.fsigma


def test_tinker10_neg_eta():
    with pytest.raises(ValueError):
        h = MassFunction(hmf_model="Tinker10", hmf_params={"eta_200": -1}, transfer_model="EH")
        h.fsigma


def test_tinker10_neg_etaphi():
    with pytest.raises(ValueError):
        h = MassFunction(
            hmf_model="Tinker10",
            hmf_params={"eta_200": -1, "phi_200": 0},
            transfer_model="EH",
        )
        h.fsigma


def test_tinker10_neg_beta():
    with pytest.raises(ValueError):
        h = MassFunction(hmf_model="Tinker10", hmf_params={"beta_200": -1}, transfer_model="EH")
        h.fsigma


@pytest.mark.filterwarnings("ignore:.*does not match the mass definition.*:UserWarning")
@pytest.mark.filterwarnings("ignore:.*Halo-Exclusion models.*:UserWarning")
@pytest.mark.parametrize("fit", ["Behroozi", "SMT"])
@pytest.mark.parametrize("z", [0.0, 1.0, 2.0])
def test_hmf_mass_definition_consistency(z, fit):
    """The dndm must be consistent between equivalent SO mass definitions.

    SOCritical(200) and SOMean(200/Omega_m(z)) define the same physical density
    threshold, so any HMF with mass-definition conversion must give the same dndm for
    both. This was broken before because the mass-definition conversion in dndm always
    used z=0 and the default Planck15 cosmology instead of the actual redshift and
    cosmology of the computation.
    """
    from astropy.cosmology import FlatLambdaCDM

    # Use non-Planck15 cosmology to verify the cosmo parameter is correctly
    # propagated in the mass-definition conversion (bug was: Planck15 used everywhere).
    cosmo_params = {"Om0": 0.3, "H0": 70.0, "Ob0": 0.05}
    cosmo = FlatLambdaCDM(**cosmo_params)

    # At redshift z, SOCritical(200) has the same density threshold as SOMean(equiv)
    equiv_overdensity = 200.0 / cosmo.Om(z)

    common = {
        "hmf_model": fit,
        "transfer_model": "EH",
        "Mmin": 10,
        "Mmax": 15,
        "dlog10m": 0.1,
        "z": z,
        "disable_mass_conversion": False,
        "cosmo_params": cosmo_params,
    }

    h_crit = MassFunction(
        mdef_model="SOCritical",
        mdef_params={"overdensity": 200},
        **common,
    )
    h_mean = MassFunction(
        mdef_model="SOMean",
        mdef_params={"overdensity": equiv_overdensity},
        **common,
    )

    assert _conversion_is_active(h_crit)
    assert _conversion_is_active(h_mean)

    # Allow ~2% tolerance: a small residual comes from the floating-point
    # imprecision of equiv_overdensity = 200/Om(z) and the NFW c-M approximation.
    # The original bug produced 20-50% errors, so this tolerance is well above that.
    np.testing.assert_allclose(h_crit.dndm, h_mean.dndm, rtol=2e-2)


def test_yung24_units_switch():
    common = {
        "mdef_model": "SOVirial",
        "z": 10.0,
        "transfer_model": "EH",
        "Mmin": 6,
        "Mmax": 13,
        "dlog10m": 0.1,
    }
    h_h = MassFunction(hmf_model="Yung24", hmf_params={"units": "h"}, **common)
    h_phys = MassFunction(hmf_model="Yung24", hmf_params={"units": "physical"}, **common)
    assert not np.allclose(h_h.fsigma, h_phys.fsigma)


def test_yung24_invalid_units_raises():
    with pytest.raises(ValueError, match="units must be 'h' or 'physical'"):
        MassFunction(
            hmf_model="Yung24",
            hmf_params={"units": "bogus"},
            mdef_model="SOVirial",
            z=10.0,
            transfer_model="EH",
        ).fsigma


@pytest.mark.parametrize("z", [5.999, 19.001])
def test_yung24_invalid_z_raises(z):
    with pytest.raises(ValueError, match=r"Yung24 fit is only valid for z in \[6.0, 19.0\]"):
        MassFunction(
            hmf_model="Yung24",
            mdef_model="SOVirial",
            z=z,
            transfer_model="EH",
        )


def test_yung24_cutmask():
    m = np.array([1e5, 1e6, 1e7, 1e12, 1e13, 1e14])
    nu2 = np.ones_like(m)

    fit_h = ff.Yung24(nu2=nu2, m=m, z=10.0, units="h")
    np.testing.assert_array_equal(fit_h.cutmask, (np.log10(m) >= 6.0) & (np.log10(m) <= 13.0))

    fit_phys = ff.Yung24(nu2=nu2, m=m, z=10.0, units="physical")
    np.testing.assert_array_equal(fit_phys.cutmask, (np.log10(m) >= 5.0) & (np.log10(m) <= 13.0))

    fit_no_m = ff.Yung24(nu2=nu2, z=10.0)
    np.testing.assert_array_equal(fit_no_m.cutmask, np.ones(len(nu2), dtype=bool))


def _yung24_sigma0_phys(mvir):
    """sigma(Mvir) at z=0, physical units (no h), from Yung+24 Eq. A4.

    This is a *different* equation to the one implemented in ``Yung24.fsigma``
    (which uses hmf's own transfer-function-based sigma(M)); it lets us derive an
    independent sigma(M, z) to convert the digitized Fig. A1 points below into
    fsigma, without going anywhere near the code under test.
    """
    m12 = mvir / 1e12
    y = 1.0 / m12
    return (
        26.80004233 * y**0.40695158 / (1 + 6.18130098 * y**0.23076433 + 4.64104008 * y**0.36760939)
    )


def _yung24_dlnsigma_dlnm(mvir, eps=1e-4):
    lnm = np.log(mvir)
    s_plus = _yung24_sigma0_phys(np.exp(lnm + eps))
    s_minus = _yung24_sigma0_phys(np.exp(lnm - eps))
    return (np.log(s_plus) - np.log(s_minus)) / (2 * eps)


def _yung24_growth_factor(z, om0=0.307, h=0.678):
    """Linear growth factor D(z), normalized to D(0)=1, for Yung+24's flat LCDM cosmology.

    Reuses hmf's own (independently-tested) growth factor machinery rather than
    re-deriving it -- growth factor correctness isn't what this test is about.
    """
    from astropy.cosmology import FlatLambdaCDM

    from hmf.cosmology.growth_factor import GrowthFactor

    cosmo = FlatLambdaCDM(H0=100 * h, Om0=om0)
    return float(GrowthFactor(cosmo).growth_factor(z)[0])


def _yung24_fsigma_sensitivity(sigma, z, frac=1e-3):
    """Local d ln(fsigma) / d ln(sigma), via finite difference on the code's own fsigma.

    Used to scale the test tolerance below: f(sigma) ~ exp(-c/sigma^2) is extremely
    sensitive to sigma when sigma is small (high z / high mass), so a fixed sigma
    error propagates to a much larger fsigma error there than at low z / low mass.
    """
    delta_c = 1.68647

    def fsigma_at(s):
        nu2 = np.array([(delta_c / s) ** 2])
        return float(ff.Yung24(nu2=nu2, z=z, units="physical").fsigma[0])

    f_plus = fsigma_at(sigma * (1 + frac))
    f_minus = fsigma_at(sigma * (1 - frac))
    return (np.log(f_plus) - np.log(f_minus)) / (2 * np.log(1 + frac))


# (z, log10(Mvir/Msun), dn/dlogM [Mpc^-3 dex^-1]) digitized from the *fitted* curve
# (not the noisier n-body points) in the physical-units panel of Fig. A1 of Yung+24
# (arXiv:2309.14408). Mass points were chosen to stay within each redshift's plotted
# range: the fit only extends to ~11 dex by z~12 and ~12 dex by z~8.
_YUNG24_FIG_A1_FITTED_POINTS = [
    (6.0, 6.0, 1.7942e3),
    (6.0, 9.0, 1.6546e0),
    (6.0, 11.0, 3.4614e-3),
    (6.0, 12.0, 2.4035e-5),
    (7.0, 9.0, 1.1748e0),
    (7.0, 11.0, 1.1918e-3),
    (7.0, 12.0, 3.1176e-6),
    (8.0, 9.0, 7.8391e-1),
    (8.0, 11.0, 3.4873e-4),
    (8.0, 12.0, 3.0396e-7),
    (9.0, 9.0, 4.8877e-1),
    (9.0, 11.0, 8.7234e-5),
    (10.0, 6.0, 1.2678e3),
    (10.0, 9.0, 2.8509e-1),
    (10.0, 11.0, 1.8723e-5),
    (11.0, 9.0, 1.5496e-1),
    (11.0, 11.0, 3.4676e-6),
    (12.0, 9.0, 7.8735e-2),
    (13.0, 9.0, 3.7477e-2),
    (14.0, 9.0, 1.6795e-2),
    (15.0, 6.0, 3.9080e2),
    (15.0, 9.0, 7.1454e-3),
    (16.0, 9.0, 2.9023e-3),
    (17.0, 9.0, 1.1316e-3),
    (18.0, 9.0, 4.2158e-4),
    (19.0, 6.0, 7.2514e1),
    (19.0, 9.0, 1.3263e-4),
]


@pytest.mark.parametrize(("z", "log10_mvir", "dn_dlogm"), _YUNG24_FIG_A1_FITTED_POINTS)
def test_yung24_fsigma_matches_paper_fit(z, log10_mvir, dn_dlogm):
    """Check fsigma's functional form against the fitted curve in Fig. A1 of Yung+24.

    This tests specifically whether ``Yung24.fsigma`` (the functional form A2, with its
    A/a/b/c(z) coefficients) matches the paper -- it does *not* test whether that
    fsigma gets propagated into dn/dm the same way as the paper (mass definitions,
    rho_m conventions etc. are not exercised here; see
    ``test_yung24_dndlogm_matches_paper_fit`` for an end-to-end check of that).

    To isolate fsigma's functional form from any difference in *how sigma is
    computed*, sigma(M, z) is derived independently of the code: from Yung+24's own
    sigma(Mvir) fit (Eq. A4, transcribed from the paper) and hmf's own growth factor
    (not the code under test). The digitized ``dn_dlogm`` values (read off the fitted
    curve, not the noisier n-body points, per Eq. A1) are converted to an expected
    fsigma using this same independent sigma and its analytic log-derivative.

    fsigma is proportional to exp(-c/sigma^2), so it is extremely sensitive to sigma
    when sigma is small (high z / high mass) -- and Eq. A4 is explicitly described in
    the paper as an "updated approximation", not exact. A cross-check against hmf's own
    (unrelated) transfer-function-based sigma confirms Eq. A4 is accurate to ~5-8% in
    sigma itself, but that gets amplified by the exponential sensitivity. So rather than
    a single fixed tolerance, we scale the allowed deviation by the local sensitivity
    d ln(fsigma)/d ln(sigma), computed from the code's own fsigma.
    """
    mvir = 10**log10_mvir
    sigma = _yung24_sigma0_phys(mvir) * _yung24_growth_factor(z)
    dlnsigma_dlnm = _yung24_dlnsigma_dlnm(mvir)

    h = 0.678
    rho_m0 = 0.307 * 2.7754e11 * h**2  # Msun/Mpc^3
    fsigma_from_fit = dn_dlogm * mvir / (rho_m0 * np.log(10) * abs(dlnsigma_dlnm))

    delta_c = 1.68647
    nu2 = np.array([(delta_c / sigma) ** 2])
    fsigma_code = ff.Yung24(nu2=nu2, z=z, units="physical").fsigma[0]

    sensitivity = _yung24_fsigma_sensitivity(sigma, z)
    base_rtol, sigma_fit_rtol = 0.06, 0.08
    rtol = base_rtol + abs(sensitivity) * sigma_fit_rtol

    np.testing.assert_allclose(fsigma_code, fsigma_from_fit, rtol=rtol)


def _yung24_dndlogm_phys(z, log10_mphys, sigma_8=0.829, dlog10m=0.02):
    """dn/dlogM [Mpc^-3 dex^-1] at a physical mass, via the full ``MassFunction`` pipeline.

    ``MassFunction`` always works in Msun/h and (Mpc/h)^-3 internally (regardless of
    Yung24's ``units`` setting, which only selects which fit-coefficient table is
    used) -- so to get the prediction at a *physical* mass we query it at the
    equivalent Msun/h mass (``m_phys * h``), then convert the (h-based comoving)
    output back to physical units: dn/dlogM_phys = dn/dlogM_h * h^3, since a
    (Mpc/h)^-3 comoving volume is h^3 times smaller than a physical Mpc^-3 one.
    """
    h = 0.678
    mmin_h = log10_mphys + np.log10(h) - 0.5
    mmax_h = log10_mphys + np.log10(h) + 0.5
    hmf_ = MassFunction(
        Mmin=mmin_h,
        Mmax=mmax_h,
        dlog10m=dlog10m,
        hmf_model="Yung24",
        hmf_params={"units": "physical"},
        mdef_model="SOVirial",
        z=z,
        transfer_model="EH",
        cosmo_params={"Om0": 0.307, "H0": 67.8, "Ob0": 0.0486},
        sigma_8=sigma_8,
        n=0.960,
    )
    m_phys = hmf_.m / h
    dndlog10m_phys = hmf_.dndlnm * np.log(10) * h**3
    idx = np.argmin(np.abs(np.log10(m_phys) - log10_mphys))
    return dndlog10m_phys[idx]


def _yung24_dndlogm_sensitivity(z, log10_mphys, frac=1e-3):
    """Local d ln(dn/dlogM) / d ln(sigma_8), via finite difference on the full pipeline.

    sigma_8 rescales sigma(M) at all M by (approximately) the same fraction, so this
    is a convenient proxy for the same exp(-c/sigma^2) sensitivity that motivates the
    scaled tolerance in ``test_yung24_fsigma_matches_paper_fit``.
    """
    f_plus = _yung24_dndlogm_phys(z, log10_mphys, sigma_8=0.829 * (1 + frac))
    f_minus = _yung24_dndlogm_phys(z, log10_mphys, sigma_8=0.829 * (1 - frac))
    return (np.log(f_plus) - np.log(f_minus)) / (2 * np.log(1 + frac))


@pytest.mark.parametrize(("z", "log10_mvir", "dn_dlogm"), _YUNG24_FIG_A1_FITTED_POINTS)
def test_yung24_dndlogm_matches_paper_fit(z, log10_mvir, dn_dlogm):
    """End-to-end check of whether ``MassFunction(hmf_model="Yung24")`` reproduces Fig. A1.

    Unlike ``test_yung24_fsigma_matches_paper_fit`` above (which isolates fsigma's
    functional form using an independently-derived sigma), this exercises the full
    pipeline -- hmf's own transfer-function-based sigma(M), mass definition handling,
    and the dn/dm relation -- exactly as a user would call it. It uses the same
    digitized points from Fig. A1's fitted curve, but is a genuinely different check:
    it can catch bugs in how fsigma is *used* (mass definitions, rho_m, unit handling)
    that the fsigma-only test above cannot.

    The same exp(-c/sigma^2) sensitivity applies here too, plus hmf's EH transfer
    function is only an approximation to the actual GUREFT/MultiDark power spectrum.
    We reuse the same scaled-tolerance strategy, measuring local sensitivity via a
    small sigma_8 perturbation through the full pipeline (a proxy for the same
    sigma-sensitivity, since sigma_8 rescales sigma(M) at fixed M by roughly the
    same fraction at all M).
    """
    dndlogm_code = _yung24_dndlogm_phys(z, log10_mvir)

    sensitivity = _yung24_dndlogm_sensitivity(z, log10_mvir)
    base_rtol, sigma_fit_rtol = 0.08, 0.08
    rtol = base_rtol + abs(sensitivity) * sigma_fit_rtol

    np.testing.assert_allclose(dndlogm_code, dn_dlogm, rtol=rtol)


@pytest.mark.filterwarnings("ignore:.*does not match the mass definition.*:UserWarning")
@pytest.mark.parametrize("z", [0.0, 1.0, 2.0])
def test_tinker08_native_so_definition_consistency(z):
    """Tinker08 should agree between equivalent native SO definitions.

    Tinker08 uses `SOGeneric` as its measured mass definition, so it natively supports
    spherical-overdensity definitions without entering the mass-definition conversion
    branch used by the PR #281 regression above.
    """
    from astropy.cosmology import FlatLambdaCDM

    cosmo_params = {"Om0": 0.3, "H0": 70.0, "Ob0": 0.05}
    cosmo = FlatLambdaCDM(**cosmo_params)
    equiv_overdensity = 200.0 / cosmo.Om(z)

    common = {
        "hmf_model": "Tinker08",
        "transfer_model": "EH",
        "Mmin": 10,
        "Mmax": 15,
        "dlog10m": 0.1,
        "z": z,
        "disable_mass_conversion": False,
        "cosmo_params": cosmo_params,
    }

    h_crit = MassFunction(
        mdef_model="SOCritical",
        mdef_params={"overdensity": 200},
        **common,
    )
    h_mean = MassFunction(
        mdef_model="SOMean",
        mdef_params={"overdensity": equiv_overdensity},
        **common,
    )

    assert not _conversion_is_active(h_crit)
    assert not _conversion_is_active(h_mean)
    np.testing.assert_allclose(h_crit.dndm, h_mean.dndm, rtol=2e-2)


@pytest.mark.parametrize(
    ("z", "rtol"),
    [(0.0, 5e-3), (2.0, 1e-2), (4.0, 5e-3), (6.0, 3e-2), (8.0, 8e-2), (10.0, 1.5e-1)],
)
def test_tinker08_matches_colossus(z, rtol):
    """Tinker08 should remain reasonably close to the Colossus implementation.

    This compares the native `200m` Tinker08 prediction against Colossus at several
    redshifts using a matched cosmology. The tolerance widens with redshift because the
    remaining mismatch is still driven mostly by different high-z growth treatments and
    by the fact that `hmf` uses more precise Tinker08 coefficients while Colossus uses
    rounded table values. After tightening the selector threshold, the agreement is good
    enough to use a meaningfully stricter external regression than before. See
    `docs/technical/colossus_comparison.rst` for a longer explanation.
    """
    cosmo_params = {"H0": 67.74, "Om0": 0.3089, "Ob0": 0.0486}
    setCosmology(
        "hmf-tinker08-test",
        {
            "flat": True,
            "H0": cosmo_params["H0"],
            "Om0": cosmo_params["Om0"],
            "Ob0": cosmo_params["Ob0"],
            "sigma8": 0.8159,
            "ns": 0.9667,
        },
    )

    hmf = MassFunction(
        hmf_model="Tinker08",
        transfer_model="EH",
        mdef_model="SOMean",
        mdef_params={"overdensity": 200},
        Mmin=10,
        Mmax=14.2,
        dlog10m=0.02,
        z=z,
        cosmo_params=cosmo_params,
        sigma_8=0.8159,
        n=0.9667,
    )

    for mass in (1e11, 1e12, 1e13):
        hmf_dndlnm = np.exp(np.interp(np.log(mass), np.log(hmf.m), np.log(hmf.dndlnm)))
        colossus_dndlnm = massFunction(
            mass,
            z,
            q_in="M",
            q_out="dndlnM",
            mdef="200m",
            model="tinker08",
        )
        assert hmf_dndlnm == pytest.approx(colossus_dndlnm, rel=rtol)
