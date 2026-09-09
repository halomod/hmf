import numpy as np
import pytest
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from hmf import MassFunction
from hmf.halos import mass_definitions as md
from hmf.mass_function import fitting_functions as ff


def test_fittingfunction_requires_mass():
    with pytest.raises(ValueError):
        ff.Warren(nu2=np.array([1.0]))


def test_fittingfunction_requires_neff():
    with pytest.raises(ValueError):
        ff.Reed07(nu2=np.array([1.0]))


def test_fittingfunction_sets_measured_mdef():
    fit = ff.SMT(nu2=np.array([1.0, 2.0]))

    assert isinstance(fit.mass_definition, md.SOVirial)


def test_get_measured_mdef_unrecognized_overdensity_warns():
    class _BadSO(ff.BaseFittingFunction):
        sim_definition = ff.SimDetails(
            L=1,
            N=1,
            halo_finder_type="SO",
            omegam=0.3,
            sigma_8=0.8,
            halo_overdensity="bad",
        )

    with pytest.warns(UserWarning, match="Unrecognized overdensity criterion format"):
        assert _BadSO.get_measured_mdef() is None


def test_get_measured_mdef_unknown_halo_finder_warns():
    class _BadFinder(ff.BaseFittingFunction):
        sim_definition = ff.SimDetails(
            L=1,
            N=1,
            halo_finder_type="??",
            omegam=0.3,
            sigma_8=0.8,
            halo_overdensity=200,
        )

    with pytest.warns(UserWarning, match="Unknown halo finder type"):
        assert _BadFinder.get_measured_mdef() is None


def test_base_cutmask_all_true():
    fit = ff.PS(nu2=np.array([1.0, 2.0, 3.0]))
    mask = fit.cutmask

    assert mask.dtype == bool
    assert mask.shape == (3,)
    assert np.all(mask)


def test_smt_validation_p():
    with pytest.raises(ValueError):
        ff.SMT(nu2=np.array([1.0]), p=0.6)


def test_smt_validation_a():
    with pytest.raises(ValueError):
        ff.SMT(nu2=np.array([1.0]), a=-1.0)


def test_jenkins_cutmask_shape():
    fit = ff.Jenkins(nu2=np.array([0.5, 1.0, 2.0]))
    mask = fit.cutmask

    assert mask.shape == (3,)


def test_bhattacharya_q_validation():
    with pytest.raises(ValueError):
        ff.Bhattacharya(nu2=np.array([1.0]), m=np.array([1e12]), q=0)


def test_bhattacharya_pq_validation():
    with pytest.raises(ValueError):
        ff.Bhattacharya(nu2=np.array([1.0]), m=np.array([1e12]), p=1.0, q=1.0)


def test_bhattacharya_cutmask():
    fit = ff.Bhattacharya(nu2=np.array([1.0, 2.0]), m=np.array([1e12, 1e16]))
    mask = fit.cutmask

    assert mask.shape == (2,)
    assert np.array_equal(mask, np.array([True, False]))


def test_bhattacharya_norm_returns_existing_a():
    fit = ff.Bhattacharya(nu2=np.array([1.0]), m=np.array([1e12]))

    assert fit._norm() == fit.params["A"]


def test_bhattacharya_normed_uses_norm():
    fit = ff.Bhattacharya(nu2=np.array([1.0]), m=np.array([1e12]), normed=True)

    assert np.isfinite(fit.params["A"])


def test_tinker10_non_so_raises():
    with pytest.raises(ValueError):
        ff.Tinker10(nu2=np.array([1.0]), mass_definition=md.FOF(linking_length=0.2))


def test_tinker10_interpolates_params():
    fit = ff.Tinker10(nu2=np.array([1.0]), mass_definition=md.SOMean(overdensity=250))

    assert fit.delta_halo == 250
    assert np.isfinite(fit.beta)


def test_tinker10_non_terminate_clamps():
    class _Tinker10NoTerminate(ff.Tinker10):
        sim_definition = None
        terminate = False

    fit = _Tinker10NoTerminate(
        nu2=np.array([1.0]),
        gamma_200=-1.0,
        eta_200=-1.0,
        phi_200=1.0,
        beta_200=-1.0,
    )

    assert fit.gamma == pytest.approx(1e-3)
    assert fit.eta == pytest.approx(-0.499)
    assert fit.phi == pytest.approx(fit.eta + 0.499)
    assert fit.beta == pytest.approx(1e-3)


def test_tinker10_normalise_alpha_at_z0():
    class _Tinker10NoSim(ff.Tinker10):
        sim_definition = None

    fit = _Tinker10NoSim(nu2=np.array([1.0]))

    assert fit._normalise == fit.params["alpha_200"]


@pytest.mark.parametrize(
    ("fit_cls", "m", "expected"),
    [
        (ff.Warren, np.array([1e9, 1e12, 1e16]), np.array([False, True, False])),
        (ff.Angulo, np.array([1e7, 1e10, 1e17]), np.array([False, True, False])),
        (ff.Crocce, np.array([1e10, 1e12, 1e16]), np.array([False, True, False])),
        (ff.Ishiyama, np.array([1e7, 1e9, 1e17]), np.array([False, True, False])),
    ],
)
def test_mass_cutmasks(fit_cls, m, expected):
    fit = fit_cls(nu2=np.ones_like(m), m=m, z=0.5)

    assert np.array_equal(fit.cutmask, expected)


@pytest.mark.parametrize(
    ("fit_cls", "nu2", "expected"),
    [
        (ff.Reed03, np.array([1e-6, 1.0, 1e6]), np.array([False, True, False])),
        (ff.Reed07, np.array([1e-6, 2.0, 1e6]), np.array([False, True, False])),
        (ff.Courtin, np.array([1e-6, 1.0, 1e6]), np.array([False, True, False])),
        (ff.Watson_FoF, np.array([1e-6, 1.0, 1e6]), np.array([False, True, False])),
    ],
)
def test_lnsigma_cutmasks(fit_cls, nu2, expected):
    kwargs = {}
    if fit_cls is ff.Reed07:
        kwargs["n_eff"] = np.zeros_like(nu2)

    fit = fit_cls(nu2=nu2, **kwargs)

    assert np.array_equal(fit.cutmask, expected)


def test_peacock_cutmask_all_false():
    m = np.array([1e8, 1e12, 1e16])
    fit = ff.Peacock(nu2=np.ones_like(m), m=m)

    assert np.all(~fit.cutmask)


def test_tinker08_non_so_raises():
    with pytest.raises(ValueError):
        ff.Tinker08(
            nu2=np.array([1.0]),
            mass_definition=md.FOF(linking_length=0.2),
        )


def test_tinker08_cutmask_z_positive():
    fit = ff.Tinker08(
        nu2=np.array([1.0, 2.0, 10.0]),
        z=1.0,
        mass_definition=md.SOMean(overdensity=200),
    )

    assert np.array_equal(fit.cutmask, np.array([False, True, True]))


def test_tinker08_cutmask_z_zero():
    fit = ff.Tinker08(
        nu2=np.array([0.01, 1.0, 100.0]),
        z=0.0,
        mass_definition=md.SOMean(overdensity=200),
    )

    assert np.array_equal(fit.cutmask, np.array([False, True, False]))


def test_tinker10_eta_phi_validation():
    class _Tinker10NoSim(ff.Tinker10):
        sim_definition = None

    with pytest.raises(ValueError):
        _Tinker10NoSim(nu2=np.array([1.0]), eta_200=0.0, phi_200=1.0)


def test_tinker10_cutmask_z_zero():
    class _Tinker10NoSim(ff.Tinker10):
        sim_definition = None

    fit = _Tinker10NoSim(nu2=np.array([1.0, 2.0, 10.0]), z=0.0)

    assert np.array_equal(fit.cutmask, np.array([True, True, True]))


def test_tinker10_cutmask_z_positive():
    class _Tinker10NoSim(ff.Tinker10):
        sim_definition = None

    fit = _Tinker10NoSim(nu2=np.array([1.0, 2.0, 10.0]), z=1.0)

    assert np.array_equal(fit.cutmask, np.array([False, True, True]))


def test_watson_gamma_and_fsigma_branches():
    class _WatsonNoSim(ff.Watson):
        sim_definition = None

    fit_z0 = _WatsonNoSim(nu2=np.array([1.0, 2.0]), z=0.0)
    fit_zhi = _WatsonNoSim(nu2=np.array([1.0, 2.0]), z=7.0)

    assert np.isfinite(fit_z0.gamma()).all()
    assert np.isfinite(fit_z0.fsigma).all()
    assert np.isfinite(fit_zhi.fsigma).all()


def test_watson_gamma_non_so_raises():
    class _WatsonNoSim(ff.Watson):
        sim_definition = None

    fit = _WatsonNoSim(nu2=np.array([1.0]), mass_definition=md.FOF(linking_length=0.2))

    with pytest.raises(ValueError):
        fit.gamma()


def test_watson_cutmask():
    class _WatsonNoSim(ff.Watson):
        sim_definition = None

    fit = _WatsonNoSim(nu2=np.array([1e-6, 1.0, 1e6]))

    assert np.array_equal(fit.cutmask, np.array([False, True, False]))


def test_behroozi_modify_dndm_handles_nan():
    fit = ff.Behroozi(
        nu2=np.array([1.0, 2.0]),
        mass_definition=md.SOMean(overdensity=200),
    )
    m = np.array([-1e11, 1e12])
    dndm = np.array([1.0, 1.0])
    ngtm = np.array([1.0, 1.0])

    with pytest.warns(RuntimeWarning):
        res = fit._modify_dndm(m, dndm, z=1.0, ngtm_tinker=ngtm)

    assert res[0] == 0
    assert np.isfinite(res[1])


def _behroozi_theta(fit, m, z):
    """Isolate the multiplicative correction 10**theta(M,z) from Eqs. G2/G3.

    With dndm=1 and ngtm_tinker=0, `_modify_dndm` reduces to `dndm * theta`
    exactly (the ngtm_tinker*dthetadM term vanishes with no NaN), giving us
    theta itself without touching any dndm/ngtm plumbing.
    """
    return fit._modify_dndm(np.atleast_1d(m), np.array([1.0]), z, np.array([0.0]))[0]


def _behroozi_dthetadm(fit, m, z):
    """Isolate -dtheta/dM by setting dndm=0, ngtm_tinker=1.

    With dndm=0 and ngtm_tinker=1, `_modify_dndm` reduces to `-dthetadM` exactly.
    """
    return -fit._modify_dndm(np.atleast_1d(m), np.array([0.0]), z, np.array([1.0]))[0]


@pytest.mark.parametrize("z", [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 9.0, 15.0])
@pytest.mark.parametrize("logm", [8.0, 9.0, 10.0, 11.5, 13.0, 15.0])
def test_behroozi_theta_matches_closed_form(z, logm):
    """theta(M,z) should exactly match Eqs. G2/G3, independent of dndm/ngtm."""
    fit = ff.Behroozi(nu2=np.array([1.0]), mass_definition=md.SOMean(overdensity=200))
    m = 10.0**logm

    a = 1 / (1 + z)
    alpha = 0.144 / (1 + np.exp(14.79 * (a - 0.213)))
    gamma = 0.5 / (1 + np.exp(6.5 * a))
    expected = 10 ** (alpha * (m / 10**11.5) ** gamma)

    assert _behroozi_theta(fit, m, z) == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("z", [1.0, 4.0, 9.0])
@pytest.mark.parametrize("logm", [9.0, 11.5, 13.5])
def test_behroozi_dthetadm_matches_finite_difference(z, logm):
    """Check that dthetadM is the actual derivative of theta, not just similarly-shaped.

    Rather than hand-deriving the "correct" formula as ground truth, this checks
    the code's derivative against a finite difference of the code's own theta,
    so it stays correct even if Eq. G2/G3's functional form is ever revised.
    """
    fit = ff.Behroozi(nu2=np.array([1.0]), mass_definition=md.SOMean(overdensity=200))
    m0 = 10.0**logm
    h = m0 * 1e-6

    fd_deriv = (_behroozi_theta(fit, m0 + h, z) - _behroozi_theta(fit, m0 - h, z)) / (2 * h)

    assert _behroozi_dthetadm(fit, m0, z) == pytest.approx(fd_deriv, rel=1e-4)


@pytest.mark.parametrize("z", [0.5, 2.0, 6.0, 9.0, 15.0])
@pytest.mark.parametrize("logm", [7.0, 9.0, 11.5, 14.0, 16.0])
def test_behroozi_never_suppresses_relative_to_tinker(z, logm):
    """alpha>0 and gamma>0 for every physical z, so theta>=1 always.

    Behroozi's high-z correction is a pure boost: it should never predict
    *fewer* halos than the underlying Tinker08 fit, at any mass or redshift.
    """
    fit = ff.Behroozi(nu2=np.array([1.0]), mass_definition=md.SOMean(overdensity=200))

    assert _behroozi_theta(fit, 10.0**logm, z) >= 1.0 - 1e-12


def test_behroozi_correction_monotonic_in_mass():
    """theta(M) must be non-decreasing in M at fixed z.

    Eq. G3's exponent gamma is always positive: more massive halos collapsed
    earlier and so pick up a larger correction.
    """
    fit = ff.Behroozi(nu2=np.array([1.0]), mass_definition=md.SOMean(overdensity=200))
    logm = np.linspace(8, 16, 50)
    theta = np.array([_behroozi_theta(fit, 10.0**lm, z=9.0) for lm in logm])

    assert np.all(np.diff(theta) >= 0)


def test_behroozi_correction_vanishes_at_z_zero():
    """Behroozi should collapse onto the bare Tinker08 prediction at z=0.

    As z->0, a->1 and alpha decays super-exponentially to 0.
    """
    fit = ff.Behroozi(nu2=np.array([1.0]), mass_definition=md.SOMean(overdensity=200))
    logm = np.linspace(8, 16, 20)
    theta = np.array([_behroozi_theta(fit, 10.0**lm, z=0.0) for lm in logm])

    assert theta == pytest.approx(1.0, abs=1e-4)


def test_behroozi_ngtm():
    """Ensure that ngtm for Behroozi / Tinker matches Behroozi Fig. 23.

    See https://arxiv.org/pdf/1207.6105.
    """
    # Behroozi's correction is calibrated against the virial mass function, so both
    # models must be evaluated in the same mass definition -- otherwise the comparison
    # also picks up the (unrelated) shift between SOMean(200) and SOVirial.
    common_kwargs = {
        "z": 9,
        "Mmin": 9,
        "Mmax": 15.5,
        "dlog10m": 0.05,
        "cosmo_params": {"H0": 70.0},
        "mdef_model": md.SOVirial,
    }
    tinker = MassFunction(hmf_model="Tinker08", **common_kwargs)
    behroozi = MassFunction(hmf_model="Behroozi", **common_kwargs)

    masses_msun = tinker.m / tinker.cosmo.h
    ngtm_tinker = Spline(np.log10(masses_msun), np.log10(tinker.ngtm))
    ngtm_behroozi = Spline(np.log10(masses_msun), np.log10(behroozi.ngtm))

    assert np.isclose(ngtm_behroozi(10) - ngtm_tinker(10), 0.07, atol=0.005)
    assert np.isclose(ngtm_behroozi(11.5) - ngtm_tinker(11.5), 0.12, atol=0.005)
    assert np.isclose(ngtm_behroozi(13) - ngtm_tinker(13), 0.22, atol=0.005)

    # Now do a test at z=2, where there is a very small difference between masses.
    tinker.update(z=2)
    behroozi.update(z=2)
    ngtm_tinker = Spline(np.log10(masses_msun), np.log10(tinker.ngtm))
    ngtm_behroozi = Spline(np.log10(masses_msun), np.log10(behroozi.ngtm))

    assert np.isclose(ngtm_behroozi(10) - ngtm_tinker(10), 0.02, atol=0.005)
    assert np.isclose(ngtm_behroozi(11.5) - ngtm_tinker(11.5), 0.02, atol=0.005)
    assert np.isclose(ngtm_behroozi(13) - ngtm_tinker(13), 0.02, atol=0.005)
