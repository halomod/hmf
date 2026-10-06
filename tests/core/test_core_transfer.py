"""Tests of the v4 transfer models and the Transfer stage.

Physical references: the k -> 0 limit, an independent implementation of EH98
(colossus), the paper's own exact formulae against its fits, astropy's background
(z_eq, k_eq), known limits of the BBKS and BondEfs fits, and bounds against CAMB.
"""

import pickle

import astropy.cosmology.units as cu
import astropy.units as u
import numpy as np
import pytest
from astropy import constants as const
from astropy.cosmology import FlatLambdaCDM, Planck18, wCDM

from hmf.core import transfer_models as tm
from hmf.core.accuracy import KAccuracy
from hmf.core.domain import DomainError
from hmf.core.transfer import Transfer
from hmf.core.units import UnitBoundaryError, h_Mpc

ANALYTIC = [tm.EH_BAO, tm.EH_NoBAO, tm.BBKS, tm.BondEfs]
ACC = KAccuracy()


def ln_t(model, k, cosmo=Planck18, species="cb"):
    return model.solve(cosmo, ACC).ln_transfer(np.asarray(k, dtype=float), species)


# ---------------------------------------------------------------------------------
# Every model
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize("model", [*ANALYTIC, tm.CAMB], ids=lambda m: m.__name__)
def test_transfer_tends_to_unity_on_large_scales(model):
    """Modes far outside the horizon at equality are unprocessed: T(k -> 0) = 1."""
    t = np.exp(ln_t(model(), [1e-8, 1e-7]))
    # Tolerance: the leading correction is O(k / Gamma) for BBKS (the slowest); at
    # k = 1e-8 h/Mpc that is ~1e-6. Measured max 1.3e-6 (BBKS), 1e-10 for the rest.
    np.testing.assert_allclose(t, 1.0, atol=1e-5)


def test_fromarray_tends_to_unity_even_if_not_normalised():
    k = np.geomspace(1e-4, 10, 200)
    t = 3.7 * np.exp(ln_t(tm.EH_NoBAO(), k))
    model = tm.FromArray(k=k * h_Mpc, t=t)
    np.testing.assert_allclose(np.exp(ln_t(model, [1e-8, 1e-6])), 1.0, atol=1e-6)


@pytest.mark.parametrize("model", [tm.EH_NoBAO, tm.BBKS, tm.BondEfs], ids=lambda m: m.__name__)
def test_transfer_decreases_monotonically_without_bao(model):
    lnt = ln_t(model(), np.logspace(-6, 4, 500))
    assert np.all(np.diff(lnt) < 0)


@pytest.mark.parametrize("model", ANALYTIC, ids=lambda m: m.__name__)
def test_fitting_formulae_give_the_same_transfer_for_both_species(model):
    sol = model().solve(Planck18, ACC)
    k = np.logspace(-3, 2, 20)
    np.testing.assert_array_equal(sol.ln_transfer(k, "cb"), sol.ln_transfer(k, "tot"))


# ---------------------------------------------------------------------------------
# The Eisenstein & Hu fits
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cosmo",
    [Planck18, FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05, Tcmb0=2.725)],
    ids=["Planck18", "LCDM"],
)
@pytest.mark.parametrize(
    ("model", "reference", "rtol"),
    [
        # colossus' BAO form uses rounded constants in a few places (e.g. its q);
        # measured max 1.9e-4.
        (tm.EH_BAO, "modelEisenstein98", 5e-4),
        # The no-wiggle form is coded identically (Eqs. 26-31); measured 1e-15.
        (tm.EH_NoBAO, "modelEisenstein98ZeroBaryon", 1e-12),
    ],
)
def test_eh98_matches_colossus(cosmo, model, reference, rtol):
    """EH98 agrees with colossus' independent implementation of the same paper."""
    ps = pytest.importorskip("colossus.cosmology.power_spectrum")
    k = np.logspace(-4, 2, 60)
    ref = getattr(ps, reference)(k, cosmo.h, cosmo.Om0, cosmo.Ob0, cosmo.Tcmb0.value)
    np.testing.assert_allclose(np.exp(ln_t(model(), k, cosmo)), ref, rtol=rtol)


def test_eh98_sound_horizon_fit_matches_exact_integral():
    """EH98 Eq. 26 approximates the sound horizon of Eq. 6 to 2% (EH98, text)."""
    s = tm._eh98_scales(Planck18)
    assert s.sound_horizon_fit == pytest.approx(s.sound_horizon, rel=0.02)


def test_eh98_sound_horizon_close_to_planck_measurement():
    """Planck 2018 (VI, Table 2) measures r_drag = 147.09 Mpc."""
    # Tolerance: EH98's drag-redshift fit is good to a few percent; measured +2.7%.
    assert tm._eh98_scales(Planck18).sound_horizon == pytest.approx(147.09, rel=0.05)


def test_eh98_equality_scales_match_astropy_background():
    """z_eq (EH98 Eq. 2) and k_eq (Eq. 3) agree with astropy's background.

    At equality rho_m = rho_r, so 1 + z_eq = Om0 / Or0, and k_eq = a_eq H(a_eq) / c.
    EH98 assume N_eff ~ 3 massless neutrinos; the cosmology here has massless
    neutrinos with Neff = 3.046.
    """
    cosmo = FlatLambdaCDM(H0=67.7, Om0=0.31, Ob0=0.049, Tcmb0=2.7255, Neff=3.046)
    s = tm._eh98_scales(cosmo)
    zp1_eq = cosmo.Om0 / (cosmo.Ogamma0 + cosmo.Onu0)
    k_eq = (cosmo.H(zp1_eq - 1) / (zp1_eq * const.c)).to_value(1 / u.Mpc)
    # Tolerance: EH98's 2.50e4 corresponds to Neff = 2.7 rather than 3.046 (a 1.0%
    # difference in Or0); measured 0.7% in z_eq and 0.4% in k_eq.
    assert s.z_eq == pytest.approx(zp1_eq, rel=1.5e-2)
    assert s.k_eq == pytest.approx(k_eq, rel=1.5e-2)


def test_eh_bao_needs_baryons():
    with pytest.raises(ValueError, match="baryons"):
        Transfer(cosmology=FlatLambdaCDM(H0=70, Om0=0.3), model="EH")


def test_eh_bao_and_nobao_agree_outside_bao_range():
    k = np.concatenate([np.logspace(-6, -3.5, 10), np.logspace(0.5, 2, 10)])
    ratio = np.exp(ln_t(tm.EH_BAO(), k) - ln_t(tm.EH_NoBAO(), k))
    # Tolerance: EH98 build the no-wiggle form to track the envelope to 1-2%;
    # measured 1.2%.
    np.testing.assert_allclose(ratio, 1.0, atol=2e-2)


# ---------------------------------------------------------------------------------
# BBKS and BondEfs
# ---------------------------------------------------------------------------------


def test_bbks_at_q_equal_one():
    """BBKS Eq. G3 at q = k/Gamma = 1, evaluated by hand."""
    model = tm.BBKS(baryons="none")
    k = Planck18.Om0 * Planck18.h  # q = 1
    expected = np.log(3.34) / 2.34 * (1 + 3.89 + 16.1**2 + 5.46**3 + 6.71**4) ** -0.25
    assert np.exp(ln_t(model, [k]))[0] == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("baryons", ["sugiyama95", "sugiyama95_preprint"])
def test_bbks_baryon_correction_vanishes_without_baryons(baryons):
    cosmo = FlatLambdaCDM(H0=67.0, Om0=0.3, Ob0=1e-10)
    k = np.logspace(-4, 1, 30)
    np.testing.assert_allclose(
        ln_t(tm.BBKS(baryons=baryons), k, cosmo), ln_t(tm.BBKS(baryons="none"), k, cosmo), atol=1e-8
    )


def test_bbks_sugiyama_forms_agree_at_h_one_half():
    """sqrt(2h) = 1 at h = 0.5, so the published and preprint forms coincide."""
    cosmo = FlatLambdaCDM(H0=50.0, Om0=0.3, Ob0=0.05)
    k = np.logspace(-4, 1, 30)
    np.testing.assert_allclose(
        ln_t(tm.BBKS(), k, cosmo),
        ln_t(tm.BBKS(baryons="sugiyama95_preprint"), k, cosmo),
        rtol=1e-12,
    )


def test_bbks_matches_eh_zero_baryon_shape():
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3, Ob0=1e-6, Tcmb0=2.7255)
    k = np.logspace(-3, 1, 40)
    ratio = np.exp(ln_t(tm.BBKS(baryons="none"), k, cosmo) - ln_t(tm.EH_NoBAO(), k, cosmo))
    # Tolerance: BBKS is a ~10% fit; measured max 5.9% from EH98's zero-baryon form.
    np.testing.assert_allclose(ratio, 1.0, atol=0.1)


def test_bondefs_depends_only_on_k_over_gamma():
    k = np.logspace(-3, 1, 40)
    t1 = ln_t(tm.BondEfs(), k, FlatLambdaCDM(H0=70.0, Om0=0.3, Ob0=0.04))
    t2 = ln_t(tm.BondEfs(), k, FlatLambdaCDM(H0=50.0, Om0=0.42, Ob0=0.04))
    np.testing.assert_allclose(t1, t2, rtol=1e-12)


@pytest.mark.parametrize("om0", [0.2, 0.3, 0.4])
def test_bondefs_matches_eh_zero_baryon_shape(om0):
    cosmo = FlatLambdaCDM(H0=70.0, Om0=om0, Ob0=1e-6, Tcmb0=2.7255)
    k = np.logspace(-3, 0, 40)
    ratio = np.exp(ln_t(tm.BondEfs(), k, cosmo) - ln_t(tm.EH_NoBAO(), k, cosmo))
    # Tolerance: Jenkins et al. (1998) quote up to 6% between this fit and CMBFAST;
    # measured max 5.5%.
    np.testing.assert_allclose(ratio, 1.0, atol=0.07)


# ---------------------------------------------------------------------------------
# Boltzmann codes against the fits
# ---------------------------------------------------------------------------------


def test_camb_is_close_to_eh98():
    """EH98 fit CMBFAST to a few percent; CAMB must agree with it to that level."""
    k = np.logspace(-4, 1, 100)
    ratio = np.exp(ln_t(tm.CAMB(), k) - ln_t(tm.EH_BAO(), k))
    # Tolerance: EH98's quoted accuracy is a few percent, and Planck18 has a 0.06 eV
    # neutrino that EH98 lack (about -0.5% in T at high k). Measured max 3.2%.
    np.testing.assert_allclose(ratio, 1.0, atol=0.06)


def test_fromarray_reproduces_its_table_and_extrapolates_with_eh_shape():
    """A table of EH98 no-wiggle itself is reproduced inside and beyond the table.

    The residual from the EH98 reference is identically zero, so the interpolation is
    exact and the extrapolation (EH shape, matched in value and slope) is too.
    """
    k = np.geomspace(1e-4, 10, 120)
    model = tm.FromArray(k=k * h_Mpc, t=np.exp(ln_t(tm.EH_NoBAO(), k)))
    kk = np.geomspace(1e-7, 1e4, 300)
    np.testing.assert_allclose(ln_t(model, kk), ln_t(tm.EH_NoBAO(), kk), atol=1e-12)


def test_fromfile_reads_camb_columns(tmp_path):
    k = np.geomspace(1e-4, 10, 100)
    t_cb = np.exp(ln_t(tm.EH_NoBAO(), k))
    t_tot = t_cb * (1 - 0.02 * k / (1 + k))
    data = np.zeros((k.size, 8))
    data[:, 0], data[:, 6], data[:, 7] = k, t_tot, t_cb
    np.savetxt(tmp_path / "camb.dat", data)
    sol = tm.FromFile(fname=tmp_path / "camb.dat").solve(Planck18, ACC)
    kk = np.geomspace(1e-3, 5, 20)
    # The cb column is EH98 no-wiggle itself, which the table reproduces exactly.
    np.testing.assert_allclose(sol.transfer(kk, "cb"), np.exp(ln_t(tm.EH_NoBAO(), kk)), rtol=1e-10)
    assert np.all(sol.transfer(kk[-5:], "tot") < sol.transfer(kk[-5:], "cb"))

    np.savetxt(tmp_path / "two.dat", np.column_stack([k, t_cb]))
    two = tm.FromFile(fname=tmp_path / "two.dat").solve(Planck18, ACC)
    np.testing.assert_allclose(two.transfer(kk, "tot"), two.transfer(kk, "cb"))


# ---------------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------------


def test_models_are_registered_under_their_v3_names():
    for name in [
        "CAMB",
        "CLASS",
        "EH",
        "EH_BAO",
        "EH_NoBAO",
        "BBKS",
        "BondEfs",
        "FromFile",
        "FromArray",
    ]:
        assert issubclass(tm.TransferModel.get(name), tm.TransferModel)


def test_fromarray_needs_wavenumber_quantities():
    with pytest.raises(UnitBoundaryError):
        tm.FromArray(k=np.ones(5), t=np.ones(5))
    with pytest.raises(u.UnitConversionError):
        tm.FromArray(k=np.ones(5) / u.Mpc, t=np.ones(5))
    with pytest.raises(ValueError, match="same length"):
        tm.FromArray(k=np.ones(5) * h_Mpc, t=np.ones(4))


def test_camb_k_max_is_a_quantity():
    k_max = tm.CAMB(k_max=10 * h_Mpc).k_max
    assert k_max.unit is h_Mpc
    assert k_max.shape == ()
    assert k_max.value == 10.0
    assert tm.CAMB().k_max == 20.0 * h_Mpc
    # Stored in the canonical unit, and compared by value in it.
    camb = tm.CAMB(k_max=0.01 * cu.littleh / u.kpc)
    assert camb.k_max.unit is h_Mpc
    np.testing.assert_allclose(camb.k_max.value, 10.0, rtol=1e-15)
    with pytest.raises(
        UnitBoundaryError, match=r"CAMB: argument 'k_max'.*k_max \* hmf.core.units.h_Mpc"
    ):
        tm.CAMB(k_max=10)
    with pytest.raises(UnitBoundaryError, match="CLASS: argument 'k_max'"):
        tm.CLASS(k_max=10)
    with pytest.raises(u.UnitConversionError, match=r"CAMB: argument 'k_max'.*no H0"):
        tm.CAMB(k_max=10 / u.Mpc)
    with pytest.raises(ValueError, match="must be a scalar"):
        tm.CAMB(k_max=[10, 20] * h_Mpc)
    with pytest.raises(ValueError, match="k_max must be finite and > 0"):
        tm.CAMB(k_max=0 * h_Mpc)


def test_class_params_cannot_set_the_cosmology():
    with pytest.raises(ValueError, match="omega_b"):
        tm.CLASS(class_params={"omega_b": 0.02})


def test_boltzmann_models_reject_unsupported_cosmologies():
    from astropy.cosmology import Flatw0wzCDM

    with pytest.raises(TypeError, match="LambdaCDM"):
        Transfer(cosmology=Flatw0wzCDM(H0=70, Om0=0.3, Ob0=0.05, Tcmb0=2.7), model="CAMB")
    with pytest.raises(ValueError, match="Ob0"):
        Transfer(cosmology=FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.7), model="CAMB")
    with pytest.raises(ValueError, match="Tcmb0"):
        Transfer(cosmology=FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05), model="CAMB")


def test_models_are_hashable_and_compare_by_value():
    assert tm.CAMB(settings={"Accuracy.AccuracyBoost": 2}) == tm.CAMB(
        settings=(("Accuracy.AccuracyBoost", 2),)
    )
    assert len({tm.CAMB(), tm.CAMB(), tm.CLASS(), tm.BBKS()}) == 3


# ---------------------------------------------------------------------------------
# The Transfer stage
# ---------------------------------------------------------------------------------


def test_k_must_be_a_quantity():
    t = Transfer(model="EH")
    with pytest.raises(UnitBoundaryError):
        t.transfer_function(0.1)
    with pytest.raises(UnitBoundaryError):
        t.unnormalised_power(np.array([0.1, 1.0]))


def test_physical_wavenumbers_are_converted_with_h0():
    """K = 0.1 / Mpc is k = 0.1 / h h/Mpc."""
    cosmo = FlatLambdaCDM(H0=60, Om0=0.3, Ob0=0.05, Tcmb0=2.7255)
    t = Transfer(cosmology=cosmo, model="EH")
    k = np.array([0.01, 0.1, 1.0])
    np.testing.assert_allclose(
        t.transfer_function(k / u.Mpc), t.transfer_function(k / 0.6 * h_Mpc), rtol=1e-14
    )


def test_k_must_be_positive():
    with pytest.raises(DomainError):
        Transfer(model="EH").transfer_function(np.array([0.0, 1.0]) * h_Mpc)


def test_outputs_broadcast_like_k():
    t = Transfer(model="EH")
    assert isinstance(t.transfer_function(0.1 * h_Mpc), float)
    assert t.transfer_function(np.ones((2, 3)) * h_Mpc).shape == (2, 3)


def test_unnormalised_power_is_k_ns_t_squared():
    t = Transfer(model="EH_NoBAO", n_s=0.95)
    k = np.logspace(-3, 1, 10)
    np.testing.assert_allclose(
        t.unnormalised_power(k * h_Mpc), k**0.95 * t.transfer_function(k * h_Mpc) ** 2, rtol=1e-14
    )
    kernel = t.power_kernel("tot")
    np.testing.assert_allclose(np.exp(kernel.ln_power(np.log(k))), kernel.power(k), rtol=1e-13)


def test_power_kernel_is_batch_size_independent():
    kernel = Transfer(model="CAMB").power_kernel()
    k = np.logspace(-4, 3, 101)
    full = kernel.power(k)
    assert np.array_equal(full[17:40], kernel.power(k[17:40]))
    assert np.array_equal(full[5:6], kernel.power(k[5:6]))


def test_unknown_species_raises():
    with pytest.raises(ValueError, match="species"):
        Transfer(model="EH").transfer_function(1 * h_Mpc, species="nu")


def test_stage_compares_cosmologies_by_value_not_name():
    clone = FlatLambdaCDM(
        H0=Planck18.H0, Om0=Planck18.Om0, Tcmb0=Planck18.Tcmb0, Neff=Planck18.Neff,
        m_nu=Planck18.m_nu, Ob0=Planck18.Ob0, name="not Planck18",
    )  # fmt: skip
    a, b = Transfer(model="EH"), Transfer(model="EH", cosmology=clone)
    assert a == b
    assert hash(a) == hash(b)
    assert a != Transfer(model="EH", cosmology=Planck18.clone(H0=70))
    assert a != Transfer(model="EH", cosmology=wCDM(H0=67.66, Om0=0.31, Ode0=0.69, Ob0=0.05))


def test_stage_model_accepts_names_and_classes():
    assert Transfer(model="BBKS").model == tm.BBKS()
    assert Transfer(model=tm.BBKS).model == tm.BBKS()
    with pytest.raises(TypeError):
        Transfer(model=3)


def test_evolve_and_pickle():
    t = Transfer(model="EH")
    value = t.transfer_function(1 * h_Mpc)
    t2 = pickle.loads(pickle.dumps(t))
    assert t2 == t
    assert t2.transfer_function(1 * h_Mpc) == value
    assert t.evolve(n_s=1.0).unnormalised_power(1 * h_Mpc) == pytest.approx(value**2)


def test_evolve_revalidates_the_model_against_the_cosmology():
    t = Transfer(model="EH")
    with pytest.raises(ValueError, match="baryons"):
        t.evolve(cosmology=FlatLambdaCDM(H0=70, Om0=0.3))
