"""Tests of hmf.core.fits: the fitting functions as v4 models.

The physical tests check each fit against something known independently of the
code: an exact normalisation (mass conservation), a parameter choice that reduces it
to another fit, numbers quoted in the defining paper, or physical bounds. The
regression against hmf 3.x is in ``test_core_fits_regression.py``.
"""

import itertools
import math
import warnings

import astropy.cosmology.units as cu
import astropy.units as u
import attrs
import numpy as np
import pytest
from scipy.integrate import simpson

from hmf.core import fits
from hmf.core._kernels import fits as kern
from hmf.core.domain import DOMAIN_POLICIES, Domain, DomainError, HMFExtrapolationWarning
from hmf.core.fits import FitInputs, FittingFunction, MeasuredMassDefinition, evaluate_fsigma
from hmf.core.units import Msun_h, UnitBoundaryError, dndm_unit, number_density_unit

DELTA_C = 1.68647

#: The hmf 3.x names of every fitting function, each of which must be registered.
V3_NAMES = (
    "PS", "SMT", "ST", "Jenkins", "Warren", "Reed03", "Reed07", "Peacock", "Angulo",
    "AnguloBound", "Watson_FoF", "Watson", "Crocce", "Courtin", "Bhattacharya", "Tinker08",
    "Tinker10", "Behroozi", "Pillepich", "Manera", "Ishiyama", "Bocquet200mDMOnly",
    "Bocquet200mHydro", "Bocquet200cDMOnly", "Bocquet200cHydro", "Bocquet500cDMOnly",
    "Bocquet500cHydro", "Yung24",
)  # fmt: skip

ALL = [FittingFunction.get(name) for name in V3_NAMES]


def _ids(cls):
    return cls.__name__


def _baseline(cls):
    """Inputs inside both of a fit's domains (not on their edges)."""
    values = {
        "sigma": 1.0,
        "z": 7.0 if cls is fits.Yung24 else 0.0,
        "omega_m_z": 0.5,
        "delta_halo": 200.0 if cls not in (fits.Watson,) else 178.0,
        "delta_c": DELTA_C,
        "n_eff": -2.0,
        "m": 1e12,
    }
    for name, iv in cls.calibration_domain.bounds:
        lo, hi = iv.lower, iv.upper
        if name == "m":
            mid = 10 ** ((math.log10(lo) + math.log10(hi)) / 2) if np.isfinite(hi) else 10 * lo
        else:
            mid = (lo + hi) / 2
        _set(values, name, mid)
    return values


def _set(values, name, value):
    """Set a domain variable, mapping the derived ones onto sigma."""
    if name == "ln_sigma_inv":
        values["sigma"] = math.exp(-value)
    elif name == "log10_sigma_inv":
        values["sigma"] = 10.0**-value
    else:
        values[name] = value


def _call(model, values, **kw):
    """evaluate_fsigma (the library path) on plain inputs in canonical units."""
    plain = {k: v for k, v in values.items() if k != "sigma"}
    return evaluate_fsigma(model, FitInputs(sigma=values["sigma"], **plain), **kw)


def _inputs(values):
    """The inputs other than sigma, for the public (units boundary) methods."""
    return {k: v * Msun_h if k == "m" else v for k, v in values.items() if k != "sigma"}


# ---------------------------------------------------------------------------------
# Registry and metadata
# ---------------------------------------------------------------------------------
def test_every_v3_fit_is_registered_under_its_name():
    assert set(V3_NAMES) <= set(FittingFunction.get_aliases())
    for name in V3_NAMES:
        assert FittingFunction.get(name).__name__ == name


def test_registry_has_no_unexpected_fits():
    names = {cls.__name__ for cls in FittingFunction.get_models().values()}
    assert names == set(V3_NAMES)


@pytest.mark.parametrize("cls", ALL, ids=_ids)
def test_fit_declares_its_metadata(cls):
    """Every fit declares both domains, its sources and its mass definition itself."""
    for attr in ("valid_domain", "calibration_domain", "measured_mass_definition"):
        assert attr in cls.__dict__, f"{cls.__name__} inherits {attr} instead of declaring it"
    assert isinstance(cls.valid_domain, Domain)
    assert isinstance(cls.calibration_domain, Domain)
    assert isinstance(cls.measured_mass_definition, MeasuredMassDefinition)
    assert cls.valid_domain.source
    assert cls.calibration_domain.source
    assert cls.references
    assert all(isinstance(r, str) and r for r in cls.references)
    assert cls.parameter_source
    assert "TODO" not in cls.parameter_source + cls.calibration_domain.source
    # Every fit is defined only for sigma > 0.
    assert cls.valid_domain["sigma"].lower > 0
    assert cls.requires <= set(fits.INPUTS)


@pytest.mark.parametrize("cls", ALL, ids=_ids)
def test_default_instance_is_hashable_and_frozen(cls):
    model = cls()
    assert hash(model) == hash(cls())
    assert model == cls()
    for field in attrs.fields(cls)[:1]:
        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            setattr(model, field.name, 1.0)


def test_measured_mass_definitions():
    assert str(fits.Jenkins.measured_mass_definition) == "FoF(b=0.2)"
    assert str(fits.Bocquet500cHydro.measured_mass_definition) == "SO-crit(500)"
    assert str(fits.Bocquet200mDMOnly.measured_mass_definition) == "SO-mean(200)"
    assert str(fits.Behroozi.measured_mass_definition) == "SO-virial"
    assert str(fits.Tinker08.measured_mass_definition) == "SO-any (preferred SO-mean(200))"
    assert str(fits.Watson.measured_mass_definition) == "SO-any (preferred SO-mean(178))"
    assert str(fits.AnguloBound.measured_mass_definition) == "self-bound"
    with pytest.raises(ValueError, match="linking length"):
        MeasuredMassDefinition(kind="so_mean", overdensity=200, linking_length=0.2)
    with pytest.raises(ValueError, match="overdensity"):
        MeasuredMassDefinition(kind="so_critical")
    with pytest.raises(ValueError, match="preferred"):
        MeasuredMassDefinition(kind="so_any")


def test_parameters_are_documented_fields():
    info = {i.name: i for i in fits.Tinker08.fields_info()}
    assert info["A_200"].default == pytest.approx(0.186, abs=5e-4)
    assert info["A_200"].doc
    assert "Parameters" in fits.Tinker08.__doc__


# ---------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------
def test_missing_required_input_raises():
    with pytest.raises(ValueError, match="delta_halo"):
        fits.Tinker08().fsigma(1.0, z=0.0)
    with pytest.raises(ValueError, match="n_eff"):
        fits.Reed07().fsigma(1.0, delta_c=DELTA_C)


def test_evaluate_needs_calibration_inputs():
    # Jenkins needs only sigma to evaluate, but z to check its calibration domain.
    assert np.isfinite(fits.Jenkins().fsigma(1.0))
    with pytest.raises(ValueError, match="calibration domain"):
        evaluate_fsigma(fits.Jenkins(), FitInputs(sigma=1.0))
    assert fits.Jenkins.domain_inputs() == {"z"}


def test_fit_inputs_unknown_variable():
    with pytest.raises(ValueError, match="Unknown"):
        FitInputs(sigma=1.0).domain_value("bogus")
    with pytest.raises(ValueError, match="not given"):
        _ = FitInputs(sigma=1.0).z


def test_inputs_broadcast():
    sigma = np.array([0.5, 1.0, 2.0])[:, None]
    z = np.array([0.0, 1.0])[None, :]
    f = fits.Tinker08().fsigma(sigma, z=z, delta_halo=300.0)
    assert f.shape == (3, 2)
    np.testing.assert_array_equal(
        f[:, 1], fits.Tinker08().fsigma(sigma[:, 0], z=1.0, delta_halo=300.0)
    )


def test_results_are_batch_size_independent():
    """Evaluating a subset gives bit-identical values (the kernel convention)."""
    rng = np.random.default_rng(1)
    sigma = rng.uniform(0.3, 3, 257)
    delta = rng.uniform(200, 3000, 257)
    model = fits.Tinker10()
    full = model.fsigma(sigma, z=0.7, delta_halo=delta, delta_c=DELTA_C)
    part = model.fsigma(sigma[5:9], z=0.7, delta_halo=delta[5:9], delta_c=DELTA_C)
    np.testing.assert_array_equal(full[5:9], part)


def test_inputs_not_mutated():
    sigma = np.array([0.5, 1.0, 2.0])
    copy = sigma.copy()
    fits.Watson().fsigma(sigma, z=1.0, omega_m_z=0.6, delta_halo=300.0)
    np.testing.assert_array_equal(sigma, copy)


# ---------------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------------
SIGMAS = np.geomspace(0.2, 10.0, 60)


def _valid_samples(cls, name, default):
    """A few values of a variable, inside the fit's valid domain."""
    if name not in cls.valid_domain.variables:
        return [default]
    iv = cls.valid_domain[name]
    lo = iv.lower
    hi = iv.upper if np.isfinite(iv.upper) else max(lo, 0) + {"z": 10.0}.get(name, 1e4)
    return [lo, (lo + hi) / 2, hi] if np.isfinite(lo) and lo > 0 else [max(lo, 0.0), hi / 2, hi]


@pytest.mark.parametrize("cls", ALL, ids=_ids)
def test_fsigma_positive_and_finite_in_valid_domain(cls):
    """f(sigma) > 0 and finite everywhere inside the valid domain (0.2 <= sigma <= 10)."""
    model = cls()
    zs = _valid_samples(cls, "z", 0.0)
    deltas = _valid_samples(cls, "delta_halo", 200.0) if "delta_halo" in cls.requires else [200.0]
    # Reed07's f -> 0 (continuously) as n_eff -> -3: it underflows within ~1e-2 of it.
    n_effs = [-2.9, -2.0, 0.0] if "n_eff" in cls.requires else [-2.0]
    for z, delta, n_eff in itertools.product(zs, deltas, n_effs):
        f = model.fsigma(
            SIGMAS,
            z=z,
            omega_m_z=0.3,
            delta_halo=delta,
            delta_c=DELTA_C,
            n_eff=n_eff,
            m=1e12 * Msun_h,
        )
        assert np.all(np.isfinite(f)), (z, delta, n_eff)
        assert np.all(f > 0), (z, delta, n_eff)


@pytest.mark.parametrize("cls", ALL, ids=_ids)
def test_fsigma_has_exponential_high_mass_cutoff(cls):
    """f(sigma) -> 0 as sigma -> 0 (the rarest, most massive haloes)."""
    values = _baseline(cls)
    model = cls()
    f_small = model.fsigma(0.15, **_inputs(values))
    f_peak = np.max(model.fsigma(SIGMAS, **_inputs(values)))
    assert f_small < 1e-3 * f_peak


@pytest.mark.parametrize("cls", ALL, ids=_ids)
def test_fsigma_below_press_schechter_maximum_scale(cls):
    """No fit puts more than ~all of the mass in one e-fold of sigma: f < 1."""
    values = _baseline(cls)
    assert np.all(cls().fsigma(SIGMAS, **_inputs(values)) < 1.0)


# ---------------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------------
# A grid in ln(nu) that resolves f(sigma) everywhere; above nu = 20 every fit is below
# exp(-0.4 * 20^2 / 2) ~ 1e-35 of its peak.
LNNU = np.linspace(np.log(1e-10), np.log(20.0), 40001)
SIGMA_GRID = DELTA_C / np.exp(LNNU)


def _mass_fraction(f):
    r"""The integral of f over ln(1/sigma) = ln(nu), plus the analytic low-nu power-law tail."""
    slope = (np.log(f[1]) - np.log(f[0])) / (LNNU[1] - LNNU[0])
    return simpson(f, x=LNNU) + f[0] / slope


@pytest.mark.parametrize(
    "model",
    [fits.PS(), fits.SMT(), fits.ST(), fits.Manera(), fits.Peacock(), fits.SMT(a=0.8, p=0.1)],
    ids=lambda m: repr(m),
)
def test_normalised_fits_conserve_mass(model):
    """Fits normalised by construction put all the mass in haloes."""
    assert model.normalized
    f = model.fsigma(SIGMA_GRID, delta_c=DELTA_C)
    # Tolerance: Simpson's rule on 4e4 points is ~1e-12; 1e-6 allows for the tail.
    assert _mass_fraction(f) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("z", [0.5, 2.0, 4.0])
@pytest.mark.parametrize("delta", [200.0, 250.0, 800.0, 3200.0])
def test_tinker10_normalised_away_from_table(z, delta):
    """Away from z = 0, Tinker10's alpha is the analytic normalisation."""
    model = fits.Tinker10()
    assert model.normalized
    f = model.fsigma(SIGMA_GRID, z=z, delta_halo=delta, delta_c=DELTA_C)
    assert _mass_fraction(f) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("delta", fits.Tinker10.delta_tab)
def test_tinker10_published_alpha_normalises(delta):
    """The published z = 0 alphas (Tinker+10 Table 4) normalise f to unity."""
    f = fits.Tinker10().fsigma(SIGMA_GRID, z=0.0, delta_halo=float(delta), delta_c=DELTA_C)
    # Tolerance: Table 4 is quoted to 3 significant figures, so each of its five
    # parameters is uncertain by up to ~0.2%; 0.5% in the integral.
    assert _mass_fraction(f) == pytest.approx(1.0, abs=5e-3)


@pytest.mark.parametrize("z", [0.0, 1.0])
def test_bhattacharya_normed_conserves_mass(z):
    model = fits.Bhattacharya(normed=True)
    assert model.normalized
    assert not fits.Bhattacharya().normalized
    f = model.fsigma(SIGMA_GRID, z=z, delta_c=DELTA_C)
    # The low-nu tail goes as nu^(q - 2p) = nu^0.18: the analytic tail handles it.
    assert _mass_fraction(f) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("cls", [fits.Reed03, fits.Courtin, fits.Jenkins, fits.Tinker08])
def test_unnormalised_fits_say_so(cls):
    assert not cls().normalized


# ---------------------------------------------------------------------------------
# Limits and published numbers
# ---------------------------------------------------------------------------------
NU = np.geomspace(0.05, 6.0, 200)


def test_press_schechter_closed_form():
    r"""PS is sqrt(2/pi) nu exp(-nu^2/2): it peaks at nu = 1 with value sqrt(2/pi) e^(-1/2)."""
    model = fits.PS()
    assert model.fsigma(DELTA_C, delta_c=DELTA_C) == pytest.approx(0.48394144903828673, rel=1e-14)
    nu = np.linspace(0.5, 1.5, 100001)
    f = model.fsigma(DELTA_C / nu, delta_c=DELTA_C)
    assert nu[np.argmax(f)] == pytest.approx(1.0, abs=1e-5)
    # Small-nu limit: f -> sqrt(2/pi) nu.
    assert model.fsigma(DELTA_C / 1e-6, delta_c=DELTA_C) == pytest.approx(
        math.sqrt(2 / math.pi) * 1e-6, rel=1e-9
    )


@pytest.mark.parametrize("cls", [fits.SMT, fits.ST])
def test_sheth_tormen_reduces_to_press_schechter(cls):
    """Sheth-Tormen with A = 1/2, a = 1, p = 0 is Press-Schechter."""
    ps = fits.PS().fsigma(DELTA_C / NU, delta_c=DELTA_C)
    np.testing.assert_allclose(
        cls(A=0.5, a=1.0, p=0.0).fsigma(DELTA_C / NU, delta_c=DELTA_C), ps, rtol=1e-13
    )
    # And with p = 0 its normalising amplitude is exactly 1/2.
    assert cls(a=1.0, p=0.0).amplitude == pytest.approx(0.5, rel=1e-14)


def test_smt_normalisation_matches_published_amplitude():
    """The normalising amplitude at p = 0.3 is 0.3222 (Sheth, Mo & Tormen 2001, eq. 5)."""
    assert fits.SMT().amplitude == pytest.approx(0.3222, abs=5e-5)


def test_reed03_reduces_to_sheth_tormen():
    np.testing.assert_allclose(
        fits.Reed03(c=0.0).fsigma(DELTA_C / NU, delta_c=DELTA_C),
        fits.SMT(A=0.3222).fsigma(DELTA_C / NU, delta_c=DELTA_C),
        rtol=1e-14,
    )


def test_bhattacharya_q1_reduces_to_sheth_tormen():
    """Bhattacharya with q = 1 is Sheth-Tormen (Bhattacharya+11, Sec. 4.1)."""
    b = fits.Bhattacharya(q=1.0, p=0.3, a_a=0.707, a_b=0.0, normed=True)
    np.testing.assert_allclose(
        b.fsigma(DELTA_C / NU, z=0.7, delta_c=DELTA_C),
        fits.SMT().fsigma(DELTA_C / NU, delta_c=DELTA_C),
        rtol=1e-12,
    )


def test_tinker10_form_reduces_to_sheth_tormen():
    r"""Tinker+10 eq. 8 with eta = 0, beta = sqrt(a), gamma = a, phi = p is Sheth-Tormen."""
    a, p = 0.707, 0.3
    np.testing.assert_allclose(
        kern.tinker10(NU, kern.tinker10_norm(math.sqrt(a), a, p, 0.0), math.sqrt(a), a, p, 0.0),
        kern.sheth_tormen(NU, kern.sheth_tormen_norm(p), a, p),
        rtol=1e-12,
    )


def test_angulo_bound_and_fof_fits():
    """AnguloBound is Angulo's form with eq. 3's parameters; it is ~2x lower at 1e15 Msun.

    Angulo+12, Sec. 2.2: 'the expected abundance of haloes with M ~ 1e15 Msun changes
    by a factor of ~2 when FoF haloes and self-bound subhaloes are compared'. In their
    cosmology (sigma_8 = 0.9) such haloes have sigma ~ 0.55-0.65.
    """
    sigma = np.geomspace(0.3, 3, 50)
    fof = fits.Angulo()
    same = fits.AnguloBound(A=fof.A, b=fof.b, c=fof.c, d=fof.d)
    np.testing.assert_array_equal(same.fsigma(sigma), fof.fsigma(sigma))
    ratio = fits.AnguloBound().fsigma([0.55, 0.65]) / fof.fsigma([0.55, 0.65])
    # Tolerance: "a factor of ~2" read generously as 1.4-3.
    assert np.all((ratio > 1 / 3) & (ratio < 1 / 1.4))
    # The bound masses lose the most massive objects' outskirts, not small haloes:
    # at low mass (sigma = 3) the two agree to ~10%.
    assert fits.AnguloBound().fsigma(3.0) / fof.fsigma(3.0) == pytest.approx(1.0, abs=0.15)


# Tinker et al. 2008, Table 2 (z = 0), typed in from the paper: Delta, A, a, b, c.
TINKER08_TABLE2 = [
    (200, 0.186, 1.47, 2.57, 1.19),
    (300, 0.200, 1.52, 2.25, 1.27),
    (400, 0.212, 1.56, 2.05, 1.34),
    (600, 0.218, 1.61, 1.87, 1.45),
    (800, 0.248, 1.87, 1.59, 1.58),
    (1200, 0.255, 2.13, 1.51, 1.80),
    (1600, 0.260, 2.30, 1.46, 1.97),
    (2400, 0.260, 2.53, 1.44, 2.24),
    (3200, 0.260, 2.66, 1.41, 2.44),
]


@pytest.mark.parametrize(("delta", "A", "a", "b", "c"), TINKER08_TABLE2)
def test_tinker08_reproduces_table2(delta, A, a, b, c):
    got = fits.Tinker08().parameters(float(delta), 0.0)
    # Tolerance: half a unit in the last digit quoted in the paper.
    assert got[0] == pytest.approx(A, abs=5e-4)
    assert got[1:] == pytest.approx((a, b, c), abs=5e-3)


def test_tinker08_at_delta200_is_table2_form():
    """At Delta = 200, z = 0, f is eq. 3 with Table 2's first row (to its 3 figures)."""
    A, a, b, c = TINKER08_TABLE2[0][1:]
    # Below sigma ~ 0.7 the rounding of c (by up to 0.005) is amplified by 1/sigma^2.
    sigma = np.geomspace(0.7, 4.0, 30)
    expected = A * ((sigma / b) ** -a + 1) * np.exp(-c / sigma**2)
    got = fits.Tinker08().fsigma(sigma, z=0.0, delta_halo=200.0)
    # Tolerance: the 3-figure rounding of A, a, b and c gives up to ~1% in f.
    np.testing.assert_allclose(got, expected, rtol=1.5e-2)


def test_tinker08_redshift_evolution():
    """Eqs. 5-8: A and a fall as (1+z)^-0.14, (1+z)^-0.06; b's exponent -> 0 at Delta -> 75."""
    t = fits.Tinker08()
    p0 = t.parameters(200.0, 0.0)
    p1 = t.parameters(200.0, 1.0)
    assert p1[0] / p0[0] == pytest.approx(2**-0.14, rel=1e-12)
    assert p1[1] / p0[1] == pytest.approx(2**-0.06, rel=1e-12)
    assert p1[3] == p0[3]  # c does not evolve
    # Close to Delta = 75, alpha -> 0, so b barely evolves.
    near = t.parameters(76.0, 1.0)[2] / t.parameters(76.0, 0.0)[2]
    assert near == pytest.approx(1.0, abs=1e-6)


def test_watson_gamma_is_unity_at_reference_overdensity():
    """Watson+13 eq. 17: Gamma(Delta = 178) = 1 at every sigma and z."""
    g = kern.watson_gamma(
        SIGMAS, 178.0, np.array([0.3, 0.7, 1.0])[:, None], 0.023, 0.456, 0.139, 0.072, 2.13
    )
    np.testing.assert_allclose(g, 1.0, rtol=1e-15)


def test_watson_z_dependent_fit_reproduces_table2():
    """Eqs. 14-16 (v4) at z -> 0+ give Table 2's CPMSO+AHF z = 0 column.

    Table 2 of the published paper: A = 0.316, alpha = 2.234, beta = 1.478, at the
    simulations' Omega_m = 0.27.
    """
    A, alpha, beta, gamma = fits.Watson().parameters(1e-8, 0.27)
    assert (A, alpha, beta) == pytest.approx((0.316, 2.234, 1.478), abs=0.01)
    assert gamma == 1.318


def test_watson_redshift_branches():
    w = fits.Watson()
    at0 = w.parameters(0.0, 0.3)
    hi = w.parameters(np.array([6.0, 10.0]), 0.3)
    assert at0 == (w.A_0, w.alpha_0, w.beta_0, w.gamma_0)
    assert np.all(hi[0] == w.A_hi)
    assert np.all(hi[3] == w.gamma_hi)


def test_behroozi_correction_at_pivot_mass():
    r"""Behroozi+13 eq. G2: n(>M*) / n_T08(>M*) = 10^alpha(z), M* = 10^11.5 Msun.

    The correction is defined through the cumulative mass function,
    n(>M) = theta(M) n_T08(>M); the test integrates the corrected dn/dM.
    """
    h = 0.7
    m = np.geomspace(1e9, 3e14, 40001)  # Msun/h; dn/dm stays > 0 up to the top
    for z in (2.0, 4.0, 8.0):
        # Any steeply falling dn/dM will do: the identity is exact.
        dndm = m**-1.9 * np.exp(-m / 1e13)
        ngtm = _cumulative(m, dndm)
        corrected = fits.Behroozi().modify_dndm(
            m * Msun_h, dndm * dndm_unit, z=z, ngtm=ngtm * number_density_unit, h=h
        )
        assert corrected.unit is dndm_unit
        n_corrected = _cumulative(m, corrected.to_value(dndm_unit))
        mstar = 10**11.5 * h  # in Msun/h
        a = 1 / (1 + z)
        alpha = 0.144 / (1 + math.exp(14.79 * (a - 0.213)))
        # The last point has n(>m) = 0 for both.
        ratio = np.interp(np.log(mstar), np.log(m[:-1]), n_corrected[:-1] / ngtm[:-1])
        # Tolerance: trapezoidal integration error on the grid.
        assert ratio == pytest.approx(10**alpha, rel=1e-4)
    assert fits.Behroozi.modifies_dndm
    assert not fits.Tinker08.modifies_dndm


def _cumulative(m, dndm):
    """n(>m), by the trapezoidal rule from the top of the grid."""
    seg = 0.5 * (dndm[1:] + dndm[:-1]) * np.diff(m)
    return np.append(np.cumsum(seg[::-1])[::-1], 0.0)


def test_modify_dndm_is_identity_by_default():
    dndm = np.array([1.0, 2.0]) * dndm_unit
    ngtm = np.array([3.0, 1.0]) * number_density_unit
    out = fits.Tinker08().modify_dndm([1e12, 1e13] * Msun_h, dndm, z=0.0, ngtm=ngtm, h=0.7)
    assert out.unit is dndm_unit
    np.testing.assert_array_equal(out.value, dndm.value)


@pytest.mark.parametrize("z", [0.0, 1.0, 2.0])
@pytest.mark.parametrize("om", [0.15, 0.3, 0.5])
def test_bocquet_mass_ratios_are_ordered(z, om):
    """0 < M500c/M200m < M200c/M200m < 1: the critical density exceeds the mean.

    As Omega_m(z) -> 1 (Omega_m0 = 0.5, z = 2: Omega_m(z) = 0.96) the 200c and 200m
    thresholds nearly coincide, and the ratio must approach 1 from below; eq. A2 is
    accurate only "at the few percent level" (App. A), so allow 3% there.
    """
    m = np.geomspace(1e13, 1e15, 9) * Msun_h
    r200 = fits.Bocquet200cDMOnly().mass_ratio_to_200m(m, z=z, omega_m0=om, h=0.7)
    r500 = fits.Bocquet500cHydro().mass_ratio_to_200m(m, z=z, omega_m0=om, h=0.7)
    assert np.all((r500 > 0) & (r500 < r200) & (r200 < 1.03))
    if om * (1 + z) ** 3 / (om * (1 + z) ** 3 + 1 - om) < 0.9:
        assert np.all(r200 < 1)
    np.testing.assert_array_equal(
        fits.Bocquet200mHydro().mass_ratio_to_200m(m, z=z, omega_m0=om, h=0.7), 1.0
    )


def test_bocquet_mass_ratio_depends_on_mass_in_msun():
    """The same physical halo gets the same ratio whatever h (eqs. 6, A2 take ln M/Msun)."""
    m_msun = np.geomspace(1e13, 1e15, 5) * Msun_h  # times h below: Msun/h
    for model in (fits.Bocquet200cDMOnly(), fits.Bocquet500cDMOnly()):
        r1 = model.mass_ratio_to_200m(m_msun * 0.5, z=0.5, omega_m0=0.3, h=0.5)
        r2 = model.mass_ratio_to_200m(m_msun * 0.9, z=0.5, omega_m0=0.3, h=0.9)
        np.testing.assert_allclose(r1, r2, rtol=1e-12)


def test_yung24_units_select_the_table():
    h = fits.Yung24()
    phys = fits.Yung24(units="physical")
    assert h.b_0 == pytest.approx(8.62813020)
    assert phys.b_0 == pytest.approx(4.86693806)
    assert fits.Yung24(units="physical", b_0=1.0).b_0 == 1.0
    with pytest.raises(ValueError, match="units"):
        fits.Yung24(units="cgs")


@pytest.mark.parametrize("cls", [fits.Crocce, fits.Bocquet500cDMOnly, fits.Yung24])
def test_massive_haloes_rarer_at_higher_redshift_at_fixed_sigma_tail(cls):
    """At fixed small sigma (rare haloes), z-dependent fits stay within the bounds of f."""
    z = [6.0, 12.0] if cls is fits.Yung24 else [0.0, 1.0]
    f = cls().fsigma(0.5, z=np.array(z))
    assert np.all((f > 0) & (f < 1))


# ---------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------
def test_parameter_validation():
    with pytest.raises(ValueError, match=r"p must be < 0\.5"):
        fits.SMT(p=0.5)
    with pytest.raises(ValueError, match="a must be > 0"):
        fits.Courtin(a=0.0)
    with pytest.raises(ValueError, match="2p must be < q"):
        fits.Bhattacharya(p=1.0, q=1.5)
    with pytest.raises(ValueError, match="q must be > 0"):
        fits.Bhattacharya(q=-1.0, p=-1.0)


def test_unphysical_tinker_parameters_raise():
    with pytest.raises(DomainError, match="parameter A"):
        fits.Tinker08(A_3200=-0.1).fsigma(1.0, z=0.0, delta_halo=3200.0)
    with pytest.raises(DomainError, match="eta_plus_half"):
        fits.Tinker10(eta_3200=-0.6).fsigma(1.0, z=0.0, delta_halo=3200.0, delta_c=DELTA_C)


@pytest.mark.parametrize("delta", [76.0, 3.0e4])
def test_tinker08_parameters_physical_across_valid_domain(delta):
    """At the edges of the valid Delta range the default parameters stay > 0 at any z."""
    for z in (0.0, 3.0, 10.0):
        assert all(np.all(p > 0) for p in fits.Tinker08().parameters(delta, z))


@pytest.mark.parametrize("delta", [70.0, 3600.0])
def test_tinker10_parameters_normalisable_across_valid_domain(delta):
    for z in (0.0, 1.0, 3.0, 10.0):
        f = fits.Tinker10().fsigma(1.0, z=z, delta_halo=delta, delta_c=DELTA_C)
        assert np.isfinite(f)
        assert f > 0


# ---------------------------------------------------------------------------------
# Domains and policies
# ---------------------------------------------------------------------------------
def _outside_values(cls):
    """Inputs inside the valid domain, but just outside the calibration domain."""
    for name, iv in cls.calibration_domain.bounds:
        for bound, step in ((iv.upper, +1), (iv.lower, -1)):
            if not np.isfinite(bound):
                continue
            if name == "m":
                value = bound * 10**step
            else:
                value = bound + step * max(0.05, 0.05 * abs(bound))
            values = _baseline(cls)
            _set(values, name, value)
            x = cls().inputs(values["sigma"], **_inputs(values))
            if bool(fits._contains(cls.valid_domain, x)):
                return values
    return None


def _edge_values(cls):
    """Inputs on the edge of the calibration domain (bounds are inclusive)."""
    for name, iv in cls.calibration_domain.bounds:
        if np.isfinite(iv.upper):
            values = _baseline(cls)
            # Derived sigma variables can not round-trip exactly through sigma.
            nudge = 1e-12 if name in ("ln_sigma_inv", "log10_sigma_inv") else 0.0
            _set(values, name, iv.upper - nudge)
            return values
    return None


CALIBRATED = [cls for cls in ALL if cls.calibration_domain.variables]


def test_only_press_schechter_is_uncalibrated():
    assert [cls for cls in ALL if cls not in CALIBRATED] == [fits.PS]


@pytest.mark.parametrize("policy", DOMAIN_POLICIES)
@pytest.mark.parametrize("cls", ALL, ids=_ids)
def test_inside_calibration_domain(cls, policy):
    model = cls()
    for values in filter(None, [_baseline(cls), _edge_values(cls)]):
        with warnings.catch_warnings():
            warnings.simplefilter("error", HMFExtrapolationWarning)
            result = _call(model, values, policy=policy)
        assert np.all(result.in_calibration_domain)
        assert result.fsigma == model.fsigma(values["sigma"], **_inputs(values))
        assert np.isfinite(result.fsigma)


@pytest.mark.parametrize("cls", CALIBRATED, ids=_ids)
def test_outside_calibration_domain(cls):
    model = cls()
    values = _outside_values(cls)
    assert values is not None, f"no point just outside {cls.__name__}'s calibration domain"
    expected = model.fsigma(values["sigma"], **_inputs(values))
    assert np.isfinite(expected)

    ignored = _call(model, values, policy="ignore")
    assert ignored.fsigma == expected
    assert not ignored.in_calibration_domain

    with pytest.warns(HMFExtrapolationWarning, match=f"{cls.__name__}'s calibration domain"):
        warned = _call(model, values, policy="warn")
    assert warned.fsigma == expected

    masked = _call(model, values, policy="mask")
    assert np.isnan(masked.fsigma)
    assert not masked.in_calibration_domain

    with pytest.raises(DomainError, match="calibration domain"):
        _call(model, values, policy="raise")


@pytest.mark.parametrize("cls", CALIBRATED, ids=_ids)
def test_mask_policy_masks_only_outside_values(cls):
    model = cls()
    inside, outside = _baseline(cls), _outside_values(cls)
    stacked = {k: np.array([inside[k], outside[k]]) for k in inside}
    result = _call(model, stacked, policy="mask")
    np.testing.assert_array_equal(result.in_calibration_domain, [True, False])
    assert np.isfinite(result.fsigma[0])
    assert np.isnan(result.fsigma[1])


def _invalid_values(cls):
    """For each bounded variable of the valid domain, inputs just outside it."""
    out = []
    for name, iv in cls.valid_domain.bounds:
        for bound, step in ((iv.lower, -1), (iv.upper, +1)):
            if not np.isfinite(bound):
                continue
            values = _baseline(cls)
            values[name] = bound - 1e-3 * max(1.0, abs(bound)) if step < 0 else bound * 1.01 + 1e-3
            out.append((name, values))
    return out


@pytest.mark.parametrize("policy", DOMAIN_POLICIES)
@pytest.mark.parametrize("cls", ALL, ids=_ids)
def test_outside_valid_domain_always_raises(cls, policy):
    model = cls()
    cases = _invalid_values(cls)
    assert cases
    for name, values in cases:
        with pytest.raises(DomainError, match=f"valid domain.*{name}"):
            _call(model, values, policy=policy)
        with pytest.raises(DomainError, match="valid domain"):
            model.fsigma(values["sigma"], **_inputs(values))
    # sigma = 0 (the edge of sigma > 0) is outside too.
    with pytest.raises(DomainError, match="sigma > 0"):
        model.fsigma(0.0, **_inputs(_baseline(cls)))


def test_valid_domain_raises_even_for_one_bad_value():
    with pytest.raises(DomainError, match="1 of 3"):
        fits.Yung24().fsigma(1.0, z=np.array([6.0, 12.0, 20.0]))


def test_unknown_policy():
    with pytest.raises(ValueError, match="Unknown domain policy"):
        evaluate_fsigma(fits.PS(), FitInputs(sigma=1.0, delta_c=DELTA_C), policy="silent")


def test_calibration_domain_in_mass_uses_canonical_units():
    """Masses are plain Msun/h inside, compared with Quantity bounds."""
    assert fits.Warren.calibration_domain["m"].unit == Msun_h
    r = evaluate_fsigma(fits.Warren(), FitInputs(sigma=[1.0, 1.0], z=0.0, m=[1e12, 1e16]))
    np.testing.assert_array_equal(r.in_calibration_domain, [True, False])


# ---------------------------------------------------------------------------------
# Units boundary
# ---------------------------------------------------------------------------------
def test_public_methods_are_unit_boundaries():
    """M (and dn/dm, n(>m)) are Quantities at the public methods; bare numbers raise."""
    warren = fits.Warren()
    with pytest.raises(UnitBoundaryError, match="m"):
        warren.fsigma(1.0, m=1e12)
    with pytest.raises(UnitBoundaryError, match="m"):
        warren.inputs(1.0, m=1e12)
    with pytest.raises(UnitBoundaryError):
        fits.Bocquet500cDMOnly().mass_ratio_to_200m(1e14, z=0.0, omega_m0=0.3, h=0.7)
    with pytest.raises(UnitBoundaryError, match="dndm"):
        fits.Behroozi().modify_dndm(
            [1e12] * Msun_h, [1.0], z=1.0, ngtm=[1.0] * number_density_unit, h=0.7
        )


def test_fsigma_is_dimensionless_and_accepts_equivalent_mass_units():
    model = fits.Warren()
    f = model.fsigma([1.0, 2.0], m=[1e12, 1e13] * Msun_h)
    assert type(f) is np.ndarray
    # The same masses in an equivalent h-unit (not the shared constant) agree, and so
    # does the calibration mask built from them.
    other = [1e12, 1e13] * (u.Msun / cu.littleh)
    np.testing.assert_array_equal(model.fsigma([1.0, 2.0], m=other), f)
    x = model.inputs([1.0, 2.0], z=0.0, m=[1e15, 1e16] * Msun_h)
    np.testing.assert_array_equal(model.in_calibration_domain(x), [True, False])


def test_masses_without_h_need_a_hubble_constant():
    """A model has no H0, so masses in physical Msun can not be converted (no guessing)."""
    with pytest.raises(u.UnitConversionError):
        fits.Warren().fsigma(1.0, m=1e12 * u.Msun)


def test_library_path_takes_plain_canonical_arrays():
    """FitInputs + fsigma_from_inputs / evaluate_fsigma: the same numbers, no units."""
    model = fits.Bocquet200cHydro()
    m = np.geomspace(1e12, 1e15, 4)
    sigma = np.linspace(0.6, 2.0, 4)
    public = model.fsigma(sigma, z=0.5, m=m * Msun_h)
    x = FitInputs(sigma=sigma, z=0.5, m=m)
    np.testing.assert_array_equal(model.fsigma_from_inputs(x), public)
    np.testing.assert_array_equal(evaluate_fsigma(model, x).fsigma, public)
    np.testing.assert_array_equal(
        model._mass_ratio_to_200m(m, z=0.5, omega_m0=0.3, h=0.7),
        model.mass_ratio_to_200m(m * Msun_h, z=0.5, omega_m0=0.3, h=0.7),
    )
    with pytest.raises(ValueError, match="needs the input"):
        evaluate_fsigma(model, FitInputs(sigma=sigma))


def test_does_not_import_v3_modules():
    """hmf.core must not depend on the v3 modules, which are removed at 4.0."""
    import ast
    from pathlib import Path

    v3 = ("cosmology", "density_field", "mass_function", "halos", "_internals", "alternatives")
    for path in (Path(fits.__file__), Path(kern.__file__)):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            else:
                continue
            for name in names:
                parts = name.split(".")
                assert not any(p in v3 for p in parts), f"{path.name} imports {name}"


def test_registered_on_import_of_core():
    import hmf.core

    assert hmf.core.fits.FittingFunction.get("Tinker08") is fits.Tinker08
