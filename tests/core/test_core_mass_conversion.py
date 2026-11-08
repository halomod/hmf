"""Tests of the profile mass conversion (NFW), its kernels, and converted mass functions."""

import copy
import math
import pickle
import subprocess
import sys
import textwrap
import warnings

import attrs
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM, Planck18
from scipy.integrate import simpson

from hmf.core import _references as refs
from hmf.core import fits
from hmf.core._kernels import mass_conversion as kern
from hmf.core.concentration import ConcentrationModel, Duffy08
from hmf.core.domain import DomainError
from hmf.core.mass_conversion import MassConversion, ProfileConversion
from hmf.core.mass_definitions import (
    FOF,
    SelfBound,
    SOCritical,
    SOGeneric,
    SOMean,
    SOVirial,
)
from hmf.core.mass_function import MassFunction
from hmf.core.profiles import NFW
from hmf.core.species import omega_m
from hmf.core.units import Msun_h, UnitBoundaryError
from hmf.exceptions import HMFExtrapolationWarning

M = np.geomspace(1e10, 1e15, 26)
OMEGA_M0 = omega_m(Planck18, 0.0, "cb")

SO_DEFINITIONS = [
    SOMean(overdensity=200),
    SOCritical(overdensity=200),
    SOVirial(),
    SOCritical(overdensity=500),
]


def _ln_q(target, source, om=OMEGA_M0):
    return np.log(target.delta_halo_mean_kernel(omega_m_z=om)) - np.log(
        source.delta_halo_mean_kernel(omega_m_z=om)
    )


def _ln_mu(x):
    """The NFW ln mu(x), through the profile."""
    return NFW().ln_mu_and_slope_kernel(x=x)[0]


def _ln_mean_density(x):
    """Ln F(x) = ln[mu(x) / x^3], the NFW mean enclosed density over 3 rho_s."""
    return _ln_mu(x) - 3 * np.log(x)


def _solve(c, ln_q):
    """The NFW concentration c' with F(c') = q F(c), through the profile."""
    return NFW().convert_concentration_kernel(c=c, slope=0.0, ln_q=ln_q)[1]


def _nfw_convert(c, slope, ln_q):
    return NFW().convert_concentration_kernel(c=c, slope=slope, ln_q=ln_q)


# ---------------------------------------------------------------------------------
# The NFW profile's kernels
# ---------------------------------------------------------------------------------


def test_ln_mu_closed_form_and_series():
    """mu(1) = ln 2 - 1/2; mu ~ x^2 / 2 as x -> 0; the series joins the closed form."""
    assert _ln_mu(1.0) == pytest.approx(math.log(math.log(2) - 0.5), rel=1e-15)
    x = np.array([1e-8, 1e-6, 1e-4])
    np.testing.assert_allclose(
        np.exp(_ln_mu(x)), x**2 / 2 * (1 - 4 * x / 3 + 1.5 * x**2), rtol=1e-11
    )
    below, above = np.nextafter(0.01, 0), 0.01
    assert _ln_mu(below) == pytest.approx(float(_ln_mu(above)), abs=1e-13)
    # psi = dln mu / dln x, from 2 at small x to 0 at large x.
    x = np.geomspace(1e-6, 1e6, 50)
    psi = NFW().ln_mu_and_slope_kernel(x=x)[1]
    assert psi[0] == pytest.approx(2.0, rel=1e-5)
    assert np.all(np.diff(psi) < 0)
    assert psi[-1] < 0.1
    h = 1e-6
    numeric = (_ln_mu(x * math.exp(h)) - _ln_mu(x * math.exp(-h))) / (2 * h)
    np.testing.assert_allclose(psi, numeric, rtol=1e-7, atol=1e-9)


C_GRID = np.geomspace(1e-3, 1e4, 120)[:, None]
LN_Q_GRID = np.linspace(math.log(1e-4), math.log(1e4), 81)[None, :]


def test_root_solve_reaches_round_off_over_a_wide_range():
    c_new = _solve(C_GRID, LN_Q_GRID)
    residual = _ln_mean_density(c_new) - _ln_mean_density(C_GRID) - LN_Q_GRID
    assert np.all(np.isfinite(c_new))
    assert np.abs(residual).max() < 2e-13
    # A denser definition has a smaller concentration (F decreases monotonically).
    assert np.all(np.diff(c_new, axis=1) < 0)


def test_root_solve_is_elementwise_and_exact_for_the_same_density():
    c = np.geomspace(0.5, 50, 101)
    ln_q = np.linspace(-2, 2, 101)
    full = _nfw_convert(c, -0.1, ln_q)
    for i in (0, 17, 100):
        part = _nfw_convert(c[i : i + 1], -0.1, ln_q[i : i + 1])
        for got, expected in zip(part, full, strict=True):
            np.testing.assert_array_equal(got, expected[i : i + 1])
    np.testing.assert_array_equal(_solve(c, 0.0), c)
    ln_ratio, c_new, dln_m, _ = _nfw_convert(c, -0.1, 0.0)
    np.testing.assert_array_equal(ln_ratio, 0.0)
    np.testing.assert_array_equal(c_new, c)
    np.testing.assert_array_equal(dln_m, 1.0)


def test_root_solve_agrees_with_v3_brentq():
    """Regression: v3's _find_new_concentration (one brentq per mass), elementwise."""
    from hmf.halos.mass_definitions import _find_new_concentration

    c = np.geomspace(0.5, 60, 25)
    for ln_q in (math.log(0.3), math.log(200 / 333), math.log(1600 / 200), math.log(5.0)):
        rho_s = c**3 / (3 * np.exp(_ln_mu(c)))  # for a halo density of 1
        v3 = np.array(
            [
                # A guess within v3's first bracket (x5) of the root: q^(-1/3) c.
                _find_new_concentration(r, math.exp(ln_q), x_guess=cc * math.exp(-ln_q / 3))
                for r, cc in zip(rho_s, c, strict=True)
            ]
        )
        np.testing.assert_allclose(_solve(c, ln_q), v3, rtol=1e-10)


def _hu_kravtsov_mass_error(c, ratio):
    """|M_HK / M - 1| for the mass ratio M_h / M_v = (D_h / D_v) (c_h / c_v)^3 (eq. C10)."""
    ln_q = np.log(ratio)
    c_hk = kern.hu_kravtsov_inverse(kern.nfw_ln_mean_density(c) + ln_q)
    exact = np.exp(_ln_mu(_solve(c, ln_q)) - _ln_mu(c))
    return np.abs(ratio * (c_hk / c) ** 3 / exact - 1)


def test_hu_kravtsov_approximation_to_its_quoted_accuracy():
    """The approximation of Hu & Kravtsov (2003, eq. C11) is as accurate as they quote.

    Masses converted by it are typically good to 0.3% for galaxy and cluster
    concentrations (c < 20), and to about 1% for 0 < c < 400 (App. C).

    The exact solve agrees: below 0.4% (median 0.3%) for 4 < c < 20 and density ratios
    0.5-2, and below 1.25% for 0.1 < c < 400, the worst case (1.22%) being near c ~ 1.
    """
    ratio = np.geomspace(0.5, 2.0, 21)[None, :]
    typical = _hu_kravtsov_mass_error(np.geomspace(4, 20, 50)[:, None], ratio)
    assert typical.max() < 4e-3
    assert np.median(typical) < 3e-3
    wide = _hu_kravtsov_mass_error(np.geomspace(0.1, 400, 200)[:, None], ratio)
    assert wide.max() < 1.25e-2
    # Exact as c -> 0 (f -> infinity), by construction.
    assert _hu_kravtsov_mass_error(np.array([[1e-4]]), ratio).max() < 1e-3


def test_jacobian_is_the_derivative_of_the_mapping():
    c = np.geomspace(1, 30, 20)
    slope = -0.1
    ln_q = math.log(2.5)
    _, _, dln_m, dln_c = _nfw_convert(c, slope, ln_q)
    h = 1e-5
    # c(M) = c0 (M / M0)^slope: a step h in ln M is a step slope h in ln c.
    lo = _nfw_convert(c * math.exp(-slope * h), slope, ln_q)
    hi = _nfw_convert(c * math.exp(slope * h), slope, ln_q)
    np.testing.assert_allclose(dln_m, 1 + (hi[0] - lo[0]) / (2 * h), rtol=1e-8)
    np.testing.assert_allclose(dln_c, (np.log(hi[1]) - np.log(lo[1])) / (2 * h), rtol=1e-7)


# ---------------------------------------------------------------------------------
# The ProfileConversion model
# ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def nfw():
    return ProfileConversion()


def test_registered_and_cited(nfw):
    assert MassConversion.get("profile") is ProfileConversion
    assert nfw.profile == NFW()
    assert nfw.concentration == Duffy08()
    assert ProfileConversion(profile="NFW", concentration="Duffy08") == nfw
    assert refs.HU03 in NFW.references
    assert pickle.loads(pickle.dumps(nfw)) == nfw
    assert hash(nfw) == hash(ProfileConversion())


@pytest.mark.parametrize("definition", [*SO_DEFINITIONS, FOF()], ids=str)
def test_converting_to_the_same_definition_is_the_identity(nfw, definition):
    out = nfw.convert_kernel(m=M, z=0.5, omega_m_z=0.6, source=definition, target=definition)
    np.testing.assert_array_equal(out.m, M)
    np.testing.assert_array_equal(out.dm_dm, 1.0)


def test_virial_is_so_mean_18pi2_in_einstein_de_sitter(nfw):
    """Omega_m(z) = 1: the same halo density, so the conversion is exactly the identity."""
    eds = SOMean(overdensity=18 * math.pi**2)
    for source, target in ((SOVirial(), eds), (eds, SOVirial())):
        out = nfw.convert_kernel(m=M, z=1.0, omega_m_z=1.0, source=source, target=target)
        np.testing.assert_array_equal(out.m, M)
        np.testing.assert_array_equal(out.dm_dm, 1.0)


@pytest.mark.parametrize("concentration", [Duffy08(), Duffy08(definition="200m")], ids=str)
@pytest.mark.parametrize(
    ("a", "b"),
    [(a, b) for i, a in enumerate(SO_DEFINITIONS) for b in SO_DEFINITIONS[i + 1 :]],
    ids=str,
)
def test_round_trip(a, b, concentration):
    """A -> B -> A returns the masses to 1e-13, and the Jacobians are reciprocal."""
    conv = ProfileConversion(concentration=concentration)
    z, om = 0.7, float(omega_m(Planck18, 0.7))
    there = conv.convert_kernel(m=M, z=z, omega_m_z=om, source=a, target=b)
    back = conv.convert_kernel(m=there.m, z=z, omega_m_z=om, source=b, target=a)
    np.testing.assert_allclose(back.m, M, rtol=1e-13)
    np.testing.assert_allclose(there.dm_dm * back.dm_dm, 1.0, rtol=1e-12)


@pytest.mark.parametrize(
    ("source", "concentration"),
    [
        (SOMean(), Duffy08(definition="200m")),  # the relation's own definition
        (SOMean(), Duffy08()),  # converted from 200c first
        (SOCritical(overdensity=500), Duffy08(definition="vir")),
    ],
    ids=str,
)
def test_jacobian_matches_a_finite_difference(source, concentration):
    conv = ProfileConversion(concentration=concentration)
    target = SOVirial()
    out = conv.convert_kernel(m=M, z=0.3, omega_m_z=0.4, source=source, target=target)
    h = 1e-5
    hi = conv.convert_kernel(m=M * math.exp(h), z=0.3, omega_m_z=0.4, source=source, target=target)
    lo = conv.convert_kernel(m=M * math.exp(-h), z=0.3, omega_m_z=0.4, source=source, target=target)
    numeric = (np.log(hi.m) - np.log(lo.m)) / (2 * h) * out.m / M
    np.testing.assert_allclose(out.dm_dm, numeric, rtol=1e-8)


def test_concentration_is_converted_to_the_source_definition():
    """A halo converted to the relation's definition has the relation's concentration."""
    conv = ProfileConversion(concentration=Duffy08())  # in 200c
    z, om = 0.5, 0.55
    out = conv.convert_kernel(m=M, z=z, omega_m_z=om, source=SOMean(), target=SOCritical())
    # Through the kernel: c in 200m from c in 200c, at the 200c masses found.
    ln_c, slope = Duffy08().ln_concentration_and_slope_kernel(m=out.m, z=z)
    ln_ratio, _, _, _ = _nfw_convert(np.exp(ln_c), slope, _ln_q(SOMean(), SOCritical(), om))
    np.testing.assert_allclose(out.m * np.exp(ln_ratio), M, rtol=1e-13)


def test_concentration_and_mass_ordering(nfw):
    """At fixed halo, a denser definition has a smaller radius, concentration and mass.

    At z = 0 for Planck18 the mean-density overdensities are 200 (200m) < 333 (vir) <
    645 (200c) < 1613 (500c), so c_500c < c_200c < c_vir < c_200m and
    M_500c < M_200c < M_vir < M_200m.
    """
    om = OMEGA_M0
    ln_c, slope = Duffy08().ln_concentration_and_slope_kernel(m=M, z=0.0)
    order = [SOCritical(overdensity=500), SOCritical(), SOVirial(), SOMean()]
    concentrations, masses = [], []
    for target in order:
        ln_ratio, c_new, _, _ = _nfw_convert(np.exp(ln_c), slope, _ln_q(target, SOCritical()))
        concentrations.append(c_new)
        masses.append(M * np.exp(ln_ratio))
        np.testing.assert_allclose(
            nfw.convert_kernel(m=M, z=0.0, omega_m_z=om, source=SOCritical(), target=target).m,
            masses[-1],
            rtol=1e-14,
        )
    assert np.all(np.diff(np.array(concentrations), axis=0) > 0)
    assert np.all(np.diff(np.array(masses), axis=0) > 0)
    # Typical numbers: M_500c / M_200c ~ 0.65-0.8, M_200m / M_200c ~ 1.1-1.5.
    assert np.all((masses[0] / M > 0.6) & (masses[0] / M < 0.82))
    assert np.all((masses[3] / M > 1.1) & (masses[3] / M < 1.5))


@attrs.frozen(kw_only=True)
class _ConstantConcentration(ConcentrationModel, abstract=True):
    """c = 5 in 200m at every mass and z, without a calibration domain (a user's model)."""

    @property
    def mass_definition(self):
        return SOMean()

    def _ln_concentration_and_slope(self, m, z):
        ln_c = np.full(np.broadcast_shapes(m.shape, z.shape), math.log(5.0))
        return ln_c, np.zeros_like(ln_c)


def test_a_user_concentration_model():
    """With a constant c the mapping is a constant ratio mu(c') / mu(c), and dM'/dM it."""
    conv = ProfileConversion(concentration=_ConstantConcentration())
    out = conv.convert_kernel(m=M, z=0.0, omega_m_z=0.3, source=SOMean(), target=SOCritical())
    c_new = _solve(5.0, _ln_q(SOCritical(), SOMean(), 0.3))
    ratio = math.exp(_ln_mu(c_new) - _ln_mu(5.0))
    np.testing.assert_allclose(out.m / M, ratio, rtol=1e-14)
    np.testing.assert_allclose(out.dm_dm, ratio, rtol=1e-14)
    # No calibration domain: everything is inside, and no policy applies to it.
    assert out.in_calibration_domain.all()
    assert out.calibration_description == ""
    mf = MassFunction.build(
        **{"transfer.model": "EH"},
        fit="SMT",
        mass_definition="200c",
        mass_conversion=conv,
        domain_policy="raise",
    )
    assert np.all(np.isfinite(mf.dndm(m=M[10:20] * Msun_h, z=0.5)))


def test_fof_converts_through_its_overdensity(nfw):
    """FoF(b=0.2) is ~179 times the mean, so a little bigger than 200m: a few per cent."""
    out = nfw.convert_kernel(m=M, z=0.0, omega_m_z=OMEGA_M0, source=FOF(), target=SOMean())
    assert np.all((out.m / M < 1.0) & (out.m / M > 0.95))


def test_definitions_without_an_overdensity_raise(nfw):
    for bad in (SelfBound(), SOGeneric(preferred=SOMean())):
        with pytest.raises(ValueError, match="Can't convert masses"):
            nfw.convert_kernel(m=M, z=0.0, omega_m_z=0.3, source=bad, target=SOMean())
        with pytest.raises(ValueError, match="Can't convert masses"):
            nfw.check_definitions(SOMean(), bad)


def test_inputs_are_checked(nfw):
    args = {"source": SOMean(), "target": SOCritical()}
    with pytest.raises(DomainError, match="m in"):
        nfw.convert_kernel(m=[1e12, 0.0], z=0.0, omega_m_z=0.3, **args)
    with pytest.raises(DomainError, match="z in"):
        nfw.convert_kernel(m=1e12, z=-1.0, omega_m_z=0.3, **args)
    with pytest.raises(DomainError, match="omega_m_z"):
        nfw.convert_kernel(m=1e12, z=0.0, omega_m_z=np.nan, **args)


def test_calibration_domain_is_reported(nfw):
    out = nfw.convert_kernel(
        m=np.array([1e9, 1e12, 1e16]), z=0.0, omega_m_z=0.3, source=SOCritical(), target=SOMean()
    )
    np.testing.assert_array_equal(out.in_calibration_domain, [False, True, False])
    assert "Duffy08's calibration domain" in out.calibration_description
    high_z = nfw.convert_kernel(m=1e12, z=3.0, omega_m_z=0.9, source=SOMean(), target=SOVirial())
    assert not high_z.in_calibration_domain


def test_public_convert(nfw):
    m, jac = nfw.convert(
        m=1e12 * Msun_h, z=0.0, omega_m_z=0.3, source=SOMean(), target=SOCritical()
    )
    assert m.unit == Msun_h
    assert m.shape == ()
    assert isinstance(jac, float)
    with pytest.raises(UnitBoundaryError):
        nfw.convert(m=1e12, z=0.0, omega_m_z=0.3, source=SOMean(), target=SOCritical())
    with pytest.raises(TypeError, match="positional"):
        nfw.convert(1e12 * Msun_h, z=0.0, omega_m_z=0.3, source=SOMean(), target=SOCritical())
    with pytest.raises(TypeError, match="positional"):
        nfw.convert_kernel(1e12, z=0.0, omega_m_z=0.3, source=SOMean(), target=SOCritical())


# ---------------------------------------------------------------------------------
# Converted mass functions
# ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def smt():
    """SMT, measured in SO-virial, with EH."""
    return MassFunction.build(**{"transfer.model": "EH"}, fit="SMT")


@pytest.fixture(scope="module")
def smt200(smt):
    return smt.evolve(mass_definition="200m", mass_conversion="profile")


def test_default_mass_function_does_not_convert(smt):
    assert smt.mass_definition is None
    assert smt.mass_conversion is None
    assert not smt.converts_masses
    assert smt.resolved_mass_definition == SOVirial()
    assert smt.fit_mass_definition == SOVirial()
    tinker = smt.evolve(fit="Tinker08")
    assert tinker.resolved_mass_definition == SOMean()
    assert tinker.fit_mass_definition == SOMean()


def test_accepted_definition_is_evaluated_directly(smt):
    """Tinker08 (SOGeneric) in 200c: no conversion, with the overdensity 200 / Omega_m(z)."""
    mf = smt.evolve(fit="Tinker08", mass_definition=SOCritical())
    assert not mf.converts_masses
    assert mf.fit_mass_definition == SOCritical()
    z = 0.5
    sigma = mf.sigma(m=M * Msun_h, z=z)
    om = omega_m(Planck18, z, "cb")
    expected = fits.Tinker08().fsigma(sigma, z=z, delta_halo=200 / om)
    np.testing.assert_allclose(mf.fsigma(m=M * Msun_h, z=z), expected, rtol=1e-14)
    # The same, with a conversion given but not needed.
    with_conversion = mf.evolve(mass_conversion="profile")
    np.testing.assert_array_equal(
        with_conversion.dndm(m=M * Msun_h, z=z), mf.dndm(m=M * Msun_h, z=z)
    )
    # Its own definition, asked for explicitly, is the default.
    np.testing.assert_array_equal(
        smt.evolve(mass_definition=SOVirial()).dndm(m=M * Msun_h, z=z),
        smt.dndm(m=M * Msun_h, z=z),
    )


def test_mismatched_definition_needs_a_conversion(smt):
    with pytest.raises(ValueError, match=r"mass_conversion=ProfileConversion\(\)") as info:
        smt.evolve(mass_definition="200m")
    assert "SMT was measured in vir, which does not accept mass_definition=200m" in str(info.value)
    with pytest.raises(ValueError, match="can't be an SOGeneric"):
        smt.evolve(mass_definition=SOGeneric(preferred=SOMean()), mass_conversion="profile")
    with pytest.raises(ValueError, match="self-bound masses have no overdensity"):
        smt.evolve(fit="AnguloBound", mass_definition="200m", mass_conversion="profile")
    with pytest.raises(ValueError, match="self-bound masses have no overdensity"):
        smt.evolve(mass_definition=SelfBound(), mass_conversion="profile")


@pytest.mark.parametrize("z", [0.0, 1.0])
@pytest.mark.parametrize("target", [SOMean(overdensity=200), SOMean(overdensity=1600)], ids=str)
def test_halo_number_is_conserved(smt, z, target):
    """n_new(>M) = n_meas(>M_meas(M)), with n_new integrated from the converted dn/dm.

    The test of #362: the converted dn/dm, integrated numerically (Simpson in ln m on
    a fine grid), counts the same haloes as the measured cumulative function above the
    corresponding masses. It checks dn/dm at M' and the Jacobian together.
    """
    mf = smt.evolve(mass_definition=target, mass_conversion="profile")
    m_lo = np.geomspace(1e10, 3e15, 12)
    m_top = 1e16
    om = omega_m(Planck18, z, "cb")
    m_meas_top = mf.mass_conversion.convert_kernel(
        m=m_top, z=z, omega_m_z=om, source=target, target=SOVirial()
    ).m
    for m0 in m_lo:
        grid = np.geomspace(m0, m_top, 801)
        dndlnm = grid * mf.dndm(m=grid * Msun_h, z=z).value
        counted = simpson(dndlnm, x=np.log(grid))
        m_meas = mf.mass_conversion.convert_kernel(
            m=m0, z=z, omega_m_z=om, source=target, target=SOVirial()
        ).m
        expected = (
            smt.ngtm(m=m_meas * Msun_h, z=z).value - smt.ngtm(m=m_meas_top * Msun_h, z=z).value
        )
        assert counted == pytest.approx(expected, rel=2e-5)
        # And ngtm of the converted mass function is that, by construction.
        assert mf.ngtm(m=m0 * Msun_h, z=z).value == pytest.approx(
            smt.ngtm(m=m_meas * Msun_h, z=z).value, rel=1e-14
        )


def test_rho_gtm_is_the_mass_in_the_new_definition(smt200):
    """rho(>M) is the integral of M dn/dM in the new definition."""
    z = 0.5
    m0 = 1e12
    grid = np.geomspace(m0, 10**smt200.mass_accuracy.log10_m_max / 1.5, 3001)
    dndm = smt200.dndm(m=grid * Msun_h, z=z).value
    counted = simpson(grid**2 * dndm, x=np.log(grid))
    assert smt200.rho_gtm(m=m0 * Msun_h, z=z).value == pytest.approx(counted, rel=1e-4)


def test_converting_to_an_equal_overdensity_is_the_identity(smt):
    """SO-mean of the virial overdensity at z is the virial definition there (#362)."""
    z = 0.5
    delta = float(SOVirial().delta_halo_mean_kernel(omega_m_z=omega_m(Planck18, z, "cb")))
    mf = smt.evolve(mass_definition=SOMean(overdensity=delta), mass_conversion="profile")
    assert mf.converts_masses
    m = M * Msun_h
    np.testing.assert_allclose(mf.dndm(m=m, z=z), smt.dndm(m=m, z=z), rtol=1e-12)
    np.testing.assert_allclose(mf.ngtm(m=m, z=z), smt.ngtm(m=m, z=z), rtol=1e-12)
    np.testing.assert_allclose(mf.rho_gtm(m=m, z=z), smt.rho_gtm(m=m, z=z), rtol=1e-12)


def test_einstein_de_sitter_virial_and_so_mean_18pi2_agree():
    eds = FlatLambdaCDM(H0=70, Om0=1.0, Ob0=0.05, Tcmb0=0.0, meta={"sigma8": 0.8, "n": 1.0})
    mf = MassFunction.build(cosmology=eds, **{"transfer.model": "EH"}, fit="SMT")
    converted = mf.evolve(
        mass_definition=SOMean(overdensity=18 * math.pi**2), mass_conversion="profile"
    )
    m = M * Msun_h
    for z in (0.0, 2.0):
        np.testing.assert_allclose(converted.dndm(m=m, z=z), mf.dndm(m=m, z=z), rtol=1e-12)


def test_converted_mass_function_is_physical(smt200, smt):
    """More haloes are above any mass in 200m than in the virial definition.

    Each halo is heavier in 200m (Delta_m = 200 < 333), so n_200m(>M) > n_vir(>M).
    """
    m = M * Msun_h
    ratio = (smt200.ngtm(m=m, z=0.0) / smt.ngtm(m=m, z=0.0)).value
    # The same haloes, each heavier in 200m: n_200m(>M) >= n_vir(>M).
    assert np.all(ratio > 1)
    assert np.all(np.isfinite(smt200.dndm(m=m, z=0.0)))


@pytest.mark.parametrize("name", ["dndm", "ngtm", "rho_gtm"])
def test_converted_quantities_are_batch_independent(smt200, name):
    method = getattr(smt200, name)
    full = method(m=M * Msun_h, z=0.5).value
    for i in (0, 9, 25):
        np.testing.assert_array_equal(method(m=M[i : i + 1] * Msun_h, z=0.5).value, full[i : i + 1])
    grid = method(m=M[None, :] * Msun_h, z=np.array([0.0, 0.5])[:, None]).value
    np.testing.assert_array_equal(grid[1], full)


def test_view_matches_the_methods(smt200):
    view = smt200.at(z=0.5, m=M * Msun_h)
    np.testing.assert_array_equal(view.dndm, smt200.dndm(m=M * Msun_h, z=0.5))
    np.testing.assert_array_equal(view.ngtm, smt200.ngtm(m=M * Msun_h, z=0.5))


def test_behroozi_uses_the_measured_ngtm(smt):
    """Behroozi's correction takes n(>M') of the measured (virial) definition."""
    behroozi = smt.evolve(fit="Behroozi")
    converted = behroozi.evolve(mass_definition="200m", mass_conversion="profile")
    z = 3.0
    om = omega_m(Planck18, z, "cb")
    out = converted.mass_conversion.convert_kernel(
        m=M, z=z, omega_m_z=om, source=SOMean(), target=SOVirial()
    )
    np.testing.assert_allclose(
        converted.dndm(m=M * Msun_h, z=z).value,
        behroozi.dndm(m=out.m * Msun_h, z=z).value * out.dm_dm,
        rtol=1e-13,
    )


def test_bocquet_mass_ratio_is_applied_once(smt):
    """A Bocquet 200c fit converted to 200m: its own M200c/M200m ratio, then NFW.

    The fit's mass function, with its published ratio, is in 200c; the conversion maps
    it to 200m like any other fit, so the 200m result is the 200c one at M200c(M), times
    dM200c/dM, and not also multiplied by the ratio again.
    """
    bocquet = smt.evolve(fit="Bocquet200cDMOnly")
    assert bocquet.fit.modifies_dndm
    converted = bocquet.evolve(mass_definition="200m", mass_conversion="profile")
    z = 0.5
    m = np.geomspace(1e13, 1e15, 9)
    out = converted.mass_conversion.convert_kernel(
        m=m, z=z, omega_m_z=omega_m(Planck18, z, "cb"), source=SOMean(), target=SOCritical()
    )
    np.testing.assert_allclose(
        converted.dndm(m=m * Msun_h, z=z).value,
        bocquet.dndm(m=out.m * Msun_h, z=z).value * out.dm_dm,
        rtol=1e-13,
    )


def test_domain_policy_covers_the_concentration_model(smt):
    m = np.array([1e9, 1e12]) * Msun_h  # 1e9 is below Duffy08's calibration
    mf = smt.evolve(mass_definition="200m", mass_conversion="profile")
    np.testing.assert_array_equal(mf.in_calibration_domain(m=m, z=0.5), [False, True])
    masked = mf.evolve(domain_policy="mask")
    for name in ("dndm", "ngtm", "rho_gtm"):
        out = getattr(masked, name)(m=m, z=0.5).value
        assert np.isnan(out[0])
        assert np.isfinite(out[1])
    with pytest.raises(DomainError, match="Duffy08's calibration domain"):
        mf.evolve(domain_policy="raise").dndm(m=m, z=0.5)
    with pytest.warns(HMFExtrapolationWarning, match="Duffy08's calibration domain"):
        mf.evolve(domain_policy="warn").dndm(m=m, z=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mf.evolve(domain_policy="warn").dndm(m=m[1:], z=0.5)


def test_routing_and_introspection(smt, smt200):
    info = MassFunction.parameter_info()
    assert info["mass_definition"].path == "mass_definition"
    assert info["mass_definition"].default is None
    assert info["mass_conversion"].default is None
    assert "ProfileConversion" in info["mass_conversion"].doc
    assert MassFunction.invalidated_by("mass_definition") == ("",)
    assert MassFunction.invalidated_by("mass_conversion") == ("",)
    assert smt200.mass_definition == SOMean()
    assert smt200.mass_conversion == ProfileConversion()
    # Dotted paths into the models.
    critical = smt200.evolve(mass_definition=SOCritical())
    assert critical.evolve(**{"mass_definition.overdensity": 500}).mass_definition == SOCritical(
        overdensity=500
    )
    assert smt200.evolve(**{"mass_conversion.profile": "NFW"}) == smt200
    relaxed = smt200.evolve(**{"mass_conversion.concentration.sample": "relaxed"})
    assert relaxed.mass_conversion.concentration == Duffy08(sample="relaxed")
    # The variance and the linear power are shared.
    assert smt200.variance is smt.variance
    assert relaxed.linear_power is smt200.linear_power
    # Built from names, and round-tripped through to_flat.
    built = MassFunction.build(
        **{"transfer.model": "EH"}, fit="SMT", mass_definition="200m", mass_conversion="profile"
    )
    assert built == smt200
    assert MassFunction.from_flat(smt200.to_flat()) == smt200


def test_pickle_and_copy(smt200):
    m = M * Msun_h
    expected = smt200.dndm(m=m, z=0.5)
    for clone in (pickle.loads(pickle.dumps(smt200)), copy.deepcopy(smt200)):
        assert clone == smt200
        np.testing.assert_array_equal(clone.dndm(m=m, z=0.5), expected)


def test_conversion_does_not_import_halomod():
    """hmf.core converts masses without halomod: it is never imported (#392)."""
    code = textwrap.dedent(
        """
        import sys, warnings
        warnings.simplefilter("ignore")
        sys.modules["halomod"] = None  # any import of it fails
        import numpy as np
        from hmf.core.mass_function import MassFunction
        from hmf.core.units import Msun_h
        mf = MassFunction.build(
            **{"transfer.model": "EH"}, fit="SMT", mass_definition="200c",
            mass_conversion="profile",
        )
        out = mf.dndm(m=np.logspace(10, 15, 11) * Msun_h, z=0.5)
        assert np.all(np.isfinite(out))
        del sys.modules["halomod"]
        print(sorted(name for name in sys.modules if name.split(".")[0] == "halomod"))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


# ---------------------------------------------------------------------------------
# Regression against hmf 3.x
# ---------------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("z", [0.0, 1.0])
def test_converted_dndm_matches_v3(monkeypatch, z):
    """v3's converted dn/dm (#362), given v4's concentrations through its profile-free path.

    v3's ``change_definition`` takes the concentrations and a profile; here they are
    Duffy08's in the source definition, and a minimal NFW profile (v3 gets both from
    halomod, whose Duffy08 uses its 200c parameters for masses in any definition). The
    ratio of converted to unconverted dn/dm cancels the differences of the two
    pipelines elsewhere; v3's Jacobian is a finite difference on its grid (dlog10m =
    0.01), v4's analytic.
    """
    from hmf import MassFunction as V3MassFunction
    from hmf.halos import mass_definitions as v3_mdefs

    duffy = Duffy08(definition="200m")

    class _NFW:
        def __init__(self, mdef, z, cosmo):
            self.mdef, self.z, self.cosmo = mdef, z, cosmo

        def _h(self, c):
            return np.log1p(c) - c / (1 + c)

        def _rho_s(self, c):
            return c**3 * self.mdef.halo_density(self.z, self.cosmo) / (3 * self._h(c))

    original = v3_mdefs.BaseMassDefinition.change_definition

    def profile_free(self, m, mdef, profile=None, c=None, z=0, cosmo=None):
        c = duffy.concentration(m=np.asarray(m) * Msun_h, z=z)
        return original(self, m, mdef, profile=_NFW(self, z, cosmo), c=c, z=z, cosmo=cosmo)

    monkeypatch.setattr(v3_mdefs.BaseMassDefinition, "change_definition", profile_free)
    kwargs = {
        "cosmo_model": Planck18,
        "sigma_8": 0.8102,
        "n": 0.9665,
        "hmf_model": "SMT",
        "transfer_model": "EH",
        "Mmin": 10,
        "Mmax": 15,
        "dlog10m": 0.01,
        "z": z,
    }
    converted = V3MassFunction(mdef_model="SOMean", disable_mass_conversion=False, **kwargs)
    native = V3MassFunction(**kwargs)
    v3_ratio = converted.dndm / native.dndm

    mf = MassFunction.build(**{"transfer.model": "EH"}, fit="SMT")
    mf200 = mf.evolve(
        mass_definition="200m", mass_conversion=ProfileConversion(concentration=duffy)
    )
    m = converted.m * Msun_h
    v4_ratio = (mf200.dndm(m=m, z=z) / mf.dndm(m=m, z=z)).value
    np.testing.assert_allclose(v4_ratio, v3_ratio, rtol=2e-3)
