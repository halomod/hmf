"""Tests of the MassFunction stage: the halo mass function as a function of (m, z)."""

import copy
import inspect
import math
import pickle
import warnings

import astropy.units as u
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM, Planck18
from power_models import UnitTransfer
from scipy import special

from hmf.core import fits
from hmf.core._kernels import mass_function as mf_kernels
from hmf.core.accuracy import KAccuracy, MassAccuracy
from hmf.core.domain import DomainError
from hmf.core.fits import FittingFunction
from hmf.core.growth import Growth
from hmf.core.linear_power import LinearPower
from hmf.core.mass_function import MassFunction, MassFunctionView
from hmf.core.mass_variance import INTERPOLATION_RTOL, MassVariance
from hmf.core.transfer import Transfer
from hmf.core.units import (
    Mpc_h,
    Msun_h,
    UnitBoundaryError,
    dndm_unit,
    number_density_unit,
    rho_unit,
)
from hmf.exceptions import HMFExtrapolationWarning

M = np.logspace(9, 15.5, 27)
Z = np.array([0.0, 0.5, 1.0, 2.0])


@pytest.fixture(scope="module")
def mf():
    """A Planck18 mass function with EH: no Boltzmann run, Tinker08."""
    return MassFunction.build(transfer_model="EH")


def _rho_cb(cosmo) -> float:
    """The mean density of CDM + baryons in Msun h^2 / Mpc^3, from astropy alone."""
    rho = (cosmo.critical_density0 * cosmo.Om0).to_value(u.Msun / u.Mpc**3)
    return rho / cosmo.h**2


def _power_law(n: float, sigma_8: float, fit="PS", cosmo=Planck18, **kwargs) -> MassFunction:
    """A mass function of the power law P = A k^n, normalised to sigma_8 of the same field."""
    transfer = Transfer(cosmology=cosmo, model=UnitTransfer(), n_s=n)
    lp = LinearPower(
        transfer=transfer,
        growth=Growth.from_transfer(transfer),
        sigma_8=sigma_8,
        species="cb",
        sigma_8_species="cb",
    )
    return MassFunction(linear_power=lp, fit=fit, **kwargs)


def _power_law_sigma(m, n, sigma_8, cosmo=Planck18):
    """sigma(m) of a power law with a top-hat: sigma_8 (R / 8)^(-(n + 3) / 2)."""
    r = (3 * m / (4 * math.pi * _rho_cb(cosmo))) ** (1 / 3)
    return sigma_8 * (r / 8.0) ** (-(n + 3) / 2)


def _all_fits() -> list[str]:
    return sorted(
        name
        for name in fits.__all__
        if inspect.isclass(getattr(fits, name))
        and issubclass(getattr(fits, name), FittingFunction)
        and getattr(fits, name) is not FittingFunction
    )


# ---------------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------------


def test_sigma_8_is_recovered(mf):
    """sigma(R = 8 Mpc/h, z = 0) is sigma_8 (the lattice against the direct evaluation)."""
    m8 = mf.m_from_radius(8.0 * Mpc_h)
    assert mf.sigma(m8, 0.0) == pytest.approx(mf.linear_power.sigma_8, rel=INTERPOLATION_RTOL)


def test_sigma_8_is_recovered_for_each_species():
    """With massive neutrinos (CAMB), sigma_8 is that of sigma_8_species.

    The power of the species normalised to its own sigma_8 recovers it; the CDM +
    baryon power normalised to the total-matter sigma_8 (the default) is slightly
    above it (neutrinos suppress the total power), by less than 5%.
    """
    cosmo = Planck18.clone(m_nu=[0.1] * 3 * u.eV)
    for species in ("cb", "tot"):
        mf = MassFunction.build(
            cosmology=cosmo, species=species, sigma_8_species=species, sigma_8=0.8
        )
        m8 = mf.m_from_radius(8.0 * Mpc_h)
        assert mf.sigma(m8, 0.0) == pytest.approx(0.8, rel=INTERPOLATION_RTOL)
    default = MassFunction.build(cosmology=cosmo, sigma_8=0.8)
    s = default.sigma(default.m_from_radius(8.0 * Mpc_h), 0.0)
    assert 0.8 < s < 0.8 * 1.05


def test_growth_in_einstein_de_sitter():
    """In Einstein-de Sitter, D = a, so sigma(m, z) = sigma(m, 0) / (1 + z)."""
    eds = FlatLambdaCDM(H0=70.0, Om0=1.0, Ob0=0.05, Tcmb0=0.0)
    mf = MassFunction.build(cosmology=eds, transfer_model="EH", sigma_8=0.8, n_s=0.96)
    s = mf.sigma(M[None, :] * Msun_h, Z[:, None])
    expected = s[0] / (1 + Z[:, None])
    np.testing.assert_allclose(s, expected, rtol=1e-6)


@pytest.mark.parametrize("n", [-2.0, -2.2])
def test_press_schechter_with_a_power_law(n):
    """With P = A k^n, PS's dn/dm is the closed form.

    sigma = sigma_8 (R/8)^(-(n+3)/2), dln sigma/dln m = -(n+3)/6, and
    dn/dm = sqrt(2/pi) rho / m^2 nu exp(-nu^2/2) |dln sigma/dln m|, nu = delta_c/sigma.
    rtol 1e-4: the dn/dm target of the lattice (sigma to 1e-5, amplified by nu^2 ~ 10).
    """
    sigma_8 = 0.8
    mf = _power_law(n, sigma_8)
    m = np.logspace(8, 15, 15)
    sigma = _power_law_sigma(m, n, sigma_8)
    nu = 1.686 / sigma
    slope = -(n + 3) / 6
    expected = math.sqrt(2 / math.pi) * _rho_cb(Planck18) / m**2 * nu * np.exp(-(nu**2) / 2)
    expected *= abs(slope)
    np.testing.assert_allclose(mf.sigma(m * Msun_h, 0.0), sigma, rtol=1e-5)
    np.testing.assert_allclose(mf.dlnsigma_dlnm(m * Msun_h, 0.0), slope, rtol=1e-4)
    np.testing.assert_allclose(mf.dndm(m * Msun_h, 0.0).to_value(dndm_unit), expected, rtol=1e-4)


def _ps_fraction_above(nu):
    """The mass fraction in PS haloes above peak height nu: erfc(nu / sqrt(2))."""
    return special.erfc(nu / math.sqrt(2))


def _st_fraction_above(nu, a=0.707, p=0.3):
    """The mass fraction in (normalised) Sheth-Tormen haloes above peak height nu.

    The integral of f over ln nu from nu to infinity, with x = a nu^2 / 2:
    A [erfc(sqrt(x)) + 2^-p Gamma(1/2 - p, x) / sqrt(pi)], and
    A = 1 / (1 + 2^-p Gamma(1/2 - p) / sqrt(pi)) (Sheth, Mo & Tormen 2001, eq. 5).
    """
    x = a * nu**2 / 2
    A = 1 / (1 + 2**-p * special.gamma(0.5 - p) / math.sqrt(math.pi))
    upper_gamma = special.gammaincc(0.5 - p, x) * special.gamma(0.5 - p)
    return A * (special.erfc(np.sqrt(x)) + 2**-p * upper_gamma / math.sqrt(math.pi))


@pytest.mark.parametrize(
    ("fit", "fraction", "rtol"),
    [("PS", _ps_fraction_above, 1e-5), ("ST", _st_fraction_above, 1e-5)],
)
def test_mass_conservation(fit, fraction, rtol):
    """A normalised fit puts all the mass in haloes: rho(>m) -> rho_cb as m -> 0.

    A power law n = -2 normalised to sigma_8 = 1 spans peak heights from 0.01 (at
    10 Msun/h, about the lowest mass whose sigma the default k grid resolves for this
    spectrum, which has much more small-scale power than CDM) to 5.8 (at the top of the
    lattice, 10^17.5, where less than 1e-6 of the mass is above). The mass in haloes
    above m is rho_cb times the fit's mass fraction between nu(m) and nu(m_top), in
    closed form for PS and ST: rho(>m) matches
    it to rtol everywhere. The mass in haloes above 10 Msun/h is rho_cb, to the fraction
    the closed form puts below that mass (0.8% for PS, 10% for ST, whose low-mass tail
    falls only as nu^0.4).
    """
    n, sigma_8 = -2.0, 1.0
    mf = _power_law(n, sigma_8, fit=fit)
    m = np.logspace(1, 16, 31)
    rho = _rho_cb(Planck18)
    nu, nu_top = (1.686 / _power_law_sigma(x, n, sigma_8) for x in (m, mf.m_top))
    expected = rho * (fraction(nu) - fraction(nu_top))
    np.testing.assert_allclose(mf.rho_gtm(m * Msun_h, 0.0).to_value(rho_unit), expected, rtol=rtol)
    below = 1 - fraction(nu[0])
    assert below < (0.01 if fit == "PS" else 0.1)
    total = mf.rho_gtm(m[0] * Msun_h, 0.0).to_value(rho_unit) / rho
    assert total == pytest.approx(1 - below, rel=rtol)
    assert fraction(nu_top) < 1e-6


def test_ngtm_is_monotonic_and_its_derivative_is_dndm(mf):
    """n(>m) decreases, and -dn(>m)/dm = dn/dm, to the interpolation tolerance."""
    m = np.logspace(6, 16, 201)
    n = mf.ngtm_kernel(m[None, :], Z[:, None])
    assert np.all(np.diff(n, axis=1) < 0)
    eps = 1e-4
    up = mf.ngtm_kernel(m[None, :] * math.exp(eps), Z[:, None])
    down = mf.ngtm_kernel(m[None, :] * math.exp(-eps), Z[:, None])
    derivative = -(up - down) / (2 * eps) / m
    dndm = mf.dndm_kernel(m[None, :], Z[:, None])
    nu = mf.ln_sigma_and_slope_kernel(m[None, :], Z[:, None])[0]
    nu = 1.686 / np.exp(nu)
    ok = nu < 5
    np.testing.assert_allclose(derivative[ok], dndm[ok], rtol=1e-4)


def test_ngtm_is_continuous_across_nodes(mf):
    """n(>m) is continuous where the panel of m changes (at the lattice nodes)."""
    node = 10 ** (600 * 0.02)  # a node of the default lattice
    below, above = mf.ngtm_kernel(np.array([node * (1 - 1e-12), node * (1 + 1e-12)]), 0.0)
    assert below == pytest.approx(above, rel=1e-10)


def test_ngtm_truncation_at_the_top_is_negligible():
    """Integrating to 10^18.5 instead of 10^17.5 changes n(>m <= 10^16) by < 1e-12."""
    base = MassFunction.build(transfer_model="EH")
    wider = MassFunction.build(transfer_model="EH", mass_accuracy=MassAccuracy(log10_m_max=18.5))
    m = np.logspace(8, 16, 9)
    for z in (0.0, 2.0):
        np.testing.assert_allclose(base.ngtm_kernel(m, z), wider.ngtm_kernel(m, z), rtol=1e-12)
    assert base.ngtm_kernel(base.m_top, 0.0) == pytest.approx(0.0, abs=1e-300)


def test_peak_height_times_sigma_is_delta_c(mf):
    m = M[None, :] * Msun_h
    product = mf.peak_height(m, Z[:, None]) * mf.sigma(m, Z[:, None])
    np.testing.assert_allclose(product, mf.delta_c, rtol=2e-16)
    assert mf.evolve(delta_c=1.5).peak_height(m, 0.0)[0] == pytest.approx(
        1.5 / mf.sigma(m, 0.0)[0], rel=2e-16
    )


def test_converters_round_trip(mf):
    m = M * Msun_h
    for z in (0.0, 1.5):
        np.testing.assert_allclose(mf.m_from_sigma(mf.sigma(m, z), z), m, rtol=1e-10)
        np.testing.assert_allclose(mf.m_from_peak_height(mf.peak_height(m, z), z), m, rtol=1e-10)
    r = mf.variance.radius_from_m(m)
    np.testing.assert_allclose(mf.m_from_radius(r), m, rtol=1e-12)
    # Broadcasting: sigma against z.
    assert mf.m_from_sigma(np.array([1.0, 2.0])[None, :], Z[:, None]).shape == (4, 2)


def test_dndm_variants(mf):
    m = M * Msun_h
    dndm = mf.dndm(m, 1.0)
    np.testing.assert_allclose(mf.dndlnm(m, 1.0).value, M * dndm.value, rtol=1e-15)
    np.testing.assert_allclose(
        mf.dndlog10m(m, 1.0).value, math.log(10) * M * dndm.value, rtol=1e-15
    )


def test_dndm_is_fsigma_times_its_factor(mf):
    """dn/dm = f(sigma) rho_cb / m^2 |dln sigma/dln m| for a fit that does not modify it."""
    m = M * Msun_h
    expected = mf.fsigma(m, 0.5) * _rho_cb(Planck18) / M**2 * np.abs(mf.dlnsigma_dlnm(m, 0.5))
    np.testing.assert_allclose(mf.dndm(m, 0.5).value, expected, rtol=1e-12)


# ---------------------------------------------------------------------------------
# Units, broadcasting, the view
# ---------------------------------------------------------------------------------

_UNITS = {
    "sigma": None,
    "dlnsigma_dlnm": None,
    "peak_height": None,
    "fsigma": None,
    "dndm": dndm_unit,
    "dndlnm": number_density_unit,
    "dndlog10m": number_density_unit,
    "ngtm": number_density_unit,
    "rho_gtm": rho_unit,
    "in_calibration_domain": None,
}


@pytest.mark.parametrize("name", list(_UNITS))
def test_units_and_scalar_rule(mf, name):
    method = getattr(mf, name)
    unit = _UNITS[name]
    scalar = method(1e12 * Msun_h, 0.5)
    array = method(M * Msun_h, 0.5)
    physical = method(M / Planck18.h * u.Msun, 0.5)
    if unit is None:
        assert np.isscalar(scalar)
        assert isinstance(array, np.ndarray)
        np.testing.assert_allclose(physical, array, rtol=1e-12)
    else:
        assert scalar.unit == unit
        assert scalar.shape == ()
        assert array.unit == unit
        np.testing.assert_allclose(physical.value, array.value, rtol=1e-12)
    assert np.shape(array) == M.shape
    with pytest.raises(UnitBoundaryError):
        method(1e12, 0.5)


def test_converter_units(mf):
    assert mf.m_from_sigma(1.0, 0.0).unit == Msun_h
    assert mf.m_from_peak_height(1.0, 0.0).shape == ()
    assert mf.m_from_radius(8 * Mpc_h).unit == Msun_h
    assert mf.m_from_radius(8 / Planck18.h * u.Mpc).value == pytest.approx(
        mf.m_from_radius(8 * Mpc_h).value, rel=1e-12
    )


@pytest.mark.parametrize("name", ["sigma", "fsigma", "dndm", "ngtm", "rho_gtm"])
def test_broadcasting_and_batch_independence(mf, name):
    """f(m[None, :], z[:, None]) is (nz, nm), and each row equals the call at that z alone."""
    method = getattr(mf, name)
    grid = method(M[None, :] * Msun_h, Z[:, None])
    assert np.shape(grid) == (Z.size, M.size)
    for i, z in enumerate(Z):
        assert np.array_equal(np.asarray(method(M * Msun_h, z)), np.asarray(grid[i]))
    # The same (m, z) in another arrangement.
    assert np.array_equal(np.asarray(method(M[:, None] * Msun_h, Z[None, :])), np.asarray(grid).T)


def test_view_matches_the_methods(mf):
    m = M * Msun_h
    view = mf.at(1.0, m)
    assert isinstance(view, MassFunctionView)
    assert view.z == 1.0
    np.testing.assert_array_equal(view.m, m)
    for name in _UNITS:
        expected = getattr(mf, name)(m, 1.0)
        assert np.array_equal(np.asarray(getattr(view, name)), np.asarray(expected)), name
        if hasattr(expected, "unit"):
            assert getattr(view, name).unit == expected.unit


def test_view_default_masses_are_the_lattice(mf):
    view = mf.at(0.0)
    acc = mf.variance.mass_accuracy
    np.testing.assert_allclose(np.log10(view.m.value[[0, -1]]), [acc.log10_m_min, 17.5], atol=1e-12)
    np.testing.assert_allclose(np.diff(np.log10(view.m.value)), acc.dlog10_m, rtol=1e-9)
    assert view.ngtm[-1].value == pytest.approx(0.0, abs=1e-300)
    with pytest.raises(ValueError, match="scalar"):
        mf.at(Z)
    with pytest.raises(ValueError, match="1D"):
        mf.at(0.0, M[None, :] * Msun_h)
    with pytest.raises(UnitBoundaryError):
        mf.at(0.0, M)


# ---------------------------------------------------------------------------------
# Domains
# ---------------------------------------------------------------------------------


def test_invalid_inputs_raise(mf):
    m = M * Msun_h
    for method in (mf.sigma, mf.dndm, mf.ngtm):
        with pytest.raises(DomainError):
            method(-1.0 * Msun_h, 0.0)
        with pytest.raises(DomainError):
            method(m, -1.0)
        with pytest.raises(DomainError):
            method(m, np.nan)
    with pytest.raises(DomainError, match="m_integral"):
        mf.ngtm(1e18 * Msun_h, 0.0)
    with pytest.raises(DomainError):
        mf.ngtm_kernel(np.array([1e12, 1e18]), 0.0)
    with pytest.raises(DomainError):
        mf.m_from_sigma(-1.0, 0.0)
    with pytest.raises(DomainError):
        mf.m_from_peak_height(0.0, 0.0)
    with pytest.raises(DomainError):
        mf.m_from_radius(0.0 * Mpc_h)
    with pytest.raises(DomainError):
        mf.dndm_kernel(np.array([1e12, np.inf]), 0.0)


def test_extension_raise_bounds_the_masses():
    mf = MassFunction.build(
        transfer_model="EH", mass_accuracy=MassAccuracy(log10_m_min=8.0, extension="raise")
    )
    assert mf.ngtm(1e8 * Msun_h, 0.0).value > 0
    with pytest.raises(DomainError):
        mf.dndm(1e7 * Msun_h, 0.0)


def test_valid_domain_always_raises(mf):
    """Outside the fit's valid domain (Yung24 below z = 6) every policy raises."""
    for policy in ("ignore", "warn", "mask", "raise"):
        stage = mf.evolve(fit="Yung24", domain_policy=policy)
        with pytest.raises(DomainError, match="valid domain"):
            stage.dndm(1e10 * Msun_h, 1.0)
    assert mf.evolve(fit="Yung24").dndm(1e10 * Msun_h, 7.0).value > 0


def test_domain_policy(mf):
    """Outside Tinker08's calibration domain (z <= 2.5, 0.4 <= sigma <= 4) the policy applies."""
    m, z = M[None, :] * Msun_h, np.array([0.0, 3.0])[:, None]
    inside = mf.in_calibration_domain(m, z)
    assert inside.any()
    assert not inside.all()
    reference = {name: getattr(mf, name)(m, z) for name in ("fsigma", "dndm", "ngtm", "rho_gtm")}

    ignore = mf.evolve(domain_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name, value in reference.items():
            assert np.array_equal(np.asarray(getattr(ignore, name)(m, z)), np.asarray(value))

    masked = mf.evolve(domain_policy="mask")
    for name, value in reference.items():
        out = np.asarray(getattr(masked, name)(m, z))
        assert np.all(np.isnan(out[~inside]))
        assert np.array_equal(out[inside], np.asarray(value)[inside])

    warn = mf.evolve(domain_policy="warn")
    with pytest.warns(HMFExtrapolationWarning, match="calibration domain"):
        warn.dndm(m, z)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.array_equal(warn.ngtm(m, z).value, reference["ngtm"].value)  # once per stage

    raising = mf.evolve(domain_policy="raise")
    for name in reference:
        with pytest.raises(DomainError, match="calibration domain"):
            getattr(raising, name)(m, z)
    inside_m = 1e13 * Msun_h
    assert raising.dndm(inside_m, 0.0) == mf.dndm(inside_m, 0.0)
    with pytest.raises(ValueError):
        mf.evolve(domain_policy="clip")


@pytest.mark.parametrize("name", _all_fits())
def test_every_fit_runs(mf, name):
    """Every registered fit gives a finite, positive mass function and n(>m)."""
    z = 7.0 if name == "Yung24" else 0.5
    stage = mf.evolve(fit=name)
    m = np.logspace(10, 14, 9)
    dndm = stage.dndm_kernel(m, z)
    ngtm = stage.ngtm_kernel(m, z)
    assert np.all(np.isfinite(dndm) & (dndm > 0))
    assert np.all(np.isfinite(ngtm) & (ngtm > 0))
    assert np.all(np.diff(ngtm) < 0)


def test_behroozi_uses_the_tinker_ngtm(mf):
    """Behroozi's dn/dm is Tinker08's corrected with Tinker08's n(>m) (App. G).

    n(>m) of Behroozi is theta n_T08(>m), so its n(>m) and Tinker08's satisfy that
    relation (the correction's own definition), up to the integration's accuracy.
    """
    behroozi = mf.evolve(fit="Behroozi")
    tinker = mf.evolve(fit=fits.Tinker08())  # Behroozi is Tinker08 at the virial Delta
    m = np.logspace(10, 14, 9)
    z = 3.0
    a = 1 / (1 + z)
    alpha = 0.144 / (1 + math.exp(14.79 * (a - 0.213)))
    gamma = 0.5 / (1 + math.exp(6.5 * a))
    theta = 10 ** (alpha * (m / Planck18.h / 10**11.5) ** gamma)
    # Tinker08 at Behroozi's (virial) overdensity: the same f(sigma) as Behroozi's raw one.
    virial = tinker.evolve(fit=fits.Behroozi())
    n_t08 = virial._integral(m, z, raw=True, moment=0)
    np.testing.assert_allclose(behroozi.ngtm_kernel(m, z), theta * n_t08, rtol=1e-4)


# ---------------------------------------------------------------------------------
# Construction, evolve, copies
# ---------------------------------------------------------------------------------


def test_build_defaults():
    mf = MassFunction.build()
    lp = mf.linear_power
    assert lp.cosmology is Planck18
    assert lp.sigma_8 == Planck18.meta["sigma8"]
    assert lp.transfer.n_s == Planck18.meta["n"]
    assert type(lp.transfer.model).__name__ == "CAMB"
    assert type(lp.growth.model).__name__ == "ODEGrowth"
    assert (lp.species, lp.sigma_8_species) == ("cb", "tot")
    assert type(mf.fit).__name__ == "Tinker08"
    assert type(mf.variance.filter).__name__ == "TopHat"
    assert mf.delta_c == 1.686
    assert mf.domain_policy == "ignore"
    assert lp.k_accuracy == lp.growth.k_accuracy == mf.variance.k_accuracy == KAccuracy()
    assert mf.variance.mass_accuracy == MassAccuracy()


def test_build_accepts_names_or_instances():
    by_name = MassFunction.build(transfer_model="EH", fit="ST", filter="SharpK", growth_model="ODE")
    by_instance = MassFunction.build(
        transfer_model=Transfer(model="EH").model,
        fit=fits.ST(),
        filter=MassVariance(power=by_name.variance.power, filter="SharpK").filter,
        growth_model=Growth().model,
    )
    assert by_name == by_instance


def test_build_needs_sigma_8_and_n_s_without_cosmology_meta():
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3, Ob0=0.05, Tcmb0=2.7255)
    with pytest.raises(ValueError, match="sigma_8"):
        MassFunction.build(cosmology=cosmo, transfer_model="EH", n_s=0.96)
    with pytest.raises(ValueError, match="n_s"):
        MassFunction.build(cosmology=cosmo, transfer_model="EH", sigma_8=0.8)
    mf = MassFunction.build(cosmology=cosmo, transfer_model="EH", sigma_8=0.8, n_s=0.96)
    assert mf.linear_power.sigma_8 == 0.8


def test_default_variance_is_the_linear_powers(mf):
    stage = MassFunction(linear_power=mf.linear_power)
    assert stage.variance == mf.variance
    assert stage.variance.power == mf.linear_power.power_source


def test_inconsistent_stages_raise(mf):
    lp = mf.linear_power
    tot = MassVariance(power=lp.transfer.power_source("tot"), k_accuracy=lp.k_accuracy)
    with pytest.raises(ValueError, match="power_source"):
        MassFunction(linear_power=lp, variance=tot)
    other = Transfer(model="EH", n_s=0.9).power_source("cb")
    with pytest.raises(ValueError, match="power_source"):
        MassFunction(linear_power=lp, variance=MassVariance(power=other))
    with pytest.raises(ValueError, match="KAccuracy"):
        MassFunction(
            linear_power=lp,
            variance=MassVariance(power=lp.power_source, k_accuracy=KAccuracy.fast()),
        )
    with pytest.raises(ValueError, match="delta_c"):
        mf.evolve(delta_c=0.0)


def test_evolve_shares_the_variance(mf):
    """Changing sigma_8, z-independent fields or the fit keeps the same variance object."""
    for changed in (
        mf.evolve(linear_power=mf.linear_power.evolve(sigma_8=0.7)),
        mf.evolve(fit="ST"),
        mf.evolve(delta_c=1.7),
        mf.evolve(domain_policy="mask"),
    ):
        assert changed.variance is mf.variance


def test_evolve_is_atomic(mf):
    m = M * Msun_h
    before = mf.dndm(m, 0.0)
    fields = {f.name: getattr(mf, f.name) for f in mf.fields_info()}
    bad = MassVariance(power=Transfer(model="EH", n_s=0.9).power_source("cb"))
    for changes in ({"variance": bad}, {"delta_c": -1.0}, {"fit": "NotAFit"}):
        with pytest.raises((ValueError, LookupError)):
            mf.evolve(**changes)
    assert {f.name: getattr(mf, f.name) for f in mf.fields_info()} == fields
    assert np.array_equal(mf.dndm(m, 0.0), before)


def _rebuilt(mf: MassFunction, **changes) -> MassFunction:
    """A fresh MassFunction, built from scratch with ``changes`` to build()'s arguments."""
    lp = mf.linear_power
    kwargs = {
        "cosmology": lp.cosmology,
        "transfer_model": lp.transfer.model,
        "n_s": lp.transfer.n_s,
        "sigma_8": lp.sigma_8,
        "fit": mf.fit,
        "filter": mf.variance.filter,
        "delta_c": mf.delta_c,
    }
    kwargs.update(changes)
    return MassFunction.build(**kwargs)


def test_no_stale_results(mf):
    """Every change changes the result, and gives what a fresh stage gives (#382).

    v3 recorded a quantity's dependencies on its first evaluation, so a parameter read
    only on another code branch (e.g. Omega_m(z), read by Watson but not by PS) was
    missed after a change of fit. Here a chain of changes, made on a stage whose
    results are already cached, must give bit for bit what a stage built fresh gives.
    """
    m = M * Msun_h
    stage = mf.evolve(fit="PS")
    first = stage.dndm(m, 1.0)

    stage = stage.evolve(fit="Watson")  # reads omega_m(z) and delta_halo
    second = stage.dndm(m, 1.0)
    assert not np.allclose(second, first, rtol=1e-6, atol=0)

    cosmo = Planck18.clone(Om0=0.28)
    transfer = stage.linear_power.transfer.evolve(cosmology=cosmo)
    lp = stage.linear_power.evolve(
        transfer=transfer, growth=Growth.from_transfer(transfer), cosmology=cosmo
    )
    stage = stage.evolve(linear_power=lp, variance=stage.variance.evolve(power=lp.power_source))
    third = stage.dndm(m, 1.0)
    assert not np.allclose(third, second, rtol=1e-6, atol=0)
    fresh = _rebuilt(stage, cosmology=cosmo)
    assert np.array_equal(third, fresh.dndm(m, 1.0))
    assert np.array_equal(stage.ngtm(m, 1.0), fresh.ngtm(m, 1.0))

    # Each other field changes what depends on it. Watson's f does not depend on
    # delta_c: delta_c changes the peak height, and the mass function of PS.
    ps, st = stage.evolve(fit="PS"), stage.evolve(fit="ST")
    for before, changed, name in (
        (stage, stage.evolve(linear_power=stage.linear_power.evolve(sigma_8=0.75)), "dndm"),
        (st, st.evolve(fit=fits.ST(a=0.75)), "dndm"),
        (stage, stage.evolve(delta_c=1.6), "peak_height"),
        (ps, ps.evolve(delta_c=1.6), "dndm"),
    ):
        old, new = getattr(before, name)(m, 1.0), getattr(changed, name)(m, 1.0)
        assert not np.allclose(new, old, rtol=1e-6, atol=0)


def test_pickle_and_deepcopy(mf):
    m = M * Msun_h
    expected = mf.ngtm(m, 0.5)
    for other in (pickle.loads(pickle.dumps(mf)), copy.deepcopy(mf)):
        assert other == mf
        assert np.array_equal(other.ngtm(m, 0.5), expected)
        assert np.array_equal(other.dndm(m, 0.5), mf.dndm(m, 0.5))


# ---------------------------------------------------------------------------------
# Determinism of n(>m)
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize("filt", ["TopHat", "SharpK"])
def test_ngtm_is_bit_identical_under_extension(filt):
    """n(>m) is the same, to the bit, whatever was asked before (#384).

    Narrow masses first then wide, or wide first then narrow, and one mass at a time:
    every arrangement gives the same values, on separate stages (separate lattices).
    """
    narrow = np.logspace(10, 14, 41)
    wide = np.logspace(5, 16, 111)

    def build():
        return MassFunction.build(transfer_model="EH", filter=filt, fit="ST")

    narrow_first = build()
    n1 = narrow_first.ngtm_kernel(narrow, 1.0)
    w1 = narrow_first.ngtm_kernel(wide, 1.0)

    wide_first = build()
    w2 = wide_first.ngtm_kernel(wide, 1.0)
    n2 = wide_first.ngtm_kernel(narrow, 1.0)

    assert np.array_equal(n1, n2)
    assert np.array_equal(w1, w2)
    one_by_one = build()
    alone = np.array([one_by_one.ngtm_kernel(x, 1.0) for x in narrow[::-1]])[::-1]
    assert np.array_equal(alone, n1)
    # With several redshifts at once.
    grid = build().ngtm_kernel(narrow[None, :], np.array([0.0, 1.0, 3.0])[:, None])
    assert np.array_equal(grid[1], n1)
    assert np.array_equal(
        narrow_first.rho_gtm_kernel(narrow, 1.0), wide_first.rho_gtm_kernel(narrow, 1.0)
    )


# ---------------------------------------------------------------------------------
# The quadrature kernels
# ---------------------------------------------------------------------------------


def test_log_lagrange_integrals_of_an_exponential():
    """Y = exp(a + b ln m) has a linear ln y, which the cubic reproduces exactly.

    The integral over [u_lo, u_hi] of a panel is then exp(a) (e^(b x_hi) - e^(b x_lo)) / b,
    to the accuracy of the 4-point Gauss rule (~1e-12 for b * step ~ 0.1).
    """
    step = 0.05
    a, b = -3.0, -2.0
    j = 7
    for offset in (-2, -1, 0):
        nodes = (j + offset + np.arange(4)) * step
        y = np.exp(a + b * nodes)
        u_lo, u_hi = np.array([0.0, 0.3]), np.array([1.0, 1.0])
        out = mf_kernels.log_lagrange_integrals(y, offset, u_lo, u_hi, step)
        x_lo, x_hi = (j + u_lo) * step, (j + u_hi) * step
        exact = np.exp(a) * (np.exp(b * x_hi) - np.exp(b * x_lo)) / b
        np.testing.assert_allclose(out, exact, rtol=1e-12)


def test_log_lagrange_integrals_fall_back_to_linear():
    """A stencil with a zero (an underflowed y) integrates y linearly over the panel."""
    y = np.array([4.0, 2.0, 1.0, 0.0])
    out = mf_kernels.log_lagrange_integrals(y, -1, 0.0, 1.0, 0.5)
    assert out == pytest.approx(0.5 * (2.0 + 1.0) / 2, rel=1e-14)


def test_cumulative_from_top():
    panels = np.array([[1.0, 2.0, 4.0], [0.5, 0.25, 0.125]])
    out = mf_kernels.cumulative_from_top(panels)
    np.testing.assert_array_equal(out, [[7.0, 6.0, 4.0, 0.0], [0.875, 0.375, 0.125, 0.0]])
    # The sum at a node does not depend on the panels below it.
    np.testing.assert_array_equal(mf_kernels.cumulative_from_top(panels[:, 1:]), out[:, 1:])


def test_tinker_interpolants_are_cached_and_unchanged():
    """The Tinker splines are built once per fit, and give log10_delta_spline's values."""
    from hmf.core._kernels import fits as fit_kernels

    fit = fits.Tinker08()
    assert fit._delta_interpolants is fit._delta_interpolants
    delta = np.array([200.0, 250.0, 1000.0, 3200.0])
    expected = fit_kernels.log10_delta_spline(
        fit.delta_tab, [getattr(fit, f"A_{d}") for d in fit.delta_tab], delta
    )
    np.testing.assert_array_equal(fit._delta_interpolants[0](delta), expected)
    assert pickle.loads(pickle.dumps(fit)) == fit


@pytest.mark.parametrize("name", list(_UNITS))
def test_empty_masses_give_empty_results(mf, name):
    out = getattr(mf, name)(np.zeros(0) * Msun_h, 0.5)
    assert np.shape(out) == (0,)
    assert np.shape(getattr(mf, name)(np.zeros((0, 3)) * Msun_h, np.zeros(3))) == (0, 3)
