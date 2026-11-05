"""Tests of hmf.core.mass_definitions: hashable mass definitions and their overdensities."""

import copy
import math
import pickle

import astropy.units as u
import attrs
import numpy as np
import pytest
from astropy.cosmology import Planck18

from hmf.core import fits
from hmf.core.domain import DomainError
from hmf.core.mass_definitions import (
    FOF,
    MassDefinition,
    SelfBound,
    SOCritical,
    SOGeneric,
    SOMean,
    SOVirial,
    coerce,
)
from hmf.core.species import omega_m, rho_mean0
from hmf.core.units import Mpc_h, Msun_h, rho_unit

_OMEGA_M_Z = np.array([0.05, 0.3, 0.7, 1.0])

ALL = [
    SOMean(),
    SOMean(overdensity=500),
    SOCritical(),
    SOCritical(overdensity=500),
    SOVirial(),
    SOGeneric(preferred=SOMean()),
    SOGeneric(preferred=SOVirial()),
    FOF(),
    FOF(linking_length=0.3),
    SelfBound(),
]


# ---------------------------------------------------------------------------------
# Values: equality, hashing, compatibility
# ---------------------------------------------------------------------------------


def test_equality_is_by_class_and_value_and_symmetric():
    assert SOMean(overdensity=200) == SOMean(overdensity=200.0) == SOMean()
    assert hash(SOMean(overdensity=200)) == hash(SOMean())
    for a in ALL:
        for b in ALL:
            assert (a == b) is (b == a)
            assert (a == b) is (a is b), (a, b)
    # Same number, different reference density: different definitions.
    assert SOMean(overdensity=200) != SOCritical(overdensity=200)
    # An SOGeneric is equal to no concrete definition, in either order.
    generic = SOGeneric(preferred=SOMean())
    assert generic != SOMean()
    assert SOMean() != generic


def test_definitions_are_hashable_keys():
    table = {d: i for i, d in enumerate(ALL)}
    assert len(table) == len(ALL)
    assert table[SOCritical(overdensity=500.0)] == 3


def test_accepts():
    generic = SOGeneric(preferred=SOMean())
    for d in ALL:
        assert d.accepts(d)
        assert d.accepts(attrs.evolve(d))
    for so in (SOMean(), SOCritical(overdensity=500), SOVirial(), SOGeneric(preferred=SOVirial())):
        assert generic.accepts(so)
    for other in (FOF(), SelfBound()):
        assert not generic.accepts(other)
    # Not symmetric: a concrete definition does not accept the generic one.
    assert not SOMean().accepts(generic)
    assert not SOMean().accepts(SOCritical())
    assert not SOVirial().accepts(SOMean(overdensity=18 * math.pi**2))
    assert not FOF().accepts(FOF(linking_length=0.3))


def test_resolved():
    assert SOGeneric(preferred=SOCritical()).resolved == SOCritical()
    for d in (SOMean(), SOVirial(), FOF(), SelfBound()):
        assert d.resolved is d


@pytest.mark.parametrize("definition", [d for d in ALL if not isinstance(d, SOGeneric)], ids=str)
def test_names_round_trip(definition):
    assert MassDefinition.from_name(definition.name) == definition
    assert coerce(definition.name) == definition
    assert coerce(definition) is definition


def test_names():
    assert MassDefinition.from_name(" 200M ") == SOMean()
    assert MassDefinition.from_name("fof") == FOF()
    with pytest.raises(TypeError, match="abstract"):
        MassDefinition()
    assert str(SOCritical(overdensity=500)) == "500c"
    assert str(SOGeneric(preferred=SOMean(overdensity=178))) == "so(preferred 178m)"
    with pytest.raises(ValueError, match="Unknown mass definition name"):
        MassDefinition.from_name("200x")
    with pytest.raises(TypeError, match="Expected a MassDefinition"):
        coerce(200)


def test_invalid_values():
    with pytest.raises(ValueError, match="overdensity must be > 0"):
        SOMean(overdensity=0)
    with pytest.raises(ValueError, match="linking_length must be > 0"):
        FOF(linking_length=-0.2)
    with pytest.raises(TypeError, match="preferred must be"):
        SOGeneric(preferred=SOGeneric(preferred=SOMean()))
    with pytest.raises(TypeError, match="preferred must be"):
        SOGeneric(preferred=FOF())
    with pytest.raises(attrs.exceptions.FrozenInstanceError):
        SOMean().overdensity = 300.0  # type: ignore[misc]


@pytest.mark.parametrize("definition", ALL, ids=str)
def test_pickle_and_copy(definition):
    for clone in (pickle.loads(pickle.dumps(definition)), copy.deepcopy(definition)):
        assert clone == definition
        assert hash(clone) == hash(definition)


# ---------------------------------------------------------------------------------
# The overdensity relative to the mean density
# ---------------------------------------------------------------------------------


def test_so_mean_overdensity_is_unchanged():
    mdef = SOMean(overdensity=200)
    np.testing.assert_array_equal(mdef.delta_halo_mean_kernel(_OMEGA_M_Z), 200.0)
    assert mdef.delta_halo_mean_kernel(0.3).shape == ()


def test_so_critical_overdensity_is_the_same_halo_density():
    """Delta_c rho_c(z) = Delta_m rho_m(z), with rho_m(z) = Omega_m0 rho_c0 (1 + z)^3.

    The halo density is computed both ways from astropy's densities (rho_c(z) from
    H(z), rho_m(z) from today's), with Omega_m(z) from hmf.core.species. In
    Einstein-de Sitter (Omega_m = 1) the two definitions coincide.
    """
    mdef = SOCritical(overdensity=500)
    z = np.array([0.0, 0.5, 2.0, 6.0])
    delta_m = mdef.delta_halo_mean_kernel(omega_m(Planck18, z))
    rho_m = Planck18.Om0 * Planck18.critical_density0 * (1 + z) ** 3
    np.testing.assert_allclose(
        (delta_m * rho_m).to_value(u.g / u.cm**3),
        (500 * Planck18.critical_density(z)).to_value(u.g / u.cm**3),
        rtol=1e-13,
    )
    assert mdef.delta_halo_mean_kernel(1.0) == 500.0
    assert np.all(mdef.delta_halo_mean_kernel(_OMEGA_M_Z[:-1]) > 500.0)


def test_bryan_norman_virial_overdensity():
    """18 pi^2 for Omega_m = 1; ~330 times the mean density for Planck18 today."""
    mdef = SOVirial()
    assert mdef.delta_halo_mean_kernel(1.0) == 18 * np.pi**2
    # Bryan & Norman (1998): Delta_c ~ 100 for Omega_m ~ 0.3, so Delta_m ~ 330.
    delta_m0 = mdef.delta_halo_mean_kernel(omega_m(Planck18, 0.0))
    assert 300 < delta_m0 < 360
    # It decreases towards 18 pi^2 as Omega_m(z) -> 1 at high z.
    delta = mdef.delta_halo_mean_kernel(omega_m(Planck18, np.array([0.0, 1.0, 3.0, 10.0])))
    assert np.all(np.diff(delta) < 0)
    assert delta[-1] == pytest.approx(18 * np.pi**2, rel=1e-2)


def test_virial_is_so_mean_18pi2_in_einstein_de_sitter():
    """With Omega_m(z) = 1 the two are the same overdensity, to the last bit."""
    assert SOVirial().delta_halo_mean_kernel(1.0) == SOMean(
        overdensity=18 * math.pi**2
    ).delta_halo_mean_kernel(1.0)


def test_fof_overdensity():
    """9 / (2 pi b^3): about 179 for b = 0.2, falling as b^-3, whatever Omega_m(z)."""
    delta = FOF(linking_length=0.2).delta_halo_mean_kernel(_OMEGA_M_Z)
    np.testing.assert_allclose(delta, 9 / (2 * np.pi * 0.2**3), rtol=1e-15)
    assert 178 < delta[0] < 180
    wide = FOF(linking_length=0.4)
    assert wide.delta_halo_mean_kernel(0.3) == pytest.approx(delta[0] / 8, rel=1e-14)


def test_so_generic_uses_its_preferred_definition_and_self_bound_has_none():
    watson = fits.Watson.measured_mass_definition
    assert isinstance(watson, SOGeneric)
    np.testing.assert_array_equal(
        watson.delta_halo_mean_kernel(_OMEGA_M_Z),
        watson.preferred.delta_halo_mean_kernel(_OMEGA_M_Z),
    )
    with pytest.raises(ValueError, match="no overdensity"):
        SelfBound().delta_halo_mean_kernel(0.3)
    with pytest.raises(ValueError, match="no overdensity"):
        SelfBound().m_to_r_kernel(1e12, omega_m_z=0.3, rho_mean0=1e11)


def test_delta_halo_mean_checks_omega_m():
    with pytest.raises(DomainError, match="omega_m_z"):
        SOVirial().delta_halo_mean_kernel([0.3, 0.0])
    with pytest.raises(DomainError, match="omega_m_z"):
        SOGeneric(preferred=SOMean()).delta_halo_mean_kernel(np.nan)


# ---------------------------------------------------------------------------------
# Halo densities and radii
# ---------------------------------------------------------------------------------


def test_halo_density_is_comoving_and_of_cdm_plus_baryons():
    """SO-crit(200): 200 rho_crit(z), physical, is (1 + z)^3 times the comoving density."""
    z = np.array([0.0, 1.0, 3.0])
    rho = SOCritical().halo_density_kernel(
        omega_m_z=omega_m(Planck18, z, "cb"), rho_mean0=rho_mean0(Planck18, "cb")
    )
    physical = (200 * Planck18.critical_density(z)).to_value(u.Msun / u.Mpc**3) / Planck18.h**2
    np.testing.assert_allclose(rho * (1 + z) ** 3, physical, rtol=1e-12)


def test_radius_of_unit_mean_overdensity_is_the_lagrangian_radius():
    """SO-mean(1) encloses the mean density: r = (3m / (4 pi rho_cb))^(1/3), the top hat's."""
    m = np.logspace(8, 16, 9)
    rho = rho_mean0(Planck18, "cb")
    r = SOMean(overdensity=1).m_to_r_kernel(m, omega_m_z=0.3, rho_mean0=rho)
    np.testing.assert_allclose(4 * np.pi * r**3 * rho / 3, m, rtol=1e-14)
    # Haloes are denser, so smaller: r scales as overdensity^(-1/3).
    r200 = SOMean().m_to_r_kernel(m, omega_m_z=0.3, rho_mean0=rho)
    np.testing.assert_allclose(r200 / r, 200 ** (-1 / 3), rtol=1e-14)
    # 1e12 Msun/h at 200m: about 0.2 Mpc/h (comoving).
    assert 0.15 < SOMean().m_to_r_kernel(1e12, omega_m_z=0.3, rho_mean0=rho) < 0.25


@pytest.mark.parametrize("definition", [SOMean(), SOCritical(), SOVirial(), FOF()], ids=str)
def test_mass_radius_round_trip(definition):
    m = np.logspace(6, 16, 11)
    om = omega_m(Planck18, 1.0)
    rho = rho_mean0(Planck18)
    r = definition.m_to_r_kernel(m, omega_m_z=om, rho_mean0=rho)
    np.testing.assert_allclose(
        definition.r_to_m_kernel(r, omega_m_z=om, rho_mean0=rho), m, rtol=1e-14
    )


def test_public_methods_take_and_give_quantities():
    rho = rho_mean0(Planck18) * rho_unit
    r = SOMean().m_to_r(1e12 * Msun_h, omega_m_z=0.3, rho_mean0=rho)
    assert r.unit == Mpc_h
    assert r.shape == ()
    m = SOMean().r_to_m(r, omega_m_z=0.3, rho_mean0=rho)
    assert m.unit == Msun_h
    assert m.value == pytest.approx(1e12, rel=1e-14)
    assert SOMean().halo_density(omega_m_z=0.3, rho_mean0=rho).unit == rho_unit
    assert isinstance(SOVirial().delta_halo_mean(1.0), float)
