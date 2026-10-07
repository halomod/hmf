"""Tests of the private helpers shared across hmf.core.

Arrays (``_arrays``, ``_kernels.arrays``), lattices (``_kernels.lattice``), the matter
species (``_species``), citations (``_references``), the ``Documented`` mixin and the
critical density.
"""

import importlib
import inspect
import pkgutil
import re

import astropy.units as u
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM, LambdaCDM, Planck18

import hmf.core
from hmf.core import _boltzmann, _references, _species, transfer_models
from hmf.core._arrays import dimensionless_floats, float_array, optional_float_array, read_only
from hmf.core._fields import Documented
from hmf.core._kernels import growth as kg
from hmf.core._kernels.lattice import LATTICE_RTOL, lattice, lattice_index
from hmf.core._serialise import cosmology_key
from hmf.core.accuracy import Accuracy, KAccuracy
from hmf.core.cache import DiskCache
from hmf.core.model import Model, qualified_name
from hmf.core.stage import Stage
from hmf.core.units import RHO_CRIT0_H2, rho_unit

# ---------------------------------------------------------------------------------
# Arrays
# ---------------------------------------------------------------------------------


def test_read_only_is_a_frozen_copy():
    a = np.array([1.0, 2.0])
    out = read_only(a)
    a[0] = 9.0
    assert out[0] == 1.0
    with pytest.raises(ValueError, match="read-only"):
        out[0] = 3.0
    assert read_only([1, 2]).dtype == np.float64


def test_read_only_keeps_a_quantity():
    q = [1.0, 2.0] * u.Mpc
    out = read_only(q)
    assert isinstance(out, u.Quantity)
    assert out.unit == u.Mpc
    assert not out.flags.writeable
    assert q.flags.writeable


def test_float_array_does_not_copy_a_float_array():
    a = np.arange(3.0)
    assert float_array(a) is a
    assert float_array([1, 2]).dtype == np.float64
    assert optional_float_array(None) is None
    np.testing.assert_array_equal(optional_float_array([1, 2]), [1.0, 2.0])


def test_dimensionless_floats():
    assert dimensionless_floats(1) == (1.0,)
    assert dimensionless_floats(np.array([1, 2])) == (1.0, 2.0)
    assert dimensionless_floats([50, 100] * u.percent) == (0.5, 1.0)
    with pytest.raises(u.UnitConversionError):
        dimensionless_floats([1, 2] * u.km)


# ---------------------------------------------------------------------------------
# Lattices
# ---------------------------------------------------------------------------------


def test_lattice_index_rounds_onto_nearby_nodes():
    # 0.3 / 0.1 is 2.9999999999999996 in floating point: it is node 3, either way.
    assert lattice_index(0.3, 0.1, up=True) == 3
    assert lattice_index(0.3, 0.1, up=False) == 3
    step = 0.02
    for i in (-500, -1, 0, 7, 875):
        for shift in (-0.5, 0.5):
            x = (i + shift * LATTICE_RTOL) * step
            assert lattice_index(x, step, up=True) == i
            assert lattice_index(x, step, up=False) == i


def test_lattice_index_between_nodes():
    assert lattice_index(0.25, 0.1, up=True) == 3
    assert lattice_index(0.25, 0.1, up=False) == 2
    assert lattice_index(-0.25, 0.1, up=True) == -2
    assert lattice_index(-0.25, 0.1, up=False) == -3
    # Further than the tolerance from a node: not on it.
    x = (4 + 10 * LATTICE_RTOL) * 0.1
    assert lattice_index(x, 0.1, up=True) == 5
    assert lattice_index(x, 0.1, up=False) == 4


def test_lattices_share_their_nodes_bit_for_bit():
    step = 0.02
    wide = lattice(-1000, 1000, step)
    narrow = lattice(-3, 17, step)
    np.testing.assert_array_equal(narrow, wide[1000 - 3 : 1000 + 18])
    assert lattice(0, 0, step)[0] == 0.0
    assert lattice(2, 4, 0.5).tolist() == [1.0, 1.5, 2.0]


def test_ln_a_grids_share_their_nodes_and_end_at_zero():
    a = kg.ln_a_grid(1e-8, 0.01)
    b = kg.ln_a_grid(1e-3, 0.01)
    assert a[-1] == 0.0
    assert not np.signbit(a[-1])
    np.testing.assert_array_equal(a[-b.size :], b)
    # The grid starts at the first node at or below a_min.
    assert a[0] <= np.log(1e-8) < a[1]
    # a_min on a node (to rounding) is the first node: no extra node below it.
    assert kg.ln_a_grid(np.exp(-0.3), 0.1).size == 4


# ---------------------------------------------------------------------------------
# Species
# ---------------------------------------------------------------------------------


def test_species_are_defined_once():
    assert _species.MATTER_SPECIES == ("cb", "tot")
    assert transfer_models.MATTER_SPECIES is _species.MATTER_SPECIES
    assert _boltzmann.MATTER_SPECIES is _species.MATTER_SPECIES
    assert transfer_models.Species is _species.Species
    assert transfer_models.check_species is _species.check_species
    assert _species.check_species("tot") == "tot"
    with pytest.raises(ValueError, match="species must be one of"):
        _species.check_species("nu")


def test_camb_columns_are_those_of_a_camb_transfer_file():
    # CAMB's columns: k, CDM, baryons, photons, massless nu, massive nu, total, no-nu.
    assert dict(_species.CAMB_COLUMNS) == {"tot": 6, "cb": 7}


#: Planck18's other parameters, for cosmologies that differ in their neutrinos.
_P18 = {"H0": Planck18.H0, "Om0": Planck18.Om0, "Ob0": Planck18.Ob0, "Tcmb0": Planck18.Tcmb0}


@pytest.mark.parametrize(
    "m_nu",
    [[0.0, 0.0, 0.06], [0.1, 0.1, 0.1], [0.0, 0.05, 0.1], [0.2, 0.2, 0.2]],
    ids=["planck18", "3x0.1", "0+0.05+0.1", "3x0.2"],
)
def test_tot_adds_the_massive_neutrino_density(m_nu):
    """rho_mean0(tot) - rho_mean0(cb) is the density of the massive neutrinos.

    Omega_nu h^2 = sum(m_nu) / 93.14 eV (e.g. Lesgourgues & Pastor 2006), for the
    standard decoupling (Neff = 3.046). Astropy instead scales each species' density
    by Neff / 3, and computes it with the Komatsu et al. (2011) fit, which differ
    from 93.14 eV by under 1% (0.3-0.6% here), the tolerance.
    """
    cosmo = FlatLambdaCDM(**_P18, m_nu=m_nu * u.eV)
    expected = sum(m_nu) / 93.14 / cosmo.h**2 * RHO_CRIT0_H2.to_value(rho_unit)
    diff = _species.rho_mean0(cosmo, "tot") - _species.rho_mean0(cosmo, "cb")
    assert diff == pytest.approx(expected, rel=1e-2)
    assert _species.rho_mean0(cosmo, "cb") == pytest.approx(
        (cosmo.Om0 * cosmo.critical_density0 / cosmo.h**2).to_value(u.Msun / u.Mpc**3), rel=1e-12
    )


@pytest.mark.parametrize("tcmb0", [Planck18.Tcmb0, 0 * u.K], ids=["massless", "no-neutrinos"])
def test_cb_is_tot_without_massive_neutrinos(tcmb0):
    cosmo = FlatLambdaCDM(**{**_P18, "Tcmb0": tcmb0}, m_nu=0 * u.eV)
    assert _species.rho_mean0(cosmo, "tot") == _species.rho_mean0(cosmo, "cb")
    z = np.array([0.0, 1.0, 100.0])
    np.testing.assert_array_equal(
        _species.omega_m(cosmo, z, "tot"), _species.omega_m(cosmo, z, "cb")
    )


def test_omega_m_of_cb_is_astropys():
    z = np.linspace(0, 10, 11)
    np.testing.assert_array_equal(_species.omega_m(Planck18, z), Planck18.Om(z))
    assert _species.omega_m(Planck18, 0.0, "tot") == pytest.approx(
        _species.omega_m0(Planck18, "tot"), rel=1e-15
    )
    with pytest.raises(ValueError, match="species"):
        _species.omega_m(Planck18, 0.0, "nu")


def test_omega_m_is_one_in_einstein_de_sitter():
    cosmo = FlatLambdaCDM(H0=70, Om0=1.0, Tcmb0=0)
    z = np.array([0.0, 0.5, 3.0, 1e3])
    for species in ("cb", "tot"):
        np.testing.assert_allclose(_species.omega_m(cosmo, z, species), 1.0, rtol=1e-14)


def test_omega_m_tends_to_one_at_high_z_in_flat_lcdm():
    """Without radiation, matter dominates flat LCDM at high z: Omega_m -> 1 from below.

    1 - Omega_m(z) = Omega_L / (Omega_m0 (1 + z)^3 + Omega_L).
    """
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=0)
    z = np.array([0.0, 1.0, 10.0, 100.0, 1e4])
    om = _species.omega_m(cosmo, z)
    assert np.all(np.diff(om) > 0)
    assert np.all(om < 1)
    assert 1 - om[-1] < 1e-11
    # With radiation, matter dominates only in between: Omega_m < 1, falling at high z.
    om = _species.omega_m(Planck18, np.array([10.0, 1e3, 1e5]), "tot")
    assert np.all((om > 0) & (om < 1))
    assert om[-1] < 0.1
    # A curved universe without dark energy too (Omega_k > 0 fades as (1 + z)^-1).
    open_cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.0, Tcmb0=0)
    assert 1 - _species.omega_m(open_cosmo, 1e6) < 1e-5


# ---------------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------------


def _citations() -> dict[str, str]:
    return {k: v for k, v in vars(_references).items() if k.isupper()}


def _models() -> list[type[Model]]:
    out = []
    for info in pkgutil.walk_packages(hmf.core.__path__, "hmf.core."):
        module = importlib.import_module(info.name)
        out.extend(
            obj
            for obj in vars(module).values()
            if inspect.isclass(obj) and issubclass(obj, Model) and obj.__module__ == info.name
        )
    return out


def test_every_model_cites_from_the_references_module():
    known = set(_citations().values())
    models = _models()
    assert len(models) > 50
    for model in models:
        unknown = set(model.references) - known
        assert not unknown, f"{model.__qualname__} cites {unknown} outside _references"


def test_no_paper_is_cited_under_two_strings():
    """Each paper (first author and year, or DOI/arXiv id) has one citation string."""
    by_paper: dict[object, str] = {}
    for name, citation in _citations().items():
        author_year = re.match(r"([^,]+), .*?(\d{4})\.", citation)
        assert author_year, name
        ids = re.findall(r"doi\.org/(\S+)|arXiv:(\S+)|adsabs\.harvard\.edu/abs/(\S+)", citation)
        assert ids, f"{name} has no DOI, arXiv or ADS link"
        for key in (author_year.groups(), *ids):
            assert key not in by_paper, f"{name} and {by_paper[key]} cite the same paper"
            by_paper[key] = name


# ---------------------------------------------------------------------------------
# Documented classes and qualified names
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [Model, Stage, Accuracy, DiskCache])
def test_documented_classes(cls):
    assert issubclass(cls, Documented)
    assert "fields_info" not in cls.__dict__


def test_documented_subclasses_get_parameters_and_fields_info():
    assert "Parameters\n" in DiskCache.__doc__
    assert [f.name for f in DiskCache.fields_info()] == ["directory"]
    assert "dln_k" in [f.name for f in KAccuracy.fields_info()]
    # The mixin adds no instance dict to slotted classes.
    assert not hasattr(KAccuracy(), "__dict__")


def test_one_qualified_name():
    assert cosmology_key(Planck18)[0] == qualified_name(type(Planck18))
    assert qualified_name(KAccuracy) == "hmf.core.accuracy:KAccuracy"


# ---------------------------------------------------------------------------------
# Critical density
# ---------------------------------------------------------------------------------


def test_critical_density():
    """rho_crit,0 / h^2 = 3 H^2 / (8 pi G) with H = 100 km/s/Mpc.

    Checked with G = 4.3009e-9 Mpc (km/s)^2 / Msun, independently of astropy's
    constants.
    """
    assert RHO_CRIT0_H2.unit == rho_unit
    g = 4.3009e-9
    expected = 3 * 100.0**2 / (8 * np.pi * g)
    assert RHO_CRIT0_H2.value == pytest.approx(expected, rel=1e-4)
    assert RHO_CRIT0_H2.value == pytest.approx(2.775366e11, rel=1e-6)
