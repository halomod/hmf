"""Tests of hmf.core.concentration: concentration-mass relations (Duffy et al. 2008)."""

import pickle

import attrs
import numpy as np
import pytest

from hmf.core import _references as refs
from hmf.core._kernels import mass_conversion as kern
from hmf.core.concentration import DUFFY08_PARAMETERS, ConcentrationModel, Duffy08
from hmf.core.domain import DomainError
from hmf.core.mass_definitions import SOCritical, SOMean, SOVirial
from hmf.core.units import Msun_h, UnitBoundaryError

M = np.geomspace(1e11, 1e15, 9)


def test_registered_and_cited():
    assert ConcentrationModel.get("Duffy08") is Duffy08
    assert isinstance(ConcentrationModel.coerce("Duffy08"), Duffy08)
    assert refs.DUFFY08 in Duffy08.references


def test_every_published_set_is_there():
    """Table 1, NFW: 3 definitions x 2 samples x 2 redshift ranges."""
    assert len(DUFFY08_PARAMETERS) == 12
    for (definition, sample, redshifts), (a, b, c) in DUFFY08_PARAMETERS.items():
        model = Duffy08(definition=definition, sample=sample, redshifts=redshifts)
        assert model.parameters == (a, b, c)
        # Concentration falls with mass and (for z = 0-2) with redshift; C = 0 at z = 0.
        assert -0.1 < b < -0.08
        assert (c == 0) is (redshifts == "0")
        assert c <= 0
    assert Duffy08().parameters == (5.71, -0.084, -0.47)
    assert Duffy08(definition="200m", sample="relaxed").parameters == (11.93, -0.090, -0.99)


def test_definitions():
    assert Duffy08().mass_definition == SOCritical(overdensity=200)
    assert Duffy08(definition="vir").mass_definition == SOVirial()
    assert Duffy08(definition="200m").mass_definition == SOMean(overdensity=200)


def test_power_law_at_the_pivot():
    """C = A (M / 2e12)^B (1 + z)^C: A at the pivot today, and slope B everywhere."""
    for model in (Duffy08(), Duffy08(definition="vir", sample="relaxed")):
        a, b, c = model.parameters
        assert model.concentration(2e12 * Msun_h, z=0.0) == pytest.approx(a, rel=1e-14)
        assert model.concentration(2e12 * Msun_h, z=1.0) == pytest.approx(a * 2**c, rel=1e-14)
        ln_c, slope = model.ln_concentration_and_slope_kernel(M, z=0.5)
        np.testing.assert_allclose(np.diff(ln_c) / np.diff(np.log(M)), b, rtol=1e-12)
        np.testing.assert_array_equal(slope, b)


def test_concentrations_are_physical():
    """Duffy et al. (2008), Sec. 3: the concentrations are as they describe them.

    c_200 ~ 4-8 at 1e11-1e15 today, c_mean nearly twice c_200, relaxed haloes about
    10% more concentrated, all falling with z.
    """
    c200 = Duffy08(redshifts="0").concentration(M * Msun_h, z=0.0)
    assert np.all((c200 > 3) & (c200 < 9))
    cmean = Duffy08(definition="200m", redshifts="0").concentration(M * Msun_h, z=0.0)
    assert np.all((cmean / c200 > 1.6) & (cmean / c200 < 2.0))
    relaxed = Duffy08(sample="relaxed", redshifts="0").concentration(M * Msun_h, z=0.0)
    assert np.all((relaxed / c200 > 1.05) & (relaxed / c200 < 1.25))
    z = np.array([0.0, 1.0, 2.0])
    assert np.all(np.diff(Duffy08().concentration(1e12 * Msun_h, z=z)) < 0)


@pytest.mark.parametrize("sample", ["full", "relaxed"])
@pytest.mark.parametrize("definition", ["vir", "200m"])
def test_duffy_relations_are_nfw_conversions_of_each_other(sample, definition):
    """Duffy's c_200 relation, converted by NFW to another definition, is their own there.

    It agrees to about their bootstrap errors (1-2%), over 1e11-1e15 at z = 0.

    They fit NFW profiles to the same haloes in each definition (Sec. 2.1), so the
    relations must agree, up to the scatter of the power-law fits; WMAP5 Omega_m.
    """
    omega_m = 0.258
    c200 = Duffy08(sample=sample, redshifts="0")
    other = Duffy08(definition=definition, sample=sample, redshifts="0")
    ln_c, slope = c200.ln_concentration_and_slope_kernel(M, z=0.0)
    ln_q = np.log(other.mass_definition.delta_halo_mean_kernel(omega_m)) - np.log(
        SOCritical().delta_halo_mean_kernel(omega_m)
    )
    ln_ratio, c_new, _, _ = kern.nfw_convert(np.exp(ln_c), slope, ln_q)
    expected = other.concentration(M * np.exp(ln_ratio) * Msun_h, z=0.0)
    np.testing.assert_allclose(c_new, expected, rtol=0.015)


def test_domains():
    assert Duffy08().calibration.contains(m=1e12 * Msun_h, z=1.5)
    assert not Duffy08(redshifts="0").calibration.contains(m=1e12 * Msun_h, z=1.5)
    assert not Duffy08().calibration.contains(m=1e16 * Msun_h, z=0.0)
    assert "Duffy et al. 2008" in Duffy08().calibration.source
    assert Duffy08.calibration_domain is Duffy08().calibration
    np.testing.assert_array_equal(
        Duffy08().in_calibration_domain_kernel(np.array([1e10, 1e12, 1e16]), z=0.0),
        [False, True, False],
    )
    with pytest.raises(DomainError, match="m in"):
        Duffy08().ln_concentration_and_slope_kernel(np.array([1e12, -1.0]), z=0.0)
    with pytest.raises(DomainError, match="z in"):
        Duffy08().concentration(1e12 * Msun_h, z=-0.5)
    with pytest.raises(UnitBoundaryError):
        Duffy08().concentration(1e12, z=0.0)


def test_invalid_options():
    for kwargs in ({"definition": "500c"}, {"sample": "all"}, {"redshifts": "0-6"}):
        with pytest.raises(ValueError, match="must be one of"):
            Duffy08(**kwargs)


def test_values_hash_and_pickle():
    model = Duffy08(sample="relaxed")
    assert model == Duffy08(sample="relaxed")
    assert hash(model) == hash(Duffy08(sample="relaxed"))
    assert model != Duffy08()
    assert pickle.loads(pickle.dumps(model)) == model
    assert attrs.evolve(model, definition="vir").mass_definition == SOVirial()


def test_scalar_rule_and_broadcasting():
    c = Duffy08().concentration(1e12 * Msun_h, z=0.0)
    assert isinstance(c, float)
    shape = Duffy08().concentration(M[None, :] * Msun_h, z=np.array([0.0, 1.0])[:, None]).shape
    assert shape == (2, M.size)
