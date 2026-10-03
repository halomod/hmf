"""Tests of writing frameworks to TOML-able configs."""

import tomllib
from datetime import UTC, datetime

import numpy as np
import pytest
import tomli_w

from hmf import MassFunction
from hmf.helpers.cfg_utils import framework_to_dict, to_toml_compatible


def _roundtrip(dct: dict) -> dict:
    return tomllib.loads(tomli_w.dumps(to_toml_compatible(dct)))


@pytest.mark.parametrize("hmf_model", ["PS", "Tinker08"])
def test_unset_mdef_model_written_as_unset(hmf_model):
    """An unset mass definition stays unset, so it is re-resolved on reload (#374)."""
    mf = MassFunction(hmf_model=hmf_model, transfer_model="EH")
    dct = framework_to_dict(mf)
    assert dct["params"]["mdef_model"] is None
    # TOML has no null, so the key is dropped and reloads as the default (None).
    assert "mdef_model" not in _roundtrip(dct)["params"]


def test_explicit_mdef_model_written():
    with pytest.warns(UserWarning, match="does not match the mass definition"):
        mf = MassFunction(
            hmf_model="PS",
            transfer_model="EH",
            mdef_model="SOCritical",
            disable_mass_conversion=False,
        )
    dct = framework_to_dict(mf)
    assert dct["params"]["mdef_model"] == "SOCritical"
    assert dct["params"]["mdef_params"] == {"overdensity": 200}


def test_to_toml_compatible_roundtrip():
    """NumPy values become plain values, and None is dropped, so TOML can write it."""
    now = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)
    dct = {
        "created_on": now,
        "params": {
            "z": np.float64(1.5),
            "n": np.int64(3),
            "flag": np.bool_(True),
            "m_nu": np.array([0.06, 0.0, 0.0]),
            "unset": None,
            "nested": {"value": np.float32(2.0), "unit": "K", "unset": None},
            "names": ("a", "b"),
        },
    }
    out = _roundtrip(dct)
    assert out == {
        "created_on": now,
        "params": {
            "z": 1.5,
            "n": 3,
            "flag": True,
            "m_nu": [0.06, 0.0, 0.0],
            "nested": {"value": 2.0, "unit": "K"},
            "names": ["a", "b"],
        },
    }
    assert type(out["params"]["n"]) is int
    assert type(out["params"]["flag"]) is bool


def test_framework_to_dict_toml_roundtrip():
    """A framework's config survives writing to and reading from TOML."""
    mf = MassFunction(transfer_model="EH", z=0.5, hmf_model="Tinker08")
    dct = framework_to_dict(mf)
    out = _roundtrip(dct)
    assert out["hmf_version"] == dct["hmf_version"]
    assert out["params"]["z"] == 0.5
    assert out["params"]["hmf_model"] == "Tinker08"
    assert out["params"]["transfer_model"] == "EH"
    assert out["params"]["cosmo_params"]["H0"]["value"] == pytest.approx(mf.cosmo.H0.value)
    assert out["params"]["cosmo_params"]["m_nu"]["value"] == pytest.approx(
        mf.cosmo.m_nu.value.tolist()
    )

    # Reloading the written params gives the same mass function.
    params = dict(out["params"])
    params.pop("cosmo_params")
    mf2 = MassFunction(**params)
    np.testing.assert_allclose(mf2.dndm, mf.dndm, rtol=1e-10)
