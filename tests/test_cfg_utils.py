"""Tests of writing frameworks to TOML-able configs."""

import pytest
import toml

from hmf import MassFunction
from hmf.helpers.cfg_utils import framework_to_dict


@pytest.mark.parametrize("hmf_model", ["PS", "Tinker08"])
def test_unset_mdef_model_written_as_unset(hmf_model):
    """An unset mass definition stays unset, so it is re-resolved on reload (#374)."""
    mf = MassFunction(hmf_model=hmf_model, transfer_model="EH")
    dct = framework_to_dict(mf)
    assert dct["params"]["mdef_model"] is None
    # TOML has no null, so the key is dropped and reloads as the default (None).
    assert "mdef_model" not in toml.loads(toml.dumps(dct))["params"]


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
