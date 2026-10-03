"""Tests for loading externally-defined models by import path and via ``plugins``."""

import re
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import toml
from click.testing import CliRunner

from hmf import MassFunction
from hmf._cli import main
from hmf._internals._framework import Component, get_mdl
from hmf.helpers.cfg_utils import framework_to_dict
from hmf.mass_function import PS

# An external fitting function that is exactly ``scale`` times Press-Schechter, so
# that dndm must be exactly ``scale`` times the PS dndm.
EXTERNAL_MODULE = """
import typing

from hmf.mass_function import PS, SMT
from hmf.mass_function.fitting_functions import BaseFittingFunction


class ScaledPS(BaseFittingFunction):
    _defaults: typing.ClassVar = {"scale": 0.5}
    # A fit with no sim_definition (like PS) can't round-trip through a written
    # config, because the default mass definition is then written explicitly.
    sim_definition = SMT.sim_definition
    req_sigma = False
    req_z = False
    normalized = False

    @property
    def fsigma(self):
        return self.params["scale"] * PS.fsigma.fget(self)


class NotAFit:
    pass
"""

TRANSFER = {"transfer_model": "EH"}


def _all_pluggable_bases():
    """Yield every class that owns a plugin registry."""
    stack = [Component]
    while stack:
        cls = stack.pop()
        if "_plugins" in cls.__dict__:
            yield cls
        stack.extend(cls.__subclasses__())


@pytest.fixture
def ext_module(request) -> str:
    """Write an importable module defining an external model.

    The plugin registries and ``sys.modules`` are restored afterwards, so the
    model does not leak into other tests.
    """
    tmp_path: Path = request.getfixturevalue("tmp_path")
    monkeypatch = request.getfixturevalue("monkeypatch")
    name = "_hmf_ext_" + re.sub(r"\W", "_", request.node.name)
    (tmp_path / f"{name}.py").write_text(EXTERNAL_MODULE)
    monkeypatch.syspath_prepend(str(tmp_path))

    registries = {base: dict(base._plugins) for base in _all_pluggable_bases()}
    try:
        yield name
    finally:
        for base, plugins in registries.items():
            base._plugins.clear()
            base._plugins.update(plugins)
        sys.modules.pop(name, None)


@pytest.fixture
def ps_dndm():
    return MassFunction(hmf_model=PS, **TRANSFER).dndm


@pytest.mark.parametrize("sep", [":", "."])
def test_import_path_in_framework(ext_module, ps_dndm, sep):
    mf = MassFunction(
        hmf_model=f"{ext_module}{sep}ScaledPS", hmf_params={"scale": 0.25}, **TRANSFER
    )
    assert mf.hmf_model.__name__ == "ScaledPS"
    assert mf.hmf_model.__module__ == ext_module
    np.testing.assert_allclose(mf.dndm, 0.25 * ps_dndm, rtol=1e-10)


def test_import_path_registers_short_name(ext_module):
    cls = get_mdl(f"{ext_module}:ScaledPS", "BaseFittingFunction")
    assert get_mdl("ScaledPS", "BaseFittingFunction") is cls


def test_import_path_without_kind(ext_module):
    cls = get_mdl(f"{ext_module}.ScaledPS")
    assert cls.__name__ == "ScaledPS"


def test_import_path_not_a_subclass(ext_module):
    with pytest.raises(TypeError, match="is not a subclass of BaseFittingFunction"):
        MassFunction(hmf_model=f"{ext_module}:NotAFit")


def test_import_path_wrong_kind(ext_module):
    with pytest.raises(TypeError, match="is not a subclass of BaseFilter"):
        MassFunction(filter_model=f"{ext_module}:ScaledPS")


def test_import_path_missing_module():
    with pytest.raises(ValueError, match="Could not import module '_hmf_no_such_module'"):
        get_mdl("_hmf_no_such_module:Cls", "BaseFittingFunction")


def test_import_path_missing_attribute(ext_module):
    with pytest.raises(ValueError, match="has no attribute 'Nope'"):
        get_mdl(f"{ext_module}:Nope", "BaseFittingFunction")


def test_unknown_name_lists_available_and_suggests():
    with pytest.raises(ValueError, match="not a defined BaseFittingFunction model") as exc:
        get_mdl("Tinker8", "BaseFittingFunction")
    msg = str(exc.value)
    assert "Did you mean" in msg
    assert "'Tinker08'" in msg
    assert "'PS'" in msg  # available names are listed
    assert "package.module:Class" in msg  # import-path form is suggested


def test_unknown_name_without_close_match_has_no_suggestion():
    with pytest.raises(ValueError, match="not a defined BaseFittingFunction model") as exc:
        get_mdl("Zzzzzzzzzz", "BaseFittingFunction")
    assert "Did you mean" not in str(exc.value)


def test_unknown_name_without_kind_suggests():
    with pytest.raises(ValueError, match="No model found with name 'Tinkr08'") as exc:
        get_mdl("Tinkr08")
    assert "'Tinker08'" in str(exc.value)


def _run(tmp_path: Path, cfg: str, outdir: Path):
    outdir.mkdir()
    cfgfile = tmp_path / f"{outdir.name}.toml"
    cfgfile.write_text(textwrap.dedent(cfg))
    result = CliRunner().invoke(main, ["run", "-i", str(cfgfile), "-o", str(outdir)])
    assert result.exit_code == 0, result.output
    return np.genfromtxt(outdir / "hmf_dndm.txt")


def test_cli_import_path(tmp_path, ext_module, ps_dndm):
    cfg = f"""
    [params]
    transfer_model = "EH"
    hmf_model = "{ext_module}:ScaledPS"
    hmf_params = {{scale = 0.25}}
    """
    dndm = _run(tmp_path, cfg, tmp_path / "out")
    np.testing.assert_allclose(dndm, 0.25 * ps_dndm, rtol=1e-6)


def test_cli_plugins_key(tmp_path, ext_module, ps_dndm):
    cfg = f"""
    plugins = ["{ext_module}"]
    [params]
    transfer_model = "EH"
    hmf_model = "ScaledPS"
    """
    dndm = _run(tmp_path, cfg, tmp_path / "out")
    # Default scale is 0.5
    np.testing.assert_allclose(dndm, 0.5 * ps_dndm, rtol=1e-6)

    written = toml.load(tmp_path / "out" / "hmf_cfg.toml")
    assert written["plugins"] == [ext_module]
    assert written["params"]["hmf_model"] == "ScaledPS"


def test_cli_plugins_roundtrip_in_fresh_registry(tmp_path, ext_module, ps_dndm):
    """A written config must reload even when the plugin was not imported yet."""
    cfg = f"""
    [params]
    transfer_model = "EH"
    hmf_model = "{ext_module}:ScaledPS"
    """
    first = _run(tmp_path, cfg, tmp_path / "first")

    # Forget the plugin, as a fresh Python process would.
    for base in _all_pluggable_bases():
        base._plugins.pop("ScaledPS", None)
    sys.modules.pop(ext_module)

    written = (tmp_path / "first" / "hmf_cfg.toml").read_text()
    second = _run(tmp_path, written, tmp_path / "second")
    np.testing.assert_allclose(second, first, rtol=1e-10)
    np.testing.assert_allclose(second, 0.5 * ps_dndm, rtol=1e-6)


def test_cli_plugins_must_be_list_of_str(tmp_path):
    cfgfile = tmp_path / "cfg.toml"
    cfgfile.write_text("plugins = 3\n")
    result = CliRunner().invoke(main, ["run", "-i", str(cfgfile), "-o", str(tmp_path)])
    assert result.exit_code != 0
    assert isinstance(result.exception, TypeError)
    assert "plugins" in str(result.exception)


def test_framework_to_dict_plugins(ext_module):
    mf = MassFunction(hmf_model=f"{ext_module}:ScaledPS", **TRANSFER)
    assert framework_to_dict(mf)["plugins"] == [ext_module]
    assert framework_to_dict(mf, plugins=["foo.bar"])["plugins"] == ["foo.bar", ext_module]


def test_framework_to_dict_no_plugins_for_builtin_models():
    assert "plugins" not in framework_to_dict(MassFunction(**TRANSFER))


@pytest.mark.parametrize("path", [":Cls", "mod:", ".Cls"])
def test_import_path_malformed(path):
    with pytest.raises(ValueError, match="Could not interpret"):
        get_mdl(path, "BaseFittingFunction")


def test_cli_plugins_single_string(tmp_path, ext_module, ps_dndm):
    cfg = f"""
    plugins = "{ext_module}"
    [params]
    transfer_model = "EH"
    hmf_model = "ScaledPS"
    """
    dndm = _run(tmp_path, cfg, tmp_path / "out")
    np.testing.assert_allclose(dndm, 0.5 * ps_dndm, rtol=1e-6)
    assert toml.load(tmp_path / "out" / "hmf_cfg.toml")["plugins"] == [ext_module]
