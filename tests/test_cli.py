from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from hmf._cli import main


@pytest.fixture
def tmpdir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("cli-tests")


@pytest.mark.filterwarnings("ignore:'extrapolate_with_eh' was not set")
@pytest.mark.filterwarnings("ignore:matter_species was not set")
def test_no_config_or_args(tmpdir: Path):
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--outdir", str(tmpdir)])
    assert result.exit_code == 0

    assert (tmpdir / "hmf_m.txt").exists()
    assert (tmpdir / "hmf_dndm.txt").exists()
    assert (tmpdir / "hmf_cfg.toml").exists()


def test_with_config(tmpdir: Path):
    runner = CliRunner()

    cfg = """
    framework = "hmf.MassFunction"
    quantities = ["m", "dndm", "sigma"]
    [params]
    z = 1.0
    transfer_model = 'EH'
    """

    with (tmpdir / "cfg.toml").open("w") as fl:
        fl.write(cfg)

    result = runner.invoke(main, ["run", "-i", str(tmpdir / "cfg.toml"), "-o", str(tmpdir)])
    print(result.stdout)
    assert result.exit_code == 0


def test_config_vs_cli(tmpdir: Path):
    cfgdir = tmpdir / "cfg"
    cfgdir.mkdir()

    clidir = tmpdir / "cli"
    clidir.mkdir()

    runner = CliRunner()

    cfg = """
    framework = "hmf.MassFunction"
    quantities = ["m", "dndm"]

    [params]
    z = 1.0
    transfer_model = 'EH'
    """

    with (tmpdir / "cfg.toml").open("w") as fl:
        fl.write(cfg)

    result_cfg = runner.invoke(main, ["run", "-i", str(tmpdir / "cfg.toml"), "-o", str(cfgdir)])
    result_cli = runner.invoke(
        main, ["run", "-o", str(clidir), "--", "--z=1.0", '--transfer_model="EH"']
    )

    assert result_cfg.exit_code == 0
    assert result_cli.exit_code == 0

    dndm_cfg = np.genfromtxt(cfgdir / "hmf_dndm.txt")
    dndm_cli = np.genfromtxt(clidir / "hmf_dndm.txt")

    assert np.allclose(dndm_cfg, dndm_cli)


def test_cli_value_containing_equals(tmpdir: Path):
    """A ``--key=value`` option keeps the whole value, even when it contains "="."""
    cfgdir = tmpdir / "cfg"
    cfgdir.mkdir()

    clidir = tmpdir / "cli"
    clidir.mkdir()

    runner = CliRunner()

    cfg = """
    [params]
    hmf_model = 'ST'
    hmf_params = {A = 0.3}
    transfer_model = 'EH'
    """

    with (tmpdir / "cfg.toml").open("w") as fl:
        fl.write(cfg)

    result_cfg = runner.invoke(main, ["run", "-i", str(tmpdir / "cfg.toml"), "-o", str(cfgdir)])
    result_cli = runner.invoke(
        main,
        [
            "run",
            "-o",
            str(clidir),
            "--",
            '--hmf_model="ST"',
            "--hmf_params=dict(A=0.3)",
            '--transfer_model="EH"',
        ],
    )

    assert result_cfg.exit_code == 0, result_cfg.output
    assert result_cli.exit_code == 0, result_cli.output

    np.testing.assert_allclose(
        np.genfromtxt(clidir / "hmf_dndm.txt"), np.genfromtxt(cfgdir / "hmf_dndm.txt"), rtol=1e-10
    )


def test_list_of_parameters(tmpdir):
    runner = CliRunner()

    result_cli = runner.invoke(
        main,
        ["run", "-o", str(tmpdir), "--", "--z=[0.0,1.0,2.0]", '--transfer_model="EH"'],
    )

    assert result_cli.exit_code == 0

    assert (tmpdir / "z=0.0_m.txt").exists()
    assert (tmpdir / "z=1.0_dndm.txt").exists()


def test_roundtrip_cfg(tmpdir):
    runner = CliRunner()

    cfg = """
        framework = "hmf.MassFunction"
        quantities = ["m", "dndm", "sigma"]
        [params]
        z = 1.0
        transfer_model = 'EH'
        """

    with (tmpdir / "cfg.toml").open("w") as fl:
        fl.write(cfg)

    result = runner.invoke(main, ["run", "-i", str(tmpdir / "cfg.toml"), "-o", str(tmpdir)])

    assert result.exit_code == 0

    clidir = tmpdir / "cli"
    clidir.mkdir()

    result2 = runner.invoke(main, ["run", "-i", str(tmpdir / "hmf_cfg.toml"), "-o", str(clidir)])

    assert result2.exit_code == 0

    first = np.genfromtxt(tmpdir / "hmf_dndm.txt")
    second = np.genfromtxt(clidir / "hmf_dndm.txt")

    assert np.allclose(first, second)


@pytest.mark.parametrize("hmf_model", ["PS", "Tinker08"])
def test_roundtrip_cfg_default_mdef(tmpdir, hmf_model):
    """A written config must re-run when the mass definition was left unset.

    PS has no measured mass definition, so an unset ``mdef_model`` falls back to
    SOMean. Writing that fallback out explicitly made the re-run fail (#374).
    """
    runner = CliRunner()

    cfg = f"""
        [params]
        hmf_model = '{hmf_model}'
        transfer_model = 'EH'
        """

    with (tmpdir / "cfg.toml").open("w") as fl:
        fl.write(cfg)

    first = tmpdir / "first"
    first.mkdir()
    result = runner.invoke(main, ["run", "-i", str(tmpdir / "cfg.toml"), "-o", str(first)])
    assert result.exit_code == 0, result.output

    second = tmpdir / "second"
    second.mkdir()
    result2 = runner.invoke(main, ["run", "-i", str(first / "hmf_cfg.toml"), "-o", str(second)])
    assert result2.exit_code == 0, result2.output

    np.testing.assert_allclose(
        np.genfromtxt(second / "hmf_dndm.txt"), np.genfromtxt(first / "hmf_dndm.txt"), rtol=1e-10
    )


def test_ctx_to_dct_bare_string_with_equals():
    """Bare string option values containing '=' are preserved as strings without SyntaxError."""
    from hmf._cli import _ctx_to_dct

    res = _ctx_to_dct(["--z=1.0", "--hmf_model=ST", "--custom_tag=foo=bar"])
    assert res == {"z": 1.0, "hmf_model": "ST", "custom_tag": "foo=bar"}


def test_ctx_to_dct_safe_eval():
    """Verify that _ctx_to_dct safely parses literals without code execution."""
    from hmf._cli import _ctx_to_dct

    # Numbers and booleans
    assert _ctx_to_dct(["--z=1.0", "--flag=True"]) == {"z": 1.0, "flag": True}
    # Builtin names must not evaluate to functions
    assert _ctx_to_dct(["--hmf_model=sum"]) == {"hmf_model": "sum"}
    # Import-path models containing colons and dots must not raise SyntaxError
    assert _ctx_to_dct(["--hmf_model=hmf.mass_function.fitting_functions:PS"]) == {
        "hmf_model": "hmf.mass_function.fitting_functions:PS"
    }
    # Arbitrary code strings must remain unexecuted strings
    assert _ctx_to_dct(["--z=__import__('os').getcwd()"]) == {"z": "__import__('os').getcwd()"}
    # Dicts from dict() call or JSON-like dict syntax
    assert _ctx_to_dct(["--hmf_params=dict(A=0.3)"]) == {"hmf_params": {"A": 0.3}}
    assert _ctx_to_dct(['--hmf_params={"A": 0.3}']) == {"hmf_params": {"A": 0.3}}
