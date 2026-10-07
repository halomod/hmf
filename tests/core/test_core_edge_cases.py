"""Input validation, error paths and Boltzmann inputs of the transfer and growth code.

These are the cheap paths: no Boltzmann code is run here.
"""

import importlib.metadata
import math

import astropy.units as u
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM, FlatwCDM, Planck18, w0waCDM

from hmf.core import _boltzmann, _serialise
from hmf.core import growth_models as gm
from hmf.core import transfer_models as tm
from hmf.core._kernels import growth as kg
from hmf.core._kernels import transfer as kt
from hmf.core.accuracy import KAccuracy
from hmf.core.cache import DiskCache
from hmf.core.growth import Growth
from hmf.core.transfer import Transfer
from hmf.core.units import h_Mpc

# ---------------------------------------------------------------------------------
# Building the Boltzmann codes' input
# ---------------------------------------------------------------------------------


def test_camb_splits_distinct_neutrino_masses_into_eigenstates():
    """Distinct masses are separate eigenstates; the total density is unchanged.

    The neutrino density is fixed by the sum of the masses:
    Omega_nu h^2 = sum(m_nu) / 93.14 eV.
    """
    cosmo = FlatLambdaCDM(H0=67.7, Om0=0.31, Ob0=0.049, Tcmb0=2.7255, m_nu=[0, 0.05, 0.1] * u.eV)
    p = _boltzmann._camb_params(tm.CAMB().run_input(cosmo, KAccuracy()))
    assert p.nu_mass_eigenstates == 2
    assert list(p.nu_mass_numbers[:2]) == [1, 1]
    np.testing.assert_allclose(p.nu_mass_fractions[:2], [1 / 3, 2 / 3], rtol=1e-12)
    # Tolerance: CAMB's 93.14 eV conversion depends slightly on Tcmb and Neff (~0.1%).
    assert p.omnuh2 == pytest.approx(0.15 / 93.14, rel=5e-3)


def test_camb_settings_are_applied_by_path():
    model = tm.CAMB(settings={"Accuracy.AccuracyBoost": 1.5, "Transfer.high_precision": True})
    p = _boltzmann._camb_params(model.run_input(Planck18, KAccuracy()))
    assert p.Accuracy.AccuracyBoost == 1.5
    assert p.Transfer.high_precision


@pytest.mark.parametrize("name", ["Accuracy.NoSuchThing", "NoSuchGroup.AccuracyBoost", "Nope"])
def test_camb_unknown_setting_raises_when_built(name):
    """An unknown CAMBparams setting is a bad option: a ValueError, before CAMB runs."""
    pytest.importorskip("camb")
    with pytest.raises(ValueError, match=f"CAMBparams has no setting '{name}'"):
        tm.CAMB(settings={name: 1})


def test_camb_and_class_inputs_describe_dark_energy():
    cosmo = w0waCDM(H0=70, Om0=0.3, Ode0=0.7, w0=-0.9, wa=0.2, Ob0=0.05, Tcmb0=2.7255)
    camb_in = tm.CAMB().run_input(cosmo, KAccuracy())["cosmology"]
    assert (camb_in["w"], camb_in["wa"]) == (-0.9, 0.2)
    class_in = tm.CLASS().run_input(cosmo, KAccuracy())["class"]
    assert class_in["Omega_Lambda"] == 0.0
    assert (class_in["w0_fld"], class_in["wa_fld"]) == (-0.9, 0.2)
    flat = FlatwCDM(H0=70, Om0=0.3, w0=-0.8, Ob0=0.05, Tcmb0=2.7255)
    assert tm.CLASS().run_input(flat, KAccuracy())["class"]["wa_fld"] == 0.0


def test_class_neutrinos_conserve_neff():
    cosmo = FlatLambdaCDM(H0=67.7, Om0=0.31, Ob0=0.049, Tcmb0=2.7255, m_nu=[0, 0.05, 0.1] * u.eV)
    params = _boltzmann.class_cosmology_input(cosmo)
    deg = [float(x) for x in params["deg_ncdm"].split(",")]
    assert params["N_ncdm"] == 2
    assert params["N_ur"] + sum(deg) == pytest.approx(cosmo.Neff, rel=1e-12)


def test_finer_accuracy_samples_the_boltzmann_codes_more_densely():
    fine = KAccuracy(dln_k=0.01)
    camb_in = tm.CAMB().run_input(Planck18, fine)["transfer"]
    assert camb_in["k_per_logint"] == math.ceil(0.2 / 0.01)
    assert camb_in["high_precision"]
    assert tm.CAMB().run_input(Planck18, KAccuracy())["transfer"]["k_per_logint"] == 0
    class_in = tm.CLASS().run_input(Planck18, fine)["class"]
    assert class_in["k_per_decade_for_pk"] == math.ceil(0.2 * math.log(10) / 0.01)
    assert "k_per_decade_for_pk" not in tm.CLASS().run_input(Planck18, KAccuracy())["class"]
    # A user's own setting wins.
    user = tm.CLASS(class_params={"k_per_decade_for_pk": 7}).run_input(Planck18, fine)
    assert user["class"]["k_per_decade_for_pk"] == 7


# ---------------------------------------------------------------------------------
# The run memo and the disk cache, with a stand-in runner
# ---------------------------------------------------------------------------------


@pytest.fixture
def fake_runner(monkeypatch):
    calls = []

    def run(inputs):
        calls.append(inputs)
        k = np.geomspace(1e-4, 10, 8)
        return _boltzmann.make_run(
            "camb",
            k=k,
            transfer={"cb": 1 / (1 + k), "tot": 1 / (1 + k)},
            growth_z=[0.0, 1.0],
            growth={"cb": [1.0, 0.6], "tot": [1.0, 0.6]},
        )

    monkeypatch.setattr(_boltzmann, "_RUNNERS", {"camb": run})
    _boltzmann.clear_memo()
    yield calls
    _boltzmann.clear_memo()


def test_memo_keeps_only_the_latest_runs(fake_runner, monkeypatch):
    monkeypatch.setattr(_boltzmann, "MEMO_SIZE", 2)
    for i in (1, 2, 3, 1):
        _boltzmann.get_run("camb", {"i": i})
    # The run of 1 was evicted by 3, so it ran again.
    assert [c["i"] for c in fake_runner] == [1, 2, 3, 1]
    _boltzmann.get_run("camb", {"i": 3})
    assert len(fake_runner) == 4


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown Boltzmann code"):
        _boltzmann.get_run("cmbfast", {})


def test_incomplete_cache_entry_is_a_miss(fake_runner, tmp_path):
    cache = DiskCache(directory=tmp_path)
    key = _boltzmann.run_key("camb", {"i": 0})
    cache.store(key, {"k": np.ones(3)})  # readable, but not a run
    run = _boltzmann.get_run("camb", {"i": 0}, disk_cache=cache)
    assert len(fake_runner) == 1
    assert set(run.transfer) == {"cb", "tot"}
    assert set(cache.load(key)) >= {"k", "transfer_cb", "growth_tot"}


def test_failed_cache_write_leaves_no_temp_file(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez", broken)
    cache = DiskCache(directory=tmp_path)
    with pytest.raises(OSError, match="disk full"):
        cache.store("abc", {"k": np.ones(3)})
    assert list((tmp_path / "boltzmann").iterdir()) == []


def test_unknown_package_version(monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    assert _boltzmann._version("camb") == "unknown"


# ---------------------------------------------------------------------------------
# Canonical serialisation
# ---------------------------------------------------------------------------------


def test_canonical_handles_every_supported_type(tmp_path):
    from pathlib import Path

    obj = {
        "path": Path("a/b"),
        "scalar": np.float32(0.5),
        "array": np.arange(3),
        "quantity": [1, 2] * u.Mpc,
        "unit": h_Mpc,
        "cosmology": Planck18,
        "model": tm.BBKS(),
        "nested": ({"x": None},),
    }
    out = _serialise.canonical(obj)
    assert out["path"] == "a/b"
    assert out["scalar"] == 0.5
    assert out["array"] == {"__ndarray__": [0, 1, 2], "dtype": np.arange(3).dtype.str}
    assert out["quantity"]["unit"] == "Mpc"
    assert "littleh" in out["unit"]["__unit__"]
    assert out["cosmology"] == _serialise.canonical(Planck18.clone(name="x"))
    assert out["model"]["__class__"] == "hmf.core.transfer_models:BBKS"
    assert out["nested"] == [{"x": None}]
    with pytest.raises(TypeError, match="collide"):
        _serialise.canonical({1: "a", "1": "b"})


def test_plain_converts_sequences():
    assert _serialise._plain((1, np.float64(2.5), [3 * u.eV])) == (1, 2.5, ((3.0, "eV"),))


# ---------------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------------


def test_tabulate_transfer_validates_its_table():
    scales = tm._eh98_scales(Planck18)
    k = np.geomspace(1e-3, 1, 10)
    with pytest.raises(ValueError, match="same length"):
        kt.tabulate_transfer(k, np.ones(9), scales)
    with pytest.raises(ValueError, match="n_slope"):
        kt.tabulate_transfer(k, np.ones(10), scales, n_slope=1)
    with pytest.raises(ValueError, match="at least"):
        kt.tabulate_transfer(k[:3], np.ones(3), scales)
    with pytest.raises(ValueError, match="increasing"):
        kt.tabulate_transfer(k[::-1], np.ones(10), scales)
    with pytest.raises(ValueError, match="positive"):
        kt.tabulate_transfer(k, -np.ones(10), scales)
    table = kt.tabulate_transfer(k, np.ones(10), scales)
    assert table.k_min == pytest.approx(1e-3)
    assert table.k_max == pytest.approx(1.0)


def test_growth_kernels_validate_and_report_range():
    with pytest.raises(ValueError, match="odd number"):
        kg.solve_growth_ode(np.zeros(4), np.zeros(4), np.zeros(4), 1.0, 1.0)
    ln_a = kg.ln_a_grid(1e-3, 0.01)
    assert ln_a[-1] == 0.0
    table = kg.tabulate_growth(ln_a, np.exp(ln_a))
    assert table.z_max == pytest.approx(1 / np.exp(ln_a[0]) - 1)


# ---------------------------------------------------------------------------------
# Model and stage validation
# ---------------------------------------------------------------------------------


def test_transfer_model_field_validation():
    with pytest.raises(u.UnitConversionError):
        tm.CAMB(k_max=1 / u.Mpc)
    with pytest.raises(ValueError, match="> 0"):
        tm.CAMB(tail_decay_ln_k=-1)
    with pytest.raises(TypeError, match="bool, int, float or str"):
        tm.CAMB(settings={"Accuracy.AccuracyBoost": [1, 2]})
    assert tm.CAMB(settings=None).settings == ()
    with pytest.raises(ValueError, match="same length"):
        tm.FromArray(k=np.arange(1, 6) * h_Mpc, t=np.ones(5), t_tot=np.ones(4))
    model = tm.FromArray(k=[1, 2, 3, 4] * h_Mpc, t=np.ones(4) * u.dimensionless_unscaled)
    assert model.t == (1.0, 1.0, 1.0, 1.0)


def test_fromfile_rejects_unknown_formats(tmp_path):
    np.savetxt(tmp_path / "three.dat", np.ones((5, 3)))
    with pytest.raises(ValueError, match="columns"):
        tm.FromFile(fname=tmp_path / "three.dat").solve(Planck18, KAccuracy())


def test_transfer_solution_and_stage_properties():
    t = Transfer(model="EH")
    assert t.solution.species == ("cb", "tot")
    assert t.k_max_table is None
    assert t.boltzmann_run is None


def test_growth_model_field_validation():
    with pytest.raises(ValueError, match="> 0"):
        gm.ODEGrowth(dln_a=0)
    with pytest.raises(ValueError, match="a_min must be < 1"):
        gm.ODEGrowth(a_min=2.0)
    z = np.linspace(0, 3, 5)
    with pytest.raises(ValueError, match="same length"):
        gm.FromArray(z=z, d=np.ones(4))
    with pytest.raises(ValueError, match="positive"):
        gm.FromArray(z=z, d=-np.ones(5))
    with pytest.raises(TypeError, match="GrowthModel"):
        Growth(model=3)


def test_missing_classy_names_the_extra(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "classy", None)
    with pytest.raises(ImportError, match=r"hmf\[class\]"):
        _boltzmann._import_classy()
