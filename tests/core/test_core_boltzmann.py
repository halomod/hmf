"""Tests of the Boltzmann runs behind the transfer and growth stages.

* One run per distinct (model, cosmology, accuracy) input, exposing every species,
  shared by the transfer and growth stages (call counters), at the stages' one
  ``KAccuracy``.
* Massive neutrinos: P_cb and P_tot from one run, and the small-scale suppression of
  the total-matter power against bounds around the -8 f_nu rule (Hu, Eisenstein &
  Tegmark 1998).
* The opt-in disk cache: hits and misses, keys, atomic writes.
"""

import sys

import astropy.units as u
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM, Planck18

from hmf.core import _boltzmann
from hmf.core._serialise import canonical_json, content_hash, cosmology_key
from hmf.core.accuracy import KAccuracy
from hmf.core.cache import DiskCache, default_cache_dir
from hmf.core.growth import Growth
from hmf.core.transfer import Transfer
from hmf.core.transfer_models import CAMB
from hmf.core.units import h_Mpc

# A cosmology no other test module uses, so that runs counted here are fresh.
COSMO = FlatLambdaCDM(
    H0=68.1, Om0=0.302, Ob0=0.0481, Tcmb0=2.7255, m_nu=[0, 0, 0.06] * u.eV, name="test"
)


@pytest.fixture
def fresh():
    """Clear the in-process memo, and count runs from here."""
    _boltzmann.clear_memo()
    before = dict(_boltzmann.run_counts())
    yield lambda backend="camb": _boltzmann.run_counts().get(backend, 0) - before.get(backend, 0)
    _boltzmann.clear_memo()


# ---------------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------------


def test_every_species_comes_from_one_run(fresh):
    t = Transfer(cosmology=COSMO, model="CAMB")
    k = np.logspace(-3, 1, 10) * h_Mpc
    for species in ("cb", "tot"):
        t.transfer_function(k, species)
        t.unnormalised_power(k, species)
        t.power_source(species).power_kernel(np.array([0.1]))
    assert fresh() == 1


def test_stages_with_the_same_run_input_share_one_run(fresh):
    t = Transfer(cosmology=COSMO, model="CAMB")
    t.transfer_function(1 * h_Mpc)
    # n_s, the extrapolation and the disk cache don't change the run.
    t.evolve(n_s=0.9).transfer_function(1 * h_Mpc)
    Transfer(cosmology=COSMO, model=CAMB(tail_decay_ln_k=2.0)).transfer_function(1 * h_Mpc)
    assert fresh() == 1
    # A new cosmology, model setting or accuracy does.
    t.evolve(cosmology=COSMO.clone(H0=70)).transfer_function(1 * h_Mpc)
    t.evolve(model=CAMB(k_max=10 * h_Mpc)).transfer_function(1 * h_Mpc)
    t.evolve(k_accuracy=KAccuracy.high()).transfer_function(1 * h_Mpc)
    assert fresh() == 4


def test_growth_shares_the_transfer_run(fresh):
    t = Transfer(cosmology=COSMO, model=CAMB(k_max=10 * h_Mpc))
    g = Growth.from_transfer(t, model="CAMB")
    g.growth_factor(np.array([0.0, 1.0, 5.0]), "cb")
    g.growth_rate(2.0, "tot")
    t.transfer_function(1 * h_Mpc, "tot")
    assert fresh() == 1
    assert g.solution.run is t.boltzmann_run


def test_growth_without_a_transfer_stage_shares_a_default_one(fresh):
    """A standalone CambGrowth makes the run of CAMB() at default accuracy."""
    Growth(cosmology=COSMO, model="CAMB").growth_factor(1.0)
    Transfer(cosmology=COSMO, model="CAMB").transfer_function(1 * h_Mpc)
    assert fresh() == 1


def test_growth_runs_at_its_own_k_accuracy(fresh):
    """KAccuracy.high() refines a growth model's own run too, as it does the transfer's.

    The growth's own run at high accuracy is the run of a high-accuracy transfer
    stage (one run for both), not the default one; a growth stage whose accuracy
    differs from its transfer stage's does not take that stage's run.
    """
    high = KAccuracy.high()
    growth = Growth(cosmology=COSMO, model="CAMB", k_accuracy=high)
    growth.growth_factor(1.0)
    transfer_high = Transfer(cosmology=COSMO, model="CAMB", k_accuracy=high)
    transfer_high.transfer_function(1 * h_Mpc)
    assert fresh() == 1
    assert growth.solution.run is transfer_high.boltzmann_run

    transfer = Transfer(cosmology=COSMO, model="CAMB")
    mixed = Growth(cosmology=COSMO, model="CAMB", transfer=transfer, k_accuracy=high)
    assert mixed.solution.run is transfer_high.boltzmann_run
    assert fresh() == 1  # the default transfer's run is not needed
    assert Growth.from_transfer(transfer_high, model="CAMB").k_accuracy == high


def test_growth_needing_higher_z_than_the_run_makes_its_own(fresh):
    t = Transfer(cosmology=COSMO, model=CAMB(z_max=5.0))
    g = Growth.from_transfer(t, model="CAMB")
    assert g.growth_factor(15.0) < 1
    t.transfer_function(1 * h_Mpc)
    assert fresh() == 2


def test_growth_of_another_code_does_not_use_the_transfer_run(fresh):
    t = Transfer(cosmology=COSMO, model="CAMB")
    Growth.from_transfer(t).growth_factor(1.0)  # ODE: no run at all
    assert fresh() == 0


def test_growth_cosmology_must_match_the_transfer():
    t = Transfer(cosmology=COSMO, model="EH")
    with pytest.raises(ValueError, match="cosmology"):
        Growth(cosmology=Planck18, transfer=t)


def test_runs_are_read_only(fresh):
    run = Transfer(cosmology=COSMO, model="CAMB").boltzmann_run
    with pytest.raises(ValueError, match="read-only"):
        run.k[0] = 1.0
    with pytest.raises(TypeError):
        run.transfer["cb"] = run.k


# ---------------------------------------------------------------------------------
# Massive neutrinos
# ---------------------------------------------------------------------------------


def _nu_cosmo(m_nu):
    """Flat LCDM with fixed total Omega_m h^2 (CDM + baryons + neutrinos)."""
    h, om_total, ob = 0.677, 0.31, 0.049
    omega_nu = m_nu / 93.14 / h**2
    return FlatLambdaCDM(
        H0=100 * h, Om0=om_total - omega_nu, Ob0=ob, Tcmb0=2.7255, m_nu=[0, 0, m_nu] * u.eV
    )


@pytest.fixture(scope="module")
def nu_runs():
    massive, massless = _nu_cosmo(0.3), _nu_cosmo(0.0)
    f_nu = (0.3 / 93.14 / 0.677**2) / 0.31
    return (
        Transfer(cosmology=massive, model="CAMB"),
        Transfer(cosmology=massless, model="CAMB"),
        f_nu,
    )


def test_cb_and_tot_differ_on_small_scales_only(nu_runs):
    t, _, f_nu = nu_runs
    k = np.array([1e-4, 1e-3]) * h_Mpc
    # Above the free-streaming scale both fields are the same.
    # Tolerance: measured 2.4e-5 at k = 1e-3 h/Mpc.
    np.testing.assert_allclose(
        t.transfer_function(k, "tot"), t.transfer_function(k, "cb"), rtol=1e-4
    )
    # Far below it the neutrinos don't cluster: delta_tot = (1 - f_nu) delta_cb.
    ratio = t.unnormalised_power(5 * h_Mpc, "tot") / t.unnormalised_power(5 * h_Mpc, "cb")
    # Tolerance: residual neutrino clustering at k = 5 h/Mpc; measured 8e-5.
    assert ratio == pytest.approx((1 - f_nu) ** 2, rel=1e-3)


@pytest.mark.parametrize("species", ["tot", "cb"])
def test_small_scale_suppression_follows_the_8_f_nu_rule(nu_runs, species):
    """At fixed Omega_m h^2 and A_s, small-scale power is suppressed by ~8 f_nu (HET98).

    The -8 f_nu rule is the leading order in f_nu; the suppression grows faster than
    linearly (it is a power of the growth since neutrinos became non-relativistic), so
    for 0.3 eV it is larger. The cb field is less suppressed than total matter, by
    the factor (1 - f_nu)^2 (about 2 f_nu).
    """
    massive, massless, f_nu = nu_runs
    k = np.array([3.0, 5.0]) * h_Mpc
    # The unnormalised P is k^n_s T^2, and A_s is the same, so the ratio is the
    # suppression.
    supp = massive.unnormalised_power(k, species) / massless.unnormalised_power(k, "tot") - 1
    # Bounds: measured -9.6 f_nu (tot) and -8.0 f_nu (cb) for f_nu = 0.023. The
    # bounds allow for the rule's higher-order terms, and still fail if the
    # neutrinos were not suppressing power (0) or were counted twice (~ -20 f_nu).
    lo, hi = (-11.0, -7.0) if species == "tot" else (-9.5, -6.0)
    assert np.all((supp / f_nu > lo) & (supp / f_nu < hi))


# ---------------------------------------------------------------------------------
# The disk cache
# ---------------------------------------------------------------------------------


def test_disk_cache_miss_then_hit(fresh, tmp_path):
    cache = DiskCache(directory=tmp_path)
    t = Transfer(cosmology=COSMO, model="CAMB", disk_cache=cache)
    first = t.transfer_function(np.logspace(-3, 2, 30) * h_Mpc, "tot")
    assert fresh() == 1
    files = list((tmp_path / "boltzmann").glob("*.npz"))
    assert len(files) == 1
    assert not list((tmp_path / "boltzmann").glob(".*tmp"))  # no temp files left

    _boltzmann.clear_memo()  # as a new process would
    again = Transfer(cosmology=COSMO, model="CAMB", disk_cache=cache)
    np.testing.assert_array_equal(
        again.transfer_function(np.logspace(-3, 2, 30) * h_Mpc, "tot"), first
    )
    assert fresh() == 1  # loaded, not re-run


def test_disk_cache_is_off_by_default_and_not_part_of_the_value(tmp_path):
    assert Transfer().disk_cache is None
    assert Transfer(disk_cache=tmp_path) == Transfer()
    assert Transfer(disk_cache=True).disk_cache == DiskCache()
    assert Transfer(disk_cache=False).disk_cache is None


def test_default_cache_dir(monkeypatch, tmp_path):
    import platformdirs

    monkeypatch.setenv("HMF_CACHE_DIR", str(tmp_path / "a"))
    assert default_cache_dir() == tmp_path / "a"
    monkeypatch.delenv("HMF_CACHE_DIR")
    assert default_cache_dir() == platformdirs.user_cache_path("hmf", appauthor=False)
    assert default_cache_dir().name == "hmf"


@pytest.mark.skipif(sys.platform != "linux", reason="XDG_CACHE_HOME is Linux's")
def test_default_cache_dir_follows_xdg_on_linux(monkeypatch, tmp_path):
    monkeypatch.delenv("HMF_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "b"))
    assert default_cache_dir() == tmp_path / "b" / "hmf"


def test_corrupt_cache_entry_is_a_miss(fresh, tmp_path):
    cache = DiskCache(directory=tmp_path)
    model = CAMB(k_max=5 * h_Mpc)
    key = _boltzmann.run_key("camb", model.run_input(COSMO, KAccuracy()))
    cache.path(key).parent.mkdir(parents=True)
    cache.path(key).write_bytes(b"not an npz file")
    Transfer(cosmology=COSMO, model=model, disk_cache=cache).transfer_function(1 * h_Mpc)
    assert fresh() == 1
    assert cache.load(key) is not None  # overwritten with a good entry


def _key(model=None, cosmo=COSMO, k_accuracy=None):
    model = model or CAMB()
    return _boltzmann.run_key("camb", model.run_input(cosmo, k_accuracy or KAccuracy()))


def test_cache_key_changes_with_every_input():
    keys = {
        _key(),
        _key(cosmo=COSMO.clone(Ob0=0.049)),
        _key(cosmo=COSMO.clone(m_nu=[0, 0.03, 0.03] * u.eV)),
        _key(model=CAMB(k_max=30 * h_Mpc)),
        _key(model=CAMB(z_max=10)),
        _key(model=CAMB(dark_energy_model="ppf")),
        _key(model=CAMB(settings={"Accuracy.AccuracyBoost": 2})),
        _key(k_accuracy=KAccuracy.high()),
    }
    assert len(keys) == 8
    # ...but not with fields the run doesn't depend on, or the cosmology's name.
    assert _key(model=CAMB(tail_decay_ln_k=3)) == _key()
    assert _key(cosmo=COSMO.clone(name="other")) == _key()


def test_cache_key_includes_the_versions(monkeypatch):
    base = _key()
    real = _boltzmann._version
    monkeypatch.setattr(_boltzmann, "_version", lambda d: "0.0" if d == "camb" else real(d))
    assert _key() != base
    monkeypatch.setattr(_boltzmann, "_version", lambda d: "0.0" if d == "hmf" else real(d))
    assert _key() != base


def test_canonical_serialisation_is_stable():
    obj = {"b": [1, 2.5, None], "a": {"y": True, "x": np.float64(0.1)}, "c": 3 * u.Mpc}
    assert canonical_json(obj) == canonical_json(dict(reversed(list(obj.items()))))
    assert canonical_json({"x": 0.1}) == '{"x":0.1}'
    assert content_hash(CAMB()) == content_hash(CAMB())
    assert content_hash(CAMB()) != content_hash(CAMB(k_max=1 * h_Mpc))
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json(float("nan"))
    with pytest.raises(TypeError):
        canonical_json(object())


def test_cosmology_key_ignores_name_and_meta():
    assert cosmology_key(Planck18) == cosmology_key(Planck18.clone(name="x", meta={"a": 1}))
    assert cosmology_key(Planck18) != cosmology_key(Planck18.clone(Om0=0.3))
    hash(cosmology_key(Planck18))
