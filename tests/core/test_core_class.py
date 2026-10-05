"""Tests of the CLASS-backed transfer and growth models (need the optional classy)."""

import ast
import subprocess
import sys
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.cosmology import FlatLambdaCDM

import hmf.core
from hmf.core import _boltzmann
from hmf.core.growth import Growth
from hmf.core.transfer import Transfer
from hmf.core.units import h_Mpc

COSMO = FlatLambdaCDM(
    H0=67.3, Om0=0.315, Ob0=0.0493, Tcmb0=2.7255, m_nu=[0, 0, 0.1] * u.eV, name="class-test"
)


def test_hmf_core_imports_without_classy():
    code = (
        "import sys, warnings\n"
        "sys.modules['classy'] = None\n"
        "warnings.simplefilter('ignore')\n"
        "import hmf.core.transfer, hmf.core.growth\n"
        "from hmf.core.units import h_Mpc\n"
        "t = hmf.core.transfer.Transfer(model='EH')\n"
        "print(t.transfer_function(1 * h_Mpc))\n"
        "try:\n"
        "    hmf.core.transfer.Transfer(model='CLASS').solution\n"
        "except ImportError as e:\n"
        "    print('ImportError', 'hmf[class]' in str(e))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.splitlines()[-1] == "ImportError True"


def test_hmf_core_never_imports_a_boltzmann_code_at_module_level():
    """CAMB and classy are imported lazily (v3's ``import hmf`` still needs CAMB)."""
    root = Path(hmf.core.__file__).parent
    for path in root.rglob("*.py"):
        for node in ast.parse(path.read_text()).body:
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            assert not {n.split(".")[0] for n in names} & {"camb", "classy"}, path


classy = pytest.importorskip("classy")


@pytest.fixture
def fresh():
    _boltzmann.clear_memo()
    before = dict(_boltzmann.run_counts())
    yield lambda: _boltzmann.run_counts().get("class", 0) - before.get("class", 0)
    _boltzmann.clear_memo()


def test_class_transfer_and_growth_share_one_run(fresh):
    t = Transfer(cosmology=COSMO, model="CLASS")
    g = Growth.from_transfer(t, model="CLASS")
    for species in ("cb", "tot"):
        t.transfer_function(np.logspace(-3, 1, 5) * h_Mpc, species)
        g.growth_factor(np.array([0.0, 2.0, 10.0]), species)
        g.growth_rate(1.0, species)
    assert fresh() == 1


def test_class_transfer_tends_to_unity_and_joins_smoothly():
    sol = Transfer(cosmology=COSMO, model="CLASS").solution
    np.testing.assert_allclose(sol.transfer(np.array([1e-8, 1e-7])), 1, atol=1e-6)
    x, eps = np.log(sol.k_max_table), 1e-4
    for species in ("cb", "tot"):
        f = sol.ln_transfer(np.exp(np.array([x - eps, x, x + eps])), species)
        slope = (f[1] - f[0]) / eps
        jump = (f[2] - f[1]) / eps - slope
        assert abs(jump) < 0.1 * eps * abs(slope)


@pytest.mark.parametrize("species", ["cb", "tot"])
def test_class_agrees_with_camb(species):
    """Two independent Boltzmann codes agree on T(k) for the same cosmology."""
    k = np.logspace(-4, 1.2, 80) * h_Mpc
    t_class = Transfer(cosmology=COSMO, model="CLASS").transfer_function(k, species)
    t_camb = Transfer(cosmology=COSMO, model="CAMB").transfer_function(k, species)
    # Tolerance: default precision of both codes; measured max 1.8e-3.
    np.testing.assert_allclose(t_class, t_camb, rtol=5e-3)


@pytest.mark.parametrize("species", ["cb", "tot"])
def test_class_growth_agrees_with_camb_growth(species):
    z = np.array([0.5, 1.0, 3.0, 10.0])
    d_class = Growth(cosmology=COSMO, model="CLASS").growth_factor(z, species)
    d_camb = Growth(cosmology=COSMO, model="CAMB").growth_factor(z, species)
    # Tolerance: both evaluate at k = 0.01/Mpc; CLASS's P(k, z) is tabulated at few
    # redshifts. Measured max 2.2e-4.
    np.testing.assert_allclose(d_class, d_camb, rtol=1e-3)
