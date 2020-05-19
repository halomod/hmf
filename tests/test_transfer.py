import numpy as np
from hmf.density_field.transfer import Transfer
from hmf.density_field.transfer_models import EH_BAO
import pytest

# def rms(a):
#     print(a)
#     print("RMS: ", np.sqrt(np.mean(np.square(a))))
#     return np.sqrt(np.mean(np.square(a)))

#
# def check_close(t, t2, fit):
#     t.update(transfer_model=fit)
#     assert np.mean(np.abs((t.power - t2.power) / t.power)) < 1


@pytest.fixture
def transfers():
    return Transfer(), Transfer()


@pytest.mark.parametrize(
    ["name", "val",],
    [("z", 0.1), ("sigma_8", 0.82), ("n", 0.95), ("cosmo_params", {"H0": 68.0})],
)
def test_updates(transfers, name, val):
    t, t2 = transfers
    t.update(**{name: val})
    assert (
        np.mean(np.abs((t.power - t2.power) / t.power)) < 1
        and np.mean(np.abs((t.power - t2.power) / t.power)) > 1e-6
    )


def test_halofit():
    t = Transfer(lnk_min=-20, lnk_max=20, dlnk=0.05, transfer_model="EH")
    print(EH_BAO._defaults)
    print("in test_transfer, params are: ", t.transfer_params)
    assert np.isclose(t.power[0], t.nonlinear_power[0])
    assert 5 * t.power[-1] < t.nonlinear_power[-1]


def test_ehnobao():
    t = Transfer(transfer_model="EH")
    tnobao = Transfer(transfer_model="EH_NoBAO")
    assert np.isclose(t._unnormalised_lnT[0], tnobao._unnormalised_lnT[0], rtol=1e-5)


def test_bondefs():
    t = Transfer(transfer_model="BondEfs")
    print(np.exp(t._unnormalised_lnT))
    assert np.isclose(np.exp(t._unnormalised_lnT[0]), 1, rtol=1e-5)


@pytest.mark.skip("Too slow and needs to be constantly updated.")
def test_data(datadir):
    import camb
    from astropy.cosmology import LambdaCDM

    cp = camb.CAMBparams()
    cp.set_matter_power(kmax=100.0)
    t = Transfer(
        cosmo_model=LambdaCDM(Om0=0.3, Ode0=0.7, H0=70.0, Ob0=0.05),
        sigma_8=0.8,
        n=1,
        transfer_params={"camb_params": cp},
        lnk_min=np.log(1e-11),
        lnk_max=np.log(1e11),
    )
    pdata = np.genfromtxt(datadir / "power_for_hmf_tests.dat")

    assert np.sqrt(np.mean(np.square(t.power - pdata[:, 1]))) < 0.001
