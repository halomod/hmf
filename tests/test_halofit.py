import numpy as np

from hmf.density_field import transfer


def test_takahashi():
    t = transfer.Transfer(transfer_model="EH", takahashi=False, lnk_max=7)
    tt = transfer.Transfer(transfer_model="EH", takahashi=True, lnk_max=7)

    assert np.isclose(t.nonlinear_power[0], tt.nonlinear_power[0], rtol=1e-4)
    print(t.nonlinear_power[-1] / tt.nonlinear_power[-1])
    assert np.logical_not(
        np.isclose(t.nonlinear_power[-1] / tt.nonlinear_power[-1], 1, rtol=0.4)
    )


def test_takahashi_hiz():
    # This test should do the HALOFIT WARNING
    t = transfer.Transfer(transfer_model="EH", takahashi=False, lnk_max=7, z=8.0)
    tt = transfer.Transfer(transfer_model="EH", takahashi=True, lnk_max=7, z=8.0)

    assert np.isclose(t.nonlinear_power[0], tt.nonlinear_power[0], rtol=1e-4)
    print(t.nonlinear_power[-1] / tt.nonlinear_power[-1])
    assert np.logical_not(
        np.isclose(t.nonlinear_power[-1] / tt.nonlinear_power[-1], 1, rtol=0.4)
    )

    t.update(z=0)

    assert np.logical_not(
        np.isclose(t.nonlinear_power[0] / tt.nonlinear_power[0], 0.9, rtol=0.1)
    )
    assert np.logical_not(
        np.isclose(t.nonlinear_power[-1] / tt.nonlinear_power[-1], 0.99, rtol=0.1)
    )


def test_halofit_high_s8():
    t = transfer.Transfer(transfer_model="EH", lnk_max=7, sigma_8=0.999)
    thi = transfer.Transfer(
        transfer_model="EH", lnk_max=7, sigma_8=1.001
    )  # just above threshold

    print(
        t.nonlinear_power[0] / thi.nonlinear_power[0] - 1,
        t.nonlinear_power[-1] / thi.nonlinear_power[-1] - 1,
    )
    assert np.isclose(t.nonlinear_power[0], thi.nonlinear_power[0], rtol=2e-2)
    assert np.isclose(t.nonlinear_power[-1], thi.nonlinear_power[-1], rtol=5e-2)
