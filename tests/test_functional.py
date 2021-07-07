import numpy as np

from hmf import MassFunction, get_hmf


def test_order():
    order = [
        "sigma_8: 0.7, ST, z: 0",
        "sigma_8: 0.7, PS, z: 0",
        "sigma_8: 0.7, ST, z: 1",
        "sigma_8: 0.7, PS, z: 1",
        "sigma_8: 0.7, ST, z: 2",
        "sigma_8: 0.7, PS, z: 2",
        "sigma_8: 0.8, ST, z: 0",
        "sigma_8: 0.8, PS, z: 0",
        "sigma_8: 0.8, ST, z: 1",
        "sigma_8: 0.8, PS, z: 1",
        "sigma_8: 0.8, ST, z: 2",
        "sigma_8: 0.8, PS, z: 2",
    ]

    for i, (quants, mf, label) in enumerate(
        get_hmf(
            ["dndm", "ngtm"],
            z=list(range(3)),
            hmf_model=["ST", "PS"],
            sigma_8=[0.7, 0.8],
        )
    ):
        print(i)
        assert len(label) == len(order[i])
        assert sorted(label.split(", ")) == sorted(order[i].split(", "))
        assert isinstance(mf, MassFunction)
        assert np.allclose(quants[0], mf.dndm)
        assert np.allclose(quants[1], mf.ngtm)
