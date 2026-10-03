import numpy as np
import pytest

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
            transfer_model="EH",
        )
    ):
        print(i)
        assert len(label) == len(order[i])
        assert sorted(label.split(", ")) == sorted(order[i].split(", "))
        assert isinstance(mf, MassFunction)
        assert np.allclose(quants[0], mf.dndm)
        assert np.allclose(quants[1], mf.ngtm)


def test_collected_iterations_are_independent():
    """Collecting the iterator must give one independent result per parameter value."""
    zs = [0.0, 1.0, 2.0]
    out = list(get_hmf(["dndm", "m", "mean_density"], z=zs, transfer_model="EH"))

    assert len(out) == 3
    assert [mf.z for _, mf, _ in out] == zs
    assert len({id(mf) for _, mf, _ in out}) == 3

    # The mean matter density scales as (1+z)^3.
    rho0 = out[0][0][2]
    for (quants, _, _), z in zip(out, zs, strict=True):
        np.testing.assert_allclose(quants[2] / rho0, (1 + z) ** 3, rtol=1e-10)

    # Each stored dndm is the one for its own redshift.
    for (quants, mf, _), z in zip(out, zs, strict=True):
        ref = MassFunction(z=z, transfer_model="EH")
        np.testing.assert_allclose(quants[0], ref.dndm, rtol=1e-8, atol=0)
        np.testing.assert_allclose(mf.dndm, ref.dndm, rtol=1e-8, atol=0)

    # The abundance of massive (cluster-scale) haloes drops steeply with redshift.
    massive = out[0][0][1] > 1e14
    assert massive.any()
    assert np.all(out[0][0][0][massive] > out[1][0][0][massive])
    assert np.all(out[1][0][0][massive] > out[2][0][0][massive])

    # No quantity array is shared between iterations, even ones that don't depend
    # on the looped parameter (like the mass vector).
    for i in range(3):
        for j in range(i + 1, 3):
            for qi, qj in zip(out[i][0], out[j][0], strict=True):
                assert not np.shares_memory(qi, qj)


def test_collected_iterations_are_independent_multiple_lists():
    out = list(
        get_hmf(
            ["dndm", "mean_density"],
            get_label=False,
            z=[0.0, 1.0],
            sigma_8=[0.7, 0.8],
            transfer_model="EH",
        )
    )
    combos = {(mf.z, mf.sigma_8) for _, mf in out}
    assert combos == {(0.0, 0.7), (0.0, 0.8), (1.0, 0.7), (1.0, 0.8)}

    for quants, mf in out:
        ref = MassFunction(z=mf.z, sigma_8=mf.sigma_8, transfer_model="EH")
        np.testing.assert_allclose(quants[0], ref.dndm, rtol=1e-8, atol=0)
        # The mean density depends on z but not on the normalisation.
        np.testing.assert_allclose(quants[1] / out[0][0][1], (1 + mf.z) ** 3, rtol=1e-10)


def test_non_parameter_list_kwarg_raises():
    """A list-valued kwarg that is not a framework parameter gives a clear error."""
    with pytest.raises(ValueError, match="Invalid arguments"):
        list(get_hmf("dndm", z=[0, 1], not_a_param=[1, 2], transfer_model="EH"))


def test_req_quantities_alias():
    ((quants, mf, _label),) = list(get_hmf(req_quantities="dndm", transfer_model="EH"))
    np.testing.assert_allclose(quants[0], mf.dndm, rtol=1e-12, atol=0)

    ((quants_old, _, _),) = list(get_hmf(req_qauntities="dndm", transfer_model="EH"))
    np.testing.assert_allclose(quants_old[0], quants[0], rtol=1e-12, atol=0)

    with pytest.raises(TypeError, match="only one"):
        list(get_hmf("dndm", req_quantities="dndm", transfer_model="EH"))

    with pytest.raises(TypeError, match="missing required argument"):
        list(get_hmf(transfer_model="EH"))
