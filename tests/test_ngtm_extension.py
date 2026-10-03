"""Tests that ngtm/rho_gtm are unchanged by computing the mass-range extension directly.

``MassFunction._gtm`` extends the mass range up to 1e18 to integrate the mass function.
It used to do so by deep-copying the framework and updating ``Mmin``/``Mmax``. The
reference here does exactly that, so the direct computation is checked against it.
"""

import copy
import warnings

import numpy as np
import pytest

from hmf import MassFunction
from hmf.alternatives.wdm import MassFunctionWDM
from hmf.mass_function import fitting_functions as ff
from hmf.mass_function.integrate_hmf import hmf_integral_gtm

RTOL = 1e-14


def _reference_gtm(mf, dndm, mass_density=False):
    """The pre-optimisation ``MassFunction._gtm``, deep-copying the framework."""
    size = len(dndm)
    m = mf.m
    if (m[-1] < 10**16.5 and not np.isnan(dndm[-1]) and dndm[-1] != 0) and not isinstance(
        mf.hmf, ff.Behroozi
    ):
        new_mf = copy.deepcopy(mf)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_mf.update(Mmin=np.log10(mf.m[-1]) + mf.dlog10m, Mmax=18)
        dndm = np.concatenate((dndm, new_mf.dndm))
        m = np.concatenate((m, new_mf.m))

    ngtm = hmf_integral_gtm(m[dndm > 0], dndm[dndm > 0], mass_density)
    if len(ngtm) < len(m):
        ngtm_temp = np.zeros(len(dndm))
        ngtm_temp[dndm > 0] = ngtm
        ngtm = ngtm_temp
    return ngtm[:size]


def _reference_dndm(mf):
    """dndm, re-derived from the reference _gtm for Behroozi (which calls _gtm)."""
    if not isinstance(mf.hmf, ff.Behroozi):
        return mf.dndm
    dndm = mf.fsigma * mf.mean_density0 * np.abs(mf._dlnsdlnm) / mf.m**2
    ngtm_tinker = _reference_gtm(mf, dndm)
    return mf.hmf._modify_dndm(mf.m / mf.cosmo.h, dndm, mf.z, ngtm_tinker, h=mf.cosmo.h)


def _assert_gtm_unchanged(mf):
    dndm = _reference_dndm(mf)
    np.testing.assert_allclose(mf.dndm, dndm, rtol=RTOL, atol=0)
    np.testing.assert_allclose(mf.ngtm, _reference_gtm(mf, dndm), rtol=RTOL, atol=0)
    np.testing.assert_allclose(
        mf.rho_gtm, _reference_gtm(mf, dndm, mass_density=True), rtol=RTOL, atol=0
    )


@pytest.mark.parametrize(
    "hmf_model", ["Tinker08", "ST", "PS", "Watson", "Reed07", "Bhattacharya", "Behroozi"]
)
@pytest.mark.parametrize("mrange", [(10, 15), (8, 13.5), (12, 17)], ids=str)
def test_gtm_unchanged_across_fits_z_and_mass_ranges(hmf_model, mrange):
    mf = MassFunction(
        transfer_model="EH", hmf_model=hmf_model, Mmin=mrange[0], Mmax=mrange[1], dlog10m=0.05
    )
    for z in [0.0, 1.0, 3.0]:
        mf.update(z=z)
        _assert_gtm_unchanged(mf)


@pytest.mark.parametrize("filter_model", ["SharpK", "SmoothK", "SharpKEllipsoid"])
def test_gtm_unchanged_other_filters(filter_model):
    mf = MassFunction(transfer_model="EH", filter_model=filter_model, dlog10m=0.05)
    for z in [0.0, 2.0]:
        mf.update(z=z)
        _assert_gtm_unchanged(mf)


def test_gtm_unchanged_after_parameter_updates():
    """The cached z-independent extension is invalidated by the parameters it uses."""
    mf = MassFunction(transfer_model="EH", dlog10m=0.05)
    mf.ngtm
    for kwargs in [{"n": 0.9}, {"sigma_8": 0.7}, {"Mmax": 14}, {"dlog10m": 0.1}, {"z": 1.0}]:
        mf.update(**kwargs)
        _assert_gtm_unchanged(mf)


def test_gtm_unchanged_with_mass_conversion():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mf = MassFunction(
            transfer_model="EH",
            hmf_model="ST",
            mdef_model="SOMean",
            disable_mass_conversion=False,
            dlog10m=0.1,
        )
        _assert_gtm_unchanged(mf)


def test_gtm_unchanged_for_subclass_overriding_dndm():
    """WDM overrides dndm, so its extension must still use the overridden dndm."""
    mf = MassFunctionWDM(transfer_model="EH", dlog10m=0.05)
    _assert_gtm_unchanged(mf)


def test_gtm_does_not_copy_framework(monkeypatch):
    mf = MassFunction(transfer_model="EH", dlog10m=0.05)
    mf.dndm

    def no_copy(self, memo):
        raise AssertionError("_gtm should not deep-copy the framework")

    monkeypatch.setattr(MassFunction, "__deepcopy__", no_copy, raising=False)
    assert mf.ngtm[0] > 0


def test_extension_sigma_is_z_independent():
    """Changing z re-uses the cached sigma at the extension masses."""
    mf = MassFunction(transfer_model="EH", dlog10m=0.05)
    mf.ngtm
    ext = mf._gtm_extension_unn_sigma0_and_dlnss_dlnm
    mf.update(z=1.0)
    mf.ngtm
    assert mf._gtm_extension_unn_sigma0_and_dlnss_dlnm is ext


def test_ngtm_physical_limits():
    """Ngtm is positive and decreasing, and rho_gtm(Mmin) is at most the mean density."""
    mf = MassFunction(transfer_model="EH", hmf_model="PS", Mmin=3, Mmax=15, dlog10m=0.05)
    assert np.all(mf.ngtm > 0)
    assert np.all(np.diff(mf.ngtm) < 0)
    # Press-Schechter puts all mass in haloes, so most of it is above 1e3 Msun/h.
    assert 0.5 < mf.rho_gtm[0] / mf.mean_density0 < 1.0
