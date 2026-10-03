import numpy as np
import pytest

import hmf
from hmf.alternatives import wdm


def test_null():
    w = wdm.WDM(mx=1.0)
    with pytest.raises(NotImplementedError):
        w.transfer(1.0)


class TestViel:
    def setup_class(self):
        self.cls = wdm.Viel05(mx=1.0)

    def test_lowk_transfer(self):
        assert np.isclose(self.cls.transfer(1e-5), 1, rtol=1e-4)

    def test_lam_eff(self):
        assert self.cls.lam_eff_fs > 0

    def test_m_eff(self):
        assert self.cls.m_fs > 0

    def test_lam_hm(self):
        assert self.cls.lam_hm > self.cls.lam_eff_fs

    def test_m_hm(self):
        assert self.cls.m_hm > self.cls.m_fs


class TestBode(TestViel):
    def setup_class(self):
        self.cls = wdm.Bode01(mx=1.0)


class TestSchneider12_vCDM:
    def setup_class(self):
        self.cdm = hmf.MassFunction(transfer_model="EH")
        self.cls = wdm.Schneider12_vCDM(
            m=self.cdm.m,
            dndm0=self.cdm.dndm,
        )

    def test_high_m(self):
        assert np.isclose(self.cls.dndm_alter()[-1], self.cdm.dndm[-1], rtol=1e-3)


class TestSchneider12(TestSchneider12_vCDM):
    def setup_class(self):
        self.cdm = hmf.MassFunction(transfer_model="EH")
        self.cls = wdm.Schneider12(
            m=self.cdm.m,
            dndm0=self.cdm.dndm,
        )


class TestLovell14(TestSchneider12_vCDM):
    def setup_class(self):
        self.cdm = hmf.MassFunction(transfer_model="EH")
        self.cls = wdm.Lovell14(
            m=self.cdm.m,
            dndm0=self.cdm.dndm,
        )


class TestTransfer:
    def setup_class(self):
        self.wdm = wdm.TransferWDM(wdm_mass=3.0, wdm_model=wdm.Viel05, transfer_model="EH")
        self.cdm = hmf.MassFunction(transfer_model="EH")

    def test_wdm_model(self):
        assert isinstance(self.wdm.wdm, wdm.Viel05)

    def test_wrong_model_type(self):
        with pytest.raises(ValueError, match="must be str or Component subclass"):
            wdm.TransferWDM(wdm_mass=3.0, wdm_model=3, transfer_model="EH")

    def test_power(self):
        print(
            self.wdm.power[0],
            self.cdm.power[0],
            self.wdm.power[0] / self.cdm.power[0] - 1,
        )
        assert np.isclose(self.wdm.power[0], self.cdm.power[0], rtol=1e-4)
        assert self.wdm.power[-1] < self.cdm.power[-1]


class TestMassFunction:
    def setup_class(self):
        self.wdm = wdm.MassFunctionWDM(
            alter_model=None, wdm_mass=3.0, wdm_model=wdm.Viel05, transfer_model="EH"
        )
        self.cdm = hmf.MassFunction(transfer_model="EH")

    def test_dndm(self):
        assert np.isclose(self.cdm.dndm[-1], self.wdm.dndm[-1], rtol=1e-3)
        assert self.cdm.dndm[0] > self.wdm.dndm[0]


class TestMassFunctionAlter(TestMassFunction):
    def setup_class(self):
        self.wdm = wdm.MassFunctionWDM(
            alter_model=wdm.Schneider12_vCDM,
            wdm_mass=3.0,
            wdm_model=wdm.Viel05,
            transfer_model="EH",
        )
        self.cdm = hmf.MassFunction(transfer_model="EH")


class TestHalfModeMassComoving:
    """M_fs and M_hm are comoving, so they must not depend on redshift.

    Reference: Schneider, Smith, Maccio & Moore (2012), MNRAS 424, 684, Eqs. 6-9.
    The WDM transfer function is T(k) = [1 + (alpha k)^(2 nu)]^(-5/nu) (Eq. 6) with
    nu = 1.12 and (Viel et al. 2005, Eq. 7)
    alpha = 0.049 (m_x / keV)^-1.11 (Omega_wdm / 0.25)^0.11 (h / 0.7)^1.22 Mpc/h.
    The half-mode scale, where T = 1/2, is
    lambda_hm = 2 pi alpha (2^(nu/5) - 1)^(-1/(2 nu)) (Eq. 8), and
    M_hm = (4 pi / 3) rho_bar (lambda_hm / 2)^3 (Eq. 9), with rho_bar the (comoving)
    mean matter density.
    """

    # rho_crit,0 / h^2 = 3 (100 km/s/Mpc)^2 / (8 pi G) in Msun / Mpc^3.
    RHO_CRIT_H2 = 2.77536627e11

    @pytest.mark.parametrize("alter", ["Schneider12", "Lovell14", "Schneider12_vCDM"])
    def test_masses_independent_of_z(self, alter):
        mf = wdm.MassFunctionWDM(transfer_model="EH", alter_model=alter, wdm_mass=1.0)
        m_hm, m_fs = [], []
        for z in (0, 1, 3):
            mf.update(z=z)
            m_hm.append(mf.wdm.m_hm)
            m_fs.append(mf.wdm.m_fs)
        assert np.allclose(m_hm, m_hm[0], rtol=1e-12, atol=0)
        assert np.allclose(m_fs, m_fs[0], rtol=1e-12, atol=0)

    @pytest.mark.parametrize("mx", [0.5, 1.0, 3.0])
    @pytest.mark.parametrize("z", [0, 3])
    def test_m_hm_matches_schneider12(self, mx, z):
        mf = wdm.MassFunctionWDM(transfer_model="EH", wdm_mass=mx, z=z)
        cosmo = mf.cosmo
        nu = 1.12

        alpha = (
            0.049 * mx**-1.11 * ((cosmo.Om0 - cosmo.Ob0) / 0.25) ** 0.11 * (cosmo.h / 0.7) ** 1.22
        )
        lam_hm = 2 * np.pi * alpha * (2 ** (nu / 5) - 1) ** (-1 / (2 * nu))

        # lambda_hm is the scale at which the WDM transfer function is halved.
        assert np.isclose(mf.wdm.transfer(2 * np.pi / lam_hm), 0.5, rtol=1e-10)

        rho_bar0 = cosmo.Om0 * self.RHO_CRIT_H2  # h^2 Msun / Mpc^3, comoving
        expected = (4 * np.pi / 3) * rho_bar0 * (lam_hm / 2) ** 3
        assert np.isclose(mf.wdm.m_hm, expected, rtol=1e-4)

    def test_z_update_does_not_recompute_wdm_transfer(self, monkeypatch):
        calls = []
        orig = wdm.Viel05.transfer

        def counting_transfer(self, k):
            calls.append(1)
            return orig(self, k)

        monkeypatch.setattr(wdm.Viel05, "transfer", counting_transfer)

        mf = wdm.MassFunctionWDM(transfer_model="EH", wdm_mass=1.0)
        assert "z" not in mf.get_dependencies("wdm", "_unnormalised_lnT")

        mf.dndm
        n = len(calls)
        assert n > 0
        for z in (1, 3):
            mf.update(z=z)
            mf.dndm
        assert len(calls) == n

    def test_z_argument_deprecated(self):
        with pytest.warns(DeprecationWarning, match="z"):
            w = wdm.Viel05(mx=1.0, z=3)
        assert w.m_hm == wdm.Viel05(mx=1.0).m_hm
