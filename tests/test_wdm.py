from pytest import raises
from hmf.alternatives import wdm
from hmf.mass_function import hmf
import numpy as np

def test_null():
    with raises(NotImplementedError):
        w = wdm.WDM(mx=1.0)
        w.transfer(1.)


class TestViel(object):
    def setup_method(self, test_method):
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
    def setup_method(self, test_method):
        self.cls = wdm.Bode01(mx=1.0)


class TestSchneider12_vCDM(object):
    def setup_method(self, test_method):
        self.cdm = hmf.MassFunction()
        self.cls = wdm.Schneider12_vCDM(m=self.cdm.m, dndm0=self.cdm.dndm)

    def test_high_m(self):
        assert np.isclose(self.cls.dndm_alter()[-1], self.cdm.dndm[-1], rtol=1e-3)


class TestSchneider12(TestSchneider12_vCDM):
    def setup_method(self, test_method):
        self.cdm = hmf.MassFunction()
        self.cls = wdm.Schneider12(m=self.cdm.m, dndm0=self.cdm.dndm)


class TestLovell14(TestSchneider12_vCDM):
    def setup_method(self, test_method):
        self.cdm = hmf.MassFunction()
        self.cls = wdm.Lovell14(m=self.cdm.m, dndm0=self.cdm.dndm)


class TestTransfer(object):
    def setup_method(self, test_method):
        self.wdm = wdm.TransferWDM(wdm_mass=3.0, wdm_model=wdm.Viel05)
        self.cdm = hmf.MassFunction()

    def test_wdm_model(self):
        assert isinstance(self.wdm.wdm, wdm.Viel05)


    def test_wrong_model_type(self):
        with raises(ValueError):
            cls = wdm.TransferWDM(wdm_mass=3.0, wdm_model=3)

    def test_power(self):
        print(self.wdm.power[0], self.cdm.power[0], self.wdm.power[0] / self.cdm.power[0] - 1)
        assert np.isclose(self.wdm.power[0], self.cdm.power[0], rtol=1e-4)
        assert self.wdm.power[-1] < self.cdm.power[-1]


class TestMassFunction(object):

    def __init__(self):
        self.wdm = wdm.MassFunctionWDM(alter_model=None, wdm_mass=3.0, wdm_model=wdm.Viel05)
        self.cdm = hmf.MassFunction()

    def test_dndm(self):
        assert np.isclose(self.cdm.dndm[-1], self.wdm.dndm[-1], rtol=1e-3)
        assert self.cdm.dndm[0] > self.wdm.dndm[0]


class TestMassFunctionAlter(TestMassFunction):
    def __init__(self):
        self.wdm = wdm.MassFunctionWDM(alter_model=wdm.Schneider12_vCDM, wdm_mass=3.0, wdm_model=wdm.Viel05)
        self.cdm = hmf.MassFunction()
