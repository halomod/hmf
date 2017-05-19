import inspect
import os

LOCATION = "/".join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split("/")[:-1])
from nose.tools import raises
import sys
sys.path.insert(0, LOCATION)
from hmf import _framework as fm
from hmf import hmf

@raises(TypeError)
def test_incorrect_argument():
    t = hmf.MassFunction(wrong_arg=3)

@raises(ValueError)
def test_incorrect_update_arg():
    t = hmf.MassFunction()
    t.update(wrong_arg=3)


class TestIntrospection(object):
    def __init__(self):
        self.cls = hmf.MassFunction
        self.inst = self.cls(z=10)

    def test_parameter_names(self):
        assert "cosmo_model" in self.cls.get_all_parameter_names()

    def test_parameter_defaults(self):
        assert type(self.cls.get_all_parameter_defaults(recursive=False)) is dict

        assert self.cls.get_all_parameter_defaults()['z'] == 0

    def test_parameter_default_rec(self):
        pd = self.cls.get_all_parameter_defaults(recursive=True)
        assert type(pd['cosmo_params']) is dict

    def test_param_values(self):
        assert type(self.inst.parameter_values) is dict
        assert self.inst.parameter_values['z'] == 10

    def test_qnt_avail(self):
        assert 'dndm' in self.cls.quantities_available()

    def test_parameter_info(self):
        assert self.cls.parameter_info() is None
        assert self.cls.parameter_info(names=['z']) is None