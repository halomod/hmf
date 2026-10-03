"""Tests for the auto-generated documentation of fitting functions."""

import doctest
import re

import numpy as np
import pytest

from hmf.mass_function import fitting_functions as ff

MODELS = dict(ff.BaseFittingFunction.get_models())


@pytest.mark.parametrize("name", sorted(MODELS))
def test_docstring_lists_all_defaults(name):
    cls = MODELS[name]
    assert cls.__doc__ is not None
    for key, value in cls._defaults.items():
        # Each parameter appears as a table row with its default value.
        pattern = rf"``{re.escape(key)}``\s+{re.escape(repr(value))}\s*$"
        assert re.search(pattern, cls.__doc__, flags=re.MULTILINE), (name, key)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_overridden_properties_have_docstrings(name):
    cls = MODELS[name]
    assert cls.cutmask.__doc__
    assert cls.fsigma.__doc__


def test_defaults_table_not_duplicated():
    # AnguloBound/ST subclass a documented fit; each must show only its own table.
    for cls in (ff.AnguloBound, ff.ST, ff.Bocquet200mHydro):
        assert cls.__doc__.count("Default model parameters") == 1


def test_st_docstring():
    assert "Sheth-Mo-Tormen" in ff.ST.__doc__
    assert ":class:`SMT`" in ff.ST.__doc__
    assert len(ff.ST.__doc__) > 200


def test_math_in_all_docstrings():
    for name, cls in MODELS.items():
        if cls.__module__ == ff.__name__:
            assert ".. math::" in cls.__doc__, name


def test_nu_docstring():
    doc = ff.BaseFittingFunction.nu.__doc__
    assert "sigma/delta_c" not in doc
    assert "delta_c/\\sigma" in doc


def test_undocumented_subclass_gets_defaults_table():
    class _Undocumented(ff.BaseFittingFunction, abstract=True):
        _defaults = {"alpha": 1.5}  # noqa: RUF012

        @property
        def fsigma(self):
            return np.ones_like(self.nu2)

    assert "``alpha``" in _Undocumented.__doc__
    assert _Undocumented.fsigma.__doc__ == ff.BaseFittingFunction.fsigma.__doc__


def test_base_class_example_runs():
    """The usage example in the base-class docstring must run and give SMT."""
    plugins = dict(ff.BaseFittingFunction._plugins)
    try:
        globs = {"np": np}
        parser = doctest.DocTestParser()
        test = parser.get_doctest(
            ff.BaseFittingFunction.__doc__, globs, "BaseFittingFunction", None, 0
        )
        assert test.examples
        runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
        runner.run(test, clear_globs=False)
        assert runner.failures == 0
        mysmt = test.globs["MySMT"]
        nu2 = np.linspace(0.5, 10, 20)
        np.testing.assert_allclose(
            mysmt(nu2=nu2).fsigma,
            ff.SMT(nu2=nu2, A=0.3222).fsigma,
            rtol=1e-12,
        )
    finally:
        ff.BaseFittingFunction._plugins.clear()
        ff.BaseFittingFunction._plugins.update(plugins)
