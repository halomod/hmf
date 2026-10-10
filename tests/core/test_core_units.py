"""Tests of hmf.core.units: the units boundary between public methods and kernels."""

import functools
import re

import astropy.cosmology.units as cu
import astropy.units as u
import attrs
import numpy as np
import pytest

from hmf.core import _kernels, units
from hmf.core.stage import Stage
from hmf.core.units import (
    CANONICAL_UNITS,
    H0_unit,
    Mpc_h,
    Msun_h,
    UnitBoundaryError,
    UnitContext,
    dndm_unit,
    h_Mpc,
    littleh,
    littleh_power,
    littleh_units_enabled,
    parse_unit,
    unit_boundary,
)

H0 = 70.0 * H0_unit
LITTLE_H = 0.7


@attrs.frozen(kw_only=True)
class Toy(Stage):
    """A stage with a cosmology (just H0), and boundary methods."""

    H0: u.Quantity = H0

    @functools.cached_property
    def _unit_context(self) -> UnitContext:
        return UnitContext(self.H0)

    @unit_boundary(m=Msun_h, returns=Msun_h)
    def mass(self, m, z=0.0):
        """Return m unchanged (the identity kernel)."""
        return m

    @unit_boundary(r=Mpc_h, k=h_Mpc, returns=(Mpc_h, h_Mpc))
    def both(self, r, *, k):
        return r, k

    @unit_boundary(m=Msun_h, r=Mpc_h, returns=Msun_h)
    def mass_and_radius(self, m, r, z=0.0):
        """Return m, checking that both arrive without units."""
        assert not isinstance(m, u.Quantity)
        assert not isinstance(r, u.Quantity)
        return m

    @unit_boundary(m=Msun_h, r=Mpc_h)
    def ratio(self, m, r):
        return m / r

    @unit_boundary(r=Mpc_h, m=Msun_h)
    def ratio_named_backwards(self, m, r):
        """The decorator names the arguments in the other order."""
        return m / r

    @unit_boundary(m=Msun_h, r=Mpc_h, returns=(Msun_h, Mpc_h))
    def after_z(self, z, m, r):
        """Two dimensional arguments that are not the first two."""
        return m, r

    @unit_boundary(m=Msun_h, r=Mpc_h, k=h_Mpc, returns=(Msun_h, Mpc_h, h_Mpc))
    def three(self, m, r, k):
        return m, r, k

    @unit_boundary(m=Msun_h, r=Mpc_h, k=h_Mpc, returns=Msun_h)
    def three_mass(self, m, r, k):
        return m

    @unit_boundary(m=Msun_h, r=Mpc_h, k=h_Mpc)
    def three_product(self, m, r, k):
        return m * r * k

    @unit_boundary(m=Msun_h, returns=dndm_unit)
    def sqrt_like(self, m):
        # np.sqrt of a 0-d array is a numpy scalar, not an array.
        return np.sqrt(m) * 0 + 1.0

    @unit_boundary(m=Msun_h, returns=None)
    def raw(self, m, z=0.0):
        return m, z

    @unit_boundary(m=Msun_h)
    def optional(self, m=None):
        return m

    @unit_boundary(m=Msun_h, returns=Msun_h)
    def bad_kernel(self, m):
        return m * Msun_h


class NoContext:
    """An object without a ``_unit_context``: it can only take h-units."""

    @unit_boundary(m=Msun_h, returns=Msun_h)
    def mass(self, m):
        return m


@pytest.fixture
def toy():
    return Toy()


# ---------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------
def test_constants():
    assert Msun_h == u.Msun / cu.littleh
    assert dndm_unit == littleh**4 / u.Msun / u.Mpc**3
    assert units.number_density_unit == littleh**3 / u.Mpc**3
    assert units.power_unit == u.Mpc**3 / littleh**3
    assert units.rho_unit == u.Msun * littleh**2 / u.Mpc**3
    # dn/dm * m is a number density; P(k) * k^3 is dimensionless.
    assert dndm_unit * Msun_h == units.number_density_unit
    assert (units.power_unit * h_Mpc**3).decompose() == u.dimensionless_unscaled


def test_littleh_power():
    assert littleh_power(Msun_h) == -1
    assert littleh_power(dndm_unit) == 4
    assert littleh_power(u.Msun) == 0
    assert littleh_power(u.Msun / littleh**2) == -2


def test_kernel_docstring_table_matches_canonical_units():
    rows = re.findall(r"^(\w+)\s+.+?\s+``(\w+)``\s*$", _kernels.__doc__, re.MULTILINE)
    table = {kind: name for kind, name in rows if kind != "Kind"}
    assert set(table) == set(CANONICAL_UNITS)
    for kind, name in table.items():
        assert getattr(units, name) is CANONICAL_UNITS[kind], kind


def test_parse_unit_littleh():
    assert parse_unit("Msun / littleh") is Msun_h
    assert parse_unit("littleh4 / (Msun Mpc3)") is dndm_unit
    assert parse_unit("Msun / littleh2") == u.Msun / littleh**2
    # Outside the helper, littleh does not parse.
    with pytest.raises(ValueError):
        u.Unit("Msun / littleh")


def test_littleh_units_enabled_does_not_enable_with_h0():
    with littleh_units_enabled():
        assert u.Unit("Mpc/littleh") == Mpc_h
        with pytest.raises(u.UnitConversionError):
            (1 * u.Msun).to(Msun_h)


# ---------------------------------------------------------------------------------
# The boundary
# ---------------------------------------------------------------------------------
def test_identity_fast_path(toy, monkeypatch):
    m = np.logspace(10, 15, 5) * Msun_h
    # The fast path never consults the context.
    monkeypatch.setattr(UnitContext, "factor", lambda *a: pytest.fail("not the fast path"))
    out = toy.mass(m)
    assert out.unit is Msun_h
    np.testing.assert_array_equal(out.value, m.value)
    # No copy, in or out.
    assert np.shares_memory(out, m)


def test_equal_but_not_identical_unit(toy):
    fresh = u.Msun / cu.littleh
    assert fresh == Msun_h
    assert fresh is not Msun_h
    out = toy.mass(1e12 * fresh)
    assert out.unit is Msun_h
    assert out.value == 1e12


def test_physical_mass_converted_with_h0(toy):
    # m [Msun/h] = m [Msun] * h, with h = H0 / (100 km/s/Mpc) = 0.7.
    out = toy.mass(1e12 * u.Msun)
    assert out.unit is Msun_h
    np.testing.assert_allclose(out.value, 1e12 * LITTLE_H, rtol=1e-14)

    # Another H0 gives another factor: the instance's own H0 is used.
    other = toy.evolve(H0=50 * H0_unit)
    np.testing.assert_allclose(other.mass(1e12 * u.Msun).value, 0.5e12, rtol=1e-14)


def test_physical_length_and_wavenumber(toy):
    r, k = toy.both(1.0 * u.Mpc, k=1.0 / u.Mpc)
    # r [Mpc/h] = r [Mpc] * h; k [h/Mpc] = k [1/Mpc] / h.
    np.testing.assert_allclose(r.value, LITTLE_H, rtol=1e-14)
    np.testing.assert_allclose(k.value, 1 / LITTLE_H, rtol=1e-14)
    assert r.unit is Mpc_h
    assert k.unit is h_Mpc


def test_scaled_unit(toy):
    out = toy.mass(2.0 * u.Unit(1e10 * Msun_h))
    np.testing.assert_allclose(out.value, 2e10, rtol=1e-14)


def test_wrong_littleh_power_raises(toy):
    # with_H0 alone would silently "convert" this.
    with pytest.raises(u.UnitConversionError, match=r"littleh\*\*-2"):
        toy.mass(1e12 * u.Msun / littleh**2)
    with pytest.raises(u.UnitConversionError, match="argument 'm'"):
        toy.mass(1e12 * u.Msun * littleh)


@pytest.mark.parametrize("bare", [1e12, np.array([1e12, 1e13]), 3])
def test_bare_value_raises(toy, bare):
    with pytest.raises(UnitBoundaryError) as e:
        toy.mass(bare)
    msg = str(e.value)
    assert "Toy.mass()" in msg
    assert "'m'" in msg
    assert "solMass / littleh" in msg
    assert "m * hmf.core.units.Msun_h" in msg
    assert "m * u.Msun" in msg
    assert isinstance(e.value, TypeError)


def test_bare_value_message_names_the_unit_kind(toy):
    with pytest.raises(UnitBoundaryError, match=r"k \* hmf.core.units.h_Mpc"):
        toy.both(1 * Mpc_h, k=0.1)


def test_bare_value_message_for_non_canonical_unit():
    class Timed:
        @unit_boundary(t=u.s)
        def f(self, t):
            return t

    with pytest.raises(UnitBoundaryError, match=r"`t \* u.Unit\('s'\)`"):
        Timed().f(1.0)


def test_wrong_dimension_raises(toy):
    with pytest.raises(u.UnitConversionError, match="argument 'm'"):
        toy.mass(1.0 * Mpc_h)
    with pytest.raises(u.UnitConversionError, match="argument 'r'"):
        toy.both(1.0 * u.s, k=1 * h_Mpc)


def test_dimensionless_pass_through(toy):
    z = np.array([0.0, 1.0])
    _, z_out = toy.raw(1e12 * Msun_h, z=z)
    assert z_out is z
    _, z_out = toy.raw(1e12 * Msun_h, 0.5)
    assert z_out == 0.5


def test_kernel_receives_plain_arrays(toy):
    m, _ = toy.raw(np.array([1.0, 2.0]) * u.Msun)
    assert type(m) is np.ndarray
    m, _ = toy.raw(1.0 * Msun_h)
    assert type(m) is np.ndarray
    assert m.shape == ()


def test_scalar_in_scalar_out(toy):
    out = toy.mass(1e12 * Msun_h)
    assert isinstance(out, u.Quantity)
    assert out.isscalar
    assert out.shape == ()
    # A kernel that returns a numpy scalar.
    out = toy.sqrt_like(4.0 * Msun_h)
    assert out.isscalar
    assert out.unit is dndm_unit


def test_array_in_array_out(toy):
    m = np.ones((2, 3)) * u.Msun
    out = toy.mass(m)
    assert out.shape == (2, 3)
    assert out.unit is Msun_h


def test_keyword_and_tuple_returns(toy):
    r, k = toy.both(r=2 * Mpc_h, k=3 * h_Mpc)
    assert (r.value, k.value) == (2.0, 3.0)


def test_none_passes_through(toy):
    assert toy.optional() is None
    assert toy.optional(m=None) is None


def test_kernel_returning_quantity_is_an_error(toy):
    with pytest.raises(TypeError, match="returned a Quantity"):
        toy.bad_kernel(1 * Msun_h)


def test_factor_cache_per_instance(toy):
    ctx = toy._unit_context
    assert ctx is toy._unit_context  # cached on the instance
    assert ctx.n_computed == 0

    toy.mass(1.0 * u.Msun)
    assert ctx.n_computed == 1
    # Same unit object: identity hit.
    toy.mass(2.0 * u.Msun)
    assert ctx.n_computed == 1
    # Equal unit, different object: equality hit.
    toy.mass(2.0 * u.Unit("solMass"))
    toy.mass(2.0 * (u.Msun * u.dimensionless_unscaled))
    assert ctx.n_computed == 1
    # Another unit: another computation.
    toy.mass(2.0 * u.kg)
    assert ctx.n_computed == 2

    # A new instance has its own cache.
    other = toy.evolve(H0=60 * H0_unit)
    assert other._unit_context is not ctx
    other.mass(1.0 * u.Msun)
    assert other._unit_context.n_computed == 1
    assert ctx.n_computed == 2


def test_no_context_accepts_h_units_only():
    obj = NoContext()
    assert obj.mass(1 * Msun_h).value == 1
    np.testing.assert_allclose(obj.mass(1 * u.Unit(1e10 * Msun_h)).value, 1e10, rtol=1e-14)
    with pytest.raises(u.UnitConversionError, match="no H0"):
        obj.mass(1 * u.Msun)


def test_base_stage_has_no_h0():
    assert Stage()._unit_context.H0 is None


def test_unit_context_requires_quantity_h0():
    with pytest.raises(UnitBoundaryError):
        UnitContext(70.0)


def test_decoration_errors():
    with pytest.raises(TypeError, match="has no argument 'mm'"):

        class Bad:
            @unit_boundary(mm=Msun_h)
            def f(self, m):
                return m

    with pytest.raises(TypeError, match="must be an astropy unit"):
        unit_boundary(m="Msun/littleh")

    # The output unit is attached without astropy's checks, so it is checked here.
    with pytest.raises(TypeError, match="'returns' must be an astropy unit"):
        unit_boundary(returns="Msun/littleh")
    with pytest.raises(TypeError, match="'returns' must be an astropy unit"):
        unit_boundary(returns=(Msun_h, "Msun/littleh"))


def test_decorator_metadata(toy):
    assert Toy.mass.__name__ == "mass"
    assert Toy.mass.__doc__ == "Return m unchanged (the identity kernel)."
    assert Toy.mass.__unit_boundary__ == {"inputs": {"m": Msun_h}, "returns": Msun_h}


def test_overhead_sanity(toy):
    """A generous check that the boundary is not grossly slow.

    The real budget (2 us per call) is measured by benchmarks/test_core_units.py; this
    only catches an order-of-magnitude regression, without being flaky.
    """
    import timeit

    m = np.ones(10) * Msun_h
    n = 2000
    per_call = min(timeit.repeat(lambda: toy.mass(m, z=0.0), number=n, repeat=5)) / n
    assert per_call < 50e-6


class _SubQuantity(u.Quantity):
    """A Quantity subclass: not the exact type the boundary's fast path checks for."""


@pytest.mark.parametrize("by_keyword", [False, True])
def test_fast_and_general_paths_agree(toy, by_keyword):
    """The boundary's inlined path and its general one give the kernel the same input.

    The inlined path takes an exact Quantity in the canonical unit; a subclass, or a
    physical unit, takes the general one. Both, by position or by keyword.
    """
    m = np.logspace(10, 15, 7)
    inputs = {
        "canonical": m * Msun_h,  # fast path
        "subclass": _SubQuantity(m, Msun_h),  # general path, no conversion
        "physical": (m / toy.H0.to_value(H0_unit) * 100) * u.Msun,  # general, converted
    }
    for name, q in inputs.items():
        out, _ = toy.raw(m=q) if by_keyword else toy.raw(q)
        assert type(out) is np.ndarray, name
        np.testing.assert_allclose(out, m, rtol=1e-14, err_msg=name)
    # The input is viewed, not copied, on the fast path.
    q = m * Msun_h
    assert np.shares_memory(toy.raw(q)[0], q)


def test_dimensional_argument_after_others():
    """A dimensional argument after others is converted in place.

    The arguments around it pass through unchanged.
    """

    class Later:
        @unit_boundary(m=Msun_h, returns=None)
        def f(self, z, m, scale=1.0):
            return z, m, scale

    m = np.logspace(10, 12, 3)
    z, out, scale = Later().f(0.5, m * Msun_h, 2.0)
    assert (z, scale) == (0.5, 2.0)
    assert type(out) is np.ndarray
    np.testing.assert_array_equal(out, m)


def test_two_dimensional_arguments_canonical_fast_path(toy):
    """With two dimensional arguments, canonical inputs are viewed, not copied."""
    m = np.logspace(10, 15, 5) * Msun_h
    r = np.linspace(1, 5, 5) * Mpc_h
    for out in (toy.mass_and_radius(m, r), toy.mass_and_radius(m, r=r, z=1.0)):
        assert type(out) is u.Quantity
        assert out.unit is Msun_h
        assert np.shares_memory(out, m)


def test_two_dimensional_arguments_convert_and_follow_the_scalar_rule(toy):
    out = toy.mass_and_radius(1e12 * u.Msun, 1.0 * u.Mpc)
    assert out.unit is Msun_h
    np.testing.assert_allclose(out.value, 1e12 * LITTLE_H, rtol=1e-14)
    ratio = toy.ratio(1e12 * Msun_h, r=2.0 * u.Mpc)
    assert type(ratio) is np.float64
    np.testing.assert_allclose(ratio, 1e12 / (2.0 * LITTLE_H), rtol=1e-14)
    assert toy.ratio(np.ones(3) * Msun_h, np.ones(3) * Mpc_h).shape == (3,)
    with pytest.raises(UnitBoundaryError):
        toy.mass_and_radius(1e12 * Msun_h, 1.0)


@pytest.mark.parametrize(
    "call",
    [
        lambda toy: toy.ratio(2e12 * u.Msun, 2.0 * u.Mpc),
        lambda toy: toy.ratio(2e12 * u.Msun, r=2.0 * u.Mpc),
        lambda toy: toy.ratio(m=2e12 * u.Msun, r=2.0 * u.Mpc),
        lambda toy: toy.ratio(r=2.0 * u.Mpc, m=2e12 * u.Msun),
        lambda toy: toy.ratio_named_backwards(2e12 * u.Msun, 2.0 * u.Mpc),
        lambda toy: toy.ratio_named_backwards(2e12 * u.Msun, r=2.0 * u.Mpc),
    ],
    ids=["positional", "mixed", "keywords", "keywords-reversed", "named-backwards", "nb-mixed"],
)
def test_two_dimensional_arguments_by_position_or_keyword(toy, call):
    """Every way of passing the two arguments converts both, to the same result."""
    out = call(toy)
    assert type(out) is np.float64
    np.testing.assert_allclose(out, 1e12, rtol=1e-14)


def test_two_dimensional_arguments_by_keyword_keep_arrays(toy):
    out = toy.ratio(m=np.full(3, 2.0) * Msun_h, r=np.full(3, 2.0) * Mpc_h)
    assert type(out) is np.ndarray
    np.testing.assert_array_equal(out, np.ones(3))


def test_two_dimensional_arguments_errors_are_the_methods(toy):
    with pytest.raises(TypeError, match="ratio"):
        toy.ratio(1e12 * Msun_h)
    with pytest.raises(TypeError, match="ratio"):
        toy.ratio()
    with pytest.raises(UnitBoundaryError):
        toy.ratio(1e12, 1.0 * Mpc_h)
    with pytest.raises(UnitBoundaryError):
        toy.ratio(r=1.0, m=1e12 * Msun_h)


def test_two_dimensional_arguments_after_another(toy):
    """Two dimensional arguments that are not the first two, by position or keyword."""
    for m, r in (
        toy.after_z(0.5, 1e12 * Msun_h, 2.0 * Mpc_h),
        toy.after_z(0.5, 1e12 * u.Msun, r=2.0 * u.Mpc),
        toy.after_z(0.5, 1e12 * Msun_h, 2.0 * LITTLE_H * u.Mpc),
        toy.after_z(z=0.5, m=1e12 * Msun_h, r=2.0 * Mpc_h),
    ):
        assert (m.unit, r.unit) == (Msun_h, Mpc_h)
    np.testing.assert_allclose(m.value, 1e12, rtol=1e-14)
    np.testing.assert_allclose(r.value, 2.0, rtol=1e-14)


def test_three_dimensional_arguments(toy):
    m, r, k = toy.three(1e12 * Msun_h, 1.0 * u.Mpc, k=1.0 / u.Mpc)
    assert (m.unit, r.unit, k.unit) == (Msun_h, Mpc_h, h_Mpc)
    np.testing.assert_allclose([m.value, r.value, k.value], [1e12, LITTLE_H, 1 / LITTLE_H])
    m, r, k = toy.three(m=1e12 * Msun_h, r=1.0 * Mpc_h, k=1.0 * h_Mpc)
    np.testing.assert_allclose([m.value, r.value, k.value], [1e12, 1.0, 1.0], rtol=1e-14)


def test_three_dimensional_arguments_array_outputs(toy):
    m = np.logspace(10, 12, 3) * Msun_h
    out = toy.three_mass(m, 1.0 * Mpc_h, 1.0 * h_Mpc)
    assert type(out) is u.Quantity
    assert out.unit is Msun_h
    assert np.shares_memory(out, m)
    product = toy.three_product(np.ones(3) * Msun_h, np.ones(3) * Mpc_h, np.ones(3) * h_Mpc)
    assert type(product) is np.ndarray
    assert type(toy.three_product(1.0 * Msun_h, 2.0 * Mpc_h, 3.0 * h_Mpc)) is np.float64
