r"""Smoothing filters: the window functions that define the mass variance.

A filter is a :class:`~hmf.core.model.Model` of the :class:`Filter` kind. It defines
a window :math:`W(x)` of the dimensionless :math:`x = kR`, its logarithmic
derivatives, and the mass assignment :math:`m = \frac{4\pi}{3} \bar\rho\, (cR)^3`.
The mass variance itself is computed by :class:`~hmf.core.mass_variance.MassVariance`.

Every method here is dimensionless, so none needs a units boundary.

Three filters are provided, ported from hmf 3.x:

* :class:`TopHat` (alias ``"TopHat"``): the real-space top-hat;
* :class:`SharpK` (alias ``"SharpK"``): the sharp-k filter, a top-hat in k-space;
* :class:`SmoothK` (alias ``"SmoothK"``): the smooth-k filter of Leo et al. (2018).
"""

from __future__ import annotations

import abc
from typing import ClassVar, NamedTuple

import attrs
import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._fields import field
from ._kernels import filters as kernels
from .model import Model

__all__ = ["Envelope", "Filter", "SharpK", "SmoothK", "TailBounds", "TopHat"]

#: A bound :math:`\sum_i c_i x^{p_i}` (with :math:`c_i \ge 0`) on a function of
#: :math:`x = kR`, as ``((c_0, p_0), (c_1, p_1), ...)``.
Envelope = tuple[tuple[float, float], ...]


class TailBounds(NamedTuple):
    r"""Bounds on the window products beyond the ends of the k grid.

    Each field is an :data:`Envelope` bounding one of the integrands' window factors,
    :math:`W^2`, :math:`W W'` and :math:`W'^2 + W W''` (the factors of
    :math:`s`, :math:`s'/2` and :math:`s''/2`):

    * ``high_*``: for large :math:`x` (beyond :math:`k_{\max} R`, at least a few),
      split into a smooth part and the amplitude (``high_*_oscillating``) of a part
      oscillating as :math:`\sin(\omega x + \phi)`, whose tail integral is much
      smaller than that of its envelope;
    * ``low_*``: for :math:`x = k_{\min} R \ll 1`.
    """

    high_w2: Envelope
    high_wdw: Envelope
    low_w2: Envelope
    low_wdw: Envelope
    high_w2_oscillating: Envelope = ()
    high_wdw_oscillating: Envelope = ()
    omega: float = 1.0
    high_ddw: Envelope = ()
    high_ddw_oscillating: Envelope = ()
    low_ddw: Envelope = ()


def _positive(instance: object, attribute: attrs.Attribute[float], value: float) -> None:
    if not (np.isfinite(value) and value > 0):
        raise ValueError(f"{type(instance).__name__}.{attribute.name} must be > 0, got {value}.")


@attrs.frozen(kw_only=True)
class Filter(Model, kind=True):
    r"""The kind of smoothing filters.

    A filter defines:

    * the window :math:`W(x)` of :math:`x = kR` (:meth:`window`), with
      :math:`W(0) = 1`;
    * its derivatives :math:`W' = dW/d\ln x` and :math:`W'' = d^2W/d(\ln x)^2`
      (:meth:`window_derivatives`);
    * the mass-assignment constant :math:`c` (:attr:`mass_assignment`), in
      :math:`m = \frac{4\pi}{3} \bar\rho\, (cR)^3`;
    * envelopes bounding :math:`W^2` and :math:`|W W'|` far beyond the ends of the
      k grid (:meth:`tail_bounds`), for the truncation estimator of
      :class:`~hmf.core.mass_variance.MassVariance`.

    A filter whose window is a sharp cut-off sets :attr:`sharp_cutoff`; its variance
    is then integrated up to the cut-off exactly, not through the window.
    """

    #: Whether the window is the sharp cut-off :math:`\Theta(1 - x)`.
    sharp_cutoff: ClassVar[bool] = False

    @property
    def mass_assignment(self) -> float:
        r"""The constant :math:`c` of the mass assignment :math:`m \propto (cR)^3`."""
        return 1.0

    @abc.abstractmethod
    def window_derivatives(self, x: ArrayLike, order: int = 2) -> tuple[NDArray[np.float64], ...]:
        r"""The window and its logarithmic derivatives at :math:`x = kR`.

        Parameters
        ----------
        x
            :math:`kR`, dimensionless, ``>= 0``.
        order
            The highest derivative to return: 0, 1 or 2.

        Returns
        -------
        tuple of ndarray
            :math:`(W,)`, :math:`(W, W')` or :math:`(W, W', W'')`.
        """

    def window(self, x: ArrayLike) -> NDArray[np.float64]:
        r"""The window :math:`W(x)`, with :math:`x = kR`.

        Parameters
        ----------
        x
            :math:`kR`, dimensionless, ``>= 0``.

        Returns
        -------
        ndarray
            The window.
        """
        return self.window_derivatives(x, order=0)[0]

    def dwindow_dlnx(self, x: ArrayLike) -> NDArray[np.float64]:
        r"""The derivative :math:`dW/d\ln x`, with :math:`x = kR`.

        Parameters
        ----------
        x
            :math:`kR`, dimensionless, ``>= 0``.

        Returns
        -------
        ndarray
            The derivative.
        """
        return self.window_derivatives(x, order=1)[1]

    @abc.abstractmethod
    def tail_bounds(self) -> TailBounds:
        """Bounds on the window beyond the ends of the k grid.

        Returns
        -------
        TailBounds
            The bounds on :math:`W^2` and :math:`|W W'|` at both ends.
        """


@attrs.frozen(kw_only=True)
class TopHat(Filter, alias="TopHat"):
    r"""The real-space top-hat filter.

    .. math:: W(x) = \frac{3 (\sin x - x \cos x)}{x^3}, \qquad
              m = \frac{4\pi}{3} \bar\rho R^3.

    Below :math:`x = 0.1` the window and its derivatives are evaluated from their
    Taylor series (to :math:`x^{10}`), since the closed forms lose precision to
    cancellation. This keeps the leading :math:`dW/d\ln x \approx -x^2/5`, which
    carries all of :math:`d\ln\sigma/d\ln m` at small masses when the power spectrum
    is truncated at high k (e.g. warm dark matter); see hmf 3.7.2.
    """

    def window_derivatives(self, x: ArrayLike, order: int = 2) -> tuple[NDArray[np.float64], ...]:
        """See :meth:`Filter.window_derivatives`."""
        return kernels.tophat_window_derivatives(x, order)

    def tail_bounds(self) -> TailBounds:
        r"""See :meth:`Filter.tail_bounds`.

        Exactly, with :math:`\omega = 2`,

        .. math::

            W^2 &= \frac{9 (1 + x^2)}{2 x^6} \left[1 + \cos(2x + \phi)\right], \\
            W W' &= -\frac{9 (x^2 + 3/2)}{x^6} + \frac{9}{2x^6}\left[(3 - 4x^2)
                \cos 2x + (6x - x^3) \sin 2x\right],

        so the oscillating amplitude of :math:`W W'` is at most
        :math:`\frac{9}{2}(x^{-3} + 4x^{-4} + 6x^{-5} + 3x^{-6})`, and

        .. math::

            W'^2 + W W'' = \frac{36x^2 + 81 + (\frac{99}{2}x^3 - 162x) \sin 2x
                + (-9x^4 + 126x^2 - 81) \cos 2x}{x^6}.

        For :math:`x \ll 1`, :math:`W \le 1`, :math:`|W'| \le x^2/5` and
        :math:`|W''| \le 2x^2/5`.
        """
        return TailBounds(
            high_w2=((4.5, -4.0), (4.5, -6.0)),
            high_wdw=((9.0, -4.0), (13.5, -6.0)),
            low_w2=((1.0, 0.0),),
            low_wdw=((0.2, 2.0),),
            high_w2_oscillating=((4.5, -4.0), (4.5, -6.0)),
            high_wdw_oscillating=((4.5, -3.0), (18.0, -4.0), (27.0, -5.0), (13.5, -6.0)),
            omega=2.0,
            high_ddw=((36.0, -4.0), (81.0, -6.0)),
            high_ddw_oscillating=(
                (9.0, -2.0),
                (49.5, -3.0),
                (126.0, -4.0),
                (162.0, -5.0),
                (81.0, -6.0),
            ),
            low_ddw=((0.4, 2.0), (0.04, 4.0)),
        )


@attrs.frozen(kw_only=True)
class SharpK(Filter, alias="SharpK"):
    r"""The sharp-k filter: a top-hat in k-space.

    .. math:: W(x) = \Theta(1 - x), \qquad m = \frac{4\pi}{3} \bar\rho\, (cR)^3.

    Its derivative is a Dirac delta, :math:`dW/d\ln x = -\delta_D(\ln x)`, so the
    variance and its derivatives have closed forms in the power at the cut-off
    :math:`k = 1/R`:

    .. math::

        \sigma^2(R) = \frac{1}{2\pi^2} \int_0^{1/R} k^3 P(k)\, d\ln k, \qquad
        \frac{d\ln\sigma^2}{d\ln R} = -\frac{P(1/R)}{2\pi^2 \sigma^2 R^3}.

    :class:`~hmf.core.mass_variance.MassVariance` integrates the first exactly up to
    the cut-off and evaluates the power at the cut-off directly (not through a
    spline). The derivative is a ratio of the power to the variance, so it does not
    depend on the normalisation of the power.
    """

    sharp_cutoff: ClassVar[bool] = True
    parameter_source: ClassVar[str] = "hmf 3.x default"

    c: float = field(
        default=2.5,
        converter=float,
        validator=_positive,
        doc="The mass-assignment constant c, in m = (4 pi / 3) rho_mean (c R)^3.",
    )

    @property
    def mass_assignment(self) -> float:
        """See :attr:`Filter.mass_assignment`: the parameter ``c``."""
        return self.c

    def window_derivatives(self, x: ArrayLike, order: int = 2) -> tuple[NDArray[np.float64], ...]:
        """See :meth:`Filter.window_derivatives`; only ``order=0`` exists.

        Raises
        ------
        ValueError
            If ``order > 0``: the derivative of the sharp-k window is a Dirac delta.
        """
        if order > 0:
            raise ValueError(
                "The derivative of the sharp-k window is a Dirac delta, -delta(ln x), so "
                "it can not be evaluated pointwise. MassVariance handles it in closed form."
            )
        return (kernels.sharpk_window(x),)

    def tail_bounds(self) -> TailBounds:
        """See :meth:`Filter.tail_bounds`. The window is exactly 0 beyond the cut-off."""
        return TailBounds(high_w2=(), high_wdw=(), low_w2=((1.0, 0.0),), low_wdw=())


@attrs.frozen(kw_only=True)
class SmoothK(Filter, alias="SmoothK"):
    r"""The smooth-k filter of Leo et al. (2018), their eq. 4.1.

    .. math:: W(x) = \frac{1}{1 + x^\beta}, \qquad
              m = \frac{4\pi}{3} \bar\rho\, (cR)^3.

    It interpolates between the top-hat and the sharp-k filter (the limit
    :math:`\beta \to \infty`), for power spectra truncated at small scales. The
    variance of a power law :math:`P \propto k^n` converges for :math:`n < 2\beta - 3`.
    """

    references: ClassVar[tuple[str, ...]] = (
        "Leo, M., Baugh, C. M., Li, B., Pascoli, S., 2018. JCAP 04, 010. arXiv:1801.02547",
    )
    parameter_source: ClassVar[str] = "Leo et al. 2018, JCAP 04, 010, Section 5 (best fit)"

    beta: float = field(
        default=4.8,
        converter=float,
        validator=_positive,
        doc="The steepness of the cut-off, beta > 0.",
    )
    c: float = field(
        default=3.3,
        converter=float,
        validator=_positive,
        doc="The mass-assignment constant c, in m = (4 pi / 3) rho_mean (c R)^3.",
    )

    @property
    def mass_assignment(self) -> float:
        """See :attr:`Filter.mass_assignment`: the parameter ``c``."""
        return self.c

    def window_derivatives(self, x: ArrayLike, order: int = 2) -> tuple[NDArray[np.float64], ...]:
        """See :meth:`Filter.window_derivatives`."""
        return kernels.smoothk_window_derivatives(x, self.beta, order)

    def tail_bounds(self) -> TailBounds:
        r"""See :meth:`Filter.tail_bounds`.

        :math:`W \le x^{-\beta}`, :math:`|W'| = \beta W(1-W) \le \beta x^{-\beta}`
        and :math:`|W'^2 + W W''| = \beta^2 W^2 (1-W) |2 - 3W| \le 2\beta^2 W^2` for
        large x; :math:`W \le 1`, :math:`|W'| \le \beta x^\beta` and
        :math:`|W'^2 + W W''| \le 2\beta^2 x^\beta` for small x.
        """
        b = self.beta
        return TailBounds(
            high_w2=((1.0, -2 * b),),
            high_wdw=((b, -2 * b),),
            low_w2=((1.0, 0.0),),
            low_wdw=((b, b),),
            high_ddw=((2 * b * b, -2 * b),),
            low_ddw=((2 * b * b, b),),
        )
