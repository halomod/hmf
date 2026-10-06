Computing Growth Factors
========================

``hmf`` provides a number of different methods for computing the cosmological growth
factor. This document outlines the relationship between these approaches, and provides
a handy reference for their derivation (and derivation of various corollaries, such as
the growth rate). This is not meant so much to be a tutorial on how to use the different
growth factor classes, but rather a reference that gathers the different approaches
together and outlines their assumptions and limitations.

Notation
--------
Much of my own confusion in understanding the growth factor came from the fact that
different sources use different notation, and sometimes this notation is not specified.
In particular, whether :math:`\Omega_m` refers to the matter density at a given redshift, or
the matter density today, is not always clear. In this document, we will use the following notation:
density parameters (:math:`\Omega_m`, :math:`\Omega_\Lambda`, etc.) will always refer to
the density parameters at z=0, while the density parameter at a given redshift will be
denoted by :math:`\Omega_m(z)`.

Throughout, we will write the "normalized" hubble parameter as :math:`E(z) = H(z)/H_0`.
This is a common notation in the literature, and is also the notation used by the
``astropy.cosmology`` module. Since this quantity will appear frequently in the equations
for the growth factor, we will often use the notation :math:`E` without an explicit
argument to refer to :math:`E(z)`.

The scale factor will be denoted by :math:`a \equiv 1/(1+z)`, and the growth factor will
be denoted by :math:`D_+(a)`. Unless otherwise noted, primes will denote derivatives
with respect to *scale factor* (e.g. :math:`D'(a)`), while dots will denote derivatives
with respect to time (e.g. :math:`\dot{D}`).

.. _label-definitions:

Definitions
-----------

The growth factor, :math:`D_+(a)`, is the amplitude of a linear density perturbation
as a function of time, :math:`\delta(a) \propto D_+(a)`. It is defined up to a constant
factor, which is fixed by a choice of normalization (see below).

On sub-horizon scales, in an FLRW cosmological background and under the Newtonian
approximation, the growth factor satisfies the following second-order differential
equation [Heath77]_:

.. math:: \ddot{D} + 2 H \dot{D} - 4 \pi G \rho_m D = 0

where :math:`\rho_m` is the matter density.

In terms of the scale factor, this can be rewritten as [Haude22]_:

.. math:: D''(a) + \left(\frac{3}{a} + \frac{E'(a)}{E(a)}\right) D'(a) - \frac{3}{2} \frac{\Omega_m}{a^5 E(a)^2} D(a) = 0

This equation is the most general form of the growth factor that we will consider in
this document. ``hmf`` therefore doesn't support non-FLRW cosmologies, nor
scale-dependent growth factors at this time.

A closely related quantity is what we shall call the "growth function", which is
simply defined as :math:`G \equiv D(a)/a`. This is a convenient quantity to work with,
since it is constant in a matter-dominated universe (as we shall see later).
In terms of the growth function, the growth factor equation can be rewritten as:

.. math:: G''(a) + \left(\frac{5}{a} + \frac{E'(a)}{E(a)}\right) G'(a) + \left(\frac{3}{a^2} + \frac{E'(a)}{aE(a)} - \frac{3}{2} \frac{\Omega_m}{a^5 E(a)^2}\right) G(a) = 0

A final related quantity is the "growth rate", which is defined as

.. math:: f \equiv d\ln D / d\ln a.

Note that :math:`f` does not depend on the normalization of the growth factor, since it
is a logarithmic derivative.



Normalization
~~~~~~~~~~~~~

Since the growth factor satisfies a second-order differential equation, the equation
has two independent solutions. The "growing mode" solution is the one that we are interested in, and is
typically denoted by :math:`D_+(a)`. In this document we will simply use :math:`D(a)` to
refer to the growing mode solution, since it is the only solution that we will be interested in.

Second-order ODE's require two boundary conditions (or initial conditions) to specify a
unique solution. One such condition will define the overall normalization of :math:`D(a)`.
Most commonly in the literature, the growth factor is normalized such that
:math:`D(a) \to a` as :math:`a \to 0` (or, :math:`G \to 1` as :math:`a \to 0`).
When computing halo mass functions and related quantities, we generally first compute
the power spectrum at z=0, and then use the growth factor to scale it back to higher
redshifts. In this case, it is convenient to use the normalization :math:`D(a=1) = 1`,
so that we simply have :math:`P(k, z) = D(z)^2 P(k, z=0)`. Thus, in ``hmf``,
when you call ``growth_factor(z)`` you will get the growth factor normalized such that
:math:`D(a=1) = 1`. However, under the hood, most of the growth factor models are
specified such that :math:`D(a) \to a` as :math:`a \to 0`, and then the growth factor is
normalized at the end by dividing by :math:`D(a=1)`. In any case, none of this matters
too much -- the important thing is that whatever normalization that is chosen is used
consistently.

The second boundary condition selects the growing mode. During matter domination the
two solutions are :math:`D \propto a` (the growing mode, with :math:`G` constant) and
:math:`D \propto a^{-3/2}` (the decaying mode, with :math:`G \propto a^{-5/2}`).
Requiring :math:`G` to stay finite as :math:`a \to 0` removes the decaying mode, and
:math:`G \to 1` then fixes the normalization. The condition :math:`G'(a) \to 0` follows:
in a flat universe with a cosmological constant, for example,
:math:`G = 1 - \frac{2}{11} \frac{\Omega_\Lambda}{\Omega_m} a^3 + O(a^6)`. So the choice is
not arbitrary: it is the statement that there is no decaying mode. (Radiation changes
the early-time behaviour; see :ref:`label-radiation-dom`.)

Limiting Cases and Assumptions
------------------------------

The solutions to the growth factor equation depend critically on the form of :math:`E(a)`,
which in turn depends on the cosmological parameters and the assumptions that are made
about the contents of the universe. For the cosmologies we will actually consider (with
massless neutrinos), :math:`E(a)` is given by:

.. math:: E^2(a) = \Omega_m a^{-3} + \Omega_r a^{-4} + \Omega_{\rm DE} f_{\rm DE}(a) + \Omega_k a^{-2}

where :math:`\Omega_m`, :math:`\Omega_r`, :math:`\Omega_{\rm DE}`, and :math:`\Omega_k`
are the matter, radiation, dark energy, and curvature density parameters at z=0,
respectively, and

.. math:: f_{\rm DE}(a) = \exp\left(3\int_a^1 \frac{1 + w(a')}{a'} da'\right),

which is :math:`a^{-3(1+w)}` for a constant equation of state :math:`w`, and 1 for a
cosmological constant (:math:`w=-1`, when we write :math:`\Omega_\Lambda` for
:math:`\Omega_{\rm DE}`).

No solution to the growth factor equation is known in closed form for the general case,
but there are a number of limiting cases and approximations that are commonly used in the literature
(generally when one or more of the terms in the above equation can be neglected).

Negligible Radiation with a Cosmological Constant
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The most general solution that is known is the case in which radiation can be neglected
(i.e. at late times), and where the dark energy is a cosmological constant (i.e. :math:`w=-1`).
In this case, a solution is given by the following integral [Heath77]_:

.. math:: D(a) = C_1 E(a) \int_0^a \frac{da'}{(a'^3 E(a')^3)}.

Typically the coefficient :math:`C_1` is chosen such that :math:`D(a) \to a` as
:math:`a \to 0`, which gives :math:`C_1 = \frac{5}{2} \Omega_m`.

This solution comes from *reduction of order*. For these cosmologies, :math:`E(a)`
itself solves the growth factor equation (we show this below): it is the decaying mode,
since :math:`E \propto a^{-3/2}` during matter domination. Given one solution
:math:`D_1`, a second is :math:`D_2 = D_1 \int W/D_1^2 \, da`, where the Wronskian
:math:`W \propto \exp\left(-\int (3/a + E'/E) da\right) = 1/(a^3 E)`. With
:math:`D_1 = E` this gives :math:`D_2 = E \int da/(a^3 E^3)`. The general solution is
therefore :math:`E(a) \left[c + C_1 \int_0^a da'/(a'^3 E(a')^3)\right]`, and choosing
the lower limit 0 with :math:`c = 0` is exactly the choice of no decaying mode (the
second boundary condition above).

We can verify that this solution satisfies the growth factor equation by plugging it in and
differentiating (taking :math:`C_1 = 1` for simplicity and without loss of generality):

.. math:: D'(a) = E' D / E + \frac{1}{a^3 E^2}
.. math:: D''(a) = \frac{E'' D }{E} +  E' D' / E - E'^2 D / E^2 - \frac{3}{a^4 E^2} - \frac{2 E'}{a^3 E^3}

Plugging into the growth factor equation and collecting terms gives:

.. math:: D \left( E''/E - E'^2 / E^2 - \frac{3\Omega_m}{2a^5 E^2}\right)  + D' \left(2 E'/E + 3/a\right) - \frac{1}{a^3 E^2} \left(3/a + 2 E'/E \right) = 0
.. math:: D \left( E''/E - E'^2 / E^2 - \frac{3\Omega_m}{2a^5 E^2}\right)  + \left(2 E'/E + 3/a\right) \left( E'D/E + \frac{1}{a^3 E^2} - \frac{1}{a^3 E^2} \right) = 0
.. math:: D \left( E''/E + E'^2/E^2 + 3E'/(aE) - \frac{3\Omega_m}{2a^5 E^2} \right) = 0

The term in parentheses is exactly what we get by substituting :math:`D = E` into the
growth factor equation (divided by :math:`E`), so it vanishes precisely when :math:`E`
is itself a solution. Which choices of :math:`E(a)` will cause the term in parentheses
to be zero? We know that :math:`E(a)` must have the
form of a square root of some function. Let's say that :math:`E^2 = Q`.
Thus:

.. math:: E' = \frac{1}{2E} Q'
.. math:: E'' = -\frac{E'^2}{E} + \frac{1}{2E} Q''.

Plugging these back into the term in parentheses gives:

.. math:: -\frac{E'^2}{E^2} + \frac{1}{2E^2} Q'' + \frac{E'^2}{E^2} + 3E'/(aE) - \frac{3\Omega_m}{2a^5 E^2}
.. math:: \frac{1}{2E^2} Q'' + 3Q'/(2aE^2) - \frac{3\Omega_m}{2a^5 E^2}
.. math:: \frac{1}{2E^2} \left(Q'' + \frac{3}{a} Q' - \frac{3\Omega_m}{a^5} \right)

Now we need only solve the much simpler ODE defined by setting the term in parentheses
to zero, :math:`Q'' + 3Q'/a = 3\Omega_m a^{-5}`. The homogeneous equation gives
:math:`Q' \propto a^{-3}`, i.e. :math:`Q = K_1 a^{-2} + K_2`, and :math:`Q = \Omega_m a^{-3}`
is a particular solution, so

.. math:: Q = \Omega_m a^{-3} + K_1 a^{-2} + K_2.

Thus, this solution is only valid for cosmologies in which :math:`E^2` has the form of a
linear combination of :math:`a^{-3}`, :math:`a^{-2}`, and a constant. This corresponds
to cosmologies with matter, curvature (:math:`K_1 = \Omega_k`), and a cosmological
constant (:math:`K_2 = \Omega_\Lambda`), but no radiation or
other forms of dark energy.

Growth Rate
+++++++++++

The growth rate, :math:`f`, can be calculated from the growth factor analytically in the
case of negligible radiation and a cosmological constant (i.e. by using the
integral formula above for the growth factor). In this case, we can simply plug the
integral formula into the definition of the growth rate and differentiate. This gives:

.. math:: f = \frac{d\ln D}{d\ln a} = \frac{a D'}{D} = \frac{a E'}{E} + \frac{C_1}{a^2 E^2 D}.

Note that in the "further simplification" cases below, the growth rate can use this
formula as well, since they are all special cases of the integral formula.

Further Simplification: Flat Cosmology
++++++++++++++++++++++++++++++++++++++
If we further assume that the universe is flat (i.e. :math:`\Omega_k = 0` and
:math:`\Omega_\Lambda = 1 - \Omega_m`), then there
exists a closed-form solution for the growth factor, as derived in [Eisenstein97]_.
In this case, the growth factor is given by the following formula (their Eqs. 8-10):

.. math:: D(a) = a \times d(v), \qquad v = \frac{1}{a} \left(\frac{\Omega_m}{1 - \Omega_m}\right)^{1/3}

where

.. math:: d(v) = \frac{5}{3} v \left\{ \sqrt[4]{3} \sqrt{1 + v^3} \left[ E(\beta, \sin^2 75^\circ)  - \frac{1}{3 + \sqrt{3}} F(\beta, \sin^2 75^\circ)\right] + \frac{1 - (\sqrt{3} + 1)v^2}{v + 1 + \sqrt{3}}  \right\}

and

.. math:: \beta = \arccos \left( \frac{1 + v - \sqrt{3}}{1 + v + \sqrt{3}} \right)

and :math:`E` and :math:`F` (not to be confused with :math:`E(a)`) are the incomplete
elliptic integrals of the second and first kind, respectively. Their second argument is
the *parameter* :math:`m = k^2 = \sin^2 75^\circ`, as in ``scipy.special.ellipeinc`` and
``scipy.special.ellipkinc``; texts that use the modulus :math:`k` write
:math:`\sin 75^\circ` instead.

This solution is normalized in the standard way, i.e. :math:`D(a) \to a` as :math:`a \to 0`.
For computational purposes, [Eisenstein97]_ also provides an asymptotic expansion for
the growth factor at high redshift:

.. math:: d(v) \approx 1 - \frac{2}{11} v^{-3} + \frac{16}{187} v^{-6} + O(v^{-9}).

Further Simplification: No Cosmological Constant
++++++++++++++++++++++++++++++++++++++++++++++++
If we further assume that the cosmological constant is negligible (i.e. :math:`\Omega_\Lambda = 0`),
then [Heath77]_ provides a closed-form solution for the growth factor. Normalized
so that :math:`D \to a` as :math:`a \to 0`, it is

.. math:: D(z) = \frac{5\Omega_m}{2|\Omega_m - 1|} \left[ \frac{3\Omega_m z + 2\Omega_m + 1}{|\Omega_m - 1|} - \frac{3\Omega_m \theta (1 + z) \sqrt{1 + \Omega_m z}}{2|\Omega_m - 1|^{3/2}} \right],

where

.. math:: \theta = \begin{cases} \text{arccosh}(x) & \text{if } \Omega_m < 1 \text{ (open)} \\ \arccos(x) & \text{if } \Omega_m > 1 \text{ (closed)} \end{cases}

and

.. math:: x = 1 + \frac{2(1 - \Omega_m)}{\Omega_m (1 + z)}.

Further Simplification: Einstein-de Sitter
++++++++++++++++++++++++++++++++++++++++++
Finally, if we further assume that the universe is matter-dominated (i.e. :math:`\Omega_m = 1` and
:math:`\Omega_\Lambda = 0`), then the growth factor is simply given by :math:`D(a) = a`.

This can be proven simply by plugging :math:`D(a) = a` into the growth factor equation
and noting that :math:`E(a) = a^{-3/2}` in this case, which gives:

.. math:: D''(a) + \left(\frac{3}{a} + \frac{E'(a)}{E(a)}\right) D'(a) - \frac{3}{2} \frac{\Omega_m}{a^5 E(a)^2} D(a) = 0
.. math:: 0 + \left(\frac{3}{a} - \frac{3}{2a} \right) - \frac{3}{2} \frac{1}{a^5 \Omega_m a^{-3}} a = 0
.. math:: \frac{3}{2a} - \frac{3}{2a} = 0.

.. _label-radiation-dom:

Matter-Radiation Domination
~~~~~~~~~~~~~~~~~~~~~~~~~~~
At early times, radiation cannot be neglected, and the growth factor is not given by the
integral formula above.

At the very earliest times, the universe is radiation-dominated, and the growth factor
satisfies (with :math:`E^2 = \Omega_r a^{-4}`) :math:`x D_{xx} + D_x - \frac{3}{2} D = 0`,
where :math:`x = \Omega_m a / \Omega_r` (i.e. the scale factor as a ratio to the scale
factor at radiation-matter equality, :math:`a_{\rm eq} = \Omega_r/\Omega_m`). Its
solutions are

.. math:: D(a) = C_1 I_0(\sqrt{6x}) + C_2 K_0(\sqrt{6x})

where :math:`I_0` and :math:`K_0` are the *modified* Bessel functions of the first and
second kind, respectively (the ordinary Bessel functions would oscillate).

:math:`K_0` diverges as :math:`x \to 0`, so we choose :math:`C_2 = 0` to get the growing
mode solution.

:math:`I_0` approaches unity as :math:`x \to 0`. Thus, the
growth factor approaches a constant as :math:`a \to 0` in the radiation-dominated era.

This is in contrast to the matter-dominated case, where the growth factor approaches
zero as :math:`a \to 0`. Thus, the growth factor grows more slowly in the
radiation-dominated era than in the matter-dominated era.

This also helps us set an appropriate initial condition for the growth factor when
solving the growth factor equation numerically. Before getting to that though, we first
solve the growth factor equation in the era when only matter and radiation are important
(i.e. when the cosmological constant and curvature can be neglected).
This can be solved with Mathematica, *as long as you give it reasonable initial
conditions* -- here we simply assert that the growth factor approaches a constant as
:math:`a \to 0` and get:

.. math:: D(a) = a + 2a_{\rm eq}/3.

Note that one of the ODE coefficients has implicitly been chosen so that we asymptote
to a constant to enable this form, the second is the overall normalization, which we
choose here such that during matter domination (i.e. late times in this scenario) we have
:math:`D(a) \to a`. This can help us connect this solution to the full solution at late
times.

Note also that if the universe is flat and has no dark energy, this is the exact solution.

Growth Rate at z=0 for Cosmological Constant
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The growth factor at z=0 is always normalized to 1. However, the Growth Rate is not
dependent on this normalization, and therefore can change with cosmological parameters.

Assuming that radiation can be neglected (very likely to be a good approximation at z=0),
[Hamilton01]_ provides an exact solution for the growth rate in both flat and non-flat
universes. Unfortunately, I think the formula itself probably takes as much time to
evaluate as the ODE, so I do not include it in ``hmf`` at this time.

However, it is worth noting that, since we are ignoring radiation and assuming a cosmological
constant, the integral formula above applies. In this case, the growth rate
has a simple form (already stated above), and is even simpler at :math:`a = 1`, where
:math:`E = 1` (Eq. 4 of [Hamilton01]_). With :math:`E^2 = \Omega_m a^{-3} + \Omega_k a^{-2} + \Omega_\Lambda`
and :math:`\Omega_k = 1 - \Omega_m - \Omega_\Lambda`, we have
:math:`E'(1) = -\frac{3}{2}\Omega_m - \Omega_k`, so

.. math:: f(z=0) = E'(1) + \frac{C_1}{D(1)} = -1 - \frac{\Omega_m}{2} + \Omega_\Lambda + \left[\int_0^1 \frac{da}{a^3 E(a)^3}\right]^{-1}.

Here :math:`D` is the growth factor normalized so that :math:`D \to a` as :math:`a \to 0`
(i.e. with :math:`C_1 = \frac{5}{2}\Omega_m`), *not* the one normalized to
:math:`D(1) = 1`.

Dark Energy with w != -1 in a Flat Universe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In [SW94]_ they give a solution for the growth factor in a flat universe with a dark
energy component with constant equation of state w. This solution is given in more modern
notation in [VT20]_ as:

.. math:: D(a) = a \times {}_2F_1\left(-\frac{1}{3w}, \frac{w-1}{2w}, 1 - \frac{5}{6w}, a^{-3w}\left(1 - \frac{1}{\Omega_m}\right)\right)

where :math:`w` is the equation of state parameter for the dark energy component, and
:math:`{}_2F_1` is the hypergeometric function.

The case :math:`w=-1` is included: it gives
:math:`D(a) = a \, {}_2F_1\left(\frac{1}{3}, 1; \frac{11}{6}; -a^3 \Omega_\Lambda/\Omega_m\right)`,
which is the flat case of the integral formula above (and of [Eisenstein97]_).

Solving the ODE
---------------
The full ODE (without any assumptions/approximations on :math:`E(a)`) does not admit
closed-form solutions. However, it can be solved numerically. A convenient way to setup
the problem is as an initial value problem for a coupled system of first-order ODEs,
which can be solved with standard ODE solvers. In this case, the relevant system is
(see :ref:`label-definitions`):

.. math:: \begin{cases} x_1' = x_2 \\ x_2' = -\left(\frac{3}{a} + \frac{E'}{E}\right) x_2 + \frac{3}{2} \frac{\Omega_m}{a^5 E^2} x_1 \end{cases}

where :math:`x_1 = D` and :math:`x_2 = D'`.

The only question is how to properly set the initial conditions for this system.
It is natural to set the initial conditions in the early universe, but it is common
to find initial conditions actually set in the matter-dominated era, where :math:`D(a) \to a`
and :math:`D'(a) \to 1`. However, choosing such initial conditions restricts the range
of redshifts in the solution to begin in matter-domination---probably enough for most
applications but still a needless restriction. Furthermore, it is also a little arbitrary,
since one must decide on which redshift is best representative of matter domination for
the particular cosmology on hand. A more natural choice is to set the initial conditions
in the radiation-dominated era, where we can start at arbitrarily high redshift and use
the solution for the growth factor in the radiation-dominated era to set the initial
conditions. In this case, :math:`D(a) = a + 2 a_{\rm eq}/3`, so that
:math:`D \to 2 a_{\rm eq}/3` and :math:`D' = 1` as :math:`a \to 0`
(see :ref:`label-radiation-dom`). ``hmf`` v3 starts the integration from
:math:`D = 2a_{\rm eq}/3` and :math:`D' = 0` instead. That adds a small decaying-mode
component, but after normalizing to :math:`D(a=1) = 1` its effect is below
:math:`10^{-5}` at :math:`z \lesssim 1000`.


Approximations
--------------

Beyond the limiting cases and simplifying assumptions laid out in the previous section,
there exist some commonly used approximations. Most popular is the approximation of
[Lahav91]_, which gives the following formula for the growth rate at :math:`z=0`:

.. math:: f(z=0) \approx \Omega_m^{0.6} + \frac{1}{70} \Omega_\Lambda \left(1 + \frac{\Omega_m}{2}\right).

They argue that the growth rate should be largely dependent on the matter density at any
particular epoch, and give the formula

.. math:: f(z) \approx \Omega_m(z)^{0.6}

which is a good approximation for any :math:`\Lambda` and :math:`\Omega_m`,
and also a similar formula as a function of redshift which is an even better
approximation in a flat universe:

.. math:: f(z) \approx \Omega_m(z)^{0.6} + \frac{1}{70} \left(1 - \frac{\Omega_m(z)}{2}(1 + \Omega_m(z))\right).

It is also very common to use the approximation :math:`f(z) \approx \Omega_m(z)^\gamma`,
where :math:`\gamma` is a free parameter, which can be fit to data or predicted by a
given theory of gravity. This is a
convenient formula, since it allows for a simple way to parametrize deviations from the
standard growth rate in modified gravity theories, and is commonly used in the
literature for this purpose. For a flat universe with a cosmological constant,
:math:`\gamma \approx 0.55` gives a good fit to the growth rate at z=0.

Since the growth rate approximation is a good approximation for any redshift (not including
when radiation becomes important), we can use the analytic formula for the growth rate
in the "integral" case above to get a similar approximation for the growth factor itself
[Carroll92]_:

.. math:: D(a) \approx a \times \frac{5}{2} \Omega_m(a) \left[ \Omega_m(a)^{4/7} - \Omega_\Lambda(a) + \left(1 + \frac{\Omega_m(a)}{2}\right)\left(1 + \frac{\Omega_\Lambda(a)}{70}\right) \right]^{-1}.


References
----------
.. [Heath77] Heath, D. J. ‘The Growth of Density Perturbations in Zero Pressure Friedmann-Lemaître Universes.’ Monthly Notices of the Royal Astronomical Society 179 (May 1977): 351–58. https://doi.org/10.1093/mnras/179.3.351.
.. [Haude22] Haude, Sophia, Shabnam Salehi, Sofía Vidal, Matteo Maturi, and Matthias Bartelmann. ‘Model-Independent Determination of the Cosmic Growth Factor’. SciPost Astronomy 2, no. 1 (2022): 001. https://doi.org/10.21468/SciPostAstro.2.1.001.
.. [Eisenstein97] Eisenstein, Daniel J. ‘An Analytic Expression for the Growth Function in a Flat Universe with a Cosmological Constant’. arXiv:astro-Ph/9709054. Preprint, arXiv, 12 September 1997. https://doi.org/10.48550/arXiv.astro-ph/9709054.
.. [Hamilton01] Hamilton, A. J. S. ‘Formulae for Growth Factors In Expanding Universes Containing Matter and a Cosmological Constant’. Monthly Notices of the Royal Astronomical Society 322, no. 2 (2001): 419–25. https://doi.org/10.1046/j.1365-8711.2001.04137.x.
.. [Lahav91] Lahav, Ofer, Per B. Lilje, Joel R. Primack, and Martin J. Rees. ‘Dynamical Effects of the Cosmological Constant’. Monthly Notices of the Royal Astronomical Society 251, no. 1 (1991): 128–36. https://doi.org/10.1093/mnras/251.1.128.
.. [Carroll92] Carroll, Sean M., William H. Press, and Edwin L. Turner. ‘The Cosmological Constant.’ Annual Review of Astronomy and Astrophysics 30 (January 1992): 499–542. https://doi.org/10.1146/annurev.aa.30.090192.002435.
.. [SW94] Silveira, V., and I. Waga. ‘Decaying Λ Cosmologies and Power Spectrum’. Physical Review D 50, no. 8 (1994): 4890–94. https://doi.org/10.1103/PhysRevD.50.4890.
.. [VT20] Velasquez-Toribio, A. M., and Júlio C. Fabris. ‘The Growth Factor Parametrization versus Numerical Solutions in Flat and Non-Flat Dark Energy Models’. The European Physical Journal C 80, no. 12 (2020): 1210. https://doi.org/10.1140/epjc/s10052-020-08785-z.
