FAQ
---

What are the units of hmf's outputs?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Firstly, the units of all quantities should be specified in their docstring (please
`<https://github.com/steven-murray/hmf/issues/new>make an issue`_ if this isn't the case!).
This can also be seen in the `<https://hmf.readthedocs.io/en/latest/_autosummary/hmf/hmf.mass_function.hmf.MassFunction.html#hmf.mass_function.hmf.MassFunction>API docs`_.

Nevertheless, it should be said that *all* quantities in ``hmf`` include little-*h*. So,
for example, masses are in units of Msun/h, and distances in Mpc/h. This is consistent
for every quantity in ``hmf``. Furthermore, little-h is defined as the value of the
Hubble constant divided by 100 km/s/Mpc (i.e. it is h_100, not h_70).
