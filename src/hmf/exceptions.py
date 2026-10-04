"""Warnings and exceptions specific to hmf.

hmf follows a single policy for inputs outside the domain in which a model (or a
numerical approximation) is valid:

* if the result is still finite and physically sensible, but relies on extrapolating a
  model beyond its calibrated or sampled range, a :class:`HMFExtrapolationWarning` is
  emitted;
* if the result would be unphysical (e.g. negative or NaN parameters), a
  :class:`ValueError` is raised.

hmf never silently returns 0 or NaN in place of a result.

To turn extrapolation warnings into errors, or to silence them, use the standard
:mod:`warnings` filters, e.g.::

    import warnings
    from hmf import HMFExtrapolationWarning

    warnings.simplefilter("error", HMFExtrapolationWarning)
"""


class HMFExtrapolationWarning(UserWarning):
    """A model or approximation is being used outside its calibrated/sampled domain.

    The returned values are finite, but rely on extrapolation, so they should be
    treated with caution.
    """


class HMFCoreExperimentalWarning(FutureWarning):
    """Emitted on ``import hmf.core``: the v4 core is an experimental preview.

    Its API may change before hmf 4.0. The warning is defined here, rather than in
    :mod:`hmf.core`, so that it can be filtered *before* :mod:`hmf.core` is imported::

        import warnings
        from hmf.exceptions import HMFCoreExperimentalWarning

        warnings.simplefilter("ignore", HMFCoreExperimentalWarning)
        import hmf.core
    """
