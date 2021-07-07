"""Functions for generating several :class:`hmf.hmf.MassFunction` instances at once.

The underlying idea here is that typically modifying say the redshift has a smaller
number of re-computations than modifying the base cosmological parameters. Thus, in a
nested loop, the redshift should be the inner loop.

It is not always obvious which order the loops should be, so this module provides
functions to determine that order, and indeed perform the loops.
"""

import collections
import itertools

from ..mass_function import hmf


def get_best_param_order(kls, q="dndm", **kwargs):
    """
    Get an optimal parameter order for nested loops.

    The underlying idea here is that typically modifying say the redshift
    has a smaller number of re-computations than modifying the base cosmological
    parameters. Thus, in a nested loop, the redshift should be the inner loop.

    It is not always obvious which order the loops should be, so this function
    determines that order.

    .. note :: The order is calculated based on actually running one iteration
               of the loop, so providing arguments which enable a fast calculation
               is helpful.

    Parameters
    ----------
    kls : :class:`hmf._framwork.Framework` class
        An arbitrary framework for which to determine parameter ordering.
    q : str or list of str
        A string specifying the desired output (e.g. ``"dndm"``), or a list of
        such strings.
    kwargs : unpacked-dict
        Arbitrary keyword arguments to the framework initialiser. These are only
        for determination of parameter order and so may be very poor in resolution
        to improve efficiency.

    Returns
    -------
    final_list : list
        An ordered list of parameters, with the first corresponding to the outer-most
        loop.

    Examples
    --------
    >>> from src.hmf import MassFunction
    >>> print get_best_param_order(
    >>>     MassFunction,"dndm",transfer_model="BBKS",dlnk=1,dlog10m=1
    >>> )[::3]
    ['z2', 'hmf_model', 'delta_wrt', 'growth_params', 'filter_params',
     'Mmin', 'transfer_params', 'dlnk', 'cosmo_params']
    """
    a = kls(**kwargs)

    if isinstance(q, str):
        getattr(a, q)
    else:
        for qq in q:
            getattr(a, qq)

    final_list = []
    final_num = []
    for i, (k, v) in enumerate(
        getattr(a, "_" + a.__class__.__name__ + "__recalc_par_prop").items()
    ):
        num = len(v)
        for ln in final_num:
            if ln >= num:
                break
        else:
            final_list += [k]
            final_num += [num]
            continue
        final_list.insert(i, k)
        final_num.insert(i, num)
    return final_list[::-1]


def get_hmf(
    req_qauntities,
    get_label=True,
    framework=hmf.MassFunction,
    fast_kwargs={  # noqa: B006
        "transfer_model": "BBKS",
        "lnk_min": -1,
        "lnk_max": 1,
        "dlnk": 1,
        "Mmin": 10,
        "Mmax": 11.5,
        "dlog10m": 0.5,
    },
    label_kind="display",
    label_kwargs=None,
    **kwargs,
):
    """
    Yield framework instances for all combinations of parameters supplied.

    The underlying idea here is that typically modifying say the redshift
    has a smaller number of re-computations than modifying the base
    cosmological parameters. Thus, in a nested loop, the redshift should
    be the inner loop.

    It is not always obvious which order the loops should be, but this function
    internally determines the order, and calculates the requisite quantities in
    a series of framework instances.

    Parameters
    ----------
    req_qauntities : str or list of str
        A string defining the quantities that should be pre-cached in the output
        instances. It is advisable that *any* required quantities for a given
        application be provided here, to ensure proper optimization.
    get_label : bool, optional
        Whether to return a list of string labels designating each combination
        of parameters.
    framework : :class:`hmf._framework.Framework` class, optional
        A framework for which to perform the optimization.
    fast_kwargs : dict, optional
        Parameters to be used in the initial run to determine optimal order.
        These should be set to provide very quick calculation, and do not affect
        the final result. This will need to be over-ridden for frameworks other
        than :class:`hmf.MassFunction`.
    kwargs : unpacked-dict
        Any of the parameters to the initialiser of `framework` which should be
        calculated. These may be scalar or lists. The total number of calculations
        will be the total combination of all parameters.

    Yields
    ------
    quantities : list
        A list of quantities, specified by the `req_quantities` arguments
    x : Framework instance
        An instance of `framework`, with the requisite quantities pre-cached.
    label : optional
        If `get_label` is True, also returns a string label uniquely specifying
        the current parameter combination.

    Examples
    --------
    The following operation will run 12 iterations, yielding the desired quantities,
    an instance containing those and other quantities, and a unique label at every
    iteration.

    >>> for quants, mf, label in get_hmf(
    >>>     ['dndm','ngtm'], z=range(3),hmf_model=["ST","PS"],sigma_8=[0.7,0.8]
    >>> ):
    >>>     print label
    sigma.8: 0.7, ST, z: 0
    sigma.8: 0.7, PS, z: 0
    sigma.8: 0.7, ST, z: 1
    sigma.8: 0.7, PS, z: 1
    sigma.8: 0.7, ST, z: 2
    sigma.8: 0.7, PS, z: 2
    sigma.8: 0.8, ST, z: 0
    sigma.8: 0.8, PS, z: 0
    sigma.8: 0.8, ST, z: 1
    sigma.8: 0.8, PS, z: 1
    sigma.8: 0.8, ST, z: 2
    sigma.8: 0.8, PS, z: 2

    To calculate all of them and keep the results as a list:

    >>> big_list = list(get_hmf('mean_density',z=range(8)))
    >>> print [x[0][0]/1e10 for x in big_list]
    [8.531878308131338, 68.2550264650507, 230.36071431954613, 546.0402117204056,
     1066.4847885164174, 1842.885714556369, 2926.434259689049, 4368.321693763245]
    """
    label_kwargs = label_kwargs or {}

    if isinstance(req_qauntities, str):
        req_qauntities = [req_qauntities]
    lists = {}
    for k, v in list(kwargs.items()):
        if isinstance(v, (list, tuple)):
            if len(v) > 1:
                lists[k] = kwargs.pop(k)
            else:
                kwargs[k] = v[0]

    x = framework(**kwargs)
    if not lists:
        if get_label:
            yield [[getattr(x, a) for a in req_qauntities], x, ""]
        else:
            yield [[getattr(x, a) for a in req_qauntities], x]

    if len(lists) == 1:
        for k, v in lists.items():
            for vv in v:
                x.update(**{k: vv})
                if get_label:
                    yield [getattr(x, a) for a in req_qauntities], x, _make_label(
                        {k: vv}, kind=label_kind, **label_kwargs
                    )
                else:
                    yield [getattr(x, a) for a in req_qauntities], x
    elif len(lists) > 1:
        # should be really fast.
        order = get_best_param_order(framework, req_qauntities, **fast_kwargs)[::-1]

        ordered_kwargs = collections.OrderedDict([])
        for item in order:
            try:
                if isinstance(lists[item], (list, tuple)):
                    ordered_kwargs[item] = lists.pop(item)
            except KeyError:
                pass

        # add the rest in any order (there shouldn't actually be any)
        for k in list(lists.items()):
            if isinstance(lists[k], (list, tuple)):
                ordered_kwargs[k] = lists.pop(k)

        ordered_list = [ordered_kwargs[k] for k in ordered_kwargs]
        final_list = [
            collections.OrderedDict(list(zip(list(ordered_kwargs.keys()), v)))
            for v in itertools.product(*ordered_list)
        ]

        for vals in final_list:
            x.update(**vals)
            if not get_label:
                yield [[getattr(x, q) for q in req_qauntities], x]

            else:
                yield [
                    [getattr(x, q) for q in req_qauntities],
                    x,
                    _make_label(vals, kind=label_kind, **label_kwargs),
                ]


def _make_label(d, no_spaces=None, equals=None, delim=None, kind="display"):
    if kind == "display":
        space = " " if not no_spaces else ""
        equals = f":{space}" if equals is None else equals
        delim = f",{space}" if delim is None else delim
    elif kind == "filename":
        space = ""
        equals = "=" if equals is None else equals
        delim = "_" if delim is None else delim

    label = ""

    for key, val in d.items():
        if isinstance(val, str):
            label += f"{val}{delim}"
        elif isinstance(val, dict):
            for k, v in val.items():
                label += f"{k}{equals}{v}{delim}"
        else:
            label += f"{key}{equals}{val}{delim}"

    # Some post-formatting to make it look nicer
    label = label[: -len(delim)]

    if no_spaces:
        label = label.replace(" ", "")

    return label
