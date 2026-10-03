"""Functions for generating several :class:`hmf.hmf.MassFunction` instances at once.

The underlying idea here is that typically modifying say the redshift has a smaller
number of re-computations than modifying the base cosmological parameters. Thus, in a
nested loop, the redshift should be the inner loop.

It is not always obvious which order the loops should be, so this module provides
functions to determine that order, and indeed perform the loops.
"""

import collections
import copy
import itertools
import warnings

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
    >>> from hmf import MassFunction
    >>> print(get_best_param_order(
    ...     MassFunction, "dndm", transfer_model="BBKS", dlnk=1, dlog10m=1
    ... )[::3])
    ['disable_mass_conversion', 'hmf_params', 'mdef_model', 'Mmin', 'takahashi',
     'z', 'lnk_min', 'sigma_8', 'cosmo_model']
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
    req_qauntities: str | list[str] | None = None,
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
    *,
    req_quantities: str | list[str] | None = None,
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
        application be provided here, to ensure proper optimization. The name is
        misspelled for historical reasons: pass it positionally, or use the
        keyword ``req_quantities`` instead.
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
    label_kind : {"display", "filename"}, optional
        The style of the yielded labels.
    label_kwargs : dict, optional
        Extra keyword arguments controlling the label format.
    req_quantities : str or list of str, optional
        Correctly-spelled, keyword-only alias of ``req_qauntities``. Give one or
        the other, not both.
    kwargs : unpacked-dict
        Any of the parameters to the initialiser of `framework` which should be
        calculated. These may be scalars, lists or tuples. The total number of
        calculations will be the total combination of all parameters.

    Yields
    ------
    quantities : list
        A list of quantities, specified by the `req_quantities` arguments.
    x : Framework instance
        An instance of `framework`, with the requisite quantities pre-cached.
        Each iteration yields a new, independent copy, so the yielded instances
        and quantities may safely be stored (e.g. with ``list(get_hmf(...))``);
        they are not modified by subsequent iterations.
    label : str, optional
        If `get_label` is True, also returns a string label uniquely specifying
        the current parameter combination.

    Notes
    -----
    Internally a single instance is updated in the optimal order, so that cached
    intermediate quantities are re-used. A deep copy of it is yielded at each
    iteration, which costs about as much as one cheap re-computation (e.g.
    changing only the redshift).

    Examples
    --------
    The following operation will run 12 iterations, yielding the desired quantities,
    an instance containing those and other quantities, and a unique label at every
    iteration.

    >>> for quants, mf, label in get_hmf(
    ...     ["dndm", "ngtm"], z=[0, 1, 2], hmf_model=["ST", "PS"], sigma_8=[0.7, 0.8]
    ... ):
    ...     print(label)
    sigma_8: 0.7, z: 0, ST
    sigma_8: 0.7, z: 0, PS
    sigma_8: 0.7, z: 1, ST
    sigma_8: 0.7, z: 1, PS
    sigma_8: 0.7, z: 2, ST
    sigma_8: 0.7, z: 2, PS
    sigma_8: 0.8, z: 0, ST
    sigma_8: 0.8, z: 0, PS
    sigma_8: 0.8, z: 1, ST
    sigma_8: 0.8, z: 1, PS
    sigma_8: 0.8, z: 2, ST
    sigma_8: 0.8, z: 2, PS

    To calculate all of them and keep the results as a list:

    >>> big_list = list(get_hmf("mean_density", z=list(range(8))))
    >>> print([float(round(x[0][0] / 1e10, 1)) for x in big_list])
    [8.6, 68.8, 232.0, 550.0, 1074.3, 1856.3, 2947.8, 4400.2]
    >>> print([x[1].z for x in big_list])
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    """
    label_kwargs = label_kwargs or {}

    if req_quantities is not None:
        if req_qauntities is not None:
            raise TypeError("Pass only one of 'req_qauntities' and 'req_quantities'.")
        req_qauntities = req_quantities
    elif req_qauntities is None:
        raise TypeError("get_hmf() missing required argument: 'req_quantities'.")

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

    def _output(label_vals):
        # Compute the quantities on the working instance (so its cache is re-used in
        # the next iteration), then hand out an independent copy, so that neither the
        # yielded instance nor its quantities change as the iteration proceeds.
        for q in req_qauntities:
            getattr(x, q)
        out = copy.deepcopy(x)
        result = [[getattr(out, q) for q in req_qauntities], out]
        if get_label:
            result.append(_make_label(label_vals, kind=label_kind, **label_kwargs))
        return result

    if not lists:
        yield _output({})

    elif len(lists) == 1:
        for k, v in lists.items():
            for vv in v:
                x.update(**{k: vv})
                yield _output({k: vv})
    else:
        # should be really fast. The fast_kwargs deliberately use a crude k-range, so
        # silence the k-range coverage warning for this ordering pass only.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The k-range", category=UserWarning)
            order = get_best_param_order(framework, req_qauntities, **fast_kwargs)[::-1]

        ordered_kwargs = collections.OrderedDict([])
        for item in order:
            if item in lists:
                ordered_kwargs[item] = lists.pop(item)

        # Add the rest in any order. These are not parameters of the framework, so
        # updating with them raises an informative error below.
        ordered_kwargs.update(lists)

        ordered_list = [ordered_kwargs[k] for k in ordered_kwargs]
        final_list = [
            collections.OrderedDict(list(zip(list(ordered_kwargs.keys()), v, strict=True)))
            for v in itertools.product(*ordered_list)
        ]

        for vals in final_list:
            x.update(**vals)
            yield _output(vals)


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
