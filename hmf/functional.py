import collections
from hmf import MassFunction
import itertools
#===============================================================================
# Functions for simple dynamic getting of properties
#===============================================================================
def get_best_param_order(kls, q="dndm", **kwargs):
    a = kls(**kwargs)

    if isinstance(q, basestring):
        getattr(a, q)
    else:
        for qq in q:
            getattr(a, qq)

    final_list = []
    final_num = []
    for k, v in a._Cache__recalc_par_prop.iteritems():
        num = len(v)
        for i, l in enumerate(final_num):
            if l >= num:
                break
        else:
            final_list += [k]
            final_num += [num]
            continue
        final_list.insert(i, k)
        final_num.insert(i, num)
    return final_list

def get_hmf(required_attrs, get_label=True, kls=MassFunction,
            fast_kwargs={"transfer_fit":"BBKS",
                         "lnk_min":-4,
                         "lnk_max":2,
                         "dlnk":1,
                         "Mmin":10,
                         "Mmax":12,
                         "dlog10m":0.5},
            **kwargs):
    """
    Yield :class:`hmf.MassFunction` objects for all combinations of parameters supplied.
    """
    if isinstance(required_attrs, basestring):
        required_attrs = [required_attrs]
    lists = {}
    for k, v in kwargs.items():
        if isinstance(v, (list, tuple)):
            if len(v) > 1:
                lists[k] = kwargs.pop(k)
            else:
                kwargs[k] = v[0]

    x = kls(**kwargs)
    if not lists:
        if get_label:
            yield [[getattr(x, a) for a in required_attrs], x, ""]
        else:
            yield [[getattr(x, a) for a in required_attrs], x]

    if len(lists) == 1:
        for k, v in lists.iteritems():
            for vv in v:
                x.update(**{k:vv})
                if get_label:
                    yield [getattr(x, a) for a in required_attrs], x, make_label({k:vv})
                else:
                    yield [getattr(x, a) for a in required_attrs], x
    elif len(lists) > 1:
        # should be really fast.
        order = get_best_param_order(kls, required_attrs,
                                     **fast_kwargs)

        ordered_kwargs = collections.OrderedDict([])
        for item in order:
            try:
                if isinstance(lists[item], (list, tuple)):
                    ordered_kwargs[item] = lists.pop(item)
            except KeyError:
                pass

        # # add the rest in any order (there shouldn't actually be any)
        for k in lists.items():
            if isinstance(lists[k], (list, tuple)):
                ordered_kwargs[k] = lists.pop(k)

        ordered_list = [ordered_kwargs[k] for k in ordered_kwargs]
        final_list = [dict(zip(ordered_kwargs.keys(), v)) for v in itertools.product(*ordered_list)]

        for vals in final_list:
            x.update(**vals)
            if not get_label:
                yield [[getattr(x, q) for q in required_attrs], x]

            else:
                yield [[getattr(x, q) for q in required_attrs], x, make_label(vals)]

def make_label(d):
    label = ""
    for key, val in d.iteritems():
        if isinstance(val, basestring):
            label += val + ", "
        elif isinstance(val, dict):
            for k, v in val.iteritems():
                label += "%s: %s, " % (k, v)
        else:
            label += "%s: %s, " % (key, val)

    # Some post-formatting to make it look nicer
    label = label[:-2]
    label = label.replace("__", "_")
    label = label.replace("_", ".")

    return label
