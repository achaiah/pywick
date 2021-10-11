from six import iteritems
from .table import canmergetensor as canmerge
from .table import mergetensor


def compose(transforms):
    if not isinstance(transforms, list):
        raise AssertionError
    for tr in transforms:
        if not callable(tr):
            raise AssertionError('list of functions expected')

    def composition(z):
        for tr in transforms:
            z = tr(z)
        return z
    return composition


def tablemergekeys():
    def mergekeys(tbl):
        mergetbl = {}
        if isinstance(tbl, dict):
            for idx, elem in tbl.items():
                for key, value in elem.items():
                    if key not in mergetbl:
                        mergetbl[key] = {}
                    mergetbl[key][idx] = value
        elif isinstance(tbl, list):
            for elem in tbl:
                for key, value in elem.items():
                    if key not in mergetbl:
                        mergetbl[key] = []
                    mergetbl[key].append(value)
        return mergetbl
    return mergekeys

tableapply = lambda f: lambda d: dict(
    map(lambda kv: (kv[0], f(kv[1])), iteritems(d)))


def makebatch(merge=None):
    if merge:
        makebatch = compose([tablemergekeys(), merge])
    else:
        makebatch = compose([
            tablemergekeys(),
            tableapply(lambda field: mergetensor(field)
                       if canmerge(field) else field)
        ])

    return makebatch
