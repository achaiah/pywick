from .lookahead import *
from .ralamb import *

# RAdam + LARS + LookAHead

def RangerLars(params, alpha=0.5, k=6, *args, **kwargs):
    """
    Combination of RAdam + LARS + LookAhead

    :param params:
    :param alpha:
    :param k:
    :param args:
    :param kwargs:
    :return:
    """
    ralamb = Ralamb(params, *args, **kwargs)
    return Lookahead(ralamb, alpha, k)
