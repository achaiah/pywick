from .lookahead import *
from .ralamb import *

# RAdam + LARS + LookAHead

class RangerLars(Lookahead):

    def __init__(self, params, alpha=0.5, k=6, *args, **kwargs):
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
        super().__init__(ralamb, alpha, k)
