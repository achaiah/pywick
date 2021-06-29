from pywick.optimizers.lookahead import *
from torch.optim import SGD


class LookaheadSGD(Lookahead):

    def __init__(self, params, lr, alpha=0.5, k=6, momentum=0.9, dampening=0, weight_decay=0.0001, nesterov=False):
        """
        Combination of SGD + LookAhead

        :param params:
        :param lr:
        :param alpha:
        :param k:
        :param momentum:
        :param dampening:
        :param weight_decay:
        :param nesterov:
        """
        sgd = SGD(params, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(sgd, alpha, k)
