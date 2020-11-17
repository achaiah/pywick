# Source: https://forums.fast.ai/t/implementing-new-activation-functions-in-fastai-library/17697

import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    """
    Swish activation function, a special case of ARiA,
    for ARiA = f(x, 1, 0, 1, 1, b, 1)
    """

    def __init__(self, b = 1.):
        super(Swish, self).__init__()
        self.b = b

    def forward(self, x):
        sigmoid = F.sigmoid(x) ** self.b
        return x * sigmoid


class Aria(nn.Module):
    """
    Aria activation function described in `this paper <https://arxiv.org/abs/1805.08878/>`_.
    """

    def __init__(self, A=0, K=1., B = 1., v=1., C=1., Q=1.):
        super(Aria, self).__init__()
        # ARiA parameters
        self.A = A # lower asymptote, values tested were A = -1, 0, 1
        self.k = K # upper asymptote, values tested were K = 1, 2
        self.B = B # exponential rate, values tested were B = [0.5, 2]
        self.v = v # v > 0 the direction of growth, values tested were v = [0.5, 2]
        self.C = C # constant set to 1
        self.Q = Q # related to initial value, values tested were Q = [0.5, 2]

    def forward(self, x):
        aria = self.A + (self.k - self.A) / ((self.C + self.Q * F.exp(-x) ** self.B) ** (1/self.v))
        return x * aria


class Aria2(nn.Module):
    """
    ARiA2 activation function, a special case of ARiA, for ARiA = f(x, 1, 0, 1, 1, b, 1/a)
    """

    def __init__(self, a=1.5, b = 2.):
        super(Aria2, self).__init__()
        self.alpha = a
        self.beta = b

    def forward(self, x):
        return x * torch.sigmoid(self.beta*x) ** self.alpha


# Source: https://github.com/rwightman/gen-efficientnet-pytorch/blob/master/geffnet/activations/activations.py (Apache 2.0)
def hard_swish(x, inplace: bool = False):
    inner = F.relu6(x + 3.).div_(6.)
    return x.mul_(inner) if inplace else x.mul(inner)


class HardSwish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_swish(x, self.inplace)
