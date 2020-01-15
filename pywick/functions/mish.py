# Source: https://github.com/rwightman/gen-efficientnet-pytorch/blob/master/geffnet/activations/activations.py (Apache 2.0)
# Note. Cuda-compiled source can be found here: https://github.com/thomasbrandon/mish-cuda (MIT)

import torch.nn as nn
import torch.nn.functional as F

def mish(x, inplace: bool = False):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    """
    return x.mul(F.softplus(x).tanh())

class Mish(nn.Module):
    """
        Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
        https://arxiv.org/abs/1908.08681v1
        implemented for PyTorch / FastAI by lessw2020
        github: https://github.com/lessw2020/mish
    """
    def __init__(self, inplace: bool = False):
        super(Mish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return mish(x, self.inplace)
