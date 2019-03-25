import torch.nn as nn
import numpy as np
import torch

# ==== Laplace === #
# Source: https://raw.githubusercontent.com/atlab/attorch/master/attorch/regularizers.py
# License: MIT
def laplace():
    return np.array([[0.25, 0.5, 0.25], [0.5, -3.0, 0.5], [0.25, 0.5, 0.25]]).astype(np.float32)[None, None, ...]


def laplace3d():
    l = np.zeros((3, 3, 3))
    l[1, 1, 1] = -6.
    l[1, 1, 2] = 1.
    l[1, 1, 0] = 1.
    l[1, 0, 1] = 1.
    l[1, 2, 1] = 1.
    l[0, 1, 1] = 1.
    l[2, 1, 1] = 1.
    return l.astype(np.float32)[None, None, ...]


class Laplace(nn.Module):
    """
    Laplace filter for a stack of data.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, bias=False, padding=1)
        self.conv.weight.data.copy_(torch.from_numpy(laplace()))
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)


class Laplace3D(nn.Module):
    """
    Laplace filter for a stack of data.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 1, 3, bias=False, padding=1)
        self.conv.weight.data.copy_(torch.from_numpy(laplace3d()))
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)


class LaplaceL2(nn.Module):
    """
    Laplace regularizer for a 2D convolutional layer.
    """

    def __init__(self):
        super().__init__()
        self.laplace = Laplace()

    def forward(self, x):
        ic, oc, k1, k2 = x.size()
        return self.laplace(x.view(ic * oc, 1, k1, k2)).pow(2).mean() / 2


class LaplaceL23D(nn.Module):
    """
    Laplace regularizer for a 2D convolutional layer.
    """

    def __init__(self):
        super().__init__()
        self.laplace = Laplace3D()

    def forward(self, x):
        ic, oc, k1, k2, k3 = x.size()
        return self.laplace(x.view(ic * oc, 1, k1, k2, k3)).pow(2).mean() / 2


class LaplaceL1(nn.Module):
    """
    Laplace regularizer for a 2D convolutional layer.
    """

    def __init__(self):
        super().__init__()
        self.laplace = Laplace()

    def forward(self, x):
        ic, oc, k1, k2 = x.size()
        return self.laplace(x.view(ic * oc, 1, k1, k2)).abs().mean()