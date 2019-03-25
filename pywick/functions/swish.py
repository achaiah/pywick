# Source: https://github.com/tzing/resnet-swish/blob/master/swish.py

import torch.nn as nn
import torch


class Swish(nn.Module):

    def __init__(self, inplace=False):
        super().__init__()

        self.inplace = True

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)
