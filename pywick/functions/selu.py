# Source: https://github.com/dannysdeng/selu/blob/master/selu.py

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 14:46:31 2017

@author: danny
"""
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F


class selu(nn.Module):
    def __init__(self):
        super(selu, self).__init__()
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946

    def forward(self, x):
        temp1 = self.scale * F.relu(x)
        temp2 = self.scale * self.alpha * (F.elu(-1 * F.relu(-1 * x)))
        return temp1 + temp2


class alpha_drop(nn.Module):
    def __init__(self, p=0.05, alpha=-1.7580993408473766, fixedPointMean=0, fixedPointVar=1):
        super(alpha_drop, self).__init__()
        keep_prob = 1 - p
        self.a = np.sqrt(
            fixedPointVar / (keep_prob * ((1 - keep_prob) * pow(alpha - fixedPointMean, 2) + fixedPointVar)))
        self.b = fixedPointMean - self.a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        self.alpha = alpha
        self.keep_prob = 1 - p
        self.drop_prob = p

    def forward(self, x):
        if self.keep_prob == 1 or not self.training:
            # print("testing mode, direct return")
            return x
        else:
            random_tensor = self.keep_prob + torch.rand(x.size())

            binary_tensor = torch.floor(random_tensor)

            if torch.cuda.is_available():
                binary_tensor = binary_tensor.cuda()

            x = x.mul(binary_tensor)
            ret = x + self.alpha * (1 - binary_tensor)
            ret.mul_(self.a).add_(self.b)
            return ret

# Selu = selu()
# dropout_selu = alpha_drop(0.05)
# x = torch.normal(torch.rand(1000, 3, 3, 224), 1)
# w = Selu(x)
# y = dropout_selu(w)

