"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.nn as nn
from .config import cfg


def Norm2d(in_channels):
    """
    Custom Norm Function to allow flexible switching
    """
    layer = cfg.MODEL.BNFUNC
    normalizationLayer = layer(in_channels)
    return normalizationLayer


def initialize_weights(*models):
   for model in models:
        for module in model.modules():
            if isinstance(module(nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
