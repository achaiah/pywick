#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2018-03-26

# Source: https://github.com/kazuto1011/deeplab-pytorch/tree/master/libs/models

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import _ConvBatchNormReLU, _ResBlock
from .msc import MSC


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.weight, 1)


def DeepLabV3_ResNet101_MSC(n_classes, output_stride):
    if output_stride == 16:
        pyramids = [6, 12, 18]
    elif output_stride == 8:
        pyramids = [12, 24, 36]
    else:
        pass

    return MSC(
        scale=DeepLabV3(
            n_classes=n_classes,
            n_blocks=[3, 4, 23, 3],
            pyramids=pyramids,
            grids=[1, 2, 4],
            output_stride=output_stride,
        ),
        pyramids=[0.5, 0.75],
    )

class _ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling with image pool"""

    def __init__(self, in_channels, out_channels, pyramids):
        super(_ASPPModule, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module(
            "c0", _ConvBatchNormReLU(in_channels, out_channels, 1, 1, 0, 1)
        )
        for i, (dilation, padding) in enumerate(zip(pyramids, pyramids)):
            self.stages.add_module(
                "c{}".format(i + 1),
                _ConvBatchNormReLU(in_channels, out_channels, 3, 1, padding, dilation),
            )
        self.imagepool = nn.Sequential(
            OrderedDict(
                [
                    ("pool", nn.AdaptiveAvgPool2d(1)),
                    ("conv", _ConvBatchNormReLU(in_channels, out_channels, 1, 1, 0, 1)),
                ]
            )
        )

    def forward(self, x):
        h = self.imagepool(x)
        h = [F.interpolate(h, size=x.shape[2:], mode="bilinear")]
        for stage in self.stages.children():
            h += [stage(x)]
        h = torch.cat(h, dim=1)
        return h


class DeepLabV3(nn.Sequential):
    """DeepLab v3"""

    def __init__(self, n_classes, n_blocks, pyramids, grids, output_stride):
        super(DeepLabV3, self).__init__()

        if output_stride == 8:
            stride = [1, 2, 1, 1]
            dilation = [1, 1, 2, 2]
        elif output_stride == 16:
            stride = [1, 2, 2, 1]
            dilation = [1, 1, 1, 2]

        self.add_module(
            "layer1",
            nn.Sequential(
                OrderedDict(
                    [
                        ("conv1", _ConvBatchNormReLU(3, 64, 7, 2, 3, 1)),
                        ("pool", nn.MaxPool2d(3, 2, 1, ceil_mode=True)),
                    ]
                )
            ),
        )
        self.add_module(
            "layer2", _ResBlock(n_blocks[0], 64, 64, 256, stride[0], dilation[0])
        )
        self.add_module(
            "layer3", _ResBlock(n_blocks[1], 256, 128, 512, stride[1], dilation[1])
        )
        self.add_module(
            "layer4", _ResBlock(n_blocks[2], 512, 256, 1024, stride[2], dilation[2])
        )
        self.add_module(
            "layer5",
            _ResBlock(n_blocks[3], 1024, 512, 2048, stride[3], dilation[3], mg=grids),
        )
        self.add_module("aspp", _ASPPModule(2048, 256, pyramids))
        self.add_module("fc1", _ConvBatchNormReLU(256 * (len(pyramids) + 2), 256, 1, 1, 0, 1))
        self.add_module("fc2", nn.Conv2d(256, n_classes, kernel_size=1))

    def forward(self, x):
        # return super(DeepLabV3, self).forward(x)
        logits = super(DeepLabV3, self).forward(x)
        logits = F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=True)
        return logits

    def freeze_bn(self):
        for m in self.named_modules():
            if isinstance(m[1], nn.BatchNorm2d):
                m[1].eval()

