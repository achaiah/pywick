# Source: https://github.com/flixpar/VisDa/tree/master/models

"""
Implementation of `Large Kernel Matters <https://arxiv.org/pdf/1703.02719>`_ with PSP backend
"""

from math import floor

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


class _GlobalConvModule(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(_GlobalConvModule, self).__init__()

        pad0 = floor((kernel_size[0] - 1) / 2)
        pad1 = floor((kernel_size[1] - 1) / 2)

        self.conv_l1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size[0], 1), padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1, kernel_size[1]), padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1, kernel_size[1]), padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size[0], 1), padding=(pad0, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r
        return x


class _BoundaryRefineModule(nn.Module):
    def __init__(self, dim):
        super(_BoundaryRefineModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        out = x + residual
        return out


class _PyramidSpatialPoolingModule(nn.Module):
    def __init__(self, in_channels, down_channels, out_size, levels=(1, 2, 3, 6)):
        super(_PyramidSpatialPoolingModule, self).__init__()

        self.out_channels = len(levels) * down_channels

        self.layers = nn.ModuleList()
        for level in levels:
            layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(level),
                nn.Conv2d(in_channels, down_channels, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(down_channels),
                nn.ReLU(inplace=True),
                nn.Upsample(size=out_size, mode='bilinear')
            )
            self.layers.append(layer)

    def forward(self, x):
        features = [layer(x) for layer in self.layers]
        out = torch.cat(features, 1)

        return out


class GCN_PSP(nn.Module):
    def __init__(self, num_classes, input_size, k=7, pretrained=True):
        super(GCN_PSP, self).__init__()

        self.K = k
        self.input_size = input_size

        resnet = models.resnet152(pretrained=pretrained)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.gcm1 = _GlobalConvModule(2048, num_classes, (self.K, self.K))
        self.gcm2 = _GlobalConvModule(1024, num_classes, (self.K, self.K))
        self.gcm3 = _GlobalConvModule(512, num_classes, (self.K, self.K))
        self.gcm4 = _GlobalConvModule(256, num_classes, (self.K, self.K))

        self.brm1 = _BoundaryRefineModule(num_classes)
        self.brm2 = _BoundaryRefineModule(num_classes)
        self.brm3 = _BoundaryRefineModule(num_classes)
        self.brm4 = _BoundaryRefineModule(num_classes)
        self.brm5 = _BoundaryRefineModule(num_classes)
        self.brm6 = _BoundaryRefineModule(num_classes)
        self.brm7 = _BoundaryRefineModule(num_classes)
        self.brm8 = _BoundaryRefineModule(num_classes)
        self.brm9 = _BoundaryRefineModule(num_classes)

        self.psp = _PyramidSpatialPoolingModule(num_classes, 10, input_size, levels=(1, 2, 3, 6, 8))
        self.final = nn.Sequential(
            nn.Conv2d(num_classes + self.psp.out_channels, num_classes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0)
        )

        initialize_weights(self.gcm1, self.gcm2, self.gcm3, self.gcm4, self.brm1, self.brm2, self.brm3,
                           self.brm4, self.brm5, self.brm6, self.brm7, self.brm8, self.brm9,
                           self.psp, self.final)

    def forward(self, x):
        fm0 = self.layer0(x)
        fm1 = self.layer1(fm0)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        gcfm1 = self.brm1(self.gcm1(fm4))
        gcfm2 = self.brm2(self.gcm2(fm3))
        gcfm3 = self.brm3(self.gcm3(fm2))
        gcfm4 = self.brm4(self.gcm4(fm1))

        fs1 = self.brm5(F.interpolate(gcfm1, fm3.size()[2:], mode='bilinear') + gcfm2)
        fs2 = self.brm6(F.interpolate(fs1, fm2.size()[2:], mode='bilinear') + gcfm3)
        fs3 = self.brm7(F.interpolate(fs2, fm1.size()[2:], mode='bilinear') + gcfm4)
        fs4 = self.brm8(F.interpolate(fs3, fm0.size()[2:], mode='bilinear'))
        fs5 = self.brm9(F.interpolate(fs4, self.input_size, mode='bilinear'))

        ppm = torch.cat([self.psp(fs5), fs5], 1)
        out = self.final(ppm)

        return out


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
