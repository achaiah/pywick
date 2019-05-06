# Source: https://github.com/flixpar/VisDa/tree/master/models


"""
Implementation of `Large Kernel Matters <https://arxiv.org/pdf/1703.02719>`_ with Resnext backend
"""

from .. import resnext101_64x4d
from math import floor

import numpy as np
import torch
import torch.nn as nn


################## GCN Modules #####################

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


class _DeconvModule(nn.Module):
    def __init__(self, channels):
        super(_DeconvModule, self).__init__()
        self.deconv = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)
        self.deconv.weight.data = self.make_bilinear_weights(4, channels)
        self.deconv.bias.data.zero_()

    def forward(self, x):
        out = self.deconv(x)
        return out

    def make_bilinear_weights(self, size, num_channels):
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        filt = torch.from_numpy(filt)
        w = torch.zeros(num_channels, num_channels, size, size)
        for i in range(num_channels):
            w[i, i] = filt
        return w


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

########################### ResNeXt ###########################

class LambdaBase(nn.Sequential):
	def __init__(self, fn, *args):
		super(LambdaBase, self).__init__(*args)
		self.lambda_func = fn

	def forward_prepare(self, input):
		output = []
		for module in self._modules.values():
			output.append(module(input))
		return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class ResNeXt(nn.Module):

    def __init__(self, pretrained=True):
        super(ResNeXt, self).__init__()

        if pretrained:
            self.resnext = resnext101_64x4d()
        else:
            self.resnext = resnext101_64x4d(pretrained=None)

        self.layer0 = nn.Sequential(
            self.resnext.features[0],
            self.resnext.features[1],
            self.resnext.features[2],
            self.resnext.features[3]
        )

        self.layer1 = self.resnext.features[4]
        self.layer2 = self.resnext.features[5]
        self.layer3 = self.resnext.features[6]
        self.layer4 = self.resnext.features[7]

        self.layer5 = nn.Sequential(
            nn.AvgPool2d((7, 7), (1, 1)),
            Lambda(lambda x: x.view(x.size(0), -1)),  # View,
            nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x), nn.Linear(2048, 1000))
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x


############################## GCN #################################

class GCN_RESNEXT(nn.Module):

    def __init__(self, num_classes, input_size, k=7, pretrained=True):
        super(GCN_RESNEXT, self).__init__()

        self.num_classes = num_classes
        self.K = k
        num_imd_feats = 40

        self.resnext = ResNeXt(pretrained)

        self.gcm1 = _GlobalConvModule(2048, num_imd_feats, (self.K, self.K))
        self.gcm2 = _GlobalConvModule(1024, num_imd_feats, (self.K, self.K))
        self.gcm3 = _GlobalConvModule(512, num_imd_feats, (self.K, self.K))
        self.gcm4 = _GlobalConvModule(256, num_imd_feats, (self.K, self.K))

        self.brm1 = _BoundaryRefineModule(num_imd_feats)
        self.brm2 = _BoundaryRefineModule(num_imd_feats)
        self.brm3 = _BoundaryRefineModule(num_imd_feats)
        self.brm4 = _BoundaryRefineModule(num_imd_feats)
        self.brm5 = _BoundaryRefineModule(num_imd_feats)
        self.brm6 = _BoundaryRefineModule(num_imd_feats)
        self.brm7 = _BoundaryRefineModule(num_imd_feats)
        self.brm8 = _BoundaryRefineModule(num_imd_feats)
        self.brm9 = _BoundaryRefineModule(num_imd_feats)

        self.deconv = _DeconvModule(num_imd_feats)

        self.psp_module = _PyramidSpatialPoolingModule(num_imd_feats, 30, input_size, levels=(1, 2, 3, 6))
        self.final = nn.Sequential(
            nn.Conv2d(num_imd_feats + self.psp_module.out_channels, num_imd_feats, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_imd_feats),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_imd_feats, num_classes, kernel_size=1, padding=0)
        )

        self.initialize_weights(self.gcm1, self.gcm2, self.gcm3, self.gcm4)
        self.initialize_weights(self.brm1, self.brm2, self.brm3, self.brm4, self.brm5, self.brm6, self.brm7, self.brm8, self.brm9)
        self.initialize_weights(self.psp_module, self.final)

    def forward(self, x):

        fm0 = self.resnext.layer0(x)
        fm1 = self.resnext.layer1(fm0)
        fm2 = self.resnext.layer2(fm1)
        fm3 = self.resnext.layer3(fm2)
        fm4 = self.resnext.layer4(fm3)

        gcfm1 = self.brm1(self.gcm1(fm4))
        gcfm2 = self.brm2(self.gcm2(fm3))
        gcfm3 = self.brm3(self.gcm3(fm2))
        gcfm4 = self.brm4(self.gcm4(fm1))

        fs1 = self.brm5(self.deconv(gcfm1) + gcfm2)
        fs2 = self.brm6(self.deconv(fs1) + gcfm3)
        fs3 = self.brm7(self.deconv(fs2) + gcfm4)
        fs4 = self.brm8(self.deconv(fs3))
        fs5 = self.brm9(self.deconv(fs4))

        p = torch.cat([self.psp_module(fs5), fs5], 1)
        out = self.final(p)

        return out

    def initialize_weights(self, *models):
        for model in models:
            for module in model.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()
