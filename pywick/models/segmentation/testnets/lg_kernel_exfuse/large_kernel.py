#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Implementation of Large Kernel Matters Paper (face++)
# Author: Xiangtai(lxtpku@pku.edu.cn)

import torch
from torch import nn


from .deeplab_resnet import ModelBuilder


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


class _GlobalConvModule(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(_GlobalConvModule, self).__init__()
        pad0 = int((kernel_size[0] - 1) / 2)
        pad1 = int((kernel_size[1] - 1) / 2)
        # kernel size had better be odd number so as to avoid alignment error
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


class GCN(nn.Module):
    def __init__(self, num_classes, kernel_size=7):
        super(GCN, self).__init__()
        self.resnet_features = ModelBuilder().build_encoder("resnet101")
        self.layer0 = nn.Sequential(self.resnet_features.conv1, self.resnet_features.bn1,
                                    self.resnet_features.relu1, self.resnet_features.conv3,
                                    self.resnet_features.bn3, self.resnet_features.relu3
                                    )
        self.layer1 = nn.Sequential(self.resnet_features.maxpool, self.resnet_features.layer1)
        self.layer2 = self.resnet_features.layer2
        self.layer3 = self.resnet_features.layer3
        self.layer4 = self.resnet_features.layer4

        self.gcm1 = _GlobalConvModule(2048, num_classes, (kernel_size, kernel_size))
        self.gcm2 = _GlobalConvModule(1024, num_classes, (kernel_size, kernel_size))
        self.gcm3 = _GlobalConvModule(512, num_classes, (kernel_size, kernel_size))
        self.gcm4 = _GlobalConvModule(256, num_classes, (kernel_size, kernel_size))

        self.brm1 = _BoundaryRefineModule(num_classes)
        self.brm2 = _BoundaryRefineModule(num_classes)
        self.brm3 = _BoundaryRefineModule(num_classes)
        self.brm4 = _BoundaryRefineModule(num_classes)
        self.brm5 = _BoundaryRefineModule(num_classes)
        self.brm6 = _BoundaryRefineModule(num_classes)
        self.brm7 = _BoundaryRefineModule(num_classes)
        self.brm8 = _BoundaryRefineModule(num_classes)
        self.brm9 = _BoundaryRefineModule(num_classes)

        self.deconv1 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv5 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):
        # suppose input = x , if x 512
        f0 = self.layer0(x)  # 256
        f1 = self.layer1(f0)  # 128
        f2 = self.layer2(f1)  # 64
        f3 = self.layer3(f2)  # 32
        f4 = self.layer4(f3)  # 16

        gcfm1 = self.brm1(self.gcm1(f4))  # 16
        gcfm2 = self.brm2(self.gcm2(f3))  # 32
        gcfm3 = self.brm3(self.gcm3(f2))  # 64
        gcfm4 = self.brm4(self.gcm4(f1))  # 128

        fs1 = self.brm5(self.deconv1(gcfm1) + gcfm2)  # 32
        fs2 = self.brm6(self.deconv2(fs1) + gcfm3)  # 64
        fs3 = self.brm7(self.deconv3(fs2) + gcfm4)  # 128
        fs4 = self.brm8(self.deconv4(fs3))  # 256
        out = self.brm9(self.deconv5(fs4))

        return out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

if __name__ == '__main__':
    model = GCN(20).cuda()
    model.freeze_bn()
    model.eval()
    image = torch.autograd.Variable(torch.randn(1, 3, 512, 512), volatile=True).cuda()
    print (model(image).size())