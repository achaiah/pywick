# Source: https://github.com/lxtGH/GALD-DGCNet

#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Xiangtai(lxt@pku.edu.cn)
# GA module is borrowed from CGNL paper directly
# Pytorch implementation of GALD-Net


import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm2d


__all__ = ['GALDNet', 'GALD_res101', 'GALD_res50']


class SpatialCGNL(nn.Module):
    """Spatial CGNL block with dot production kernel for image classfication.
    """
    def __init__(self, inplanes, planes, use_scale=False, groups=8):
        self.use_scale = use_scale
        self.groups = groups

        super(SpatialCGNL, self).__init__()
        # conv theta
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv phi
        self.p = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv g
        self.g = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv z
        self.z = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1, groups=self.groups, bias=False)
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=inplanes)

        if self.use_scale:
            print("=> WARN: SpatialCGNL block uses 'SCALE'", \
                   'yellow')
        if self.groups:
            print("=> WARN: SpatialCGNL block uses '{}' groups".format(self.groups), \
                   'yellow')

    def kernel(self, t, p, g, b, c, h, w):
        """The linear kernel (dot production).
        Args:
            t: output of conv theata
            p: output of conv phi
            g: output of conv g
            b: batch size
            c: channels number
            h: height of featuremaps
            w: width of featuremaps
        """
        t = t.view(b, 1, c * h * w)
        p = p.view(b, 1, c * h * w)
        g = g.view(b, c * h * w, 1)

        att = torch.bmm(p, g)

        if self.use_scale:
            att = att.div((c*h*w)**0.5)

        x = torch.bmm(att, t)
        x = x.view(b, c, h, w)

        return x

    def forward(self, x):
        residual = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)
        b, c, h, w = t.size()

        if self.groups and self.groups > 1:
            _c = c // self.groups

            ts = torch.split(t, split_size_or_sections=_c, dim=1)
            ps = torch.split(p, split_size_or_sections=_c, dim=1)
            gs = torch.split(g, split_size_or_sections=_c, dim=1)

            _t_sequences = []

            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i],
                                 b, _c, h, w)
                _t_sequences.append(_x)

            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(t, p, g,
                            b, c, h, w)

        x = self.z(x)
        x = self.gn(x) + residual

        return x


class GALDBlock(nn.Module):
    def __init__(self, inplane, plane):
        super(GALDBlock, self).__init__()
        """
            Note down the spatial into 1/16
        """
        self.down = nn.Sequential(
            nn.Conv2d(inplane, inplane,kernel_size=3, groups=inplane, stride=2),
            BatchNorm2d(inplane),
            nn.ReLU(inplace=False)
        )
        self.long_relation = SpatialCGNL(inplane, plane)
        self.local_attention = LocalAttenModule(inplane)

    def forward(self, x):
        size = x.size()[2:]
        x = self.down(x)
        x = self.long_relation(x)
        # local attention
        x = F.interpolate(x,size=size, mode="bilinear", align_corners=True)
        res = x
        x = self.local_attention(x)
        return x + res


class LocalAttenModule(nn.Module):
    def __init__(self, inplane):
        super(LocalAttenModule, self).__init__()
        self.dconv1 = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            BatchNorm2d(inplane),
            nn.ReLU(inplace=False)
        )
        self.dconv2 = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            BatchNorm2d(inplane),
            nn.ReLU(inplace=False)
        )
        self.dconv3 = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            BatchNorm2d(inplane),
            nn.ReLU(inplace=False)
        )
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        res1 = x
        res2 = x
        x = self.dconv1(x)
        x = self.dconv2(x)
        x = self.dconv3(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
        x_mask = self.sigmoid_spatial(x)

        res1 = res1 * x_mask

        return res2 + res1


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out


class GALDHead(nn.Module):
    def __init__(self, inplanes, interplanes, num_classes):
        super(GALDHead, self).__init__()
        self.conva = nn.Sequential(nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False),
                                   BatchNorm2d(interplanes),
                                   nn.ReLU(interplanes))
        self.a2block = GALDBlock(interplanes, interplanes//2)
        self.convb = nn.Sequential(nn.Conv2d(interplanes, interplanes, 3, padding=1, bias=False),
                                   BatchNorm2d(interplanes),
                                   nn.ReLU(interplanes))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inplanes + interplanes, interplanes, kernel_size=3, padding=1, dilation=1, bias=False),
            BatchNorm2d(interplanes),
            nn.ReLU(interplanes),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        output = self.conva(x)
        output = self.a2block(output)
        output = self.convb(output)
        output = self.bottleneck(torch.cat([x, output], 1))
        return output


class GALDNet(nn.Module):
    def __init__(self, block, layers, num_classes, avg=False, aux=False, **_):
        self.inplanes = 128
        super(GALDNet, self).__init__()
        self.aux = aux
        self.conv1 = nn.Sequential(
            conv3x3(3, 64, stride=2),
            BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv3x3(64, 64),
            BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv3x3(64, 128))
        self.bn1 = BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1, 2, 4))

        self.head = GALDHead(2048, 512, num_classes=num_classes)
        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=True))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x):
        size = x.size()[2:]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.training and self.aux:
            x_dsn = self.dsn(x)
        x = self.layer4(x)
        x = self.head(x)

        if not self.training or not self.aux:
            return F.interpolate(x, size, mode='bilinear', align_corners=True)
        else:
            return [F.interpolate(x_dsn, size, mode='bilinear', align_corners=True), F.interpolate(x, size, mode='bilinear', align_corners=True)]


def GALD_res101(num_classes=21, **kwargs):
    model = GALDNet(Bottleneck, [3, 4, 23, 3], num_classes, **kwargs)
    return model


def GALD_res50(num_classes=21, **kwargs):
    model = GALDNet(Bottleneck, [3, 4, 6, 3], num_classes, **kwargs)
    return model


if __name__ == '__main__':
    i = torch.Tensor(1, 3, 769, 769).cuda()
    m = GALD_res50(19).cuda()
    m.eval()
    o = m(i)
    print(o[0].size())
