# Source: https://github.com/BloodAxe/segmentation-networks-benchmark/tree/master/lib/models
# Pretrained: No
# Out of memory even on 1080TI

"""
https://arxiv.org/abs/1611.05431
official code:
https://github.com/facebookresearch/ResNeXt
"""
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

"""
NOTICE:
    BasicBlock_B is not implemented
    BasicBlock_C is recommendation
    The full architecture consist of BasicBlock_A is not implemented.
"""


class ResBottleBlock(nn.Module):

    def __init__(self, in_planes, bottleneck_width=4, stride=1, expansion=1):
        super(ResBottleBlock, self).__init__()
        self.conv0 = nn.Conv2d(in_planes, bottleneck_width, 1, stride=1, bias=False)
        self.bn0 = nn.BatchNorm2d(bottleneck_width)
        self.conv1 = nn.Conv2d(bottleneck_width, bottleneck_width, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_width)
        self.conv2 = nn.Conv2d(bottleneck_width, expansion * in_planes, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(expansion * in_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or expansion != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, in_planes * expansion, 1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn0(self.conv0(x)))
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlock_A(nn.Module):
    def __init__(self, in_planes, num_paths=32, bottleneck_width=4, expansion=1, stride=1):
        super(BasicBlock_A, self).__init__()
        self.num_paths = num_paths
        for i in range(num_paths):
            setattr(self, 'path' + str(i), self._make_path(in_planes, bottleneck_width, stride, expansion))

        # self.paths=self._make_path(in_planes,bottleneck_width,stride,expansion)
        self.conv0 = nn.Conv2d(in_planes * expansion, expansion * in_planes, 1, stride=1, bias=False)
        self.bn0 = nn.BatchNorm2d(in_planes * expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or expansion != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, in_planes * expansion, 1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.path0(x)
        for i in range(1, self.num_paths):
            if hasattr(self, 'path' + str(i)):
                out + getattr(self, 'path' + str(i))(x)
            # out+=self.paths(x)
            # getattr
        # out = torch.sum(out, dim=1)
        out = self.bn0(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def _make_path(self, in_planes, bottleneck_width, stride, expansion):
        layers = []
        layers.append(ResBottleBlock(
            in_planes, bottleneck_width, stride, expansion))
        return nn.Sequential(*layers)


class BasicBlock_C(nn.Module):
    """
    increasing cardinality is a more effective way of
    gaining accuracy than going deeper or wider
    """

    def __init__(self, in_planes, bottleneck_width=4, cardinality=32, stride=1, expansion=2):
        super(BasicBlock_C, self).__init__()
        inner_width = cardinality * bottleneck_width
        self.expansion = expansion
        self.basic = nn.Sequential(OrderedDict(
            [
                ('conv1_0', nn.Conv2d(in_planes, inner_width, 1, stride=1, bias=False)),
                ('bn1', nn.BatchNorm2d(inner_width)),
                ('act0', nn.ReLU()),
                ('conv3_0',
                 nn.Conv2d(inner_width, inner_width, 3, stride=stride, padding=1, groups=cardinality, bias=False)),
                ('bn2', nn.BatchNorm2d(inner_width)),
                ('act1', nn.ReLU()),
                ('conv1_1', nn.Conv2d(inner_width, inner_width * self.expansion, 1, stride=1, bias=False)),
                ('bn3', nn.BatchNorm2d(inner_width * self.expansion))
            ]
        ))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != inner_width * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, inner_width * self.expansion, 1, stride=stride, bias=False)
            )
        self.bn0 = nn.BatchNorm2d(self.expansion * inner_width)

    def forward(self, x):
        out = self.basic(x)
        out += self.shortcut(x)
        out = F.relu(self.bn0(out))
        return out


class ResNeXt(nn.Module):
    def __init__(self, num_blocks, cardinality, bottleneck_width, expansion=2, num_classes=10):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64
        self.expansion = expansion
        self.n_out_filters = [0, 0, 0, 0]

        self.conv0 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(self.in_planes)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1, self.n_out_filters[0] = self._make_layer(num_blocks[0], 1)
        self.layer2, self.n_out_filters[1] = self._make_layer(num_blocks[1], 2)
        self.layer3, self.n_out_filters[2] = self._make_layer(num_blocks[2], 2)
        self.layer4, self.n_out_filters[3] = self._make_layer(num_blocks[3], 2)
        self.linear = nn.Linear(self.cardinality * self.bottleneck_width, num_classes)

    def forward(self, x):
        out = F.relu(self.bn0(self.conv0(x)))
        # out = self.pool0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock_C(self.in_planes, self.bottleneck_width, self.cardinality, stride, self.expansion))
            self.in_planes = self.expansion * self.bottleneck_width * self.cardinality
        self.bottleneck_width *= 2
        return nn.Sequential(*layers), self.in_planes


def resnext26_2x64d():
    return ResNeXt(num_blocks=[2, 2, 2, 2], cardinality=2, bottleneck_width=64)


def resnext26_4x32d():
    return ResNeXt(num_blocks=[2, 2, 2, 2], cardinality=4, bottleneck_width=32)


def resnext26_8x16d():
    return ResNeXt(num_blocks=[2, 2, 2, 2], cardinality=8, bottleneck_width=16)


def resnext26_16x8d():
    return ResNeXt(num_blocks=[2, 2, 2, 2], cardinality=16, bottleneck_width=8)


def resnext26_32x4d():
    return ResNeXt(num_blocks=[2, 2, 2, 2], cardinality=32, bottleneck_width=4)


def resnext26_64x2d():
    return ResNeXt(num_blocks=[2, 2, 2, 2], cardinality=32, bottleneck_width=4)


def resnext50_2x64d():
    return ResNeXt(num_blocks=[3, 4, 6, 3], cardinality=2, bottleneck_width=64)


def resnext50_32x4d():
    return ResNeXt(num_blocks=[3, 4, 6, 3], cardinality=32, bottleneck_width=4)


class DecoderBlockLinkNet(nn.Module):
    def __init__(self, in_channels, n_filters, drop_rate=0.):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C/4, 2 * H, 2 * W
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=4,
                                          stride=2, padding=1, output_padding=0)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.drop = nn.Dropout2d(drop_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        x = self.drop(x)
        return x


class LinkNext(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super().__init__()
        assert num_channels == 3
        self.num_classes = num_classes
        resnext = resnext26_2x64d()
        # resnext = resnext26_8x16d()
        filters = resnext.n_out_filters

        self.encoder0 = nn.Sequential(resnext.conv0, resnext.bn0, nn.ReLU())

        # self.firstmaxpool = resnext.maxpool
        self.encoder1 = resnext.layer1
        self.encoder2 = resnext.layer2
        self.encoder3 = resnext.layer3
        self.encoder4 = resnext.layer4

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2], drop_rate=0.1)
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1], drop_rate=0.2)
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0], drop_rate=0.3)
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0], drop_rate=0.4)

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, x):
        # Encoder
        x = self.encoder0(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        return f5
