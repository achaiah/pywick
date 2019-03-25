# Source: https://github.com/eladhoffer/bigBatch/tree/master/models
# Pretrained: No

import torch.nn as nn
import torchvision.transforms as transforms
import math

__all__ = ['wide_WResNet']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
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

        out += residual
        out = self.relu(out)

        return out


class WResNet(nn.Module):

    def __init__(self):
        super(WResNet, self).__init__()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feats(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)

        return x


class WResNet_imagenet(WResNet):

    def __init__(self, num_classes=1000, block=Bottleneck, layers=[3, 4, 23, 3]):
        super(WResNet_imagenet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,  bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.feats = nn.Sequential(self.conv1,
                                   self.bn1,
                                   self.relu,
                                   self.maxpool,

                                   self.layer1,
                                   self.layer2,
                                   self.layer3,
                                   self.layer4,

                                   self.avgpool)
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

        init_model(self)
        self.regime = {
            0: {'optimizer': 'SGD', 'lr': 1e-1, 'weight_decay': 1e-4, 'momentum': 0.9},
            30: {'lr': 1e-2},
            60: {'lr': 1e-3},
            90: {'lr': 1e-4}
        }


class WResNet_cifar10(WResNet):

    def __init__(self, num_classes=10, multiplier=1,
                 block=BasicBlock, depth=18):
        super(WResNet_cifar10, self).__init__()
        self.inplanes = 16 * multiplier
        n = int((depth - 2) / 6)
        self.conv1 = nn.Conv2d(3, 16 * multiplier, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16 * multiplier)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = lambda x: x
        self.layer1 = self._make_layer(block, 16 * multiplier, n)
        self.layer2 = self._make_layer(block, 32 * multiplier, n, stride=2)
        self.layer3 = self._make_layer(block, 64 * multiplier, n, stride=2)
        self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(8)
        self.last_linear = nn.Linear(64 * multiplier, num_classes)
        self.feats = nn.Sequential(self.conv1,
                                   self.bn1,
                                   self.relu,
                                   self.layer1,
                                   self.layer2,
                                   self.layer3,
                                   self.avgpool)
        init_model(self)

        self.regime = {
            0: {'optimizer': 'SGD', 'lr':  1e-1,
                'weight_decay': 1e-4, 'momentum': 0.9},
            60: {'lr': 2e-2},
            120: {'lr':  4e-3},
            140: {'lr':  1e-4}
        }


def wide_WResNet(**kwargs):
    num_classes, depth, dataset = map(
        kwargs.get, ['num_classes', 'depth', 'dataset'])
    if dataset == 'imagenet':
        num_classes = num_classes or 1000
        depth = depth or 50
        if depth == 18:
            return WResNet_imagenet(num_classes=num_classes, block=BasicBlock, layers=[2, 2, 2, 2])
        if depth == 34:
            return WResNet_imagenet(num_classes=num_classes, block=BasicBlock, layers=[3, 4, 6, 3])
        if depth == 50:
            return WResNet_imagenet(num_classes=num_classes, block=Bottleneck, layers=[3, 4, 6, 3])
        if depth == 101:
            return WResNet_imagenet(num_classes=num_classes, block=Bottleneck, layers=[3, 4, 23, 3])
        if depth == 152:
            return WResNet_imagenet(num_classes=num_classes, block=Bottleneck, layers=[3, 8, 36, 3])

    elif dataset == 'cifar10':
        num_classes = num_classes or 10
        depth = depth or 16
        return WResNet_cifar10(num_classes=num_classes,
                               block=BasicBlock, depth=depth,  multiplier=4)
    elif dataset == 'cifar100':
        num_classes = num_classes or 100
        depth = depth or 16
        return WResNet_cifar10(num_classes=num_classes,
                               block=BasicBlock, depth=depth, multiplier=4)