# Source: https://github.com/hysts/pytorch_resnet_preact (License: MIT)

"""
`Preact_Resnet models <https://github.com/hysts/pytorch_resnet_preact>`_. Not pretrained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

__all__ = ['PreactResnet110', 'PreactResnet164_bottleneck']

def PreactResnet110(num_classes):
    model_config = OrderedDict([
            ('arch', 'resnet_preact'),
            ('block_type', 'basic'),
            ('depth', 110),
            ('base_channels', 16),
            ('remove_first_relu', True),
            ('add_last_bn', True),
            ('preact_stage', [True, True, True]),
            ('input_shape', (1, 3, 32, 32)),
            ('n_classes', num_classes)
    ])
    return Network(model_config)


def PreactResnet164_bottleneck(num_classes):
    model_config = OrderedDict([
            ('arch', 'resnet_preact'),
            ('block_type', 'bottleneck'),
            ('depth', 164),
            ('base_channels', 16),
            ('remove_first_relu', True),
            ('add_last_bn', True),
            ('preact_stage', [True, True, True]),
            ('input_shape', (1, 3, 32, 32)),
            ('n_classes', num_classes)
    ])
    return Network(model_config)


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 remove_first_relu,
                 add_last_bn,
                 preact=False):
        super(BasicBlock, self).__init__()

        self._remove_first_relu = remove_first_relu
        self._add_last_bn = add_last_bn
        self._preact = preact

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,  # downsample with first conv
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)

        if add_last_bn:
            self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # downsample
                    padding=0,
                    bias=False))

    def forward(self, x):
        if self._preact:
            x = F.relu(
                self.bn1(x), inplace=True)  # shortcut after preactivation
            y = self.conv1(x)
        else:
            # preactivation only for residual path
            y = self.bn1(x)
            if not self._remove_first_relu:
                y = F.relu(y, inplace=True)
            y = self.conv1(y)

        y = F.relu(self.bn2(y), inplace=True)
        y = self.conv2(y)

        if self._add_last_bn:
            y = self.bn3(y)

        y += self.shortcut(x)
        return y


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 remove_first_relu,
                 add_last_bn,
                 preact=False):
        super(BottleneckBlock, self).__init__()

        self._remove_first_relu = remove_first_relu
        self._add_last_bn = add_last_bn
        self._preact = preact

        bottleneck_channels = out_channels // self.expansion

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,  # downsample with 3x3 conv
            padding=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)

        if add_last_bn:
            self.bn4 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()  # identity
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # downsample
                    padding=0,
                    bias=False))

    def forward(self, x):
        if self._preact:
            x = F.relu(
                self.bn1(x), inplace=True)  # shortcut after preactivation
            y = self.conv1(x)
        else:
            # preactivation only for residual path
            y = self.bn1(x)
            if not self._remove_first_relu:
                y = F.relu(y, inplace=True)
            y = self.conv1(y)

        y = F.relu(self.bn2(y), inplace=True)
        y = self.conv2(y)
        y = F.relu(self.bn3(y), inplace=True)
        y = self.conv3(y)

        if self._add_last_bn:
            y = self.bn4(y)

        y += self.shortcut(x)
        return y


class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()

        input_shape = config['input_shape']
        n_classes = config['n_classes']

        base_channels = config['base_channels']
        self._remove_first_relu = config['remove_first_relu']
        self._add_last_bn = config['add_last_bn']
        block_type = config['block_type']
        depth = config['depth']
        preact_stage = config['preact_stage']

        if block_type not in ['basic', 'bottleneck']:
            raise AssertionError
        if block_type == 'basic':
            block = BasicBlock
            n_blocks_per_stage = (depth - 2) // 6
            if n_blocks_per_stage * 6 + 2 != depth:
                raise AssertionError
        else:
            block = BottleneckBlock
            n_blocks_per_stage = (depth - 2) // 9
            if n_blocks_per_stage * 9 + 2 != depth:
                raise AssertionError

        n_channels = [
            base_channels,
            base_channels * 2 * block.expansion,
            base_channels * 4 * block.expansion,
        ]

        self.conv = nn.Conv2d(
            input_shape[1],
            n_channels[0],
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=False)

        self.stage1 = self._make_stage(
            n_channels[0],
            n_channels[0],
            n_blocks_per_stage,
            block,
            stride=1,
            preact=preact_stage[0])
        self.stage2 = self._make_stage(
            n_channels[0],
            n_channels[1],
            n_blocks_per_stage,
            block,
            stride=2,
            preact=preact_stage[1])
        self.stage3 = self._make_stage(
            n_channels[1],
            n_channels[2],
            n_blocks_per_stage,
            block,
            stride=2,
            preact=preact_stage[2])
        self.bn = nn.BatchNorm2d(n_channels[2])

        # compute conv feature size
        with torch.no_grad():
            self.feature_size = self._forward_conv(
                torch.zeros(*input_shape)).view(-1).shape[0]

        self.fc = nn.Linear(self.feature_size, n_classes)

        # initialize weights
        self.apply(initialize_weights)

    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride,
                    preact):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = 'block{}'.format(index + 1)
            if index == 0:
                stage.add_module(
                    block_name,
                    block(
                        in_channels,
                        out_channels,
                        stride=stride,
                        remove_first_relu=self._remove_first_relu,
                        add_last_bn=self._add_last_bn,
                        preact=preact))
            else:
                stage.add_module(
                    block_name,
                    block(
                        out_channels,
                        out_channels,
                        stride=1,
                        remove_first_relu=self._remove_first_relu,
                        add_last_bn=self._add_last_bn,
                        preact=False))
        return stage

    def _forward_conv(self, x):
        x = self.conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.relu(
            self.bn(x),
            inplace=True)  # apply BN and ReLU before average pooling
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
