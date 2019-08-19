# Source: https://github.com/yaq007/Autofocus-Layer (MIT)

"""
Autofocus Layer - `Autofocus Layer for Semantic Segmentation <https://arxiv.org/abs/1805.08403>`_.

Only applicable to 3D targets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AFN1', 'AFN2', 'AFN3', 'AFN4', 'AFN5', 'AFN6', 'AFN_ASPP_c', 'AFN_ASPP_s', 'get_AFN']


class ModelBuilder():
    def build_net(self, arch='AFN1', num_input=4, num_classes=5, num_branches=4, padding_list=[0, 4, 8, 12], dilation_list=[2, 6, 10, 14], **kwargs):
        # parameters in the architecture
        channels = [num_input - 1, 30, 30, 40, 40, 40, 40, 50, 50, num_classes]
        kernel_size = 3

        # Baselines
        if arch == 'Basic':
            network = Basic(channels, kernel_size)
            return network
        elif arch == 'ASPP_c':
            network = ASPP_c(dilation_list, channels, kernel_size, num_branches)
            return network
        elif arch == 'ASPP_s':
            network = ASPP_s(dilation_list, channels, kernel_size, num_branches)
            return network

        # Autofocus Neural Networks
        elif arch == 'AFN1':
            blocks = [BasicBlock, BasicBlock, Autofocus_single]
        elif arch == 'AFN2':
            blocks = [BasicBlock, BasicBlock, Autofocus]
        elif arch == 'AFN3':
            blocks = [BasicBlock, Autofocus_single, Autofocus]
        elif arch == 'AFN4':
            blocks = [BasicBlock, Autofocus, Autofocus]
        elif arch == 'AFN5':
            blocks = [Autofocus_single, Autofocus, Autofocus]
        elif arch == 'AFN6':
            blocks = [Autofocus, Autofocus, Autofocus]
        else:
            raise Exception('Architecture undefined')

        network = AFN(blocks, padding_list, dilation_list, channels, kernel_size, num_branches)
        return network


class BasicBlock(nn.Module):
    def __init__(self, inplanes1, outplanes1, outplanes2, kernel=3, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes1, outplanes1, kernel_size=kernel, dilation=2)
        self.bn1 = nn.BatchNorm3d(outplanes1)
        self.conv2 = nn.Conv3d(outplanes1, outplanes2, kernel_size=kernel, dilation=2)
        self.bn2 = nn.BatchNorm3d(outplanes2)
        self.relu = nn.ReLU(inplace=True)
        if inplanes1 == outplanes2:
            self.downsample = downsample
        else:
            self.downsample = nn.Sequential(nn.Conv3d(inplanes1, outplanes2, kernel_size=1), nn.BatchNorm3d(outplanes2))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        residual = x[:, :, 4:-4, 4:-4, 4:-4]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)
        return x


class Autofocus_single(nn.Module):
    def __init__(self, inplanes1, outplanes1, outplanes2, padding_list, dilation_list, num_branches, kernel=3):
        super(Autofocus_single, self).__init__()
        self.padding_list = padding_list
        self.dilation_list = dilation_list
        self.num_branches = num_branches
        self.conv1 = nn.Conv3d(inplanes1, outplanes1, kernel_size=kernel, dilation=2)
        self.bn1 = nn.BatchNorm3d(outplanes1)

        self.bn_list2 = nn.ModuleList()
        for i in range(len(self.padding_list)):
            self.bn_list2.append(nn.BatchNorm3d(outplanes2))

        self.conv2 = nn.Conv3d(outplanes1, outplanes2, kernel_size=kernel, dilation=self.dilation_list[0])
        self.convatt1 = nn.Conv3d(outplanes1, int(outplanes1 / 2), kernel_size=kernel)
        self.convatt2 = nn.Conv3d(int(outplanes1 / 2), self.num_branches, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        if inplanes1 == outplanes2:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(nn.Conv3d(inplanes1, outplanes2, kernel_size=1), nn.BatchNorm3d(outplanes2))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        residual = x[:, :, 4:-4, 4:-4, 4:-4]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # compute attention weights for the second layer
        feature = x.detach()
        att = self.relu(self.convatt1(feature))
        att = self.convatt2(att)
        # att = torch.sigmoid(att)
        att = F.softmax(att, dim=1)
        att = att[:, :, 1:-1, 1:-1, 1:-1]

        # linear combination of different dilation rates
        x1 = self.conv2(x)
        shape = x1.size()
        x1 = self.bn_list2[0](x1) * att[:, 0:1, :, :, :].expand(shape)

        # sharing weights in parallel convolutions
        for i in range(1, self.num_branches):
            x2 = F.conv3d(x, self.conv2.weight, padding=self.padding_list[i], dilation=self.dilation_list[i])
            x2 = self.bn_list2[i](x2)
            x1 += x2 * att[:, i:(i + 1), :, :, :].expand(shape)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x = x1 + residual
        x = self.relu(x)
        return x


class Autofocus(nn.Module):
    def __init__(self, inplanes1, outplanes1, outplanes2, padding_list, dilation_list, num_branches, kernel=3):
        super(Autofocus, self).__init__()
        self.padding_list = padding_list
        self.dilation_list = dilation_list
        self.num_branches = num_branches

        self.conv1 = nn.Conv3d(inplanes1, outplanes1, kernel_size=kernel, dilation=self.dilation_list[0])
        self.convatt11 = nn.Conv3d(inplanes1, int(inplanes1 / 2), kernel_size=kernel)
        self.convatt12 = nn.Conv3d(int(inplanes1 / 2), self.num_branches, kernel_size=1)
        self.bn_list1 = nn.ModuleList()
        for i in range(self.num_branches):
            self.bn_list1.append(nn.BatchNorm3d(outplanes1))

        self.conv2 = nn.Conv3d(outplanes1, outplanes2, kernel_size=kernel, dilation=self.dilation_list[0])
        self.convatt21 = nn.Conv3d(outplanes1, int(outplanes1 / 2), kernel_size=kernel)
        self.convatt22 = nn.Conv3d(int(outplanes1 / 2), self.num_branches, kernel_size=1)
        self.bn_list2 = nn.ModuleList()
        for i in range(self.num_branches):
            self.bn_list2.append(nn.BatchNorm3d(outplanes2))

        self.relu = nn.ReLU(inplace=True)
        if inplanes1 == outplanes2:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(nn.Conv3d(inplanes1, outplanes2, kernel_size=1), nn.BatchNorm3d(outplanes2))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        residual = x[:, :, 4:-4, 4:-4, 4:-4]
        # compute attention weights in the first autofocus convolutional layer
        feature = x.detach()
        att = self.relu(self.convatt11(feature))
        att = self.convatt12(att)
        att = F.softmax(att, dim=1)
        att = att[:, :, 1:-1, 1:-1, 1:-1]

        # linear combination of different rates
        x1 = self.conv1(x)
        shape = x1.size()
        x1 = self.bn_list1[0](x1) * att[:, 0:1, :, :, :].expand(shape)

        for i in range(1, self.num_branches):
            x2 = F.conv3d(x, self.conv1.weight, padding=self.padding_list[i], dilation=self.dilation_list[i])
            x2 = self.bn_list1[i](x2)
            x1 += x2 * att[:, i:(i + 1), :, :, :].expand(shape)

        x = self.relu(x1)

        # compute attention weights for the second autofocus layer
        feature2 = x.detach()
        att2 = self.relu(self.convatt21(feature2))
        att2 = self.convatt22(att2)
        att2 = F.softmax(att2, dim=1)
        att2 = att2[:, :, 1:-1, 1:-1, 1:-1]

        # linear combination of different rates
        x21 = self.conv2(x)
        shape = x21.size()
        x21 = self.bn_list2[0](x21) * att2[:, 0:1, :, :, :].expand(shape)

        for i in range(1, self.num_branches):
            x22 = F.conv3d(x, self.conv2.weight, padding=self.padding_list[i], dilation=self.dilation_list[i])
            x22 = self.bn_list2[i](x22)
            x21 += x22 * att2[:, i:(i + 1), :, :, :].expand(shape)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x = x21 + residual
        x = self.relu(x)
        return x


class Basic(nn.Module):
    def __init__(self, channels, kernel_size):
        super(Basic, self).__init__()

        # parameters in the architecture
        self.channels = channels
        self.kernel_size = kernel_size

        # network architecture
        self.conv1 = nn.Conv3d(self.channels[0], self.channels[1], kernel_size=self.kernel_size)
        self.bn1 = nn.BatchNorm3d(self.channels[1])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(self.channels[1], self.channels[2], kernel_size=self.kernel_size)
        self.bn2 = nn.BatchNorm3d(self.channels[2])

        self.layer3 = BasicBlock(self.channels[2], self.channels[3], self.channels[4])
        self.layer4 = BasicBlock(self.channels[4], self.channels[5], self.channels[6])
        self.layer5 = BasicBlock(self.channels[6], self.channels[7], self.channels[8])

        self.fc = nn.Conv3d(self.channels[8], self.channels[9], kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.fc(x)
        return x


class ASPP_c(nn.Module):
    def __init__(self, dilation_list, channels, kernel_size, num_branches):
        super(ASPP_c, self).__init__()

        # parameters in the architecture
        channels.insert(-1, 30)
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding_list = dilation_list
        self.dilation_list = dilation_list
        self.num_branches = num_branches

        # network architecture
        self.conv1 = nn.Conv3d(self.channels[0], self.channels[1], kernel_size=self.kernel_size)
        self.bn1 = nn.BatchNorm3d(self.channels[1])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(self.channels[1], self.channels[2], kernel_size=self.kernel_size)
        self.bn2 = nn.BatchNorm3d(self.channels[2])

        self.layer3 = BasicBlock(self.channels[2], self.channels[3], self.channels[4])
        self.layer4 = BasicBlock(self.channels[4], self.channels[5], self.channels[6])
        self.layer5 = BasicBlock(self.channels[6], self.channels[7], self.channels[8])

        self.aspp = nn.ModuleList()
        for i in range(self.num_branches):
            self.aspp.append(nn.Conv3d(self.channels[8], self.channels[9], kernel_size=self.kernel_size, padding=self.padding_list[i], dilation=self.dilation_list[i]))

        self.bn9 = nn.BatchNorm3d(4 * self.channels[9])
        self.fc9 = nn.Conv3d(4 * self.channels[9], self.channels[10], kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        aspp_out = []
        for aspp_scale in self.aspp:
            aspp_out.append(aspp_scale(x))
        aspp_out = torch.cat(aspp_out, 1)

        out = self.bn9(aspp_out)
        out = self.relu(out)
        out = self.fc9(out)
        return out


class ASPP_s(nn.Module):
    def __init__(self, dilation_list, channels, kernel_size, num_branches):
        super(ASPP_s, self).__init__()

        # parameters in the architecture
        channels.insert(-1, 120)
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding_list = dilation_list
        self.dilation_list = dilation_list
        self.num_branches = num_branches

        # network architecture
        self.conv1 = nn.Conv3d(self.channels[0], self.channels[1], kernel_size=self.kernel_size)
        self.bn1 = nn.BatchNorm3d(self.channels[1])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(self.channels[1], self.channels[2], kernel_size=self.kernel_size)
        self.bn2 = nn.BatchNorm3d(self.channels[2])

        self.layer3 = BasicBlock(self.channels[2], self.channels[3], self.channels[4])
        self.layer4 = BasicBlock(self.channels[4], self.channels[5], self.channels[6])
        self.layer5 = BasicBlock(self.channels[6], self.channels[7], self.channels[8])

        self.conv_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.last_list = nn.ModuleList()
        for i in range(self.num_branches):
            self.conv_list.append(nn.Conv3d(self.channels[8], self.channels[9], kernel_size=self.kernel_size, padding=self.padding_list[i], dilation=self.dilation_list[i]))
            self.bn_list.append(nn.BatchNorm3d(self.channels[9]))
            self.last_list.append(nn.Conv3d(self.channels[9], self.channels[10], kernel_size=1))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        out = self.conv_list[0](x)
        out = self.relu(self.bn_list[0](out))
        out = self.last_list[0](out)
        for i in range(1, self.num_branches):
            out1 = self.conv_list[i](x)
            out1 = self.relu(self.bn_list[i](out1))
            out += self.last_list[i](out1)

        return out


class AFN(nn.Module):
    def __init__(self, blocks, padding_list, dilation_list, channels, kernel_size, num_branches):
        super(AFN, self).__init__()

        # parameters in the architecture
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding_list = padding_list
        self.dilation_list = dilation_list
        self.num_branches = num_branches
        self.blocks = blocks

        # network architecture
        self.conv1 = nn.Conv3d(self.channels[0], self.channels[1], kernel_size=self.kernel_size)
        self.bn1 = nn.BatchNorm3d(self.channels[1])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(self.channels[1], self.channels[2], kernel_size=self.kernel_size)
        self.bn2 = nn.BatchNorm3d(self.channels[2])

        self.layers = nn.ModuleList()
        for i in range(len(blocks)):
            block = blocks[i]
            index = int(2 * i + 2)
            if block == BasicBlock:
                self.layers.append(block(self.channels[index], self.channels[index + 1], self.channels[index + 2]))
            else:
                self.layers.append(block(self.channels[index], self.channels[index + 1], self.channels[index + 2], self.padding_list, self.dilation_list, self.num_branches))

        self.fc = nn.Conv3d(self.channels[8], self.channels[9], kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        for layer in self.layers:
            x = layer(x)

        x = self.fc(x)
        return x


def get_AFN(num_classes=1, arch='AFN4', **kwargs):
    return ModelBuilder().build_net(arch=arch, num_classes=num_classes, **kwargs)

def AFN_ASPP_c(num_classes=1, **kwargs):
    return get_AFN(arch='ASPP_c', num_classes=num_classes, **kwargs)

def AFN_ASPP_s(num_classes=1, **kwargs):
    return get_AFN(arch='ASPP_s', num_classes=num_classes, **kwargs)

def AFN1(num_classes=1, **kwargs):
    return get_AFN(arch='AFN1', num_classes=num_classes, **kwargs)

def AFN2(num_classes=1, **kwargs):
    return get_AFN(arch='AFN2', num_classes=num_classes, **kwargs)

def AFN3(num_classes=1, **kwargs):
    return get_AFN(arch='AFN3', num_classes=num_classes, **kwargs)

def AFN4(num_classes=1, **kwargs):
    return get_AFN(arch='AFN4', num_classes=num_classes, **kwargs)

def AFN5(num_classes=1, **kwargs):
    return get_AFN(arch='AFN5', num_classes=num_classes, **kwargs)

def AFN6(num_classes=1, **kwargs):
    return get_AFN(arch='AFN6', num_classes=num_classes, **kwargs)
