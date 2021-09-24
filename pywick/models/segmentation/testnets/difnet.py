# Source: https://github.com/sdujump/DifNet

import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

affine_par = True

__all__ = ['DifNet', 'DifNet101', 'DifNet152']

def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=False)


class Mask(nn.Module):
    def __init__(self, inplanes=21):
        super(Mask, self).__init__()
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(4)
        self.conv1 = nn.Conv2d(inplanes, 8, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv2 = nn.Conv2d(8, 4, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv2d(4, 1, kernel_size=5, stride=1, padding=2, bias=False)

    def forward(self, x):
        x = self.relu(self.bn0(x))
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.sig(self.conv3(x))
        return x


class Diffuse(nn.Module):
    def __init__(self, inplanes, outplanes=64, clamp=False):
        super(Diffuse, self).__init__()
        self.alpha = Parameter(torch.Tensor(1))
        self.beta = Parameter(torch.Tensor(1))
        self.alpha.data.fill_(0)
        self.beta.data.fill_(0)
        self.clamp = clamp
        self.softmax = nn.Softmax(2)
        self.conv = nn.Conv2d(in_channels=inplanes, out_channels=outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(outplanes)

    def forward(self, F, pred, seed):
        b, c, h, w = pred.size()

        F = self.bn(self.conv(F))
        F = nn.functional.adaptive_max_pool2d(F, (h, w))
        F = F.view(b, -1, h * w)
        W = torch.bmm(F.transpose(1, 2), F)
        P = self.softmax(W)

        if self.clamp:
            self.alpha.data = torch.clamp(self.alpha.data, 0, 1)
            self.beta.data = torch.clamp(self.beta.data, 0, 1)

        pred_vec = pred.view(b, c, -1)
        out_vec = torch.bmm(P, pred_vec.transpose(1, 2)).transpose(1, 2).contiguous()
        out = (1 / (1 + torch.exp(self.beta))) * ((1 / (1 + torch.exp(self.alpha))) * out_vec.view(b, c, h, w) + (torch.exp(self.alpha) / (1 + torch.exp(self.alpha))) * seed) + (
                    torch.exp(self.beta) / (1 + torch.exp(self.beta))) * pred
        return out, P


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)

        padding = dilation
        self.conv2 = conv3x3(planes, planes, stride=1, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
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

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)

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


class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes, inplane):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(inplane, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, isseed=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        if isseed:
            if block.__name__ == 'Bottleneck':
                self.layer5 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], num_classes, 2048)
            else:
                self.layer5 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], num_classes, 512)
        self.isseed = isseed

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    @staticmethod
    def _make_pred_layer(block, dilation_series, padding_series, num_classes, inplane):
        return block(dilation_series, padding_series, num_classes, inplane)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        if self.isseed:
            out = self.layer5(x4)
        else:
            out = (x1, x2, x3, x4)
        return out


class DifNet(nn.Module):
    def __init__(self, num_classes, layers, **kwargs):
        super(DifNet, self).__init__()
        if layers <= 34:
            self.diffuse0 = Diffuse(3)
            self.diffuse1 = Diffuse(64)
            self.diffuse2 = Diffuse(128)
            self.diffuse3 = Diffuse(256)
            self.diffuse4 = Diffuse(512)
        else:
            self.diffuse0 = Diffuse(3)
            self.diffuse1 = Diffuse(64 * 4)
            self.diffuse2 = Diffuse(128 * 4)
            self.diffuse3 = Diffuse(256 * 4)
            self.diffuse4 = Diffuse(512 * 4)

        if layers == 18:
            self.model_sed = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
            self.model_dif = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, isseed=False)
        elif layers == 34:
            self.model_sed = ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
            self.model_dif = ResNet(BasicBlock, [3, 4, 6, 3], num_classes, isseed=False)
        elif layers == 50:
            self.model_sed = ResNet(Bottleneck, [3, 4, 6, 3], num_classes)
            self.model_dif = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, isseed=False)
        elif layers == 101:
            self.model_sed = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
            self.model_dif = ResNet(Bottleneck, [3, 4, 23, 3], num_classes, isseed=False)
        elif layers == 152:
            self.model_sed = ResNet(Bottleneck, [3, 8, 36, 3], num_classes)
            self.model_dif = ResNet(Bottleneck, [3, 8, 36, 3], num_classes, isseed=False)
        elif layers == 1850:
            self.model_sed = ResNet(BasicBlock, [3, 2, 2, 2], num_classes)
            self.model_dif = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, isseed=False)
        else:
            raise Exception('unsupport layer number: {}'.format(layers))
        self.mask_layer = Mask(inplanes=num_classes)

    def get_alpha(self):
        return torch.stack((self.diffuse0.alpha.data, self.diffuse1.alpha.data, self.diffuse2.alpha.data, self.diffuse3.alpha.data, self.diffuse4.alpha.data)).t()

    def get_beta(self):
        return torch.stack((self.diffuse0.beta.data, self.diffuse1.beta.data, self.diffuse2.beta.data, self.diffuse3.beta.data, self.diffuse4.beta.data)).t()

    def forward(self, x):
        sed = self.model_sed(x)
        sed_out = sed.clone()
        mask = self.mask_layer(sed)
        sed = sed * mask

        dif = self.model_dif(x)

        pred0, P0 = self.diffuse0(x, sed, sed)
        pred1, P1 = self.diffuse1(dif[0], pred0, sed)
        pred2, P2 = self.diffuse2(dif[1], pred1, sed)
        pred3, P3 = self.diffuse3(dif[2], pred2, sed)
        pred4, P4 = self.diffuse4(dif[3], pred3, sed)

        # return mask, sed_out, (pred0,pred1,pred2,pred3,pred4), torch.stack((P0,P1,P2,P3,P4))
        return F.interpolate(pred4, size=x.shape[2:], mode="bilinear", align_corners=True)        # mask, sed_out, pred4


def DifNet152(num_classes=1, **kwargs):
    difnet = DifNet(num_classes=num_classes, layers=152, **kwargs)
    return difnet


def DifNet101(num_classes=1, **kwargs):
    difnet = DifNet(num_classes=num_classes, layers=101, **kwargs)
    return difnet
