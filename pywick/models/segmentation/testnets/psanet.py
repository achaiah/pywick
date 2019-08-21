# Source: https://github.com/Tramac/awesome-semantic-segmentation-pytorch/blob/master/core/models/psanet.py (License: Apache 2.0)

"""
Implementation of `PSANet: Point-wise Spatial AttentionNetwork for Scene Parsing <https://hszhao.github.io/papers/eccv18_psanet.pdf>`_
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pywick.models.segmentation.da_basenets.basic import _ConvBNReLU
from pywick.models.segmentation.da_basenets.segbase import SegBaseModel
from pywick.models.segmentation.da_basenets.fcn import _FCNHead

__all__ = ['PSANet', 'get_psanet', 'PSANet_Resnet50', 'PSANet_Resnet101', 'PSANet_Resnet152']


class PSANet(SegBaseModel):
    r"""PSANet
    Parameters
    ----------
    :param num_classes : int
        Number of categories for the training dataset.
    :param backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    :param norm_layer : object
        Normalization layer used in backbone network (default: :class:`nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    :param aux : (bool, default=False)  Whether to use auxiliary loss.
    """

    def __init__(self, num_classes, pretrained=True, backbone='resnet101', aux=False, **kwargs):
        super(PSANet, self).__init__(num_classes, pretrained=pretrained, aux=aux, backbone=backbone, **kwargs)
        self.head = _PSAHead(num_classes, **kwargs)
        if aux:
            self.auxlayer = _FCNHead(1024, num_classes, **kwargs)

        self.__setattr__('exclusive', ['head', 'auxlayer'] if aux else ['head'])

    def forward(self, x):
        size = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)
        outputs = list()
        x = self.head(c4)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
            return tuple(outputs)
        else:
            return outputs[0]


class _PSAHead(nn.Module):
    def __init__(self, nclass, input_size, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_PSAHead, self).__init__()
        psa_out_channels = (input_size // 8) ** 2
        # self.psa = _PointwiseSpatialAttention(2048, 3600, norm_layer) # <-- original definition. Does not work. Why 3600?
        self.psa = _PointwiseSpatialAttention(2048, psa_out_channels, norm_layer)

        self.conv_post = _ConvBNReLU(1024, 2048, 1, norm_layer=norm_layer)
        self.project = nn.Sequential(
            _ConvBNReLU(4096, 512, 3, padding=1, norm_layer=norm_layer),
            nn.Dropout2d(0.1, False),
            nn.Conv2d(512, nclass, 1))

    def forward(self, x):
        global_feature = self.psa(x)
        out = self.conv_post(global_feature)
        out = torch.cat([x, out], dim=1)
        out = self.project(out)

        return out


class _PointwiseSpatialAttention(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_PointwiseSpatialAttention, self).__init__()
        reduced_channels = 512
        self.collect_attention = _AttentionGeneration(in_channels, reduced_channels, out_channels, norm_layer)
        self.distribute_attention = _AttentionGeneration(in_channels, reduced_channels, out_channels, norm_layer)

    def forward(self, x):
        collect_fm = self.collect_attention(x)
        distribute_fm = self.distribute_attention(x)
        psa_fm = torch.cat([collect_fm, distribute_fm], dim=1)
        return psa_fm


class _AttentionGeneration(nn.Module):
    def __init__(self, in_channels, reduced_channels, out_channels, norm_layer, **kwargs):
        super(_AttentionGeneration, self).__init__()
        self.conv_reduce = _ConvBNReLU(in_channels, reduced_channels, 1, norm_layer=norm_layer)
        self.attention = nn.Sequential(
            _ConvBNReLU(reduced_channels, reduced_channels, 1, norm_layer=norm_layer),
            nn.Conv2d(reduced_channels, out_channels, 1, bias=False))

        self.reduced_channels = reduced_channels

    def forward(self, x):
        reduce_x = self.conv_reduce(x)
        attention = self.attention(reduce_x)
        n, c, h, w = attention.size()
        attention = attention.view(n, c, -1)
        reduce_x = reduce_x.view(n, self.reduced_channels, -1)
        fm = torch.bmm(reduce_x, torch.softmax(attention, dim=1))
        fm = fm.view(n, self.reduced_channels, h, w)

        return fm


def get_psanet(num_classes=1, backbone='resnet50', pretrained=True, **kwargs):
    r"""PS Attention Network

        Parameters
        ----------
        num_classes : int
            Number of classes
        pretrained : bool, default True
            This will load pretrained backbone network, that was trained on ImageNet.
        """

    model = PSANet(num_classes=num_classes, backbone=backbone, pretrained=pretrained, **kwargs)
    return model


def PSANet_Resnet50(num_classes=1, **kwargs):
    return get_psanet(num_classes=num_classes, backbone='resnet50', **kwargs)


def PSANet_Resnet101(num_classes=1, backbone='resnet101', **kwargs):
    return get_psanet(num_classes=num_classes, backbone=backbone, **kwargs)


def PSANet_Resnet152(num_classes=1, backbone='resnet152', **kwargs):
    return get_psanet(num_classes=num_classes, backbone=backbone, **kwargs)


if __name__ == '__main__':
    model = PSANet_Resnet50()
    img = torch.randn(1, 3, 480, 480)
    output = model(img)
