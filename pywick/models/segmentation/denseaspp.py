# Source: https://github.com/Tramac/awesome-semantic-segmentation-pytorch/blob/master/core/models/denseaspp.py (License: Apache 2.0)

"""
Implementation of `DenseASPP for Semantic Segmentation in Street Scenes <http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.pdf>`_
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pywick.models.segmentation.da_basenets.densenet import *
from pywick.models.segmentation.da_basenets.fcn import _FCNHead

__all__ = ['DenseASPP', 'DenseASPP_121', 'DenseASPP_161', 'DenseASPP_169', 'DenseASPP_201']


class DenseASPP(nn.Module):
    def __init__(self, num_classes, pretrained=True, backbone='densenet161', aux=False, dilate_scale=8, **kwargs):
        super(DenseASPP, self).__init__()
        self.nclass = num_classes
        self.aux = aux
        self.dilate_scale = dilate_scale
        if backbone == 'densenet121':
            self.pretrained = dilated_densenet121(dilate_scale, pretrained=pretrained, **kwargs)
        elif backbone == 'densenet161':
            self.pretrained = dilated_densenet161(dilate_scale, pretrained=pretrained, **kwargs)
        elif backbone == 'densenet169':
            self.pretrained = dilated_densenet169(dilate_scale, pretrained=pretrained, **kwargs)
        elif backbone == 'densenet201':
            self.pretrained = dilated_densenet201(dilate_scale, pretrained=pretrained, **kwargs)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        in_channels = self.pretrained.num_features

        self.head = _DenseASPPHead(in_channels, num_classes, **kwargs)

        if aux:
            self.auxlayer = _FCNHead(in_channels, num_classes, **kwargs)

        self.__setattr__('exclusive', ['head', 'auxlayer'] if aux else ['head'])

    def forward(self, x):
        size = x.size()[2:]
        features = self.pretrained.features(x)
        if self.dilate_scale > 8:
            features = F.interpolate(features, scale_factor=2, mode='bilinear', align_corners=True)
        outputs = []
        x = self.head(features)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)

        if self.aux and self.training:
            auxout = self.auxlayer(features)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
            return tuple(outputs)
        else:
            return outputs[0]


class _DenseASPPHead(nn.Module):
    def __init__(self, in_channels, nclass, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DenseASPPHead, self).__init__()
        self.dense_aspp_block = _DenseASPPBlock(in_channels, 256, 64, norm_layer, norm_kwargs)
        self.block = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(in_channels + 5 * 64, nclass, 1)
        )

    def forward(self, x):
        x = self.dense_aspp_block(x)
        return self.block(x)


class _DenseASPPConv(nn.Sequential):
    def __init__(self, in_channels, inter_channels, out_channels, atrous_rate,
                 drop_rate=0.1, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_DenseASPPConv, self).__init__()
        self.add_module('conv1', nn.Conv2d(in_channels, inter_channels, 1))
        self.add_module('bn1', norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)))
        self.add_module('relu1', nn.ReLU(True))
        self.add_module('conv2', nn.Conv2d(inter_channels, out_channels, 3, dilation=atrous_rate, padding=atrous_rate))
        self.add_module('bn2', norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)))
        self.add_module('relu2', nn.ReLU(True))
        self.drop_rate = drop_rate

    def forward(self, x):
        features = super(_DenseASPPConv, self).forward(x)
        if self.drop_rate > 0:
            features = F.dropout(features, p=self.drop_rate, training=self.training)
        return features


class _DenseASPPBlock(nn.Module):
    def __init__(self, in_channels, inter_channels1, inter_channels2,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_DenseASPPBlock, self).__init__()
        self.aspp_3 = _DenseASPPConv(in_channels, inter_channels1, inter_channels2, 3, 0.1,
                                     norm_layer, norm_kwargs)
        self.aspp_6 = _DenseASPPConv(in_channels + inter_channels2 * 1, inter_channels1, inter_channels2, 6, 0.1,
                                     norm_layer, norm_kwargs)
        self.aspp_12 = _DenseASPPConv(in_channels + inter_channels2 * 2, inter_channels1, inter_channels2, 12, 0.1,
                                      norm_layer, norm_kwargs)
        self.aspp_18 = _DenseASPPConv(in_channels + inter_channels2 * 3, inter_channels1, inter_channels2, 18, 0.1,
                                      norm_layer, norm_kwargs)
        self.aspp_24 = _DenseASPPConv(in_channels + inter_channels2 * 4, inter_channels1, inter_channels2, 24, 0.1,
                                      norm_layer, norm_kwargs)

    def forward(self, x):
        aspp3 = self.aspp_3(x)
        x = torch.cat([aspp3, x], dim=1)

        aspp6 = self.aspp_6(x)
        x = torch.cat([aspp6, x], dim=1)

        aspp12 = self.aspp_12(x)
        x = torch.cat([aspp12, x], dim=1)

        aspp18 = self.aspp_18(x)
        x = torch.cat([aspp18, x], dim=1)

        aspp24 = self.aspp_24(x)
        x = torch.cat([aspp24, x], dim=1)

        return x


def get_denseaspp(num_classes=1, backbone='densenet169', pretrained=True, **kwargs):
    r"""DenseASPP

    Parameters
    ----------
    dataset : str, default citys
        The dataset that model pretrained on. (pascal_voc, ade20k)
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    pretrained_base : bool or str, default True
        This will load pretrained backbone network, that was trained on ImageNet.

   """

    return DenseASPP(num_classes=num_classes, pretrained=pretrained, backbone=backbone, **kwargs)


def DenseASPP_121(num_classes=1, **kwargs):
    return get_denseaspp(num_classes=num_classes, backbone='densenet121', **kwargs)


def DenseASPP_161(num_classes=1, **kwargs):
    return get_denseaspp(num_classes=num_classes, backbone='densenet161', **kwargs)


def DenseASPP_169(num_classes=1, **kwargs):
    return get_denseaspp(num_classes=num_classes, backbone='densenet169', **kwargs)


def DenseASPP_201(num_classes=1, **kwargs):
    return get_denseaspp(num_classes=num_classes, backbone='densenet201', **kwargs)


if __name__ == '__main__':
    img = torch.randn(2, 3, 480, 480)
    model = DenseASPP_121()
    outputs = model(img)
