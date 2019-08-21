# Source: https://github.com/wutianyiRosun/TKCN (MIT)

###########################################################################
# Created by: Tianyi Wu 
# Email: wutianyi@ict.ac.cn
# Copyright (c) 2018
###########################################################################
from __future__ import division
import torch
import torch.nn as nn
from torch.nn.functional import upsample
from .base import BaseNet


__all__ = ['TKCNet', 'get_tkcnet', 'TKCNet_Resnet101']

class TKCNet(BaseNet):
    """Tree-structured Kronecker Convolutional Networks for Semantic Segmentation, 
      Note that:
        In our pytorch implementation of TKCN: for KConv(r_1,r_2), we use AvgPool2d(kernel_size = r_2, stride=1) 
        and Conv2d( kernel_size =3, dilation = r_1) to approximate it.
        The original codes (caffe)  will be relesed later .

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object

    """
    def __init__(self, num_classes, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(TKCNet, self).__init__(num_classes, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = TFAHead(2048, num_classes, norm_layer, r1=[10, 20, 30], r2=[7, 15, 25])

    def forward(self, x):
        #print("in tkcnet.forward(): input_size: ", x.size())  #input.size == crop_size
        imsize = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)
        x = self.head(c4)
        out = upsample(x, imsize, **self._up_kwargs)
        out1 = [out]

        if self.aux:
            return tuple(out1)
        else:
            return out1[0]
        
class TFAHead(nn.Module):
    """
       input:
        x: B x C x H x W  (C = 2048)
       output: B x nClass x H x W
    """
    def __init__(self, in_channels, out_channels, norm_layer, r1, r2):
        super(TFAHead, self).__init__()
        # TFA module
        inter_channels = in_channels // 4  # 2048-->512
        self.TFA_level_1 = self._make_level(2048, inter_channels, r1[0], r2[0], norm_layer)
        self.TFA_level_list = nn.ModuleList()
        for i in range(1, len(r1)):
            self.TFA_level_list.append( self._make_level(inter_channels, inter_channels, r1[i], r2[i], norm_layer))

        # segmentation subnetwork 
        self.conv51 = nn.Sequential(nn.Conv2d(in_channels + inter_channels*len(r1), inter_channels, 3, padding=1, bias=False),
                                        norm_layer(inter_channels),
                                        nn.ReLU())
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
    
    def _make_level(self, inChannel, outChannel, r1, r2, norm_layer):
        avg_agg = nn.AvgPool2d(r2, stride =1, padding= r2 // 2)
        conv = nn.Sequential( nn.Conv2d(inChannel, outChannel, kernel_size= 3, stride= 1, padding = r1, dilation = r1 ),
                              norm_layer(outChannel),
                              nn.ReLU())
        return nn.Sequential(avg_agg, conv)


    def forward(self, x):
        TFA_out_list = []
        TFA_out_list.append(x)
        level_1_out = self.TFA_level_1(x)
        TFA_out_list.append(level_1_out)
        for i, layer in enumerate(self.TFA_level_list):
            if i==0:
                output1 = layer(level_1_out)
                TFA_out_list.append(output1)
            else:
                output1 = layer(output1)
                TFA_out_list.append(output1)
        TFA_out= torch.cat( TFA_out_list, 1)
        
        out = self.conv51(TFA_out) # B x 4096 x H x W  --> B x 512 x H x W
        out = self.conv6(out)  # B x nClass x H x W
        return out


def get_tkcnet(num_classes=1, backbone='resnet101', pretrained=True, **kwargs):
    """TKCN model from the paper `"Tree-structured Kronocker Convolutional Network for Semantic Segmentation"
    """

    model = TKCNet(num_classes=num_classes, backbone=backbone, pretrained=pretrained, **kwargs)
    return model


def TKCNet_Resnet101(num_classes=1, **kwargs):
    return get_tkcnet(num_classes=num_classes, backbone='resnet101', **kwargs)


