# Source: https://github.com/saeedizadi/binseg_pytoch/blob/master/models/pspnet.py

import torch
import torch.nn.init as init
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np

import math

def initialize_weights(method='kaiming', *models):
    for model in models:
        for module in model.modules():

            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear):
                if method == 'kaiming':
                    init.kaiming_normal_(module.weight.data, np.sqrt(2.0))
                elif method == 'xavier':
                    init.xavier_normal(module.weight.data, np.sqrt(2.0))
                elif method == 'orthogonal':
                    init.orthogonal(module.weight.data, np.sqrt(2.0))
                elif method == 'normal':
                    init.normal(module.weight.data,mean=0, std=0.02)
                if module.bias is not None:
                    init.constant(module.bias.data,0)

class PyramidPoolingModule(nn.Module):
    def __init__(self, in_size, in_channels, out_channels, setting):
        super(PyramidPoolingModule, self).__init__()

        self.features = []

        for s in setting:
            pool_size = int(math.ceil(float(in_size[0])/s)),int(math.ceil(float(in_size[1])/s))
            self.features.append(nn.Sequential(nn.AvgPool2d(kernel_size=pool_size,stride=pool_size, ceil_mode=True),
                                          nn.Conv2d(in_channels, out_channels,kernel_size=1, bias=False),
                                          nn.BatchNorm2d(out_channels),
                                          nn.ReLU(inplace=True),
                                          nn.UpsamplingBilinear2d(size=in_size)))

        self.features = nn.ModuleList(modules=self.features)

    def forward(self,x):
        out = []
        out.append(x)

        for m in self.features:
            out.append(m(x))

        out = torch.cat(out, 1)

        return out


class PSPNet(nn.Module):
    def __init__(self, num_classes):
        super(PSPNet, self).__init__()

        feats = list(models.resnet101(pretrained=True).modules())
        resent = models.resnet101(pretrained=True)

        self.layer0 = nn.Sequential(resent.conv1, resent.bn1, resent.relu, resent.maxpool)
        self.layer1 = resent.layer1
        self.layer2 = resent.layer2
        self.layer3 = resent.layer3
        self.layer4 = resent.layer4


        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation = (2,2)
                m.padding = (2,2)
                m.stride = (1,1)
            if 'downsample.0' in n:
                m.stride = (1,1)

        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation = (4,4)
                m.padding = (4,4)
                m.stride = (1,1)
            if 'downsample.0' in n:
                m.stride = (1,1)


        #NOte that the size of input image is assumed to be 240hx320w
        self.ppm = PyramidPoolingModule(in_size=(30,40), in_channels=2048, out_channels=512, setting=(1,2,3,6))

        #4x512 = 4096
        self.final = nn.Sequential(nn.Conv2d(4096, 512, kernel_size=1, stride=1, bias=False),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, num_classes, kernel_size=1))

        self.activation = nn.Sigmoid()
        initialize_weights(self.ppm, self.final)

    def forward(self,x):

        input_size = x.size()
        x = self.layer0(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.ppm(x)
        x = self.final(x)

        upsample = F.interpolate(x, input_size[2:], mode='bilinear')

        return upsample
        # return self.activation(upsample)



