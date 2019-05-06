# Source: https://github.com/saeedizadi/binseg_pytoch (Apache-2.0)

"""
Implementation of `FusionNet: A deep fully residual convolutional neural network for image segmentation in connectomics <https://arxiv.org/abs/1612.05360>`_
"""

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

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

class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()

        self.layer = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True),
                                   # nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                                   # nn.ReLU(inplace=True),
                                   nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))

    def forward(self,x):
        conv = self.layer(x)
        # The last relu must be applied after the sumation
        return F.relu(x.expand_as(conv)+ conv)

class ConvResConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvResConv, self).__init__()

        # Note that the block do not return ReLU version of the output. Reason: ReLU should take place after summation
        self.layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                                   nn.ReLU(inplace=True),
                                   ResBlock(out_channels),
                                   nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

    def forward(self,x):
        return self.layer(x)


class DeconvBN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeconvBN, self).__init__()
        self.layer = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))

    def forward(self,x):
        return self.layer(x)

class FusionNet(nn.Module):
    def __init__(self, num_classes):
        super(FusionNet, self).__init__()

        #Assumin input of size 240x320
        self.enc1 = ConvResConv(3, 64)
        self.enc2 = ConvResConv(64, 128)
        self.enc3 = ConvResConv(128, 256)
        self.enc4 = ConvResConv(256, 512)

        self.middle = ConvResConv(512, 1024)

        self.dec1 = ConvResConv(512, 512)
        self.dec2 = ConvResConv(256, 256)
        self.dec3 = ConvResConv(128, 128)
        self.dec4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1))
                                  # nn.Conv2d(64, num_classes, kernel_size=1, stride=1))
        # self.dec4 = ConvResConv(64, 64)

        self.deconvbn1024_512 = DeconvBN(1024,512)
        self.deconvbn512_256 = DeconvBN(512, 256)
        self.deconvbn256_128 = DeconvBN(256, 128)
        self.deconvbn128_64 = DeconvBN(128, 64)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1, stride=1)
        self.activation = nn.Sigmoid()
        initialize_weights(self)


    def forward(self,x):

        enc1 = self.enc1(x) ## 240x320x64 --> No Relu
        enc2 = self.enc2(self._do_downsample(F.relu(enc1))) ## 120x160x128 --> No relu
        enc3 = self.enc3(self._do_downsample(F.relu(enc2))) ## 60x80x256 -->
        enc4 = self.enc4(self._do_downsample(F.relu(enc3))) ## 30x40x512 --> conv4
        middle = self.deconvbn1024_512(self.middle(self._do_downsample(F.relu(enc4)))) ## 30x40x512 --> no relu


        dec1 = self.deconvbn512_256(self.dec1(F.relu(middle+enc4))) ## 60x80x256
        dec2 = self.deconvbn256_128(self.dec2(F.relu(dec1 + enc3)))  ## 120x160x128
        dec3 = self.deconvbn128_64(self.dec3(F.relu(dec2 + enc2)))  ## 240x320x64
        dec4 = self.dec4(F.relu(dec3 + enc1))  ## 240x320x64

        output = self.final(dec4)
        return self.activation(output)

    def _do_downsample(self, x, kernel_size=2, stride=2):
        return F.max_pool2d(x, kernel_size=kernel_size, stride=stride)