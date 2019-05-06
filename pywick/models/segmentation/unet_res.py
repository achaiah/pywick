"""
Implementation of `U-net: Convolutional networks for biomedical image segmentation <https://arxiv.org/pdf/1505.04597>`_
"""

# Source: https://github.com/saeedizadi/binseg_pytoch (Apache-2.0)

import torch
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


class UnetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layer = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(self.out_channels),
                                   nn.ReLU(),
                                   nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(self.out_channels),
                                   nn.ReLU())

    def forward(self, x):
        return self.layer(x)


class UnetDecoder(nn.Module):
    def __init__(self, in_channels, featrures, out_channels):
        super(UnetDecoder, self).__init__()
        self.in_channels = in_channels
        self.features = featrures
        self.out_channels = out_channels

        self.layer = nn.Sequential(nn.Conv2d(self.in_channels, self.features, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(self.features),
                                   nn.ReLU(),
                                   nn.Conv2d(self.features, self.features, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(self.features),
                                   nn.ReLU(),
                                   nn.ConvTranspose2d(self.features, self.out_channels, kernel_size=2, stride=2),
                                   nn.BatchNorm2d(self.out_channels),
                                   nn.ReLU())

    def forward(self, x):
        return self.layer(x)


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.down1 = UnetEncoder(3, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down2 = UnetEncoder(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down3 = UnetEncoder(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down4 = UnetEncoder(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.center = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(),
                                    nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU())

        self.up1 = UnetDecoder(1024, 512, 256)
        self.up2 = UnetDecoder(512, 256, 128)
        self.up3 = UnetDecoder(256, 128, 64)

        self.up4 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1),
                                 # nn.BatchNorm2d(64),
                                 # nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, padding=1))
        # nn.BatchNorm2d(64),
        # nn.ReLU())

        self.output = nn.Conv2d(64, self.num_classes, kernel_size=1, stride=1)
        self.final = nn.Sigmoid()

        # Initialize weights
        initialize_weights(self)

    def forward(self, x):
        en1 = self.down1(x)
        po1 = self.pool1(en1)
        en2 = self.down2(po1)
        po2 = self.pool2(en2)
        en3 = self.down3(po2)
        po3 = self.pool3(en3)
        en4 = self.down4(po3)
        po4 = self.pool4(en4)

        c1 = self.center(po4)

        dec1 = self.up1(torch.cat([c1, F.interpolate(en4, c1.size()[2:], mode="bilinear")], 1))
        dec2 = self.up2(torch.cat([dec1, F.interpolate(en3, dec1.size()[2:], mode="bilinear")], 1))
        dec3 = self.up3(torch.cat([dec2, F.interpolate(en2, dec2.size()[2:], mode="bilinear")], 1))
        dec4 = self.up4(torch.cat([dec3, F.interpolate(en1, dec3.size()[2:], mode="bilinear")], 1))

        out = self.output(dec4)
        return self.final(out)


# The improved version of UNet model which replaces all poolings with convolution, skip conenction goes through convolutions, and residual convlutions
class Conv2dX2_Res(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(Conv2dX2_Res, self).__init__()

        self.layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding))

    def forward(self, x):
        conv = self.layer(x)
        return F.relu(x.expand_as(conv) + conv)


class PassConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(PassConv, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layer(x)


class DeconvX2_Res(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(DeconvX2_Res, self).__init__()

        self.convx2_res = Conv2dX2_Res(in_channels, in_channels, kernel_size=3, padding=1)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            nn.ReLU(inplace=True))

    def forward(self, x):
        convx2_res = self.convx2_res(x)
        return self.upsample(convx2_res)


class UNetRes(nn.Module):
    def __init__(self, num_class):
        super(UNetRes, self).__init__()

        # Assuming Input as 240x320x3
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 64, 3, padding=1),
                                  nn.ReLU(inplace=True))
        self.pool1 = nn.Conv2d(64, 128, kernel_size=2, stride=2)  # Conv as Pool
        self.enc2 = Conv2dX2_Res(128, 128, 3, padding=1)
        self.pool2 = nn.Conv2d(128, 256, kernel_size=2, stride=2)  # Conv as Pool
        self.enc3 = Conv2dX2_Res(256, 256, 3, padding=1)
        self.pool3 = nn.Conv2d(256, 512, kernel_size=2, stride=2)  # Conv as Pool
        self.enc4 = Conv2dX2_Res(512, 512, 3, padding=1)
        self.pool4 = nn.Conv2d(512, 1024, kernel_size=2, stride=2)  # Conv as Pool

        self.middle = nn.Sequential(Conv2dX2_Res(1024, 1024, 3, padding=1),
                                    nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
                                    nn.ReLU(inplace=True))

        self.pass_enc4 = PassConv(512, 512)
        self.pass_enc3 = PassConv(256, 256)
        self.pass_enc2 = PassConv(128, 128)
        self.pass_enc1 = PassConv(64, 64)

        self.dec1 = DeconvX2_Res(512, 256, 2, stride=2)
        self.dec2 = DeconvX2_Res(256, 128, 2, stride=2)
        self.dec3 = DeconvX2_Res(128, 64, 2, stride=2)

        self.dec4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                  # nn.Conv2d(64, 64, 3, padding=1),
                                  nn.Conv2d(64, num_class, kernel_size=1, stride=1))

        self.activation = nn.Sigmoid()

        initialize_weights(self)

    def forward(self, x):
        en1 = self.enc1(x)  ##240x320x64

        en2 = self.enc2(self.pool1(en1))  ## 120x160x128
        en3 = self.enc3(self.pool2(en2))  ## 60x80x256
        en4 = self.enc4(self.pool3(en3))  ## 30x40x512

        middle = self.middle(self.pool4(en4))  ## 30x40x512

        # pass_en4 = self.pass_enc4(en4) ## 30x40x512
        # dec1 = self.dec1(pass_en4+middle) ## 60x80x256
        dec1 = self.dec1(en4 + middle)  ## 60x80x256

        # pass_enc3 = self.pass_enc3(en3) ## 60x80x256
        # dec2 = self.dec2(pass_enc3+dec1) ## 120x160x128
        dec2 = self.dec2(en3 + dec1)  ## 120x160x128

        # pass_enc2 = self.pass_enc2(en2) ## 120x160x128
        # dec3 = self.dec3(pass_enc2+dec2) ## 240x320x64
        dec3 = self.dec3(en2 + dec2)  ## 240x320x64

        # pass_enc1 = self.pass_enc1(enc1) ## 240x320x64
        # dec4 = self.dec4(pass_enc1+dec3) ## 240x320x1
        dec4 = self.dec4(en1 + dec3)  ## 240x320x1

        return self.activation(dec4)