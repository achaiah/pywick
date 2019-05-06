# Source: https://github.com/snakers4/mnasnet-pytorch/blob/master/src/models/linknet.py

"""
Implementation of `LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation <https://arxiv.org/pdf/1505.04597>`_
"""

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .resnext import resnext101_32x4d
from .inception_resnet import inceptionresnetv2
from .inception4 import inceptionv4
from .decoder import DecoderBlockLinkNetV2 as DecoderBlock
from .decoder import DecoderBlockLinkNetInceptionV2 as DecoderBlockInception

nonlinearity = nn.ReLU


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)  # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001,  # value found in tensorflow
                                 momentum=0.1,  # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LinkNet18(nn.Module):
    def __init__(self,
                 num_classes,
                 num_channels=3,
                 is_deconv=False,
                 decoder_kernel_size=4,
                 pretrained=True
                 ):
        super().__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet18(pretrained=pretrained)

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        if num_channels == 3:
            self.firstconv = resnet.conv1
        else:
            self.firstconv = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))

        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlock(in_channels=filters[3],
                                     n_filters=filters[2],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder3 = DecoderBlock(in_channels=filters[2],
                                     n_filters=filters[1],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder2 = DecoderBlock(in_channels=filters[1],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder1 = DecoderBlock(in_channels=filters[0],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def freeze(self):
        self.require_encoder_grad(False)

    def unfreeze(self):
        self.require_encoder_grad(True)

    def require_encoder_grad(self, requires_grad):
        blocks = [self.firstconv,
                  self.encoder1,
                  self.encoder2,
                  self.encoder3,
                  self.encoder4]

        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        return f5


class LinkNet34(nn.Module):
    def __init__(self,
                 num_classes,
                 num_channels=3,
                 is_deconv=False,
                 decoder_kernel_size=4,
                 pretrained=True
                 ):
        super().__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=pretrained)

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        if num_channels == 3:
            self.firstconv = resnet.conv1
        else:
            self.firstconv = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))

        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlock(in_channels=filters[3],
                                     n_filters=filters[2],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder3 = DecoderBlock(in_channels=filters[2],
                                     n_filters=filters[1],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder2 = DecoderBlock(in_channels=filters[1],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder1 = DecoderBlock(in_channels=filters[0],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def freeze(self):
        self.require_encoder_grad(False)

    def unfreeze(self):
        self.require_encoder_grad(True)

    def require_encoder_grad(self, requires_grad):
        blocks = [self.firstconv,
                  self.encoder1,
                  self.encoder2,
                  self.encoder3,
                  self.encoder4]

        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        return f5


class LinkNet50(nn.Module):
    def __init__(self,
                 num_classes,
                 num_channels=3,
                 is_deconv=False,
                 decoder_kernel_size=4,
                 pretrained=True
                 ):
        super().__init__()

        filters = [256, 512, 1024, 2048]
        resnet = models.resnet50(pretrained=pretrained)

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        # self.firstconv = resnet.conv1
        # assert num_channels == 3, "num channels not used now. to use changle first conv layer to support num channels other then 3"
        # try to use 8-channels as first input
        if num_channels == 3:
            self.firstconv = resnet.conv1
        else:
            self.firstconv = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))

        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlock(in_channels=filters[3],
                                     n_filters=filters[2],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder3 = DecoderBlock(in_channels=filters[2],
                                     n_filters=filters[1],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder2 = DecoderBlock(in_channels=filters[1],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder1 = DecoderBlock(in_channels=filters[0],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def freeze(self):
        self.require_encoder_grad(False)

    def unfreeze(self):
        self.require_encoder_grad(True)

    def require_encoder_grad(self, requires_grad):
        blocks = [self.firstconv,
                  self.encoder1,
                  self.encoder2,
                  self.encoder3,
                  self.encoder4]

        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

                # noinspection PyCallingNonCallable

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        return f5


class LinkNet101(nn.Module):
    def __init__(self,
                 num_classes,
                 num_channels=3,
                 is_deconv=False,
                 decoder_kernel_size=4,
                 pretrained=True
                 ):
        super().__init__()

        filters = [256, 512, 1024, 2048]
        resnet = models.resnet101(pretrained=pretrained)

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        # self.firstconv = resnet.conv1
        # assert num_channels == 3, "num channels not used now. to use changle first conv layer to support num channels other then 3"
        # try to use 8-channels as first input
        if num_channels == 3:
            self.firstconv = resnet.conv1
        else:
            self.firstconv = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))

        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlock(in_channels=filters[3],
                                     n_filters=filters[2],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder3 = DecoderBlock(in_channels=filters[2],
                                     n_filters=filters[1],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder2 = DecoderBlock(in_channels=filters[1],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder1 = DecoderBlock(in_channels=filters[0],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def freeze(self):
        self.require_encoder_grad(False)

    def unfreeze(self):
        self.require_encoder_grad(True)

    def require_encoder_grad(self, requires_grad):
        blocks = [self.firstconv,
                  self.encoder1,
                  self.encoder2,
                  self.encoder3,
                  self.encoder4]

        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

                # noinspection PyCallingNonCallable

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        return f5


class LinkNeXt(nn.Module):
    def __init__(self,
                 num_classes,
                 num_channels=3,
                 is_deconv=False,
                 decoder_kernel_size=4,
                 pretrained=True
                 ):
        super().__init__()

        filters = [256, 512, 1024, 2048]
        # only pretrained
        resnet = resnext101_32x4d(num_classes=1000, pretrained='imagenet')

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        self.stem = resnet.stem
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlock(in_channels=filters[3],
                                     n_filters=filters[2],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder3 = DecoderBlock(in_channels=filters[2],
                                     n_filters=filters[1],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder2 = DecoderBlock(in_channels=filters[1],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder1 = DecoderBlock(in_channels=filters[0],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def freeze(self):
        self.require_encoder_grad(False)

    def unfreeze(self):
        self.require_encoder_grad(True)

    def require_encoder_grad(self, requires_grad):
        blocks = [self.stem,
                  self.encoder1,
                  self.encoder2,
                  self.encoder3,
                  self.encoder4]

        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        x = self.stem(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4) + e3
        # d4 = e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        # return F.sigmoid(f5)
        return f5


class LinkNet152(nn.Module):
    def __init__(self,
                 num_classes,
                 num_channels=3,
                 is_deconv=False,
                 decoder_kernel_size=3,
                 pretrained=True
                 ):
        super().__init__()

        filters = [256, 512, 1024, 2048]
        resnet = models.resnet152(pretrained=pretrained)

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        # self.firstconv = resnet.conv1
        # assert num_channels == 3, "num channels not used now. to use changle first conv layer to support num channels other then 3"
        # try to use 8-channels as first input
        if num_channels == 3:
            self.firstconv = resnet.conv1
        else:
            self.firstconv = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))

        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlock(in_channels=filters[3],
                                     n_filters=filters[2],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder3 = DecoderBlock(in_channels=filters[2],
                                     n_filters=filters[1],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder2 = DecoderBlock(in_channels=filters[1],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder1 = DecoderBlock(in_channels=filters[0],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def freeze(self):
        self.require_encoder_grad(False)

    def unfreeze(self):
        self.require_encoder_grad(True)

    def require_encoder_grad(self, requires_grad):
        blocks = [self.firstconv,
                  self.encoder1,
                  self.encoder2,
                  self.encoder3,
                  self.encoder4]

        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

                # noinspection PyCallingNonCallable

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        return f5


class LinkCeption(nn.Module):
    def __init__(self,
                 num_classes,
                 num_channels=3,
                 is_deconv=False,
                 decoder_kernel_size=4,
                 pretrained=True
                 ):
        super().__init__()

        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)

        filters = [64, 384, 384, 1024, 1536]

        # only pre-trained
        inception = inceptionv4(num_classes=1000, pretrained='imagenet')

        if num_channels == 3:
            self.stem1 = nn.Sequential(
                inception.features[0],
                inception.features[1],
                inception.features[2],
            )
        else:
            self.stem1 = nn.Sequential(
                BasicConv2d(num_channels, 32, kernel_size=3, stride=2),
                inception.features[1],
                inception.features[2],
            )

        self.stem2 = nn.Sequential(
            inception.features[3],
            inception.features[4],
            inception.features[5],
        )

        self.block1 = nn.Sequential(
            inception.features[6],
            inception.features[7],
            inception.features[8],
            inception.features[9],
        )

        self.tr1 = inception.features[10]

        self.block2 = nn.Sequential(
            inception.features[11],
            inception.features[12],
            inception.features[13],
            inception.features[14],
            inception.features[15],
            inception.features[16],
            inception.features[17],
        )

        self.tr2 = inception.features[18]

        self.block3 = nn.Sequential(
            inception.features[19],
            inception.features[20],
            inception.features[21]
        )

        # Decoder
        self.decoder4 = DecoderBlockInception(in_channels=filters[4],
                                              out_channels=filters[3],
                                              n_filters=filters[3],
                                              last_padding=0,
                                              kernel_size=decoder_kernel_size,
                                              is_deconv=is_deconv)
        self.decoder3 = DecoderBlockInception(in_channels=filters[3],
                                              out_channels=filters[2],
                                              n_filters=filters[2],
                                              last_padding=0,
                                              kernel_size=decoder_kernel_size,
                                              is_deconv=is_deconv)
        self.decoder2 = DecoderBlockInception(in_channels=filters[2],
                                              out_channels=filters[1],
                                              n_filters=filters[1],
                                              last_padding=0,
                                              kernel_size=decoder_kernel_size,
                                              is_deconv=is_deconv)
        self.decoder1 = DecoderBlockInception(in_channels=filters[1],
                                              out_channels=filters[0],
                                              n_filters=filters[0],
                                              last_padding=0,
                                              kernel_size=decoder_kernel_size,
                                              is_deconv=is_deconv)

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 1, stride=2)
        self.finalnorm1 = nn.BatchNorm2d(32)
        self.finalrelu1 = nonlinearity(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalnorm2 = nn.BatchNorm2d(32)
        self.finalrelu2 = nonlinearity(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=0)

    def freeze(self):
        self.require_encoder_grad(False)

    def unfreeze(self):
        self.require_encoder_grad(True)

    def require_encoder_grad(self, requires_grad):
        blocks = [self.stem1,
                  self.stem2,
                  self.block1,
                  self.tr1,
                  self.block2,
                  self.tr2,
                  self.block3]

        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

                # noinspection PyCallingNonCallable

    def forward(self, x):
        final_shape = x.shape[2:]

        # Encoder
        x = self.stem1(x)
        e1 = self.stem2(x)
        e2 = self.block1(e1)
        e3 = self.tr1(e2)
        e3 = self.block2(e3)
        e4 = self.tr2(e3)
        e4 = self.block3(e4)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4)[:, :, 0:e3.size(2), 0:e3.size(3)] + e3
        d3 = self.decoder3(d4)[:, :, 0:e2.size(2), 0:e2.size(3)] + e2
        d2 = self.decoder2(d3)[:, :, 0:self.decoder2(e1).size(2), 0:self.decoder2(e1).size(3)] + self.decoder2(e1)
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f1 = self.finalnorm1(f1)
        f2 = self.finalrelu1(f1)
        f2 = self.finalnorm2(f2)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        out = F.interpolate(f5, size=final_shape, mode="bilinear")

        return out


class LinkInceptionResNet(nn.Module):
    def __init__(self,
                 num_classes,
                 num_channels=3,
                 is_deconv=False,
                 decoder_kernel_size=3,
                 pretrained=True
                 ):
        super().__init__()

        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)

        filters = [64, 192, 320, 1088, 2080]

        # only pre-trained
        ir = inceptionresnetv2(pretrained='imagenet', num_classes=1000)

        if num_channels == 3:
            self.stem1 = nn.Sequential(
                ir.conv2d_1a,
                ir.conv2d_2a,
                ir.conv2d_2b,
            )
        else:
            self.stem1 = nn.Sequential(
                BasicConv2d(num_channels, 32, kernel_size=3, stride=2),
                ir.conv2d_2a,
                ir.conv2d_2b,
            )

        self.maxpool_3a = ir.maxpool_3a

        self.stem2 = nn.Sequential(
            ir.conv2d_3b,
            ir.conv2d_4a,
        )

        self.maxpool_5a = ir.maxpool_5a
        self.mixed_5b = ir.mixed_5b

        self.mixed_6a = ir.mixed_6a
        self.mixed_7a = ir.mixed_7a
        self.skip1 = ir.repeat
        self.skip2 = ir.repeat_1
        self.skip3 = ir.repeat_2

        # Decoder
        self.decoder3 = DecoderBlockInception(in_channels=filters[4],
                                              out_channels=filters[3],
                                              n_filters=filters[3],
                                              last_padding=0,
                                              kernel_size=decoder_kernel_size,
                                              is_deconv=is_deconv)
        self.decoder2 = DecoderBlockInception(in_channels=filters[3],
                                              out_channels=filters[2],
                                              n_filters=filters[2],
                                              last_padding=0,
                                              kernel_size=decoder_kernel_size,
                                              is_deconv=is_deconv)
        self.decoder1 = DecoderBlockInception(in_channels=filters[2],
                                              out_channels=filters[1],
                                              n_filters=filters[1],
                                              last_padding=0,
                                              kernel_size=decoder_kernel_size,
                                              is_deconv=is_deconv)
        self.decoder0 = DecoderBlockInception(in_channels=filters[1],
                                              out_channels=filters[0],
                                              n_filters=filters[0],
                                              last_padding=2,
                                              kernel_size=decoder_kernel_size,
                                              is_deconv=is_deconv)

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalnorm1 = nn.BatchNorm2d(32)
        self.finalrelu1 = nonlinearity(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalnorm2 = nn.BatchNorm2d(32)
        self.finalrelu2 = nonlinearity(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def freeze(self):
        self.require_encoder_grad(False)

    def unfreeze(self):
        self.require_encoder_grad(True)

    def require_encoder_grad(self, requires_grad):
        blocks = [self.stem1,
                  self.stem2,
                  self.mixed_5b,
                  self.mixed_6a,
                  self.mixed_7a,
                  self.skip1,
                  self.skip2,
                  self.skip3]

        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

    # noinspection PyCallingNonCallable
    def forward(self, x):

        # Encoder
        x = self.stem1(x)
        x1 = self.maxpool_3a(x)
        x1 = self.stem2(x1)
        x2 = self.maxpool_3a(x1)
        x2 = self.mixed_5b(x2)

        e1 = self.skip1(x2)
        e1_resume = self.mixed_6a(e1)
        e2 = self.skip2(e1_resume)
        e2_resume = self.mixed_7a(e2)
        e3 = self.skip3(e2_resume)

        # Decoder with Skip Connections
        d3 = self.decoder3(e3)[:, :, 0:e2.size(2), 0:e2.size(3)] + e2
        d2 = self.decoder2(d3)[:, :, 0:e1.size(2), 0:e1.size(3)] + e1
        d1 = self.decoder1(d2)[:, :, 0:x1.size(2), 0:x1.size(3)] + x1
        d0 = self.decoder0(d1)

        # Final Classification
        f1 = self.finaldeconv1(d0)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        return f5


class LinkDenseNet161(nn.Module):
    def __init__(self,
                 num_classes,
                 num_channels=3,
                 is_deconv=False,
                 decoder_kernel_size=4,
                 pretrained=True
                 ):
        super().__init__()

        filters = [384, 768, 2112, 2208]
        densenet = models.densenet161(pretrained=pretrained)

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        if num_channels == 3:
            self.firstconv = densenet.features.conv0
        else:
            self.firstconv = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))

        self.stem = nn.Sequential(
            self.firstconv,
            densenet.features.norm0,
            densenet.features.relu0,
            densenet.features.pool0,
        )

        self.encoder1 = nn.Sequential(densenet.features.denseblock1)
        self.encoder2 = nn.Sequential(densenet.features.transition1,
                                      densenet.features.denseblock2)
        self.encoder3 = nn.Sequential(densenet.features.transition2,
                                      densenet.features.denseblock3)
        self.encoder4 = nn.Sequential(densenet.features.transition3,
                                      densenet.features.denseblock4)

        # Decoder
        self.decoder4 = DecoderBlock(in_channels=filters[3],
                                     n_filters=filters[2],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder3 = DecoderBlock(in_channels=filters[2],
                                     n_filters=filters[1],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder2 = DecoderBlock(in_channels=filters[1],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder1 = DecoderBlock(in_channels=filters[0],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def require_encoder_grad(self, requires_grad):
        blocks = [self.stem,
                  self.encoder1,
                  self.encoder2,
                  self.encoder3,
                  self.encoder4]

        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

    def freeze(self):
        self.require_encoder_grad(False)

    def unfreeze(self):
        self.require_encoder_grad(True)

        # noinspection PyCallingNonCallable

    def forward(self, x):
        # Encoder
        x = self.stem(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        return f5


class LinkDenseNet121(nn.Module):
    def __init__(self,
                 num_classes,
                 num_channels=3,
                 is_deconv=False,
                 decoder_kernel_size=4,
                 pretrained=True
                 ):
        super().__init__()

        filters = [256, 512, 1024, 1024]
        densenet = models.densenet121(pretrained=pretrained)

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        if num_channels == 3:
            self.firstconv = densenet.features.conv0
        else:
            self.firstconv = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))

        self.stem = nn.Sequential(
            self.firstconv,
            densenet.features.norm0,
            densenet.features.relu0,
            densenet.features.pool0,
        )

        self.encoder1 = nn.Sequential(densenet.features.denseblock1)
        self.encoder2 = nn.Sequential(densenet.features.transition1,
                                      densenet.features.denseblock2)
        self.encoder3 = nn.Sequential(densenet.features.transition2,
                                      densenet.features.denseblock3)
        self.encoder4 = nn.Sequential(densenet.features.transition3,
                                      densenet.features.denseblock4)

        # Decoder
        self.decoder4 = DecoderBlock(in_channels=filters[3],
                                     n_filters=filters[2],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder3 = DecoderBlock(in_channels=filters[2],
                                     n_filters=filters[1],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder2 = DecoderBlock(in_channels=filters[1],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder1 = DecoderBlock(in_channels=filters[0],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def require_encoder_grad(self, requires_grad):
        blocks = [self.stem,
                  self.encoder1,
                  self.encoder2,
                  self.encoder3,
                  self.encoder4]

        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

    def freeze(self):
        self.require_encoder_grad(False)

    def unfreeze(self):
        self.require_encoder_grad(True)

        # noinspection PyCallingNonCallable

    def forward(self, x):
        # Encoder
        x = self.stem(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        return f5


class CoarseLinkNet50(nn.Module):
    def __init__(self,
                 num_classes,
                 num_channels=3,
                 is_deconv=False,
                 decoder_kernel_size=4,
                 pretrained=True
                 ):
        super().__init__()

        filters = [256, 512, 1024, 2048]
        resnet = models.resnet50(pretrained=pretrained)

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        # self.firstconv = resnet.conv1
        # assert num_channels == 3, "num channels not used now. to use changle first conv layer to support num channels other then 3"
        # try to use 8-channels as first input
        if num_channels == 3:
            self.firstconv = resnet.conv1
        else:
            self.firstconv = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))

        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlock(in_channels=filters[3],
                                     n_filters=filters[2],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder3 = DecoderBlock(in_channels=filters[2],
                                     n_filters=filters[1],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder2 = DecoderBlock(in_channels=filters[1],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder1 = DecoderBlock(in_channels=filters[0],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        # Final Classifier
        self.finalconv1 = nn.Conv2d(filters[0], 32, 2, padding=1)
        self.finalrelu1 = nonlinearity(inplace=True)
        self.finalconv2 = nn.Conv2d(32, num_classes, 2, padding=1)

    def freeze(self):
        self.require_encoder_grad(False)

    def unfreeze(self):
        self.require_encoder_grad(True)

    def require_encoder_grad(self, requires_grad):
        blocks = [self.firstconv,
                  self.encoder1,
                  self.encoder2,
                  self.encoder3,
                  self.encoder4]

        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

                # noinspection PyCallingNonCallable

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finalconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)

        return f3