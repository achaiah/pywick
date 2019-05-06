# Another implementation of GCN
# Source: https://github.com/saeedizadi/binseg_pytoch/tree/master/models (Apache-2.0)

"""
Implementation of `Large Kernel Matters <https://arxiv.org/pdf/1703.02719>`_ with Resnet backend.
"""

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.models as models
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


class GlobalConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k):

        super(GlobalConvolutionBlock, self).__init__()
        self.left = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=(k[0],1), padding=(k[0]//2,0)),
                                  nn.Conv2d(out_channels, out_channels, kernel_size=(1,k[1]), padding=(0,k[1]//2)))

        self.right = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=(1,k[1]), padding=(0,k[1]//2)),
                                   nn.Conv2d(out_channels, out_channels, kernel_size=(k[0],1), padding=(k[0]//2,0)))


    def forward(self,x):
        left = self.left(x)
        right = self.right(x)
        return left + right



class BoundaryRefine(nn.Module):
    def __init__(self, in_channels):
        super(BoundaryRefine, self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(in_channels),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(in_channels))

    def forward(self,x):
        convs = self.layer(x)
        return x.expand_as(convs)+convs

class ResnetGCN(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ResnetGCN, self).__init__()

        resent = models.resnet101(pretrained=pretrained)
        self.layer0 = nn.Sequential(resent.conv1, resent.bn1, resent.relu, resent.maxpool)
        self.layer1 = resent.layer1
        self.layer2 = resent.layer2
        self.layer3 = resent.layer3
        self.layer4 = resent.layer4

        #Assuming input of size 240x320
        ks = 7
        self.gcn256 = GlobalConvolutionBlock(256, num_classes, (59,79))
        self.br256 = BoundaryRefine(num_classes)
        self.gcn512 = GlobalConvolutionBlock(512, num_classes, (29,39))
        self.br512 = BoundaryRefine(num_classes)
        self.gcn1024 = GlobalConvolutionBlock(1024, num_classes, (13,19))
        self.br1024 = BoundaryRefine(num_classes)
        self.gcn2048 = GlobalConvolutionBlock(2048, num_classes, (7,9))
        self.br2048 = BoundaryRefine(num_classes)


        self.br1 = BoundaryRefine(num_classes)
        self.br2 = BoundaryRefine(num_classes)
        self.br3 = BoundaryRefine(num_classes)
        self.br4 = BoundaryRefine(num_classes)
        self.br5 = BoundaryRefine(num_classes)

        self.activation = nn.Sigmoid()

        self.deconv1 = nn.ConvTranspose2d(1,1,2,stride=2)
        self.deconv2 = nn.ConvTranspose2d(1, 1, 2, stride=2)

        initialize_weights(self.gcn256,self.gcn512,self.gcn1024, self.gcn2048,
                           self.br5,self.br4,self.br3, self.br2, self.br1,
                           self.br256, self.br512, self.br1024, self.br2048,
                           self.deconv1, self.deconv2)

    def forward(self,x):

        # Assuming input of size 240x320

        x = self.layer0(x) ## 120x160x64

        layer1 = self.layer1(x) ## 60x80x256
        layer2 = self.layer2(layer1) ## 30x40x512
        layer3 = self.layer3(layer2) ## 15x 20x1024
        layer4 = self.layer4(layer3) ## 7x10x2048

        enc1 = self.br256(self.gcn256(layer1))
        enc2 = self.br512(self.gcn512(layer2))
        enc3 = self.br1024(self.gcn1024(layer3))
        enc4 = self.br2048(self.gcn2048(layer4)) ## 8x10x1

        dec1 = self.br1(F.interpolate(enc4, size=enc3.size()[2:], mode='bilinear')+ enc3)
        dec2 = self.br2(F.interpolate(dec1, enc2.size()[2:], mode='bilinear') + enc2)
        dec3 = self.br3(F.interpolate(dec2, enc1.size()[2:], mode='bilinear') + enc1)
        dec4 = self.br4(self.deconv1(dec3))

        score_map = self.br5(self.deconv2(dec4))

        return self.activation(score_map)



    def _do_upsample(self, num_classes=1, kernel_size=2, stride=2):
        return nn.ConvTranspose2d(num_classes, num_classes, kernel_size=kernel_size, stride=stride)
