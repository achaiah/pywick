# Source: https://github.com/Hsuxu/carvana-pytorch-uNet/blob/master/model.py

"""
Implementation of `U-net: Convolutional networks for biomedical image segmentation <https://arxiv.org/pdf/1505.04597>`_ with dilation convolution operation
"""

import torch
import torch.nn as nn




class Conv_transition(nn.Module):
    '''
    resnet block contains inception
    '''

    def __init__(self, kernel_size, in_channels, out_channels):
        super(Conv_transition, self).__init__()
        if not kernel_size:
            kernel_size = [1, 3, 5]
        paddings = [int(a / 2) for a in kernel_size]
        # self.Conv0=nn.Conv2d(in_channels,out_channels,3,stride=1,padding=1)
        self.Conv1 = nn.Conv2d(in_channels, out_channels, kernel_size[0], stride=1, padding=paddings[0])
        self.Conv2 = nn.Conv2d(in_channels, out_channels, kernel_size[1], stride=1, padding=paddings[1])
        self.Conv3 = nn.Conv2d(in_channels, out_channels, kernel_size[2], stride=1, padding=paddings[2])
        self.Conv_f = nn.Conv2d(3 * out_channels, out_channels, 3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU()

    def forward(self, x):
        # x = self.Conv0(x)
        x1 = self.act(self.Conv1(x))
        x2 = self.act(self.Conv2(x))
        x3 = self.act(self.Conv3(x))
        x = torch.cat([x1, x2, x3], dim=1)
        return self.act(self.bn(self.Conv_f(x)))


class Dense_layer(nn.Module):
    """
    an two-layer
    """

    def __init__(self, in_channels, growth_rate):
        super(Dense_layer, self).__init__()
        # self.bn0=nn.BatchNorm2d(in_channels)
        self.Conv0 = nn.Conv2d(in_channels, in_channels + growth_rate, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels + growth_rate)
        self.Conv1 = nn.Conv2d(in_channels + growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(in_channels + growth_rate)
        self.Conv2 = nn.Conv2d(in_channels + growth_rate, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(in_channels)

        # self.Conv1=nn.Conv2d(in_channels+growth_rate,growth_rate,kernel_size=3,stride=1,padding=1,bias=False)

        self.act = nn.PReLU()

    def forward(self, x):
        x1 = self.act(self.bn1(self.Conv0(x)))
        x1 = self.act(self.bn2(torch.cat([self.Conv1(x1), x], dim=1)))

        return self.act(self.bn3(self.Conv2(x1)))


class Fire_Down(nn.Module):
    def __init__(self, kernel_size, in_channels, inner_channels, out_channels):
        super(Fire_Down, self).__init__()
        dilations = [1, 3, 5]
        self.Conv1 = nn.Conv2d(in_channels, inner_channels, kernel_size=kernel_size, stride=1, padding=dilations[0],
                               dilation=dilations[0])
        self.Conv4 = nn.Conv2d(in_channels, inner_channels, kernel_size=kernel_size, stride=1, padding=dilations[1],
                               dilation=dilations[1])
        self.Conv8 = nn.Conv2d(in_channels, inner_channels, kernel_size=kernel_size, stride=1, padding=dilations[2],
                               dilation=dilations[2])
        self.Conv_f3 = nn.Conv2d(3 * inner_channels, out_channels, kernel_size=kernel_size, stride=2, padding=1)
        self.Conv_f1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU()

    def forward(self, x):
        x1 = self.act(self.Conv1(x))
        x2 = self.act(self.Conv4(x))
        x3 = self.act(self.Conv8(x))
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.act(self.Conv_f3(x))
        return self.act(self.bn1(self.Conv_f1(x)))


class Fire_Up(nn.Module):
    def __init__(self, kernel_size, in_channels, inner_channels, out_channels, out_padding=(1, 1)):
        super(Fire_Up, self).__init__()
        padds = int(kernel_size / 2)
        self.Conv1 = nn.Conv2d(in_channels, inner_channels, kernel_size=3, stride=1, padding=1)
        if not out_padding:
            out_padding = (1, 1)
        # self.ConvT1=nn.ConvTranspose2d(inner_channels,out_channels,kernel_size=1,stride=2,padding=0,output_padding=out_padding)
        self.ConvT4 = nn.ConvTranspose2d(inner_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padds,
                                         output_padding=out_padding)
        # self.ConvT8=nn.ConvTranspose2d(inner_channels,out_channels,kernel_size=5,stride=2,padding=2,output_padding=out_padding)
        self.Conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU()

    def forward(self, x):
        x = self.act(self.Conv1(x))
        # x1=self.act(self.ConvT1(x))
        x = self.act(self.ConvT4(x))
        # x8=self.act(self.ConvT8(x))
        # x=torch.cat([x1,x4],dim=1)
        x = self.act(self.bn1(self.Conv2(x)))
        return x


class uNetDilated(nn.Module):
    def __init__(self, num_classes):
        super(uNetDilated, self).__init__()
        self.Conv0 = self._transition(3, 8)  # 1918
        self.down1 = self._down_block(8, 16, 16)  # 959
        self.down2 = self._down_block(16, 16, 32)  # 480
        self.down3 = self._down_block(32, 32, 64)  # 240
        self.down4 = self._down_block(64, 64, 96)  # 120
        self.down5 = self._down_block(96, 96, 128)  # 60
        self.tran0 = self._transition(128, 256)
        self.db0 = self._dense_block(256, 32)

        self.up1 = self._up_block(256, 96, 96)  # 120
        self.db1 = self._dense_block(96, 32)
        self.conv1 = nn.Conv2d(96 * 2, 96, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(96)

        self.up2 = self._up_block(96, 64, 64)  # 240
        self.db2 = self._dense_block(64, 24)
        self.conv2 = nn.Conv2d(64 * 2, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.up3 = self._up_block(64, 32, 32)  # 480
        self.db3 = self._dense_block(32, 10)
        self.conv3 = nn.Conv2d(32 * 2, 32, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.up4 = self._up_block(32, 16, 16) #, output_padding=(1, 0))  # 959
        self.db4 = self._dense_block(16, 8)
        self.conv4 = nn.Conv2d(16 * 2, 16, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(16)

        self.up5 = self._up_block(16, 16, 16)  # 1918
        self.db5 = self._dense_block(16, 4)
        self.conv5 = nn.Conv2d(16, num_classes, 3, stride=1, padding=1)

        self.clss = nn.LogSoftmax()
        self.act = nn.PReLU()

    def forward(self, x):

        x1 = self.Conv0(x)
        down1 = self.down1(x1)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        down5 = self.tran0(down5)
        down5 = self.db0(down5)

        ## TODO Problem here:
        # self.up1(down5).data.shape    =>  torch.Size([2, 96, 64, 44])
        #                       -- MISMATCH WITH --
        # down4.data.shape              =>  torch.Size([2, 96, 64, 43])

        up1 = self.act(self.bn1(self.conv1(torch.cat([self.db1(self.up1(down5)), down4], dim=1))))
        del down5, down4

        up2 = self.act(self.bn2(self.conv2(torch.cat([self.db2(self.up2(up1)), down3], dim=1))))
        del down3

        up3 = self.act(self.bn3(self.conv3(torch.cat([self.db3(self.up3(up2)), down2], dim=1))))
        del down2

        up4 = self.act(self.bn4(self.conv4(torch.cat([self.db4(self.up4(up3)), down1], dim=1))))
        del down1

        up5 = self.up5(up4)
        # up5=self.conv5(up5)

        # return self.clss(self.conv5(up5))
        return self.conv5(up5)


    def _transition(self, in_channels, out_channels):
        layers = []
        layers.append(Conv_transition([1, 3, 5], in_channels, out_channels))
        return nn.Sequential(*layers)

    def _down_block(self, in_channels, inner_channels, out_channels):
        layers = []
        layers.append(Fire_Down(3, in_channels, inner_channels, out_channels))
        return nn.Sequential(*layers)

    def _up_block(self, in_channels, inner_channels, out_channels, output_padding=(1, 1)):
        layers = []
        layers.append(Fire_Up(3, in_channels, inner_channels, out_channels, output_padding))
        return nn.Sequential(*layers)

    def _dense_block(self, in_channels, growth_rate):
        layers = []
        layers.append(Dense_layer(in_channels, growth_rate))
        return nn.Sequential(*layers)