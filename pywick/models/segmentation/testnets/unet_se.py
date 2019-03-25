# Source: https://github.com/areum-lee/SENet_Segmentation

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetDec(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UNetDec, self).__init__()

        self.up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.ConvTranspose2d(out_channels, out_channels, 2, stride=2),
            nn.ReLU(inplace=True),
        )

        self.SE1 = nn.Conv2d(in_channels, in_channels // 16, kernel_size=1)
        self.SE2 = nn.Conv2d(in_channels // 16, in_channels, kernel_size=1)

    def forward(self, x):
        fm_size = x.size()[2]
        scale_weight = F.avg_pool2d(x, fm_size)
        scale_weight = F.relu(self.SE1(scale_weight))
        scale_weight = F.sigmoid(self.SE2(scale_weight))
        x = x * scale_weight.expand_as(x)
        out = self.up(x)
        return out


class Dilated_UNetEnc(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False):
        super(Dilated_UNetEnc, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)

        ]
        if dropout:
            layers += [nn.Dropout(.5)]

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        out = self.down(x)

        return out


class Dilated_Bottleneck_block(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate, dropout=False):
        super(Dilated_Bottleneck_block, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation_rate, dilation=dilation_rate, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)]

        if dropout:
            layers += [nn.Dropout(.5)]

        self.center = nn.Sequential(*layers)

    def forward(self, x):
        out = self.center(x)

        return out


# Example: model = Dilated_UNet(inChannel=2, num_classes=2, init_features=32, network_depth=3, bottleneck_layers=3))
class Dilated_UNet(nn.Module):

    def __init__(self, inChannel, num_classes, init_features, network_depth, bottleneck_layers):
        super(Dilated_UNet, self).__init__()

        self.network_depth = network_depth
        self.bottleneck_layers = bottleneck_layers
        skip_connection_channel_counts = []

        self.add_module('firstconv', nn.Conv2d(in_channels=inChannel,
                                               out_channels=init_features, kernel_size=3,
                                               stride=1, padding=1, bias=True))

        self.encodingBlocks = nn.ModuleList([])
        features = init_features

        for i in range(self.network_depth):
            self.encodingBlocks.append(Dilated_UNetEnc(features, 2 * features))

            skip_connection_channel_counts.insert(0, 2 * features)
            features *= 2
        final_encoding_channels = skip_connection_channel_counts[0]

        self.bottleNecks = nn.ModuleList([])
        for i in range(self.bottleneck_layers):
            dilation_factor = 1
            self.bottleNecks.append(Dilated_Bottleneck_block(final_encoding_channels,
                                                             final_encoding_channels, dilation_rate=dilation_factor))

        self.decodingBlocks = nn.ModuleList([])
        for i in range(self.network_depth):
            if i == 0:
                prev_deconv_channels = final_encoding_channels
            self.decodingBlocks.append(UNetDec(prev_deconv_channels + skip_connection_channel_counts[i],
                                               skip_connection_channel_counts[i]))
            prev_deconv_channels = skip_connection_channel_counts[i]

        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        out = self.firstconv(x)

        skip_connections = []
        for i in range(self.network_depth):
            out = self.encodingBlocks[i](out)
            skip_connections.append(out)

        for i in range(self.bottleneck_layers):
            out = self.bottleNecks[i](out)

        for i in range(self.network_depth):
            skip = skip_connections.pop()
            out = self.decodingBlocks[i](torch.cat([out, skip], 1))

        out = self.final(out)
        return out