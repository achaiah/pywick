# Source: https://github.com/areum-lee/SENet_Segmentation/blob/master/2d_densenetse_model.py

import torch
import torch.functional as F
import torch.nn as nn

def center_crop(layer, max_height, max_width):
    batch_size, n_channels, layer_height, layer_width = layer.size()
    xy1 = (layer_width - max_width) // 2
    xy2 = (layer_height - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]


class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))

        self.add_module('conv', nn.Conv2d(in_channels=in_channels,
                                          out_channels=growth_rate, kernel_size=3, stride=1,
                                          padding=1, bias=True))

        self.add_module('drop', nn.Dropout2d(0.2))

    def forward(self, x):
        out = super(DenseLayer, self).forward(x)
        return out


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, upsample=False):
        super(DenseBlock, self).__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([DenseLayer(
            in_channels + i * growth_rate, growth_rate)
            for i in range(n_layers)])

        self.SE_upsample1 = nn.Conv2d(growth_rate * n_layers, growth_rate * n_layers // 16, kernel_size=1)
        self.SE_upsample2 = nn.Conv2d(growth_rate * n_layers // 16, growth_rate * n_layers, kernel_size=1)
        self.SE1 = nn.Conv2d((in_channels + growth_rate * n_layers), (in_channels + growth_rate * n_layers) // 16, kernel_size=1)
        self.SE2 = nn.Conv2d((in_channels + growth_rate * n_layers) // 16, (in_channels + growth_rate * n_layers), kernel_size=1)

    def forward(self, x):
        if self.upsample:
            new_features = []
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            out = torch.cat(new_features, 1)
            fm_size = out.size()[2]
            scale_weight = F.avg_pool2d(out, fm_size)
            scale_weight = F.relu(self.SE_upsample1(scale_weight))
            scale_weight = F.sigmoid(self.SE_upsample2(scale_weight))
            out = out * scale_weight.expand_as(out)
            return out
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)  # 1 = channel axis
            fm_size = x.size()[2]
            scale_weight = F.avg_pool2d(x, fm_size)
            scale_weight = F.relu(self.SE1(scale_weight))
            scale_weight = F.sigmoid(self.SE2(scale_weight))
            x = x * scale_weight.expand_as(x)
            return x


class TransitionDown(nn.Sequential):
    def __init__(self, in_channels):
        super(TransitionDown, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels=in_channels,
                                          out_channels=in_channels, kernel_size=1, stride=1,
                                          padding=0, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))
        self.add_module('maxpool', nn.MaxPool2d(2))

    def forward(self, x):
        out = super(TransitionDown, self).forward(x)

        return out


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionUp, self).__init__()
        self.convTrans = nn.ConvTranspose2d(in_channels=in_channels,
                                            out_channels=out_channels, kernel_size=3, stride=2,
                                            padding=0,
                                            bias=True)  # crop = 'valid' means padding=0. Padding has reverse effect for transpose conv (reduces output size)

    def forward(self, x, skip):
        out = self.convTrans(x)
        out = center_crop(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)

        return out


class Bottleneck(nn.Sequential):
    def __init__(self, in_channels, growth_rate, n_layers):
        super(Bottleneck, self).__init__()
        self.add_module('bottleneck', DenseBlock(in_channels, growth_rate, n_layers, upsample=True))

    def forward(self, x):
        out = super(Bottleneck, self).forward(x)
        return out


class FCDenseNet(nn.Module):
    def __init__(self, in_channels=3, down_blocks=(5, 5, 5, 5, 5),
                 up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48, n_classes=12):
        super(FCDenseNet, self).__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks

        cur_channels_count = 0
        skip_connection_channel_counts = []

        #####################
        # First Convolution #
        #####################

        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
                                               out_channels=out_chans_first_conv, kernel_size=3,
                                               stride=1, padding=1, bias=True))
        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################

        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i]))
            cur_channels_count += (growth_rate * down_blocks[i])
            skip_connection_channel_counts.insert(0, cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))

        #####################
        #     Bottleneck    #
        #####################

        self.add_module('bottleneck', Bottleneck(cur_channels_count,
                                                 growth_rate, bottleneck_layers))
        prev_block_channels = growth_rate * bottleneck_layers
        cur_channels_count += prev_block_channels

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks) - 1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlocksUp.append(DenseBlock(
                cur_channels_count, growth_rate, up_blocks[i],
                upsample=True))
            prev_block_channels = growth_rate * up_blocks[i]
            cur_channels_count += prev_block_channels

        # One final dense block
        self.transUpBlocks.append(TransitionUp(
            prev_block_channels, prev_block_channels))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]

        self.denseBlocksUp.append(DenseBlock(
            cur_channels_count, growth_rate, up_blocks[-1],
            upsample=False))
        cur_channels_count += growth_rate * up_blocks[-1]

        #####################
        #      Softmax      #
        #####################

        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
                                   out_channels=n_classes, kernel_size=1, stride=1,
                                   padding=0, bias=True)
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        out = self.firstconv(x)

        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)

        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)

        out = self.finalConv(out)
        out = self.softmax(out)
        return out


def FCDenseNet57(n_classes):
    return FCDenseNet(in_channels=2, down_blocks=(4, 4, 4, 4, 4),
                      up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=4,
                      growth_rate=4, out_chans_first_conv=48, n_classes=n_classes)


def FCDenseNet67(n_classes):
    return FCDenseNet(in_channels=2, down_blocks=(5, 5, 5, 5, 5),
                      up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
                      growth_rate=16, out_chans_first_conv=48, n_classes=n_classes)


def FCDenseNet103(n_classes):
    return FCDenseNet(in_channels=2, down_blocks=(4, 5, 7, 10, 12),
                      up_blocks=(12, 10, 7, 5, 4), bottleneck_layers=15,
                      growth_rate=16, out_chans_first_conv=48, n_classes=n_classes)