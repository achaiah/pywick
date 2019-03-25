# Source: https://github.com/keng000/pytorch_unet_plus_plus/blob/master/unet_plus_plus.py (License: MIT)

import torch
import torch.nn as nn


class ConvSamePad2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias: bool = True):
        super().__init__()

        left_top_pad = right_bottom_pad = kernel_size // 2
        if kernel_size % 2 == 0:
            right_bottom_pad -= 1

        self.layer = nn.Sequential(
            nn.ReflectionPad2d((left_top_pad, right_bottom_pad, left_top_pad, right_bottom_pad)),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias)
        )

    def forward(self, inputs):
        return self.layer(inputs)


class StandardUnit(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate=0.5):
        super().__init__()
        self.layer = nn.Sequential(
            ConvSamePad2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3),
            nn.Dropout2d(p=drop_rate),
            ConvSamePad2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3),
            nn.Dropout2d(p=drop_rate)
        )

    def forward(self, inputs):
        return self.layer(inputs)


class Final1x1ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layer = nn.Sequential(
            ConvSamePad2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        return self.layer(inputs)


class NestNet(nn.Module):
    def __init__(self, in_channels, n_classes, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision

        filters = [32, 64, 128, 256, 512]

        # j == 0
        self.x_00 = StandardUnit(in_channels=in_channels, out_channels=filters[0])
        self.pool0 = nn.MaxPool2d(kernel_size=2)

        self.x_01 = StandardUnit(in_channels=filters[0] * 2, out_channels=filters[0])
        self.x_02 = StandardUnit(in_channels=filters[0] * 3, out_channels=filters[0])
        self.x_03 = StandardUnit(in_channels=filters[0] * 4, out_channels=filters[0])
        self.x_04 = StandardUnit(in_channels=filters[0] * 5, out_channels=filters[0])

        self.up_10_to_01 = nn.ConvTranspose2d(in_channels=filters[1], out_channels=filters[0], kernel_size=2, stride=2)
        self.up_11_to_02 = nn.ConvTranspose2d(in_channels=filters[1], out_channels=filters[0], kernel_size=2, stride=2)
        self.up_12_to_03 = nn.ConvTranspose2d(in_channels=filters[1], out_channels=filters[0], kernel_size=2, stride=2)
        self.up_13_to_04 = nn.ConvTranspose2d(in_channels=filters[1], out_channels=filters[0], kernel_size=2, stride=2)

        # j == 1
        self.x_10 = StandardUnit(in_channels=filters[0], out_channels=filters[1])
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.x_11 = StandardUnit(in_channels=filters[1] * 2, out_channels=filters[1])
        self.x_12 = StandardUnit(in_channels=filters[1] * 3, out_channels=filters[1])
        self.x_13 = StandardUnit(in_channels=filters[1] * 4, out_channels=filters[1])

        self.up_20_to_11 = nn.ConvTranspose2d(in_channels=filters[2], out_channels=filters[1], kernel_size=2, stride=2)
        self.up_21_to_12 = nn.ConvTranspose2d(in_channels=filters[2], out_channels=filters[1], kernel_size=2, stride=2)
        self.up_22_to_13 = nn.ConvTranspose2d(in_channels=filters[2], out_channels=filters[1], kernel_size=2, stride=2)

        # j == 2
        self.x_20 = StandardUnit(in_channels=filters[1], out_channels=filters[2])
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.x_21 = StandardUnit(in_channels=filters[2] * 2, out_channels=filters[2])
        self.x_22 = StandardUnit(in_channels=filters[2] * 3, out_channels=filters[2])

        self.up_30_to_21 = nn.ConvTranspose2d(in_channels=filters[3], out_channels=filters[2], kernel_size=2, stride=2)
        self.up_31_to_22 = nn.ConvTranspose2d(in_channels=filters[3], out_channels=filters[2], kernel_size=2, stride=2)

        # j == 3
        self.x_30 = StandardUnit(in_channels=filters[2], out_channels=filters[3])
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.x_31 = StandardUnit(in_channels=filters[3] * 2, out_channels=filters[3])

        self.up_40_to_31 = nn.ConvTranspose2d(in_channels=filters[4], out_channels=filters[3], kernel_size=2, stride=2)

        # j == 4
        self.x_40 = StandardUnit(in_channels=filters[3], out_channels=filters[4])

        # 1x1 conv layer
        self.final_1x1_x01 = Final1x1ConvLayer(in_channels=filters[0], out_channels=n_classes)
        self.final_1x1_x02 = Final1x1ConvLayer(in_channels=filters[0], out_channels=n_classes)
        self.final_1x1_x03 = Final1x1ConvLayer(in_channels=filters[0], out_channels=n_classes)
        self.final_1x1_x04 = Final1x1ConvLayer(in_channels=filters[0], out_channels=n_classes)

    def forward(self, inputs, L=4):
        if not (1 <= L <= 4):
            raise ValueError("the model pruning factor `L` should be 1 <= L <= 4")

        x_00_output = self.x_00(inputs)
        x_10_output = self.x_10(self.pool0(x_00_output))
        x_10_up_sample = self.up_10_to_01(x_10_output)
        x_01_output = self.x_01(torch.cat([x_00_output, x_10_up_sample], 1))
        nestnet_output_1 = self.final_1x1_x01(x_01_output)

        if L == 1:
            return nestnet_output_1

        x_20_output = self.x_20(self.pool1(x_10_output))
        x_20_up_sample = self.up_20_to_11(x_20_output)
        x_11_output = self.x_11(torch.cat([x_10_output, x_20_up_sample], 1))
        x_11_up_sample = self.up_11_to_02(x_11_output)
        x_02_output = self.x_02(torch.cat([x_00_output, x_01_output, x_11_up_sample], 1))
        nestnet_output_2 = self.final_1x1_x01(x_02_output)

        if L == 2:
            if self.deep_supervision:
                # return the average of output layers
                return (nestnet_output_1 + nestnet_output_2) / 2
            else:
                return nestnet_output_2

        x_30_output = self.x_30(self.pool2(x_20_output))
        x_30_up_sample = self.up_30_to_21(x_30_output)
        x_21_output = self.x_21(torch.cat([x_20_output, x_30_up_sample], 1))
        x_21_up_sample = self.up_21_to_12(x_21_output)
        x_12_output = self.x_12(torch.cat([x_10_output, x_11_output, x_21_up_sample], 1))
        x_12_up_sample = self.up_12_to_03(x_12_output)
        x_03_output = self.x_03(torch.cat([x_00_output, x_01_output, x_02_output, x_12_up_sample], 1))
        nestnet_output_3 = self.final_1x1_x01(x_03_output)

        if L == 3:
            # return the average of output layers
            if self.deep_supervision:
                return (nestnet_output_1 + nestnet_output_2 + nestnet_output_3) / 3
            else:
                return nestnet_output_3

        x_40_output = self.x_40(self.pool3(x_30_output))
        x_40_up_sample = self.up_40_to_31(x_40_output)
        x_31_output = self.x_31(torch.cat([x_30_output, x_40_up_sample], 1))
        x_31_up_sample = self.up_31_to_22(x_31_output)
        x_22_output = self.x_22(torch.cat([x_20_output, x_21_output, x_31_up_sample], 1))
        x_22_up_sample = self.up_22_to_13(x_22_output)
        x_13_output = self.x_13(torch.cat([x_10_output, x_11_output, x_12_output, x_22_up_sample], 1))
        x_13_up_sample = self.up_13_to_04(x_13_output)
        x_04_output = self.x_04(torch.cat([x_00_output, x_01_output, x_02_output, x_03_output, x_13_up_sample], 1))
        nestnet_output_4 = self.final_1x1_x01(x_04_output)

        if L == 4:
            if self.deep_supervision:
                # return the average of output layers
                return (nestnet_output_1 + nestnet_output_2 + nestnet_output_3 + nestnet_output_4) / 4
            else:
                return nestnet_output_4


if __name__ == '__main__':
    inputs = torch.rand((3, 1, 96, 96)).cuda()

    unet_plus_plus = NestNet(in_channels=1, n_classes=3).cuda()

    from datetime import datetime

    st = datetime.now()
    output = unet_plus_plus(inputs, L=1)
    print(f"{(datetime.now() - st).total_seconds(): .4f}s")
